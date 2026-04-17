#!/usr/bin/env python3
"""
weather_agent.py — multimodal weather agent · Claude Code-style UI
Modularized version using src.common and src.modules.weather_logic
"""

from __future__ import annotations

import sys
from pathlib import Path

# Fix sys.path for robust modular imports when run as a script
_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)


import argparse
import asyncio
import sys
import time
from pathlib import Path
from textwrap import dedent
from typing import Optional

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

# Common imports
from src.common.ui import UIState
from src.common.ollama import setup_ollama, _stream_chat, _chat_json, _warmup
from src.common.types import TokenUsage

# Weather logic imports
from src.modules.weather_logic import (
    WeatherReport, WeatherComparison, MemorySnapshot,
    geocode, fetch_weather, load_memory, save_memory,
    find_prev_snapshot, compare, extract_location,
    snapshot_from_report, upsert_snapshot
)

# Weather UI imports
from src.modules.weather_ui import (
    _cstyle, _uv_col, _bar, _moon_em, _day_label, _delta_text,
    _coord, _uv_label, _obs_to_local
)

# ══════════════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════════════

OLLAMA_URL      = "http://localhost:11434"
AGENT_MODEL     = "llama3.1:8b"
CHECKER_MODEL   = "llama3.1:8b"
UI_REFRESH_HZ   = 10

# ══════════════════════════════════════════════════════════════════════
#  Prompts
# ══════════════════════════════════════════════════════════════════════

AGENT_SYSTEM = dedent("""\
    You are a concise, accurate weather assistant.

    Rules:
      1. Use ONLY the supplied weather data — never invent numbers.
      2. If an image is attached, describe only weather-relevant content.
      3. If prior-day memory exists, note how conditions changed.
      4. Never apologise for wttr.in area-name differences (nearest-area behaviour).
      5. Treat hourly forecast as 3-hour slots, not exact hour-by-hour certainty.
      6. Separate observed conditions from forecast conditions.
      7. If the signal is weak or mixed, say so directly instead of overstating confidence.
      8. Use rain chance, cloud cover, pressure, wind, UV, and temp progression when describing trends.
      9. Format your response as four sections, each starting on a new line:
           CURRENT: one-sentence summary with observation date and key numbers
           CHANGES: how it changed vs yesterday (if data exists)
           HOURLY: 2-3 sentence narrative on the day's progression
           OUTLOOK: summary of the 3-day forecast window (e.g. Oct 14-16)
""")

VISION_SYSTEM = "You are a weather-satellite and radar-image analyst. Describe the key visual features."

CLASSIFIER_SYSTEM = dedent("""\
    Identify if the user is asking about weather. Return JSON:
    { "is_weather_query": bool, "location": "string", "confidence": float, "reason": "string" }
""")

RESULT_CHECK_SYSTEM = dedent("""\
    As a Quality Assurance agent, verify if the weather answer is grounded in the provided data.
    Return JSON: { "is_valid": bool, "quality_score": float, "notes": "string", "issues": [] }
""")

# ══════════════════════════════════════════════════════════════════════
#  Context builder
# ══════════════════════════════════════════════════════════════════════

def build_context(query: str, r: WeatherReport, comp: WeatherComparison, vision: str) -> str:
    lines = [
        f"Query: {query}",
        f"Location requested: {r.queried_location}",
        f"wttr.in grid: {r.location_label}",
        f"Coordinates: {_coord(r.lat, r.lon)}",
        "",
        "Current conditions:",
        f"  {r.desc},  {r.temp_f}°F (feels {r.feels_f}°F)",
        f"  Humidity {r.humidity}%  Wind {r.wind_dir} {r.wind_mph} mph"
        + (f" ({r.wind_deg}°)" if r.wind_deg else ""),
        f"  UV {_uv_label(r.uv_index)}  Visibility {r.visibility_mi} mi",
        f"  Pressure {r.pressure_mb} mb  Precip {r.precip_in} in",
        f"  Cloud cover {r.cloud_cover}%",
        f"  Observed {_obs_to_local(r.obs_time, r.tz_name)}" if r.obs_time else "",
        "",
        "Prior-day comparison:",
        f"  Found: {'yes' if comp.found else 'no'}",
        f"  Summary: {comp.summary}",
    ]
    if comp.bullets and comp.found:
        for b in comp.bullets:
            lines.append(f"  • {b}")

    if r.forecast:
        today = r.forecast[0]
        lines += ["", f"Today ({today.date}) hourly observations (3-hour slots):"]
        for s in today.hourly:
            now_mark = " ← NOW" if s.is_current else ""
            lines.append(
                f"  {s.time_label}  {s.desc:20s}  {s.temp_f}°F (feels {s.feels_f}°F)"
                f"  hum {s.humidity}%  wind {s.wind_dir} {s.wind_mph} mph"
                f"  rain {s.rain_chance}%  snow {s.snow_chance}%  UV {s.uv_index}"
                f"  cloud {s.cloud_cover}%{now_mark}"
            )

    lines += ["", "3-day forecast:"]
    for i, day in enumerate(r.forecast):
        tag = ("Today", "Tomorrow", day.date)[min(i, 2)]
        lines += [
            f"  {tag}: {day.desc}  H {day.high_f}°F / L {day.low_f}°F",
            f"    Rain {day.rain_chance}%  Snow {day.snow_chance}%  UV {_uv_label(day.uv_index)}",
            f"    Sunrise {day.sunrise}  Sunset {day.sunset}" if day.sunrise else "",
            f"    Moon: {day.moon_phase} ({day.moon_illumination}% lit)" if day.moon_phase else "",
        ]
    if vision:
        lines += ["", "Image analysis:", vision]

    return "\n".join(x for x in lines if x != "")

# ══════════════════════════════════════════════════════════════════════
#  Weather card (Rich)
# ══════════════════════════════════════════════════════════════════════

def print_weather_card(
    console: Console,
    r: WeatherReport,
    comp: WeatherComparison,
    prev_snap: Optional[MemorySnapshot],
    answer: str,
    vision: str,
    harness: bool = False,
) -> None:
    em, ccol = _cstyle(r.desc)
    hdr = Text(justify="left" if harness else "center")
    hdr.append(f"\n  {em}  ", f"bold {ccol}")
    hdr.append(r.queried_location, "bold white")
    if r.location_label and r.location_label != r.queried_location:
        hdr.append(f"\n  wttr.in grid: {r.location_label}", "grey50")
    if r.lat and r.lon:
        hdr.append(f"\n  {_coord(r.lat, r.lon)}", "grey50")
    if r.obs_time:
        hdr.append(f"\n  Observed {_obs_to_local(r.obs_time, r.tz_name)}", "grey50")
    hdr.append("\n")
    if not harness:
        console.print(Panel(hdr, border_style="blue", expand=True, box=box.DOUBLE_EDGE))
    else:
        # In harness mode, just print the header text to avoid double-boxing
        console.print(hdr)

    L = Table.grid(padding=(0, 2)); L.add_column(style="grey58"); L.add_column(style="bold white")
    R = Table.grid(padding=(0, 2)); R.add_column(style="grey58"); R.add_column(style="bold white")

    tt = Text(); tt.append(f"{r.temp_f}°F", f"bold {ccol}"); tt.append(f"  feels {r.feels_f}°F", "white")
    uvt = Text(); uvt.append(_uv_label(r.uv_index), _uv_col(r.uv_index))

    L.add_row("🌡 Temp",       tt)
    L.add_row("☁ Conditions",  Text(r.desc, ccol))
    L.add_row("💦 Humidity",   Text(f"{_bar(r.humidity)}  {r.humidity}%", "cyan"))
    L.add_row("🌬 Wind",       Text(f"{_bar(r.wind_mph, step=5)}  {r.wind_mph} mph", "sky_blue1"))
    L.add_row("🧭 Direction",  Text(f"{r.wind_dir}  ({r.wind_deg}°)" if r.wind_deg else r.wind_dir, "sky_blue1"))

    R.add_row("☀ UV Index",    uvt)
    R.add_row("👁 Visibility",  Text(f"{r.visibility_mi} mi",   "white"))
    R.add_row("📊 Pressure",    Text(f"{r.pressure_mb} mb",     "white"))
    R.add_row("🌧 Precip",      Text(f"{r.precip_in} in",       "sky_blue1"))
    R.add_row("🌥 Cloud cover", Text(f"{_bar(r.cloud_cover)}  {r.cloud_cover}%", "grey74"))

    # Current Conditions - remove equal=True to save space on labels
    cond_node = Columns([L, R], equal=False, expand=True)
    if not harness:
        console.print(Panel(cond_node, title="[bold]Current Conditions[/bold]", expand=True, border_style=ccol, box=box.ROUNDED))
    else:
        console.print(Rule("[bold]Current Conditions[/bold]", style=ccol))
        console.print(cond_node)

    if r.forecast and r.forecast[0].hourly:
        ht = Table(box=box.SIMPLE_HEAD, expand=True, show_header=True, header_style="bold white")
        ht.add_column("Time",       style="bold white")
        ht.add_column("Conditions")
        ht.add_column("Temp",       justify="right", style="bold")
        if not harness:
            ht.add_column("Feels",      justify="right", style="dim white")
            ht.add_column("Hum",        justify="right", style="cyan")
        ht.add_column("Wind",       justify="right", style="sky_blue1")
        if not harness:
            ht.add_column("UV",         justify="center")
        ht.add_column("🌧",         justify="right", style="sky_blue1")
        if not harness:
            ht.add_column("❄",          justify="right", style="cyan")
            ht.add_column("Cloud",      justify="right", style="grey74")

        for slot in r.forecast[0].hourly:
            se, sc = _cstyle(slot.desc)
            row_style = "on grey11" if slot.is_current else ""
            now_mark = " ◀" if slot.is_current else ""
            try:
                tf = int(slot.temp_f)
                tc = ("bold cyan" if tf < 32 else "bold sky_blue1" if tf < 50 else
                      "bold yellow3" if tf < 80 else "bold red")
            except Exception: tc = "white"
            uvt2 = Text(_uv_label(slot.uv_index), _uv_col(slot.uv_index))
            row_args = [slot.time_label + now_mark, Text(f"{se} {slot.desc}", sc), Text(f"{slot.temp_f}°F", tc)]
            if not harness:
                row_args.extend([f"{slot.feels_f}°F", f"{slot.humidity}%"])
            row_args.append(f"{slot.wind_dir} {slot.wind_mph}")
            if not harness:
                row_args.append(uvt2)
            row_args.append(f"{slot.rain_chance}%")
            if not harness:
                row_args.extend([f"{slot.snow_chance}%", f"{slot.cloud_cover}%"])
            
            ht.add_row(*row_args, style=row_style)

        if not harness:
            console.print(Panel(ht, title=f"[bold]Today's Hourly Forecast[/bold]", expand=True, border_style="cyan", box=box.ROUNDED))
        else:
            console.print(Rule("[bold]Today's Hourly Forecast[/bold]", style="cyan"))
            console.print(ht)

    if comp.found and prev_snap:
        dt = Table(box=box.SIMPLE, expand=True, show_header=True, header_style="bold grey50")
        dt.add_column("Field",    style="grey58",   min_width=14); dt.add_column("Previous", justify="right",  style="white")
        dt.add_column("Now",      justify="right",  style="bold white"); dt.add_column("Δ",        justify="right")
        rows = [("🌡 Temp °F",  prev_snap.temp_f,  r.temp_f,   "°F"), ("🌡 Feels °F", prev_snap.feels_f, r.feels_f,  "°F"),
                ("💦 Humidity", prev_snap.humidity, r.humidity, "%"), ("🌬 Wind mph", prev_snap.wind_mph, r.wind_mph, ""),
                ("☁ Cond.",    prev_snap.desc,     r.desc,     None), ("🌥 Cloud %",  prev_snap.cloud_cover, r.cloud_cover, "%")]
        for label, old, new, unit in rows:
            delta = Text("changed", "yellow") if unit is None and old != new else Text("same", "grey50") if unit is None else _delta_text(new, old, unit)
            dt.add_row(label, str(old), str(new), delta)
        console.print(Panel(dt, title=f"[bold]vs {comp.previous_date}[/bold]",
                            expand=False if harness else True,
                            border_style="magenta", box=box.SIMPLE if harness else box.ROUNDED))
    elif not comp.found:
        console.print(Panel(Text("No prior snapshot — run again tomorrow to compare.", "grey50"),
                            title="[bold]Prior-Day Comparison[/bold]", 
                            expand=False if harness else True,
                            border_style="grey35", box=box.ROUNDED))

    if r.forecast:
        ft = Table(box=box.SIMPLE_HEAD, expand=True, show_header=True, header_style="bold white")
        ft.add_column("Day",       style="bold white"); ft.add_column("Conditions")
        ft.add_column("High",      justify="right", style="bold red"); ft.add_column("Low",       justify="right", style="bold cyan")
        if not harness:
            ft.add_column("UV",        justify="center")
        ft.add_column("🌧 Rain",   justify="right", style="sky_blue1")
        if not harness:
            ft.add_column("❄ Snow",    justify="right", style="cyan")
            ft.add_column("Sunrise",   justify="right", style="yellow3")
            ft.add_column("Sunset",    justify="right", style="orange3")
            ft.add_column("Moon",      justify="center")
        for i, day in enumerate(r.forecast):
            fe, _ = _cstyle(day.desc); moon = f"{_moon_em(day.moon_phase)} {day.moon_phase}"
            if day.moon_illumination: moon += f" {day.moon_illumination}%"
            row_args = [_day_label(day.date, i), f"{fe} {day.desc}" if day.desc else "—", f"{day.high_f}°F", f"{day.low_f}°F"]
            if not harness:
                row_args.append(Text(_uv_label(day.uv_index), _uv_col(day.uv_index)))
            row_args.append(f"{day.rain_chance}%")
            if not harness:
                row_args.extend([f"{day.snow_chance}%", day.sunrise or "—", day.sunset or "—", moon or "—"])
            
            ft.add_row(*row_args)
        console.print(Panel(ft, title="[bold]3-Day Forecast[/bold]",
                            expand=True,
                            border_style="yellow3", box=box.SIMPLE if harness else box.ROUNDED))

    if vision:
        console.print(Panel(Text(vision, "white"), title="[bold]🛰  Image Analysis[/bold]", 
                            expand=False if harness else True,
                            border_style="blue", box=box.ROUNDED))

    if answer:
        sections: dict[str, str] = {}
        current_key = None
        for line in answer.splitlines():
            for key in ("CURRENT:", "CHANGES:", "HOURLY:", "OUTLOOK:"):
                if line.upper().startswith(key):
                    current_key = key.rstrip(":"); sections[current_key] = line[len(key):].strip(); break
            else:
                if current_key and line.strip(): sections[current_key] = sections[current_key] + "\n" + line
        if sections:
            sec_table = Table.grid(padding=(0, 2)); sec_table.add_column(style="bold cyan", min_width=10); sec_table.add_column(style="white")
            for key in ("CURRENT", "CHANGES", "HOURLY", "OUTLOOK"):
                if key in sections: sec_table.add_row(key, sections[key].strip())
            console.print(Panel(sec_table, title=f"[bold]🤖 {AGENT_MODEL}[/bold]", border_style="green", box=box.ROUNDED))
        else:
            console.print(Panel(Text(answer, "white"), title=f"[bold]🤖 {AGENT_MODEL}[/bold]", border_style="green", box=box.ROUNDED))

# ══════════════════════════════════════════════════════════════════════
#  Main pipeline
# ══════════════════════════════════════════════════════════════════════

async def run(query: str, image_path: Optional[str] = None, harness: bool = False) -> int:
    ui      = UIState(agent_name="weather-agent", model_info=f"{AGENT_MODEL} · {CHECKER_MODEL}")
    if harness: ui.harness_mode = True

    report:     Optional[WeatherReport]    = None
    comp:       Optional[WeatherComparison]= None
    prev_snap:  Optional[MemorySnapshot]   = None
    answer      = ""
    vision_note = ""
    expert_manual = ""

    async with ui:
        def refresh() -> None:
            ui.refresh()

        try:
            s_init = ui.add_step("connect + warmup").start(); refresh()
            client = await setup_ollama(OLLAMA_URL, [AGENT_MODEL, CHECKER_MODEL])
            
            # Parallel warmup using asyncio.gather
            await asyncio.gather(
                _warmup(client, AGENT_MODEL),
                _warmup(client, CHECKER_MODEL)
            )
            
            # ── Load Domain Knowledge ──
            try:
                from src.modules.knowledge_logic import get_logic
                logic = get_logic()
                if logic.load():
                    results = logic.query("weather protocol visual assessment coordinates")
                    expert_manual = "\n".join(r.get("text", "") for r in results)
            except Exception:
                pass
                
            s_init.done("Ollama ready · models warmed"); refresh()

            loc_hint = extract_location(query) or query.strip().rstrip(" ?.")
            s_cls  = ui.add_step("classify query").start(); refresh()
            s_geo  = ui.add_step("geocode").start();        refresh()

            f_cls_task = _chat_json(client, CHECKER_MODEL, CLASSIFIER_SYSTEM, f"Classify: {query!r}", ui, refresh, "classify")
            f_geo_task = asyncio.to_thread(geocode, loc_hint)
            
            pc_raw, pc_usage = await f_cls_task
            coords = await f_geo_task

            is_weather = bool(pc_raw.get("is_weather_query", False))
            location   = str(pc_raw.get("location", "")).strip() or loc_hint
            s_cls.done(f"{'✓ weather' if is_weather else '✗ not weather'} ({pc_raw.get('confidence',0):.0%})")
            s_geo.done(f"{coords[0]:.4f}°N, {coords[1]:.4f}°W" if coords else "geocode failed")
            refresh()

            if not is_weather:
                ui.push_chunk("This doesn't appear to be a weather query.\n" + str(pc_raw.get("reason", "")))
                return 1

            s_wx  = ui.add_step("fetch weather").start();  refresh()
            s_mem = ui.add_step("load memory").start();    refresh()

            f_wx_task = asyncio.to_thread(fetch_weather, location, coords)
            f_mem_task = asyncio.to_thread(load_memory)
            
            report = await f_wx_task
            snapshots = await f_mem_task

            s_wx.done(f"{report.desc} · {report.temp_f}°F")
            prev_snap = find_prev_snapshot(report, snapshots)
            comp      = compare(prev_snap, report)
            s_mem.done(f"vs {comp.previous_date}" if comp.found else "no prior data"); refresh()

            if image_path:
                img = Path(image_path).expanduser()
                if img.exists():
                    s_vis = ui.add_step("vision analysis").start(); refresh()
                    vision_note, _ = await _stream_chat(client, AGENT_MODEL, [{"role": "user", "content": f"Analyse context: {report.desc}", "images": [str(img)]}], ui, refresh, phase="vision")
                    s_vis.done(vision_note[:60]); refresh()

            s_ans = ui.add_step("generate answer").start(); refresh()
            ctx   = build_context(query, report, comp, vision_note)
            sys_prompt = AGENT_SYSTEM
            if expert_manual:
                sys_prompt = f"EXPERT MANUAL:\n{expert_manual}\n\n{AGENT_SYSTEM}"

            answer, _ = await _stream_chat(client, AGENT_MODEL, 
                                     [{"role": "system", "content": sys_prompt},
                                      {"role": "user", "content": f"Query: {query}\n\n{ctx}"}],
                                     ui, refresh, phase="answer")
            s_ans.done(answer[:60]); refresh()

            s_val = ui.add_step("validate result").start(); refresh()
            res, _ = await _chat_json(client, CHECKER_MODEL, RESULT_CHECK_SYSTEM, f"Query: {query}\nAnswer: {answer}", ui, refresh, "validate")
            s_val.done(f"valid={res.get('is_valid', True)} score={res.get('quality_score', 1):.0%}"); refresh()

            s_save = ui.add_step("save memory").start(); refresh()
            snapshots = upsert_snapshot(snapshots, snapshot_from_report(report))
            save_memory(snapshots)
            s_save.done("memory updated"); refresh()

        except Exception as exc:
            ui.push_chunk(f"[error] {exc}")
            return 1

    if report and comp:
        if not harness: ui.console.print()
        print_weather_card(ui.console, report, comp, prev_snap, answer, vision_note, harness=harness)
    return 0


async def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal weather agent")
    parser.add_argument("query", nargs="*", help="Weather question")
    parser.add_argument("--image", metavar="PATH", help="Image for vision analysis")
    parser.add_argument("--harness", action="store_true", help="Harness mode")
    args = parser.parse_args()
    query = " ".join(args.query) if args.query else input("Weather query: ")
    res = await run(query, args.image, harness=args.harness)
    sys.exit(res)

if __name__ == "__main__":
    asyncio.run(main())
