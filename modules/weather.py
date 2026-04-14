#!/usr/bin/env python3
"""
weather_agent.py — multimodal weather agent · Claude Code-style UI

Models : llava:7b-v1.6-mistral-q4_0 (agent, ~4.1 GB) · llama3.2:3b (checker, ~2.8 GB)

Speed vs prior version
  ● Geocoding + wttr.in fetch execute concurrently WITH the prompt classifier
    (HTTP calls overlap the LLM call → ~400-600 ms saved per run)
  ● Nominatim results cached to ~/.weather_geocache.json (subsequent runs: ~0 ms)
  ● Memory snapshot loaded concurrently with the weather fetch
  ● Models warmed up at startup so the first real call skips cold-load delay
  ● Ollama's own OLLAMA_NUM_PARALLEL env var does NOT help on 8 GB — two LLM
    calls at once would thrash VRAM.  All LLM calls remain sequential.
    Only the HTTP / disk work is parallelised.

Hourly forecast
  ● wttr.in j1 returns 8 × 3-hour slots per day (00:00–21:00 local)
  ● Current hour is highlighted in the table
  ● All 8 slots fed to the agent → agent produces an educated hourly narrative

UI  Claude Code aesthetic
  ◆  weather-agent  header
  ✓/✗/⠸  per-step status with elapsed time
  ⚡  geocode cache hit indicator
  inline streaming output with ▌ cursor

pip install ollama rich
python agent.py "weather in Ashland, Kentucky"
python agent.py "radar update for Ashland KY" --image radar.png
python agent.py --clear-memory
"""




from __future__ import annotations

import argparse
import json
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import date as Date, datetime, timedelta, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any, Optional
from urllib.parse import quote
from urllib.request import Request, urlopen

from ollama import Client, ResponseError
from rich import box
from rich.columns import Columns
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

import shutil

# ══════════════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════════════

OLLAMA_URL    = "http://localhost:11434"
AGENT_MODEL   = "llava:7b-v1.6-mistral-q4_0"
CHECKER_MODEL = "llama3.2:3b"

MEMORY_PATH   = Path.home() / ".weather_agent_memory.json"
GEOCACHE_PATH = Path.home() / ".weather_geocache.json"

UI_REFRESH_HZ   = 10
DEFAULT_TIMEOUT = 20
MAX_STREAM_LINES = 8

SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

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
           CURRENT: one-sentence summary with key numbers
           CHANGES: delta vs yesterday (or "No prior data.")
           HOURLY: cautious narrative of today's 3-hour forecast progression (≤ 5 sentences)
           OUTLOOK: concise 3-day forecast with confidence wording when needed
""")


CLASSIFIER_SYSTEM = dedent("""\
    Respond with ONLY a JSON object — no markdown, no prose.
    {"is_weather_query":bool,"location":"place name or empty","confidence":0.0,"reason":"one sentence"}
""")

RESULT_CHECK_SYSTEM = dedent("""\
    Respond with ONLY a JSON object — no markdown, no prose.
    {"is_valid":bool,"issues":[],"quality_score":0.0,"notes":"brief"}
    NOTE: wttr.in uses nearest-area naming. Never flag area-name differences as issues.
    Validate: plausible temperatures, humidity 0-100%, no hallucinated values.
""")

VISION_SYSTEM = dedent("""\
    Analyse this weather image. Identify:
      • Radar: reflectivity, storm cells, precip type and intensity
      • Satellite: cloud bands, fronts, coverage extent
      • Sky photo: cloud type, coverage, visibility clues
    State your confidence. Be specific and concise (3-5 sentences).
""")

# ══════════════════════════════════════════════════════════════════════
#  Data models
# ══════════════════════════════════════════════════════════════════════

@dataclass
class TokenUsage:
    prompt_tokens:     int   = 0
    completion_tokens: int   = 0
    total_duration_ms: float = 0.0
    estimated:         bool  = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class HourlySlot:
    time_label:  str          # "06:00"
    temp_f:      str
    feels_f:     str
    desc:        str
    wind_mph:    str
    wind_dir:    str
    rain_chance: int
    snow_chance: int
    uv_index:    str
    humidity:    str
    cloud_cover: str
    is_current:  bool = False


@dataclass
class DayForecast:
    date:              str
    desc:              str
    high_f:            str
    low_f:             str
    uv_index:          str
    rain_chance:       int
    snow_chance:       int
    sunrise:           str
    sunset:            str
    moon_phase:        str
    moon_illumination: str
    hourly:            list[HourlySlot] = field(default_factory=list)


@dataclass
class WeatherReport:
    location_label:   str
    lat:              str
    lon:              str
    population:       str
    queried_location: str
    desc:             str
    temp_f:           str
    feels_f:          str
    humidity:         str
    wind_mph:         str
    wind_dir:         str
    wind_deg:         str
    uv_index:         str
    visibility_mi:    str
    pressure_mb:      str
    precip_in:        str
    cloud_cover:      str
    obs_time:         str
    tz_name: str = ""
    forecast:         list[DayForecast] = field(default_factory=list)


@dataclass
class MemorySnapshot:
    date:             str
    location_key:     str
    location_label:   str
    queried_location: str
    desc:             str
    temp_f:           str
    feels_f:          str
    humidity:         str
    wind_mph:         str
    wind_dir:         str
    uv_index:         str
    precip_in:        str
    cloud_cover:      str
    obs_time:         str


@dataclass
class WeatherComparison:
    found:         bool
    previous_date: str       = ""
    summary:       str       = ""
    bullets:       list[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════
#  Claude Code-style UI
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Step:
    """One pipeline step rendered as a status line."""
    name:   str
    state:  str   = "pending"   # pending | running | done | error | skipped
    detail: str   = ""
    elapsed_ms: float = 0.0
    cached: bool  = False
    _t0: float    = field(default_factory=time.monotonic, repr=False)

    def start(self) -> "Step":
        self.state = "running"
        self._t0   = time.monotonic()
        return self

    def done(self, detail: str = "", cached: bool = False) -> "Step":
        self.elapsed_ms = (time.monotonic() - self._t0) * 1000
        self.state  = "done"
        self.detail = detail
        self.cached = cached
        return self

    def error(self, detail: str = "") -> "Step":
        self.elapsed_ms = (time.monotonic() - self._t0) * 1000
        self.state  = "error"
        self.detail = detail
        return self

    def skip(self, detail: str = "") -> "Step":
        self.state  = "skipped"
        self.detail = detail
        return self

    def render(self, sp_frame: int) -> Text:
        t = Text()
        t.append("  ")
        if self.state == "running":
            t.append(SPINNER_FRAMES[sp_frame % len(SPINNER_FRAMES)], "bold cyan")
        elif self.state == "done":
            t.append("✓", "bold green")
        elif self.state == "error":
            t.append("✗", "bold red")
        elif self.state == "skipped":
            t.append("–", "dim")
        else:
            t.append("·", "dim")

        col = "white" if self.state not in ("pending",) else "grey50"
        label = f"  {self.name}"
        t.append(f"{label:<24}", col)

        if self.detail:
            preview = self.detail[:52]
            t.append(preview, "dim")

        if self.elapsed_ms:
            elapsed = self.elapsed_ms
            ts = f"{elapsed:.0f}ms" if elapsed < 2000 else f"{elapsed / 1000:.1f}s"
            padding = max(1, 58 - len(self.name) - len(self.detail[:52]))
            t.append(" " * padding)
            t.append(ts, "dim")
            if self.cached:
                t.append("  ⚡", "yellow")
        elif self.state == "running":
            t.append(f"  {(time.monotonic()-self._t0):.1f}s", "dim")

        return t


class UIState:
    """Owns all mutable display state; thread-safe via a reentrant lock."""

    def __init__(self) -> None:
        self._lock         = threading.RLock()
        self.steps:   list[Step]              = []
        self.stream_chunks: list[str]         = []
        self.usage:   dict[str, TokenUsage]   = {}
        self.running: bool                    = True
        self._frame:  int                     = 0

    # ── mutation helpers ──────────────────────────────────────────────

    def add_step(self, name: str) -> Step:
        s = Step(name=name)
        with self._lock:
            self.steps.append(s)
        return s

    def push_chunk(self, piece: str) -> None:
        with self._lock:
            full = "".join(self.stream_chunks) + piece
            self.stream_chunks = full.split("\n")[-MAX_STREAM_LINES:]

    def set_usage(self, phase: str, u: TokenUsage) -> None:
        with self._lock:
            self.usage[phase] = u

    def clear_stream(self) -> None:
        with self._lock:
            self.stream_chunks = []

    # ── renderer ─────────────────────────────────────────────────────

    def render(self) -> RenderableType:
        with self._lock:
            self._frame = (self._frame + 1) % len(SPINNER_FRAMES)
            frame = self._frame
            steps = list(self.steps)
            chunks = list(self.stream_chunks)
            usage  = dict(self.usage)
            running = self.running

        parts: list[Any] = []

        # ── header ────────────────────────────────────────────────────
        hdr = Text()
        hdr.append("◆", "bold cyan")
        hdr.append(" weather-agent", "bold white")
        hdr.append(f"   {AGENT_MODEL}", "dim")
        hdr.append(" · ", "dim")
        hdr.append(CHECKER_MODEL, "dim")
        dot = "●" if running else "◉"
        hdr.append(f"   {dot}", "green" if running else "dim")
        parts.append(hdr)
        parts.append(Text(""))

        # ── steps ─────────────────────────────────────────────────────
        for s in steps:
            parts.append(s.render(frame))

        # ── streaming output ──────────────────────────────────────────
        if any(c.strip() for c in chunks):
            parts.append(Text(""))
            parts.append(Text("  " + "─" * 64, "dim"))
            for i, line in enumerate(chunks):
                row = Text("  ")
                row.append(line, "white")
                if i == len(chunks) - 1 and running:
                    row.append("▌", "blink bold cyan")
                parts.append(row)
            parts.append(Text("  " + "─" * 64, "dim"))

        # ── footer ────────────────────────────────────────────────────
        parts.append(Text(""))
        total_tok = sum(u.total_tokens for u in usage.values())
        gen_tok   = sum(u.completion_tokens for u in usage.values())
        foot = Text("  ")
        foot.append("esc", "bold dim")
        foot.append(" to interrupt", "dim")
        if total_tok:
            foot.append(f"  ·  prompt {total_tok - gen_tok:,}  gen {gen_tok:,}  total {total_tok:,}", "dim")
        active = next((s for s in steps if s.state == "running"), None)
        if active:
            foot.append(f"  ·  {time.monotonic() - active._t0:.1f}s", "dim")
        parts.append(foot)

        return Group(*parts)


# ══════════════════════════════════════════════════════════════════════
#  Ollama helpers
# ══════════════════════════════════════════════════════════════════════

def _wait_ollama(client: Client, timeout: int = 30) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            client.list()
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"Ollama not reachable after {timeout}s")


def _ensure_model(client: Client, model: str) -> None:
    try:
        client.show(model)
    except ResponseError as exc:
        if getattr(exc, "status_code", None) == 404 or "not found" in str(exc).lower():
            print(f"  ↓ Pulling {model}…")
            client.pull(model)
        else:
            raise


def _warmup(client: Client, model: str) -> None:
    """Send a trivial request so the model is resident in VRAM before the real call."""
    try:
        client.chat(model=model,
                    messages=[{"role": "user", "content": "hi"}],
                    options={"num_predict": 1, "num_gpu": 99})
    except Exception:
        pass


def setup_ollama() -> Client:
    client = Client(host=OLLAMA_URL)
    try:
        client.list()
    except Exception:
        subprocess.Popen(["ollama", "serve"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _wait_ollama(client)
    for m in {AGENT_MODEL, CHECKER_MODEL}:
        _ensure_model(client, m)
    return client


def _get(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _msg_content(resp: Any) -> str:
    msg = _get(resp, "message", {})
    c = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
    return c or ""


def _usage_from(resp: Any) -> TokenUsage:
    return TokenUsage(
        prompt_tokens     = int(_get(resp, "prompt_eval_count", 0) or 0),
        completion_tokens = int(_get(resp, "eval_count", 0) or 0),
        total_duration_ms = float(_get(resp, "total_duration", 0) or 0) / 1_000_000,
        estimated         = False,
    )


def _est_tokens(text: str) -> int:
    return max(1, len(text.strip()) // 4)


def _stream_chat(
    client: Client,
    model: str,
    messages: list[dict],
    ui: UIState,
    refresh: Any,
    phase: str,
    temperature: float = 0.25,
) -> tuple[str, TokenUsage]:
    ui.clear_stream()
    est = TokenUsage(
        prompt_tokens = sum(_est_tokens(
            m.get("content", "") if isinstance(m.get("content"), str)
            else json.dumps(m.get("content", ""))
        ) for m in messages),
        completion_tokens = 0,
        estimated = True,
    )
    ui.set_usage(phase, est)

    parts: list[str] = []
    final_u = est

    for chunk in client.chat(model=model, messages=messages, stream=True,
                              options={"temperature": temperature, "num_gpu": 99}):
        piece = _msg_content(chunk)
        if piece:
            parts.append(piece)
            ui.push_chunk(piece)
            cur = ui.usage.get(phase, est)
            cur.completion_tokens = _est_tokens("".join(parts))
            ui.set_usage(phase, cur)
        if _get(chunk, "eval_count", None) is not None:
            final_u = _usage_from(chunk)
        refresh()

    if final_u.estimated and parts:
        final_u.completion_tokens = _est_tokens("".join(parts))
    ui.set_usage(phase, final_u)
    refresh()
    return "".join(parts).strip(), final_u


def _chat_json(
    client: Client,
    model: str,
    system: str,
    user: str,
    ui: UIState,
    refresh: Any,
    phase: str,
    retries: int = 2,
) -> tuple[dict, TokenUsage]:
    last = ""
    u = TokenUsage()
    for attempt in range(retries + 1):
        resp = client.chat(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
            options={"temperature": 0.05, "num_gpu": 99},
        )
        u = _usage_from(resp)
        ui.set_usage(phase, u); refresh()
        raw = re.sub(r"```(?:json)?|```", "", _msg_content(resp).strip()).strip()
        last = raw
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group()), u
            except json.JSONDecodeError:
                pass
        if attempt < retries:
            user += "\n\nReturn ONLY the JSON object. No prose."
    raise ValueError(f"No valid JSON from {model}: {last!r}")


# ══════════════════════════════════════════════════════════════════════
#  Geocoding + cache
# ══════════════════════════════════════════════════════════════════════

_geocache: dict[str, tuple[float, float]] = {}
_geocache_lock = threading.Lock()


def _load_geocache() -> None:
    global _geocache
    if GEOCACHE_PATH.exists():
        try:
            raw = json.loads(GEOCACHE_PATH.read_text())
            _geocache = {k: tuple(v) for k, v in raw.items()}  # type: ignore
        except Exception:
            _geocache = {}


def _save_geocache() -> None:
    GEOCACHE_PATH.write_text(
        json.dumps({k: list(v) for k, v in _geocache.items()}, indent=2)
    )


def geocode(location: str) -> Optional[tuple[float, float]]:
    key = location.lower().strip()
    with _geocache_lock:
        if key in _geocache:
            return _geocache[key]

    url = (
        "https://nominatim.openstreetmap.org/search"
        f"?q={quote(location)}&format=json&limit=1&addressdetails=0"
    )
    req = Request(url, headers={"User-Agent": "weather-agent/2.1 (local)"})
    try:
        with urlopen(req, timeout=DEFAULT_TIMEOUT) as resp:
            results = json.loads(resp.read().decode())
        if results:
            coords: tuple[float, float] = (float(results[0]["lat"]), float(results[0]["lon"]))
            with _geocache_lock:
                _geocache[key] = coords
                _save_geocache()
            return coords
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════════════
#  Weather fetch + parse (with hourly slots)
# ══════════════════════════════════════════════════════════════════════

def _fv(lst: list, key: str, fb: str = "N/A") -> str:
    if isinstance(lst, list) and lst:
        return str(lst[0].get(key) or fb).strip()
    return fb


def _parse_time(raw: str) -> str:
    """wttr.in time like '600' → '06:00'."""
    try:
        h = int(raw) // 100
        return f"{h:02d}:00"
    except Exception:
        return raw


def _parse_hourly(day: dict, today: bool) -> list[HourlySlot]:
    slots: list[HourlySlot] = []
    now_h = datetime.now().hour if today else -1
    for h in day.get("hourly", []):
        label  = _parse_time(str(h.get("time", "0")))
        slot_h = int(h.get("time", 0)) // 100
        # mark the slot whose window covers the current hour
        is_now = today and (slot_h <= now_h < slot_h + 3)
        slots.append(HourlySlot(
            time_label  = label,
            temp_f      = h.get("tempF",          "N/A"),
            feels_f     = h.get("FeelsLikeF",     "N/A"),
            desc        = _fv(h.get("weatherDesc", [{"value": ""}]), "value", ""),
            wind_mph    = h.get("windspeedMiles",  "N/A"),
            wind_dir    = h.get("winddir16Point",  ""),
            rain_chance = int(h.get("chanceofrain",  0)),
            snow_chance = int(h.get("chanceofsnow",  0)),
            uv_index    = str(h.get("uvIndex",     "")),
            humidity    = h.get("humidity",        "N/A"),
            cloud_cover = h.get("cloudcover",      "N/A"),
            is_current  = is_now,
        ))
    return slots


# -- Timezone helpers --------------------------------------------------------

_tf_instance = None


def _get_tz(lat: str, lon: str) -> str:
    """Return IANA timezone name for coordinates, e.g. 'America/New_York'.

    Requires:  pip install timezonefinder
    Falls back to a rough longitude / 15 UTC-offset when absent.
    """
    global _tf_instance
    try:
        if _tf_instance is None:
            from timezonefinder import TimezoneFinder
            _tf_instance = TimezoneFinder()
        return _tf_instance.timezone_at(lat=float(lat), lng=float(lon)) or 'UTC'
    except ImportError:
        try:
            offset_h = round(float(lon) / 15)
            return f'Etc/GMT{-offset_h:+d}'
        except Exception:
            return 'UTC'
    except Exception:
        return 'UTC'


def _obs_to_local(obs_str: str, tz_name: str) -> str:
    """Convert wttr.in observation_time ('07:00 PM', UTC) to local time.

    wttr.in provides only a time component with no date, so we anchor
    to today's UTC date before converting.
    """
    if not obs_str:
        return ""
    if not tz_name or tz_name == 'UTC':
        return obs_str.strip() + " UTC"
    try:
        from zoneinfo import ZoneInfo
        t      = datetime.strptime(obs_str.strip(), "%I:%M %p")
        utc_dt = datetime.now(tz=timezone.utc).replace(
            hour=t.hour, minute=t.minute, second=0, microsecond=0)
        local  = utc_dt.astimezone(ZoneInfo(tz_name))
        return local.strftime("%I:%M %p %Z").lstrip("0")
    except Exception:
        return obs_str.strip() + " UTC"


def parse_weather(data: dict, queried: str) -> WeatherReport:
    cur   = (data.get("current_condition") or [{}])[0]
    near  = (data.get("nearest_area")      or [{}])[0]
    days  =  data.get("weather") or []

    area    = _fv(near.get("areaName",  [{"value": queried}]), "value", queried)
    region  = _fv(near.get("region",    []), "value", "")
    country = _fv(near.get("country",   []), "value", "")
    parts   = [v for v in (area, region, country) if v and v != "N/A"]
    label   = ", ".join(dict.fromkeys(parts))

    forecast: list[DayForecast] = []
    today_date = Date.today().isoformat()
    for i, day in enumerate(days[:3]):
        astro  = (day.get("astronomy") or [{}])[0]
        hourly = day.get("hourly") or []
        mid    = hourly[len(hourly) // 2] if hourly else {}
        is_today = (day.get("date", "") == today_date) or (i == 0)
        forecast.append(DayForecast(
            date              = day.get("date", ""),
            desc              = _fv(mid.get("weatherDesc", [{"value": ""}]), "value", ""),
            high_f            = day.get("maxtempF", "N/A"),
            low_f             = day.get("mintempF", "N/A"),
            uv_index          = str(day.get("uvIndex", "")),
            rain_chance       = max((int(h.get("chanceofrain", 0)) for h in hourly), default=0),
            snow_chance       = max((int(h.get("chanceofsnow", 0)) for h in hourly), default=0),
            sunrise           = astro.get("sunrise",           ""),
            sunset            = astro.get("sunset",            ""),
            moon_phase        = astro.get("moon_phase",        ""),
            moon_illumination = astro.get("moon_illumination", ""),
            hourly            = _parse_hourly(day, is_today),
        ))

    return WeatherReport(
        location_label   = label,
        lat              = near.get("latitude",   ""),
        lon              = near.get("longitude",  ""),
        population       = near.get("population", ""),
        queried_location = queried,
        desc             = _fv(cur.get("weatherDesc", [{"value": "Unknown"}]), "value", "Unknown"),
        temp_f           = cur.get("temp_F",          "N/A"),
        feels_f          = cur.get("FeelsLikeF",      "N/A"),
        humidity         = cur.get("humidity",        "N/A"),
        wind_mph         = cur.get("windspeedMiles",  "N/A"),
        wind_dir         = cur.get("winddir16Point",  ""),
        wind_deg         = cur.get("winddirDegree",   ""),
        uv_index         = cur.get("uvIndex",         ""),
        visibility_mi    = cur.get("visibilityMiles", "N/A"),
        pressure_mb      = cur.get("pressure",        "N/A"),
        precip_in        = cur.get("precipInches",    "0.0"),
        cloud_cover      = cur.get("cloudcover",      "N/A"),
        obs_time         = cur.get("observation_time",""),
        tz_name       = _get_tz(
            near.get("latitude",  ""),
            near.get("longitude", ""),
        ),
        forecast         = forecast,
    )


def fetch_weather(location: str, coords: Optional[tuple[float, float]]) -> WeatherReport:
    if coords:
        query_str = f"{coords[0]:.4f},{coords[1]:.4f}"
    else:
        query_str = location.strip().rstrip(" ?.")

    url = f"https://wttr.in/{quote(query_str, safe=',')}?format=j1"
    req = Request(url, headers={"User-Agent": "curl/8.0", "Accept": "application/json"})
    with urlopen(req, timeout=DEFAULT_TIMEOUT) as resp:
        data = json.loads(resp.read().decode())
    return parse_weather(data, location.strip().rstrip(" ?."))


def compact_summary(r: WeatherReport) -> str:
    s = (
        f"{r.queried_location}: {r.desc}, {r.temp_f}°F (feels {r.feels_f}°F), "
        f"humidity {r.humidity}%, wind {r.wind_dir} {r.wind_mph} mph, "
        f"UV {r.uv_index}, visibility {r.visibility_mi} mi, "
        f"pressure {r.pressure_mb} mb, precip {r.precip_in} in."
    )
    if r.forecast:
        d = r.forecast[0]
        s += f" Today H {d.high_f}°F / L {d.low_f}°F, rain {d.rain_chance}%."
    return s


# ══════════════════════════════════════════════════════════════════════
#  Memory
# ══════════════════════════════════════════════════════════════════════

def _loc_key(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")


def load_memory() -> list[MemorySnapshot]:
    if not MEMORY_PATH.exists():
        return []
    try:
        raw = json.loads(MEMORY_PATH.read_text())
        snaps = []
        for item in raw.get("snapshots", []):
            item.setdefault("queried_location", item.get("location_label", ""))
            snaps.append(MemorySnapshot(**item))
        return snaps
    except Exception:
        return []


def save_memory(snapshots: list[MemorySnapshot]) -> None:
    MEMORY_PATH.write_text(
        json.dumps({"snapshots": [asdict(s) for s in snapshots]}, indent=2)
    )


def snapshot_from_report(r: WeatherReport) -> MemorySnapshot:
    return MemorySnapshot(
        date=Date.today().isoformat(),
        location_key=_loc_key(r.queried_location or r.location_label or "unknown"),
        location_label=r.location_label,
        queried_location=r.queried_location,
        desc=r.desc, temp_f=r.temp_f, feels_f=r.feels_f,
        humidity=r.humidity, wind_mph=r.wind_mph, wind_dir=r.wind_dir,
        uv_index=r.uv_index, precip_in=r.precip_in,
        cloud_cover=r.cloud_cover, obs_time=r.obs_time,
    )


def find_prev_snapshot(r: WeatherReport,
                       snaps: list[MemorySnapshot]) -> Optional[MemorySnapshot]:
    key   = _loc_key(r.queried_location or r.location_label or "unknown")
    today = Date.today()
    cands: list[tuple[Date, MemorySnapshot]] = []
    for s in snaps:
        if s.location_key != key:
            continue
        try:
            d = Date.fromisoformat(s.date)
        except Exception:
            continue
        if d < today:
            cands.append((d, s))
    if not cands:
        return None
    exact = [s for d, s in cands if d == today - timedelta(days=1)]
    if exact:
        return exact[-1]
    cands.sort(key=lambda x: x[0], reverse=True)
    return cands[0][1]


def upsert_snapshot(snaps: list[MemorySnapshot],
                    cur: MemorySnapshot) -> list[MemorySnapshot]:
    for i, s in enumerate(snaps):
        if s.location_key == cur.location_key and s.date == cur.date:
            snaps[i] = cur
            return snaps[-500:]
    snaps.append(cur)
    snaps.sort(key=lambda s: (s.location_key, s.date))
    return snaps[-500:]


def _safe_int(v: Any) -> Optional[int]:
    try:
        return int(str(v).strip())
    except Exception:
        return None


def compare(prev: Optional[MemorySnapshot], cur: WeatherReport) -> WeatherComparison:
    if prev is None:
        return WeatherComparison(found=False,
                                 summary="No prior weather memory for this location.")
    bullets: list[str] = []

    def _d(now: Any, old: Any, label: str, unit: str) -> None:
        a, b = _safe_int(now), _safe_int(old)
        if a is None or b is None:
            return
        diff = a - b
        if diff:
            bullets.append(f"{label} {'up' if diff>0 else 'down'} {abs(diff)}{unit}")
        else:
            bullets.append(f"{label} unchanged")

    _d(cur.temp_f,   prev.temp_f,   "Temp",      "°F")
    _d(cur.feels_f,  prev.feels_f,  "Feels-like","°F")
    _d(cur.humidity, prev.humidity, "Humidity",  " pp")
    _d(cur.wind_mph, prev.wind_mph, "Wind",      " mph")
    if cur.desc.lower() != prev.desc.lower():
        bullets.append(f"Conditions: '{prev.desc}' → '{cur.desc}'")
    if cur.wind_dir and prev.wind_dir and cur.wind_dir != prev.wind_dir:
        bullets.append(f"Wind direction: {prev.wind_dir} → {cur.wind_dir}")
    if not bullets:
        bullets.append(f"Conditions similar to {prev.date}")

    return WeatherComparison(
        found=True, previous_date=prev.date,
        summary=". ".join(bullets) + ".",
        bullets=bullets,
    )


# ══════════════════════════════════════════════════════════════════════
#  Checkers
# ══════════════════════════════════════════════════════════════════════

def check_prompt(client: Client, query: str, ui: UIState, refresh: Any) -> dict:
    d, _ = _chat_json(client, CHECKER_MODEL, CLASSIFIER_SYSTEM,
                       f"Classify: {query!r}", ui, refresh, "classify")
    return d


def check_result(client: Client, query: str, answer: str,
                 queried: str, grid: str, ui: UIState, refresh: Any) -> dict:
    ctx = (f"Query: {query!r}\nUser location: {queried!r}\n"
           f"wttr.in grid: {grid!r}\nAnswer:\n{answer}")
    d, _ = _chat_json(client, CHECKER_MODEL, RESULT_CHECK_SYSTEM,
                       ctx, ui, refresh, "validate")
    return d


# ══════════════════════════════════════════════════════════════════════
#  Location extraction (regex fallback)
# ══════════════════════════════════════════════════════════════════════

_LOC_RE = [
    r"(?:current )?weather (?:in|for) (.+)$",
    r"(?:forecast|temperature|radar|satellite) (?:in|for) (.+)$",
    r"what(?:'s| is) (?:the )?(?:current )?weather (?:in|for) (.+)$",
    r"how (?:hot|cold|warm) is it in (.+)$",
    r"(?:rain|snow|storm) (?:in|near) (.+)$",
]

def extract_location(query: str) -> str:
    q = query.strip().rstrip(" ?.")
    for pat in _LOC_RE:
        m = re.search(pat, q, re.IGNORECASE)
        if m:
            return q[m.start(1):].strip().rstrip(" ?.")
    return ""


# ══════════════════════════════════════════════════════════════════════
#  Context builder (feeds the LLM)
# ══════════════════════════════════════════════════════════════════════

def _coord(lat: str, lon: str) -> str:
    try:
        la, lo = float(lat), float(lon)
        return (f"{abs(la):.4f}°{'N' if la>=0 else 'S'}, "
                f"{abs(lo):.4f}°{'E' if lo>=0 else 'W'}")
    except Exception:
        return f"{lat}, {lon}"


def _uv_label(uv: str) -> str:
    try:
        v = int(uv)
        w = ("Low","Moderate","High","Very High","Extreme")[
            0 if v<=2 else 1 if v<=5 else 2 if v<=7 else 3 if v<=10 else 4]
        return f"{v} {w}"
    except Exception:
        return uv or "N/A"


def build_context(query: str, r: WeatherReport,
                  comp: WeatherComparison, vision: str) -> str:
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

    # Include today's hourly data (first day)
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
#  Weather card (Rich, printed after live UI closes)
# ══════════════════════════════════════════════════════════════════════

_COND_MAP = [
    ("thunder","⛈","bold yellow"), ("blizzard","❄","bold cyan"), ("snow","❄","bold cyan"),
    ("sleet","🌨","cyan"), ("drizzle","🌦","sky_blue1"), ("rain","🌧","sky_blue1"),
    ("shower","🌦","sky_blue1"), ("fog","🌫","grey70"), ("mist","🌫","grey70"),
    ("haze","🌫","grey70"), ("overcast","☁","grey70"), ("cloudy","🌥","grey74"),
    ("partly","⛅","yellow3"), ("clear","☀","bright_yellow"), ("sunny","☀","bright_yellow"),
    ("wind","💨","cyan"),
]

def _cstyle(desc: str) -> tuple[str, str]:
    d = desc.lower()
    for kw, em, col in _COND_MAP:
        if kw in d:
            return em, col
    return "🌡", "white"


def _uv_col(uv: str) -> str:
    try:
        v = int(uv)
        return ("green","yellow3","orange3","red","bright_red")[
            0 if v<=2 else 1 if v<=5 else 2 if v<=7 else 3 if v<=10 else 4]
    except Exception:
        return "white"


def _bar(val: str, step: int = 10, width: int = 10) -> str:
    try:
        v = int(val)
        filled = min(width, max(0, v // step))
        return "█" * filled + "░" * (width - filled)
    except Exception:
        return "?" * width


def _moon_em(phase: str) -> str:
    p = phase.lower()
    m = {"new":"🌑","waxing crescent":"🌒","first quarter":"🌓","waxing gibbous":"🌔",
         "full":"🌕","waning gibbous":"🌖","last quarter":"🌗","waning crescent":"🌘"}
    for k, v in m.items():
        if k in p:
            return v
    return "🌙"


def _day_label(date_str: str, i: int) -> str:
    try:
        d = Date.fromisoformat(date_str)
        s = d.strftime("%a %b %d").replace(" 0", " ")
        return ("Today", "Tomorrow")[i] + f" ({s})" if i < 2 else s
    except Exception:
        return date_str


def _delta_text(now: Any, prev: Any, unit: str = "") -> Text:
    t = Text()
    try:
        d = int(str(now)) - int(str(prev))
        if d > 0:   t.append(f"+{d}{unit}", "green")
        elif d < 0: t.append(f"{d}{unit}", "red")
        else:       t.append(f"±0{unit}", "grey50")
    except Exception:
        pass
    return t


def _term_width(console: Console) -> int:
    try:
        return max(40, int(console.size.width))
    except Exception:
        return shutil.get_terminal_size((100, 30)).columns


def _compact_mode(console: Console) -> bool:
    w = _term_width(console)
    return w < 110


def _tiny_mode(console: Console) -> bool:
    w = _term_width(console)
    return w < 78


def _safe_text(value: Any, fallback: str = "—") -> str:
    if value is None:
        return fallback
    s = str(value).strip()
    return s if s else fallback


def _limit(s: str, width: int) -> str:
    s = str(s)
    if width <= 3:
        return s[:width]
    return s if len(s) <= width else s[: width - 1] + "…"


def _weather_ascii(desc: str, is_day: bool = True) -> str:
    d = (desc or "").lower()

    if "thunder" in d:
        return (
            "   .-.   \n"
            "  (⚡ )  \n"
            " (___) ) \n"
            "  ʻ ʻ ʻ  "
        )
    if "snow" in d or "blizzard" in d or "sleet" in d or "ice" in d:
        return (
            "   .-.   \n"
            "  (   ). \n"
            " (___(__)\n"
            "  * * *  "
        )
    if "rain" in d or "drizzle" in d or "shower" in d:
        return (
            "   .-.   \n"
            "  (   ). \n"
            " (___(__)\n"
            "  ʻ ʻ ʻ  "
        )
    if "fog" in d or "mist" in d or "haze" in d:
        return (
            " _ - _ - \n"
            "  _ - _  \n"
            " _ - _ - \n"
            "  - _ -  "
        )
    if "cloud" in d or "overcast" in d:
        return (
            "    .--.  \n"
            " .-(    ).\n"
            "(___.__)__)"
        )
    if is_day:
        return (
            "   \\ | /  \n"
            " '-.☼.-' \n"
            "   / | \\  "
        )
    return (
        "   _..._  \n"
        " .:::::::. \n"
        " ::::::::: \n"
        " `:::::::' "
    )


def _wind_arrow(deg: Optional[int]) -> str:
    if deg is None:
        return "•"
    dirs = [
        (22, "↑"), (67, "↗"), (112, "→"), (157, "↘"),
        (202, "↓"), (247, "↙"), (292, "←"), (337, "↖"), (361, "↑"),
    ]
    for limit, arrow in dirs:
        if deg < limit:
            return arrow
    return "•"


def _feels_comment(temp_f: int, feels_f: int) -> str:
    delta = feels_f - temp_f
    if delta >= 5:
        return "feels warmer than air"
    if delta <= -5:
        return "feels cooler than air"
    return "feels near actual temp"


def _pressure_trend_label(pressure_mb: Optional[int]) -> str:
    if pressure_mb is None:
        return "unknown"
    if pressure_mb >= 1022:
        return "higher / more settled"
    if pressure_mb <= 1008:
        return "lower / less settled"
    return "fairly neutral"


def _hourly_temp_trend(hourly) -> str:
    vals = [getattr(h, "temp_f", None) for h in hourly if getattr(h, "temp_f", None) is not None]
    if len(vals) < 2:
        return "insufficient data"
    if vals[-1] - vals[0] >= 6:
        return "warming through the day"
    if vals[0] - vals[-1] >= 6:
        return "cooling through the day"
    return "temperatures stay fairly steady"


def _hourly_precip_signal(hourly) -> str:
    rainy = []
    for h in hourly:
        chance = getattr(h, "chance_rain", 0) or 0
        desc = (getattr(h, "desc", "") or "").lower()
        if chance >= 40 or "rain" in desc or "drizzle" in desc or "shower" in desc or "storm" in desc:
            rainy.append(h)
    if not rainy:
        return "no strong rain signal"
    first = rainy[0]
    return f"rain risk builds around {_safe_text(getattr(first, 'time_label', None), 'later')}"


def _prediction_notes(r) -> list[str]:
    notes = []

    if r.forecast:
        today = r.forecast[0]
        hourly = getattr(today, "hourly", []) or []
        notes.append(_hourly_temp_trend(hourly))
        notes.append(_hourly_precip_signal(hourly))

        if getattr(today, "uv_index", None) is not None:
            uv = today.uv_index
            if uv >= 8:
                notes.append("very high UV near midday")
            elif uv >= 6:
                notes.append("high UV near midday")
            elif uv >= 3:
                notes.append("moderate UV during the day")

    notes.append(_pressure_trend_label(getattr(r, "pressure_mb", None)))
    return [n for n in notes if n]

def _daypart_label(hour_24: int) -> str:
    if hour_24 < 6:
        return "overnight"
    if hour_24 < 12:
        return "morning"
    if hour_24 < 18:
        return "afternoon"
    return "evening"


def _slot_hour(slot) -> Optional[int]:
    t = getattr(slot, "time", None)
    if t is None:
        return None
    try:
        s = str(t).strip()
        if s.isdigit():
            n = int(s)
            if n == 2400:
                return 0
            return max(0, min(23, n // 100))
    except Exception:
        pass
    return None


def _normalize_hourly_slot(slot):
    hour = _slot_hour(slot)
    if hour is not None:
        slot.time_label = f"{hour:02d}:00"
        slot.daypart = _daypart_label(hour)
    else:
        slot.time_label = _safe_text(getattr(slot, "time", None))
        slot.daypart = "period"
    return slot


def _normalize_report(report):
    if not report or not getattr(report, "forecast", None):
        return report

    for day in report.forecast:
        hourly = getattr(day, "hourly", None) or []
        norm = []
        for slot in hourly:
            norm.append(_normalize_hourly_slot(slot))
        day.hourly = norm

        if not getattr(day, "best_desc", None) and norm:
            day.best_desc = max(
                norm,
                key=lambda s: (getattr(s, "chance_rain", 0) or 0, getattr(s, "humidity", 0) or 0)
            ).desc

    return report




def print_weather_card(
    console: Console,
    r: WeatherReport,
    comp: WeatherComparison,
    prev_snap: Optional[MemorySnapshot],
    answer: str,
    vision: str,
) -> None:

    em, ccol = _cstyle(r.desc)

    # ── location header ───────────────────────────────────────────────
    hdr = Text(justify="center")
    hdr.append(f"\n  {em}  ", f"bold {ccol}")
    hdr.append(r.queried_location, "bold white")
    if r.location_label and r.location_label != r.queried_location:
        hdr.append(f"\n  wttr.in grid: {r.location_label}", "grey50")
    if r.lat and r.lon:
        hdr.append(f"\n  {_coord(r.lat, r.lon)}", "grey50")
        if r.population and r.population.isdigit():
            hdr.append(f"   pop. {int(r.population):,}", "grey50")
    if r.obs_time:
        hdr.append(f"\n  Observed {_obs_to_local(r.obs_time, r.tz_name)}", "grey50")
    hdr.append("\n")
    console.print(Panel(hdr, border_style="blue", box=box.DOUBLE_EDGE))

    # ── current conditions ────────────────────────────────────────────
    L = Table.grid(padding=(0, 2)); L.add_column(style="grey58", min_width=15); L.add_column(style="bold white")
    R = Table.grid(padding=(0, 2)); R.add_column(style="grey58", min_width=15); R.add_column(style="bold white")

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

    console.print(Panel(Columns([L, R], equal=True, expand=True),
                        title="[bold]Current Conditions[/bold]",
                        border_style=ccol, box=box.ROUNDED))

    # ── today's hourly table ──────────────────────────────────────────
    if r.forecast and r.forecast[0].hourly:
        ht = Table(box=box.SIMPLE_HEAD, expand=True,
                   show_header=True, header_style="bold white")
        ht.add_column("Time",       style="bold white", min_width=6)
        ht.add_column("Conditions", min_width=18)
        ht.add_column("Temp",       justify="right", style="bold")
        ht.add_column("Feels",      justify="right", style="dim white")
        ht.add_column("Hum",        justify="right", style="cyan")
        ht.add_column("Wind",       justify="right", style="sky_blue1")
        ht.add_column("UV",         justify="center")
        ht.add_column("🌧",         justify="right", style="sky_blue1")
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
            except Exception:
                tc = "white"
            uvt2 = Text(_uv_label(slot.uv_index), _uv_col(slot.uv_index))
            ht.add_row(
                slot.time_label + now_mark,
                Text(f"{se} {slot.desc}", sc),
                Text(f"{slot.temp_f}°F", tc),
                f"{slot.feels_f}°F",
                f"{slot.humidity}%",
                f"{slot.wind_dir} {slot.wind_mph}",
                uvt2,
                f"{slot.rain_chance}%",
                f"{slot.snow_chance}%",
                f"{slot.cloud_cover}%",
                style=row_style,
            )

        console.print(Panel(ht, title=f"[bold]Today's Hourly Forecast[/bold]",
                            border_style="cyan", box=box.ROUNDED))

    # ── prior-day delta ───────────────────────────────────────────────
    if comp.found and prev_snap:
        dt = Table(box=box.SIMPLE, expand=True, show_header=True,
                   header_style="bold grey50")
        dt.add_column("Field",    style="grey58",   min_width=14)
        dt.add_column("Previous", justify="right",  style="white")
        dt.add_column("Now",      justify="right",  style="bold white")
        dt.add_column("Δ",        justify="right")

        rows = [("🌡 Temp °F",  prev_snap.temp_f,  r.temp_f,   "°F"),
                ("🌡 Feels °F", prev_snap.feels_f, r.feels_f,  "°F"),
                ("💦 Humidity", prev_snap.humidity, r.humidity, "%"),
                ("🌬 Wind mph", prev_snap.wind_mph, r.wind_mph, ""),
                ("☁ Cond.",    prev_snap.desc,     r.desc,     None),
                ("🌥 Cloud %",  prev_snap.cloud_cover, r.cloud_cover, "%")]
        for label, old, new, unit in rows:
            if unit is None:
                if old != new: delta = Text("changed", "yellow")
                else:          delta = Text("same",    "grey50")
            else:
                delta = _delta_text(new, old, unit)
            dt.add_row(label, str(old), str(new), delta)

        console.print(Panel(dt, title=f"[bold]vs {comp.previous_date}[/bold]",
                            border_style="magenta", box=box.ROUNDED))
    elif not comp.found:
        console.print(Panel(
            Text("No prior snapshot — run again tomorrow to compare.", "grey50"),
            title="[bold]Prior-Day Comparison[/bold]",
            border_style="grey35", box=box.ROUNDED))

    # ── 3-day forecast ────────────────────────────────────────────────
    if r.forecast:
        ft = Table(box=box.SIMPLE_HEAD, expand=True,
                   show_header=True, header_style="bold white")
        ft.add_column("Day",       style="bold white", min_width=18)
        ft.add_column("Conditions")
        ft.add_column("High",      justify="right", style="bold red")
        ft.add_column("Low",       justify="right", style="bold cyan")
        ft.add_column("UV",        justify="center")
        ft.add_column("🌧 Rain",   justify="right", style="sky_blue1")
        ft.add_column("❄ Snow",    justify="right", style="cyan")
        ft.add_column("Sunrise",   justify="right", style="yellow3")
        ft.add_column("Sunset",    justify="right", style="orange3")
        ft.add_column("Moon",      justify="center")

        for i, day in enumerate(r.forecast):
            fe, _ = _cstyle(day.desc)
            moon  = f"{_moon_em(day.moon_phase)} {day.moon_phase}"
            if day.moon_illumination:
                moon += f" {day.moon_illumination}%"
            ft.add_row(
                _day_label(day.date, i),
                f"{fe} {day.desc}" if day.desc else "—",
                f"{day.high_f}°F", f"{day.low_f}°F",
                Text(_uv_label(day.uv_index), _uv_col(day.uv_index)),
                f"{day.rain_chance}%", f"{day.snow_chance}%",
                day.sunrise or "—", day.sunset or "—",
                moon or "—",
            )
        console.print(Panel(ft, title="[bold]3-Day Forecast[/bold]",
                            border_style="yellow3", box=box.ROUNDED))

    # ── vision note ───────────────────────────────────────────────────
    if vision:
        console.print(Panel(Text(vision, "white"), title="[bold]🛰  Image Analysis[/bold]",
                            border_style="blue", box=box.ROUNDED))

    # ── AI answer ─────────────────────────────────────────────────────
    if answer:
        # Parse the four sections from the answer
        sections: dict[str, str] = {}
        current_key = None
        for line in answer.splitlines():
            for key in ("CURRENT:", "CHANGES:", "HOURLY:", "OUTLOOK:"):
                if line.upper().startswith(key):
                    current_key = key.rstrip(":")
                    sections[current_key] = line[len(key):].strip()
                    break
            else:
                if current_key and line.strip():
                    sections[current_key] = sections[current_key] + "\n" + line

        if sections:
            sec_table = Table.grid(padding=(0, 2))
            sec_table.add_column(style="bold cyan", min_width=10)
            sec_table.add_column(style="white")
            for key in ("CURRENT", "CHANGES", "HOURLY", "OUTLOOK"):
                if key in sections:
                    sec_table.add_row(key, sections[key].strip())
            console.print(Panel(sec_table, title=f"[bold]🤖 {AGENT_MODEL}[/bold]",
                                border_style="green", box=box.ROUNDED))
        else:
            console.print(Panel(Text(answer, "white"),
                                title=f"[bold]🤖 {AGENT_MODEL}[/bold]",
                                border_style="green", box=box.ROUNDED))


# ══════════════════════════════════════════════════════════════════════
#  Main pipeline
# ══════════════════════════════════════════════════════════════════════

def run(query: str, image_path: Optional[str]) -> int:
    console = Console()
    ui      = UIState()

    report:     Optional[WeatherReport]    = None
    comp:       Optional[WeatherComparison]= None
    prev_snap:  Optional[MemorySnapshot]   = None
    answer      = ""
    vision_note = ""
    valid       = True

    with Live(ui.render(), refresh_per_second=UI_REFRESH_HZ,
              console=console, screen=False) as live:

        def refresh() -> None:
            live.update(ui.render())

        try:
            # ── 1. setup ──────────────────────────────────────────────
            s_init = ui.add_step("connect + warmup").start(); refresh()
            client = setup_ollama()
            # Warm up both models concurrently (fire-and-forget HTTP pings)
            with ThreadPoolExecutor(max_workers=2) as warmpool:
                warmpool.submit(_warmup, client, AGENT_MODEL)
                warmpool.submit(_warmup, client, CHECKER_MODEL)
            s_init.done("Ollama ready · models warmed"); refresh()

            # ── 2. extract a location hint before firing threads ──────
            loc_hint = extract_location(query)
            if not loc_hint:
                loc_hint = query.strip().rstrip(" ?.")

            # ── 3. parallel: classify (LLM) + geocode (HTTP) ──────────
            s_cls  = ui.add_step("classify query").start(); refresh()
            s_geo  = ui.add_step("geocode").start();        refresh()

            with ThreadPoolExecutor(max_workers=2) as p:
                f_cls  = p.submit(check_prompt, client, query, ui, lambda: None)
                f_geo  = p.submit(geocode, loc_hint)

                # poll both, refreshing live UI while we wait
                while not (f_cls.done() and f_geo.done()):
                    refresh()
                    time.sleep(0.08)

                pc_raw = f_cls.result()
                coords = f_geo.result()

            is_weather = bool(pc_raw.get("is_weather_query", False))
            location   = str(pc_raw.get("location", "")).strip() or loc_hint
            confidence = float(pc_raw.get("confidence", 0.0))
            reason     = str(pc_raw.get("reason", ""))

            s_cls.done(f"{'✓ weather' if is_weather else '✗ not weather'}  ({confidence:.0%})  {reason[:40]}")
            cached = (location.lower().strip() in _geocache)
            s_geo.done(
                f"{coords[0]:.4f}°N, {coords[1]:.4f}°W" if coords else "geocode failed (using raw string)",
                cached=cached,
            )
            refresh()

            if not is_weather:
                ui.push_chunk("This doesn't appear to be a weather query.\n" + reason)
                ui.running = False; refresh()
                return 1

            # ── 4. parallel: wttr.in fetch + memory load ──────────────
            s_wx  = ui.add_step("fetch weather").start();  refresh()
            s_mem = ui.add_step("load memory").start();    refresh()

            with ThreadPoolExecutor(max_workers=2) as p2:
                f_wx  = p2.submit(fetch_weather, location, coords)
                f_mem = p2.submit(load_memory)

                while not (f_wx.done() and f_mem.done()):
                    refresh()
                    time.sleep(0.08)

                report    = f_wx.result()
                snapshots = f_mem.result()

            s_wx.done(f"{report.desc} · {report.temp_f}°F · hum {report.humidity}%")
            prev_snap  = find_prev_snapshot(report, snapshots)
            comp       = compare(prev_snap, report)
            mem_detail = f"vs {comp.previous_date}" if comp.found else "no prior data"
            s_mem.done(mem_detail); refresh()

            # ── 5. optional vision ────────────────────────────────────
            if image_path:
                img = Path(image_path).expanduser()
                if img.exists():
                    s_vis = ui.add_step("vision analysis").start(); refresh()
                    vision_note, _ = _stream_chat(
                        client, AGENT_MODEL,
                        [
                            {"role": "system", "content": VISION_SYSTEM},
                            {
                                "role": "user",
                                "content": (
                                    f"Analyse this image.\n"
                                    f"Context: {compact_summary(report)}"
                                ),
                                "images": [str(img)],
                            },
                        ],
                        ui, refresh, phase="vision", temperature=0.15,
                    )
                    s_vis.done(vision_note[:60]); refresh()
                else:
                    ui.add_step("vision analysis").skip(f"file not found: {image_path}")
                    refresh()

            # ── 6. generate answer ────────────────────────────────────
            s_ans = ui.add_step("generate answer").start(); refresh()
            ctx   = build_context(query, report, comp, vision_note)
            answer, _ = _stream_chat(
                client, AGENT_MODEL,
                [
                    {"role": "system", "content": AGENT_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"Using the data below, answer: {query}\n\n"
                            f"Use the four-section format in your instructions.\n\n"
                            f"{ctx}"
                        ),
                    },
                ],
                ui, refresh, phase="answer", temperature=0.25,
            )
            s_ans.done(answer[:60]); refresh()

            # ── 7. validate ───────────────────────────────────────────
            s_val = ui.add_step("validate result").start(); refresh()
            res   = check_result(client, query, answer,
                                 report.queried_location, report.location_label,
                                 ui, refresh)
            valid = bool(res.get("is_valid", True))
            score = float(res.get("quality_score", 1.0))
            notes = str(res.get("notes", ""))
            issues = res.get("issues", [])
            val_detail = f"valid={valid}  score={score:.0%}  {notes[:45]}"
            if issues:
                val_detail += "  issues: " + "; ".join(issues)[:30]
            s_val.done(val_detail); refresh()

            # ── 8. save memory ────────────────────────────────────────
            s_save = ui.add_step("save memory").start(); refresh()
            snapshots = upsert_snapshot(snapshots, snapshot_from_report(report))
            save_memory(snapshots)
            s_save.done(f"{MEMORY_PATH.name}"); refresh()

            ui.running = False; refresh()

        except KeyboardInterrupt:
            ui.running = False
            ui.add_step("interrupted").error("KeyboardInterrupt"); refresh()
            return 130
        except Exception as exc:
            ui.running = False
            ui.add_step("fatal error").error(str(exc)[:80]); refresh()
            raise

    # ── post-live: full weather card ───────────────────────────────────
    report = _normalize_report(report)
    if report is not None and comp is not None:
        console.print()
        console.print(Rule("[bold blue]Weather Report[/bold blue]", style="blue"))
        print_weather_card(console, report, comp, prev_snap, answer, vision_note)
        console.print(Rule(style="blue"))
        console.print()
        console.print(Panel.fit(
            f"[dim]memory:[/dim] {MEMORY_PATH}   "
            f"[dim]geocache:[/dim] {GEOCACHE_PATH}",
            title="[dim]Run complete[/dim]",
            border_style="dim",
        ))

    return 0 if valid else 2


# ══════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════

def main() -> int:
    _load_geocache()

    ap = argparse.ArgumentParser(description="Multimodal weather agent")
    ap.add_argument("query", nargs="*", help="Weather question")
    ap.add_argument("--image", metavar="PATH",
                    help="Radar / satellite / sky photo for vision analysis")
    ap.add_argument("--clear-memory", action="store_true",
                    help="Delete stored snapshots and exit")
    ap.add_argument("--clear-geocache", action="store_true",
                    help="Delete geocode cache and exit")
    args = ap.parse_args()

    if args.clear_memory:
        if MEMORY_PATH.exists():
            MEMORY_PATH.unlink()
            print(f"Memory cleared: {MEMORY_PATH}")
        else:
            print("No memory file found.")
        return 0

    if args.clear_geocache:
        if GEOCACHE_PATH.exists():
            GEOCACHE_PATH.unlink()
            print(f"Geocache cleared: {GEOCACHE_PATH}")
        else:
            print("No geocache file found.")
        return 0

    query = " ".join(args.query).strip()
    if not query:
        try:
            query = input("\n  What would you like to know?  ").strip()
        except (EOFError, KeyboardInterrupt):
            return 0

    if not query:
        print("Empty query.")
        return 1

    return run(query, args.image)


if __name__ == "__main__":
    raise SystemExit(main())
