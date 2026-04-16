#!/usr/bin/env python3
"""
harness.py — multi-agent orchestration harness · Claude Code-style UI

Models : llama3.2:3b  (sanity-check + final validation, ~2.0 GB)
         qwen2.5:3b   (task planning + structured output,  ~1.9 GB)

Pipeline
  1. sanity-check   llama3.2:3b decides if the suite can handle the query
  2. plan           qwen2.5:3b produces an ordered, tool-call plan (JSON)
  3. execute        each planned step runs sequentially; results flow forward
  4. validate       llama3.2:3b checks final output against the original intent

Register tools
  @register_tool("my_tool", "One-sentence description shown to the planner.")
  def my_tool(arg1: str, arg2: int = 0) -> str:
      ...
      return "result string"

  Reference a prior step's output in args with the placeholder {step_N}
  (e.g. {"text": "{step_0}"} becomes the result of step 0).

Usage
  python harness.py "summarise the latest Hacker News front page"
  python harness.py --list-tools
  python harness.py --dry-run "make a plan but don't execute"
  python harness.py --clear-history
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is in sys.path for robust absolute imports
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)


import argparse
import inspect
import json
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent
from typing import Any, Callable
from urllib.request import Request, urlopen

from ollama import Client, ResponseError
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

# Common imports
from src.common.ui import UIState
from src.common.ollama import setup_ollama, _stream_chat, _chat_json, _warmup
from src.common.types import TokenUsage

# ══════════════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════════════

OLLAMA_URL      = "http://localhost:11434"
CHECKER_MODEL   = "llama3.2:3b"
PLANNER_MODEL   = "qwen2.5:3b"

HISTORY_PATH    = Path.home() / ".harness_history.json"
UI_REFRESH_HZ   = 10
MAX_STREAM_LINES = 8
DEFAULT_TIMEOUT  = 30

# ══════════════════════════════════════════════════════════════════════
#  Prompts
# ══════════════════════════════════════════════════════════════════════

SANITY_SYSTEM = dedent("""\
    You are a routing agent for a multi-tool AI system.
    Given a user query and the list of available tools, decide whether
    the query can be fully or partially handled.

    Respond with ONLY a JSON object — no markdown, no prose.
    {
      "can_handle": bool,
      "confidence": 0.0,
      "reason": "one sentence",
      "relevant_tools": ["tool_name", ...]
    }
""")

PLANNER_SYSTEM = dedent("""\
    You are a task-planning agent. Given a user query and a catalogue of
    available tools, produce a minimal, ordered execution plan.

    Rules:
      1. Use only tools listed in the catalogue.
      2. Prefer the fewest steps that fully address the query.
      3. To pass a prior step's output as an arg value, use the string "{step_N}"
         where N is the 0-based index of the prior step (e.g. "{step_0}").
      4. Every step must have a clear, human-readable "description".
      5. Respond with ONLY a JSON object — no markdown, no prose.

    {
      "rationale": "why this plan works",
      "steps": [
        {"tool": "tool_name", "args": {"key": "value"}, "description": "what this step does"}
      ]
    }
""")

VALIDATOR_SYSTEM = dedent("""\
    You are a quality-check agent. Given the original user query and the
    aggregated output produced by a multi-tool pipeline, decide whether
    the output satisfactorily answers the query.

    Respond with ONLY a JSON object — no markdown, no prose.
    {
      "aligned": bool,
      "quality_score": 0.0,
      "issues": ["issue1", ...],
      "notes": "brief summary"
    }
""")

# ══════════════════════════════════════════════════════════════════════
#  Tool registry
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ToolDef:
    name:        str
    description: str
    fn:          Callable
    signature:   str   # human-readable for planner prompt

    def call(self, args: dict) -> str:
        try:
            return str(self.fn(**args))
        except TypeError as exc:
            return f"[tool error] bad args for {self.name!r}: {exc}"
        except Exception as exc:
            return f"[tool error] {self.name!r} raised {type(exc).__name__}: {exc}"


_TOOL_REGISTRY: dict[str, ToolDef] = {}


def register_tool(name: str, description: str) -> Callable:
    """Decorator: @register_tool("name", "description")"""
    def decorator(fn: Callable) -> Callable:
        def _ann(p: inspect.Parameter) -> str:
            a = p.annotation
            if a is inspect.Parameter.empty:
                return "str"
            if isinstance(a, type):
                return a.__name__
            return str(a)  # handles string annotations from __future__

        sig = ", ".join(
            f"{p.name}: {_ann(p)}"
            + (f" = {p.default!r}" if p.default != inspect.Parameter.empty else "")
            for p in inspect.signature(fn).parameters.values()
        )
        _TOOL_REGISTRY[name] = ToolDef(
            name=name,
            description=description,
            fn=fn,
            signature=f"{name}({sig})",
        )
        return fn
    return decorator


def _tool_catalogue() -> str:
    """Compact catalogue string fed to the planner prompt."""
    if not _TOOL_REGISTRY:
        return "  (no tools registered)"
    lines = []
    for td in _TOOL_REGISTRY.values():
        lines.append(f"  • {td.signature}")
        lines.append(f"    {td.description}")
    return "\n".join(lines)


def _resolve_args(args: dict, results: list[str]) -> dict:
    """Replace {step_N} placeholders with the Nth step's result string."""
    resolved = {}
    for k, v in args.items():
        if isinstance(v, str):
            def _sub(m: re.Match) -> str:
                idx = int(m.group(1))
                return results[idx] if idx < len(results) else m.group(0)
            v = re.sub(r"\{step_(\d+)\}", _sub, v)
        resolved[k] = v
    return resolved


# ══════════════════════════════════════════════════════════════════════
#  Built-in tools  (replace / extend as needed for your suite)
# ══════════════════════════════════════════════════════════════════════

@register_tool("echo", "Return the input string unchanged. Useful for passing data between steps.")
def tool_echo(text: str) -> str:
    return text


@register_tool("read_file", "Read a local text file and return its contents.")
def tool_read_file(path: str) -> str:
    p = Path(path).expanduser()
    if not p.exists():
        return f"[error] file not found: {path}"
    return p.read_text(errors="replace")


@register_tool("write_file", "Write text to a local file. Returns the absolute path on success.")
def tool_write_file(path: str, content: str) -> str:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return str(p.resolve())


@register_tool("http_get", "Fetch a URL via HTTP GET and return the response body (plain text).")
def tool_http_get(url: str, timeout: int = 15) -> str:
    req = Request(url, headers={"User-Agent": "harness/1.0"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.read().decode(errors="replace")[:8_000]
    except Exception as exc:
        return f"[error] {exc}"


@register_tool("shell", "Run a shell command and return stdout+stderr (max 4 KB).")
def tool_shell(cmd: str, timeout: int = 30) -> str:
    try:
        out = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        result = (out.stdout + out.stderr).strip()
        return result[:4_096] or "(no output)"
    except subprocess.TimeoutExpired:
        return "[error] command timed out"
    except Exception as exc:
        return f"[error] {exc}"


@register_tool(
    "llm_summarize",
    "Summarize a block of text using the planner model. Returns a concise summary.",
)
def tool_llm_summarize(text: str, max_sentences: int = 5) -> str:
    # Lazy import so the tool works without a live client at import time.
    client = Client(host=OLLAMA_URL)
    resp = client.chat(
        model=PLANNER_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    f"Summarise the following text in at most {max_sentences} sentences. "
                    "Be concise and factual. Return only the summary."
                ),
            },
            {"role": "user", "content": text[:6_000]},
        ],
        options={"temperature": 0.2, "num_gpu": 99},
    )
    msg = resp.get("message", {}) if isinstance(resp, dict) else resp.message
    content = msg.get("content", "") if isinstance(msg, dict) else msg.content
    return (content or "").strip()




@register_tool("weather", "Get current weather and forecast for a location.")
def tool_weather(location: str) -> str:
    weather_path = Path(__file__).parent / "modules" / "weather.py"
    out = subprocess.run(
        [
            "uv", "run",
            "--with", "browser-use",
            "--with", "ollama",
            "--with", "rich",
            "--with", "timezonefinder",
            "python", str(weather_path),
            f"weather in {location}",
            "--harness",
        ],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0 and not out.stdout.strip():
        # Capture more of the stderr for better tracebacks
        err = out.stderr.strip()[:1000]
        return f"[tool error] weather module failed: {err}"
    return out.stdout.strip()


@register_tool("browser", "Execute a multi-step web task using an autonomous agent (e.g. 'Find current star-count of X on GitHub').")
def tool_browser(task: str) -> str:
    agent_path = Path(__file__).parent / "modules" / "browser_agent.py"
    out = subprocess.run(
        [
            "uv", "run",
            "--with", "browser-use",
            "--with", "langchain-ollama",
            "--with", "ollama",
            "--with", "rich",
            "python", str(agent_path),
            task,
            "--harness",
        ],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        return f"[tool error] browser module failed: {out.stderr.strip()[:1000]}"
    # Return everything after the Rich header/init lines if possible, or just the whole output
    return out.stdout.strip()


@register_tool("knowledge_index", "Index a local directory (path) for semantic search.")
def tool_knowledge_index(path: str) -> str:
    agent_path = Path(__file__).parent / "modules" / "knowledge_agent.py"
    out = subprocess.run(
        [
            "uv", "run",
            "--with", "faiss-cpu",
            "--with", "sentence-transformers",
            "--with", "numpy",
            "--with", "rich",
            "python", str(agent_path),
            "--index", path,
            "--harness",
        ],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        return f"[tool error] indexing failed: {out.stderr.strip()[:500]}"
    return out.stdout.strip()


@register_tool("knowledge_query", "Search the local knowledge base for specific information / code context.")
def tool_knowledge_query(query: str) -> str:
    agent_path = Path(__file__).parent / "modules" / "knowledge_agent.py"
    out = subprocess.run(
        [
            "uv", "run",
            "--with", "faiss-cpu",
            "--with", "sentence-transformers",
            "--with", "numpy",
            "--with", "rich",
            "python", str(agent_path),
            "--query", query,
            "--harness",
        ],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        return f"[tool error] query failed: {out.stderr.strip()[:500]}"
    return out.stdout.strip()

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
class PlanStep:
    tool:        str
    args:        dict
    description: str


@dataclass
class StepResult:
    plan_step: PlanStep
    output:    str
    elapsed_ms: float = 0.0
    error:      bool  = False



# ══════════════════════════════════════════════════════════════════════
#  Run history (simple JSON log)
# ══════════════════════════════════════════════════════════════════════

def load_history() -> list[dict]:
    if HISTORY_PATH.exists():
        try:
            return json.loads(HISTORY_PATH.read_text())
        except Exception:
            pass
    return []


def save_history(entries: list[dict]) -> None:
    HISTORY_PATH.write_text(json.dumps(entries[-50:], indent=2))  # keep last 50


# ══════════════════════════════════════════════════════════════════════
#  Core pipeline functions
# ══════════════════════════════════════════════════════════════════════

def sanity_check(
    client: Client,
    query: str,
    ui: UIState,
    refresh: Any,
) -> dict:
    catalogue = _tool_catalogue()
    user_msg = (
        f"Query: {query}\n\n"
        f"Available tools:\n{catalogue}"
    )
    result, _ = _chat_json(
        client, CHECKER_MODEL, SANITY_SYSTEM, user_msg, ui, refresh, "sanity"
    )
    return result


def make_plan(
    client: Client,
    query: str,
    relevant_tools: list[str],
    ui: UIState,
    refresh: Any,
) -> tuple[list[PlanStep], str]:
    # Restrict catalogue to relevant tools only (keeps context tight)
    catalogue_lines = []
    for name, td in _TOOL_REGISTRY.items():
        if not relevant_tools or name in relevant_tools:
            catalogue_lines.append(f"  • {td.signature}\n    {td.description}")
    catalogue = "\n".join(catalogue_lines) or _tool_catalogue()

    user_msg = (
        f"Query: {query}\n\n"
        f"Available tools:\n{catalogue}"
    )
    result, _ = _chat_json(
        client, PLANNER_MODEL, PLANNER_SYSTEM, user_msg, ui, refresh, "plan"
    )
    steps_raw = result.get("steps", [])
    rationale = result.get("rationale", "")
    steps = [
        PlanStep(
            tool        = s.get("tool", "echo"),
            args        = s.get("args", {}),
            description = s.get("description", ""),
        )
        for s in steps_raw
        if isinstance(s, dict)
    ]
    return steps, rationale


def execute_plan(
    plan: list[PlanStep],
    ui: UIState,
    refresh: Any,
) -> list[StepResult]:
    results: list[StepResult] = []
    result_strings: list[str] = []

    for i, step in enumerate(plan):
        label = f"step {i}  [{step.tool}]"
        s = ui.add_step(label).start()
        refresh()

        if step.tool not in _TOOL_REGISTRY:
            out = f"[error] unknown tool: {step.tool!r}"
            s.error(out[:50])
            results.append(StepResult(plan_step=step, output=out, error=True,
                                       elapsed_ms=s.elapsed_ms))
            result_strings.append(out)
            refresh()
            continue

        resolved_args = _resolve_args(step.args, result_strings)
        t0 = time.monotonic()
        out = _TOOL_REGISTRY[step.tool].call(resolved_args)
        elapsed = (time.monotonic() - t0) * 1000

        is_err = out.startswith("[error]") or out.startswith("[tool error]")
        preview = (out[:50] + "…") if len(out) > 50 else out
        if is_err:
            s.error(preview)
        else:
            s.done(preview)

        results.append(StepResult(
            plan_step  = step,
            output     = out,
            elapsed_ms = elapsed,
            error      = is_err,
        ))
        result_strings.append(out)
        refresh()

    return results


def validate_result(
    client: Client,
    query: str,
    results: list[StepResult],
    ui: UIState,
    refresh: Any,
) -> dict:
    # Aggregate all step outputs into a readable block for the validator
    aggregate = "\n\n".join(
        f"[Step {i} – {r.plan_step.tool}]\n{r.output}"
        for i, r in enumerate(results)
    )
    user_msg = (
        f"Original query: {query}\n\n"
        f"Pipeline output:\n{aggregate[:4_000]}"
    )
    result, _ = _chat_json(
        client, CHECKER_MODEL, VALIDATOR_SYSTEM, user_msg, ui, refresh, "validate"
    )
    return result


# ══════════════════════════════════════════════════════════════════════
#  Result card  (printed after Live block closes)
# ══════════════════════════════════════════════════════════════════════

def print_result_card(
    console: Console,
    query: str,
    plan: list[PlanStep],
    results: list[StepResult],
    rationale: str,
    validation: dict,
) -> None:
    console.print()
    console.print(Rule("[bold blue]Harness Result[/bold blue]", style="blue"))

    # Plan summary table
    tbl = Table(
        show_header=True,
        header_style="bold blue",
        box=None,
        padding=(0, 2),
    )
    tbl.add_column("#",    style="dim",        width=4)
    tbl.add_column("Tool", style="cyan bold",  width=18)
    tbl.add_column("Description",              width=34)
    tbl.add_column("Status",                   width=8)
    tbl.add_column("Output preview",           width=36)

    for i, r in enumerate(results):
        status = "[red]error[/red]" if r.error else "[green]ok[/green]"
        preview = r.output[:35].replace("\n", " ") + ("…" if len(r.output) > 35 else "")
        tbl.add_row(
            str(i),
            r.plan_step.tool,
            r.plan_step.description[:34],
            status,
            preview,
        )

    console.print(
        Panel(tbl, title="[bold]Execution plan[/bold]",
              subtitle=f"[dim]{rationale[:80]}[/dim]",
              border_style="blue", padding=(1, 2))
    )

    # Final output of last step
    if results:
        final_out = results[-1].output
        console.print(
            Panel(
                Text.from_ansi(final_out[:8000]),
                title="[bold]Final output[/bold]",
                border_style="green",
                padding=(1, 2),
                expand=False
            )
        )

    # Validation panel
    aligned = validation.get("aligned", True)
    score   = float(validation.get("quality_score", 1.0))
    notes   = validation.get("notes", "")
    issues  = validation.get("issues", [])

    val_color = "green" if aligned and score >= 0.7 else "yellow" if score >= 0.4 else "red"
    val_lines = Text()
    val_lines.append(f"{'✓' if aligned else '✗'} aligned  ", f"bold {val_color}")
    val_lines.append(f"quality {score:.0%}  ", "white")
    val_lines.append(notes[:80], "dim")
    if issues:
        val_lines.append("\n  issues: " + " · ".join(str(x) for x in issues[:3]), "yellow")

    console.print(
        Panel(val_lines,
              title="[bold]Validation[/bold]",
              border_style=val_color,
              padding=(0, 2))
    )
    console.print(Rule(style="blue"))
    console.print()
    console.print(
        Panel.fit(
            f"[dim]history:[/dim] {HISTORY_PATH}",
            title="[dim]Run complete[/dim]",
            border_style="dim",
        )
    )


# ══════════════════════════════════════════════════════════════════════
#  Main orchestration
# ══════════════════════════════════════════════════════════════════════

def run(query: str, dry_run: bool = False) -> int:
    console = Console()
    ui      = UIState(agent_name="agent-harness", model_info=f"{CHECKER_MODEL} · {PLANNER_MODEL}")

    plan: list[PlanStep]       = []
    results: list[StepResult]  = []
    rationale  = ""
    validation: dict           = {}
    aligned    = True

    with Live(ui.render(), refresh_per_second=UI_REFRESH_HZ,
              console=console, screen=False) as live:

        def refresh() -> None:
            live.update(ui.render())

        try:
            # ── 1. connect + warmup ───────────────────────────────────
            s_init = ui.add_step("connect + warmup").start(); refresh()
            client = setup_ollama(OLLAMA_URL, [CHECKER_MODEL, PLANNER_MODEL])
            with ThreadPoolExecutor(max_workers=2) as wp:
                wp.submit(_warmup, client, CHECKER_MODEL)
                wp.submit(_warmup, client, PLANNER_MODEL)
            s_init.done("Ollama ready · models warmed"); refresh()

            # ── 2. sanity check ───────────────────────────────────────
            s_chk = ui.add_step("sanity check").start(); refresh()
            check_raw = sanity_check(client, query, ui, refresh)

            can_handle = bool(check_raw.get("can_handle", False))
            confidence = float(check_raw.get("confidence", 0.0))
            reason     = str(check_raw.get("reason", ""))
            rel_tools  = check_raw.get("relevant_tools", [])

            chk_label = (
                f"{'✓ in scope' if can_handle else '✗ out of scope'}  "
                f"({confidence:.0%})  {reason[:40]}"
            )
            if can_handle:
                s_chk.done(chk_label)
            else:
                s_chk.error(chk_label)
            refresh()

            if not can_handle:
                ui.push_chunk(
                    f"This query is outside the current tool suite.\n{reason}"
                )
                ui.running = False; refresh()
                return 1

            # ── 3. plan ───────────────────────────────────────────────
            s_plan = ui.add_step("build plan").start(); refresh()
            plan, rationale = make_plan(client, query, rel_tools, ui, refresh)
            s_plan.done(
                f"{len(plan)} step{'s' if len(plan) != 1 else ''}  ·  "
                + rationale[:40]
            )
            refresh()

            if not plan:
                ui.push_chunk("Planner returned an empty plan.")
                ui.running = False; refresh()
                return 1

            if dry_run:
                # Print the plan and exit without executing
                for i, step in enumerate(plan):
                    ui.add_step(f"  step {i}  [{step.tool}]").skip(step.description[:50])
                ui.running = False; refresh()
                return 0

            # ── 4. execute ────────────────────────────────────────────
            results = execute_plan(plan, ui, refresh)

            # ── 5. validate ───────────────────────────────────────────
            s_val = ui.add_step("validate result").start(); refresh()
            validation = validate_result(client, query, results, ui, refresh)
            aligned    = bool(validation.get("aligned", True))
            score      = float(validation.get("quality_score", 1.0))
            notes      = str(validation.get("notes", ""))
            issues     = validation.get("issues", [])

            val_detail = f"{'✓' if aligned else '✗'}  score={score:.0%}  {notes[:45]}"
            if issues:
                val_detail += "  issues: " + "; ".join(str(x) for x in issues)[:30]

            if aligned:
                s_val.done(val_detail)
            else:
                s_val.error(val_detail)
            refresh()

            # ── 6. save history ───────────────────────────────────────
            s_save = ui.add_step("save history").start(); refresh()
            history = load_history()
            history.append({
                "query":      query,
                "plan":       [{"tool": s.tool, "description": s.description} for s in plan],
                "aligned":    aligned,
                "score":      score,
                "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            save_history(history)
            s_save.done(f"{HISTORY_PATH.name}"); refresh()

            ui.running = False; refresh()

        except KeyboardInterrupt:
            ui.running = False
            ui.add_step("interrupted").error("KeyboardInterrupt"); refresh()
            return 130
        except Exception as exc:
            ui.running = False
            ui.add_step("fatal error").error(str(exc)[:80]); refresh()
            raise

    # ── post-live: result card ─────────────────────────────────────────
    if results and not dry_run:
        print_result_card(
            console, query, plan, results, rationale, validation
        )

    return 0 if aligned else 2


# ══════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Multi-agent orchestration harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent("""\
            examples:
              python harness.py "summarise the Hacker News front page"
              python harness.py --dry-run "fetch and summarise https://example.com"
              python harness.py --list-tools
              python harness.py --clear-history
        """),
    )
    ap.add_argument("query", nargs="*", help="Task / query for the harness")
    ap.add_argument("--dry-run", action="store_true",
                    help="Plan the steps but do not execute them")
    ap.add_argument("--list-tools", action="store_true",
                    help="Print registered tools and exit")
    ap.add_argument("--clear-history", action="store_true",
                    help="Delete run history and exit")
    ap.add_argument("--models", action="store_true",
                    help="Show configured model names and exit")
    args = ap.parse_args()

    if args.list_tools:
        print("\nRegistered tools:\n")
        for td in _TOOL_REGISTRY.values():
            print(f"  {td.signature}")
            print(f"    {td.description}\n")
        return 0

    if args.clear_history:
        if HISTORY_PATH.exists():
            HISTORY_PATH.unlink()
            print(f"History cleared: {HISTORY_PATH}")
        else:
            print("No history file found.")
        return 0

    if args.models:
        print(f"  checker/validator : {CHECKER_MODEL}")
        print(f"  planner           : {PLANNER_MODEL}")
        return 0

    query = " ".join(args.query).strip()
    if not query:
        try:
            query = input("\n  What would you like to do?  ").strip()
        except (EOFError, KeyboardInterrupt):
            return 0

    if not query:
        print("Empty query.")
        return 1

    return run(query, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
