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
import asyncio
import inspect
import json
import re
import shlex
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent
from typing import Any, Callable, Optional

from ollama import AsyncClient, ResponseError
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
from src.common.config import OLLAMA_URL, PLANNER_MODEL, CHECKER_MODEL, REVIEW_MODEL

# ══════════════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════════════

# Moved to config.py
# OLLAMA_URL      = "http://localhost:11434"
# CHECKER_MODEL   = "llama3.2:3b"
# PLANNER_MODEL   = "qwen2.5:3b"

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

    Protocol Priority: 
    - If a query is about weather, MUST include 'weather' in relevant_tools.
    - If a query is about site navigation or deep research, MUST include 'browser'.
    - If a query is about local code or files, MUST include 'knowledge_query'.

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
      2. Specialist Priority: ALWAYS prefer specialized agents (weather, browser, knowledge_query) 
         over raw utilities (http_get, shell) for their respective domains.
      3. Minimal Hallucination: Do NOT invent URLs or paths for http_get/shell unless they are provided in the query or by a previous step.
      4. To pass a prior step's output as an arg value, use the string "{step_N}"
         where N is the 0-based index of the prior step (e.g. "{step_0}").
      5. Every step must have a clear, human-readable "description".
      6. Respond with ONLY a JSON object — no markdown, no prose.

    {
      "rationale": "why this plan works",
      "steps": [
        {"tool": "tool_name", "args": {"key": "value"}, "description": "what this step does"}
      ]
    }
""")

VALIDATOR_SYSTEM = dedent("""\
    You are a high-fidelity Quality Assurance agent.
    Given the original user query and the aggregated output produced by a multi-tool pipeline, 
    decide whether the output satisfactorily and accurately answers the query.

    Evaluation Criteria:
      1. Grounding: All numbers/facts must come from tool outputs.
      2. Completeness: Did we answer every part of the user's multi-step request?
      3. Protocol Check: If the output contains specialized data (e.g. weather), does it follow expert protocols?
      4. Formatting: Is the output visually clean and descriptive?

    Respond with ONLY a JSON object:
    {
      "aligned": bool,
      "quality_score": 0.0 (0.0 to 1.0),
      "issues": ["missing X", "incorrect Y", ...],
      "notes": "critical summary of quality"
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

    async def call(self, args: dict, ui: UIState, refresh: Callable, client: AsyncClient) -> str:
        try:
            # We check if the function accepts ui/refresh/client
            params = inspect.signature(self.fn).parameters
            extra = {}
            if "ui" in params: extra["ui"] = ui
            if "refresh" in params: extra["refresh"] = refresh
            if "client" in params: extra["client"] = client
            
            if inspect.iscoroutinefunction(self.fn):
                res = await self.fn(**args, **extra)
            else:
                res = self.fn(**args, **extra)
            return str(res)
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
                if idx < 0 or idx >= len(results):
                    return f"[error: step_{idx} not found]"
                return results[idx]
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
async def tool_read_file(path: str) -> str:
    p = Path(path).expanduser()
    if not await asyncio.to_thread(p.exists):
        return f"[error] file not found: {path}"
    return await asyncio.to_thread(p.read_text, errors="replace")


@register_tool("write_file", "Write text to a local file. Returns the absolute path on success.")
async def tool_write_file(path: str, content: str) -> str:
    p = Path(path).expanduser()
    await asyncio.to_thread(p.parent.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(p.write_text, content)
    return str(p.resolve())


@register_tool("http_get", "Fetch a URL via HTTP GET and return the response body (plain text). DO NOT USE this for weather, search, or browsing if a specialist tool (weather, browser, knowledge) is available.")
async def tool_http_get(url: str, timeout: int = 15) -> str:
    from urllib.request import Request, urlopen
    def _sync_get():
        req = Request(url, headers={"User-Agent": "harness/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            return resp.read().decode(errors="replace")[:8_000]
    try:
        return await asyncio.to_thread(_sync_get)
    except Exception as exc:
        return f"[error] {exc}"


@register_tool("shell", "Run a shell command and return stdout+stderr (max 4 KB).")
async def tool_shell(cmd: str, ui: UIState, refresh: Callable) -> str:
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        
        # Read output in chunks to keep UI responsive
        chunks = []
        while True:
            line = await proc.stdout.readline()
            if not line: break
            text = line.decode(errors="replace").strip()
            if text:
                ui.push_chunk(text)
                chunks.append(text)
                refresh()
                
        await proc.wait()
        result = "\n".join(chunks)
        return result[:4_096] or "(no output)"
    except Exception as exc:
        return f"[error] {exc}"


@register_tool(
    "llm_summarize",
    "Summarize a block of text using the planner model. Returns a concise summary.",
)
async def tool_llm_summarize(text: str, client: AsyncClient, max_sentences: int = 5) -> str:
    resp = await client.chat(
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


async def _run_tool_subprocess(args: list[str], ui: UIState, refresh: Callable) -> str:
    """Helper to run tool modules asynchronously and stream output to UI."""
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout_chunks = []
    stderr_chunks = []
    
    async def read_stream(stream, is_err=False):
        while True:
            line = await stream.readline()
            if not line:
                break
            text = line.decode(errors="replace").strip()
            if text:
                ui.push_chunk(text)
                if is_err:
                    stderr_chunks.append(text)
                else:
                    stdout_chunks.append(text)
                refresh()

    await asyncio.gather(
        read_stream(proc.stdout, is_err=False),
        read_stream(proc.stderr, is_err=True)
    )
    
    await proc.wait()
    
    if proc.returncode != 0:
        err_msg = "\n".join(stderr_chunks)
        if err_msg:
            return f"[tool error] {err_msg}"
        return f"[tool error] module failed with code {proc.returncode}"
        
    return "\n".join(stdout_chunks)




@register_tool("weather", "PRIMARY TOOL for all weather queries, forecasts, and climate data. Use this instead of http_get for any weather-related task.")
async def tool_weather(location: str, ui: UIState, refresh: Callable) -> str:
    weather_path = Path(__file__).parent / "modules" / "weather.py"
    return await _run_tool_subprocess([
        "uv", "run",
        "--with", "browser-use",
        "--with", "ollama",
        "--with", "rich",
        "--with", "timezonefinder",
        "python", str(weather_path),
        f"weather in {location}",
        "--harness",
    ], ui, refresh)


@register_tool("browser", "PRIMARY TOOL for multi-step web tasks, site navigation, and deep research using an autonomous agent (e.g. 'Find star-count of X on GitHub').")
async def tool_browser(task: str, ui: UIState, refresh: Callable) -> str:
    agent_path = Path(__file__).parent / "modules" / "browser_agent.py"
    return await _run_tool_subprocess([
        "uv", "run",
        "--with", "browser-use",
        "--with", "langchain-ollama",
        "--with", "ollama",
        "--with", "rich",
        "python", str(agent_path),
        task,
        "--harness",
    ], ui, refresh)


@register_tool("knowledge_index", "Index a local directory (path) for semantic search.")
async def tool_knowledge_index(path: str, ui: UIState, refresh: Callable) -> str:
    agent_path = Path(__file__).parent / "modules" / "knowledge_agent.py"
    return await _run_tool_subprocess([
        "uv", "run",
        "--with", "faiss-cpu",
        "--with", "sentence-transformers",
        "--with", "numpy",
        "--with", "rich",
        "python", str(agent_path),
        "--index", path,
        "--harness",
    ], ui, refresh)


@register_tool("knowledge_query", "Search the local knowledge base for specific information / code context.")
async def tool_knowledge_query(query: str, ui: UIState, refresh: Callable) -> str:
    agent_path = Path(__file__).parent / "modules" / "knowledge_agent.py"
    return await _run_tool_subprocess([
        "uv", "run",
        "--with", "faiss-cpu",
        "--with", "sentence-transformers",
        "--with", "numpy",
        "--with", "rich",
        "python", str(agent_path),
        "--query", query,
        "--harness",
    ], ui, refresh)


@register_tool("analyze_failure", "Root-cause diagnosis — analyze why a plan failed and suggest a specific fix for the harness or tool code.")
async def tool_analyze_failure(issues: str, plan_context: str, ui: UIState, refresh: Callable, client: AsyncClient) -> str:
    # Target files for self-analysis
    files = list(Path(_root).rglob("*.py"))
    code_context = ""
    for f in files[:5]: # Cap context for prototype
        content = await asyncio.to_thread(f.read_text, errors="replace")
        code_context += f"\n--- {f.name} ---\n{content[:2000]}"
    
    prompt = f"Validation Issues: {issues}\n\nPlan Context: {plan_context}\n\nCodebase Context: {code_context}\n\nIdentify the ROOT CAUSE and suggest a specific fix."
    
    # Using the shared client and streaming chat logic
    res, _ = await _chat_json(
        client,
        PLANNER_MODEL,
        "You are a self-healing AI coordinator. Identify the root cause of failures. Return JSON: { \"cause\": \"string\", \"fix\": \"string\" }",
        prompt,
        ui,
        refresh,
        "analyzer"
    )
    return f"Cause: {res.get('cause')}\nFix Suggestion: {res.get('fix')}"

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


def get_relevant_history(query: str, limit: int = 3) -> str:
    """Finds previous similar tasks and their outcomes for 'learning'."""
    history = load_history()
    if not history:
        return "No prior project history available."
    
    # Simple recent-history strategy for the prototype
    # In a full RAG version, we'd use KnowledgeAgent to find semantically similar queries.
    relevant = history[-limit:]
    lines = []
    for h in relevant:
        q = h.get("query", "unknown")
        s = h.get("score", 0.0)
        a = "Success" if h.get("aligned") else "Failed"
        lines.append(f"  - Query: {q}\n    Outcome: {a} (Score: {s:.0%})")
    
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
#  Core pipeline functions
# ══════════════════════════════════════════════════════════════════════

async def sanity_check(
    client: AsyncClient,
    query: str,
    ui: UIState,
    refresh: Any,
) -> dict:
    catalogue = _tool_catalogue()
    user_msg = (
        f"Query: {query}\n\n"
        f"Available tools:\n{catalogue}"
    )
    result, _ = await _chat_json(
        client, CHECKER_MODEL, SANITY_SYSTEM, user_msg, ui, refresh, "sanity"
    )
    return result


async def make_plan(
    client: AsyncClient,
    query: str,
    relevant_tools: list[str],
    ui: UIState,
    refresh: Any,
    feedback: Optional[str] = None,
) -> tuple[list[PlanStep], str]:
    # Restrict catalogue to relevant tools only (keeps context tight)
    catalogue_lines = []
    for name, td in _TOOL_REGISTRY.items():
        if not relevant_tools or name in relevant_tools:
            catalogue_lines.append(f"  • {td.signature}\n    {td.description}")
    catalogue = "\n".join(catalogue_lines) or _tool_catalogue()

    history_context = get_relevant_history(query)

    user_msg = (
        f"Query: {query}\n\n"
        f"Available tools:\n{catalogue}\n\n"
        f"Past Lessons Learned:\n{history_context}"
    )
    
    if feedback:
        user_msg += f"\n\nCRITICAL: Previous attempt failed with these issues:\n{feedback}\nDO NOT repeat the same mistakes. Adjust your strategy."

    result, _ = await _chat_json(
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


async def execute_plan(
    plan: list[PlanStep],
    ui: UIState,
    refresh: Any,
    client: AsyncClient,
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
        out = await _TOOL_REGISTRY[step.tool].call(resolved_args, ui, refresh, client)
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


async def validate_result(
    client: AsyncClient,
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
    result, _ = await _chat_json(
        client, REVIEW_MODEL, VALIDATOR_SYSTEM, user_msg, ui, refresh, "validate"
    )
    return result


# ══════════════════════════════════════════════════════════════════════
#  Result card  (printed after Live block closes)
# ══════════════════════════════════════════════════════════════════════

def print_result_card(
    ui: UIState,
    query: str,
    plan: list[PlanStep],
    results: list[StepResult],
    rationale: str,
    validation: dict,
) -> None:
    # ── Plan Table ──
    tbl = Table(show_header=True, header_style="bold blue", box=None, padding=(0, 2), expand=True)
    tbl.add_column("#",    style="dim",        ratio=1)
    tbl.add_column("Tool", style="cyan bold",  ratio=3)
    tbl.add_column("Description",              ratio=8)
    tbl.add_column("Status",                   ratio=2)
    tbl.add_column("Output preview",           ratio=10)

    for i, r in enumerate(results):
        status = "[red]error[/red]" if r.error else "[green]ok[/green]"
        preview = r.output[:35].replace("\n", " ") + ("…" if len(r.output) > 35 else "")
        tbl.add_row(str(i), r.plan_step.tool, r.plan_step.description[:34], status, preview)

    ui.print_card("Execution Plan", tbl, border_color="blue", metadata=rationale[:80])

    # ── Final Output ──
    if results:
        final_out = results[-1].output
        ui.print_card("Final Output", Text.from_ansi(final_out[:8000]), border_color="green", padding=(0, 1))

    # ── Validation ──
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

    ui.print_card("Validation Result", val_lines, border_color=val_color)


# ══════════════════════════════════════════════════════════════════════
#  Main orchestration
# ══════════════════════════════════════════════════════════════════════

async def run(query: str, dry_run: bool = False) -> int:
    ui      = UIState(agent_name="agent-harness", model_info=f"{CHECKER_MODEL} · {PLANNER_MODEL} · {REVIEW_MODEL}")

    plan: list[PlanStep]       = []
    results: list[StepResult]  = []
    rationale  = ""
    validation: dict           = {}
    aligned    = False
    retry_count = 0
    max_retries = 2
    last_feedback = ""

    async with ui:
        def refresh() -> None:
            ui.refresh()

        try:
            # ── 1. connect + warmup ───────────────────────────────────
            s_init = ui.add_step("connect + warmup").start(); refresh()
            client = await setup_ollama(OLLAMA_URL, [CHECKER_MODEL, PLANNER_MODEL, REVIEW_MODEL])
            
            # Parallel warmup using asyncio.gather
            await asyncio.gather(
                _warmup(client, CHECKER_MODEL),
                _warmup(client, PLANNER_MODEL),
                _warmup(client, REVIEW_MODEL)
            )
            s_init.done("Ollama ready · models warmed"); refresh()

            # ── 2. sanity check ───────────────────────────────────────
            s_chk = ui.add_step("sanity check").start(); refresh()
            check_raw = await sanity_check(client, query, ui, refresh)

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
                return 1

            while retry_count <= max_retries and not aligned:
                # ── 3. plan ───────────────────────────────────────────
                s_plan = ui.add_step(f"plan (attempt {retry_count})").start(); refresh()
                plan, rationale = await make_plan(client, query, rel_tools, ui, refresh, feedback=last_feedback)
                s_plan.done(
                    f"{len(plan)} step{'s' if len(plan) != 1 else ''}  ·  "
                    + rationale[:40]
                )
                refresh()

                if not plan:
                    ui.push_chunk("Planner returned an empty plan.")
                    return 1

                if dry_run:
                    # Print the plan and exit without executing
                    for i, step in enumerate(plan):
                        ui.add_step(f"  step {i}  [{step.tool}]").skip(step.description[:50])
                    return 0

                # ── 4. execute ────────────────────────────────────────────
                results = await execute_plan(plan, ui, refresh, client)

                # ── 5. validate ───────────────────────────────────────────
                s_val = ui.add_step(f"validate (attempt {retry_count})").start(); refresh()
                validation = await validate_result(client, query, results, ui, refresh)
                aligned    = bool(validation.get("aligned", True))
                score      = float(validation.get("quality_score", 1.0))
                notes      = str(validation.get("notes", ""))
                issues     = validation.get("issues", [])

                val_detail = f"{'✓' if aligned else '✗'}  score={score:.0%}  {notes[:45]}"
                if issues:
                    val_detail += "  issues: " + "; ".join(str(x) for x in issues)[:30]
                    last_feedback = f"Validation issues: {'; '.join(str(x) for x in issues)}. Notes: {notes}"

                if aligned or score >= 0.7:
                    s_val.done(val_detail)
                    aligned = True # Close the loop
                else:
                    s_val.error(val_detail)
                    retry_count += 1
                
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
            ui.add_step("fatal error").error(str(exc)[:80]); refresh()
            raise

    # ── post-live: result card ─────────────────────────────────────────
    if results and not dry_run:
        print_result_card(ui, query, plan, results, rationale, validation)

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

    try:
        return asyncio.run(run(query, dry_run=args.dry_run))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
