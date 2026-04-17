#!/usr/bin/env python3
"""
harness_logic.py — Core execution engine & Tool Registry
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Dict, Optional, Tuple

from ollama import AsyncClient
from rich.console import Console

from src.common.ui import UIState
from src.common.ollama import _stream_chat
from src.common.config import (
    OLLAMA_URL, HISTORY_PATH, EXPERIENCE_PATH, 
    PLANNER_MODEL, CHECKER_MODEL, REVIEW_MODEL
)

# ══════════════════════════════════════════════════════════════════════
#  Data Models
# ══════════════════════════════════════════════════════════════════════

@dataclass
class PlanStep:
    tool:        str
    args:        dict
    description: str

@dataclass
class StepResult:
    plan_step:   PlanStep
    output:      str
    error:       bool = False

@dataclass
class ToolDef:
    name:        str
    description: str
    fn:          Callable
    signature:   str

    async def call(self, args: dict, ui: UIState, refresh: Callable, client: AsyncClient) -> str:
        try:
            params = inspect.signature(self.fn).parameters
            clean_args = {k: v for k, v in args.items() if k not in ("ui", "refresh", "client")}
            
            extra = {}
            if "ui" in params: extra["ui"] = ui
            if "refresh" in params: extra["refresh"] = refresh
            if "client" in params: extra["client"] = client
            
            if inspect.iscoroutinefunction(self.fn):
                res = await self.fn(**clean_args, **extra)
            else:
                res = self.fn(**clean_args, **extra)
            return str(res)
        except Exception as exc:
            return f"[tool error] {self.name!r} raised {type(exc).__name__}: {exc}"

# ══════════════════════════════════════════════════════════════════════
#  Tool Registry
# ══════════════════════════════════════════════════════════════════════

_TOOL_REGISTRY: Dict[str, ToolDef] = {}

def register_tool(name: str, description: str) -> Callable:
    """Decorator to register a tool with the harness."""
    def decorator(fn: Callable) -> Callable:
        def _ann(p: inspect.Parameter) -> str:
            a = p.annotation
            if a is inspect.Parameter.empty: return "str"
            return a.__name__ if isinstance(a, type) else str(a)

        sig = ", ".join(
            f"{p.name}: {_ann(p)}" + (f" = {p.default!r}" if p.default != inspect.Parameter.empty else "")
            for p in inspect.signature(fn).parameters.values()
            if p.name not in ("ui", "refresh", "client")
        )
        _TOOL_REGISTRY[name] = ToolDef(name=name, description=description, fn=fn, signature=f"{name}({sig})")
        return fn
    return decorator

def get_tool_catalogue() -> str:
    if not _TOOL_REGISTRY: return "  (no tools registered)"
    return "\n".join([f"  • {td.signature}\n    {td.description}" for td in _TOOL_REGISTRY.values()])

# ══════════════════════════════════════════════════════════════════════
#  Built-in General Tools
# ══════════════════════════════════════════════════════════════════════

@register_tool("echo", "Return the input string unchanged.")
def tool_echo(text: str) -> str: return text

@register_tool("read_file", "Read a local text file.")
async def tool_read_file(path: str) -> str:
    p = Path(path).expanduser()
    if not await asyncio.to_thread(p.exists): return f"[error] file not found: {path}"
    return await asyncio.to_thread(p.read_text, errors="replace")

@register_tool("write_file", "Write text to a local file.")
async def tool_write_file(path: str, content: str) -> str:
    p = Path(path).expanduser()
    await asyncio.to_thread(p.parent.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(p.write_text, content)
    return str(p.resolve())

@register_tool("http_get", "Fetch a URL via HTTP GET.")
async def tool_http_get(url: str, timeout: int = 15) -> str:
    from urllib.request import Request, urlopen
    def _sync():
        with urlopen(Request(url, headers={"User-Agent": "ElfWeave"}), timeout=timeout) as r:
            return r.read().decode(errors="replace")[:8000]
    try: return await asyncio.to_thread(_sync)
    except Exception as e: return f"[error] {e}"

@register_tool("shell", "Run a shell command.")
async def tool_shell(cmd: str, ui: UIState, refresh: Callable) -> str:
    try:
        proc = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        chunks = []
        while True:
            line = await proc.stdout.readline()
            if not line: break
            t = line.decode(errors="replace").strip()
            if t: ui.push_chunk(t); chunks.append(t); refresh()
        await proc.wait()
        return "\n".join(chunks)[:4096] or "(no output)"
    except Exception as e: return f"[error] {e}"

@register_tool("llm_summarize", "Summarize text using the planner model.")
async def tool_llm_summarize(text: str, client: AsyncClient, max_sentences: int = 5) -> str:
    summary, _ = await _stream_chat(
        client,
        PLANNER_MODEL,
        [
            {"role": "system", "content": f"Summarize the user's text in no more than {max_sentences} sentences."},
            {"role": "user", "content": text[:6000]},
        ],
        None,
        lambda: None,
        "summarizer",
        temperature=0.1,
    )
    return summary or "(no summary)"

# ══════════════════════════════════════════════════════════════════════
#  Specialist Wrappers
# ══════════════════════════════════════════════════════════════════════

async def _run_tool_subprocess(args: List[str], ui: UIState, refresh: Callable) -> str:
    proc = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = [], []
    async def read(stream, out, err=False):
        while True:
            line = await stream.readline()
            if not line: break
            t = line.decode(errors="replace").strip()
            if t: ui.push_chunk(t); out.append(t); refresh()
    await asyncio.gather(read(proc.stdout, stdout), read(proc.stderr, stderr, True))
    await proc.wait()
    if proc.returncode != 0:
        stderr_text = "\n".join(stderr)
        return f"[tool error] {stderr_text}" if stderr_text else f"[tool error] code {proc.returncode}"
    return "\n".join(stdout)

@register_tool("weather", "PRIMARY weather Specialist.")
async def tool_weather(location: str, ui: UIState, refresh: Callable) -> str:
    return await _run_tool_subprocess(["uv", "run", "--with", "ollama", "--with", "rich", "--with", "timezonefinder", "python", "src/modules/weather.py", f"weather in {location}", "--harness"], ui, refresh)

@register_tool("browser", "Autonomous Web Specialist.")
async def tool_browser(task: str, ui: UIState, refresh: Callable) -> str:
    return await _run_tool_subprocess(["uv", "run", "--with", "browser-use", "--with", "langchain-ollama", "--with", "ollama", "--with", "rich", "python", "src/modules/browser_agent.py", task, "--harness"], ui, refresh)

@register_tool("code_architect", "Design & Technical Debt Specialist.")
async def tool_code_architect(files: List[str], ui: UIState, refresh: Callable) -> str:
    return await _run_tool_subprocess(["uv", "run", "--with", "ollama", "--with", "rich", "python", "src/modules/code_architect.py", *files, "--harness"], ui, refresh)

@register_tool("fs_manager", "Project Explorer Specialist.")
async def tool_fs_manager(ui: UIState, refresh: Callable, path: str = ".") -> str:
    return await _run_tool_subprocess(["uv", "run", "--with", "rich", "python", "src/modules/fs_manager.py", path, "--harness"], ui, refresh)

@register_tool("knowledge_query", "Search the local knowledge base or repository text.")
async def tool_knowledge_query(query: str, ui: UIState, refresh: Callable) -> str:
    return await _run_tool_subprocess(["uv", "run", "--with", "rich", "--with", "numpy", "python", "src/modules/knowledge_agent.py", "--query", query, "--harness"], ui, refresh)

# ══════════════════════════════════════════════════════════════════════
#  Self-Repair Meta-Tools
# ══════════════════════════════════════════════════════════════════════

@register_tool("analyze_failure", "Root-cause diagnosis for plan failures.")
async def tool_analyze_failure(issues: str, plan_context: str, ui: UIState, refresh: Callable, client: AsyncClient) -> str:
    # Logic moved to harness_planner for cleaner model coordination
    from src.harness_planner import analyze_failure_logic
    return await analyze_failure_logic(issues, plan_context, ui, refresh, client)

@register_tool("repair_code", "Autonomously patch a specific file.")
async def tool_repair_code(filename: str, recommended_fix: str, ui: UIState, refresh: Callable, client: AsyncClient) -> str:
    from src.harness_planner import repair_code_logic
    return await repair_code_logic(filename, recommended_fix, ui, refresh, client)

@register_tool("research_fix", "Web Search for technical errors.")
async def tool_research_fix(issues: str, ui: UIState, refresh: Callable) -> str:
    return await tool_browser(f"How to fix this error: {issues[:200]}", ui, refresh)

# ══════════════════════════════════════════════════════════════════════
#  Execution Logistics
# ══════════════════════════════════════════════════════════════════════

def _resolve_args(args: dict, res_history: List[str]) -> dict:
    def _sub(s: str) -> str:
        return re.sub(
            r"\{step_(\d+)\}",
            lambda m: res_history[int(m.group(1))] if int(m.group(1)) < len(res_history) else "[error]",
            s,
        )

    res = {}
    for k, v in args.items():
        if isinstance(v, str):
            v = _sub(v)
        elif isinstance(v, list):
            v = [_sub(x) if isinstance(x, str) else x for x in v]
        res[k] = v
    return res

async def execute_plan(plan: List[PlanStep], ui: UIState, refresh: Callable, client: AsyncClient) -> List[StepResult]:
    results, strings = [], []
    for i, step in enumerate(plan):
        s = ui.add_step(f"step {i} [{step.tool}]").start(); refresh()
        if step.tool not in _TOOL_REGISTRY:
            results.append(StepResult(step, f"Unknown tool: {step.tool}", True)); s.error("unknown tool"); break
        tool = _TOOL_REGISTRY[step.tool]
        out = await tool.call(_resolve_args(step.args, strings), ui, refresh, client)
        stripped = out.lstrip()
        is_err = stripped.startswith("[tool error]") or stripped.startswith("[error]")
        if is_err:
            s.error("error")
            # Prepend structured context so the validator/planner knows which step failed
            out = f"[failed at step {i} · tool={step.tool!r}] {out}"
        else:
            s.done("ok")
        results.append(StepResult(step, out, is_err)); strings.append(out); refresh()
        if is_err: break
    return results

# ══════════════════════════════════════════════════════════════════════
#  History & Experience
# ══════════════════════════════════════════════════════════════════════

def load_history() -> List[Dict]:
    if not HISTORY_PATH.exists(): return []
    try: return json.loads(HISTORY_PATH.read_text())
    except: return []

def save_history(history: List[Dict]): HISTORY_PATH.write_text(json.dumps(history, indent=2))

def get_learned_lessons(limit: int = 5) -> str:
    if not EXPERIENCE_PATH.exists(): return "No past experiences."
    lessons = []
    try:
        with open(EXPERIENCE_PATH, "r") as f:
            for line in f.readlines()[-limit:]:
                e = json.loads(line)
                lessons.append(f" - {'SUCCESS' if e.get('aligned') else 'FAILURE'}: {e.get('query')}\n   Issues: {e.get('issues')}\n   Fix: {e.get('fix')}")
    except: return "Experience load error."
    return "\n".join(lessons) or "No experiences."

def save_experience(query: str, res: List[StepResult], validation: Dict, timestamp: str):
    fix = next((r.output for r in res if r.plan_step.tool == "repair_code" and not r.error), "")
    entry = {"timestamp": timestamp, "query": query, "aligned": validation.get("aligned"), "score": validation.get("quality_score"), "issues": validation.get("issues"), "fix": fix}
    try:
        with open(EXPERIENCE_PATH, "a") as f: f.write(json.dumps(entry) + "\n")
    except: pass
