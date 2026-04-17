#!/usr/bin/env python3
"""
harness.py — multi-agent orchestration harness · Slim Entry Point
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is in sys.path
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import argparse
import asyncio
import time
from textwrap import dedent

from src.common.ui import UIState
from src.common.config import (
    OLLAMA_URL, CHECKER_MODEL, PLANNER_MODEL, REVIEW_MODEL,
    HISTORY_PATH
)
from src.common.ollama import setup_ollama, _warmup
from src.harness_logic import (
    _TOOL_REGISTRY, execute_plan, load_history, save_history, save_experience
)
from src.harness_planner import sanity_check, plan_task, validate_result
from src.harness_ui import print_result_card


def _read_query(parts: list[str], prompt: str) -> str | None:
    query = " ".join(parts).strip()
    if query:
        return query
    if not sys.stdin.isatty():
        print("Error: no query provided. Pass a task on the command line.", file=sys.stderr)
        return None
    try:
        return input(prompt).strip()
    except EOFError:
        return None


def _build_retry_feedback(results: list, validation: dict) -> str:
    lines = []
    for idx, result in enumerate(results):
        status = "error" if result.error else "ok"
        lines.append(f"step {idx} [{result.plan_step.tool}] {status}: {result.output[:500]}")
    if validation.get("issues"):
        lines.append(f"validator issues: {validation.get('issues')}")
    if validation.get("suggested_fix"):
        lines.append(f"validator suggested_fix: {validation.get('suggested_fix')}")
    return "\n".join(lines)

async def run(query: str, dry_run: bool = False) -> int:
    ui = UIState(agent_name="agent-harness", model_info=f"{CHECKER_MODEL} · {PLANNER_MODEL}")
    plan, results = [], []
    aligned, retry_count, max_retries = False, 0, 2
    last_feedback = ""

    async with ui:
        def refresh(): ui.refresh()

        try:
            # 1. Init
            s_init = ui.add_step("connect + warmup").start(); refresh()
            client = await setup_ollama(OLLAMA_URL, [CHECKER_MODEL, PLANNER_MODEL, REVIEW_MODEL])
            await asyncio.gather(
                _warmup(client, CHECKER_MODEL, phase="sanity"),
                _warmup(client, PLANNER_MODEL, phase="planner"),
                _warmup(client, REVIEW_MODEL, phase="validator"),
            )
            s_init.done("Ollama ready"); refresh()

            # 2. Sanity
            s_chk = ui.add_step("sanity check").start(); refresh()
            chk = await sanity_check(client, query, ui, refresh)
            if not chk.get("can_handle"):
                s_chk.error(f"out of scope: {chk.get('reason')}"); refresh()
                return 1
            s_chk.done(f"in scope ({chk.get('confidence', 0):.0%})"); refresh()

            # 3. Mission Loop
            while retry_count <= max_retries and not aligned:
                s_plan = ui.add_step(f"plan (attempt {retry_count})").start(); refresh()
                plan_raw = await plan_task(client, query, last_feedback, ui, refresh)
                from src.harness_logic import PlanStep
                plan = [PlanStep(**s) for s in plan_raw.get("steps", [])]
                rationale = plan_raw.get("rationale", "")
                s_plan.done(f"{len(plan)} steps"); refresh()

                if dry_run:
                    ui.add_step("save history").skip("dry-run — skipped")
                    refresh()
                    return 0

                results = await execute_plan(plan, ui, refresh, client)
                
                s_val = ui.add_step(f"validate (attempt {retry_count})").start(); refresh()
                val = await validate_result(client, query, results, ui, refresh)
                aligned = bool(val.get("aligned", True))
                score = float(val.get("quality_score", 1.0))
                
                if aligned or score >= 0.7:
                    s_val.done(f"✓ score={score:.0%}"); aligned = True
                else:
                    s_val.error(f"✗ score={score:.0%}")
                    last_feedback = _build_retry_feedback(results, val)
                    retry_count += 1
                refresh()

            # 4. Finalize
            s_save = ui.add_step("save history").start(); refresh()
            hist = load_history()
            hist.append({"query": query, "aligned": aligned, "score": score, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
            save_history(hist)
            save_experience(query, results, val, time.strftime("%Y-%m-%dT%H:%M:%S"))
            s_save.done(f"history updated"); refresh()

        except Exception as e:
            ui.add_step("fatal error").error(str(e)); refresh(); raise

    if results: print_result_card(ui, query, plan, results, rationale, val)
    return 0 if aligned else 2

def main() -> int:
    ap = argparse.ArgumentParser(description="ElfWeave Orchestrator", formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", nargs="*", help="Task query")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list-tools", action="store_true")
    ap.add_argument("--clear-history", action="store_true")
    args = ap.parse_args()

    if args.list_tools:
        print("\nTools:"); [print(f"  {td.signature}\n    {td.description}") for td in _TOOL_REGISTRY.values()]; return 0
    if args.clear_history:
        if HISTORY_PATH.exists(): HISTORY_PATH.unlink(); print("History cleared")
        return 0

    query = _read_query(args.query, "\n  What would you like to do?  ")
    if query is None:
        return 1
    if not query:
        return 0
    return asyncio.run(run(query, dry_run=args.dry_run))

if __name__ == "__main__":
    sys.exit(main())
