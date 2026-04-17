#!/usr/bin/env python3
"""
browser_agent.py — autonomous web navigation assistant using browser-use.
"""

from __future__ import annotations

import sys
import argparse
import asyncio
from pathlib import Path

# Ensure the project root is in sys.path for robust absolute imports
_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.common.ui import UIState
from src.common.ollama import setup_ollama
from src.modules.browser_logic import execute_browser_task

# ══════════════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════════════

from src.common.config import OLLAMA_URL, DEFAULT_MODEL
UI_REFRESH_HZ   = 10

# ══════════════════════════════════════════════════════════════════════
#  Main pipeline
# ══════════════════════════════════════════════════════════════════════
async def run(query: str, model: str, harness: bool = False) -> int:
    ui      = UIState(agent_name="browser-agent", model_info=f"{model}")
    if harness: ui.harness_mode = True
    
    async with ui:
        def refresh() -> None:
            ui.refresh()

        try:
            s_init = ui.add_step("connect + warmup").start(); refresh()
            # Ensure we have the model (and pull it if needed)
            _ = await setup_ollama(OLLAMA_URL, [model])
            s_init.done("Ollama ready · model verified"); refresh()
            
            # ── Load Domain Knowledge ──
            expert_manual = ""
            try:
                from src.modules.knowledge_logic import get_logic
                logic = get_logic()
                if logic.load():
                    results = logic.query("browser navigator protocol stealth extraction")
                    expert_manual = "\n".join(r.get("text", "") for r in results)
            except Exception:
                pass
                
            final_task = query
            if expert_manual:
                final_task = f"ADHERE TO THESE PROTOCOLS:\n{expert_manual}\n\nTASK:\n{query}"

            result = await execute_browser_task(final_task, model, OLLAMA_URL, ui, refresh, harness=harness)
            
            if not harness:
                ui.print_card("Browser Result", result, border_color="green", metadata=f"Task: {query[:50]}...")
                
            return 0
        except Exception as e:
            ui.add_step("fatal error").error(str(e)); refresh()
            return 1

def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous Browser Agent")
    parser.add_argument("query", nargs="*", help="Web task / question")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"LLM model (default: {DEFAULT_MODEL})")
    parser.add_argument("--harness", action="store_true", help="Harness mode")
    args = parser.parse_args()
    
    if args.query:
        query = " ".join(args.query).strip()
    elif sys.stdin.isatty():
        try:
            query = input("Browser task: ").strip()
        except EOFError:
            query = ""
    else:
        print("Error: No task provided. Pass a task on the command line.", file=sys.stderr)
        sys.exit(1)
    if not query.strip():
        print("Error: No task provided.")
        sys.exit(1)
        
    sys.exit(asyncio.run(run(query, args.model, harness=args.harness)))

if __name__ == "__main__":
    main()
