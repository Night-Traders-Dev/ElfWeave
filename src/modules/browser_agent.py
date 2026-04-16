#!/usr/bin/env python3
"""
browser_agent.py — autonomous web navigation assistant using browser-use.
"""

from __future__ import annotations

import sys
import time
import argparse
import asyncio
from pathlib import Path
from typing import Optional

# Ensure the project root is in sys.path for robust absolute imports
_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from rich.console import Console
from rich.live import Live

# Common imports
from src.common.ui import UIState
from src.common.ollama import setup_ollama
from .browser_logic import execute_browser_task

# ══════════════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════════════

OLLAMA_URL      = "http://localhost:11434"
DEFAULT_MODEL   = "qwen2.5:7b" # Suggested model for browser-use logic
UI_REFRESH_HZ   = 10

# ══════════════════════════════════════════════════════════════════════
#  Main pipeline
# ══════════════════════════════════════════════════════════════════════

async def run(query: str, model: str, harness: bool = False) -> int:
    console = Console()
    ui      = UIState(agent_name="browser-agent", model_info=f"{model}")
    
    if harness:
        def refresh() -> None: pass
        live = None
    else:
        live = Live(ui.render(), refresh_per_second=UI_REFRESH_HZ, console=console, screen=False)
        live.start()
        def refresh() -> None:
            if live: live.update(ui.render())

    try:
        s_init = ui.add_step("connect + warmup").start(); refresh()
        # Ensure we have the model (and pull it if needed)
        _ = setup_ollama(OLLAMA_URL, [model])
        s_init.done("Ollama ready · model verified"); refresh()

        result = await execute_browser_task(query, model, OLLAMA_URL, ui, refresh, harness=harness)
        
        ui.running = False; refresh()
        
        if not harness:
            console.print("\n[bold green]Browser Agent Result:[/bold green]")
            console.print(result)
            
        return 0
    except Exception as e:
        ui.add_step("fatal error").error(str(e)); refresh()
        return 1
    finally:
        if not harness and live: live.stop()

def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous Browser Agent")
    parser.add_argument("query", nargs="*", help="Web task / question")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"LLM model (default: {DEFAULT_MODEL})")
    parser.add_argument("--harness", action="store_true", help="Harness mode")
    args = parser.parse_args()
    
    query = " ".join(args.query) if args.query else input("Browser task: ")
    if not query.strip():
        print("Error: No task provided.")
        sys.exit(1)
        
    sys.exit(asyncio.run(run(query, args.model, harness=args.harness)))

if __name__ == "__main__":
    main()
