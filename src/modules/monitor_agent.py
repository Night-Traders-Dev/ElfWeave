#!/usr/bin/env python3
"""
monitor_agent.py — real-time log watcher and anomaly detection agent.
"""

from __future__ import annotations

import sys
import argparse
import asyncio
import time
from pathlib import Path
from typing import Optional, List

# Ensure the project root is in sys.path for robust absolute imports
_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

# Common imports
from src.common.ui import UIState
from src.common.ollama import setup_ollama
from .monitor_logic import tail_file, classify_log_line

# ══════════════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════════════

OLLAMA_URL      = "http://localhost:11434"
MONITOR_MODEL   = "llama3.2:1b"
UI_REFRESH_HZ   = 10

# ══════════════════════════════════════════════════════════════════════
#  Main pipeline
# ══════════════════════════════════════════════════════════════════════

async def run_monitor(log_path: Path, model: str, host: str) -> int:
    console = Console()
    ui = UIState(agent_name="monitor-agent", model_info=f"{model}")
    
    live = Live(ui.render(), refresh_per_second=UI_REFRESH_HZ, console=console, screen=False)
    live.start()
    
    def refresh() -> None:
        if live: live.update(ui.render())
    
    try:
        s_init = ui.add_step("connect + warmup").start(); refresh()
        client = await setup_ollama(host, [model])
        s_init.done("monitoring active"); refresh()
        
        # Store history for the UI
        history: List[dict] = []
        
        # Start the async tailing loop
        async for line in tail_file(log_path):
            if not line.strip(): continue
            
            # When a new line arrives, classify it using the LLM
            s_ana = ui.add_step("analyzing log").start(); refresh()
            res = await classify_log_line(client, model, line, ui, refresh)
            s_ana.done(f"{res.get('severity', 'info').upper()} · {res.get('intent', 'unknown')}")
            
            # Add to history (max 10)
            history.append({"time": time.ctime(), "line": line[:80], "res": res})
            if len(history) > 10: history.pop(0)
            
            # Push chunk to TUI
            col = "red" if res.get('severity') == 'error' else "yellow" if res.get('severity') == 'warn' else "white"
            ui.push_chunk(f"[{col}]{res.get('summary', line[:60])}[/]")
            refresh()
            
            # Tiny sleep to ensure UI loop gets a chance if logs are fast
            await asyncio.sleep(0.01)

    except KeyboardInterrupt:
        return 0
    finally:
        if live:
            live.stop()
    return 0

async def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time Log Monitor Agent")
    parser.add_argument("path", help="Path to log file to monitor")
    parser.add_argument("--model", default=MONITOR_MODEL, help=f"LLM model (default: {MONITOR_MODEL})")
    args = parser.parse_args()
    
    p = Path(args.path).expanduser()
    if not p.exists():
        # Create empty if it doesn't exist
        p.touch()
        
    res = await run_monitor(p, args.model, OLLAMA_URL)
    sys.exit(res)

if __name__ == "__main__":
    asyncio.run(main())
