#!/usr/bin/env python3
"""
code_architect.py — architectural analysis agent · Design Review & Technical Debt
"""

from __future__ import annotations

import sys
from pathlib import Path

# Fix sys.path for robust modular imports
_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import argparse
import asyncio
from typing import List

from src.common.ui import UIState
from src.common.ollama import setup_ollama, _warmup
from src.common.config import AGENT_MODEL
from src.modules.code_architect_logic import analyze_code
from src.modules.code_architect_ui import print_audit_card

async def run(filenames: List[str], harness: bool = False):
    ui = UIState()
    console = ui.console
    
    client = await setup_ollama(ui)
    await _warmup(client, [AGENT_MODEL])

    paths = [Path(f).expanduser() for f in filenames]
    
    with ui.live_context() as refresh:
        s = ui.add_step(f"Analyzing {len(paths)} files...").start(); refresh()
        analysis = await analyze_code(paths, ui, client)
        s.done("Analysis complete"); refresh()
        
        ui.print_card("Architectural Audit", "Detailed report below...", border_color="cyan", padding=(0,1) if harness else (1,2))
        print_audit_card(console, analysis, harness=harness)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="Files to analyze")
    parser.add_argument("--harness", action="store_true", help="Harness mode")
    args = parser.parse_args()

    asyncio.run(run(args.files, harness=args.harness))
