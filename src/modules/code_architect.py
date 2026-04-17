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
from src.common.config import AGENT_MODEL, OLLAMA_URL
from src.modules.code_architect_logic import analyze_code
from src.modules.code_architect_ui import print_audit_card

async def run(filenames: List[str], harness: bool = False) -> int:
    ui = UIState(agent_name="code-architect", model_info=AGENT_MODEL)
    if harness:
        ui.harness_mode = True
    console = ui.console
    paths = [Path(f).expanduser() for f in filenames]
    analysis = {}

    async with ui:
        def refresh() -> None:
            ui.refresh()

        try:
            s_init = ui.add_step("connect + warmup").start(); refresh()
            client = await setup_ollama(OLLAMA_URL, [AGENT_MODEL])
            await _warmup(client, AGENT_MODEL)
            s_init.done("Ollama ready"); refresh()

            s = ui.add_step(f"Analyzing {len(paths)} files...").start(); refresh()
            analysis = await analyze_code(paths, ui, client)
            s.done("Analysis complete"); refresh()
        except Exception as exc:
            ui.add_step("fatal error").error(str(exc)); refresh()
            return 1

    print_audit_card(console, analysis, harness=harness)
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="Files to analyze")
    parser.add_argument("--harness", action="store_true", help="Harness mode")
    args = parser.parse_args()

    raise SystemExit(asyncio.run(run(args.files, harness=args.harness)))
