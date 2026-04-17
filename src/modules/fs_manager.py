#!/usr/bin/env python3
"""
fs_manager.py — filesystem explorer & repository manager
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
from rich.tree import Tree

from src.common.ui import UIState
from src.modules.fs_manager_logic import get_fs_stats
from src.modules.fs_manager_ui import render_tree, render_stats_table

async def run(target_dir: str, harness: bool = False) -> int:
    ui = UIState(agent_name="fs-manager", model_info="Local filesystem")
    if harness:
        ui.harness_mode = True
    console = ui.console
    
    root_path = Path(target_dir).expanduser().resolve()
    if not root_path.exists():
        console.print(f"[red]Error: Path {target_dir} does not exist.[/red]")
        return 1

    async with ui:
        def refresh() -> None:
            ui.refresh()

        s = ui.add_step(f"Scanning {root_path.name}...").start(); refresh()
        stats = await asyncio.to_thread(get_fs_stats, root_path)
        s.done(f"Scanned {stats['files']} files"); refresh()

        # 1. Project Tree
        tree = Tree(f"[bold white]ROOT: {root_path.name}[/bold white]")
        await asyncio.to_thread(render_tree, root_path, tree)
        ui.print_card("Repository Structure", tree, border_color="blue", padding=(0,1) if harness else (1,2))

        # 2. Stats Table
        st = render_stats_table(stats)
        ui.print_card("Filesystem Analytics", st, border_color="green", padding=(0,1) if harness else (1,2))
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default=".", help="Directory to explore")
    parser.add_argument("--harness", action="store_true", help="Harness mode")
    args = parser.parse_args()

    raise SystemExit(asyncio.run(run(args.path, harness=args.harness)))
