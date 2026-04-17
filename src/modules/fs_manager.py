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
import os
import re
from typing import List, Dict, Optional

from rich.console import Console
from rich.tree import Tree
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from src.common.ui import UIState

# ══════════════════════════════════════════════════════════════════════
#  Logic
# ══════════════════════════════════════════════════════════════════════

def build_tree(path: Path, tree: Tree, ignore_patterns: List[str] = None):
    if ignore_patterns is None:
        ignore_patterns = [".git", "__pycache__", ".venv", ".pytest_cache", ".DS_Store", "node_modules", ".gemini", ".system_generated"]

    # Sort: directories first, then files
    try:
        paths = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError:
        return

    for p in paths:
        if any(pat in p.name for pat in ignore_patterns):
            continue
            
        if p.is_dir():
            style = "bold blue"
            branch = tree.add(f"[bold blue]📂 {p.name}[/bold blue]")
            build_tree(p, branch, ignore_patterns)
        else:
            suffix = p.suffix.lower()
            icon = "📄"
            if suffix == ".py": icon = "🐍"; style = "green"
            elif suffix in [".md", ".txt"]: icon = "📝"; style = "yellow"
            elif suffix == ".json": icon = "📦"; style = "cyan"
            else: style = "white"
            
            size = p.stat().st_size
            size_str = f"{size / 1024:.1f} KB" if size > 1024 else f"{size} B"
            tree.add(Text.assemble((f"{icon} {p.name}", style), (f" ({size_str})", "dim grey50")))

def get_fs_stats(path: Path) -> Dict[str, any]:
    files = 0
    dirs = 0
    total_size = 0
    extensions = {}
    
    for root, d_names, f_names in os.walk(path):
        if any(x in root for x in [".git", ".venv", "node_modules", ".gemini"]): continue
        dirs += len(d_names)
        files += len(f_names)
        for f in f_names:
            p = Path(root) / f
            try:
                sz = p.stat().st_size
                total_size += sz
                ext = p.suffix or "no-ext"
                extensions[ext] = extensions.get(ext, 0) + 1
            except: pass
            
    return {
        "files": files,
        "dirs": dirs,
        "total_size": total_size,
        "extensions": extensions
    }

async def run(target_dir: str, harness: bool = False):
    ui = UIState()
    console = ui.console
    
    root_path = Path(target_dir).expanduser().resolve()
    if not root_path.exists():
        console.print(f"[red]Error: Path {target_dir} does not exist.[/red]")
        return

    with ui.live_context() as refresh:
        s = ui.add_step(f"Scanning {root_path.name}...").start(); refresh()
        stats = await asyncio.to_thread(get_fs_stats, root_path)
        s.done(f"Scanned {stats['files']} files"); refresh()

        # 1. Project Tree
        tree = Tree(f"[bold white]ROOT: {root_path.name}[/bold white]")
        await asyncio.to_thread(build_tree, root_path, tree)
        
        ui.print_card("Repository Structure", tree, border_color="blue", padding=(0,1) if harness else (1,2))

        # 2. Stats Table
        st = Table(box=box.SIMPLE, show_header=True, expand=True)
        st.add_column("Metric", style="bold cyan")
        st.add_column("Value", justify="right")
        
        st.add_row("Total Files", str(stats["files"]))
        st.add_row("Total Directories", str(stats["dirs"]))
        st.add_row("Total Size", f"{stats['total_size'] / 1024 / 1024:.2f} MB")
        
        # Top Extensions
        top_exts = sorted(stats["extensions"].items(), key=lambda x: x[1], reverse=True)[:5]
        ext_str = ", ".join([f"{k} ({v})" for k, v in top_exts])
        st.add_row("Primary Formats", ext_str)

        ui.print_card("Filesystem Analytics", st, border_color="green", padding=(0,1) if harness else (1,2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default=".", help="Directory to explore")
    parser.add_argument("--harness", action="store_true", help="Harness mode")
    args = parser.parse_args()

    asyncio.run(run(args.path, harness=args.harness))
