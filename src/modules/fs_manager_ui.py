#!/usr/bin/env python3
"""
fs_manager_ui.py — Filesystem visualization
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
from rich.tree import Tree
from rich.table import Table
from rich.text import Text
from rich import box

from src.modules.fs_manager_logic import get_dir_contents

def render_tree(path: Path, tree: Tree):
    paths = get_dir_contents(path)
    for p in paths:
        if p.is_dir():
            branch = tree.add(f"[bold blue]📂 {p.name}[/bold blue]")
            render_tree(p, branch)
        else:
            suffix = p.suffix.lower()
            icon, style = "📄", "white"
            if suffix == ".py": icon, style = "🐍", "green"
            elif suffix in [".md", ".txt"]: icon, style = "📝", "yellow"
            elif suffix == ".json": icon, style = "📦", "cyan"
            
            size = p.stat().st_size
            size_str = f"{size / 1024:.1f} KB" if size > 1024 else f"{size} B"
            tree.add(Text.assemble((f"{icon} {p.name}", style), (f" ({size_str})", "dim grey50")))

def render_stats_table(stats: Dict[str, Any]) -> Table:
    st = Table(box=box.SIMPLE, show_header=True, expand=True)
    st.add_column("Metric", style="bold cyan")
    st.add_column("Value", justify="right")
    
    st.add_row("Total Files", str(stats["files"]))
    st.add_row("Total Directories", str(stats["dirs"]))
    st.add_row("Total Size", f"{stats['total_size'] / 1024 / 1024:.2f} MB")
    
    top_exts = sorted(stats["extensions"].items(), key=lambda x: x[1], reverse=True)[:5]
    ext_str = ", ".join([f"{k} ({v})" for k, v in top_exts])
    st.add_row("Primary Formats", ext_str)
    return st
