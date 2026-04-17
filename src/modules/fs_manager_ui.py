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

from src.common.ui import clip_text
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


def summarize_tree(path: Path, width: int, max_depth: int = 3, max_entries: int = 40) -> str:
    lines: List[str] = [f"ROOT: {path.name}"]
    count = 1

    def walk(current: Path, prefix: str = "", depth: int = 0) -> None:
        nonlocal count
        if depth >= max_depth or count >= max_entries:
            return
        entries = get_dir_contents(current)
        for idx, entry in enumerate(entries):
            if count >= max_entries:
                break
            branch = "└── " if idx == len(entries) - 1 else "├── "
            label = f"{entry.name}/" if entry.is_dir() else entry.name
            lines.append(prefix + branch + clip_text(label, max(12, width - len(prefix) - 4)))
            count += 1
            if entry.is_dir():
                extension = "    " if idx == len(entries) - 1 else "│   "
                walk(entry, prefix + extension, depth + 1)
        if depth == 0 and count >= max_entries:
            lines.append("…")

    walk(path)
    return "\n".join(lines)


def summarize_stats(stats: Dict[str, Any]) -> str:
    top_exts = sorted(stats["extensions"].items(), key=lambda x: x[1], reverse=True)[:5]
    ext_str = ", ".join([f"{k} ({v})" for k, v in top_exts]) or "none"
    return "\n".join(
        [
            f"Files: {stats['files']}",
            f"Directories: {stats['dirs']}",
            f"Size: {stats['total_size'] / 1024 / 1024:.2f} MB",
            f"Primary formats: {ext_str}",
        ]
    )
