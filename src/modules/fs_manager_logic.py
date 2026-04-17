#!/usr/bin/env python3
"""
fs_manager_logic.py — Filesystem scanning & analytics
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Any

def get_fs_stats(path: Path) -> Dict[str, Any]:
    files = 0
    dirs = 0
    total_size = 0
    extensions = {}
    
    ignore = [".git", ".venv", "node_modules", ".gemini", "__pycache__"]
    
    for root, d_names, f_names in os.walk(path):
        if any(x in root for x in ignore): continue
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

def get_dir_contents(path: Path, ignore_patterns: List[str] = None) -> List[Path]:
    if ignore_patterns is None:
        ignore_patterns = [".git", "__pycache__", ".venv", ".pytest_cache", ".DS_Store", "node_modules", ".gemini", ".system_generated"]

    try:
        return sorted(
            [p for p in path.iterdir() if not any(pat in p.name for pat in ignore_patterns)],
            key=lambda p: (not p.is_dir(), p.name.lower())
        )
    except PermissionError:
        return []
