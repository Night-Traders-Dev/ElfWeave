#!/usr/bin/env python3
"""
knowledge_agent.py — local codebase knowledge base (RAG) agent.
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Optional, List

# Ensure the project root is in sys.path for robust absolute imports
_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from rich.console import Console
from rich.live import Live
from rich.panel import Panel

# Common imports
from src.common.ui import UIState
from .knowledge_logic import get_logic

# ══════════════════════════════════════════════════════════════════════
#  Main pipeline
# ══════════════════════════════════════════════════════════════════════

def run_index(path: str, ui: UIState, refresh: callable) -> int:
    s_idx = ui.add_step(f"indexing {path}").start(); refresh()
    logic = get_logic()
    count = logic.index_files(Path(path).expanduser())
    s_idx.done(f"indexed {count} chunks"); refresh()
    return 0

def run_query(query: str, ui: UIState, refresh: callable, harness: bool = False) -> str:
    s_q = ui.add_step("querying KB").start(); refresh()
    logic = get_logic()
    results = logic.query(query)
    
    if not results:
        s_q.error("no results found"); refresh()
        return "No relevant information found in the local knowledge base."
        
    s_q.done(f"found {len(results)} matches"); refresh()
    
    # Combine results into a context string
    context = []
    for r in results:
        context.append(f"--- File: {r['path']} ---\n{r.get('text', '')}")
    
    return "\n\n".join(context)

def main() -> None:
    parser = argparse.ArgumentParser(description="Local Knowledge Base Agent")
    parser.add_argument("--index", metavar="PATH", help="Index a directory")
    parser.add_argument("--query", metavar="TEXT", help="Query the knowledge base")
    parser.add_argument("--harness", action="store_true", help="Harness mode")
    args = parser.parse_args()
    
    console = Console()
    ui = UIState(agent_name="knowledge-agent", model_info="Local FAISS + MiniLM")
    
    if args.harness:
        def refresh() -> None: pass
    else:
        live = Live(ui.render(), refresh_per_second=10, console=console, screen=False)
        live.start()
        def refresh() -> None:
            if live: live.update(ui.render())

    try:
        if args.index:
            run_index(args.index, ui, refresh)
            if not args.harness:
                print("\n[bold green]Indexing complete.[/bold green]")
        
        if args.query:
            result = run_query(args.query, ui, refresh, harness=args.harness)
            if not args.harness:
                console.print(Panel(result, title="Search Results"))
            else:
                # In harness mode, we output the result to stdout for capture
                print(result)
                
        ui.running = False; refresh()
        if not args.harness and 'live' in locals():
            live.stop()
    except Exception as e:
        ui.add_step("fatal error").error(str(e)); refresh()
        sys.exit(1)

if __name__ == "__main__":
    main()
