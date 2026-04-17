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
import os
from textwrap import dedent
from typing import List, Dict, Any

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.common.ui import UIState
from src.common.ollama import _chat_json, _warmup, setup_ollama

# ══════════════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════════════

AGENT_MODEL   = "qwen2.5:7b"
CHECKER_MODEL = "llama3.1:8b"

# ══════════════════════════════════════════════════════════════════════
#  Prompts
# ══════════════════════════════════════════════════════════════════════

ARCHITECT_SYSTEM = dedent("""\
    You are a Senior Code Architect. Your task is to analyze the provided source code 
    and evaluate its technical quality, modularity, and adherence to design patterns.

    Rules:
      1. Identify clear "Maintenance Risks" (Complexity, duplication, tight coupling).
      2. Suggest 2-3 specific architectural improvements.
      3. Rate the following metrics (0-10): Modularity, Readability, Scalability.
      4. Be technical and precise. Use file names and line numbers if possible.

    Return JSON:
    {
      "metrics": {
        "modularity": float,
        "readability": float,
        "scalability": float
      },
      "risks": [{"file": "string", "issue": "string", "severity": "high|med|low"}],
      "suggestions": ["string"],
      "patterns_detected": ["string"],
      "summary": "string"
    }
""")

# ══════════════════════════════════════════════════════════════════════
#  Logic
# ══════════════════════════════════════════════════════════════════════

async def analyze_code(files: List[Path], ui: UIState, client) -> Dict[str, Any]:
    code_bundle = ""
    for f in files:
        if not f.exists(): continue
        try:
            content = f.read_text(errors="replace")
            code_bundle += f"\n--- FILE: {f.name} ---\n{content[:5000]}" # Limit context
        except Exception as e:
            code_bundle += f"\n[error reading {f.name}: {e}]"

    res, _ = await _chat_json(
        client,
        AGENT_MODEL,
        ARCHITECT_SYSTEM,
        f"Please analyze these files and provide a structural review:\n{code_bundle}",
        ui,
        lambda: None,
        "architect"
    )
    return res

def print_audit_card(console: Console, analysis: Dict[str, Any], harness: bool = False):
    m = analysis.get("metrics", {})
    
    # 1. Metrics Table
    mt = Table(show_header=True, header_style="bold cyan", box=box.SIMPLE if harness else box.ROUNDED, expand=True)
    mt.add_column("Category", style="bold white")
    mt.add_column("Score", justify="right")
    mt.add_column("Status")
    
    for category, score in m.items():
        status = "[bold green]EXCELLENT[/bold green]" if score > 8 else "[bold yellow]DECENT[/bold yellow]" if score > 5 else "[bold red]RISKY[/bold red]"
        mt.add_row(category.capitalize(), f"{score}/10", status)

    if not harness:
        console.print(Panel(mt, title="[bold]Architectural Health Metrics[/bold]", border_style="cyan"))
    else:
        console.print(Rule("[bold]Technical Metrics[/bold]", style="cyan"))
        console.print(mt)

    # 2. Risks & Suggestions
    rt = Table(box=box.SIMPLE, show_header=True, expand=True)
    rt.add_column("Severity", width=10)
    rt.add_column("Issue / File")
    
    for r in analysis.get("risks", []):
        col = "red" if r.get("severity") == "high" else "yellow" if r.get("severity") == "med" else "blue"
        rt.add_row(f"[{col}]{r.get('severity','').upper()}[/{col}]", f"[bold]{r.get('file')}[/bold]: {r.get('issue')}")

    if not harness:
        console.print(Panel(rt, title="[bold red]Maintenance Risks[/bold red]", border_style="red"))
    else:
        console.print(Rule("[bold red]Maintenance Risks[/bold red]", style="red"))
        console.print(rt)

    # 3. Patterns & Summary
    patterns = ", ".join(analysis.get("patterns_detected", []))
    summary = analysis.get("summary", "No summary provided.")
    
    txt = Text()
    txt.append("\nPatterns Detected: ", style="bold green")
    txt.append(patterns or "None identified")
    txt.append("\n\nExecutive Summary:\n", style="bold white")
    txt.append(summary)

    if not harness:
        console.print(Panel(txt, title="[bold green]Design Review[/bold green]", border_style="green", padding=(1, 2)))
    else:
        console.print(Rule("[bold green]Design Review[/bold green]", style="green"))
        console.print(txt)


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
    parser.get_output = lambda: None # suppress default
    parser.add_argument("files", nargs="+", help="Files to analyze")
    parser.add_argument("--harness", action="store_true", help="Harness mode")
    args = parser.parse_args()

    # Pre-flight imports for Rich UI consistency
    from rich import box
    
    asyncio.run(run(args.files, harness=args.harness))
