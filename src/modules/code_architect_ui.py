#!/usr/bin/env python3
"""
code_architect_ui.py — Design visualization
"""

from __future__ import annotations

from typing import Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich import box

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
