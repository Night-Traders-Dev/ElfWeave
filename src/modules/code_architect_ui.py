#!/usr/bin/env python3
"""
code_architect_ui.py — Design visualization
"""

from __future__ import annotations

from typing import Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from src.common.ui import clip_text, console_width, is_compact_width, panel_box_for_width, panel_padding_for_width


def format_audit_summary(analysis: Dict[str, Any], width: int = 80) -> str:
    lines = ["Technical Metrics"]
    for category, score in analysis.get("metrics", {}).items():
        lines.append(f"- {category.capitalize()}: {score}/10")

    risks = analysis.get("risks", [])
    lines.append("")
    lines.append("Maintenance Risks")
    if risks:
        for risk in risks[:5]:
            severity = str(risk.get("severity", "low")).upper()
            issue = f"{risk.get('file', 'unknown')}: {risk.get('issue', '')}"
            lines.append(f"- {severity}: {clip_text(issue, max(24, width - 10))}")
    else:
        lines.append("- None reported")

    suggestions = analysis.get("suggestions", [])
    if suggestions:
        lines.append("")
        lines.append("Suggestions")
        for suggestion in suggestions[:4]:
            lines.append(f"- {clip_text(suggestion, max(24, width - 4))}")

    patterns = analysis.get("patterns_detected", [])
    if patterns:
        lines.append("")
        lines.append("Patterns")
        lines.append(f"- {clip_text(', '.join(patterns), max(24, width - 4))}")

    lines.append("")
    lines.append("Summary")
    lines.append(clip_text(analysis.get("summary", "No summary provided."), max(24, width)))
    return "\n".join(lines)


def print_audit_card(console: Console, analysis: Dict[str, Any], harness: bool = False):
    width = console_width(console)
    compact = is_compact_width(width)
    panel_box = panel_box_for_width(width)
    padding = panel_padding_for_width(width)

    if harness:
        console.print(format_audit_summary(analysis, width=width))
        return

    m = analysis.get("metrics", {})
    
    # 1. Metrics Table
    mt = Table(show_header=True, header_style="bold cyan", box=box.SIMPLE if compact else box.ROUNDED, expand=True)
    mt.add_column("Category", style="bold white")
    mt.add_column("Score", justify="right")
    mt.add_column("Status")
    
    for category, score in m.items():
        status = "[bold green]EXCELLENT[/bold green]" if score > 8 else "[bold yellow]DECENT[/bold yellow]" if score > 5 else "[bold red]RISKY[/bold red]"
        mt.add_row(category.capitalize(), f"{score}/10", status)

    console.print(Panel(mt, title="[bold]Architectural Health Metrics[/bold]", border_style="cyan", box=panel_box, padding=padding))

    # 2. Risks & Suggestions
    rt = Table(box=box.SIMPLE, show_header=True, expand=True)
    rt.add_column("Severity", width=8 if compact else 10)
    rt.add_column("Issue / File")
    
    for r in analysis.get("risks", []):
        col = "red" if r.get("severity") == "high" else "yellow" if r.get("severity") == "med" else "blue"
        rt.add_row(f"[{col}]{r.get('severity','').upper()}[/{col}]", f"[bold]{r.get('file')}[/bold]: {r.get('issue')}")

    console.print(Panel(rt, title="[bold red]Maintenance Risks[/bold red]", border_style="red", box=panel_box, padding=padding))

    # 3. Patterns & Summary
    patterns = ", ".join(analysis.get("patterns_detected", []))
    summary = analysis.get("summary", "No summary provided.")
    
    txt = Text()
    txt.append("\nPatterns Detected: ", style="bold green")
    txt.append(patterns or "None identified")
    txt.append("\n\nExecutive Summary:\n", style="bold white")
    txt.append(summary)

    console.print(Panel(txt, title="[bold green]Design Review[/bold green]", border_style="green", box=panel_box, padding=padding))
