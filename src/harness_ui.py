#!/usr/bin/env python3
"""
harness_ui.py — Specialized TUI rendering for the Orchestrator
"""

from __future__ import annotations

from typing import List, Dict
from rich.table import Table
from rich.text import Text
from rich import box

from src.common.ui import UIState
from src.harness_logic import StepResult, PlanStep

def print_result_card(
    ui: UIState,
    query: str,
    plan: List[PlanStep],
    results: List[StepResult],
    rationale: str,
    validation: Dict,
) -> None:
    # ── Plan Table ──
    tbl = Table(show_header=True, header_style="bold blue", box=None, padding=(0, 2), expand=True)
    tbl.add_column("#",    style="dim",        ratio=1)
    tbl.add_column("Tool", style="cyan bold",  ratio=3)
    tbl.add_column("Description",              ratio=8)
    tbl.add_column("Status",                   ratio=2)
    tbl.add_column("Output preview",           ratio=10)

    for i, r in enumerate(results):
        status = "[red]error[/red]" if r.error else "[green]ok[/green]"
        preview = r.output[:35].replace("\n", " ") + ("…" if len(r.output) > 35 else "")
        tbl.add_row(str(i), r.plan_step.tool, r.plan_step.description[:34], status, preview)

    ui.print_card("Execution Plan", tbl, border_color="blue", metadata=rationale[:80])

    # ── Final Output ──
    if results:
        final_out = results[-1].output
        ui.print_card("Final Output", Text.from_ansi(final_out[:8000]), border_color="green", padding=(0, 1))

    # ── Validation ──
    aligned = validation.get("aligned", True)
    score   = float(validation.get("quality_score", 1.0))
    notes   = validation.get("notes", "")
    issues  = validation.get("issues", [])

    val_color = "green" if aligned and score >= 0.7 else "yellow" if score >= 0.4 else "red"
    val_lines = Text()
    val_lines.append(f"{'✓' if aligned else '✗'} aligned  ", f"bold {val_color}")
    val_lines.append(f"quality {score:.0%}  ", "white")
    val_lines.append(notes[:80], "dim")
    if issues:
        val_lines.append("\n  issues: " + " · ".join(str(x) for x in issues[:3]), "yellow")

    ui.print_card("Validation Result", val_lines, border_color=val_color)
