#!/usr/bin/env python3
"""
harness_ui.py — Specialized TUI rendering for the Orchestrator
"""

from __future__ import annotations

from typing import List, Dict
from rich.table import Table
from rich.text import Text

from src.common.ui import UIState, clip_text, console_width, is_compact_width, is_tiny_width
from src.harness_logic import StepResult, PlanStep

def print_result_card(
    ui: UIState,
    query: str,
    plan: List[PlanStep],
    results: List[StepResult],
    rationale: str,
    validation: Dict,
) -> None:
    width = console_width(ui.console)
    compact = is_compact_width(width)
    tiny = is_tiny_width(width)

    # ── Plan Table ──
    tbl = Table(show_header=True, header_style="bold blue", box=None, padding=(0, 1), expand=True)
    tbl.add_column("#", style="dim", width=3, no_wrap=True)
    tbl.add_column("Tool", style="cyan bold", min_width=10)
    if tiny:
        tbl.add_column("Details", ratio=10)
    else:
        tbl.add_column("Description", ratio=7)
        tbl.add_column("Status", width=8, no_wrap=True)
        if not compact:
            tbl.add_column("Output", ratio=8)

    for i, r in enumerate(results):
        status = "error" if r.error else "ok"
        desc = clip_text(r.plan_step.description, 34 if compact else 52)
        preview = clip_text(r.output, 32 if compact else 56)
        if tiny:
            detail = f"{desc}\nstatus: {status}"
            if preview:
                detail += f"\n{preview}"
            tbl.add_row(str(i), clip_text(r.plan_step.tool, 14), detail)
        elif compact:
            tbl.add_row(str(i), clip_text(r.plan_step.tool, 16), desc, status)
        else:
            tbl.add_row(str(i), clip_text(r.plan_step.tool, 18), desc, status, preview)

    ui.print_card("Execution Plan", tbl, border_color="blue", metadata=clip_text(rationale, 120))

    # ── Final Output ──
    if results:
        final_out = "\n".join(line.rstrip() for line in results[-1].output[:8000].splitlines()).strip()
        ui.print_card("Final Output", Text.from_ansi(final_out[:8000]), border_color="green", padding=(0, 1))

    # ── Validation ──
    aligned = validation.get("aligned", True)
    score   = float(validation.get("quality_score", 1.0))
    notes   = validation.get("notes", "")
    issues  = validation.get("issues", [])

    val_color = "green" if aligned and score >= 0.7 else "yellow" if score >= 0.4 else "red"
    val_lines = Table.grid(padding=(0, 1))
    val_lines.add_column(style=f"bold {val_color}", no_wrap=True)
    val_lines.add_column(style="white")
    val_lines.add_row("Aligned", "yes" if aligned else "no")
    val_lines.add_row("Score", f"{score:.0%}")
    val_lines.add_row("Notes", clip_text(notes, 120))
    if issues:
        val_lines.add_row("Issues", clip_text(" · ".join(str(x) for x in issues[:3]), 120))

    ui.print_card("Validation Result", val_lines, border_color=val_color)
