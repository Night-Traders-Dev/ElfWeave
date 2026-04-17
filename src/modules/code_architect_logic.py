#!/usr/bin/env python3
"""
code_architect_logic.py — Design analysis & prompt logic
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import List, Dict, Any

from src.common.ui import UIState
from src.common.ollama import _chat_json
from src.common.config import AGENT_MODEL

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
      "metrics": { "modularity": float, "readability": float, "scalability": float },
      "risks": [{"file": "string", "issue": "string", "severity": "high|med|low"}],
      "suggestions": ["string"],
      "patterns_detected": ["string"],
      "summary": "string"
    }
""")

async def analyze_code(files: List[Path], ui: UIState, client) -> Dict[str, Any]:
    # Load recent experiences
    exp_path = Path.home() / ".agent_experience.jsonl"
    past_lessons = ""
    if exp_path.exists():
        try:
            with open(exp_path, "r") as f:
                for line in f.readlines()[-10:]:
                    entry = json.loads(line)
                    if not entry.get("aligned", True):
                        past_lessons += f"\n- Failure in '{entry.get('query')}': {entry.get('issues')}"
        except: pass

    code_bundle = ""
    for f in files:
        if not f.exists(): continue
        try:
            content = f.read_text(errors="replace")
            code_bundle += f"\n--- FILE: {f.name} ---\n{content[:5000]}"
        except Exception as e:
            code_bundle += f"\n[error reading {f.name}: {e}]"

    prompt = (
        f"HISTORICAL FAILURES IN THIS REPO:{past_lessons or 'None'}\n\n"
        f"SOURCE CODE TO AUDIT:\n{code_bundle}\n\n"
        "Please analyze these files. Check for architectural debt AND look for patterns "
        "that might be causing the HISTORICAL FAILURES listed above."
    )

    res, _ = await _chat_json(client, AGENT_MODEL, ARCHITECT_SYSTEM, prompt, ui, lambda: None, "architect")
    return res
