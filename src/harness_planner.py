#!/usr/bin/env python3
"""
harness_planner.py — LLM Orchestration (Sanity, Planning, Validation, Repair)
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Callable
from ollama import AsyncClient

from src.common.ui import UIState
from src.common.ollama import _chat_json, _stream_chat
from src.common.config import (
    PLANNER_MODEL, CHECKER_MODEL, REVIEW_MODEL,
    PLANNER_SYSTEM, SANITY_SYSTEM, VALIDATOR_SYSTEM
)

async def sanity_check(client: AsyncClient, query: str, ui: UIState, refresh: Callable) -> Dict:
    from src.harness_logic import get_tool_catalogue
    catalogue = get_tool_catalogue()
    prompt = f"USER QUERY: {query}\n\nTOOLS AVAILABLE:\n{catalogue}"
    res, _ = await _chat_json(client, CHECKER_MODEL, SANITY_SYSTEM, prompt, ui, refresh, "sanity")
    return res

async def validate_result(client: AsyncClient, query: str, results: List, ui: UIState, refresh: Callable) -> Dict:
    if not results: return {"aligned": False, "quality_score": 0.0, "notes": "No output."}
    full_output = "\n".join([f"Step {i}: {r.output}" for i, r in enumerate(results)])
    prompt = f"USER QUERY: {query}\n\nPIPELINE OUTPUT:\n{full_output}"
    res, _ = await _chat_json(client, REVIEW_MODEL, VALIDATOR_SYSTEM, prompt, ui, refresh, "validator")
    return res

async def plan_task(client: AsyncClient, query: str, feedback: str, ui: UIState, refresh: Callable) -> Dict:
    from src.harness_logic import get_tool_catalogue, get_learned_lessons
    catalogue = get_tool_catalogue()
    lessons = get_learned_lessons(query=query, limit=10)
    prompt = f"USER QUERY: {query}\n\nTOOLS AVAILABLE:\n{catalogue}\n\nPAST LESSONS:\n{lessons}"
    if feedback: prompt += f"\n\nPRIOR ATTEMPT FEEDBACK:\n{feedback}"
    res, _ = await _chat_json(client, PLANNER_MODEL, PLANNER_SYSTEM, prompt, ui, refresh, "planner")
    return res

async def analyze_failure_logic(issues: str, plan_context: str, ui: UIState, refresh: Callable, client: AsyncClient) -> str:
    from src.harness_logic import get_tool_catalogue, get_learned_lessons
    repo_root = Path(__file__).resolve().parent.parent
    files = list(repo_root.rglob("*.py"))
    code_context = ""
    relevant = [f for f in files if f.name in issues]
    for f in (relevant + [f for f in files if f not in relevant])[:5]:
        content = await asyncio.to_thread(f.read_text, errors="replace")
        code_context += f"\n--- {f.name} ---\n{content[:2500]}"
    
    past = get_learned_lessons(query=issues, limit=10)
    prompt = f"LOGS:\n{issues}\n\nPLAN:\n{plan_context}\n\nTOOLS:\n{get_tool_catalogue()}\n\nPAST:\n{past}\n\nCODE:\n{code_context}"
    
    res, _ = await _chat_json(client, PLANNER_MODEL, "Analyze root cause and suggest fix. JSON: { \"cause\": \"...\", \"fix\": \"...\", \"needs_research\": bool }", prompt, ui, refresh, "analyzer")
    analysis = f"Cause: {res.get('cause')}\nFix: {res.get('fix')}"
    if res.get("needs_research"): analysis += "\nSUGGESTION: Call research_fix."
    return analysis

async def repair_code_logic(filename: str, recommended_fix: str, ui: UIState, refresh: Callable, client: AsyncClient) -> str:
    repo_root = Path(__file__).resolve().parent.parent
    files = list(repo_root.rglob(filename))
    if not files: return f"[error] file not found: {filename}"
    target = files[0]
    content = await asyncio.to_thread(target.read_text, errors="replace")
    
    prompt = f"FILE: {target.name}\nCONTENT:\n{content}\nFIX:\n{recommended_fix}\n\nREWRITE ENTIRE FILE with fix. No prose."
    s = ui.add_step(f"patching {target.name}").start(); refresh()
    updated, _ = await _stream_chat(client, REVIEW_MODEL, [{"role": "user", "content": prompt}], ui, refresh, "repair")
    
    if not updated or len(updated) < 10: s.error("empty"); return "[error] repair failed"
    # Strip markdown code fences that LLMs commonly wrap responses in
    import re as _re
    clean = _re.sub(r"^```[a-z]*\n?", "", updated.strip(), flags=_re.MULTILINE)
    clean = _re.sub(r"\n?```$", "", clean.strip()).strip()
    if not clean or len(clean) < 10: s.error("empty after fence-strip"); return "[error] repair produced no code"
    await asyncio.to_thread(target.write_text, clean)
    s.done("patched"); refresh()
    return f"Successfully repaired {target.name}."
