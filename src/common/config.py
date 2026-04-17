import os
from pathlib import Path
from textwrap import dedent
import multiprocessing

from .kernel_bootstrap import (
    DEFAULT_MEGAKERNEL_MODEL,
    DEFAULT_MEGAKERNEL_PHASES,
    bundled_megakernel_repo,
)

# ══════════════════════════════════════════════════════════════════════
#  Hardware & Performance Config (RTX 5060 8GB / 32GB RAM)
# ════════════════════════════════════════════════─═════════════════════

# Ollama Global Defaults
OLLAMA_URL          = "http://localhost:11434"

# Resource Limits
# For 8GB VRAM, 8k context is the "sweet spot" while staying 100% on GPU.
# Note: This is a default value; actual context size depends on model capabilities
OLLAMA_NUM_CTX      = int(os.getenv("ELFWEAVE_OLLAMA_CTX", "8192"))
OLLAMA_NUM_GPU      = 99        # Force max layers offload
OLLAMA_NUM_THREAD   = multiprocessing.cpu_count() // 2

# Model Selection (Tiered for 8GB VRAM)
# Note: qwen3.5:0.8b is ideal for lightweight tasks (CHECKER, REVIEW) using ~0.5-1GB VRAM
#       qwen3.5:2b provides better reasoning for moderate tasks (~1.5-2GB VRAM)
#       qwen3.5:4b recommended for complex planning and tool orchestration (~3-4GB VRAM)
#       qwen3.5:9b only when VRAM allows or for high-fidelity generation (~6-7GB VRAM)
PLANNER_MODEL       = os.getenv("ELFWEAVE_PLANNER_MODEL", "qwen3.5:4b")      # Stronger at tool-use/reasoning
CHECKER_MODEL       = os.getenv("ELFWEAVE_CHECKER_MODEL", "qwen3.5:0.8b")   # Lightweight sanity checks (0.8B works great here)
REVIEW_MODEL        = os.getenv("ELFWEAVE_REVIEW_MODEL", "qwen3.5:0.8b")    # Lightweight validation (0.8B works great here)
AGENT_MODEL         = os.getenv("ELFWEAVE_AGENT_MODEL", "qwen3.5:2b")       # Default persona model (balanced performance/VRAM)
DEFAULT_MODEL       = AGENT_MODEL

REPO_ROOT           = Path(__file__).resolve().parents[2]
DEFAULT_MEGAKERNEL_REPO = bundled_megakernel_repo(REPO_ROOT)

# Optional Megakernel Backend
INFERENCE_BACKEND   = os.getenv("ELFWEAVE_INFERENCE_BACKEND", "ollama").strip().lower()
MEGAKERNEL_ENABLED  = INFERENCE_BACKEND in {"hybrid", "megakernel"} or os.getenv("ELFWEAVE_MEGAKERNEL_ENABLE", "0") == "1"
MEGAKERNEL_REPO     = os.getenv(
    "ELFWEAVE_MEGAKERNEL_REPO",
    str(DEFAULT_MEGAKERNEL_REPO) if DEFAULT_MEGAKERNEL_REPO.exists() else "",
).strip()
MEGAKERNEL_MODEL    = os.getenv("ELFWEAVE_MEGAKERNEL_MODEL", DEFAULT_MEGAKERNEL_MODEL).strip()
MEGAKERNEL_MAX_TOKENS = int(os.getenv("ELFWEAVE_MEGAKERNEL_MAX_TOKENS", "256"))
MEGAKERNEL_PHASES   = tuple(
    part.strip().lower()
    for part in os.getenv(
        "ELFWEAVE_MEGAKERNEL_PHASES",
        DEFAULT_MEGAKERNEL_PHASES,
    ).split(",")
    if part.strip()
)
MEGAKERNEL_FALLBACK = os.getenv("ELFWEAVE_MEGAKERNEL_FALLBACK", "1") != "0"

# Memory & Knowledge
KNOWLEDGE_DIR       = Path.home() / ".elfweave_knowledge"
AGENT_MANUALS_DIR   = REPO_ROOT / "knowledge" / "agents"

# Global Paths & Timing
HISTORY_PATH        = Path.home() / ".harness_history.json"
EXPERIENCE_PATH     = Path.home() / ".agent_experience.jsonl"
UI_REFRESH_HZ       = 10
MAX_STREAM_LINES    = 8
DEFAULT_TIMEOUT     = 30

def get_ollama_options(ctx_override: int = None):
    return {
        "num_ctx": ctx_override or OLLAMA_NUM_CTX,
        "num_gpu": OLLAMA_NUM_GPU,
        "num_thread": OLLAMA_NUM_THREAD,
        "temperature": 0.05
    }

# ══════════════════════════════════════════════════════════════════════
#  Global Prompts
# ════════════════════════════════════════════════─═════════════════════

PLANNER_SYSTEM = dedent("""\
    You are a high-autonomy task planner. 
    You break down complex queries into sequential tool calls.
    
    Rules:
      1. Use only tools listed in the catalogue.
      2. Specialist Priority: ALWAYS prefer specialized agents (weather, browser, knowledge_query) 
         over raw utilities (http_get, shell) for their respective domains.
      3. Minimal Hallucination: Do NOT invent arguments, URLs, or paths.
      4. Signature Audit: Carefully match your "args" to the parameters in the catalogue.
      5. Research-First: If a tool fails with a technical error you don't recognize, 
         call `analyze_failure` and then `research_fix` to learn the solution from the web.
      6. Self-Healing: Use `repair_code` ONLY after identifying a specific fix (via research 
         or analysis) before retrying the task.
      7. To pass a prior step's output as an arg value, use the string "{step_N}"
         where N is the 0-based index of the prior step (e.g. "{step_0}").
      8. Use past lessons to avoid repeating failed tool sequences or bad arguments.
      9. Use `knowledge_query` when repo-local context would materially improve the plan.
      10. Respond with ONLY a JSON object — no markdown, no prose.

    {
      "rationale": "why this plan works",
      "steps": [
        {"tool": "name", "args": {...}, "description": "why this step"}
      ]
    }
""")

SANITY_SYSTEM = dedent("""\
    You are a routing agent for a multi-tool AI system.
    Respond with ONLY a JSON object:
    {
      "can_handle": bool,
      "confidence": 0.0,
      "reason": "one sentence",
      "relevant_tools": ["name"]
    }
""")

VALIDATOR_SYSTEM = dedent("""\
    You are a high-fidelity Quality Assurance agent.
    Respond with ONLY a JSON object:
    {
      "aligned": bool,
      "quality_score": float,
      "notes": "string",
      "issues": ["string"],
      "suggested_fix": "string"
    }
""")
