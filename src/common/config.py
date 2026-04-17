from textwrap import dedent

# ══════════════════════════════════════════════════════════════════════
#  Hardware & Performance Config (RTX 5060 8GB / 32GB RAM)
# ════════════════════════════════════════════════─═════════════════════

# Ollama Global Defaults
OLLAMA_URL          = "http://localhost:11434"

# Resource Limits
# For 8GB VRAM, 8k context is the "sweet spot" while staying 100% on GPU.
OLLAMA_NUM_CTX      = 8192
OLLAMA_NUM_GPU      = 99        # Force max layers offload
OLLAMA_NUM_THREAD   = multiprocessing.cpu_count() // 2

# Model Selection (Tiered for 8GB VRAM)
PLANNER_MODEL       = "qwen2.5:7b"      # Stronger at tool-use/reasoning
CHECKER_MODEL       = "llama3.1:8b"     # Reliable for sanity checks
REVIEW_MODEL        = "llama3.1:8b"     # Standard validation model
AGENT_MODEL         = "llama3.1:8b"     # Default persona model

# Memory & Knowledge
KNOWLEDGE_DIR       = Path.home() / ".elfweave_knowledge"
AGENT_MANUALS_DIR   = Path(__file__).resolve().parent.parent.parent / "knowledge" / "agents"

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
      8. Respond with ONLY a JSON object — no markdown, no prose.

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
