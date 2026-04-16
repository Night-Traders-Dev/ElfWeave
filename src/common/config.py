import multiprocessing
from pathlib import Path

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

def get_ollama_options(ctx_override: int = None):
    return {
        "num_ctx": ctx_override or OLLAMA_NUM_CTX,
        "num_gpu": OLLAMA_NUM_GPU,
        "num_thread": OLLAMA_NUM_THREAD,
        "temperature": 0.05
    }
