"""
ElfWeave: Evolutionary Multi-Agent Ecosystem

A modular multi-agent platform optimized for local execution on hardware
with 8GB VRAM (RTX 5060) and 32GB RAM, utilizing a tiered LLM architecture.

Key Features:
- Evolutionary Self-Repair: Agents can autonomously fix their own code
- Web-Learning Loop: Research fixes via browser agent
- Persistent Experience Store: Long-term memory at ~/.agent_experience.jsonl
- Responsive Terminal UI: Adapts to narrow and wide terminals
- Optional Megakernel Backend: Hybrid inference with Luce Megakernel

Package Structure:
- src.common: Shared utilities (config, ollama client, UI, types)
- src.modules: Specialized agent modules (weather, browser, code_architect, etc.)
- src.harness: Main orchestration engine
"""

__version__ = "0.1.0"
__author__ = "Night-Traders-Dev"

# Re-export common utilities for convenience
from src.common import (
    UIState,
    TokenUsage,
    OLLAMA_URL,
    AGENT_MODEL,
    PLANNER_MODEL,
    CHECKER_MODEL,
    REVIEW_MODEL,
)

# Re-export module runners
from src.modules import (
    run_weather,
    run_browser,
    run_code_architect,
    run_fs_manager,
    run_knowledge,
    run_monitor_agent,
)

__all__ = [
    "__version__",
    "__author__",
    # Common
    "UIState",
    "TokenUsage",
    "OLLAMA_URL",
    "AGENT_MODEL",
    "PLANNER_MODEL",
    "CHECKER_MODEL",
    "REVIEW_MODEL",
    # Module runners
    "run_weather",
    "run_browser",
    "run_code_architect",
    "run_fs_manager",
    "run_knowledge",
    "run_monitor_agent",
]
