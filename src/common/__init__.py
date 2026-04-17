"""
ElfWeave Common Package

Shared utilities and configuration for all ElfWeave agents:
- config.py: Global hardware settings, model selection, and prompts
- ollama.py: Async LLM client helpers and Megakernel integration
- types.py: Shared data models (TokenUsage, etc.)
- ui.py: Synchronized terminal UI manager
- kernel_bootstrap.py: Megakernel environment setup
"""

from src.common.config import (
    OLLAMA_URL,
    AGENT_MODEL,
    PLANNER_MODEL,
    CHECKER_MODEL,
    REVIEW_MODEL,
    DEFAULT_MODEL,
    get_ollama_options,
)
from src.common.types import TokenUsage
from src.common.ui import (
    UIState,
    Step,
    console_width,
    clip_text,
    is_compact_width,
    is_tiny_width,
    panel_box_for_width,
    panel_padding_for_width,
)

__all__ = [
    # Config
    "OLLAMA_URL",
    "AGENT_MODEL",
    "PLANNER_MODEL",
    "CHECKER_MODEL",
    "REVIEW_MODEL",
    "DEFAULT_MODEL",
    "get_ollama_options",
    # Types
    "TokenUsage",
    # UI
    "UIState",
    "Step",
    "console_width",
    "clip_text",
    "is_compact_width",
    "is_tiny_width",
    "panel_box_for_width",
    "panel_padding_for_width",
]
