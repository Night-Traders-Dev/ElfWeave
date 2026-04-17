"""
ElfWeave Modules Package

This package contains all specialized agent modules, each following the pattern:
- <agent>.py: Main entry point and orchestration
- <agent>_logic.py: Business logic and data processing
- <agent>_ui.py: Terminal UI rendering (where applicable)
"""

from src.modules.browser_agent import run as run_browser
from src.modules.code_architect import run as run_code_architect
from src.modules.fs_manager import run as run_fs_manager
from src.modules.knowledge_agent import main as run_knowledge
from src.modules.monitor_agent import run_monitor as run_monitor_agent
from src.modules.weather import run as run_weather

__all__ = [
    "run_browser",
    "run_code_architect",
    "run_fs_manager",
    "run_knowledge",
    "run_monitor_agent",
    "run_weather",
]
