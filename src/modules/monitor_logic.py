import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, AsyncGenerator

# ══════════════════════════════════════════════════════════════════════
#  Monitor Logic (Log Watcher)
# ══════════════════════════════════════════════════════════════════════

async def tail_file(file_path: Path) -> AsyncGenerator[str, None]:
    """Asynchronous file tailing generator."""
    try:
        with open(file_path, 'r', errors='ignore') as f:
            # Go to end
            f.seek(0, os.SEEK_END)
            while True:
                line = f.readline()
                if not line:
                    await asyncio.sleep(0.5)
                    continue
                yield line.strip()
    except Exception:
        # Stop on fatal errors (e.g. file deletion)
        return

async def classify_log_line(client: Any, model: str, line: str, ui: Any, refresh: Callable) -> Dict[str, Any]:
    """Uses LLM to classify a log line for severity and intent."""
    from src.common.ollama import _chat_json
    
    system = "Classify this log line. Return JSON: { \"severity\": \"info|warn|error\", \"intent\": \"string\", \"summary\": \"string\" }"
    try:
        res, _ = await _chat_json(client, model, system, f"Log: {line}", ui, refresh, "monitor")
        return res
    except Exception:
        return {"severity": "info", "intent": "unknown", "summary": line[:50]}
