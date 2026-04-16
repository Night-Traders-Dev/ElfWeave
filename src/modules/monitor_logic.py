import os
import time
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

# ══════════════════════════════════════════════════════════════════════
#  Monitor Logic (Log Watcher)
# ════════════════════════════════════════════════─═════════════════════

class LogTailer:
    """Tails a file and calls a callback for each new line."""
    
    def __init__(self, file_path: Path, callback: Callable[[str], None]):
        self.file_path = file_path
        self.callback = callback
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join()

    def _run(self):
        # Open and go to end
        try:
            with open(self.file_path, 'r', errors='ignore') as f:
                f.seek(0, os.SEEK_END)
                while not self._stop_event.is_set():
                    line = f.readline()
                    if not line:
                        time.sleep(0.1)
                        continue
                    self.callback(line.strip())
        except Exception:
            # Handle error (e.g. file deleted)
            pass

def classify_log_line(client: Any, model: str, line: str, ui: Any, refresh: Callable) -> Dict[str, Any]:
    """Uses LLM to classify a log line for severity and intent."""
    from src.common.ollama import _chat_json
    
    system = "Classify this log line. Return JSON: { \"severity\": \"info|warn|error\", \"intent\": \"string\", \"summary\": \"string\" }"
    try:
        res, _ = _chat_json(client, model, system, f"Log: {line}", ui, refresh, "monitor")
        return res
    except Exception:
        return {"severity": "info", "intent": "unknown", "summary": line[:50]}
