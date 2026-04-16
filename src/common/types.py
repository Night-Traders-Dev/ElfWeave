from dataclasses import dataclass
from typing import Optional

@dataclass
class TokenUsage:
    prompt_tokens:     int   = 0
    completion_tokens: int   = 0
    total_duration_ms: float = 0.0
    estimated:         bool  = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens
