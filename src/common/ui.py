import shutil
import threading
import time
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional

from rich import box
from rich.cells import cell_len
from rich.console import Group, RenderableType, Console
from rich.text import Text
from rich.panel import Panel
from rich.rule import Rule
from rich.live import Live

SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def console_width(console: Optional[Console] = None, minimum: int = 40) -> int:
    if console is not None:
        try:
            return max(minimum, int(console.size.width))
        except Exception:
            pass
    return max(minimum, shutil.get_terminal_size((100, 30)).columns)


def is_compact_width(width: int) -> bool:
    return width < 110


def is_tiny_width(width: int) -> bool:
    return width < 78


def clip_text(text: Any, limit: int) -> str:
    value = " ".join(str(text).split())
    if limit <= 0:
        return ""
    if len(value) <= limit:
        return value
    if limit <= 3:
        return value[:limit]
    return value[: limit - 1] + "…"


def panel_padding_for_width(width: int, padding: tuple[int, int] = (1, 2)) -> tuple[int, int]:
    if is_tiny_width(width):
        return (0, 1)
    if is_compact_width(width):
        return (padding[0], 1)
    return padding


def panel_box_for_width(width: int):
    return box.SIMPLE if is_compact_width(width) else box.ROUNDED

@dataclass
class Step:
    """One pipeline step rendered as a status line."""
    name:       str
    state:      str   = "pending"   # pending | running | done | error | skipped
    detail:     str   = ""
    elapsed_ms: float = 0.0
    cached:     bool  = False
    _t0: float        = field(default_factory=time.monotonic, repr=False)

    def start(self) -> "Step":
        self.state = "running"
        self._t0   = time.monotonic()
        return self

    def done(self, detail: str = "", cached: bool = False) -> "Step":
        self.elapsed_ms = (time.monotonic() - self._t0) * 1000
        self.state  = "done"
        self.detail = detail
        self.cached = cached
        return self

    def error(self, detail: str = "") -> "Step":
        self.elapsed_ms = (time.monotonic() - self._t0) * 1000
        self.state  = "error"
        self.detail = detail
        return self

    def skip(self, detail: str = "") -> "Step":
        self.state  = "skipped"
        self.detail = detail
        return self

    def render(self, sp_frame: int, width: int, label_width: int = 24) -> Text:
        width = max(40, width)
        label_width = max(14, min(label_width, max(14, width // 3)))
        t = Text()
        t.append("  ")
        if self.state == "running":
            # Synchronized with UIState frame timing
            t.append(SPINNER_FRAMES[sp_frame], "bold cyan")
        elif self.state == "done":
            t.append("✓", "bold green")
        elif self.state == "error":
            t.append("✗", "bold red")
        elif self.state == "skipped":
            t.append("–", "dim")
        else:
            t.append("·", "dim")

        col = "white" if self.state != "pending" else "grey50"
        label = clip_text(f"  {self.name}", label_width)
        t.append(f"{label:<{label_width}}", col)

        if self.detail:
            detail_budget = max(8, width - label_width - 12)
            preview = clip_text(self.detail, detail_budget)
            if preview:
                t.append(" ")
                t.append(preview, "dim")

        if self.elapsed_ms:
            elapsed = self.elapsed_ms
            ts = f"{elapsed:.0f}ms" if elapsed < 2000 else f"{elapsed / 1000:.1f}s"
            padding = max(1, width - cell_len(t.plain) - cell_len(ts) - 2)
            t.append(" " * padding)
            t.append(ts, "dim")
            if self.cached:
                t.append("  ⚡", "yellow")
        elif self.state == "running":
            t.append(f"  {(time.monotonic()-self._t0):.1f}s", "dim")

        return t


class UIState:
    """
    Standardized UI orchestrator for ElfWeave agents.
    Provides an async context manager for Live display management.
    """

    def __init__(self, agent_name: str, model_info: str = "", max_stream_lines: int = 8, refresh_hz: int = 10) -> None:
        self._lock         = threading.RLock()
        self.agent_name    = agent_name
        self.model_info    = model_info
        self.max_stream_lines = max_stream_lines
        self.refresh_hz    = refresh_hz
        self.steps:   List[Step]              = []
        self.stream_chunks: List[str]         = []
        self.usage:   Dict[str, Any]          = {}  # TokenUsage
        self.running: bool                    = True
        
        self.console = Console()
        self.live: Optional[Live] = None
        self.harness_mode: bool = False

    async def __aenter__(self) -> "UIState":
        """Start the Live display automatically unless in harness mode."""
        if not self.harness_mode:
            self.live = Live(self.render(), refresh_per_second=self.refresh_hz, console=self.console, screen=False)
            self.live.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop the Live display and cleanup."""
        self.running = False
        self.refresh()
        if self.live:
            self.live.stop()

    def refresh(self) -> None:
        """Update the TUI render."""
        if self.live:
            self.live.update(self.render())

    def add_step(self, name: str) -> Step:
        s = Step(name=name)
        with self._lock:
            self.steps.append(s)
        return s

    def push_chunk(self, piece: str) -> None:
        with self._lock:
            full = "".join(self.stream_chunks) + piece
            self.stream_chunks = full.split("\n")[-self.max_stream_lines:]

    def set_usage(self, phase: str, u: Any) -> None:
        with self._lock:
            self.usage[phase] = u

    def clear_stream(self) -> None:
        with self._lock:
            self.stream_chunks = []

    def render(self) -> RenderableType:
        with self._lock:
            # Absolute frame timing synchronized across all render calls
            frame   = int(time.monotonic() * 10) % len(SPINNER_FRAMES)
            steps   = list(self.steps)
            chunks  = list(self.stream_chunks)
            usage   = dict(self.usage)
            running = self.running

        width = console_width(self.console)
        parts: List[Any] = []

        # ── header ────────────────────────────────────────────────────
        hdr = Text()
        hdr.append("◆", "bold cyan")
        name_budget = max(14, width // 3)
        hdr.append(f" {clip_text(self.agent_name, name_budget)}", "bold white")
        if self.model_info:
            model_budget = max(10, width - cell_len(hdr.plain) - 6)
            hdr.append(f"   {clip_text(self.model_info, model_budget)}", "dim")
        dot = "●" if running else "◉"
        hdr.append(f"   {dot}", "green" if running else "dim")
        parts.append(hdr)
        parts.append(Text(""))

        # ── steps ─────────────────────────────────────────────────────
        label_width = min(26 if "harness" in self.agent_name else 24, max(16, width // 4))
        for s in steps:
            parts.append(s.render(frame, width=width, label_width=label_width))

        # ── streaming output ──────────────────────────────────────────
        if any(c.strip() for c in chunks):
            parts.append(Text(""))
            parts.append(Rule(style="dim"))
            for i, line in enumerate(chunks):
                row = Text("  ")
                row.append(line, "white")
                if i == len(chunks) - 1 and running:
                    row.append("▌", "blink bold cyan")
                parts.append(row)
            parts.append(Rule(style="dim"))

        # ── footer ────────────────────────────────────────────────────
        parts.append(Text(""))
        total_tok = sum(u.prompt_tokens + u.completion_tokens for u in usage.values() if hasattr(u, 'prompt_tokens'))
        gen_tok   = sum(u.completion_tokens for u in usage.values() if hasattr(u, 'completion_tokens'))
        foot = Text("  ")
        foot.append("esc", "bold dim")
        foot.append(" to interrupt", "dim")
        if total_tok:
            if is_tiny_width(width):
                foot.append(f"  ·  {total_tok:,} tok", "dim")
            elif is_compact_width(width):
                foot.append(f"  ·  p {total_tok - gen_tok:,}  g {gen_tok:,}  t {total_tok:,}", "dim")
            else:
                foot.append(f"  ·  prompt {total_tok - gen_tok:,}  gen {gen_tok:,}  total {total_tok:,}", "dim")
        active = next((s for s in steps if s.state == "running"), None)
        if active:
            foot.append(f"  ·  {time.monotonic() - active._t0:.1f}s", "dim")
        parts.append(foot)

        return Group(*parts)

    def print_card(self, title: str, content: Any, border_color: str = "blue", metadata: Optional[str] = None, padding: tuple = (1, 2)) -> None:
        """Print a standard premium result card in the Claude Code / Weather UI style."""
        width = console_width(self.console)
        self.console.print()
        self.console.print(Rule(f"[bold {border_color}]{title}[/bold {border_color}]", style=border_color))
        
        self.console.print(
            Panel(
                content,
                border_style=border_color,
                box=panel_box_for_width(width),
                padding=panel_padding_for_width(width, padding),
                expand=True,
                subtitle=f"[dim]{clip_text(metadata, max(20, width - 12))}[/dim]" if metadata else None
            )
        )
        self.console.print(Rule(style=border_color))
        self.console.print()
