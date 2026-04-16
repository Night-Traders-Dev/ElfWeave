import json
import re
import subprocess
import time
import asyncio
from typing import Any, Tuple, List, Dict, Callable, Optional

from ollama import AsyncClient, ResponseError
from .types import TokenUsage
from .config import OLLAMA_URL, get_ollama_options

async def _wait_ollama(client: AsyncClient, timeout: int = 30) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            await client.list()
            return
        except Exception:
            await asyncio.sleep(1)
    raise RuntimeError(f"Ollama not reachable after {timeout}s")

async def _ensure_model(client: AsyncClient, model: str) -> None:
    try:
        await client.show(model)
    except ResponseError as exc:
        if getattr(exc, "status_code", None) == 404 or "not found" in str(exc).lower():
            # Note: client.pull(model) is an async generator but we can just await the full pull
            # if we don't need a progress bar for the pull itself.
            await client.pull(model)
        else:
            raise

async def _warmup(client: AsyncClient, model: str) -> None:
    """Send a trivial request so the model is resident in VRAM before the real call."""
    try:
        await client.chat(model=model,
                          messages=[{"role": "user", "content": "hi"}],
                          options=get_ollama_options(ctx_override=1024)) # Light warmup
    except Exception:
        pass

async def setup_ollama(url: str, models: List[str]) -> AsyncClient:
    client = AsyncClient(host=url)
    try:
        await client.list()
    except Exception:
        subprocess.Popen(["ollama", "serve"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        await _wait_ollama(client)
    for m in models:
        await _ensure_model(client, m)
    return client

def _get(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)

def _msg_content(resp: Any) -> str:
    msg = _get(resp, "message", {})
    c = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
    return c or ""

def _usage_from(resp: Any) -> TokenUsage:
    return TokenUsage(
        prompt_tokens     = int(_get(resp, "prompt_eval_count", 0) or 0),
        completion_tokens = int(_get(resp, "eval_count", 0) or 0),
        total_duration_ms = float(_get(resp, "total_duration", 0) or 0) / 1_000_000,
        estimated         = False,
    )

def _est_tokens(text: str) -> int:
    return max(1, len(text.strip()) // 4)

async def _stream_chat(
    client: AsyncClient,
    model: str,
    messages: List[Dict[str, Any]],
    ui: Any, # UIState
    refresh: Callable[[], None],
    phase: str,
    temperature: float = 0.25,
) -> Tuple[str, TokenUsage]:
    ui.clear_stream()
    est = TokenUsage(
        prompt_tokens = sum(_est_tokens(
            m.get("content", "") if isinstance(m.get("content"), str)
            else json.dumps(m.get("content", ""))
        ) for m in messages),
        completion_tokens = 0,
        estimated = True,
    )
    ui.set_usage(phase, est)

    parts: List[str] = []
    final_u = est

    opts = get_ollama_options()
    opts["temperature"] = temperature

    async for chunk in await client.chat(model=model, messages=messages, stream=True, options=opts):
        piece = _msg_content(chunk)
        if piece:
            parts.append(piece)
            ui.push_chunk(piece)
            cur = ui.usage.get(phase, est)
            cur.completion_tokens = _est_tokens("".join(parts))
            ui.set_usage(phase, cur)
        if _get(chunk, "eval_count", None) is not None:
            final_u = _usage_from(chunk)
        refresh()

    if final_u.estimated and parts:
        final_u.completion_tokens = _est_tokens("".join(parts))
    ui.set_usage(phase, final_u)
    refresh()
    return "".join(parts).strip(), final_u

async def _chat_json(
    client: AsyncClient,
    model: str,
    system: str,
    user: str,
    ui: Any, # UIState
    refresh: Callable[[], None],
    phase: str,
    retries: int = 2,
) -> Tuple[dict, TokenUsage]:
    last = ""
    current_user_msg = user
    for attempt in range(retries + 1):
        # We use a internal streaming loop to update the UI
        # even for JSON calls, so the user sees progress.
        messages = [{"role": "system", "content": system},
                    {"role": "user",   "content": current_user_msg}]
        
        # We use our existing _stream_chat logic to get the text and usage
        raw_text, u = await _stream_chat(client, model, messages, ui, refresh, phase)
        
        last = raw_text
        # Extract JSON block
        clean = re.sub(r"```(?:json)?|```", "", raw_text.strip()).strip()
        m = re.search(r"\{.*\}", clean, re.DOTALL)
        if m:
            try:
                return json.loads(m.group()), u
            except json.JSONDecodeError:
                pass
        
        if attempt < retries:
            current_user_msg += "\n\nCRITICAL: Return ONLY the JSON object. No prose. The previous attempt was unparseable."
            
    raise ValueError(f"No valid JSON from {model} after {retries} retries. Final output: {last!r}")
