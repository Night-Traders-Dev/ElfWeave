import json
import re
import subprocess
import time
from typing import Any, Tuple, List, Dict, Callable, Optional

from ollama import Client, ResponseError
from .types import TokenUsage
from .config import OLLAMA_URL, get_ollama_options

def _wait_ollama(client: Client, timeout: int = 30) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            client.list()
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"Ollama not reachable after {timeout}s")

def _ensure_model(client: Client, model: str) -> None:
    try:
        client.show(model)
    except ResponseError as exc:
        if getattr(exc, "status_code", None) == 404 or "not found" in str(exc).lower():
            # Note: client.pull(model) is blocking and doesn't provide easy streaming progress here
            # without more complex wrapping. For now, we keep it simple.
            client.pull(model)
        else:
            raise

def _warmup(client: Client, model: str) -> None:
    """Send a trivial request so the model is resident in VRAM before the real call."""
    try:
        client.chat(model=model,
                    messages=[{"role": "user", "content": "hi"}],
                    options=get_ollama_options(ctx_override=1024)) # Light warmup
    except Exception:
        pass

def setup_ollama(url: str, models: List[str]) -> Client:
    client = Client(host=url)
    try:
        client.list()
    except Exception:
        subprocess.Popen(["ollama", "serve"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _wait_ollama(client)
    for m in models:
        _ensure_model(client, m)
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

def _stream_chat(
    client: Client,
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

    for chunk in client.chat(model=model, messages=messages, stream=True, options=opts):
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

def _chat_json(
    client: Client,
    model: str,
    system: str,
    user: str,
    ui: Any, # UIState
    refresh: Callable[[], None],
    phase: str,
    retries: int = 2,
) -> Tuple[dict, TokenUsage]:
    last = ""
    u = TokenUsage()
    for attempt in range(retries + 1):
        resp = client.chat(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
            options=get_ollama_options(),
        )
        u = _usage_from(resp)
        ui.set_usage(phase, u); refresh()
        raw = re.sub(r"```(?:json)?|```", "", _msg_content(resp).strip()).strip()
        last = raw
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group()), u
            except json.JSONDecodeError:
                pass
        if attempt < retries:
            user += "\n\nReturn ONLY the JSON object. No prose."
    raise ValueError(f"No valid JSON from {model}: {last!r}")
