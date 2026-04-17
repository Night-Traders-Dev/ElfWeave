import importlib
import json
import re
import subprocess
import sys
import time
import asyncio
from pathlib import Path
from typing import Any, Tuple, List, Dict, Callable, Optional

from ollama import AsyncClient, ResponseError
from .types import TokenUsage
from .config import (
    INFERENCE_BACKEND,
    OLLAMA_URL,
    MEGAKERNEL_ENABLED,
    MEGAKERNEL_FALLBACK,
    MEGAKERNEL_MAX_TOKENS,
    MEGAKERNEL_MODEL,
    MEGAKERNEL_PHASES,
    MEGAKERNEL_REPO,
    get_ollama_options,
)
from .kernel_bootstrap import DEFAULT_MEGAKERNEL_MODEL


def _ui_call(ui: Optional[Any], method: str, *args: Any, **kwargs: Any) -> Any:
    if ui is None:
        return None
    fn = getattr(ui, method, None)
    if callable(fn):
        return fn(*args, **kwargs)
    return None


def _refresh(refresh: Optional[Callable[[], None]]) -> None:
    if callable(refresh):
        refresh()


def _split_stream_text(text: str, size: int = 64) -> List[str]:
    if not text:
        return [""]
    return [text[i:i + size] for i in range(0, len(text), size)]


class MegaKernelRuntime:
    def __init__(self, repo_path: str, model_name: str, max_tokens: int = 256):
        self.repo_path = Path(repo_path).expanduser()
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.max_context_tokens = 2048
        self._lock = asyncio.Lock()
        self._module = None
        self._decoder = None
        self._tokenizer = None

    @property
    def configured(self) -> bool:
        return bool(self.repo_path) and (self.repo_path / "model.py").exists()

    def availability_reason_sync(self) -> str:
        if not self.configured:
            return f"megakernel repo is missing model.py at {self.repo_path}"
        if not self.supports_model(self.model_name):
            return (
                f"megakernel only supports {DEFAULT_MEGAKERNEL_MODEL}; "
                f"got {self.model_name}"
            )

        repo_str = str(self.repo_path)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        if importlib.util.find_spec("torch") is None:
            return "torch is not installed in the active environment"
        if importlib.util.find_spec("transformers") is None:
            return "transformers is not installed in the active environment"
        if importlib.util.find_spec("qwen35_megakernel_bf16_C") is None:
            return (
                "the qwen35_megakernel_bf16_C extension is not built; "
                "install the submodule with `pip install -e third_party/luce-megakernel`"
            )
        try:
            importlib.import_module("torch")
            importlib.import_module("qwen35_megakernel_bf16_C")
        except Exception as exc:
            return f"megakernel extension failed to load: {exc}"
        return ""

    async def availability_reason(self) -> str:
        return await asyncio.to_thread(self.availability_reason_sync)

    def _load_module_sync(self):
        if self._module is not None:
            return self._module
        model_file = self.repo_path / "model.py"
        if not model_file.exists():
            raise FileNotFoundError(f"Megakernel model.py not found at {model_file}")
        repo_str = str(self.repo_path)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
        spec = importlib.util.spec_from_file_location("elfweave_luce_megakernel_model", model_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load megakernel module from {model_file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.max_context_tokens = int(getattr(module, "MAX_SEQ_LEN", self.max_context_tokens))
        self._module = module
        return module

    @staticmethod
    def _normalize_model_name(model: str) -> str:
        return str(model or "").strip().lower()

    def supports_model(self, requested_model: str) -> bool:
        normalized = self._normalize_model_name(requested_model)
        if not normalized:
            return True
        compatible_names = {
            self._normalize_model_name(self.model_name),
            self._normalize_model_name(DEFAULT_MEGAKERNEL_MODEL),
            self._normalize_model_name("qwen3.5:0.8b"),
            self._normalize_model_name("qwen/qwen3.5-0.8b"),
        }
        return normalized in compatible_names

    def _load_tokenizer_sync(self):
        if self._tokenizer is not None:
            return self._tokenizer
        if self._decoder is not None:
            tokenizer = getattr(self._decoder, "tokenizer", None)
            if tokenizer is not None:
                self._tokenizer = tokenizer
                return tokenizer
        if importlib.util.find_spec("transformers") is None:
            raise RuntimeError("transformers is not installed in the active environment")
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    def _ensure_ready_sync(self):
        if self._decoder is not None:
            return self._decoder
        module = self._load_module_sync()
        try:
            self._decoder = module.Decoder(model_name=self.model_name, verbose=False)
            tokenizer = getattr(self._decoder, "tokenizer", None)
            if tokenizer is not None:
                self._tokenizer = tokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Megakernel dependencies are incomplete. "
                "Install torch with CUDA support, then run "
                "`pip install -e third_party/luce-megakernel` and "
                "`pip install transformers accelerate`."
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Megakernel initialization failed: {exc}") from exc
        return self._decoder

    def supports_request(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        options = options or {}
        if not self.supports_model(model):
            return False, f"megakernel only supports {DEFAULT_MEGAKERNEL_MODEL}"
        max_tokens = int(options.get("num_predict") or options.get("max_tokens") or self.max_tokens)
        prompt_tokens = self._prompt_token_count_sync(messages)
        safety_margin = 64
        if prompt_tokens + max_tokens + safety_margin > self.max_context_tokens:
            return False, "prompt exceeds Luce Megakernel context budget"
        return True, ""

    async def warmup(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._ensure_ready_sync)

    def _messages_to_prompt_sync(self, messages: List[Dict[str, Any]]) -> str:
        tokenizer = self._load_tokenizer_sync()
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        parts = []
        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            parts.append(f"{role}:\n{content}")
        parts.append("ASSISTANT:\n")
        return "\n\n".join(parts)

    def _prompt_token_count_sync(self, messages: List[Dict[str, Any]]) -> int:
        tokenizer = self._load_tokenizer_sync()
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                token_ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
                return len(token_ids)
            except Exception:
                pass

        prompt = self._messages_to_prompt_sync(messages)
        return len(tokenizer.encode(prompt, add_special_tokens=True))

    def _generate_sync(self, messages: List[Dict[str, Any]], max_tokens: int) -> Tuple[str, int]:
        decoder = self._ensure_ready_sync()
        prompt_tokens = self._prompt_token_count_sync(messages)
        if prompt_tokens + max_tokens + 64 > self.max_context_tokens:
            raise ValueError("prompt exceeds Luce Megakernel context budget")
        prompt = self._messages_to_prompt_sync(messages)
        return decoder.generate(prompt, max_tokens=max_tokens), prompt_tokens

    async def chat(self, model: str, messages: List[Dict[str, Any]], stream: bool = True, options: Optional[Dict[str, Any]] = None):
        options = options or {}
        if not self.supports_model(model):
            raise ValueError(
                f"megakernel only supports {DEFAULT_MEGAKERNEL_MODEL}; "
                f"got {model}"
            )
        max_tokens = int(options.get("num_predict") or options.get("max_tokens") or self.max_tokens)
        if not stream:
            async with self._lock:
                started = time.perf_counter()
                text, prompt_tokens = await asyncio.to_thread(self._generate_sync, messages, max_tokens)
                duration_ms = (time.perf_counter() - started) * 1000

            completion_tokens = _est_tokens(text)
            return {
                "message": {"content": text},
                "prompt_eval_count": prompt_tokens,
                "eval_count": completion_tokens,
                "total_duration": int(duration_ms * 1_000_000),
            }

        async def _run_generate() -> Tuple[str, int, float]:
            async with self._lock:
                started = time.perf_counter()
                text, prompt_tokens = await asyncio.to_thread(self._generate_sync, messages, max_tokens)
                duration_ms = (time.perf_counter() - started) * 1000
            return text, prompt_tokens, duration_ms

        est_prompt_tokens = sum(_est_tokens(str(m.get("content", ""))) for m in messages)
        task = asyncio.create_task(_run_generate())
        task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

        async def _gen():
            try:
                yield {
                    "message": {"content": ""},
                    "prompt_eval_count": est_prompt_tokens,
                    "eval_count": 0,
                    "total_duration": 0,
                }

                text = ""
                prompt_tokens = est_prompt_tokens
                duration_ms = 0.0
                while True:
                    if task.done():
                        text, prompt_tokens, duration_ms = await task
                        break
                    await asyncio.sleep(0.1)

                completion_tokens = _est_tokens(text)
                chunks = _split_stream_text(text)
                for idx, piece in enumerate(chunks):
                    payload = {"message": {"content": piece}}
                    if idx == len(chunks) - 1:
                        payload.update(
                            {
                                "prompt_eval_count": prompt_tokens,
                                "eval_count": completion_tokens,
                                "total_duration": int(duration_ms * 1_000_000),
                            }
                        )
                    yield payload
            finally:
                if not task.done():
                    task.cancel()

        return _gen()


class InferenceClient:
    def __init__(
        self,
        ollama_client: Optional[AsyncClient],
        megakernel: Optional[MegaKernelRuntime] = None,
        megakernel_phases: Optional[List[str]] = None,
        fallback_to_ollama: bool = True,
    ) -> None:
        self.ollama_client = ollama_client
        self.megakernel = megakernel
        self.megakernel_phases = {phase.strip().lower() for phase in (megakernel_phases or []) if phase}
        self.fallback_to_ollama = fallback_to_ollama
        self.backend_note = "Ollama ready"

    async def list(self):
        if self.ollama_client is None:
            return {"models": []}
        return await self.ollama_client.list()

    async def show(self, model: str):
        if self.ollama_client is None:
            return {"model": model}
        return await self.ollama_client.show(model)

    async def pull(self, model: str):
        if self.ollama_client is None:
            return None
        return await self.ollama_client.pull(model)

    def _can_use_megakernel(
        self,
        phase: str,
        model: str,
        messages: List[Dict[str, Any]],
        options: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if self.megakernel is None:
            return False
        if phase.strip().lower() not in self.megakernel_phases:
            return False
        for message in messages:
            if message.get("images"):
                return False
            if not isinstance(message.get("content", ""), str):
                return False
        if hasattr(self.megakernel, "supports_request"):
            allowed, _reason = self.megakernel.supports_request(model=model, messages=messages, options=options)
            if not allowed:
                return False
        return True

    async def warmup(self, model: str, phase: str = "") -> None:
        if self.megakernel is not None and phase.strip().lower() in self.megakernel_phases:
            await self.megakernel.warmup()
            return
        if self.ollama_client is None:
            return
        await self.ollama_client.chat(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            options=get_ollama_options(ctx_override=1024),
        )

    async def chat_phase(
        self,
        phase: str,
        model: str,
        messages: List[Dict[str, Any]],
        stream: bool = True,
        options: Optional[Dict[str, Any]] = None,
    ):
        phase = phase.strip().lower()
        if self._can_use_megakernel(phase, model, messages, options=options):
            try:
                return await self.megakernel.chat(model=model, messages=messages, stream=stream, options=options)
            except Exception:
                if not self.fallback_to_ollama or self.ollama_client is None:
                    raise
        if self.ollama_client is None:
            raise RuntimeError("No Ollama backend available for this request.")
        return await self.ollama_client.chat(model=model, messages=messages, stream=stream, options=options)

    async def chat(self, model: str, messages: List[Dict[str, Any]], stream: bool = True, options: Optional[Dict[str, Any]] = None):
        return await self.chat_phase("", model, messages, stream=stream, options=options)

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

async def _warmup(client: Any, model: str, phase: str = "") -> None:
    """Send a trivial request so the model is resident in VRAM before the real call."""
    try:
        if hasattr(client, "warmup"):
            await client.warmup(model, phase=phase)
        else:
            await client.chat(model=model,
                              messages=[{"role": "user", "content": "hi"}],
                              options=get_ollama_options(ctx_override=1024)) # Light warmup
    except Exception:
        pass

def _rough_token_estimate(text: str) -> int:
    """Estimate token count using a consistent heuristic (chars/3)."""
    compact = " ".join(str(text).split())
    return max(1, len(compact) // 3)

# Alias for consistency - use single implementation
_est_tokens = _rough_token_estimate


async def setup_ollama(url: str, models: List[str]) -> InferenceClient:
    import shutil

    ollama_client = AsyncClient(host=url)
    try:
        await ollama_client.list()
    except Exception:
        # Check if ollama binary exists before trying to start it
        if shutil.which("ollama") is None:
            raise RuntimeError(
                "Ollama server is not running and 'ollama' binary not found in PATH. "
                "Please install Ollama from https://ollama.ai or start the server manually."
            )
        proc = subprocess.Popen(["ollama", "serve"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Store reference to prevent garbage collection
        setup_ollama._ollama_proc = proc
        await _wait_ollama(ollama_client)
    for m in models:
        await _ensure_model(ollama_client, m)

    megakernel = None
    backend_note = "Ollama ready"
    if MEGAKERNEL_ENABLED and MEGAKERNEL_REPO:
        candidate = MegaKernelRuntime(MEGAKERNEL_REPO, MEGAKERNEL_MODEL, max_tokens=MEGAKERNEL_MAX_TOKENS)
        if candidate.configured:
            reason = await candidate.availability_reason()
            if reason:
                backend_note = f"{INFERENCE_BACKEND} requested; using Ollama ({reason})"
            else:
                megakernel = candidate
                phases = ",".join(MEGAKERNEL_PHASES)
                backend_note = f"{INFERENCE_BACKEND} ready ({MEGAKERNEL_MODEL}; {phases})"
        else:
            backend_note = f"{INFERENCE_BACKEND} requested; using Ollama (repo not configured)"

    client = InferenceClient(
        ollama_client=ollama_client,
        megakernel=megakernel,
        megakernel_phases=list(MEGAKERNEL_PHASES),
        fallback_to_ollama=MEGAKERNEL_FALLBACK,
    )
    client.backend_note = backend_note
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

# Removed duplicate _est_tokens - now uses the single implementation at line 316-322

async def _stream_chat(
    client: AsyncClient,
    model: str,
    messages: List[Dict[str, Any]],
    ui: Optional[Any],  # UIState
    refresh: Callable[[], None],
    phase: str,
    temperature: float = 0.25,
) -> Tuple[str, TokenUsage]:
    _ui_call(ui, "clear_stream")
    est = TokenUsage(
        prompt_tokens = sum(_est_tokens(
            m.get("content", "") if isinstance(m.get("content"), str)
            else json.dumps(m.get("content", ""))
        ) for m in messages),
        completion_tokens = 0,
        estimated = True,
    )
    _ui_call(ui, "set_usage", phase, est)

    parts: List[str] = []
    final_u = est

    opts = get_ollama_options()
    opts["temperature"] = temperature

    if hasattr(client, "chat_phase"):
        stream_iter = await client.chat_phase(phase, model=model, messages=messages, stream=True, options=opts)
    else:
        stream_iter = await client.chat(model=model, messages=messages, stream=True, options=opts)

    async for chunk in stream_iter:
        piece = _msg_content(chunk)
        if piece:
            parts.append(piece)
            _ui_call(ui, "push_chunk", piece)
            cur = ui.usage.get(phase, est) if ui is not None and hasattr(ui, "usage") else est
            cur.completion_tokens = _est_tokens("".join(parts))
            _ui_call(ui, "set_usage", phase, cur)
        if _get(chunk, "eval_count", None) is not None:
            final_u = _usage_from(chunk)
        _refresh(refresh)

    if final_u.estimated and parts:
        final_u.completion_tokens = _est_tokens("".join(parts))
    _ui_call(ui, "set_usage", phase, final_u)
    _refresh(refresh)
    return "".join(parts).strip(), final_u

async def _chat_json(
    client: AsyncClient,
    model: str,
    system: str,
    user: str,
    ui: Optional[Any],  # UIState
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
        m = re.search(r"(\{.*\})", clean, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1)), u
            except json.JSONDecodeError:
                pass

        # Fallback: handles truncated JSON (missing closing brace)
        if clean.startswith("{") and not clean.endswith("}"):
            try:
                return json.loads(clean + "}"), u
            except json.JSONDecodeError:
                pass

        if attempt < retries:
            current_user_msg += "\n\nCRITICAL: Return ONLY the JSON object. No prose. The previous attempt was unparseable or truncated."

    raise ValueError(f"No valid JSON from {model} after {retries} retries. Final output: {last!r}")
