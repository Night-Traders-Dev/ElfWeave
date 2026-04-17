import asyncio
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.common.ollama import InferenceClient, MegaKernelRuntime


class _FakeOllama:
    def __init__(self):
        self.calls = []

    async def list(self):
        return {"models": []}

    async def show(self, model: str):
        return {"model": model}

    async def pull(self, model: str):
        return {"model": model}

    async def chat(self, model, messages, stream=True, options=None):
        self.calls.append((model, messages, stream, options))
        return {"message": {"content": "ollama"}}


class _FakeMegakernel:
    def __init__(self, fail: bool = False, supported: bool = True, reason: str = ""):
        self.fail = fail
        self.supported = supported
        self.reason = reason
        self.calls = []
        self.warmed = False

    async def warmup(self):
        self.warmed = True

    def supports_request(self, model, messages, options=None):
        self.calls.append(("supports_request", model, messages, options))
        return self.supported, self.reason

    async def chat(self, model, messages, stream=True, options=None):
        self.calls.append((model, messages, stream, options))
        if self.fail:
            raise RuntimeError("megakernel failed")
        return {"message": {"content": "megakernel"}}


class InferenceBackendTests(unittest.IsolatedAsyncioTestCase):
    def test_availability_reason_reports_extension_load_failures(self) -> None:
        with TemporaryDirectory() as tmpdir:
            runtime = MegaKernelRuntime(tmpdir, "Qwen/Qwen3.5-0.8B", max_tokens=256)
            Path(tmpdir, "model.py").write_text("# stub\n", encoding="utf-8")

            with patch("src.common.ollama.importlib.util.find_spec", return_value=object()), \
                 patch("src.common.ollama.importlib.import_module") as import_module:
                def _fake_import(name: str):
                    if name == "qwen35_megakernel_bf16_C":
                        raise ImportError("libc10.so: cannot open shared object file")
                    return object()

                import_module.side_effect = _fake_import
                reason = runtime.availability_reason_sync()

        self.assertIn("failed to load", reason)
        self.assertIn("libc10.so", reason)

    def test_runtime_rejects_requests_that_exceed_true_context_budget(self) -> None:
        runtime = MegaKernelRuntime("/tmp/megakernel", "Qwen/Qwen3.5-0.8B", max_tokens=256)
        runtime._prompt_token_count_sync = lambda messages: 1800

        allowed, reason = runtime.supports_request(
            model="qwen3.5:0.8b",
            messages=[{"role": "user", "content": "validate this"}],
            options={},
        )

        self.assertFalse(allowed)
        self.assertIn("context budget", reason)

    def test_runtime_rejects_incompatible_models(self) -> None:
        runtime = MegaKernelRuntime("/tmp/megakernel", "Qwen/Qwen3.5-0.8B", max_tokens=256)
        runtime._prompt_token_count_sync = lambda messages: 32

        allowed, reason = runtime.supports_request(
            model="llama3.1:8b",
            messages=[{"role": "user", "content": "validate this"}],
            options={},
        )

        self.assertFalse(allowed)
        self.assertIn("only supports", reason)

    async def test_runtime_stream_yields_immediate_metrics(self) -> None:
        runtime = MegaKernelRuntime("/tmp/megakernel", "Qwen/Qwen3.5-0.8B", max_tokens=8)

        def _slow_generate(messages, max_tokens):
            time.sleep(0.2)
            return "ready", 42

        runtime._generate_sync = _slow_generate

        stream = await runtime.chat(
            model="qwen3.5:0.8b",
            messages=[{"role": "user", "content": "say ready"}],
            stream=True,
            options={"num_predict": 8},
        )

        first = await asyncio.wait_for(anext(stream), timeout=0.05)
        self.assertEqual(first["prompt_eval_count"], 3)
        self.assertEqual(first["eval_count"], 0)

        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        self.assertEqual(chunks[-1]["message"]["content"], "ready")
        self.assertEqual(chunks[-1]["prompt_eval_count"], 42)
        self.assertEqual(chunks[-1]["eval_count"], 1)

    async def test_runtime_stream_close_is_non_blocking(self) -> None:
        runtime = MegaKernelRuntime("/tmp/megakernel", "Qwen/Qwen3.5-0.8B", max_tokens=8)

        def _slow_generate(messages, max_tokens):
            time.sleep(0.3)
            return "ready", 42

        runtime._generate_sync = _slow_generate

        stream = await runtime.chat(
            model="qwen3.5:0.8b",
            messages=[{"role": "user", "content": "say ready"}],
            stream=True,
            options={"num_predict": 8},
        )
        await asyncio.wait_for(anext(stream), timeout=0.05)

        t0 = time.perf_counter()
        await asyncio.wait_for(stream.aclose(), timeout=0.05)
        self.assertLess(time.perf_counter() - t0, 0.05)

    async def test_phase_routes_to_megakernel(self) -> None:
        ollama = _FakeOllama()
        megakernel = _FakeMegakernel()
        client = InferenceClient(
            ollama_client=ollama,
            megakernel=megakernel,
            megakernel_phases=["planner"],
            fallback_to_ollama=True,
        )

        result = await client.chat_phase(
            "planner",
            model="qwen2.5:7b",
            messages=[{"role": "user", "content": "plan this"}],
            stream=False,
            options={},
        )

        self.assertEqual(result["message"]["content"], "megakernel")
        self.assertEqual(len(megakernel.calls), 2)
        self.assertEqual(len(ollama.calls), 0)
        self.assertEqual(megakernel.calls[0][0], "supports_request")

    async def test_megakernel_falls_back_to_ollama_on_failure(self) -> None:
        ollama = _FakeOllama()
        megakernel = _FakeMegakernel(fail=True)
        client = InferenceClient(
            ollama_client=ollama,
            megakernel=megakernel,
            megakernel_phases=["validator"],
            fallback_to_ollama=True,
        )

        result = await client.chat_phase(
            "validator",
            model="llama3.1:8b",
            messages=[{"role": "user", "content": "validate this"}],
            stream=False,
            options={},
        )

        self.assertEqual(result["message"]["content"], "ollama")
        self.assertEqual(len(megakernel.calls), 2)
        self.assertEqual(len(ollama.calls), 1)

    async def test_image_requests_stay_off_unsupported_megakernel(self) -> None:
        ollama = _FakeOllama()
        megakernel = _FakeMegakernel()
        client = InferenceClient(
            ollama_client=ollama,
            megakernel=megakernel,
            megakernel_phases=["vision"],
            fallback_to_ollama=True,
        )

        result = await client.chat_phase(
            "vision",
            model="llama3.1:8b",
            messages=[{"role": "user", "content": "inspect", "images": ["/tmp/img.png"]}],
            stream=False,
            options={},
        )

        self.assertEqual(result["message"]["content"], "ollama")
        self.assertEqual(len(megakernel.calls), 0)
        self.assertEqual(len(ollama.calls), 1)

    async def test_model_mismatch_stays_on_ollama(self) -> None:
        ollama = _FakeOllama()
        megakernel = _FakeMegakernel(supported=False, reason="wrong model")
        client = InferenceClient(
            ollama_client=ollama,
            megakernel=megakernel,
            megakernel_phases=["validator"],
            fallback_to_ollama=True,
        )

        result = await client.chat_phase(
            "validator",
            model="llama3.1:8b",
            messages=[{"role": "user", "content": "validate this"}],
            stream=False,
            options={},
        )

        self.assertEqual(result["message"]["content"], "ollama")
        self.assertEqual(len(megakernel.calls), 1)
        self.assertEqual(megakernel.calls[0][0], "supports_request")
        self.assertEqual(len(ollama.calls), 1)


if __name__ == "__main__":
    unittest.main()
