import unittest

from src.common.ollama import InferenceClient


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
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls = []
        self.warmed = False

    async def warmup(self):
        self.warmed = True

    async def chat(self, model, messages, stream=True, options=None):
        self.calls.append((model, messages, stream, options))
        if self.fail:
            raise RuntimeError("megakernel failed")
        return {"message": {"content": "megakernel"}}


class InferenceBackendTests(unittest.IsolatedAsyncioTestCase):
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
        self.assertEqual(len(megakernel.calls), 1)
        self.assertEqual(len(ollama.calls), 0)

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
        self.assertEqual(len(megakernel.calls), 1)
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


if __name__ == "__main__":
    unittest.main()
