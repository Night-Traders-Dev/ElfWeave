import unittest
from pathlib import Path

from src.common.kernel_bootstrap import (
    DEFAULT_MEGAKERNEL_MODEL,
    DEFAULT_MEGAKERNEL_PHASES,
    apply_kernel_env,
    argv_requests_kernel,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class KernelBootstrapTests(unittest.TestCase):
    def test_flag_detection(self) -> None:
        self.assertTrue(argv_requests_kernel(["main.py", "--use-kernel", "weather"]))
        self.assertFalse(argv_requests_kernel(["main.py", "weather"]))

    def test_apply_kernel_env_sets_expected_defaults(self) -> None:
        env: dict[str, str] = {}
        applied = apply_kernel_env(True, environ=env, repo_root=REPO_ROOT)

        self.assertTrue(applied)
        self.assertEqual(env["ELFWEAVE_INFERENCE_BACKEND"], "hybrid")
        self.assertEqual(env["ELFWEAVE_MEGAKERNEL_MODEL"], DEFAULT_MEGAKERNEL_MODEL)
        self.assertEqual(env["ELFWEAVE_MEGAKERNEL_PHASES"], DEFAULT_MEGAKERNEL_PHASES)
        self.assertEqual(
            Path(env["ELFWEAVE_MEGAKERNEL_REPO"]).resolve(),
            (REPO_ROOT / "third_party" / "luce-megakernel").resolve(),
        )

    def test_existing_megakernel_overrides_are_preserved(self) -> None:
        env = {
            "ELFWEAVE_MEGAKERNEL_REPO": "/tmp/custom-kernel",
            "ELFWEAVE_MEGAKERNEL_MODEL": "custom/model",
            "ELFWEAVE_MEGAKERNEL_PHASES": "planner",
        }
        apply_kernel_env(True, environ=env, repo_root=REPO_ROOT)

        self.assertEqual(env["ELFWEAVE_INFERENCE_BACKEND"], "hybrid")
        self.assertEqual(env["ELFWEAVE_MEGAKERNEL_REPO"], "/tmp/custom-kernel")
        self.assertEqual(env["ELFWEAVE_MEGAKERNEL_MODEL"], "custom/model")
        self.assertEqual(env["ELFWEAVE_MEGAKERNEL_PHASES"], "planner")
