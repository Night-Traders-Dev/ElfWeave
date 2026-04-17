from __future__ import annotations

import os
from pathlib import Path
from typing import MutableMapping, Sequence


USE_KERNEL_FLAG = "--use-kernel"
DEFAULT_MEGAKERNEL_MODEL = "Qwen/Qwen3.5-0.8B"
# Only use megakernel for sanity (checker) and validator (review) phases
# Planner and other phases continue using Ollama with their respective models
DEFAULT_MEGAKERNEL_PHASES = "sanity,validator"


def elfweave_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def bundled_megakernel_repo(repo_root: Path | None = None) -> Path:
    root = repo_root or elfweave_repo_root()
    return root / "third_party" / "luce-megakernel"


def argv_requests_kernel(argv: Sequence[str]) -> bool:
    return USE_KERNEL_FLAG in argv


def apply_kernel_env(
    use_kernel: bool,
    environ: MutableMapping[str, str] | None = None,
    repo_root: Path | None = None,
) -> bool:
    if not use_kernel:
        return False

    env = os.environ if environ is None else environ
    env.setdefault("ELFWEAVE_INFERENCE_BACKEND", "hybrid")
    env.setdefault("ELFWEAVE_MEGAKERNEL_ENABLE", "1")

    repo_path = bundled_megakernel_repo(repo_root)
    if repo_path.exists():
        env.setdefault("ELFWEAVE_MEGAKERNEL_REPO", str(repo_path))

    env.setdefault("ELFWEAVE_MEGAKERNEL_MODEL", DEFAULT_MEGAKERNEL_MODEL)
    env.setdefault("ELFWEAVE_MEGAKERNEL_PHASES", DEFAULT_MEGAKERNEL_PHASES)
    return True
