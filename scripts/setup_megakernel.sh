#!/bin/bash
# ElfWeave Megakernel Setup Script
# This script installs and configures the Luce Megakernel for Qwen3.5-0.8B

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
else
    echo "ERROR: python3 or python is required"
    exit 1
fi

echo "═══════════════════════════════════════════════════════════"
echo "  ElfWeave Megakernel Setup"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Check for CUDA
if ! "${PYTHON_BIN}" -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "❌ ERROR: CUDA is not available or torch was installed without CUDA support"
    echo ""
    echo "Please ensure:"
    echo "  1. You have an NVIDIA GPU with CUDA support"
    echo "  2. NVIDIA drivers are installed"
    echo "  3. PyTorch with CUDA is installed:"
    echo "     ${PYTHON_BIN} -m pip install torch --index-url https://download.pytorch.org/whl/cu132"
    echo ""
    exit 1
fi

echo "✓ CUDA detected"
"${PYTHON_BIN}" -c "import torch; print(f'  PyTorch {torch.__version__} with CUDA {torch.version.cuda}')"
"${PYTHON_BIN}" -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}')"
echo ""

if ! "${PYTHON_BIN}" -m pip --version >/dev/null 2>&1; then
    echo "Bootstrapping pip for ${PYTHON_BIN}..."
    "${PYTHON_BIN}" -m ensurepip --upgrade >/dev/null
fi

# Install dependencies
echo "Installing dependencies..."
"${PYTHON_BIN}" -m pip install -q -e "${REPO_ROOT}[megakernel]"

# Build megakernel extension
echo ""
echo "Building megakernel CUDA extension..."
"${PYTHON_BIN}" -m pip install --no-build-isolation -e "${REPO_ROOT}/third_party/luce-megakernel"

echo ""
echo "Caching Qwen/Qwen3.5-0.8B weights..."
"${PYTHON_BIN}" - <<'PY'
import os
from huggingface_hub import snapshot_download

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

snapshot_download(
    repo_id="Qwen/Qwen3.5-0.8B",
    allow_patterns=[
        "config.json",
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "special_tokens_map.json",
        "generation_config.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "model.safetensors-*.safetensors",
    ],
)
PY

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ✓ Megakernel setup complete!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  - Model: Qwen/Qwen3.5-0.8B (built-in)"
echo "  - Phases: sanity, validator"
echo "  - Fallback: Ollama (qwen3.5:0.8b)"
echo ""
echo "Usage:"
echo "  # Run with megakernel enabled:"
echo "  ${PYTHON_BIN} src/harness.py --use-kernel \"your task here\""
echo ""
echo "  # Or set environment variable:"
echo "  export ELFWEAVE_INFERENCE_BACKEND=hybrid"
echo "  ${PYTHON_BIN} src/harness.py \"your task here\""
echo ""
echo "Model configuration:"
echo "  - CHECKER_MODEL:  qwen3.5:0.8b (megakernel for sanity checks)"
echo "  - REVIEW_MODEL:   qwen3.5:0.8b (megakernel for validation)"
echo "  - PLANNER_MODEL:  qwen3.5:4b  (Ollama for planning)"
echo "  - AGENT_MODEL:    qwen3.5:2b  (Ollama for agent tasks)"
echo ""
