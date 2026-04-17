#!/bin/bash
# ElfWeave Megakernel Setup Script
# This script installs and configures the Luce Megakernel for Qwen3.5-0.8B

set -e

echo "═══════════════════════════════════════════════════════════"
echo "  ElfWeave Megakernel Setup"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Check for CUDA
if ! python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "❌ ERROR: CUDA is not available or torch was installed without CUDA support"
    echo ""
    echo "Please ensure:"
    echo "  1. You have an NVIDIA GPU with CUDA support"
    echo "  2. NVIDIA drivers are installed"
    echo "  3. PyTorch with CUDA is installed:"
    echo "     pip install torch --index-url https://download.pytorch.org/whl/cu126"
    echo ""
    exit 1
fi

echo "✓ CUDA detected"
python3 -c "import torch; print(f'  PyTorch {torch.__version__} with CUDA {torch.version.cuda}')"
python3 -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}')"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -q transformers accelerate

# Build megakernel extension
echo ""
echo "Building megakernel CUDA extension..."
cd "$(dirname "$0")/../third_party/luce-megakernel"
pip install -e .

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
echo "  python3 src/harness.py --use-kernel \"your task here\""
echo ""
echo "  # Or set environment variable:"
echo "  export ELFWEAVE_INFERENCE_BACKEND=hybrid"
echo "  python3 src/harness.py \"your task here\""
echo ""
echo "Model configuration:"
echo "  - CHECKER_MODEL:  qwen3.5:0.8b (megakernel for sanity checks)"
echo "  - REVIEW_MODEL:   qwen3.5:0.8b (megakernel for validation)"
echo "  - PLANNER_MODEL:  qwen3.5:4b  (Ollama for planning)"
echo "  - AGENT_MODEL:    qwen3.5:2b  (Ollama for agent tasks)"
echo ""
