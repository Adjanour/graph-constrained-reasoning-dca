#!/bin/bash
# setup-env.sh — Create .venv and install dependencies.
#
# Picks the torch build from the hardware it finds: CUDA wheels when nvidia-smi
# is present (rented GPU), the much smaller CPU wheels otherwise (laptop).
#
# Usage:
#   bash scripts/setup-env.sh          # auto-detect
#   FORCE_CPU=1 bash scripts/setup-env.sh
#   FORCE_GPU=1 bash scripts/setup-env.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Setting up Python environment ==="

# Kill any leftover uv processes
pkill -f "uv " 2>/dev/null || true
sleep 1

# Remove stale venvs (already done, but safety net)
rm -rf .venv-demo .venv-manim

# Create venv
echo "Creating .venv..."
uv venv .venv --python 3.11

# Activate
source .venv/bin/activate

# torch: CUDA build on a GPU box, CPU build (~180MB vs 2.5GB) everywhere else.
#
# NOTE: `uv pip install` ignores [tool.uv.sources] in pyproject.toml, which pins
# torch to the CPU index for local dev. `uv run` does NOT — it re-syncs the
# project and would replace CUDA torch with the CPU build. On a GPU box always
# launch through .venv/bin/python, never `uv run`.
#
# The pin only holds on uv < 0.12: uv 0.12+ honours [tool.uv.sources] even
# through `uv pip install`, silently pulling the CPU wheel again.  `--no-project`
# makes the behaviour version-independent — the wheel is chosen explicitly here,
# not by the project file.
if [ -n "${FORCE_CPU:-}" ]; then
    GPU_BOX=""
elif [ -n "${FORCE_GPU:-}" ] || command -v nvidia-smi &>/dev/null; then
    GPU_BOX=1
else
    GPU_BOX=""
fi

if [ -n "$GPU_BOX" ]; then
    # Pinned below 2.9: prebuilt flash-attn 2 wheels cover cu12torch2.4–2.8 for
    # every cp39–cp313, while the newest torch releases lag the wheel matrix by
    # months (torch 2.10 has cp312 wheels only). Raise this once the matrix
    # catches up — see scripts/install_flash_attn.sh.
    echo "Installing torch (CUDA — nvidia-smi found)..."
    uv pip install --no-project "torch>=2.6,<2.9"
else
    echo "Installing torch (CPU)..."
    uv pip install --no-project torch --index-url https://download.pytorch.org/whl/cpu
fi

# Install everything else
echo "Installing remaining dependencies..."
uv pip install \
    transformers accelerate peft \
    tiktoken openai datasets python-dotenv \
    marisa-trie scikit-learn trl \
    sentencepiece protobuf wandb \
    networkx

# Install dev tools
echo "Installing dev dependencies..."
uv pip install pytest ruff

# Prebuilt flash-attn wheel (~20% faster beam search on sm_80+; never built
# from source — see the script's header).
if [ -n "$GPU_BOX" ]; then
    echo "Installing flash-attn..."
    bash "$SCRIPT_DIR/install_flash_attn.sh" || true
fi

echo ""
echo "=== Done! ==="
python - <<'PYEOF'
import torch

cuda = torch.cuda.is_available()
print(f"torch {torch.__version__}  CUDA available: {cuda}")
if cuda:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No CUDA — CPU-only build. Fine on a laptop; wrong on a rented GPU.")
PYEOF
echo ""
echo "Activate with: source .venv/bin/activate"
echo "Run tests:     pytest"
echo "Lint:          ruff check src/"
