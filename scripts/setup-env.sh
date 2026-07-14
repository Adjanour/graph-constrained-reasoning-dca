#!/bin/bash
# Run this when system is idle (close some tabs/apps first)
# Usage: bash scripts/setup-env.sh

set -e

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

# Install CPU-only torch first (fast, ~180MB vs 500MB+ for CUDA)
echo "Installing torch (CPU)..."
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install everything else
echo "Installing remaining dependencies..."
uv pip install \
    transformers accelerate peft \
    tiktoken openai datasets python-dotenv \
    marisa-trie scikit-learn trl \
    sentencepiece protobuf wandb

# Install dev tools
echo "Installing dev dependencies..."
uv pip install pytest ruff

echo ""
echo "=== Done! ==="
echo "Activate with: source .venv/bin/activate"
echo "Run tests:    pytest"
echo "Lint:         ruff check src/"
