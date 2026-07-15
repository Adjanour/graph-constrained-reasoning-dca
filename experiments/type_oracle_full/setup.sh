#!/usr/bin/env bash
# setup.sh — install all dependencies for the TypeOracle experiment.
# Run once after cloning the repo, or whenever the environment is recreated.
#
# Usage:
#   bash experiments/type_oracle_full/setup.sh
#
# flash-attn: pre-built wheels only, no source compilation.
# If no matching wheel is found, falls back to sdpa (built into transformers).

set -euo pipefail

echo "========================================"
echo "TypeOracle Experiment — Environment Setup"
echo "========================================"

# Core dependencies
pip install -q \
    "transformers==4.44.2" \
    "accelerate>=0.30.1" \
    "datasets>=2.19.2" \
    "marisa-trie>=1.2.0" \
    "networkx>=3.0" \
    "scikit-learn>=1.5.0" \
    "tiktoken>=0.7.0" \
    "sentencepiece>=0.2.0" \
    "protobuf>=5.27.1" \
    "openai>=1.31" \
    "python-dotenv>=1.0" \
    "peft>=0.11" \
    "tqdm"

echo "Core dependencies installed."

# Catch any remaining project deps (openai, dotenv, peft, etc.)
pip install -q -e . 2>/dev/null || true

# Re-pin transformers — trl wants >=4.56 but our code needs 4.44.x
pip install -q "transformers==4.44.2"

# ── Flash-Attn (pre-built wheels only) ─────────────────────────────────
# sdpa works on all GPUs with no install. flash_attention_2 is ~20% faster
# on A100 for beam search, but not required.

flash_attn_installed=false

if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    py_ver=$(python3 -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
    cuda_ver=$(python3 -c "import torch; print(torch.version.cuda.replace('.',''))")
    torch_ver=$(python3 -c "import torch; print(torch.__version__.split('+')[0])")
    torch_mm=$(echo "$torch_ver" | cut -d. -f1,2)

    echo "Detected: Python $py_ver, CUDA $cuda_ver, PyTorch $torch_ver"

    # Try wheel patterns in order: cxx11abiFALSE first (most common), then TRUE
    # Also try torch major.minor.patch variants
    for abi in FALSE TRUE; do
        for torch_tag in "$torch_mm" "$torch_ver"; do
            wheel_url="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu${cuda_ver}torch${torch_tag}cxx11abi${abi}-cp${py_ver}-cp${py_ver}-linux_x86_64.whl"
            echo "Trying: $wheel_url"
            if pip install -q "$wheel_url" 2>/dev/null; then
                flash_attn_installed=true
                echo "flash-attn installed from pre-compiled wheel."
                break 2
            fi
        done
    done

    if [ "$flash_attn_installed" = false ]; then
        echo "No matching pre-built wheel found for this Python/CUDA/PyTorch combo."
        echo "Falling back to sdpa (built into transformers, no install needed)."
    fi
else
    echo "No CUDA detected — skipping flash-attn (sdpa will be used)."
fi

if [ "$flash_attn_installed" = true ]; then
    echo "→ flash_attention_2 available."
else
    echo "→ Using sdpa (built into transformers, no install needed)."
fi

echo "========================================"
echo "Setup complete."
echo "========================================"
