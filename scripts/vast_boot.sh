#!/usr/bin/env bash
# vast_boot.sh — Runs automatically when a Vast.ai instance starts.
#
# This script is uploaded to the instance and executed via --onstart-cmd.
# It clones the repo, installs all dependencies, and signals readiness.
#
# Usage (do not run manually — called by run_vast.sh):
#   vastai create instance ... --onstart-cmd 'bash /workspace/vast_boot.sh'

set -euo pipefail

WORKSPACE="/workspace"
REPO_DIR="$WORKSPACE/graph-constrained-reasoning"
REPO_URL="https://github.com/Adjanour/graph-constrained-reasoning-dca.git"
FLAG="$WORKSPACE/setup_done.flag"
LOG="$WORKSPACE/vast_boot.log"

# Remove any stale flag
rm -f "$FLAG"

echo "========================================" | tee "$LOG"
echo "Vast.ai Boot Script — $(date)"           | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

# ── 1. Activate venv ──────────────────────────────────────────────
if [ -f /venv/main/bin/activate ]; then
    source /venv/main/bin/activate
    echo "Activated /venv/main" | tee -a "$LOG"
else
    echo "WARNING: /venv/main not found, using system Python" | tee -a "$LOG"
fi

echo "Python: $(python --version 2>&1)" | tee -a "$LOG"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>&1)" | tee -a "$LOG"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")' 2>&1)" | tee -a "$LOG"

# ── 2. Clone or update repo ───────────────────────────────────────
cd "$WORKSPACE"
if [ -d "$REPO_DIR/.git" ]; then
    echo "Repo exists, pulling latest..." | tee -a "$LOG"
    cd "$REPO_DIR"
    git pull 2>&1 | tee -a "$LOG"
else
    echo "Cloning repo..." | tee -a "$LOG"
    git clone "$REPO_URL" "$REPO_DIR" 2>&1 | tee -a "$LOG"
    cd "$REPO_DIR"
fi

# ── 3. Write .env if HF_TOKEN is set ────────────────────────────────
if [ -n "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN=$HF_TOKEN" > "$REPO_DIR/.env"
    echo "HF_TOKEN written to .env" | tee -a "$LOG"
fi

# ── 4. Keep the 16GB checkpoint off the small root filesystem ──────
export HF_HOME="$WORKSPACE/hf-cache"
mkdir -p "$HF_HOME"
echo "HF_HOME=$HF_HOME" | tee -a "$LOG"

# ── 5. Install dependencies (CUDA torch + prebuilt flash-attn) ─────
echo "Running setup-env.sh..." | tee -a "$LOG"
bash scripts/setup-env.sh 2>&1 | tee -a "$LOG"

# ── 6. Verify the venv actually sees the GPU ───────────────────────
# .venv/bin/python is what run_vast.sh launches; `uv run` would re-sync the
# project and swap in the CPU torch pinned in pyproject.toml.
"$REPO_DIR/.venv/bin/python" - 2>&1 <<'PYEOF' | tee -a "$LOG"
import torch

print(f"venv torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"venv GPU: {torch.cuda.get_device_name(0)}")
else:
    print("ERROR: .venv has no CUDA torch — the experiment would abort at preflight.")
PYEOF

# ── 7. Persist environment for SSH sessions ────────────────────────
echo "export HF_HOME=$HF_HOME" >> /root/.bashrc 2>/dev/null || true
env >> /etc/environment 2>/dev/null || true

# ── 8. Signal completion ───────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "Setup complete — $(date)"               | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
touch "$FLAG"
