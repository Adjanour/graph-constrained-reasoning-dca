#!/usr/bin/env bash
# run.sh — one-shot entry point for the TypeOracle experiment.
#
# Usage:
#   bash experiments/type_oracle_full/run.sh                     # quick test (100 samples)
#   bash experiments/type_oracle_full/run.sh --full               # full test set
#   bash experiments/type_oracle_full/run.sh --flash-attn         # install flash-attn from source
#   bash experiments/type_oracle_full/run.sh --output-dir /tmp/r  # custom output dir
#
# All extra arguments are forwarded to run.py after the script's own flags are consumed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── Parse script-level flags ───────────────────────────────────────────
INSTALL_FLASH=false
EXTRA_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --flash-attn) INSTALL_FLASH=true ;;
        *)            EXTRA_ARGS+=("$arg") ;;
    esac
done

# Default to 100 samples unless --full or --max-samples is passed
has_max_samples=false
for arg in "${EXTRA_ARGS[@]}"; do
    case "$arg" in
        --max-samples) has_max_samples=true ;;
    esac
done

if [ "$has_max_samples" = false ]; then
    EXTRA_ARGS+=("--max-samples" "100")
fi

# ── Setup ──────────────────────────────────────────────────────────────
echo "Running setup..."
bash "$SCRIPT_DIR/setup.sh" $([ "$INSTALL_FLASH" = true ] && echo "--flash-attn")

# ── Run experiment ─────────────────────────────────────────────────────
echo ""
echo "Starting experiment..."
python "$SCRIPT_DIR/run.py" "${EXTRA_ARGS[@]}"
