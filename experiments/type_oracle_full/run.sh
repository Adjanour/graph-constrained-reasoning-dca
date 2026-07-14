#!/usr/bin/env bash
# run.sh — one-shot entry point for the DCA-Trie experiment.
#
# Usage:
#   bash experiments/type_oracle_full/run.sh                     # 50 samples, both datasets, all methods
#   bash experiments/type_oracle_full/run.sh --full               # full test set
#   bash experiments/type_oracle_full/run.sh --method v1          # v1 only
#   bash experiments/type_oracle_full/run.sh --datasets RoG-webqsp
#   bash experiments/type_oracle_full/run.sh --max-samples 10
#
# All extra arguments are forwarded to run.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

EXTRA_ARGS=()

# Default to 50 samples unless --full or --max-samples is passed
has_max_samples=false
for arg in "$@"; do
    case "$arg" in
        --max-samples|--full) has_max_samples=true ;;
    esac
done

if [ "$has_max_samples" = false ]; then
    EXTRA_ARGS+=("--max-samples" "50")
fi

# Translate --full into a large --max-samples (main.py doesn't have --full)
for arg in "$@"; do
    if [ "$arg" = "--full" ]; then
        EXTRA_ARGS+=("--max-samples" "999999")
    else
        EXTRA_ARGS+=("$arg")
    fi
done

# Setup
echo "Running setup..."
bash "$SCRIPT_DIR/setup.sh"

# Run
echo ""
echo "Starting experiment..."
python "$SCRIPT_DIR/main.py" "${EXTRA_ARGS[@]}"
