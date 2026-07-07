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

# Forward all args
EXTRA_ARGS+=("$@")

# Setup
echo "Running setup..."
bash "$SCRIPT_DIR/setup.sh"

# Run
echo ""
echo "Starting experiment..."
python "$SCRIPT_DIR/run.py" "${EXTRA_ARGS[@]}"
