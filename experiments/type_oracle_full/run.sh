#!/usr/bin/env bash
# run.sh — one-shot entry point for all DCA-Trie experiments.
#
# Usage:
#   bash run.sh                                           # default: main.py, 50 samples, both datasets
#   bash run.sh --experiment main                         # main.py (baseline/v1/v2)
#   bash run.sh --experiment 4ideas                       # experiment_4_ideas.py
#   bash run.sh --experiment adaptive-budget              # experiment_adaptive_budget.py
#   bash run.sh --full                                    # full test set (999999 samples)
#   bash run.sh --method v1                               # v1 only
#   bash run.sh --dataset RoG-webqsp                      # single dataset
#   bash run.sh --dataset RoG-cwq --max-samples 10        # quick test
#
# Tip: Run one dataset at a time to avoid losing progress if interrupted.
#
# All extra arguments are forwarded to the selected experiment script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

EXPERIMENT="main"
EXPERIMENT_ARGS=()

# Parse known flags, collect rest into EXPERIMENT_ARGS
while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment)
            case "$2" in
                main|4ideas|adaptive-budget) EXPERIMENT="$2" ;;
                *) echo "ERROR: Unknown experiment '$2'. Choices: main, 4ideas, adaptive-budget"; exit 1 ;;
            esac
            shift 2
            ;;
        --dataset)
            EXPERIMENT_ARGS+=("--dataset" "$2")
            shift 2
            ;;
        --help|-h)
            head -20 "$0" | grep '^#' | sed 's/^# *//'
            exit 0
            ;;
        *)
            EXPERIMENT_ARGS+=("$1")
            shift
            ;;
    esac
done

# Default to 50 samples unless --max-samples or --full is passed
has_max_samples=false
for arg in "${EXPERIMENT_ARGS[@]}"; do
    case "$arg" in
        --max-samples|--full) has_max_samples=true ;;
    esac
done
if [ "$has_max_samples" = false ]; then
    EXPERIMENT_ARGS+=("--max-samples" "50")
fi
for i in "${!EXPERIMENT_ARGS[@]}"; do
    if [ "${EXPERIMENT_ARGS[$i]}" = "--full" ]; then
        EXPERIMENT_ARGS[$i]="--max-samples"
        EXPERIMENT_ARGS=("${EXPERIMENT_ARGS[@]:0:$((i+1))}" "999999" "${EXPERIMENT_ARGS[@]:$((i+1))}")
        break
    fi
done

Python() {
    python "$SCRIPT_DIR/$1" "${EXPERIMENT_ARGS[@]}"
}

echo "========================================"
echo "  Experiment: $EXPERIMENT"
echo "  Args: ${EXPERIMENT_ARGS[*]}"
echo "========================================"

case "$EXPERIMENT" in
    main)
        Python main.py
        ;;
    4ideas)
        Python experiment_4_ideas.py
        ;;
    adaptive-budget)
        Python experiment_adaptive_budget.py
        ;;
esac
