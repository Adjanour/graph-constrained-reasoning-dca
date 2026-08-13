#!/usr/bin/env bash
# run_ablation.sh — Run ablation studies for the v2 metrics paper.
#
# Usage:
#   bash experiments/type_oracle_full/run_ablation.sh
#   bash experiments/type_oracle_full/run_ablation.sh --max-samples 100
#   bash experiments/type_oracle_full/run_ablation.sh --datasets RoG-webqsp
#   bash experiments/type_oracle_full/run_ablation.sh --run-name abl1   # resumable
#   bash experiments/type_oracle_full/run_ablation.sh --resume          # continue latest
#   bash experiments/type_oracle_full/run_ablation.sh --cost-per-hour 0.35 --max-hours 6
#
# Any unrecognised flag is forwarded straight to main.py.
#
# What it does:
#   1. Runs v2 with gates ON (normal), collects metrics
#   2. Runs v2 with gates OFF (DoG-proxy), collects metrics
#   3. Runs baseline + v1 for comparison
#   4. All results saved to the same output directory
#
# Interrupted runs resume: rerun the same command with --output-dir/--run-name
# (or --resume) and finished questions are skipped.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Defaults
MAX_SAMPLES=300
DATASETS="RoG-webqsp"
OUTPUT_DIR=""
RUN_NAME=""
SEED=42
RESUME=""
COST_PER_HOUR="${VAST_COST_PER_HOUR:-}"
MAX_HOURS=""
EXTRA_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-samples)   MAX_SAMPLES="$2"; shift 2 ;;
        --datasets)      DATASETS="$2"; shift 2 ;;
        --output-dir)    OUTPUT_DIR="$2"; shift 2 ;;
        --run-name)      RUN_NAME="$2"; shift 2 ;;
        --seed)          SEED="$2"; shift 2 ;;
        --cost-per-hour) COST_PER_HOUR="$2"; shift 2 ;;
        --max-hours)     MAX_HOURS="$2"; shift 2 ;;
        --resume)        RESUME=1; shift ;;
        --help|-h)       head -21 "$0" | grep '^#' | sed 's/^# *//'; exit 0 ;;
        *)               EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# Where results go: --output-dir > --run-name > --resume > fresh timestamp.
# --run-name and --resume are handled by main.py itself.
DEST_ARGS=()
if [[ -n "$OUTPUT_DIR" ]]; then
    DEST_ARGS=(--output-dir "$OUTPUT_DIR")
    DEST_LABEL="$OUTPUT_DIR"
elif [[ -n "$RUN_NAME" ]]; then
    DEST_ARGS=(--run-name "$RUN_NAME")
    DEST_LABEL="results/final_experiment/$RUN_NAME"
elif [[ -n "$RESUME" ]]; then
    DEST_ARGS=(--resume)
    DEST_LABEL="(most recent run)"
else
    TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
    OUTPUT_DIR="$PROJECT_ROOT/results/ablation_${TIMESTAMP}"
    DEST_ARGS=(--output-dir "$OUTPUT_DIR")
    DEST_LABEL="$OUTPUT_DIR"
fi

BUDGET_ARGS=()
[[ -n "$COST_PER_HOUR" ]] && BUDGET_ARGS+=(--cost-per-hour "$COST_PER_HOUR")
[[ -n "$MAX_HOURS" ]]     && BUDGET_ARGS+=(--max-runtime-hours "$MAX_HOURS")

# .venv/bin/python, not `uv run`: pyproject pins torch to the CPU index for
# local dev, and `uv run` re-syncs the project — which would replace CUDA torch
# with a CPU build on a rented GPU.
if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    PY="$PROJECT_ROOT/.venv/bin/python"
else
    PY="python"
fi

echo "=============================================="
echo "  DCA-Trie Ablation Study"
echo "=============================================="
echo "  Datasets:    $DATASETS"
echo "  Max samples: $MAX_SAMPLES"
echo "  Seed:        $SEED"
echo "  Output:      $DEST_LABEL"
echo "  Python:      $PY"
[[ -n "$COST_PER_HOUR" ]] && echo "  Cost/hr:     \$$COST_PER_HOUR"
[[ -n "$MAX_HOURS" ]]     && echo "  Budget:      ${MAX_HOURS}h"
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && echo "  Extra args:  ${EXTRA_ARGS[*]}"
echo "=============================================="
echo ""

# Run all conditions (baseline, v1, v2, v2-nogates) with metrics collection
"$PY" "$SCRIPT_DIR/main.py" \
    --method ablation \
    --datasets $DATASETS \
    --max-samples "$MAX_SAMPLES" \
    --seed "$SEED" \
    --collect-metrics \
    --beam-size 5 \
    "${DEST_ARGS[@]}" \
    ${BUDGET_ARGS[@]+"${BUDGET_ARGS[@]}"} \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo ""
echo "=============================================="
echo "  Ablation complete!"
echo "  Results: $DEST_LABEL"
echo "=============================================="
echo ""
echo "Files produced:"
echo "  config.json              — configuration + provenance (git commit, versions, GPU)"
echo "  summary.json             — aggregate metrics per condition"
echo "  status.json              — final state, conditions run, elapsed, est. cost"
echo "  run.log                  — full DEBUG log"
echo "  <dataset>/predictions_*.jsonl — per-question predictions + metrics"
echo ""
echo "To analyze metrics, load the JSONL files and examine the"
echo "'ablation_metrics' field in each record."
