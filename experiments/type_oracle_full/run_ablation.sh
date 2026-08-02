#!/usr/bin/env bash
# run_ablation.sh — Run ablation studies for the v2 metrics paper.
#
# Usage:
#   bash experiments/type_oracle_full/run_ablation.sh
#   bash experiments/type_oracle_full/run_ablation.sh --max-samples 100
#   bash experiments/type_oracle_full/run_ablation.sh --datasets RoG-webqsp
#
# What it does:
#   1. Runs v2 with gates ON (normal), collects metrics
#   2. Runs v2 with gates OFF (DoG-proxy), collects metrics
#   3. Runs baseline + v1 for comparison
#   4. All results saved to the same output directory

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Defaults
MAX_SAMPLES=300
DATASETS="RoG-webqsp"
METHOD="ablation"
OUTPUT_DIR=""
SEED=42

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
        --datasets) DATASETS="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Output directory
if [[ -z "$OUTPUT_DIR" ]]; then
    TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
    OUTPUT_DIR="$PROJECT_ROOT/results/ablation_${TIMESTAMP}"
fi

echo "=============================================="
echo "  DCA-Trie Ablation Study"
echo "=============================================="
echo "  Datasets:    $DATASETS"
echo "  Max samples: $MAX_SAMPLES"
echo "  Seed:        $SEED"
echo "  Output:      $OUTPUT_DIR"
echo "=============================================="
echo ""

# Run all conditions (baseline, v1, v2, v2-nogates) with metrics collection
uv run python "$SCRIPT_DIR/main.py" \
    --method ablation \
    --datasets $DATASETS \
    --max-samples "$MAX_SAMPLES" \
    --output-dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    --collect-metrics \
    --beam-size 5

echo ""
echo "=============================================="
echo "  Ablation complete!"
echo "  Results: $OUTPUT_DIR"
echo "=============================================="
echo ""
echo "Files produced:"
echo "  config.json              — experiment configuration"
echo "  summary.json             — aggregate metrics per condition"
echo "  <dataset>/predictions_*.jsonl — per-question predictions + metrics"
echo ""
echo "To analyze metrics, load the JSONL files and examine the"
echo "'ablation_metrics' field in each record."
