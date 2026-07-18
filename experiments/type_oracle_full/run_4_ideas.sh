#!/usr/bin/env bash
# run_4_ideas.sh — Convenience launcher for experiment_4_ideas.py
#
# Usage:
#   bash experiments/type_oracle_full/run_4_ideas.sh --max-samples 10
#   bash experiments/type_oracle_full/run_4_ideas.sh --max-samples 100
#   bash experiments/type_oracle_full/run_4_ideas.sh --model-path Qwen/Qwen2.5-3B-Instruct --max-samples 10
#
# All arguments forwarded to experiment_4_ideas.py.
# Defaults: 100 samples, all 7 methods, 8B model, WebQSP

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# If not already activated (e.g. inside Docker), try the Vast.ai venv
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f /venv/main/bin/activate ]; then
    source /venv/main/bin/activate
fi

python "$SCRIPT_DIR/experiment_4_ideas.py" \
    --model-path rmanluo/GCR-Meta-Llama-3.1-8B-Instruct \
    --dataset RoG-webqsp \
    --methods baseline,filtered,validate,adaptive30,adaptive100,adaptive500,label-plan \
    --sample-timeout 180 \
    "$@"
