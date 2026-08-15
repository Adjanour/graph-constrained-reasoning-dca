#!/usr/bin/env bash
# run_premise_test.sh — Rent a GPU on Vast.ai and run the E1/E2 premise test
# across one or more models.
#
# The premise test (experiments/type_oracle_full/main.py --method premise)
# compares, on the SAME questions, four trie filters over the LLM's beam:
#
#   GCR_Baseline    — unfiltered trie (control; true rank-1 accuracy)
#   E1_GoldRelevant — keep only paths whose terminal is a gold answer entity.
#                     This is the *perfect relevance filter*: the oracle upper
#                     bound for "filtering irrelevant paths helps accuracy".
#   E2_GoldTypes    — keep only paths whose terminal passes a gate over
#                     *gold-derived* answer types (a correct ontology oracle).
#                     Tests whether the type lever helps when types are right.
#
# Usage:
#   bash scripts/run_premise_test.sh                                   # default model
#   bash scripts/run_premise_test.sh --models "gcr llama-3.1-8b"       # by alias (below)
#   bash scripts/run_premise_test.sh --models "rmanluo/GCR-Meta-Llama-3.1-8B-Instruct"
#   bash scripts/run_premise_test.sh --models "gcr llama" --samples 300 --run-name premise-full
#   bash scripts/run_premise_test.sh --offer 44169006                  # specific GPU
#   bash scripts/run_premise_test.sh --gpu A100_40GB
#   bash scripts/run_premise_test.sh --region eu
#   bash scripts/run_premise_test.sh --max-hours 6
#   bash scripts/run_premise_test.sh --search-only                     # just list offers
#   BRANCH=premise-test bash scripts/run_premise_test.sh                # run this branch's code
#
# Model aliases (resolved to HF checkpoints):
#   gcr    -> rmanluo/GCR-Meta-Llama-3.1-8B-Instruct   (the fine-tuned GCR model)
#   llama  -> meta-llama/Llama-3.1-8B-Instruct         (base Llama 3.1 8B)
#   mistral-> mistralai/Mistral-7B-Instruct-v0.3
#   qwen   -> Qwen/Qwen2.5-7B-Instruct
#
# Results land in results_from_vast/<run-name>/ as one subdirectory per model.
# NOTE: any plain (non-GCR) checkpoint must be instruction-tuned and able to
# follow the path-generation prompt; GCR_Token is only guaranteed for `gcr`.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCKER_IMAGE="vastai/pytorch:2.6.0-cuda-12.6.3-py312"
DISK_SIZE=200
MIN_RELIABILITY=95
HF_TOKEN="${HF_TOKEN:-}"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ServerAliveInterval=30"
POLL_BOOT=15
POLL_EXPERIMENT=60
RESULTS_DIR="$PROJECT_ROOT/results_from_vast"

# Model aliases
declare -A MODELS=(
  [gcr]="rmanluo/GCR-Meta-Llama-3.1-8B-Instruct"
  [llama]="meta-llama/Llama-3.1-8B-Instruct"
  [mistral]="mistralai/Mistral-7B-Instruct-v0.3"
  [qwen]="Qwen/Qwen2.5-7B-Instruct"
)

# ─── Parse arguments ───────────────────────────────────────────────
OFFER_ID=""
GPU_FILTER="RTX_4090"
REGION=""
MODEL_ARGS="gcr"
SAMPLES=300
RUN_NAME="premise-test"
MAX_HOURS=""
SEARCH_ONLY=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --offer)   OFFER_ID="$2"; shift 2 ;;
        --gpu)     GPU_FILTER="$2"; shift 2 ;;
        --region)  REGION="$2"; shift 2 ;;
        --models)  MODEL_ARGS="$2"; shift 2 ;;
        --samples) SAMPLES="$2"; shift 2 ;;
        --run-name) RUN_NAME="$2"; shift 2 ;;
        --max-hours) MAX_HOURS="$2"; shift 2 ;;
        --search-only) SEARCH_ONLY=1; shift ;;
        --help|-h) sed -n '2,34p' "$0" | sed 's/^# *//'; exit 0 ;;
        *)         EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# Resolve aliases → HF checkpoints
MODEL_PATHS=()
for m in $MODEL_ARGS; do
    if [[ -n "${MODELS[$m]:-}" ]]; then
        MODEL_PATHS+=("${MODELS[$m]}")
    else
        MODEL_PATHS+=("$m")
    fi
done

echo "========================================"
echo "  Vast.ai E1/E2 Premise Test"
echo "========================================"
echo "GPU: $GPU_FILTER  Disk: ${DISK_SIZE}GB  Region: ${REGION:-any}"
echo "Docker: $DOCKER_IMAGE"
echo "Models:"
for mp in "${MODEL_PATHS[@]}"; do
    echo "  - $mp"
done
echo "Samples per dataset: $SAMPLES  Run name: $RUN_NAME"
echo "Results: $RESULTS_DIR"
echo "Args: ${EXTRA_ARGS[*]:-none}"
echo "========================================"

# ─── Dependency checks ─────────────────────────────────────────────
for cmd in ssh scp jq; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: '$cmd' not found."
        [[ "$cmd" == "jq" ]] && echo "  → brew install jq  OR  apt install jq"
        exit 1
    fi
done

cd "$PROJECT_ROOT"
VASTAI=""
if command -v uv &>/dev/null; then
    if uv run vastai search offers "gpu_name=RTX_4090 num_gpus=1" --order dph --raw &>/dev/null 2>&1; then
        VASTAI="uv run vastai"
    elif uv run python -m vastai search offers "gpu_name=RTX_4090 num_gpus=1" --order dph --raw &>/dev/null 2>&1; then
        VASTAI="uv run python -m vastai"
    fi
fi
if [ -z "$VASTAI" ] && command -v vastai &>/dev/null; then
    VASTAI="vastai"
fi
if [ -z "$VASTAI" ]; then
    echo "ERROR: 'vastai' not found. → cd $(pwd) && uv pip install vastai && uv run vastai set api-key"
    exit 1
fi

# ─── 1. Search for an offer ────────────────────────────────────────
if [ -z "$OFFER_ID" ]; then
    echo ""
    echo "→ Searching for $GPU_FILTER offers..."
    CANDIDATES=$($VASTAI search offers \
        "gpu_name=$GPU_FILTER num_gpus=1 disk_space>=$DISK_SIZE reliability>=0.$MIN_RELIABILITY" \
        --order dph --raw 2>/dev/null)
    if [ -z "$CANDIDATES" ] || [ "$CANDIDATES" = "[]" ]; then
        echo "ERROR: No offers found. Try: $VASTAI search offers \"gpu_name=$GPU_FILTER num_gpus=1\" --order dph"
        exit 1
    fi
    N_CANDIDATES=$(echo "$CANDIDATES" | jq 'length')
    echo "  Found $N_CANDIDATES offers."

    if [ -n "$REGION" ]; then
        case "$REGION" in
            us|US)
                OFFER_ID=$(echo "$CANDIDATES" | jq -r '[.[] | select(.geolocation | test(", US$"))] | .[0].id // empty')
                ;;
            eu|EU)
                OFFER_ID=$(echo "$CANDIDATES" | jq -r '[.[] | select(.geolocation | test(", (DE|FR|NL|SE|GB|IT|ES|PL|RO|BG|HU|DK|AT|CZ|FI|NO|BE|IE|PT|CH|HR|SK|SI|LT|LV|EE|LU|MT|CY)$"))] | .[0].id // empty')
                ;;
            *)
                echo "WARNING: Unknown region '$REGION', ignoring filter."
                OFFER_ID=$(echo "$CANDIDATES" | jq -r '.[0].id // empty')
                ;;
        esac
    else
        OFFER_ID=$(echo "$CANDIDATES" | jq -r '.[0].id // empty')
    fi

    if [ -z "$OFFER_ID" ]; then
        echo "ERROR: No offers in region. Try: vastai search offers \"gpu_name=$GPU_FILTER\" --order dph"
        exit 1
    fi

    OFFER_DETAILS=$(echo "$CANDIDATES" | jq -r ".[] | select(.id == $OFFER_ID) | \"  ID: \\(.id)  Price: \\(.dph_total)/hr  Location: \\(.geolocation)  Reliability: \\(.reliability)\"")
    echo "$OFFER_DETAILS"
    DPH=$(echo "$CANDIDATES" | jq -r ".[] | select(.id == $OFFER_ID) | .dph_total // empty")
    if [ "$SEARCH_ONLY" = "1" ]; then
        echo ""
        echo "Search-only mode. Skipping rent (--search-only)"
        exit 0
    fi
else
    echo ""
    echo "→ Using offer: $OFFER_ID"
fi

# ─── 2. Rent the instance ─────────────────────────────────────────
echo ""
echo "→ Renting instance..."
RENTAL_OUTPUT=$($VASTAI create instance "$OFFER_ID" \
    --image "$DOCKER_IMAGE" \
    --disk "$DISK_SIZE" \
    --ssh --direct --raw \
    ${HF_TOKEN:+--env "-e HF_TOKEN=$HF_TOKEN"} \
    2>/dev/null)

INSTANCE_ID=$(echo "$RENTAL_OUTPUT" | jq -r '.new_contract // .new_contract_id // .id // empty')
if [ -z "$INSTANCE_ID" ]; then
    echo "ERROR: Failed to rent. Output: $RENTAL_OUTPUT"
    exit 1
fi
echo "  Instance ID: $INSTANCE_ID"

# ─── 3. Wait for instance to be running ────────────────────────────
echo ""
echo "→ Waiting for instance to start (polling every ${POLL_BOOT}s)..."
WAIT_COUNT=0
MAX_WAIT=120
while true; do
    STATUS_JSON=$($VASTAI show instance "$INSTANCE_ID" --raw 2>/dev/null)
    STATUS=$(echo "$STATUS_JSON" | jq -r '.actual_status // "unknown"')
    case "$STATUS" in
        running) echo "  Instance is running."; break ;;
        loading)
            DL_STATUS=$(echo "$STATUS_JSON" | jq -r '.status_msg // ""' 2>/dev/null)
            if [ -n "$DL_STATUS" ] && [ "$DL_STATUS" != "null" ]; then
                echo "  Loading: $DL_STATUS"
            else
                echo "  Loading image... (${WAIT_COUNT}x15s elapsed)"
            fi
            ;;
        *)  echo "  Status: $STATUS" ;;
    esac
    WAIT_COUNT=$((WAIT_COUNT + 1))
    if [ "$WAIT_COUNT" -ge "$MAX_WAIT" ]; then
        echo "ERROR: Timed out (30 min). Check: vastai show instance $INSTANCE_ID"
        exit 1
    fi
    sleep "$POLL_BOOT"
done

# ─── 4. Get SSH details ───────────────────────────────────────────
echo ""
echo "→ Getting SSH details..."
CONN_INFO=$($VASTAI show instance "$INSTANCE_ID" --raw 2>/dev/null)
SSH_HOST=$(echo "$CONN_INFO" | jq -r '.ssh_host // empty')
SSH_PORT=$(echo "$CONN_INFO" | jq -r '.ssh_port // .ports["22/tcp"][0].HostPort // empty')
if [ -z "$SSH_HOST" ] || [ -z "$SSH_PORT" ]; then
    echo "ERROR: Could not determine SSH details. Try: vastai show instance $INSTANCE_ID"
    exit 1
fi
echo "  ssh -p $SSH_PORT root@$SSH_HOST"

# ─── 5. Upload boot script and run it ─────────────────────────────
echo ""
echo "→ Uploading boot script..."
sleep 5

scp $SSH_OPTS -P "$SSH_PORT" \
    "$SCRIPT_DIR/vast_boot.sh" \
    "root@$SSH_HOST:/workspace/vast_boot.sh"

echo "→ Running boot script (clone + dependencies)..."
ssh $SSH_OPTS -p "$SSH_PORT" "root@$SSH_HOST" \
    "nohup bash /workspace/vast_boot.sh > /workspace/vast_boot.log 2>&1 &"

echo "→ Waiting for setup to finish (polling every ${POLL_BOOT}s)..."
BOOT_COUNT=0
BOOT_MAX=120
while true; do
    if ssh $SSH_OPTS -p "$SSH_PORT" "root@$SSH_HOST" \
        'test -f /workspace/setup_done.flag' 2>/dev/null; then
        echo "  Setup complete."
        break
    fi
    BOOT_COUNT=$((BOOT_COUNT + 1))
    if [ "$BOOT_COUNT" -ge "$BOOT_MAX" ]; then
        echo "ERROR: Setup timed out (30 min)."
        echo "Logs: ssh -p $SSH_PORT root@$SSH_HOST 'cat /workspace/vast_boot.log'"
        exit 1
    fi
    sleep "$POLL_BOOT"
done

# ─── 6. Run the experiment ─────────────────────────────────────────
echo ""
if [ -z "${DPH:-}" ]; then
    DPH=$(echo "$CONN_INFO" | jq -r '.dph_total // empty')
fi

# NOTE: the premise-test code (E1/E2 conditions + multi-model loop) lives on a
# branch.  Check it out on the box after boot so the run uses this code.
CHECKOUT=""
if [ -n "${BRANCH:-}" ]; then
    CHECKOUT="git fetch origin && git checkout ${BRANCH} &&"
fi
RUN_ARGS=(--method premise --max-samples "$SAMPLES" --run-name "$RUN_NAME")
[ -n "$DPH" ]       && RUN_ARGS+=(--cost-per-hour "$DPH")
[ -n "$MAX_HOURS" ] && RUN_ARGS+=(--max-runtime-hours "$MAX_HOURS")
RUN_ARGS+=(--model-path ${MODEL_PATHS[@]})
RUN_ARGS+=(${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})

echo "→ Starting experiment on the box..."
echo "  main.py ${RUN_ARGS[*]}"

ssh $SSH_OPTS -p "$SSH_PORT" "root@$SSH_HOST" \
    "cd /workspace/graph-constrained-reasoning && \
     export HF_HOME=/workspace/hf-cache && \
     ${CHECKOUT} \
     nohup ./.venv/bin/python experiments/type_oracle_full/main.py ${RUN_ARGS[*]} \
         > /workspace/experiment.log 2>&1 &"

echo "→ Experiment running. Monitoring every ${POLL_EXPERIMENT}s..."
echo "   Manual: ssh -p $SSH_PORT root@$SSH_HOST 'tail -f /workspace/experiment.log'"
echo ""

EXP_COUNT=0
while true; do
    if ssh $SSH_OPTS -p "$SSH_PORT" "root@$SSH_HOST" \
        'grep -q "ALL MODELS DONE" /workspace/experiment.log' 2>/dev/null; then
        echo "  Experiment complete!"
        break
    fi

    if ! ssh $SSH_OPTS -p "$SSH_PORT" "root@$SSH_HOST" \
        'pgrep -f "experiments/type_oracle_full/main.py" >/dev/null' 2>/dev/null; then
        echo ""
        echo "ERROR: the experiment process is gone but never finished. Last 20 log lines:"
        ssh $SSH_OPTS -p "$SSH_PORT" "root@$SSH_HOST" \
            'tail -20 /workspace/experiment.log' 2>/dev/null || true
        echo ""
        echo "Instance $INSTANCE_ID is still up — fix and rerun, or destroy it:"
        echo "  $VASTAI destroy instance $INSTANCE_ID"
        exit 1
    fi

    PROGRESS=$(ssh $SSH_OPTS -p "$SSH_PORT" "root@$SSH_HOST" \
        'cat /workspace/graph-constrained-reasoning/results/final_experiment/*/status.json 2>/dev/null | tail -20' 2>/dev/null |
        jq -r '"\(.completed_units)/\(.total_units) conditions | elapsed \(.elapsed_h)h | model \(.model // "?")"' 2>/dev/null || true)
    if [ -n "$PROGRESS" ] && [ "$PROGRESS" != "null" ]; then
        echo "  [$EXP_COUNT] $PROGRESS"
    else
        LAST_LINE=$(ssh $SSH_OPTS -p "$SSH_PORT" "root@$SSH_HOST" \
            'tail -1 /workspace/experiment.log 2>/dev/null' || echo "...")
        echo "  [$EXP_COUNT] $LAST_LINE"
    fi
    EXP_COUNT=$((EXP_COUNT + 1))
    sleep "$POLL_EXPERIMENT"
done

# ─── 7. Download results ───────────────────────────────────────────
echo ""
echo "→ Downloading results to $RESULTS_DIR ..."
mkdir -p "$RESULTS_DIR"
scp $SSH_OPTS -P "$SSH_PORT" -r \
    "root@$SSH_HOST:/workspace/graph-constrained-reasoning/results/" \
    "$RESULTS_DIR/"

scp $SSH_OPTS -P "$SSH_PORT" \
    "root@$SSH_HOST:/workspace/experiment.log" \
    "$RESULTS_DIR/" 2>/dev/null || true

echo "  Results saved to: $RESULTS_DIR"

# ─── 8. Summary ────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  DONE"
echo "========================================"
echo "Results: $RESULTS_DIR"
echo ""
echo "Per-model summaries under $RESULTS_DIR/<run-name>/<model-slug>/summary.json"
echo "  e.g. $(find "$RESULTS_DIR" -name "summary.json" -path "*$RUN_NAME*" -print -quit 2>/dev/null)"
echo ""
echo "To compare the four filters (Baseline vs E1 vs E2) across models, read each"
echo "model's summary.json — the E1 vs Baseline gap is the relevance-filter headroom;"
echo "the E2 vs Baseline gap is the value of correct types."

# ─── 9. Clean up ───────────────────────────────────────────────────
echo ""
read -p "Destroy instance now? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    $VASTAI destroy instance "$INSTANCE_ID"
    echo "Instance destroyed. Billing stopped."
else
    echo "Remember to destroy when done: vastai destroy instance $INSTANCE_ID"
fi