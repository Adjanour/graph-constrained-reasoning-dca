#!/usr/bin/env bash
# run_vast.sh — Fully orchestrate a Vast.ai run from your local machine.
#
# Searches for a GPU, rents it, uploads boot script, installs dependencies,
# runs the experiment, downloads results, and destroys the instance.
#
# Prerequisites:
#   pip install vastai
#   vastai set api-key YOUR_API_KEY
#   (SSH key added to your Vast.ai account)
#
# Usage:
#   bash scripts/run_vast.sh                              # full run
#   bash scripts/run_vast.sh --max-samples 50             # quick test
#   bash scripts/run_vast.sh --datasets RoG-webqsp        # one dataset
#   bash scripts/run_vast.sh --datasets RoG-cwq           # other dataset
#   bash scripts/run_vast.sh --method v2                  # one method
#   bash scripts/run_vast.sh --offer 44169006             # specific offer
#   bash scripts/run_vast.sh --gpu A100_40GB              # different GPU
#   bash scripts/run_vast.sh --region us                   # US hosts only
#   bash scripts/run_vast.sh --region eu                   # EU hosts only
#   bash scripts/run_vast.sh --run-name thesis             # named, resumable run dir
#   bash scripts/run_vast.sh --max-hours 6                 # stop before spending more
#
# Split-dataset strategy (recommended to avoid losing progress) — both runs
# share one results dir, so a re-rented box picks up where the last one stopped:
#   bash scripts/run_vast.sh --run-name thesis --datasets RoG-webqsp
#   bash scripts/run_vast.sh --run-name thesis --datasets RoG-cwq
#
# The offer's real $/hr is passed to main.py --cost-per-hour automatically.
# All extra arguments are forwarded to experiments/type_oracle_full/main.py.

set -euo pipefail

# ─── Configuration ─────────────────────────────────────────────────
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

# ─── Parse arguments ───────────────────────────────────────────────
OFFER_ID=""
GPU_FILTER="RTX_4090"
REGION=""
EXPERIMENT="all"   # all | baseline | v1 | v2 | v2-nogates | ablation
RUN_NAME="vast"    # stable results dir — a re-rented box resumes into it
MAX_HOURS=""       # optional spend guard, forwarded to --max-runtime-hours
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --offer)   OFFER_ID="$2";  shift 2 ;;
        --gpu)     GPU_FILTER="$2"; shift 2 ;;
        --image)   DOCKER_IMAGE="$2"; shift 2 ;;
        --disk)    DISK_SIZE="$2"; shift 2 ;;
        --region)  REGION="$2"; shift 2 ;;
        --experiment) EXPERIMENT="$2"; shift 2 ;;
        --run-name) RUN_NAME="$2"; shift 2 ;;
        --max-hours) MAX_HOURS="$2"; shift 2 ;;
        --search-only) SEARCH_ONLY=1; shift ;;
        --help|-h) head -32 "$0" | grep '^#' | sed 's/^# *//' ; exit 0 ;;
        *)         EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# An explicit --output-dir in the forwarded args wins over --run-name.
if [[ " ${EXTRA_ARGS[*]:-} " == *" --output-dir "* ]]; then
    RUN_NAME=""
fi

echo "========================================"
echo "  Vast.ai DCA-Trie Orchestrator"
echo "========================================"
echo "GPU: $GPU_FILTER  Disk: ${DISK_SIZE}GB  Region: ${REGION:-any}"
echo "Docker: $DOCKER_IMAGE"
echo "Experiment: $EXPERIMENT"
echo "Run name: ${RUN_NAME:-(from --output-dir)}"
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

# vastai: try uv-available first, then global (mise shims can be broken)
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
    echo "ERROR: 'vastai' not found or not working."
    echo "  → cd $(pwd) && uv pip install vastai"
    echo "  → uv run vastai set api-key YOUR_API_KEY"
    exit 1
fi

# ─── 1. Search for an offer ────────────────────────────────────────
if [ -z "$OFFER_ID" ]; then
    echo ""
    echo "→ Searching for $GPU_FILTER offers..."

    # Fetch offers sorted by price
    CANDIDATES=$($VASTAI search offers \
        "gpu_name=$GPU_FILTER num_gpus=1 disk_space>=$DISK_SIZE reliability>=0.$MIN_RELIABILITY" \
        --order dph --raw 2>/dev/null)

    if [ -z "$CANDIDATES" ] || [ "$CANDIDATES" = "[]" ]; then
        echo "ERROR: No offers found matching filters."
        echo "  Try: $VASTAI search offers \"gpu_name=$GPU_FILTER num_gpus=1\" --order dph"
        exit 1
    fi

    N_CANDIDATES=$(echo "$CANDIDATES" | jq 'length')
    echo "  Found $N_CANDIDATES offers."

    # Debug: show geolocation format
    if [ -n "$REGION" ]; then
        echo "  Sample locations: $(echo "$CANDIDATES" | jq -r '.[0:3][] | .geolocation // .geoloc // .country // "unknown"' 2>/dev/null)"
    fi

    if [ -n "$REGION" ]; then
        # Filter by region
        case "$REGION" in
            us|US)
                OFFER_ID=$(echo "$CANDIDATES" | jq -r '[.[] | select(.geolocation | test(", US$"))] | .[0].id // empty')
                ;;
            eu|EU)
                OFFER_ID=$(echo "$CANDIDATES" | jq -r '[.[] | select(.geolocation | test(", (DE|FR|NL|SE|GB|IT|ES|PL|RO|BG|HU|DK|AT|CZ|FI|NO|BE|IE|PT|CH|HR|SK|SI|LT|LV|EE|LU|MT|CY)$"))] | .[0].id // empty')
                ;;
            ca|CA)
                OFFER_ID=$(echo "$CANDIDATES" | jq -r '[.[] | select(.geolocation | test(", CA$"))] | .[0].id // empty')
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
        echo "ERROR: No offers found. Try: vastai search offers \"gpu_name=$GPU_FILTER\" --order dph"
        exit 1
    fi

    # Show the selected offer details
    OFFER_DETAILS=$(echo "$CANDIDATES" | jq -r ".[] | select(.id == $OFFER_ID) | \"  ID: \\(.id)  Price: \\(.dph_total)/hr  Location: \\(.geolocation)  Reliability: \\(.reliability)\"")
    echo "$OFFER_DETAILS"
    # Real price of this offer — fed to main.py --cost-per-hour so the run logs
    # what it is actually spending.
    DPH=$(echo "$CANDIDATES" | jq -r ".[] | select(.id == $OFFER_ID) | .dph_total // empty")
    if [ "${SEARCH_ONLY:-0}" = "1" ]; then
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
MAX_WAIT=120  # 30 min — first-time image pulls can be slow
while true; do
    STATUS_JSON=$($VASTAI show instance "$INSTANCE_ID" --raw 2>/dev/null)
    STATUS=$(echo "$STATUS_JSON" | jq -r '.actual_status // "unknown"')
    case "$STATUS" in
        running) echo "  Instance is running."; break ;;
        loading)
            # Show download progress if available
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
        echo "Or check dashboard: https://cloud.vast.ai"
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
sleep 5  # brief wait for SSH daemon

scp $SSH_OPTS -P "$SSH_PORT" \
    "$SCRIPT_DIR/vast_boot.sh" \
    "root@$SSH_HOST:/workspace/vast_boot.sh"

echo "→ Running boot script (clone + dependencies)..."
ssh $SSH_OPTS -p "$SSH_PORT" "root@$SSH_HOST" \
    "nohup bash /workspace/vast_boot.sh > /workspace/vast_boot.log 2>&1 &"

echo "→ Waiting for setup to finish (polling every ${POLL_BOOT}s)..."
BOOT_COUNT=0
BOOT_MAX=120  # 30 min max
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

# Price of this instance, so the run can log what it is spending. Set from the
# offer during search; recovered from the instance record when --offer was used.
if [ -z "${DPH:-}" ]; then
    DPH=$(echo "$CONN_INFO" | jq -r '.dph_total // empty')
fi

RUN_ARGS=(--method "$EXPERIMENT")
[ -n "$RUN_NAME" ]  && RUN_ARGS+=(--run-name "$RUN_NAME")
[ -n "$DPH" ]       && RUN_ARGS+=(--cost-per-hour "$DPH")
[ -n "$MAX_HOURS" ] && RUN_ARGS+=(--max-runtime-hours "$MAX_HOURS")
RUN_ARGS+=(${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})

echo "→ Starting experiment: main.py ${RUN_ARGS[*]}"

# .venv/bin/python, NOT `uv run`: pyproject pins torch to the CPU index for
# local dev, so `uv run` would re-sync the project and swap the CUDA build for
# a CPU one on this GPU box. vast_boot.sh builds .venv with CUDA torch.
# HF_HOME is set explicitly: a non-interactive ssh command does not read
# ~/.bashrc, and the 16 GB checkpoint must not land on the small root fs.
ssh $SSH_OPTS -p "$SSH_PORT" "root@$SSH_HOST" \
    "cd /workspace/graph-constrained-reasoning && \
     export HF_HOME=/workspace/hf-cache && \
     nohup ./.venv/bin/python experiments/type_oracle_full/main.py ${RUN_ARGS[*]} \
         > /workspace/experiment.log 2>&1 &"

echo "→ Experiment running. Monitoring every ${POLL_EXPERIMENT}s..."
echo "   Manual: ssh -p $SSH_PORT root@$SSH_HOST 'tail -f /workspace/experiment.log'"
echo ""

EXP_COUNT=0
while true; do
    if ssh $SSH_OPTS -p "$SSH_PORT" "root@$SSH_HOST" \
        'grep -q "Results saved to" /workspace/experiment.log' 2>/dev/null; then
        echo "  Experiment complete!"
        break
    fi

    # Died without finishing (preflight abort, OOM, CUDA error)? Say so instead
    # of polling a dead box at $/hr.
    if ! ssh $SSH_OPTS -p "$SSH_PORT" "root@$SSH_HOST" \
        'pgrep -f "experiments/type_oracle_full/main.py" >/dev/null' 2>/dev/null; then
        echo ""
        echo "ERROR: the experiment process is gone but never finished. Last 20 log lines:"
        ssh $SSH_OPTS -p "$SSH_PORT" "root@$SSH_HOST" \
            'tail -20 /workspace/experiment.log' 2>/dev/null || true
        echo ""
        echo "Instance $INSTANCE_ID is still up — fix and rerun, or destroy it:"
        echo "  ssh -p $SSH_PORT root@$SSH_HOST"
        echo "  $VASTAI destroy instance $INSTANCE_ID"
        exit 1
    fi

    # status.json is the structured heartbeat main.py keeps updated.
    PROGRESS=$(ssh $SSH_OPTS -p "$SSH_PORT" "root@$SSH_HOST" \
        'cat /workspace/graph-constrained-reasoning/results/final_experiment/*/status.json 2>/dev/null | tail -20' 2>/dev/null |
        jq -r '"\(.completed_units)/\(.total_units) conditions | elapsed \(.elapsed_h)h | ETA \(.eta_h)h | $\(.est_cost_usd // 0)"' 2>/dev/null || true)
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

SUMMARY=$(find "$RESULTS_DIR" -name "summary.json" -print -quit 2>/dev/null)
if [ -n "$SUMMARY" ]; then
    echo ""
    echo "Summary:"
    cat "$SUMMARY" | jq '.' 2>/dev/null || cat "$SUMMARY"
fi

echo ""
echo "Instance $INSTANCE_ID is still running."
echo "  SSH:     ssh -p $SSH_PORT root@$SSH_HOST"
echo "  Destroy: vastai destroy instance $INSTANCE_ID"

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