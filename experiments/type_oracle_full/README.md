# DCA-Trie Full Experiment

End-to-end experiment comparing GCR baseline vs DCA-Trie v1 (static) vs
DCA-Trie v2 (dynamic) vs DCA-Trie v3 (lazy).

For v3 — design, how to run it, how it is tested, and what is still unverified
— see [docs/LAZY_CONSTRAINT.md](../../docs/LAZY_CONSTRAINT.md).

## Quick Start

```bash
# Run all conditions on WebQSP, 50 samples
uv run python experiments/type_oracle_full/main.py --method all --max-samples 50

# Full ablation (baseline + v1 + v2 + v2-nogates) with metrics
bash experiments/type_oracle_full/run_ablation.sh --max-samples 300
```

## What It Does

Runs controlled comparisons of constrained decoding strategies on knowledge graph QA.

### Conditions

| Condition | Description |
|-----------|-------------|
| `GCR_Baseline` | Unfiltered DFS paths, standard constrained decoding |
| `DCA_v1_Static` | TypeOracle pre-filters all paths, then builds trie |
| `DCA_v2_Dynamic` | Iterative hop-by-hop trie expansion with symbolic gates |
| `DCA_v2_NoGates` | v2 with type/range gates disabled (ablation proxy for DoG) |
| `DCA_v3_Lazy` | Single decoding pass; constraint materialised at the frontier, no DFS |
| `DCA_v3_NoGates` | v3 with gates disabled — admits exactly the baseline trie's language |

### Output

Results saved to `results/<timestamp>/`:

```
<timestamp>/
  config.json     # config + provenance (git commit, package versions, GPU)
  summary.json    # rewritten after every condition, not just at the end
  status.json     # heartbeat: state, conditions done, elapsed, ETA, est. cost
  run.log
  <dataset>/
    predictions_GCR_Baseline.jsonl
    predictions_DCA_v1_Static.jsonl
    predictions_DCA_v2_Dynamic.jsonl
    predictions_DCA_v2_NoGates.jsonl   # if --method ablation
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-path` | `rmanluo/GCR-Meta-Llama-3.1-8B-Instruct` | LLM to use |
| `--datasets` | `RoG-webqsp RoG-cwq` | Datasets to run |
| `--split` | `test` | Dataset split |
| `--max-samples` | `50` | Questions per dataset |
| `--method` | `all` | `baseline`, `v1`, `v2`, `v2-nogates`, `v3`, `v3-nogates`, `all`, `ablation`, `lazy`, `lazy-ablation` |
| `--index-len` | `2` | Max hops |
| `--beam-size` | `5` | Beam width |
| `--seed` | `42` | Random seed |
| `--trace` | `false` | Print per-sample trace (7-step format) |
| `--collect-metrics` | `false` | Collect BUR/SIR/volatility/RV metrics |
| `--force-rerun` | `false` | Ignore existing checkpoints and break a stale/live lock |
| `--sample-timeout` | `120` | Per-question timeout (seconds) |
| `--full` | `false` | Run every question in the split (same as `--max-samples 0`) |
| `--run-name` | — | Stable dir under `results/final_experiment/` (resume across instances) |
| `--resume` | `false` | Continue the most recent run directory |

### Rented-GPU options

| Argument | Default | Description |
|----------|---------|-------------|
| `--attn-impl` | `auto` | `auto` picks flash-attn 2 on any sm_80+ card (A100, A6000, 4090, L40S, H100) when installed, else `sdpa` |
| `--dtype` | `bf16` | `bf16`, `fp16`, `fp32` |
| `--quant` | `none` | `8bit` / `4bit` for cards under ~17 GB VRAM |
| `--allow-cpu` | `false` | Proceed with no GPU visible (otherwise preflight aborts) |
| `--no-preflight` | `false` | Skip VRAM/disk/driver checks |
| `--cost-per-hour` | `$VAST_COST_PER_HOUR` | Logs a running cost estimate |
| `--max-runtime-hours` | `0` | Stop before starting a new condition once the budget is spent |

Preflight runs *before* the ~16 GB model download and aborts on a missing GPU,
too little VRAM, or too little free space in the HF cache — so a misconfigured
instance fails in seconds instead of ten minutes of paid GPU time. Point the
cache at the big volume first: `export HF_HOME=/workspace/hf-cache`.

## Ablation Study

```bash
# Full ablation: all 4 conditions, metrics collection, 300 samples
bash experiments/type_oracle_full/run_ablation.sh --max-samples 300

# Custom seed for reproducibility
bash experiments/type_oracle_full/run_ablation.sh --seed 123 --max-samples 500

# Lazy (v3) ablation: baseline + v1 + v3 + v3-nogates
uv run python experiments/type_oracle_full/main.py \
  --method lazy-ablation --datasets RoG-webqsp --max-samples 300
```

## Tests

CPU only — no model weights are loaded, ~15s.

```bash
uv run python -m pytest tests/ -q
```

## Checkpoint/Resume

Each condition writes predictions incrementally to JSONL. If interrupted, re-running
will skip already-processed questions. Use `--force-rerun` to start fresh.

## Reproducing Full Results

Run one dataset at a time to avoid losing progress if interrupted.
Both runs share the same output directory so results are combined.

On a GPU box use `./.venv/bin/python`, not `uv run` — `uv run` re-syncs the
project and `pyproject.toml` pins torch to the CPU index for local dev.

```bash
# Step 1: WebQSP (~1600 questions)
./.venv/bin/python experiments/type_oracle_full/main.py \
  --method all --datasets RoG-webqsp --full --run-name run1

# Step 2: CWQ (~3500 questions)
./.venv/bin/python experiments/type_oracle_full/main.py \
  --method all --datasets RoG-cwq --full --run-name run1
```

Via Vast.ai (the offer's $/hr is forwarded to `--cost-per-hour` automatically):

```bash
bash scripts/run_vast.sh --run-name run1 --datasets RoG-webqsp
bash scripts/run_vast.sh --run-name run1 --datasets RoG-cwq

# with a spend guard
bash scripts/run_vast.sh --run-name run1 --datasets RoG-cwq --max-hours 8
```

## Requirements

- GPU with 16GB+ VRAM; sm_80+ (Ampere or newer) to get flash-attn 2
- Prebuilt flash-attn wheel: `bash scripts/install_flash_attn.sh` (never build from source)
- Launch with `./.venv/bin/python`, not `uv run` — see `docs/VAST_AI_SETUP.md`
- Python 3.11+
- `transformers>=4.44,<5.0` (pinned — 5.x has breaking generation API changes)
- See `scripts/setup-env.sh` for dependencies
