# DCA-Trie Full Experiment

End-to-end experiment comparing GCR baseline vs DCA-Trie v1 (static) vs DCA-Trie v2 (dynamic).

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

### Output

Results saved to `results/<timestamp>/`:

```
<timestamp>/
  config.json
  summary.json
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
| `--method` | `all` | `baseline`, `v1`, `v2`, `v2-nogates`, `all`, or `ablation` |
| `--index-len` | `2` | Max hops |
| `--beam-size` | `5` | Beam width |
| `--seed` | `42` | Random seed |
| `--trace` | `false` | Print per-sample trace (7-step format) |
| `--collect-metrics` | `false` | Collect BUR/SIR/volatility/RV metrics |
| `--force-rerun` | `false` | Ignore existing checkpoints |
| `--sample-timeout` | `120` | Per-question timeout (seconds) |

## Ablation Study

```bash
# Full ablation: all 4 conditions, metrics collection, 300 samples
bash experiments/type_oracle_full/run_ablation.sh --max-samples 300

# Custom seed for reproducibility
bash experiments/type_oracle_full/run_ablation.sh --seed 123 --max-samples 500
```

## Checkpoint/Resume

Each condition writes predictions incrementally to JSONL. If interrupted, re-running
will skip already-processed questions. Use `--force-rerun` to start fresh.

## Reproducing Full Results

Run one dataset at a time to avoid losing progress if interrupted.
Both runs share the same output directory so results are combined.

```bash
# Step 1: WebQSP (~1600 questions)
uv run python experiments/type_oracle_full/main.py \
  --method all --datasets RoG-webqsp --full \
  --output-dir results/final_experiment/run1

# Step 2: CWQ (~3500 questions)
uv run python experiments/type_oracle_full/main.py \
  --method all --datasets RoG-cwq --full \
  --output-dir results/final_experiment/run1
```

Via Vast.ai:

```bash
bash scripts/run_vast.sh --datasets RoG-webqsp --output-dir results/final_experiment/run1
bash scripts/run_vast.sh --datasets RoG-cwq    --output-dir results/final_experiment/run1
```

## Requirements

- GPU with 16GB+ VRAM (A100 recommended for flash-attn)
- Python 3.11+
- `transformers>=4.44,<5.0` (pinned — 5.x has breaking generation API changes)
- See `scripts/setup-env.sh` for dependencies
