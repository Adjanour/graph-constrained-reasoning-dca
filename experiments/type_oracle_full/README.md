# DCA-Trie Full Experiment

End-to-end experiment comparing GCR baseline vs DCA-Trie v1 (static) vs DCA-Trie v2 (dynamic).

## Quick Start

```bash
# Run everything: setup + both datasets + all 3 conditions, 50 samples each
bash experiments/type_oracle_full/run.sh
```

## What It Does

1. **Setup** (`setup.sh`): installs dependencies, optional flash-attn wheel
2. **Run** (`run.py`): runs all three conditions on both WebQSP and CWQ

### Conditions

| Condition | Description |
|-----------|-------------|
| `GCR_Baseline` | Unfiltered DFS paths, standard constrained decoding |
| `DCA_v1_Static` | TypeOracle pre-filters all paths, then builds trie |
| `DCA_v2_Dynamic` | Iterative hop-by-hop trie expansion with symbolic gates |

### Output

Results saved to `results/final_experiment/<timestamp>/`:

```
<timestamp>/
  config.json
  summary.json
  RoG-webqsp/
    predictions_GCR_Baseline.jsonl
    predictions_DCA_v1_Static.jsonl
    predictions_DCA_v2_Dynamic.jsonl
  RoG-cwq/
    predictions_GCR_Baseline.jsonl
    predictions_DCA_v1_Static.jsonl
    predictions_DCA_v2_Dynamic.jsonl
```

## Options

```bash
# Both datasets, all methods, 50 samples (default)
bash experiments/type_oracle_full/run.sh

# One dataset only
bash experiments/type_oracle_full/run.sh --datasets RoG-webqsp

# One method only
bash experiments/type_oracle_full/run.sh --method v1

# Full test set (no subsampling)
bash experiments/type_oracle_full/run.sh --full

# Custom sample count
bash experiments/type_oracle_full/run.sh --max-samples 10

# Fresh start (ignore checkpoints)
bash experiments/type_oracle_full/run.sh --force-rerun

# Custom output directory
bash experiments/type_oracle_full/run.sh --output-dir /path/to/results
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-path` | `rmanluo/GCR-Meta-Llama-3.1-8B-Instruct` | LLM to use |
| `--datasets` | `RoG-webqsp RoG-cwq` | Datasets to run |
| `--split` | `test` | Dataset split |
| `--max-samples` | `50` | Questions per dataset |
| `--method` | `all` | `baseline`, `v1`, `v2`, or `all` |
| `--index-len` | `2` | Max hops |
| `-k` | `10` | Beam width |
| `--gen-mode` | `group-beam` | `greedy`, `group-beam`, or `beam` |
| `--force-rerun` | `false` | Ignore existing checkpoints |

## Checkpoint/Resume

Each condition writes predictions incrementally to JSONL. If interrupted, re-running
will skip already-processed questions. Use `--force-rerun` to start fresh.

## Reproducing Full Results

Run one dataset at a time to avoid losing progress if interrupted.
Both runs share the same output directory so results are combined.

```bash
# Step 1: WebQSP (~1600 questions, ~3 methods)
bash experiments/type_oracle_full/run.sh \
  --datasets RoG-webqsp --full \
  --output-dir results/final_experiment/run1

# Step 2: CWQ (~3500 questions, ~3 methods)
bash experiments/type_oracle_full/run.sh \
  --datasets RoG-cwq --full \
  --output-dir results/final_experiment/run1
```

Via Vast.ai:

```bash
bash scripts/run_vast.sh --datasets RoG-webqsp --output-dir results/final_experiment/run1
bash scripts/run_vast.sh --datasets RoG-cwq    --output-dir results/final_experiment/run1
```

## Requirements

- GPU with 16GB+ VRAM (A100 recommended for flash-attn)
- Python 3.10+
- `transformers>=4.44,<5.0` (pinned — 5.x has breaking generation API changes)
- See `setup.sh` for dependencies
