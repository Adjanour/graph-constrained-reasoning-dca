# Vast.ai Automation Scripts

One-command GPU orchestration for the DCA-Trie experiment. These scripts handle
the entire lifecycle: search → rent → setup → run → download → destroy.

## Prerequisites

```bash
pip install vastai
vastai set api-key YOUR_API_KEY
# SSH key added to your Vast.ai account (Settings → SSH Keys)
```

You also need `jq` (JSON parser):
```bash
# macOS
brew install jq
# Ubuntu/Debian
sudo apt install jq
```

## Quick Start

```bash
# Full run — both datasets, all 3 methods, ~8–12 hours, ~$3
bash scripts/run_vast.sh

# Quick test — 10 samples, both datasets, ~10 minutes
bash scripts/run_vast.sh --max-samples 10

# One dataset only
bash scripts/run_vast.sh --datasets RoG-webqsp

# One method only
bash scripts/run_vast.sh --method v2

# Use a specific offer you found
bash scripts/run_vast.sh --offer 44169006

# Different GPU
bash scripts/run_vast.sh --gpu A100_40GB
```

All arguments except `--offer`, `--gpu`, `--image`, and `--disk` are forwarded
directly to `experiments/type_oracle_full/run.sh`.

## What It Does

```
1. Search     vastai search offers for cheapest GPU matching filters
2. Rent       vastai create instance with PyTorch image + SSH
3. Wait       Poll until instance status is "running"
4. Upload     scp vast_boot.sh to /workspace/ on the instance
5. Setup      Boot script: git clone → setup.sh → pip install deps
6. Run        Start run.sh with --full (or your args) via nohup
7. Monitor    Poll experiment.log every 60s, print latest line
8. Download   scp results/ directory back to local machine
9. Clean up   Prompt to destroy instance (or keep alive)
```

## Scripts

### `scripts/run_vast.sh` (runs on your machine)

The main orchestrator. Searches, rents, connects, monitors, downloads.

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--offer ID` | auto-search | Use a specific Vast.ai offer ID |
| `--gpu NAME` | `RTX_4090` | GPU filter for auto-search |
| `--image` | `vastai/pytorch:2.6.0-cuda-12.6.3-py312` | Docker image |
| `--disk` | `200` | Disk size in GB |
| `--help` | — | Print usage |

Everything else is forwarded to `run.sh` (e.g., `--max-samples`, `--method`, `--datasets`).

### `scripts/vast_boot.sh` (runs on the instance)

Boot script uploaded and executed on the instance. It:

1. Activates the PyTorch venv (`/venv/main`)
2. Prints Python/PyTorch/CUDA versions to log
3. Clones (or `git pull`) the repo
4. Runs `experiments/type_oracle_full/setup.sh`
5. Writes `/workspace/setup_done.flag` to signal completion

Logs are written to `/workspace/vast_boot.log`.

## How Monitoring Works

The orchestrator polls two things:

1. **Boot phase**: checks for `/workspace/setup_done.flag` every 15s (max 30 min)
2. **Experiment phase**: checks for `"Results saved to"` in `experiment.log` every 60s

You can also monitor manually from another terminal:

```bash
# Watch setup progress
ssh -p PORT root@HOST 'tail -f /workspace/vast_boot.log'

# Watch experiment progress
ssh -p PORT root@HOST 'tail -f /workspace/experiment.log'

# Check GPU usage
ssh -p PORT root@HOST 'watch -n 5 nvidia-smi'
```

## Output

Results are downloaded to `results_from_vast/` in the project root:

```
results_from_vast/
├── experiment.log                         # Full experiment log
└── results/
    └── final_experiment/
        └── <timestamp>/
            ├── config.json
            ├── summary.json               # ← key metrics
            ├── run.log
            ├── RoG-webqsp/
            │   ├── predictions_GCR_Baseline.jsonl
            │   ├── predictions_DCA_v1_Static.jsonl
            │   └── predictions_DCA_v2_Dynamic.jsonl
            └── RoG-cwq/
                └── ...
```

## Error Handling

| Situation | Behavior |
|---|---|
| No offers found | Exits with error, suggests CLI search |
| Instance fails to start | Times out after 15 min |
| Boot/setup script fails | Times out after 30 min, shows log path |
| Experiment errors (OOM, etc.) | Prints warning, continues monitoring |
| SSH connection drops | Retries automatically (SSH keepalive) |
| Instance preempted (interruptible) | Experiment has checkpoint/resume — re-run picks up where it left off |

## Cost Estimate

| Scenario | GPU | Time | Cost |
|---|---|---|---|
| Quick test (10 samples) | RTX 4090 @ $0.32/hr | ~5 min | < $0.05 |
| Default (50 samples) | RTX 4090 @ $0.32/hr | ~30 min | ~$0.16 |
| Full run (all samples) | RTX 4090 @ $0.32/hr | ~10 hr | ~$3.20 |
| Full run (A100) | A100 40GB @ $0.80/hr | ~7 hr | ~$5.60 |

## Example Session

```bash
$ bash scripts/run_vast.sh --method v1

========================================
  Vast.ai DCA-Trie Orchestrator
========================================
GPU: RTX_4090  Disk: 200GB
Docker: vastai/pytorch:2.6.0-cuda-12.6.3-py312
Results: /home/bernard/.../results_from_vast
Args: --method v1
========================================

→ Searching for RTX_4090 offers...
  Found offer: 44169006

→ Renting instance...
  Instance ID: 98765

→ Waiting for instance to start (polling every 15s)...
  Loading image...
  Instance is running.

→ Getting SSH details...
  ssh -p 12345 root@52.204.230.7

→ Uploading boot script...
→ Running boot script (clone + dependencies)...
→ Waiting for setup to finish (polling every 15s)...
  Setup complete.

→ Starting experiment...
→ Experiment running. Monitoring every 60s...

  [0] [DCA_v1_Static] 10/1628 2.31 q/s | 4s | skip=0 dead=0
  [1] [DCA_v1_Static] 20/1628 2.28 q/s | 9s | skip=0 dead=0
  ...

→ Downloading results to results_from_vast/ ...
  Results saved to: results_from_vast/

========================================
  DONE
========================================

Destroy instance now? [y/N] y
Instance destroyed. Billing stopped.
```

## See Also

- [VAST_AI_SETUP.md](VAST_AI_SETUP.md) — Manual setup guide, all CLI commands, troubleshooting
- [type_oracle_full/README.md](../experiments/type_oracle_full/README.md) — Experiment docs