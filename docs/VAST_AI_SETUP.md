# Running `experiments/type_oracle_full` on Vast.ai

Step-by-step guide to rent a GPU instance on [Vast.ai](https://vast.ai) and run the
DCA-Trie full experiment. Based on current Vast.ai documentation (July 2026).

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Instance Requirements](#2-instance-requirements)
3. [Rent an Instance (GUI)](#3-rent-an-instance-gui)
4. [Rent an Instance (CLI)](#4-rent-an-instance-cli)
5. [Connect via SSH](#5-connect-via-ssh)
6. [Transfer Code to the Instance](#6-transfer-code-to-the-instance)
7. [Set Up the Environment](#7-set-up-the-environment)
8. [Run the Experiment](#8-run-the-experiment)
9. [Monitor Progress](#9-monitor-progress)
10. [Retrieve Results](#10-retrieve-results)
11. [Stop or Destroy the Instance](#11-stop-or-destroy-the-instance)
12. [Cost Estimates](#12-cost-estimates)
13. [Troubleshooting](#13-troubleshooting)
14. [Automated Setup with Provisioning Script](#14-automated-setup-with-provisioning-script)
15. [Appendix: Vast.ai Billing Rules](#15-appendix-vastai-billing-rules)

---

## 1. Prerequisites

### Vast.ai Account

1. Go to [https://cloud.vast.ai](https://cloud.vast.ai) and create an account.
2. Verify your email (check spam folder; you cannot rent until verified).
3. Go to **Billing → Add Credit** and top up with a credit card, BitPay, or Crypto.com.
   Minimum deposit is **$5 USD**. Enable **autobilling** on the Billing page to avoid
   interruptions — set an auto-charge threshold so your card is charged when your
   balance falls below a certain amount.

### SSH Key

Vast.ai uses SSH key authentication only — there is no password.

```bash
# Generate a key (if you don't already have one)
ssh-keygen -t ed25519 -C "your_email@example.com"

# Print the public key
cat ~/.ssh/id_ed25519.pub
# Output: ssh-ed25519 AAAAC3NzaC1lZ... your_email@example.com
```

Copy the **entire output** (including the `ssh-ed25519` prefix and email suffix).

Add it to Vast.ai via one of:

- **Web UI**: Go to [Settings → SSH Keys](https://cloud.vast.ai) (under the Account
  tab) and paste the key.
- **CLI**:
  ```bash
  pip install vastai
  vastai set api-key YOUR_API_KEY
  vastai create ssh-key
  ```
  This auto-generates a keypair and registers it.

> **Important**: New keys only apply to instances created *after* adding the key.
> Existing instances keep their original keys.

---

## 2. Instance Requirements

Based on `experiments/type_oracle_full/README.md` and `scripts/setup-env.sh`:

| Requirement | Minimum | Recommended | Why |
|---|---|---|---|
| **GPU VRAM** | 16 GB | 40 GB (A100 40GB) | `rmanluo/GCR-Meta-Llama-3.1-8B-Instruct` is ~16 GB in FP16; group-beam search needs headroom |
| **GPU Model** | Any NVIDIA with 16 GB+ | Any sm_80+ card (Ampere or newer) | flash-attn 2 needs sm_80+; ~20% faster beam search |
| **CUDA** | ≥ 12.1 | 12.4+ | Required by transformers 4.44+ and flash-attn |
| **System RAM** | 32 GB | 64 GB | Graph data, trie construction, and dataset loading |
| **Disk** | 80 GB | 150 GB | Model weights (~16 GB) + HuggingFace cache + datasets + results |
| **Python** | 3.11 | 3.12 | `pyproject.toml` specifies `requires-python = ">=3.11"` |

### GPU Options (cheapest → best)

| GPU | VRAM | Arch | flash-attn 2? | Typical Price | Notes |
|---|---|---|---|---|---|
| RTX 3090 | 24 GB | sm_86 | Yes | $0.15–0.30/hr | Ampere — wheels exist; older and slower |
| RTX 4090 | 24 GB | sm_89 | Yes | $0.20–0.40/hr | Best value; fits with room for beam search |
| L40S | 48 GB | sm_89 | Yes | $0.60–1.00/hr | Plenty of headroom |
| A100 40GB | 40 GB | sm_80 | Yes | $0.50–1.50/hr | Fits everything comfortably |
| A100 80GB | 80 GB | sm_80 | Yes | $0.80–2.00/hr | No VRAM worries at all |
| H100 80GB | 80 GB | sm_90 | Yes | $1.50–3.50/hr | Overkill for this experiment |
| T4 / V100 | 16 GB | sm_75/70 | **No** | $0.10–0.25/hr | Pre-Ampere: sdpa only, and 16 GB is tight |

flash-attn 2 runs on **every Ampere-or-newer card**, not just A100s — the wheel
matrix is keyed on (CUDA, torch, python, C++ ABI), not on the GPU model.
`scripts/install_flash_attn.sh` resolves and downloads the matching prebuilt
wheel; it is called automatically by `setup-env.sh` on a GPU box.

> Never let pip build flash-attn from source here: it compiles CUDA kernels for
> 30–90 minutes at $/hr and frequently OOMs. If no matching wheel exists, the
> script skips and `main.py` falls back to `sdpa`.

---

## 3. Rent an Instance (GUI)

### Step 1: Select a Template

1. Go to [cloud.vast.ai/templates](https://cloud.vast.ai/templates).
2. Find the **"PyTorch"** recommended template (built on `vastai/pytorch` base image,
   PyTorch pre-installed at `/venv/main/`).
3. Click the **pencil icon** (edit) to review settings. Key fields:
   - **Image Path:Tag**: `vastai/pytorch` — use the **Version Tag** dropdown to select
     the CUDA version matching your target GPU (e.g., `2.6.0-cuda-12.6.3-py312`).
   - **Launch Mode**: **SSH** (preferred — keeps the experiment running even if you
     disconnect). Alternatively **Jupyter + SSH** for web-based terminal access.
   - **On-start Script** (optional): paste setup commands — see
     [Section 14](#14-automated-setup-with-provisioning-script) for a ready-to-use script.
   - **Disk Space**: set to **150 GB** minimum. This cannot be changed after creation.
4. Click **Create & Use** to proceed to the GPU search page.

### Step 2: Search and Filter

You'll land on the **Search** page with the template pre-loaded. Apply filters:

| Filter | Value | Where |
|---|---|---|
| GPU Name | `A100` (or `RTX 4090`) | GPU filter dropdown |
| Min GPU Memory | `16` GB | VRAM slider |
| Min RAM | `32` GB | System RAM slider |
| Min Disk | `150` GB | Disk slider |
| Reliability | `> 95%` | Reliability slider — higher = less likely to be interrupted |
| Sort by | `Price: Low to High` | Sort dropdown |
| Instance Type | `On-demand` | For guaranteed uptime. Use `Interruptible` for cheaper but pre-emptible |

> **Tip**: Prices fluctuate in real time based on supply/demand. If a GPU you want is
> expensive, check back in a few hours — prices change frequently.

### Step 3: Review and Rent

1. Click on an offer card to review details (GPU model, bandwidth, location, reliability).
2. Verify the **disk space** shown matches your template setting.
3. Click **Rent**.
4. Wait 1–5 minutes for the instance to boot. If the Docker image needs to be pulled
   fresh (first time on that host), it can take 10–60 minutes.

> **Disk size is permanent.** Once you create the instance, you cannot change it.
> If you run out of space, you must create a new instance with a larger disk.

---

## 4. Rent an Instance (CLI)

The CLI gives you more control and is scriptable.

### Install the CLI

```bash
pip install vastai
```

### Set Your API Key

Get it from [cloud.vast.ai → Settings → API Keys](https://cloud.vast.ai), then:

```bash
vastai set api-key YOUR_API_KEY
```

It's saved to `~/.config/vastai/vast_api_key`.

### Search for Offers

```bash
# Search for A100 40GB instances with SSH, sorted by price
vastai search offers \
  "gpu_name=A100_80GB num_gpus=1 gpu_ram>=40 dph<=1.5 reliability>=0.95 \
   disk_space>=150 inet_down>=200 inet_up>=100" \
  --order dph --raw
```

Key filter fields:

| Field | Description | Example |
|---|---|---|
| `gpu_name` | Exact GPU name (with underscores) | `A100_80GB`, `RTX_4090` |
| `num_gpus` | Number of GPUs | `1` |
| `gpu_ram` | Minimum GPU VRAM in GB | `>=16` |
| `dph` | Max $/hr for GPU compute | `<=1.5` |
| `reliability` | Min reliability score (0–1) | `>=0.95` |
| `disk_space` | Min disk in GB | `>=150` |
| `inet_down` | Min download speed in Mbps | `>=200` |
| `inet_up` | Min upload speed in Mbps | `>=100` |
| `cuda_vers` | Min CUDA version | `>=12.1` |
| `order` | Sort field | `dph` (price) |
| `type` | Instance type | `on-demand`, `interruptible` |

### Create (Rent) an Instance

```bash
vastai create instance OFFER_ID \
  --image vastai/pytorch:2.6.0-cuda-12.6.3-py312 \
  --disk 150 \
  --ssh \
  --direct \
  --onstart-cmd "bash /workspace/setup_and_run.sh" \
  --env '-e TZ=UTC'
```

Where `OFFER_ID` is the numeric ID from the search results (the `id` field in `--raw`
output).

Key `create instance` options:

| Option | Description |
|---|---|
| `--image` | Docker image path:tag |
| `--disk` | Disk size in GB (cannot be changed later) |
| `--ssh` | Launch with SSH access |
| `--jupyter` | Launch with Jupyter + SSH |
| `--direct` | Try direct SSH connection (faster than proxy) |
| `--onstart-cmd` | Bash script to run on instance startup |
| `--env` | Docker environment variables (e.g., `-e FOO=bar`) |
| `--args` | Arguments to pass to docker entrypoint |
| `--entrypoint` | Override docker entrypoint command |
| `--template` | Use a saved template by name or ID |

### Manage Instances

```bash
# List your instances
vastai show instances

# Show details for a specific instance
vastai show instance INSTANCE_ID

# Stop (data persists, storage billing continues, GPU released)
vastai stop instance INSTANCE_ID

# Start a stopped instance
vastai start instance INSTANCE_ID

# Destroy (all data permanently deleted, billing stops)
vastai destroy instance INSTANCE_ID

# Execute a command on a stopped instance (e.g., to free disk space)
vastai execute INSTANCE_ID 'du -d1 -h'

# Copy files between instances
vastai copy SRC_INSTANCE_ID:/path/to/file DST_INSTANCE_ID:/path/to/dest
```

> **Note**: `INSTANCE_ID` is the numeric ID shown in `show instances` output.
> From inside the container, use `echo $VAST_CONTAINERLABEL` to get it (format:
> `C.38250`).

---

## 5. Connect via SSH

Once the instance status shows **Running**, click the instance card in the dashboard.
The **Connect** tab shows the SSH command.

### SSH Command Format

```bash
ssh -p PORT root@IP_ADDRESS
```

For example:

```bash
ssh -p 12345 root@52.204.230.7
```

### With Port Forwarding (for TensorBoard or local result viewing)

```bash
ssh -p PORT root@IP_ADDRESS -L 8080:localhost:8080
```

This forwards `localhost:8080` on your machine to port 8080 on the instance.

### Direct vs Proxy SSH

Vast.ai tries two connection methods:

- **Direct SSH**: Connects directly to the host's IP and port. Faster, lower latency,
  better for SCP transfers. Requires the host to have open ports.
- **Proxy SSH**: Routes through Vast.ai's proxy servers. Works even if the host doesn't
  have open ports. Higher latency, slower for large file transfers.

The instance page shows which method is available. If both are shown, **use direct SSH**,
especially for SCP.

### VS Code Remote SSH

1. Install the **"Remote - SSH"** extension in VS Code.
2. Click the remote connection icon (bottom-left corner).
3. Enter: `ssh -p PORT root@IP_ADDRESS`
4. VS Code will configure the instance — you can then work as if it were local.

### tmux

Vast.ai SSH instances launch inside a **tmux** session by default. This is critical —
if your SSH connection drops, tmux keeps your experiment running.

| Action | Keybinding |
|---|---|
| New terminal tab | `Ctrl+b`, then `c` |
| Switch to next tab | `Ctrl+b`, then `n` |
| Switch to previous tab | `Ctrl+b`, then `p` |
| Split horizontally | `Ctrl+b`, then `"` |
| Split vertically | `Ctrl+b`, then `%` |
| Detach (leave running) | `Ctrl+b`, then `d` |
| Re-attach | `tmux attach` |

> **Do not disable tmux.** SSH connections to Vast.ai can be unstable. tmux is your
> safety net.

---

## 6. Transfer Code to the Instance

### Option A: Git Clone (Recommended)

```bash
# SSH into the instance, then:
cd /workspace
git clone https://github.com/Adjanour/graph-constrained-reasoning-dca.git
cd graph-constrained-reasoning
```

> **Before renting**: make sure to `git push` all local changes first:
> ```bash
> cd /home/bernard/research/projects/graph-constrained-reasoning
> git add -A && git commit -m "WIP" && git push
> ```

### Option B: SCP (for uncommitted local files)

From your **local** machine:

```bash
# Copy a single file
scp -P PORT /local/file root@IP_ADDRESS:/workspace/graph-constrained-reasoning/

# Copy a directory recursively
scp -P PORT -r /local/dir/ root@IP_ADDRESS:/workspace/graph-constrained-reasoning/
```

> Use uppercase `-P` for SCP. PORT and IP_ADDRESS from the instance's Connect tab.
> Use **direct SSH** for transfers > 1 GB (proxy is slow).

### Option C: Vast CLI Copy

```bash
vastai copy LOCAL_PATH INSTANCE_ID:/workspace/
```

---

## 7. Set Up the Environment

```bash
cd /workspace/graph-constrained-reasoning

# Activate the pre-installed venv
source /venv/main/bin/activate

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# Install experiment dependencies
bash scripts/setup-env.sh
```

### What `setup-env.sh` Does

1. Creates `.venv` and installs torch — the **CUDA** build when `nvidia-smi` is
   present, the CPU build otherwise
2. Installs: `transformers`, `accelerate`, `datasets`, `marisa-trie`,
   `networkx`, `scikit-learn`, `tiktoken`, `sentencepiece`, `protobuf`
3. On a GPU box, runs `install_flash_attn.sh`, which probes torch / CUDA /
   python / C++ ABI and downloads the matching **prebuilt** flash-attn 2 wheel
4. Falls back to **sdpa** if no matching wheel exists (~20% slower beam search)
5. Prints whether the venv's torch actually sees the GPU

> Launch the experiment with `./.venv/bin/python`, **not** `uv run`.
> `pyproject.toml` pins torch to the CPU index for local dev, and `uv run`
> re-syncs the project — which would replace CUDA torch with a CPU build.
> `run_vast.sh` and `run_ablation.sh` already do this.

### Python Version Fix

The `vastai/pytorch` template ships Python 3.12 in `/venv/main/`. If you see a version
error:

```bash
python --version
# If < 3.11:
apt-get update && apt-get install -y python3.12 python3.12-venv python3.12-dev
python3.12 -m venv /workspace/venv
source /workspace/venv/bin/activate
```

---

## 8. Run the Experiment

```bash
cd /workspace/graph-constrained-reasoning
source /venv/main/bin/activate  # if not already activated

# Quick test — 10 samples, both datasets, all 3 methods
bash experiments/type_oracle_full/main.py --max-samples 10

# Default — 50 samples, both datasets, all 3 methods
bash experiments/type_oracle_full/main.py

# Full test set (no subsampling)
bash experiments/type_oracle_full/main.py --full

# One dataset only
bash experiments/type_oracle_full/main.py --datasets RoG-webqsp

# One method only (baseline, v1, v2)
bash experiments/type_oracle_full/main.py --method v2
```

### Run in Background (Recommended for Long Runs)

```bash
nohup bash experiments/type_oracle_full/main.py > /workspace/experiment.log 2>&1 &
tail -f /workspace/experiment.log
```

### All CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--model-path` | `rmanluo/GCR-Meta-Llama-3.1-8B-Instruct` | HuggingFace model path |
| `--datasets` | `RoG-webqsp RoG-cwq` | Datasets to evaluate |
| `--split` | `test` | Dataset split |
| `--max-samples` | `50` | Questions per dataset |
| `--method` | `all` | `baseline`, `v1`, `v2`, or `all` |
| `--index-len` | `2` | Max hop count |
| `-k` | `10` | Beam width |
| `--gen-mode` | `group-beam` | `greedy`, `group-beam`, or `beam` |
| `--max-new-tokens` | `256` | Max tokens per generation |
| `--sample-timeout` | `120` | Per-sample timeout in seconds |
| `--force-rerun` | `false` | Ignore checkpoints, start fresh |
| `--output-dir` | auto-timestamped | Custom results directory |

Each condition writes predictions incrementally to JSONL. If interrupted, re-running
skips already-processed questions. Use `--force-rerun` to start fresh.

---

## 9. Monitor Progress

```bash
# Live log
tail -f /workspace/graph-constrained-reasoning/results/final_experiment/*/run.log

# GPU usage (refresh every 5s)
watch -n 5 nvidia-smi

# Disk usage
df -h /
```

Your instance card on the Vast.ai dashboard shows runtime, estimated cost, and balance.

---

## 10. Retrieve Results

```bash
# From your local machine — copy entire results directory
scp -P PORT -r root@IP_ADDRESS:/workspace/graph-constrained-reasoning/results/ \
  ./results_from_vast/

# Copy just the summary
scp -P PORT \
  root@IP_ADDRESS:/workspace/graph-constrained-reasoning/results/final_experiment/*/summary.json \
  ./summary.json
```

**Cloud Sync** (works on stopped instances too): connect S3/Google Drive/Dropbox in
Account Settings, then:
```bash
vastai cloud copy INSTANCE_ID:/workspace/results s3://my-bucket/results
```

---

## 11. Stop or Destroy the Instance

| Action | GPU Billing | Storage Billing | Data | Reversible? |
|---|---|---|---|---|
| **Running** | Charged | Charged | Persists | — |
| **Stop** | Stops | Still charged | Persists | Can restart |
| **Destroy** | Stops | Stops | **Permanently deleted** | Cannot undo |

```bash
vastai stop instance INSTANCE_ID       # Release GPU
vastai start instance INSTANCE_ID      # Restart
vastai destroy instance INSTANCE_ID    # Delete everything, stop all billing
```

> **Storage charges continue on stopped instances.** Retrieve results first, then destroy.

**Lifetime**: Every offer has a max duration. When expired, instance stops. Data persists
until destroyed. Host may extend. **Always copy results before lifetime expires.**

**Budget safety**: Balance hits zero → with card: auto-recharge; without: deletion.
Enable autobilling at **Billing → Auto-Recharge**.

---

## 12. Cost Estimates

| GPU | 50-samples (2–3 hr) | Full test set (6–12 hr) |
|---|---|---|
| RTX 4090 ($0.30/hr) | $0.60–0.90 | $1.80–3.60 |
| A100 40GB ($1.00/hr) | $1.00–2.00 | $6–12 |
| A100 80GB ($1.50/hr) | $1.50–3.00 | $9–18 |

**Save money**: Use interruptible instances (~50% cheaper), validate with `--max-samples 10`,
destroy immediately after, consider reserved for long projects.

---

## 13. Troubleshooting

| Problem | Fix |
|---|---|
| **Permission denied (publickey)** | Verify key in Settings → SSH Keys. `chmod 600 ~/.ssh/id_ed25519` |
| **CUDA out of memory** | Use `-k 5` or `--gen-mode greedy`. Use A100 40GB+ |
| **Model download fails** | `curl https://huggingface.co`. If gated: `huggingface-cli login` |
| **Instance won't start** | Docker pull can take 10–60 min. Check logs. Try different host |
| **SSH drops** | Always use tmux. Reconnect to rejoin session |
| **Disk full** | Cannot resize. Create new instance. `vastai execute ID 'du -d1 -h'` |
| **flash-attn fails** | Non-critical — falls back to sdpa (~20% slower beam search) |

---

## 14. Automated Setup with Provisioning Script

**Setup script** (save as GitHub Gist):

```bash
#!/bin/bash
set -eo pipefail
source /venv/main/bin/activate
cd /workspace
if [ ! -d "graph-constrained-reasoning" ]; then
    git clone https://github.com/Adjanour/graph-constrained-reasoning-dca.git
fi
cd graph-constrained-reasoning
bash scripts/setup-env.sh
env >> /etc/environment
```

**Use via GUI**: Edit template → Env Variables → Add `PROVISIONING_SCRIPT` = URL

**Use via CLI**:
```bash
vastai create instance OFFER_ID \
  --image vastai/pytorch:2.6.0-cuda-12.6.3-py312 \
  --disk 150 --ssh \
  --env '-e PROVISIONING_SCRIPT=https://raw.githubusercontent.com/...'
```

---

## 15. Appendix: Billing Rules

| Component | Charged When | Notes |
|---|---|---|
| GPU compute | Running | Per second |
| Storage | Exists (running or stopped) | Per GB; higher when stopped |
| Bandwidth | Data transferred | Same-host free |

Stopped ≠ Destroyed. Destroyed = data gone. Balance=0 with card → auto-recharge.
Min deposit: $5 USD. `vastai show charges` / `vastai show instances` / `vastai show user`.

---

## Quick Reference

```
1. git push (from local)
2. cloud.vast.ai → Templates → PyTorch → Rent
3. ssh -p PORT root@IP
4. cd /workspace && git clone <repo> && cd graph-constrained-reasoning
5. bash scripts/setup-env.sh
6. bash experiments/type_oracle_full/main.py
7. Ctrl+b, d (tmux detach)
8. scp -P PORT -r root@IP:/workspace/.../results/ ./
9. vastai destroy instance INSTANCE_ID
```

---

## Sources

[Quickstart](https://docs.vast.ai/quickstart) · [Instances](https://docs.vast.ai/guides/instances/overview) · [Pricing](https://docs.vast.ai/guides/instances/pricing) · [SSH](https://docs.vast.ai/guides/instances/connect/ssh) · [Storage](https://docs.vast.ai/guides/instances/storage) · [Data Movement](https://docs.vast.ai/guides/instances/data-movement) · [Templates](https://docs.vast.ai/guides/templates/creating-templates) · [Template Settings](https://docs.vast.ai/guides/templates/template-settings) · [Advanced Setup](https://docs.vast.ai/guides/templates/advanced-setup) · [PyTorch](https://docs.vast.ai/pytorch) · [CLI](https://docs.vast.ai/cli/reference) · [Community Guide](https://github.com/joystiller/vast-ai-guide)

---

## Automation Architecture

One-command GPU orchestration for the DCA-Trie experiment. These scripts handle
the entire lifecycle: search → rent → setup → run → download → destroy.

### Prerequisites

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

### Quick Start

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
directly to `experiments/type_oracle_full/main.py`.

### Lifecycle

```
1. Search     vastai search offers for cheapest GPU matching filters
2. Rent       vastai create instance with PyTorch image + SSH
3. Wait       Poll until instance status is "running"
4. Upload     scp vast_boot.sh to /workspace/ on the instance
5. Setup      Boot script: git clone → setup-env.sh → pip install deps
6. Run        Start main.py with --full (or your args) via nohup
7. Monitor    Poll experiment.log every 60s, print latest line
8. Download   scp results/ directory back to local machine
9. Clean up   Prompt to destroy instance (or keep alive)
```

### Scripts

#### `scripts/run_vast.sh` (runs on your machine)

The main orchestrator. Searches, rents, connects, monitors, downloads.

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--offer ID` | auto-search | Use a specific Vast.ai offer ID |
| `--gpu NAME` | `RTX_4090` | GPU filter for auto-search |
| `--image` | `vastai/pytorch:2.6.0-cuda-12.6.3-py312` | Docker image |
| `--disk` | `200` | Disk size in GB |
| `--help` | — | Print usage |

Everything else is forwarded to `main.py` (e.g., `--max-samples`, `--method`, `--datasets`).

#### `scripts/vast_boot.sh` (runs on the instance)

Boot script uploaded and executed on the instance. It:

1. Activates the PyTorch venv (`/venv/main`)
2. Prints Python/PyTorch/CUDA versions to log
3. Clones (or `git pull`) the repo
4. Runs `scripts/setup-env.sh`
5. Writes `/workspace/setup_done.flag` to signal completion

Logs are written to `/workspace/vast_boot.log`.

### How Monitoring Works

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

### Output

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

### Error Handling

| Situation | Behavior |
|---|---|
| No offers found | Exits with error, suggests CLI search |
| Instance fails to start | Times out after 15 min |
| Boot/setup script fails | Times out after 30 min, shows log path |
| Experiment errors (OOM, etc.) | Prints warning, continues monitoring |
| SSH connection drops | Retries automatically (SSH keepalive) |
| Instance preempted (interruptible) | Experiment has checkpoint/resume — re-run picks up where it left off |

### Cost Estimate

| Scenario | GPU | Time | Cost |
|---|---|---|---|
| Quick test (10 samples) | RTX 4090 @ $0.32/hr | ~5 min | < $0.05 |
| Default (50 samples) | RTX 4090 @ $0.32/hr | ~30 min | ~$0.16 |
| Full run (all samples) | RTX 4090 @ $0.32/hr | ~10 hr | ~$3.20 |
| Full run (A100) | A100 40GB @ $0.80/hr | ~7 hr | ~$5.60 |

### Example Session

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

---

## Command Reference

### SSH + Pull (always first)

```bash
ssh -p PORT root@HOST
cd /workspace/graph-constrained-reasoning
git pull
source /venv/main/bin/activate
```

### Dry Run (10 samples, verify everything works)

#### v2 on CWQ
```bash
python experiments/type_oracle_full/main.py \
  --datasets RoG-cwq --method v2 --max-samples 10 \
  --output-dir results/cwq_dryrun
```

#### All methods on CWQ
```bash
python experiments/type_oracle_full/main.py \
  --datasets RoG-cwq --method all --max-samples 10 \
  --output-dir results/cwq_dryrun_all
```

### Full Runs (background, survive SSH disconnect)

#### v2 on CWQ (~3.5K questions, 3-4 hrs)
```bash
nohup python experiments/type_oracle_full/main.py \
  --datasets RoG-cwq --method v2 --max-samples 999999 \
  --output-dir results/cwq_v2 \
  > /workspace/cwq_v2.log 2>&1 &

tail -f /workspace/cwq_v2.log
```

#### All methods on CWQ (baseline + v1 + v2, ~8-10 hrs)
```bash
nohup python experiments/type_oracle_full/main.py \
  --datasets RoG-cwq --method all --max-samples 999999 \
  --output-dir results/cwq_all \
  > /workspace/cwq_all.log 2>&1 &

tail -f /workspace/cwq_all.log
```

#### v2 on WebQSP (compare against old buggy 54.9%)
```bash
nohup python experiments/type_oracle_full/main.py \
  --datasets RoG-webqsp --method v2 --max-samples 999999 \
  --output-dir results/webqsp_v2_fixed \
  > /workspace/webqsp_v2_fixed.log 2>&1 &

tail -f /workspace/webqsp_v2_fixed.log
```

### Retrieve Results

#### From your machine (replace PORT + HOST with Vast.ai details)
```bash
scp -P PORT -r root@HOST:/workspace/graph-constrained-reasoning/results/ ./results_from_vast/
scp -P PORT root@HOST:/workspace/cwq_v2.log ./results_from_vast/
```

### Automated Launch (local machine, ONE COMMAND)

Full lifecycle: search → rent → setup → run → download → destroy:

```bash
cd /home/bernard/research/projects/graph-constrained-reasoning

# Dry run on WebQSP (10 samples, all methods)
bash scripts/run_vast.sh --max-samples 10 --dataset RoG-webqsp --experiment 4ideas

# Dry run with adaptive-budget experiment
bash scripts/run_vast.sh --max-samples 10 --dataset RoG-webqsp --experiment adaptive-budget

# Full v2 on CWQ (main.py)
bash scripts/run_vast.sh --dataset RoG-cwq --method v2 --max-samples 999999

# Full 4-ideas run on WebQSP
bash scripts/run_vast.sh --dataset RoG-webqsp --experiment 4ideas --max-samples 999999

# Full adaptive-budget run on WebQSP
bash scripts/run_vast.sh --dataset RoG-webqsp --experiment adaptive-budget --max-samples 999999
```

### Manual SSH (if automated script fails)

```bash
# On Vast.ai instance:
cd /workspace
git clone https://github.com/Adjanour/graph-constrained-reasoning-dca.git
cd graph-constrained-reasoning-dca
pip install -e .

# Run experiment
uv run python experiments/type_oracle_full/main.py \
  --model-path rmanluo/GCR-Meta-Llama-3.1-8B-Instruct \
  --datasets RoG-webqsp --max-samples 10 \
  --method all \
  --output-dir results/dryrun
```
