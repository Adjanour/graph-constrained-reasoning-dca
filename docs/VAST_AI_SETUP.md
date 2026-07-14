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

Based on `experiments/type_oracle_full/README.md` and `setup.sh`:

| Requirement | Minimum | Recommended | Why |
|---|---|---|---|
| **GPU VRAM** | 16 GB | 40 GB (A100 40GB) | `rmanluo/GCR-Meta-Llama-3.1-8B-Instruct` is ~16 GB in FP16; group-beam search needs headroom |
| **GPU Model** | Any NVIDIA with 16 GB+ | A100 40GB or A100 80GB | flash-attn 2.x has pre-built wheels for A100; ~20% faster beam search |
| **CUDA** | ≥ 12.1 | 12.4+ | Required by transformers 4.44+ and flash-attn |
| **System RAM** | 32 GB | 64 GB | Graph data, trie construction, and dataset loading |
| **Disk** | 80 GB | 150 GB | Model weights (~16 GB) + HuggingFace cache + datasets + results |
| **Python** | 3.11 | 3.12 | `pyproject.toml` specifies `requires-python = ">=3.11"` |

### GPU Options (cheapest → best)

| GPU | VRAM | flash-attn? | Typical Price | Notes |
|---|---|---|---|---|
| RTX 4090 | 24 GB | No pre-built wheel; sdpa fallback | $0.20–0.40/hr | Budget option, ~20% slower without flash-attn |
| RTX 3090 | 24 GB | No | $0.15–0.30/hr | Older, slower, but works |
| A100 40GB | 40 GB | Yes | $0.50–1.50/hr | Best balance — flash-attn works, fits everything comfortably |
| A100 80GB | 80 GB | Yes | $0.80–2.00/hr | Most comfortable, no VRAM worries at all |
| H100 80GB | 80 GB | Yes | $1.50–3.50/hr | Overkill for this experiment |

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
bash experiments/type_oracle_full/setup.sh
```

### What `setup.sh` Does

1. Installs: `transformers==4.44.2`, `accelerate`, `datasets`, `marisa-trie`,
   `networkx`, `scikit-learn`, `tiktoken`, `sentencepiece`, `protobuf`
2. Detects Python/CUDA/PyTorch versions
3. Tries to install **flash-attn** from pre-built wheel (A100 only)
4. Falls back to **sdpa** if no matching wheel found (~20% slower beam search)

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
bash experiments/type_oracle_full/run.sh --max-samples 10

# Default — 50 samples, both datasets, all 3 methods
bash experiments/type_oracle_full/run.sh

# Full test set (no subsampling)
bash experiments/type_oracle_full/run.sh --full

# One dataset only
bash experiments/type_oracle_full/run.sh --datasets RoG-webqsp

# One method only (baseline, v1, v2)
bash experiments/type_oracle_full/run.sh --method v2
```

### Run in Background (Recommended for Long Runs)

```bash
nohup bash experiments/type_oracle_full/run.sh > /workspace/experiment.log 2>&1 &
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
bash experiments/type_oracle_full/setup.sh
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
5. bash experiments/type_oracle_full/setup.sh
6. bash experiments/type_oracle_full/run.sh
7. Ctrl+b, d (tmux detach)
8. scp -P PORT -r root@IP:/workspace/.../results/ ./
9. vastai destroy instance INSTANCE_ID
```

---

## Sources

[Quickstart](https://docs.vast.ai/quickstart) · [Instances](https://docs.vast.ai/guides/instances/overview) · [Pricing](https://docs.vast.ai/guides/instances/pricing) · [SSH](https://docs.vast.ai/guides/instances/connect/ssh) · [Storage](https://docs.vast.ai/guides/instances/storage) · [Data Movement](https://docs.vast.ai/guides/instances/data-movement) · [Templates](https://docs.vast.ai/guides/templates/creating-templates) · [Template Settings](https://docs.vast.ai/guides/templates/template-settings) · [Advanced Setup](https://docs.vast.ai/guides/templates/advanced-setup) · [PyTorch](https://docs.vast.ai/pytorch) · [CLI](https://docs.vast.ai/cli/reference) · [Community Guide](https://github.com/joystiller/vast-ai-guide)
