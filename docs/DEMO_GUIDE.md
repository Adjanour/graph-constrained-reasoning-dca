# DCA-Trie Demo Guide

Two ways to present the project: a **CLI step-by-step walkthrough** and a **Streamlit web UI**. Each has a replay mode (no GPU) and a live mode (needs GPU).

---

## 1. CLI Step-by-Step Demo (`step_by_step.py`)

Interactive terminal walkthrough that pauses at each pipeline stage. Designed for narrating during a presentation.

### Replay Mode (No GPU)

Uses saved experiment predictions. No model loading required.

```bash
# Random question
uv run python demo/step_by_step.py

# Specific question
uv run python demo/step_by_step.py --question-idx 0

# Show all three methods (baseline, v1, v2)
uv run python demo/step_by_step.py --method all

# Only show v2
uv run python demo/step_by_step.py --method v2

# Use CWQ dataset instead of WebQSP
uv run python demo/step_by_step.py --dataset RoG-cwq --question-idx 0

# Point to a specific results directory
uv run python demo/step_by_step.py --predictions-dir results/ideas_webqsp_full
```

**What you see at each step:**

| Step | Content |
|------|---------|
| 1 | Question, entities, ground-truth answers |
| 2 | KG subgraph stats (nodes, edges, Freebase IDs flagged) |
| 3 | TypeOracle (answer types, schema stats) |
| 4 | DFS path enumeration (all paths shown) |
| 5 | TypeOracle filtering (removed paths with reason) |
| 6 | Trie construction (sizes for baseline, filtered, v2) |
| 7 | Prediction per method (path + answer + correctness) |

Press **ENTER** to advance to the next step. **Ctrl+C** to exit.

### Live Mode (Needs GPU)

Loads the actual model and runs inference.

```bash
# All methods, default beam size
uv run python demo/step_by_step.py --live --method all

# V2 only, beam size 5
uv run python demo/step_by_step.py --live --method v2 --beam-size 5

# Specific question
uv run python demo/step_by_step.py --live --question-idx 42 --method all
```

---

## 2. Streamlit Web UI (`app.py`)

Browser-based visualization with interactive KG graph.

### Pre-computed Mode (No GPU)

```bash
uv run streamlit run demo/app.py
```

Or use the convenience script:

```bash
bash run_demo.sh
```

Opens at `http://localhost:8501`. Select a question from the sidebar, click through the pipeline steps. The KG is rendered as an interactive network graph.

### Live Mode (Needs GPU)

The Streamlit app also has a "Live" mode toggle in the sidebar. When enabled, it loads the model and runs inference in the browser. Requires a GPU.

---

## 3. Renting a GPU on Vast.ai

For live demos, rent a GPU on Vast.ai.

### Automated (Recommended)

The `run_vast.sh` script handles everything: search, rent, upload, run, download, destroy.

```bash
# Full run (both datasets, all methods, 50 samples)
bash scripts/run_vast.sh

# Quick test
bash scripts/run_vast.sh --max-samples 10

# One dataset at a time (avoids losing progress on timeout)
bash scripts/run_vast.sh --datasets RoG-webqsp --output-dir results/vast_run
bash scripts/run_vast.sh --datasets RoG-cwq    --output-dir results/vast_run

# Specific GPU type
bash scripts/run_vast.sh --gpu A100_40GB

# Specific offer (cheapest)
bash scripts/run_vast.sh --offer 44169006
```

### Manual

1. **Find an offer:**
   ```bash
   vastai search offers gpu_name=RTX_4090 gpu_ram=24 'dph_total < 0.3' --order dph_total --limit 5
   ```

2. **Rent:**
   ```bash
   vastai create instance <offer_id> --image nvidia/cuda:12.4.1-devel-ubuntu22.04 --disk 50 --ssh
   ```

3. **Upload and run:**
   ```bash
   # Upload project
   scp -r . root@<instance_ip>:~/graph-constrained-reasoning

   # SSH in
   ssh root@<instance_ip>

   # Install deps
   cd ~/graph-constrained-reasoning
   pip install -e .
   pip install streamlit networkx pyvis

   # Run experiment
   bash experiments/type_oracle_full/run.sh --datasets RoG-webqsp --max-samples 50

   # Download results
   exit
   scp -r root@<instance_ip>:~/graph-constrained-reasoning/results ./results
   ```

4. **Destroy:**
   ```bash
   vastai destroy instance <instance_id>
   ```

---

## 4. Regenerating Demo Data

If you run a new experiment and want the pre-computed demo to use those results:

```bash
# Generate from experiment predictions
uv run python demo/generate_demo_data.py \
    --n-samples 10 \
    --predictions-dir results/final_experiment-<timestamp>/final_experiment/<run>/RoG-webqsp \
    --output-dir demo/demo_data
```

---

## 5. Quick Reference

| Command | GPU? | Purpose |
|---------|------|---------|
| `uv run python demo/step_by_step.py` | No | CLI walkthrough (replay) |
| `uv run python demo/step_by_step.py --live` | Yes | CLI walkthrough (live) |
| `uv run streamlit run demo/app.py` | No | Web UI (pre-computed) |
| `bash run_demo.sh` | No | Web UI shortcut |
| `bash scripts/run_vast.sh` | Rented | Full experiment on Vast.ai |

---

## 6. Dependencies

```bash
# Core (already installed)
uv pip install torch transformers datasets marisa-trie networkx

# Demo-specific
uv pip install streamlit pyvis
```
