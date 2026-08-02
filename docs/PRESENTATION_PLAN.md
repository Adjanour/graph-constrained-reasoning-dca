# Presentation & Demo Plan

## Overview

Presentation flow: **Talk → Manim Visualization → Live UI Demo → Results**

1. Talk: What we set out to do, what the project is, how we did it
2. Manim animation: Animated walkthrough of KG-constrained decoding
3. Live UI demo: Side-by-side comparison (Normal LLM / RAG / DCA-Trie)
4. Results: Hits@1, F1, SIR comparison

---

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Animation | ManimCE + manim-slides | Pre-rendered, reliable, reveal.js export |
| UI Framework | Gradio | Purpose-built for ML demos, minimal code, clean layout |
| LLM Backend | Google Gemini API (free tier) | No credit card, 250 RPD, good quality |
| Alternative LLM | Ollama (local) | Fully offline, no API dependency |

### Gemini API Free Tier Limits

- Gemini 2.5 Flash: 10 RPM, 250,000 TPM, 250 RPD
- No credit card required
- More than enough for curated demo (5-6 questions × 3 methods = 18 requests)

---

## Components

### 1. Manim Animation (Highest Priority)

**Goal:** Visually explain how DCA-Trie works for the presentation

**Scenes to animate:**

| Scene | Content | Duration |
|-------|---------|----------|
| 1 | Knowledge graph as a graph (nodes + edges) | 10s |
 2 | GCR builds static trie — show all paths including irrelevant ones highlighted in red | 15s |
| 3 | SIR problem — show irrelevant paths wastes compute | 10s |
| 4 | DCA-Trie v2 — step-by-step pruning as tokens are generated, paths fade out | 20s |
| 5 | Final comparison — GCR trie size vs DCA-Trie trie size | 10s |

**Workflow:**
1. Write scenes as ManimCE `Slide` subclasses (manim-slides)
2. Render: `manim-slides render scenes.py SceneName`
3. Present: `manim-slides Scene1 Scene2 Scene3 Scene4 Scene5`
4. Export: reveal.js HTML for browser-based presentation

**Dependencies:**
- `manim` (ManimCE)
- `manim-slides`

**Status:** Not started

---

### 2. Gradio UI Demo

**Goal:** Live comparison showing constrained decoding beats normal LLM and RAG

**Layout:**
```
┌─────────────────────────────────────────────────────┐
│  DCA-Trie: Dynamic Context-Aware Constrained Decoding │
├─────────────────────────────────────────────────────┤
│  Question: [input]  [Run Comparison]                │
├─────────────────────────────────────────────────────┤
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐    │
│  │ Normal   │ │   RAG    │ │  DCA-Trie        │    │
│  │   LLM    │ │          │ │  Constrained     │    │
│  │          │ │          │ │                  │    │
│  │ answer   │ │ answer   │ │ answer + path    │    │
│  └──────────┘ └──────────┘ └──────────────────┘    │
│                                                     │
│  KG Path Visualization (optional)                   │
└─────────────────────────────────────────────────────┘
```

**Backend:**
- Normal LLM: Gemini API call, no context
- RAG: Retrieve top-K KG triples, inject into prompt
- DCA-Trie: Run constrained decoding pipeline (our implementation)

**Dependencies:**
- `gradio`
- `google-generativeai` (Gemini SDK)
- DCA-Trie pipeline code

**Status:** Not started

---

### 3. Curated Demo Questions

**Goal:** Pick 5-6 WebQSP questions that showcase the difference

**Criteria:**
- Multi-hop (2-3 hops) to show trie pruning benefit
- Questions where normal LLM hallucinates or gives wrong answer
- Questions where DCA-Trie path is clearly correct
- Mix of entity types (people, places, events)

**Source:** WebQSP test set, filtered by hop depth from notebook 06

**Status:** Not started

---

### 4. Full Evaluation (Objective v)

**Goal:** Complete Hits@1, F1 comparison for paper results section

**What's ready:**
- `experiments/type_oracle_full/main.py` — supports `--method all --datasets RoG-webqsp RoG-cwq`
- Notebook 05.5 — clean experiment notebook

**What's needed:**
- Run `uv run python experiments/type_oracle_full/main.py --method all --datasets RoG-webqsp RoG-cwq --max-samples 50`
- Collect Hits@1, F1, structural faithfulness, trie size
- Stratify by hop depth

**Status:** Not started

---

### 5. Wire Up Backend

**Goal:** Connect Gradio UI to real pipelines

**Flow:**
1. User enters question in Gradio
2. Backend calls Gemini for Normal LLM
3. Backend calls Gemini with RAG context
4. Backend runs DCA-Trie constrained decoding
5. Display all three results side-by-side

**Status:** Blocked by #2, #3

---

## Objective Status (UMaT Template)

| # | Objective | Status |
|---|-----------|--------|
| i | Define & measure SIR | ✅ Done |
| ii | Semantic relevance scoring (pivoted to TypeOracle) | ✅ Done |
| iii | DCA-Trie v1 (static, FNR < 5%) | ✅ Done |
| iv | DCA-Trie v2 (dynamic) | ✅ Done |
| v | Evaluate vs GCR + CoT (Hits@1, F1, faithfulness, SIR) | 🔲 Pending |
| vi | Interactive prototype (UI demo) | 🔲 Pending |

---

## Timeline

| Day | Task |
|-----|------|
| 1 | Manim scenes 1-3 (KG, static trie, SIR problem) |
| 2 | Manim scenes 4-5 (DCA-Trie pruning, comparison) |
| 3 | Gradio UI scaffold + Gemini integration |
| 4 | Curated questions + wire up DCA-Trie pipeline |
| 5 | Full evaluation run (objective v) |
| 6 | Polish, test, rehearse |
| 7 | Presentation |

---

## Risks

| Risk | Mitigation |
|------|------------|
| Manim rendering slow | Use `-pql` (low quality) during dev, `-pqh` for final |
| Gemini API rate limits | Curated questions only, max 18 requests per demo |
| DCA-Trie pipeline errors | Test with fixed questions beforehand |
| Gradio layout issues | Keep UI simple, minimal customization |

---

## Demo Walkthrough

Two ways to present the project: a **CLI step-by-step walkthrough** and a **Streamlit web UI**. Each has a replay mode (no GPU) and a live mode (needs GPU).

### 1. CLI Step-by-Step Demo (`step_by_step.py`)

Interactive terminal walkthrough that pauses at each pipeline stage. Designed for narrating during a presentation.

#### Replay Mode (No GPU)

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

#### Live Mode (Needs GPU)

Loads the actual model and runs inference.

```bash
# All methods, default beam size
uv run python demo/step_by_step.py --live --method all

# V2 only, beam size 5
uv run python demo/step_by_step.py --live --method v2 --beam-size 5

# Specific question
uv run python demo/step_by_step.py --live --question-idx 42 --method all
```

### 2. Streamlit Web UI (`app.py`)

Browser-based visualization with interactive KG graph.

#### Pre-computed Mode (No GPU)

```bash
uv run streamlit run demo/app.py
```

Or use the convenience script:

```bash
bash run_demo.sh
```

Opens at `http://localhost:8501`. Select a question from the sidebar, click through the pipeline steps. The KG is rendered as an interactive network graph.

#### Live Mode (Needs GPU)

The Streamlit app also has a "Live" mode toggle in the sidebar. When enabled, it loads the model and runs inference in the browser. Requires a GPU.

### 3. Renting a GPU on Vast.ai

For live demos, rent a GPU on Vast.ai.

#### Automated (Recommended)

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

#### Manual

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
   uv run python experiments/type_oracle_full/main.py --datasets RoG-webqsp --max-samples 50

   # Download results
   exit
   scp -r root@<instance_ip>:~/graph-constrained-reasoning/results ./results
   ```

4. **Destroy:**
   ```bash
   vastai destroy instance <instance_id>
   ```

### 4. Regenerating Demo Data

If you run a new experiment and want the pre-computed demo to use those results:

```bash
# Generate from experiment predictions
uv run python demo/generate_demo_data.py \
    --n-samples 10 \
    --predictions-dir results/final_experiment-<timestamp>/final_experiment/<run>/RoG-webqsp \
    --output-dir demo/demo_data
```

### 5. Quick Reference

| Command | GPU? | Purpose |
|---------|------|---------|
| `uv run python demo/step_by_step.py` | No | CLI walkthrough (replay) |
| `uv run python demo/step_by_step.py --live` | Yes | CLI walkthrough (live) |
| `uv run streamlit run demo/app.py` | No | Web UI (pre-computed) |
| `bash run_demo.sh` | No | Web UI shortcut |
| `bash scripts/run_vast.sh` | Rented | Full experiment on Vast.ai |

### 6. Dependencies

```bash
# Core (already installed)
uv pip install torch transformers datasets marisa-trie networkx

# Demo-specific
uv pip install streamlit pyvis
```
