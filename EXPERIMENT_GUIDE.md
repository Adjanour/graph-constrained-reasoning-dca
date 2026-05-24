# Experiment Guide: Running DCA-Trie

## Hardware Requirements

| Setup | GPU | VRAM | Speed (100 questions) |
|-------|-----|------|-----------------------|
| **Colab Pro** | A100 40GB | ✓ | ~15 min with flash-attn |
| **Colab Free** | T4 16GB | ✓ (8bit) | ~45 min |
| **Local** | A100/H100 40GB | ✓ | Same as Colab Pro |
| **Local** | Any 24GB+ | ✓ (8bit) | Slower, depends on GPU |

The Llama-3.1-8B model needs ~16GB at 16-bit precision. A100 40GB is ideal.
T4 16GB works with `QUANT=8bit`.

---

## Option A: Colab (Recommended)

1. Open the notebook in Colab:
   - <https://colab.research.google.com/github/your-repo/...>

2. Runtime → Change runtime type → **A100 GPU**

3. Run cell-by-cell from top. Each notebook is self-contained:
   - Installs dependencies
   - Downloads model from HuggingFace (requires HF token for gated models)
   - Loads dataset from HuggingFace
   - Runs inference
   - Saves predictions to `results/GenPaths/...`

### HF Token

GCR's Llama-3.1-8B checkpoint may be gated. To access it:

1. Accept the license at https://huggingface.co/rmanluo/GCR-Meta-Llama-3.1-8B-Instruct
2. Set your token in the notebook: `os.environ["HF_TOKEN"] = "hf_..."`  
   Or add to Colab secrets (🔑 icon in left sidebar).

---

## Option B: Local Run

### 1. Clone the repo

```bash
git clone https://github.com/rmanluo/graph-constrained-reasoning.git
cd graph-constrained-reasoning
```

### 2. Install dependencies

```bash
pip install transformers==4.44.2 accelerate peft deepspeed \
  tiktoken datasets python-dotenv marisa-trie bitsandbytes \
  trl sentencepiece protobuf wandb networkx scikit-learn

# A100 only:
pip install flash-attn --no-build-isolation
```

Or with Conda:

```bash
conda create -n dca python=3.10
conda activate dca
pip install -r requirements.txt   # if available
```

### 3. Set HF token

```bash
export HF_TOKEN="hf_your_token_here"
```

### 4. Run notebooks

```bash
jupyter notebook notebooks/01_GCR_Baseline.ipynb
```

Or convert to scripts and run headless:

```bash
jupyter nbconvert --to script notebooks/01_GCR_Baseline.ipynb --output gcr_baseline
python gcr_baseline.py
```

---

## Notebook Execution Order

| Order | Notebook | What it does | Expected output |
|-------|----------|-------------|-----------------|
| 1 | `01_GCR_Baseline.ipynb` | Run GCR baseline (no filtering) | Baseline Hits@1, F1, SIR |
| 2 | `02_DCA_Trie_v1.ipynb` | Static symbolic filtering (v1) | v1 Hits@1, path reduction |
| 3 | `03_DCA_Trie_v2.ipynb` | Step-wise symbolic expansion (v2) | v2 Hits@1, dynamic stats |
| 4 | `04_SIR_Evaluation.ipynb` | Decomposed SIR metric | SIR_type, SIR_traj breakdown |

Run **01 first** (it produces the baseline predictions), then 02–03 in any order
(they each produce their own predictions), then 04 to evaluate all of them.

---

## Configuration

All experiment settings are in each notebook's **Configuration cell** (cell 2):

| Setting | Typical value | Notes |
|---------|--------------|-------|
| `MODEL_PATH` | `rmanluo/GCR-Meta-Llama-3.1-8B-Instruct` | The GCR checkpoint |
| `DATASET` | `RoG-webqsp` or `RoG-cwq` | WebQSP or ComplexWebQuestions |
| `SPLIT` | `test` | Test split for evaluation |
| `INDEX_LEN` | `2` (WebQSP) or `4` (CWQ) | Max BFS hop depth |
| `K` | `5` | Beam size |
| `GEN_MODE` | `group-beam` | Beam search mode |
| `PROMPT_MODE` | `zero-shot` | Prompt template |
| `MAX_NEW_TOKENS` | `256` | Max generated tokens |
| `MAX_SAMPLES` | `100` for dev, `None` for full | Number of questions to process |
| `QUANT` | `False` (A100), `True` (T4) | 8-bit quantization to save VRAM |
| `ATTN_IMPL` | `flash_attention_2` (A100) or `sdpa` | Attention backend |

### Important flags

- `FORCE = True` — re-run even if output exists (set `False` to resume)
- `MAX_SAMPLES = None` — full dataset (use for final results; takes hours)
- `MAX_SAMPLES = 100` — quick dev run (use for debugging)

For **CWQ** (4-hop questions), increase `MAX_NEW_TOKENS = 512` and `INDEX_LEN = 4`.

---

## What Each Notebook Measures

| Notebook | Key outputs | How to interpret |
|----------|-------------|-----------------|
| 01 (GCR) | Hits@1, F1, faithfulness | Baseline. ~89% dev, ~92.6% reported |
| 02 (v1) | Hits@1, paths before/after, type/range blocked | How much does filtering reduce the path set? |
| 03 (v2) | Hits@1, expansion stats per hop | Does step-wise expansion help? |
| 04 (SIR) | SIR*, SIR*_type, SIR*_traj | Which failure mode dominates? |

Lower SIR* = tighter constraint = less permissive oracle = better.

---

## Output Format

Predictions are saved as JSONL:

```json
{
  "id": "WebQSP_42",
  "question": "What is the nationality of...",
  "prediction": "United States [PATH] Blue Hawaii | film.director | ...",
  "ground_truth": ["United States"],
  "dca_paths_before": 5321,
  "dca_paths_after": 187,
  "dca_type_blocked": 92,
  "dca_range_blocked": 7
}
```

Each notebook appends to `results/GenPaths/{dataset}/{model}/test/{config}/predictions.jsonl`.

---

## Three Approach Folders

| Folder | Notebook style | Notes |
|--------|---------------|-------|
| `approach1_cosine/` | Cosine similarity | Has τ, has encoder. Run on CPU if needed |
| `approach2_decomposed/` | Decomposed product | ρ_r · ρ_e · ρ_traj. Run on CPU if needed |
| `approach3_symbolic/` | Symbolic TypeOracle | **No encoder, no τ.** Fastest option |

For **final thesis results**, use `approach3_symbolic/` (the symbolic approach).
Approaches 1–2 are historical reference points.

---

## Quick Dev Run (10 min, any GPU)

```bash
# In any notebook's config cell, set:
MAX_SAMPLES = 10
K = 2
FORCE = True

# Then run all cells. This produces a quick sanity check:
# - Does the model load? (✓)
# - Does the trie build? (✓)
# - Does inference produce output? (✓)
# - Are paths being filtered? (check total_paths_before vs total_paths_after)

# For the full experiment, set:
MAX_SAMPLES = None   # processes all questions (WebQSP = 1,632, CWQ = ~3500)
K = 5
# Expect 1-2 hours for WebQSP, 3-4 for CWQ on A100
```

## Troubleshooting

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| `CUDA out of memory` | Not enough VRAM | Enable 8bit: add `QUANT = True` in config |
| `flash-attn` install fails | No A100 | Switch to `ATTN_IMPL = "sdpa"` (auto-fallback) |
| `load_in_8bit` kwarg error | Transformers version wrong | Pin `transformers==4.44.2` |
| Model not loading | HF token missing | Set `HF_TOKEN` env var or add to notebook |
| `ModuleNotFoundError: src` | Wrong working directory | Run from `graph-constrained-reasoning/` folder |
| `No module named 'type_oracle'` | Missing import path | `sys.path.insert(0, 'notebooks')` first |
