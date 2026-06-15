# TypeOracle Experiment Guide & Results

## Overview

This document covers all experiments for the symbolic TypeOracle (DCA-Trie Approach 3):
what can run on this hardware, what needs a GPU, the results so far, and commentary.

## Hardware Landscape

| Hardware | Available | What it can run |
|----------|-----------|----------------|
| **CPU** (11th Gen i7, 4 cores) | ✅ | TypeOracle SIR/FNR on full test set (~2 min) |
| **Intel Iris Xe** (integrated GPU, 6 GB free) | ✅ | llama.cpp with Q4_K_M models (Qwen2.5-3B) via Vulkan |
| **NVIDIA GPU** (16 GB+ VRAM) | ❌ | Full GCR pipeline with 8B constrained-decoding model |

**What this means:**

- TypeOracle gate evaluation (SIR, FNR) is pure CPU and runs instantly
- Small model inference via llama.cpp works (~5-15s per generation on Intel GPU)
- The full DCA-Trie v1 constrained-decoding pipeline needs an NVIDIA GPU with 16 GB+
  to load `rmanluo/GCR-Meta-Llama-3.1-8B-Instruct` via HuggingFace Transformers

---

## Unified Script

All experiments are run through a single script:

```
workflow/run_symbolic_experiment.py
```

### Commands

```bash
# Phase 1: SIR/FNR on full test set (CPU, 2 min)
python workflow/run_symbolic_experiment.py --phase sir

# Phase 2: Proxy answer generation via llama.cpp (Intel GPU)
# First start the server:
llama-server --hf-repo Qwen/Qwen2.5-3B-Instruct-GGUF \
  --hf-file qwen2.5-3b-instruct-q4_k_m.gguf -ngl 999 --port 8080

# Then run (10 questions, compares filtered vs unfiltered):
python workflow/run_symbolic_experiment.py --phase proxy --n 10

# Both phases:
python workflow/run_symbolic_experiment.py --phase all --n 10
```

### Output

Results are saved to `results/symbolic_experiment/` as JSON files with timestamps.

---

## Phase 1: TypeOracle SIR / FNR (CPU)

Evaluates how many candidate paths the two symbolic gates prune, and whether
any gold-truth paths are incorrectly blocked.

### Results (WebQSP, 1,628 test questions)

| Metric | Value |
|--------|-------|
| Total candidate paths (raw) | 4,102,833 |
| Total paths after filtering | 3,509,451 |
| **SIR (overall)** | **14.5%** |
| SIR_type (type gate) | 10.6% (435,897 blocked) |
| SIR_traj (range gate) | 3.8% (157,485 blocked) |
| **Type gate FNR** | **3.3%** (490 / 14,829 gold paths) |
| **Range gate FNR** | **2.9%** (424 / 14,829 gold paths) |

---

## Phase 2: Proxy Model Answer Generation (llama.cpp / Intel GPU)

**Caveat: this is NOT the DCA-Trie pipeline.** The real v1 uses token-level
constrained decoding via a trie (generated tokens are forced to form valid paths
from the filtered set). This experiment instead dumps paths into a text prompt
and asks Qwen2.5-3B to extract the answer — a fundamentally weaker approach.

Unsurprisingly, with ~2,000-4,000 Freebase paths (each with arcane relation names
like `location.statistical_region.gender_balance_members_of_parliament`) in
the prompt, the 3B model struggles. Results on 3 questions showed 0/3 Hit@1
even though the correct paths (`Jamaica → location.country.languages_spoken →
Jamaican English`) survived TypeOracle filtering.

**Takeaway:** Prompt-based path listing is not a substitute for constrained
decoding. The proxy model is useful for testing model-serving infrastructure
but cannot evaluate TypeOracle's impact on generation quality.

---

## Full Pipeline (NVIDIA GPU required)

The real DCA-Trie v1 experiment uses `GraphConstrainedDecodingModel` from HF
Transformers with token-level trie constraints. This requires:

- NVIDIA GPU with 16 GB+ VRAM (A100 40GB recommended)
- The GCR model: `rmanluo/GCR-Meta-Llama-3.1-8B-Instruct`
- HuggingFace token (gated model)

### Command

```bash
python workflow/predict_symbolic_dca_trie.py \
  --dca_mode v1 \
  --d RoG-webqsp \
  --split "test" \
  --index_path_length 2 \
  --model_name gcr-Llama-2-7b-chat-hf \
  --model_path rmanluo/GCR-Meta-Llama-3.1-8B-Instruct \
  --prompt_mode zero-shot \
  --k 5 \
  --force
```

### Quick dev run (10 questions, ~10 min on A100)

Same command with `--split "test[:10]"`.

### What this measures

| Output | Meaning |
|--------|---------|
| Hits@1 (with filtering) | Generation accuracy *with* TypeOracle |
| SIR, SIR_type, SIR_traj | How much the oracle prunes the path set |
| Paths before/after | Raw filtering statistics |
| Kept path samples | Examples of admitted paths |

Compare Hits@1 against the unfiltered GCR baseline (no TypeOracle) to measure
the impact of symbolic pruning on answer quality. The baseline is produced by
`notebooks/01_GCR_Baseline.ipynb` or the corresponding headless workflow.

---

## Commentary

### 1. TypeOracle works: 14.5% pruning at <3.3% FNR

The two symbolic gates eliminate ~593K spurious paths while blocking only
~3% of gold-truth paths. This is a strong result for a purely symbolic
(stdlib-only, no-GPU) oracle.

### 2. The type gate dominates pruning (10.6% vs 3.8%)

The answer-type gate does ~3× the work of the range gate. This matches
expectations: the most common failure mode is paths terminating at the wrong
entity type. A single Freebase type lookup catches most of these.

### 3. The proxy experiment is not a proxy for DCA-Trie

Feeding paths into a prompt is not the same as constrained decoding.
The DCA-Trie forces the model to generate tokens along valid paths from the
trie — the model never even sees invalid options. Prompt-based path listing
overwhelms the model with noise. The 0/3 hit rate says nothing about
TypeOracle's effectiveness in the actual constrained-decoding pipeline.

### 4. What's missing (needs NVIDIA GPU)

- **Generation comparison**: Does TypeOracle-filtered decoding improve Hits@1
  over unfiltered GCR? This is the central question and cannot be answered
  without running `predict_symbolic_dca_trie.py` on a GPU.
- **v2 evaluation**: The step-wise Algorithm 2 may have different beam dynamics.
- **CWQ**: ComplexWebQuestions (4-hop) evaluation.
- **FNR impact**: When the oracle blocks a gold path (3.3% of cases), how often
  does the LLM still produce the correct answer via a different path?

### 5. SIR/FNR numbers are stable and ready for the thesis

The Phase 1 results are deterministic (no random seed, no threshold) and
computed on the full 1,628-sample WebQSP test set. These numbers can be
reported with confidence in Chapter 3.

---

## Experimental Record

| Date | Experiment | Samples | Key result |
|------|-----------|---------|------------|
| 2026-06-04 | SIR/FNR (CPU) | 1,628 | SIR=14.5%, FNR_type=3.3%, FNR_range=2.9% |
| 2026-06-04 | Proxy answer gen (llama.cpp) | 3 | 0/3 Hit@1 (prompt-based, not DCA-Trie) |
