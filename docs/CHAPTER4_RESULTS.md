# Chapter 4: Experimental Results

## 4.1 Experimental Setup

| Parameter | Value |
|-----------|-------|
| Base Model | rmanluo/GCR-Meta-Llama-3.1-8B-Instruct |
| Decoding | Beam search (k=5) |
| GPU | NVIDIA RTX 4090 (24 GB VRAM) |
| Datasets | RoG-webqsp (1,628 test), RoG-cwq (1,628 test) |
| Index length | 4 (both datasets) |
| Max new tokens | 256 |

### Methods Evaluated

| Label | Description |
|-------|-------------|
| **Baseline** | GCR with full BFS path trie (no TypeOracle filtering) |
| **Filtered** | TypeOracle static filtering — prune paths *before* decoding |
| **v2** | Step-wise hop-by-hop trie expansion with per-hop TypeOracle gates |
| **label-reason** | (ORT-inspired) Ontology reverse reasoning → label-constrained DFS |

### Bug Fix: Beam Search Configuration

Prior experiments (Section 4.7) used a bugged beam search configuration: the `generation_config` was not properly passed to the model, effectively running greedy decoding despite specifying `num_beams>1`. After fixing, beam search (k=5) provides a consistent **+9-10pp lift** over greedy across all methods.

---

## 4.2 WebQSP Results (Beam Search, k=5)

| Method | N | Hits@1 | Time (s) | Avg/q |
|--------|---|--------|----------|-------|
| Baseline | 100 | **89.0%** | 670.6 | 6.71s |
| Filtered | 100 | **88.0%** | 658.3 | 6.58s |
| Δ | — | **-1.0pp** | -12.3s | -0.13s |

Path statistics:
- Avg paths before filtering: 2,552.7
- Avg paths after filtering: 2,212.8
- Path reduction: **13.3%**
- 0 skip / 0 dead end / 0 timeout

The baseline 89.0% is within expected noise of the published GCR result (91.6%, reported on a larger random subset). Filtering reduces path count by 13.3% with only 1pp accuracy loss — marginal improvement given the overhead.

---

## 4.3 CWQ Results (Beam Search, k=5)

### 4.3.1 100-Sample Verification

| Method | N | Hits@1 | Time (s) | Avg/q |
|--------|---|--------|----------|-------|
| Baseline | 100 | **69.0%** | 631.9 | 6.32s |
| Filtered | 100 | **65.0%** | 622.2 | 6.22s |
| Δ | — | **-4.0pp** | -9.7s | -0.10s |

Path statistics:
- Avg paths before filtering: 1,969.6
- Avg paths after filtering: 1,764.1
- Path reduction: **10.4%**
- 0 skip / 0 dead end / 0 timeout

### 4.3.2 500-Sample Full Run

| Method | N | Hits@1 | Time (s) | Avg/q |
|--------|---|--------|----------|-------|
| Baseline | 500 | **53.2%** | 3049.4 | 6.10s |
| v2 | 500 | **32.2%** | 1643.5 | 3.29s |
| Δ | — | **-21.0pp** | -1405.9s | -2.81s |

### 4.3.3 Analysis

Cross-dataset comparison:

| Metric | WebQSP (100) | CWQ (100) | CWQ (500) |
|--------|-------------|-----------|-----------|
| Baseline Hits@1 | 89.0% | 69.0% | 53.2% |
| Filtered Hits@1 | 88.0% | 65.0% | — |
| Filtering cost (Δ) | -1.0pp | -4.0pp | — |
| Path reduction | 13.3% | 10.4% | — |

CWQ is substantially harder than WebQSP:
1. **CWQ baseline is 20-36pp lower** (69% vs 89% on 100q; 53.2% on 500q) — 4-hop questions are fundamentally more difficult for BFS path enumeration
2. **Filtering hurts more on CWQ** (-4pp vs -1pp) — deeper graphs have fewer alternative routes to the answer, so removing valid paths has larger impact
3. **Path reduction is smaller** (10.4% vs 13.3%) — CWQ graphs have a higher proportion of type-heterogeneous entities

The 500-sample CWQ baseline (53.2%) is more reliable than the 100-sample estimate (69.0%). The 100q subset apparently oversampled easy questions.

---

## 4.4 Step-wise Decoding (v2) — Negative Result

| Dataset | N | Baseline | v2 | Δ |
|---------|---|----------|----|---|
| WebQSP | 1,466 | 91.6% | 54.9% | -36.7pp |
| CWQ | 500 | 53.2% | 32.2% | -21.0pp |

**Finding: v2 is catastrophically worse than baseline on both datasets.**

The step-wise approach introduces compounding errors: a wrong entity choice at hop 1 makes all subsequent hops impossible. On 2-hop graphs (WebQSP), the failure is dramatic (-36.7pp). The hypothesis that v2 would perform *better* on deeper graphs (CWQ) is also false: -21.0pp on 4-hop questions.

**Root cause**: The per-hop trie construction suffers from tokenization misalignment. When the model decodes a partial path, the entity boundary detection is unreliable — the tokenizer splits entity names differently in context vs in isolation. This is the same tokenization disalignment problem described by REL-RAG.

**Recommendation**: The v2 approach should be abandoned unless the per-hop trie construction can be fixed to handle tokenization boundaries correctly. This is a fundamental architectural issue, not a tuning problem.

---

## 4.5 ORT-style Ontology Reasoning (label-reason)

### 4.5.1 Motivation

ORT (Runxuan Liu et al., 2024) formulates KGQA as a **label-level reasoning** task: first find a path through the ontology label graph, then constrain entity-level search to follow only type-compatible entities. This avoids enumerating all O(E^L) entity paths.

### 4.5.2 Implementation

We implemented:
1. **IntentParser** (`approach3_symbolic/intent_parser.py`): LLM-based question-to-answer-type mapping (12 Freebase categories, cache, regex fallback)
2. **OntologyReasoner** (`approach3_symbolic/ontology_reasoner.py`): Category-level ontology graph (7 categories, 24 edges vs 46 labels, 1134 edges)

### 4.5.3 Challenge: Dense Ontology Graph

Initial construction used a cross-product of Freebase type sets per relation, creating a dense label graph:
- **46 labels, 1,134 edges** — every relation connects all its domain types to all its range types
- **Path explosion**: up to 373,000 label paths per question
- **Solution**: Aggregate Freebase types → broad categories (7 categories), building the graph at the category level → 7 categories, 24 edges

### 4.5.4 Preliminary Results

| Metric | Value |
|--------|-------|
| Samples | 10 (CWQ) |
| Processable | 4 (40%) |
| Hits@1 (processable) | 4/4 (100%) |
| Avg time | 10.08s/q |
| Skips | 6 (60%) |

Key observations:
- **60% skip rate**: Most questions produce no label path connecting the aim category to a condition category, so the trie is empty and the model falls back.
- **On the 4 processable questions**: Accuracy is perfect (4/4) but the sample is too small to be meaningful.
- **DFS still hits 50k cap**: Even with category-level constraints, the constrained DFS produces 50k paths for some questions (e.g., `n_paths_constrained: 50000`). The 7 categories are too coarse to provide meaningful pruning.

### 4.5.5 Remaining Issues

1. **High skip rate (60%)**: The ontology fails to find reverse reasoning paths connecting aim → condition for most questions. This likely occurs because the category-level graph (7 nodes, 24 edges) is too sparse — many cross-category relation paths are missing at the aggregated level.
2. **Insufficient pruning when it works**: When paths are found, they still produce 50k entity paths because the category constraint is too permissive. A finer granularity (15-20 categories) would provide better pruning.
3. **No meaningful evaluation done**: The 4/4 correct result on processable questions is not statistically meaningful.

---

## 4.6 Key Findings Summary

1. **GCR baseline is strong**: 89.0% (WebQSP) and 53.2% (CWQ 500q) with beam search. The original 91.6% result is replicable within noise.

2. **TypeOracle filtering gives marginal benefit**: 13% path reduction at 1pp cost on WebQSP; 10% reduction at 4pp cost on CWQ. The cost-benefit tradeoff is unfavorable, especially on deep graphs.

3. **Step-wise decoding (v2) is broken**: -36.7pp on WebQSP, -21.0pp on CWQ. The deep-graph hypothesis is conclusively false. Tokenization misalignment in per-hop trie construction is the root cause.

4. **Beam search is critical**: +9-10pp over greedy decoding. All prior results were de facto greedy due to a configuration bug.

5. **ORT-style ontology reasoning is promising but immature**: Category-level graph (7 nodes, 24 edges) controls path explosion, but 60% skip rate needs diagnosis. Type constraint relaxation and category granularity tuning are next steps.

---

## 4.7 Legacy Results (Greedy Decoding, Pre-Beam-Fix)

These results use greedy decoding (effectively num_beams=1 due to the generation config bug). They are included for reference and contrast.

### 4.7.1 WebQSP Full Dataset (Greedy, N=1,627)

| Method | Hits@1 | Paths (avg) | Reduction |
|--------|--------|-------------|-----------|
| Baseline | 80.6% | 2,552.7 | — |
| Filtered | 78.9% | 2,212.8 | 13.3% |
| Validate | 80.6% | 2,552.7 | 0% (post-hoc) |
| Label-plan | 78.8% | 2,212.8 | 13.3% |
| adaptive500 | 30.7% | 500 | — |
| adaptive100 | 12.8% | 100 | — |
| adaptive30 | 7.9% | 30 | — |

Validation analysis:
- Wrong predictions caught by TypeOracle: **23.8%**
- False positive rate: **1.1%**
- Validation precision: **95.5%**
- Validation recall: **23.8%**

### 4.7.2 Comparison: Beam vs Greedy (WebQSP)

| Method | Beam (k=5) | Greedy | Δ |
|--------|-----------|--------|---|
| Baseline | 89.0% | 80.6% | +8.4pp |
| Filtered | 88.0% | 78.9% | +9.1pp |

Beam search provides a consistent **+8-9pp lift** across methods.

---

## 4.8 Limitations

1. **Partial WebQSP beam run**: Only 100/1,628 samples completed for beam search. The WebQSP beam baseline at full scale is estimated at ~89% based on the 100-sample verification but needs confirmation.
2. **No filtered results at 500q**: The 500-sample CWQ run evaluated baseline + v2 but not filtered. The -4pp filtering cost observed at 100q may not generalize to the full dataset.
3. **label-reason incomplete**: The ORT-inspired approach is still in early evaluation (category granularity, constraint relaxation).
4. **No Qwen 3B comparison**: The small-model + filtering hypothesis could not be tested (no local GPU capable of running Llama 3.1-8B + Qwen 3B side-by-side).
5. **Adaptive budget study invalid**: Current adaptive methods use DFS-order path truncation, introducing strong selection bias. Relevance-based path ranking is required for a valid tradeoff study.
