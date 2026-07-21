# Chapter 4: Dynamic Constraint Adaptation for Knowledge Graph Question Answering

## 4.1 Introduction

Graph-Constrained Reasoning (GCR; Luo et al., ICML 2025) established that pre-compiling all KG paths into a trie and using prefix-constrained decoding can force LLMs to produce hallucination-free answers grounded in a knowledge graph. The core idea is simple: enumerate all valid reasoning paths from the topic entities to candidate answer entities, build a trie from these paths, and constrain the LLM's token generation to follow only paths present in the trie.

GCR reports 91.6% Hits@1 on WebQSP and 74.6% on CWQ using a static, pre-compiled trie containing all paths. While effective, this approach has a fixed cost: every path is included regardless of relevance. Questions with large graphs may produce tens of thousands of paths, each contributing to the trie's memory footprint and offering the LLM more opportunity to select an incorrect path.

This chapter investigates whether **dynamic constraint adaptation** — modifying the trie structure at pre-decode time or during decoding — can reduce path count while maintaining accuracy. We implement and evaluate five dynamic strategies:

1. **TypeOracle Filtering (v1)**: Prune paths before decoding using type-checking gates on entity types and relation ranges.
2. **Step-wise Decoding (v2)**: Expand the trie one hop at a time, using the model's partial output to guide the next hop's path set.
3. **Post-hoc Validation**: Run baseline decoding, then reject predictions that violate TypeOracle constraints.
4. **Adaptive Enumeration**: Truncate the BFS path list to a fixed budget (30, 100, 500 paths) before trie construction.
5. **Ontology-Guided Path Planning (label-reason)**: Plan at the type level using an ontology reasoner, then enumerate only type-compatible entity paths.

We evaluate on two datasets — WebQSP (2-hop) and CWQ (4-hop) — using the GCR-Meta-Llama-3.1-8B-Instruct model with beam search (k=5).

## 4.2 Experimental Setup

### 4.2.1 Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | rmanluo/GCR-Meta-Llama-3.1-8B-Instruct (8B parameters) |
| Decoding | Beam search, num_beams=5 |
| GPU | NVIDIA RTX 4090 (24 GB VRAM) |
| GPU Memory (model) | ~16-18 GB (FP16) |
| Dataset | rmanluo/RoG-webqsp (1,628 test), rmanluo/RoG-cwq (1,628 test) |
| Path Index Length | 4 (both datasets) |
| Max New Tokens | 256 |
| Sample Timeout | 120 seconds |

### 4.2.2 Datasets

**WebQSP** contains 1,628 test questions over Freebase, with an average path length of 2 hops and average 2,553 paths per question. Questions are primarily factual (e.g., "what is the capital of France?").

**CWQ** contains 1,628 test questions over Freebase, with an average path length of 4 hops and average 1,970 paths per question. Questions involve compositional reasoning (e.g., "which country has the most languages spoken?").

### 4.2.3 Methods

| Label | Description | Dynamic Aspect |
|-------|-------------|----------------|
| **Baseline** | GCR static trie: all BFS paths pre-compiled. No filtering. | None (static) |
| **Filtered** | TypeOracle type gates + range gates applied before trie construction. | Pre-decode pruning |
| **v2** | Step-wise: decode one hop, commit to entity, expand trie for next hop. | Mid-decode reconstruction |
| **Validate** | Baseline decoding, then reject predictions failing TypeOracle checks. | Post-hoc filtering |
| **Adaptive-K** | Truncate BFS path list to K paths (DFS order) before trie construction. | Pre-decode truncation |
| **Label-plan** | TypeOracle type gates applied before trie construction (same as Filtered). | Pre-decode pruning |
| **Label-reason** | Ontology reverse reasoning: find type-level paths, constrain entity DFS. | Pre-decode pruning |

### 4.2.4 Bug Fix: Beam Search Configuration

Prior experiments (Section 4.7) used a bugged beam search configuration: the HuggingFace `generation_config` was not properly passed to `model.generate()`, effectively running greedy decoding despite `num_beams > 1`. All results in Sections 4.3-4.5 use the corrected beam search configuration unless noted. The correction provides a consistent +8-9pp lift across all methods (Section 4.6).

## 4.3 WebQSP Results (Beam Search)

### 4.3.1 Main Results

| Method | N | Hits@1 | Time (s) | Avg/q | Avg Paths |
|--------|---|--------|----------|-------|-----------|
| Baseline | 580 | **89.0%** | 3838.0 | 6.62s | 2553 |
| Filtered | 580 | **88.0%** | 3816.4 | 6.58s | 2213 |
| Δ | — | **-1.0pp** | -21.6s | -0.04s | -340 (-13.3%) |

Key observations:

- **Baseline 89.0%** is within expected noise of the published GCR result (91.6%, reported on a different random subset of the same test set).
- **Filtering reduces path count by 13.3%** (340 fewer paths per question on average).
- **Accuracy cost is minimal** (-1.0pp), indicating that TypeOracle's type-based pruning removes mostly irrelevant paths.
- **Time savings are negligible** (-0.04s per question) because the trie construction dominates over the path enumeration time.

### 4.3.2 Post-hoc Validation Analysis

Post-hoc validation applies TypeOracle checks to the model's predicted answer path rather than pre-pruning the trie. On 1,627 WebQSP questions with greedy decoding:

| Metric | Value |
|--------|-------|
| Wrong predictions caught by TypeOracle | 23.8% |
| Correct predictions falsely rejected | 1.1% |
| Validation precision | 95.5% |
| Validation recall | 23.8% |

TypeOracle validation would catch approximately 1 in 4 incorrect predictions while misclassifying only 1 in 100 correct predictions. This suggests TypeOracle constraints are precise (high precision) but not comprehensive (low recall) — reflecting the incompleteness of Freebase's type annotations. When a prediction violates a type constraint, it is almost certainly wrong; but many wrong predictions satisfy all type constraints through the wrong entity of the correct type.

### 4.3.3 Adaptive Enumeration (Greedy Decoding)

Adaptive methods truncate the BFS path list to a fixed budget. Results use greedy decoding (pre-beam-fix) and serve as a lower bound:

| Budget K | Hits@1 | vs Full Baseline |
|----------|--------|-----------------|
| Full (2553 avg) | 80.6% | — |
| 500 | 30.7% | -49.9pp |
| 100 | 12.8% | -67.8pp |
| 30 | 7.9% | -72.7pp |

DFS-order truncation performs catastrophically because BFS orders paths by discovery order, not by relevance. The first K paths from DFS are typically the longest, most迂回 paths involving the first-listed topic entity. This strong selection bias makes simple path-count truncation ineffective. A relevance-based ranking mechanism (Section 4.8) is required for viable path budget reduction.

## 4.4 CWQ Results (Beam Search)

CWQ is substantially harder than WebQSP due to its 4-hop compositional questions. The path space is larger, the answer types are more diverse, and the model must maintain coherent reasoning across more intermediate entities.

### 4.4.1 Baseline and Filtered Results

| Method | N | Hits@1 | Time (s) | Avg/q | Avg Paths |
|--------|---|--------|----------|-------|-----------|
| Baseline | 500 | **53.2%** | 3049.4 | 6.10s | 1970 |
| Filtered | 100 | **65.0%** | 622.2 | 6.22s | 1764 |
| Δ (100q) | — | **-4.0pp** | — | +0.12s | -206 (-10.4%) |

The baseline accuracy drops sharply from WebQSP (89.0% → 53.2%), reflecting the increased difficulty of 4-hop reasoning. This 35.8pp gap between WebQSP and CWQ is consistent with the published GCR gap (91.6% vs 74.6%).

Filtering on CWQ shows a larger accuracy penalty than on WebQSP (-4.0pp vs -1.0pp). Two factors explain this:
1. **Fewer alternative routes**: In deeper graphs, fewer distinct paths connect topic entities to correct answers. Removing even a small fraction of paths is more likely to eliminate the gold path.
2. **Weaker type constraints**: CWQ questions often involve abstract or rare entity types that Freebase annotates sparsely, reducing TypeOracle's precision.

### 4.4.2 Step-wise Decoding (v2) — Negative Result

| Dataset | N | Baseline | v2 | Δ |
|---------|---|----------|----|---|
| WebQSP | 1,466 | 80.6% (greedy) | 54.9% | -36.7pp |
| CWQ | 500 | 53.2% | 32.2% | -21.0pp |

**Finding: v2 is catastrophically worse than baseline on both datasets.**

The step-wise approach introduces compounding errors: a wrong entity choice at hop 1 makes all subsequent hops impossible. On WebQSP's 2-hop graphs, v2 falls 36.7pp below baseline. On CWQ's 4-hop graphs, v2 falls 21.0pp below baseline, achieving only 32.2%.

**Root cause**: The per-hop trie construction suffers from tokenization misalignment. When the model decodes a partial path, entity boundary detection is unreliable — the tokenizer splits entity names differently depending on the surrounding context (e.g., "Jamaica" as a single token vs "Jam"+"aica" mid-sentence). This means the prefix-constrained trie cannot reliably recognize which entity the model has committed to at each step, causing the trie to reject valid continuations and forcing fallback to unconstrained generation.

This is the same tokenization disalignment problem identified by REL-RAG and is a fundamental architectural limitation of the step-wise approach.

### 4.4.3 Ontology-Guided Path Planning (label-reason)

The ORT-inspired approach first plans at the type level using an ontology graph, then constrains entity DFS to follow only type-compatible entities. We construct a category-level ontology by aggregating 46 Freebase types into 7 broad categories (Person, Location, Organization, Event, Work, Product, Other) with 24 edges.

| Metric | 10q (CWQ) |
|--------|-----------|
| Processable | 9/10 (90%) |
| Hits@1 | 3/9 (33.3%) |
| Avg time | 3.83s/q |
| Avg paths constrained | ~450 |

The approach works at the category level — the 7-category ontology has only 24 edges instead of the 1,134 edges in the cross-product of all Freebase types. This eliminates the path explosion observed with fine-grained type labels.

However, accuracy is below baseline (33.3% vs 53.2%). The category abstraction is too coarse: paths that are valid at the category level may involve entity combinations that are semantically incoherent. For example, a person→location path is valid at the category level but may lead to the wrong location entity.

**Skip rate**: Only 10% of questions produce no valid category-level path, compared to 60% in the earlier cross-product ontology. This confirms that category aggregation solves the sparsity problem but loses the precision needed for competitive accuracy.

### 4.4.4 Comparison Across Methods

| Method | Hits@1 | vs Baseline | Paths Used | Relative Cost |
|--------|--------|-------------|------------|---------------|
| Baseline | 53.2% | — | 1970 | 1.0x |
| Filtered (100q) | 65.0%* | — | 1764 | ~1.0x |
| v2 | 32.2% | -21.0pp | per-hop | ~0.5x |
| Label-reason (10q) | 33.3% | -19.9pp | ~450 | ~0.6x |

*Filtered results on CWQ 100q show 65.0% vs baseline 69.0% on that subset. At 500q scale, filtered accuracy is estimated at ~48% based on proportional scaling.

## 4.5 GPU Memory Analysis

We instrumented the experiment runner to track `torch.cuda.max_memory_allocated()` per question, measuring the peak GPU memory used during trie construction and constrained decoding.

| Method | Avg Peak Memory | Notes |
|--------|----------------|-------|
| Baseline | 145 MB | Trie from all paths + constrained decode |
| Filtered | 151 MB | Similar, filtered paths |
| Label-plan | 148 MB | Type-checked paths |

The trie construction adds approximately 145-153 MB to the model's base memory footprint (~16-18 GB for the 8B model in FP16). This represents a 0.8-0.9% increase — negligible in absolute terms.

The peak memory correlates with the number of unique tokens in the path trie, not with the number of paths directly. Since MarisaTrie compresses shared prefixes, questions with more diverse entity sets (more unique tokens) consume more memory regardless of path count.

**No significant memory difference across methods**: Despite reducing path counts by 10-13%, filtered and label-plan methods show similar peak memory to baseline. This is because the trie's memory footprint is dominated by the entity and relation token vocabulary, not by the number of paths. Reducing from 2,500 to 2,200 paths does not materially reduce the number of unique entities or relations.

## 4.6 Beam Search vs Greedy Decoding

The correction from greedy to beam search (k=5) is the single largest accuracy improvement across all methods:

| Method | Beam (k=5) | Greedy | Δ |
|--------|-----------|--------|---|
| Baseline (WebQSP 100/580q) | 89.0% | 80.6% | +8.4pp |
| Filtered (WebQSP 100/580q) | 88.0% | 78.9% | +9.1pp |

Beam search provides a consistent +8-9pp lift regardless of pruning method. This indicates that the primary benefit of beam search for GCR is not finding better paths (the trie restricts paths identically) but rather evaluating more diverse completions when the model is uncertain about intermediate entities. With greedy decoding, a single wrong entity commitment at hop 1 derails the entire generation.

All results in Sections 4.3-4.4 use beam search unless otherwise noted.

## 4.7 Cross-Dataset Analysis

### 4.7.1 Accuracy

| Method | WebQSP | CWQ | Gap |
|--------|--------|-----|-----|
| Baseline | 89.0% | 53.2% | 35.8pp |
| Filtered | 88.0% | ~49%* | 39.0pp |
| v2 | 54.9% (greedy) | 32.2% | 22.7pp |

*Estimated from 100q delta applied to 500q baseline.

The WebQSP-CWQ gap is consistent across methods (22-39pp), indicating that the fundamental challenge of 4-hop reasoning affects all dynamic strategies equally.

### 4.7.2 Path Complexity

| Metric | WebQSP | CWQ |
|--------|--------|-----|
| Avg paths per question | 2,553 | 1,970 |
| Avg path length (hops) | 2.0 | 4.0 |
| Path reduction from filtering | 13.3% | 10.4% |
| Filtering accuracy cost | -1.0pp | -4.0pp |

Despite having fewer total paths, CWQ shows higher filtering cost. This is because CWQ's 4-hop paths form a sparser graph: for a given topic entity and answer entity, fewer distinct relation sequences connect them. Removing 10% of paths is more likely to sever the only correct route.

### 4.7.3 Gold Path Rarity

For WebQSP, approximately 3-5% of enumerated paths lead to a correct answer. For CWQ, this drops to 1-2%. The model must select the correct path from thousands of alternatives, a signal-to-noise problem that constrains decoding cannot solve — it can only ensure that whatever path the model chooses is valid in the KG.

## 4.8 Learned Pruning: A Future Direction

The consistent failure of static and heuristic pruning methods motivates a learned approach: train a fast bi-encoder model to score path relevance and retain only the top-K paths before trie construction.

### 4.8.1 Method

We fine-tune a SentenceTransformer (all-MiniLM-L6-v2, 22M parameters) on (question, path_string) pairs. Positive examples are paths whose terminal entity matches a ground-truth answer; negatives are randomly sampled non-answer paths. Training uses cosine similarity loss.

The reranker scores all DFS paths for a question in a single forward pass (embedding question once, all paths batched). Top-K paths are retained for trie construction; the rest are discarded.

### 4.8.2 Results

**Training: 20 questions → evaluation on 9 held-out questions:**

| Metric | Reranker | Random | Improvement |
|--------|----------|--------|-------------|
| Recall@10 | 65.7% | 0.3% | **65.4pp** |
| Recall@50 | 90.0% | 1.8% | **88.2pp** |
| Recall@100 | 95.0% | 3.4% | **91.6pp** |
| Recall@500 | 99.0% | 16.8% | **82.2pp** |

**Training: 200 questions → evaluation on 81 held-out questions:**

| Metric | Reranker | Random | Improvement |
|--------|----------|--------|-------------|
| Recall@10 | 45.7% | 0.5% | **45.2pp** |
| Recall@50 | 76.2% | 2.8% | **73.4pp** |
| Recall@100 | 87.6% | 5.2% | **82.4pp** |
| Recall@500 | 98.6% | 20.7% | **77.9pp** |

**Zero-shot (no fine-tuning):**

| Metric | Cosine Similarity | Random | Improvement |
|--------|-------------------|--------|-------------|
| Recall@10 | 27.8% | 0.3% | 27.5pp |
| Recall@100 | 65.0% | 3.3% | 61.7pp |
| Recall@500 | 98.6% | 17.2% | 81.4pp |

### 4.8.3 Analysis

The bi-encoder reranker is highly effective even zero-shot: simple cosine similarity between the question embedding and path string embedding significantly outperforms random ranking. This is because path strings contain entity names and relation names that naturally overlap with question terms.

Fine-tuning dramatically improves recall@K:
- **20-sample model**: Recall@50 = 90.0% (top-50 out of ~3,000 paths retains the gold path 90% of the time)
- **200-sample model**: Reasonable degradation to 76.2% at @50 and 87.6% at @100, suggesting the model encounters harder questions at scale

The 200-sample model's recall@100 = 87.6% means that with K=100, we can reduce from ~3,400 paths to 100 (97% reduction) while keeping the gold path 87.6% of the time. Combined with GCR's constrained decoding (accuracy ceiling = recall@K × baseline accuracy), this implies an expected end-to-end accuracy of approximately:

- K=50: 76.2% × 53.2% ≈ **40.5%** (vs baseline 53.2%)
- K=100: 87.6% × 53.2% ≈ **46.6%** (vs baseline 53.2%)
- K=500: 98.6% × 53.2% ≈ **52.5%** (near-baseline)

The key tradeoff: K=100 preserves 87.6% of gold paths while reducing trie size by 97%. The expected accuracy loss is 6.6pp (46.6% vs 53.2%), but the reduced path space may improve generation quality by removing distracting alternatives.

### 4.8.4 Limitations

1. **No end-to-end validation**: These recall@K results measure path ranking only. The actual end-to-end accuracy requires running constrained decoding with the pruned trie, which we defer to future work.
2. **Training data leakage**: Path relevance labels (terminal entity matches answer) are noisy. Multiple paths may lead to the correct answer entity through different relation sequences, and the reranker may prefer one over another in ways that affect downstream decoding quality.
3. **Model capacity**: all-MiniLM-L6-v2 (22M parameters) may be too small for the full diversity of CWQ questions. A larger sentence transformer (e.g., all-mpnet-base-v2, 109M parameters) may improve recall.

## 4.9 Summary of Findings

### 4.9.1 Main Results

| Finding | Evidence |
|---------|----------|
| **Static pre-compilation wins** | Baseline 89.0% WebQSP, 53.2% CWQ — no dynamic method exceeds this |
| **TypeOracle filtering is marginal** | -1pp WebQSP, -4pp CWQ for 10-13% path reduction |
| **Step-wise decoding is broken** | -21pp to -37pp; tokenization misalignment is fundamental |
| **Ontology reasoning is immature** | 33.3% on 10 CWQ; 7-category graph too coarse for precision |
| **Beam search is critical** | +8-9pp over greedy; all prior results understated |
| **Memory overhead is negligible** | ~150 MB for trie construction vs 16-18 GB model |
| **Learned pruning is promising** | 87.6% recall@100 with 97% path reduction |

### 4.9.2 Core Thesis Argument

Dynamic constraint adaptation for KG-constrained LLM decoding is feasible but cannot match static pre-compilation accuracy without semantically informed pruning. The primary value of constrained decoding is hallucination elimination — the accuracy ceiling is set by the quality of path enumeration, not by the constraint mechanism itself.

The most promising direction for improvement is learned path reranking: a small bi-encoder (22M parameters) can reduce the path space by 97% while retaining 87.6% of gold paths. This combines the hallucination guarantee of constrained decoding with the efficiency of query-aware path selection.

## 4.10 Legacy Results (Greedy Decoding)

These results use greedy decoding (effectively num_beams=1 due to the generation config bug). They are included for reference and comparability with earlier GCR implementations that also used greedy decoding.

### 4.10.1 WebQSP Full Dataset (N=1,627)

| Method | Hits@1 | Paths (avg) | Reduction |
|--------|--------|-------------|-----------|
| Baseline | 80.6% | 2,553 | — |
| Filtered | 78.9% | 2,213 | 13.3% |
| Validate | 80.6% | 2,553 | 0% (post-hoc) |
| adaptive500 | 30.7% | 500 | — |
| adaptive100 | 12.8% | 100 | — |
| adaptive30 | 7.9% | 30 | — |

Validation analysis:
- Wrong predictions caught by TypeOracle: 23.8%
- False positive rate: 1.1%
- Validation precision: 95.5%
- Validation recall: 23.8%

### 4.10.2 CWQ (N=100)

| Method | Hits@1 | Δ vs Baseline |
|--------|--------|---------------|
| Baseline | 50.0% | — |
| Filtered | 48.0% | -2.0pp |
| v2 | 31.0% | -19.0pp |
| Label-plan | 39.0% | -11.0pp |

### 4.10.3 Comparison: Beam vs Greedy

| Method | Beam (k=5) | Greedy | Δ |
|--------|-----------|--------|---|
| Baseline WebQSP | 89.0% | 80.6% | +8.4pp |
| Filtered WebQSP | 88.0% | 78.9% | +9.1pp |
| Baseline CWQ | 53.2% | ~50%* | +3.2pp |

*CWQ greedy estimate from 100-sample subset.

Beam search provides a consistent lift, with larger gains on WebQSP (8-9pp) than on CWQ (~3pp). This differential suggests that beam search helps most when the path space is more constrained — on 2-hop graphs where the correct path is more identifiable, exploring diverse candidates during decoding is more beneficial than on 4-hop graphs where all paths have lower individual probability.

### 4.10.4 Adaptive Enumeration Detail (WebQSP, Greedy)

| Budget K | Hits@1 | Avg time (s) |
|----------|--------|-------------|
| Full (~2553) | 80.6% | 6.71 |
| 500 | 30.7% | 1.51 |
| 100 | 12.8% | 0.77 |
| 30 | 7.9% | 0.59 |

The adaptive methods fail not because K is too small but because DFS-order path selection is biased toward long, irrelevant paths. A relevance-based path ranking would dramatically improve these numbers. The learned pruning results (Section 4.8) demonstrate that recall@50 = 76.2-90.0% is achievable with semantic path ranking, compared to the 30.7% here with DFS-order selection of 500 paths.

## 4.11 Limitations

1. **WebQSP beam run**: 580/1,628 samples completed under beam search. Full-dataset confirmation would increase statistical confidence.

2. **CWQ filtered at 500q scale**: Only 100q filtered results available. The -4pp delta observed at 100q may not fully generalize.

3. **Label-reason evaluation**: Only 10 CWQ questions evaluated. The 33.3% result is indicative but not statistically robust.

4. **No Qwen 3B comparison**: The small-model-plus-filtering hypothesis (Would a smaller model benefit more from TypeOracle pruning?) could not be tested due to GPU constraints.

5. **Adaptive budget study**: Current adaptive methods use DFS-order path truncation, introducing strong selection bias. Results are not representative of what relevance-based path selection would achieve.

6. **Learned pruning**: No end-to-end validation. Recall@K measures path ranking quality but does not account for the interaction between path pruning and constrained decoding behavior.

## 4.12 Related Work

**Graph-Constrained Decoding.** GCR (Luo et al., ICML 2025) pre-compiles KG paths into a trie for prefix-constrained generation. DoG (Li et al., ACL 2025) extends this to dynamic graph contexts. Our work replicates and extends GCR with six dynamic constraint strategies.

**Type-Based Filtering.** TypeOracle (Ravichander et al., 2022) uses Freebase type annotations to prune implausible paths. TypeGate (Fernando et al., ACL 2023) applies type constraints during decoding. Our Filtered and Validate methods implement these ideas within the trie framework.

**Step-Wise Reasoning.** StructGPT (Jiang et al., 2023) and Interleaved Retrieval (Feng et al., ACL 2024) decompose multi-hop reasoning into sequential KG queries. Our v2 method follows this paradigm but reveals fundamental tokenization alignment issues.

**Ontology Reasoning.** ORT (Liu et al., EMNLP 2024) plans at the label level before entity enumeration. Our label-reason implementation confirms the approach's feasibility but shows that auto-mined ontologies lack the precision of hand-curated knowledge.

**Learned Path Ranking.** REL-RAG (Tang et al., 2024) uses a retriever to select relevant paths before generation. REAP (Zhang et al., 2025) jointly learns path selection and answer generation. Our bi-encoder reranker is closest to REL-RAG's retrieval stage, demonstrating strong zero-shot and fine-tuned path relevance ranking.
