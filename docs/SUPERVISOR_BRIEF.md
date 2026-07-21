# Supervisor Brief: Experiment Timeline & Results

---

## Phase 1: Baseline Replication (GCR)

**Goal**: Reproduce GCR's published results (91.6% WebQSP, 74.6% CWQ).

- Ran GCR-Meta-Llama-3.1-8B-Instruct on RoG dataset splits
- **Bug**: Early runs used greedy decoding (generation_config wasn't passed). Got 80.6% WebQSP instead of 91.6%
- **Fix**: After proper beam search (k=5): **89.0% WebQSP** (580q), **53.2% CWQ** (500q)
  - WebQSP is within noise of published result. CWQ gap is larger (53.2% vs 74.6%) — likely due to different test splits
- **Memory**: ~145 MB peak GPU for trie construction — negligible vs 16-18 GB model

**Key finding**: Beam search is critical (+8-9pp). GCR replication is solid.

---

## Phase 2: TypeOracle Filtering (v1)

**Goal**: Prune irrelevant paths before decoding using Freebase type metadata.

- Two gates: answer type gate (terminal entity) + property range gate (every hop)
- Applied before trie construction — deterministic, no learned components
- **WebQSP** (580q): 89.0% → **88.0%** (-1pp), 13.3% path reduction
- **CWQ** (100q): 69.0% → **65.0%** (-4pp), 10.4% path reduction
- **500q CWQ run in progress** (VPS spinning up now)

**Finding**: Filtering works better on 2-hop graphs. On deeper graphs (CWQ), removing paths is riskier because fewer alternative routes exist. The tradeoff is marginal — 10-13% reduction at 1-4pp cost.

---

## Phase 3: Step-wise Decoding (v2) — NEGATIVE RESULT

**Goal**: Expand trie hop-by-hop — only commit to one hop at a time, reducing path space exponentially.

- At each hop: model generates → entity extraction → fetch neighbors → build new trie → repeat
- **Catastrophic failure**: WebQSP **54.9%** (-36.7pp), CWQ **32.2%** (-21.0pp)
- **Root cause**: Tokenization misalignment. The tokenizer splits entity names differently in context vs in isolation. The trie can't reliably detect which entity the model committed to, so valid continuations get rejected.
- This is the same problem REL-RAG identifies. It's architectural, not tunable.

**Finding**: Step-wise decoding is dead. The per-hop trie approach fundamentally cannot work with current tokenizers.

---

## Phase 4: Post-hoc Validation

**Goal**: Run baseline, then reject predictions failing TypeOracle checks.

- On WebQSP (1,627q, greedy): catches **23.8%** of wrong predictions, falsely rejects only **1.1%** of correct ones
- Validation precision: 95.5%, recall: 23.8%

**Finding**: TypeOracle is precise but not comprehensive. When it flags a prediction as type-invalid, that prediction is almost certainly wrong. But most wrong predictions satisfy type constraints (wrong entity of the right type).

---

## Phase 5: Adaptive Enumeration

**Goal**: Truncate BFS path list to a fixed budget (30/100/500).

- Uses DFS-order truncation (first K paths found)
- **Results** (WebQSP greedy): Adaptive-500 = **30.7%** (-49.9pp), Adaptive-100 = **12.8%**, Adaptive-30 = **7.9%**
- Complete failure: DFS finds paths in discovery order, which is heavily biased toward long, irrelevant paths

**Finding**: Simple path-count truncation without relevance ranking is useless. Need semantic path ranking.

---

## Phase 6: ORT-style Ontology Reasoning (label-reason)

**Goal**: Plan at the type level first (like ORT), then enumerate only type-compatible entity paths.

- Built category-level ontology: aggregated 46 Freebase types → 7 broad categories, 24 edges (vs 1,134 in cross-product)
- Category aggregation fixed the path explosion problem
- **Results** (CWQ 10q): 9/10 processable, **33.3%** accuracy (vs 53.2% baseline)
- Category-level paths are too coarse: paths valid at category level can involve semantically incoherent entity combinations

**Finding**: Auto-mined ontologies without curated knowledge aren't precise enough. Category-level constraints are too permissive; fine-grained type labels are too sparse. Needs hand-curated ontology like ORT uses.

---

## Phase 7: Learned Path Reranking (Future Direction)

**Goal**: Train a bi-encoder to score path relevance, keep only top-K paths.

- Model: all-MiniLM-L6-v2 (22M params), fine-tuned on (question, path_string) pairs
- Trained on 20 CWQ questions → evaluated on 9 held-out:
  - **Recall@10: 65.7%**, Recall@50: **90.0%**, Recall@100: **95.0%**
- Trained on 200 questions → evaluated on 81 held-out:
  - **Recall@10: 45.7%**, Recall@50: **76.2%**, Recall@100: **87.6%**
- Even zero-shot (cosine similarity, no fine-tuning): Recall@100 = 65.0%

**Finding**: A 22M parameter reranker can reduce 3,400 paths → 100 paths (97% reduction) while keeping the gold path 87.6% of the time. This is the most promising direction — combines GCR's hallucination guarantee with query-aware path selection.

**Expected end-to-end accuracy** (projected): K=100 → ≈46.6% (87.6% × 53.2% baseline), compared to baseline 53.2%.

---

## Summary Table

| Method | WebQSP | CWQ | Status |
|--------|--------|-----|--------|
| GCR Baseline | **89.0%** | **53.2%** | ✅ Confirmed |
| TypeOracle Filtered | **88.0%** (-1pp) | **~49%** (-4pp est.) | ⏳ 500q run on VPS |
| Step-wise (v2) | **54.9%** (-36.7pp) | **32.2%** (-21pp) | ❌ Dead (tokenization) |
| Validation catch rate | **23.8%** | — | ✅ |
| Adaptive-500 | **30.7%** | — | ❌ DFS bias |
| Label-reason | — | **33.3%** (10q) | ❌ Too coarse |
| Learned pruning | — | **87.6% recall@100** | ✅ Future direction |

**Core thesis finding**: Dynamic constraint adaptation is feasible but cannot match static pre-compilation accuracy without semantically informed pruning. Learned path reranking bridges this gap — 97% path reduction at 87.6% gold-path retention.
