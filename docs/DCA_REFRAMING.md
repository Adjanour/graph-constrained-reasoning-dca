# DCA Reframing: What We Actually Built

## Original Claim

*"Dynamic Context-Aware Tries improve KGQA by filtering semantically irrelevant paths from the KG-Trie, reducing search space without losing accuracy."*

**Status: Disproven for WebQSP.** TypeOracle prunes 14.5% of paths but accuracy drops 5.2%. The ontology is too noisy; false positives kill the benefit.

---

## What We Actually Have

Three independent contributions that were bundled under a misleading title:

### 1. Semantic Irrelevance Ratio (SIR) — A Diagnostic Metric

**What it is:** A model-agnostic measure of path quality. For any set of KG paths rooted at a question's topic entities, SIR decomposes irrelevance into two independent failure modes:

- **SIR_type**: paths whose terminal entity type mismatches the expected answer type
- **SIR_traj**: paths containing a relation whose declared range is incompatible with the answer type

**Why it matters:** Existing KGQA evaluations treat all structurally valid paths as equally useful. SIR shows they're not — ~14.5% of valid KG paths lead to type-wrong answers. This is a property of the KG itself, not the model.

**Publishable as:** A short analysis paper or extended abstract — "Semantic Irrelevance in Knowledge Graph Paths for Question Answering."

### 2. TypeOracle — A Validation Tool (Not a Filter)

**What it is:** A symbolic classifier using Freebase type information (entity types from `notable_types`, relation domain/range from schema + auto-mining) to determine whether a path's terminal entity is type-compatible with a question.

**What it's good for:**
- **Post-hoc validation**: "Should I trust this generated answer?" (Idea 1 — check predictions, don't prune paths)
- **Diagnostic profiling**: "What fraction of wrong answers are type-plausible vs. type-impossible?"
- **Not good for**: Pre-decoding path pruning (false positives lose accuracy)

**Finding:** TypeOracle's false positive rate (rejecting correct predictions) is ~1.1%, but the catch rate (flagging wrong predictions) is only ~25%. The signal exists but is too weak to improve accuracy through pre-filtering.

### 3. DCA-v2 — Step-Wise Constrained Decoding

**What it is:** Instead of building one trie with all paths (GCR/v1), v2 builds a small trie per hop. At each step, the model generates one entity-to-entity transition, the terminal entity is committed, and a new trie is built from its neighbors.

**What it actually does differently from GCR/v1:**

| Aspect | GCR / v1 | DCA-v2 |
|--------|----------|--------|
| Trie construction | BFS all paths, one trie | DFS per-hop, one trie per step |
| Path exploration | Beam search across all paths simultaneously | Greedy entity commitment per step |
| Trie size | TotalPaths × MaxHops | AvgBranchingFactor (per hop) |
| LLM calls | 1 | MaxHops |
| Branching factor | Managed by beam search | Managed by TypeOracle per-hop |

**Where it could matter:** On CWQ (3-4 hops), the total path count can explode (thousands of paths). v1's single trie becomes enormous. v2's per-hop trie stays at the average branching factor (tens, not thousands). The model trades beam-search parallelism for a fe  w sequential decisions — each with a much smaller search space.

**Untested claim:** On CWQ, v2 achieves competitive accuracy with significantly less memory and potentially lower latency than v1, because per-hop trie construction avoids the exponential path explosion.

---

## Summary: What Papers Could Come Out of This

| Paper | Content | Difficulty | Impact |
|-------|---------|------------|--------|
| **SIR diagnostic analysis** | "14.5% of valid KG paths lead to wrong types; here's how to measure it" | Low — data analysis | Medium — fills a gap |
| **Post-hoc validation** | "TypeOracle rejects 25% of wrong preds with 1.1% false positives" | Low — clean experiment | Low-medium |
| **Step-wise decoding on CWQ** | "When path count explodes, per-hop trie construction keeps decoding tractable" | Medium — needs CWQ results | Medium-high if it beats v1 |

All three are real contributions that don't require the filtering narrative. The mistake was claiming filtering would *improve* accuracy — it doesn't. But it reveals something about the structure of KG paths that's worth knowing.
