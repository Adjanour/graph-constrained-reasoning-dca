# DCA-Trie: Symbolic Constraint Oracles for Faithful KG Reasoning

**Author:** Bernard [Surname] — BSc Computer Science, University of Mines and Technology (UMaT)

**Supervisor:** [Supervisor Name]

**Repository:** `github.com/bernard-research/graph-constrained-reasoning`

---

## What This Project Does

Large language models hallucinate on knowledge-graph question answering (KGQA).
They generate fluent reasoning paths that don't exist in the knowledge graph.

The standard fix — **constrained decoding** — builds a prefix tree (trie) of all
structurally valid paths and masks out any token that would lead outside it.
But for a 3-hop question on Freebase, this means **up to 24,000 valid paths**
compete at every decoding step. The LLM still needs to choose among thousands of
valid-but-irrelevant options using its parametric knowledge — the same mechanism
that causes hallucinations in the first place.

**DCA-Trie (Dynamic Context-Aware Trie)** replaces the loose structural constraint
with a tight **symbolic oracle** that uses the KG's own ontology to prune irrelevant
paths before they reach the decoder. Two set-containment checks — an answer type
gate and a property range gate — remove paths whose entity types or relation ranges
don't match the question. No embeddings, no thresholds, no GPU.

---

## The Core Insight

The KG already stores entity types (`common.topic.notable_types`) and relation
ranges (`rdf-schema#range`) as RDF triples alongside domain facts. This metadata
directly answers "should this path be in the constraint set?" — it is free signal
that prior methods (GCR, DoG) ignored.

Every relation in Freebase declares its expected object type. Every notable entity
has a type annotation. These are ground-truth, machine-readable, and require no
learned component to exploit.

---

## Three Approaches, One Pipeline

The project tracks three generations of the oracle design, each in its own
standalone directory:

### Approach 1 — Cosine Similarity (`approach1_cosine/`)
A single monolithic score `cos(E(path), E(question)) ≥ τ` using all-MiniLM
sentence embeddings. A threshold determines admission.

**Problems:** Threshold-dependent, encoder cost scales with candidate set size,
no type awareness, embedding noise introduces non-determinism.

### Approach 2 — Decomposed Product (`approach2_decomposed/`)
Factor the relevance score into three components: `ρ_r · ρ_e · ρ_traj`.
Each component scores relation relevance, entity relevance, and trajectory
coherence independently, with encoder caching to reduce redundant compute.

**Problems:** Still threshold-dependent, still needs a GPU encoder, still
no explicit type constraint.

### Approach 3 — Symbolic TypeOracle (`approach3_symbolic/`)
Two deterministic set-containment gates operating on KG ontology:

| Property | Cosine | Decomposed | **Symbolic** |
|----------|--------|------------|--------------|
| Encoder needed | Yes | Yes | **No** |
| Threshold τ | Yes (tuned) | Yes (tuned) | **No** |
| GPU needed | Encode+decode | Encode+decode | **Decode only** |
| Type awareness | Implicit in cosine | Implicit in components | **Explicit** |
| Range awareness | None | None | **Explicit** |
| Deterministic | No | No | **Yes** |

---

## The Two Symbolic Gates

### Gate 1 — Answer Type Gate (terminal hop only)
```
Does the terminal entity's type match what the question asks for?
  types(e_term) ∩ answer_types(q) ≠ ∅
```
- Question asks "who?" → only admit entities typed "Person"
- Question asks "where?" → only admit entities typed "Location"
- Conservative fallback: if type info missing, admit by default

### Gate 2 — Property Range Gate (every hop)
```
Does this relation's declared range match the entity it connects to?
  types(tail_entity) ∩ range(relation) ≠ ∅
```
- Relation `people.person.place_of_birth` has range `{"Location", "Country", ...}`
- If a path connects it to an entity typed "Film", block it
- Same conservative fallback: admit by default when schema info is missing

Both gates are O(1) set lookups. No floating point, no neural network, no GPU.

---

## Formal Guarantees (Machine-Checked in Lean 4)

A complete formal proof (3298 jobs, zero errors, zero warnings) is at
`proof/formal_proof/FormalProof.lean` using Lean 4 with mathlib v4.30.0-rc2.

**Theorem 1 (v1 Faithfulness):** Every token produced by DCA-Trie v1 is a
verified member of the knowledge graph. The symbolic filter is a subset of BFS,
the trie is built from the filtered set, so any token it admits traces back to
a valid KG triple.

**Theorem 2 (v2 Faithfulness):** DCA-Trie v2's step-wise expansion preserves
structural faithfulness by induction. The initial trie contains only question
entities. Each expansion step checks `(e_t, r, e') ∈ G` as a hard membership
test, and the symbolic gates only reduce the admitted set.

**Theorem 3 (Monotonicity Chain):** `P_filtered ⊆ P_typed ⊆ P_range ⊆ P_GCR`.
Every path admitted by the symbolic oracle is also in GCR's path set — no false
positives introduced.

**Theorem 4 (FNR Union Bound):** The probability that a gold path is excluded
is bounded by the sum of per-gate failure probabilities. Each gate fails only
when both the question/relation provides a type signal AND the entity has a
known incompatible type.

---

## Code Structure

```
graph-constrained-reasoning/
├── approach1_cosine/
│   └── notebooks/          # 01_GCR_Baseline, 02_DCA_Trie_v1, 03_DCA_Trie_v2, 04_SIR_Evaluation
├── approach2_decomposed/
│   └── notebooks/
├── approach3_symbolic/
│   ├── notebooks/
│   ├── type_oracle.py      # Standalone symbolic oracle (stdlib only)
│   └── algo_demo.py        # Self-contained algorithm demonstration
├── MENTAL_MODEL.md          # First-principles explanation of the ecosystem
├── FINAL_SOLUTION.md        # One-pager for supervisor review
├── EXPERIMENT_GUIDE.md      # Setup guide for Colab and local runs
├── chapter3_updated.md      # Full methodology chapter
├── PROJECT_OVERVIEW.md      # This file
└── proof/formal_proof/FormalProof.lean  # Machine-checked Lean 4 proof
```

**Key file: `type_oracle.py`** (150 lines, stdlib only) — the entire symbolic
oracle implementation. Type map extraction from subgraph triples, answer type
inference from question text, and both gates. Run it standalone with any
Freebase subgraph; no dependencies beyond Python 3.10+.

---

## How to Run

### Quick sanity check (10 minutes, any GPU with 16GB)
```bash
cd approach3_symbolic/notebooks
# Set MAX_SAMPLES=10, K=2, QUANT=8bit
# Run 01_GCR_Baseline.ipynb → 02_DCA_Trie_v1.ipynb → 04_SIR_Evaluation.ipynb
```

### Full evaluation (1-2 hours, A100 40GB recommended)
Same notebooks with `MAX_SAMPLES=100`, `K=5`, full precision.

The model (`rmanluo/GCR-Meta-Llama-3.1-8B-Instruct`) is gated — set
`HF_TOKEN` in Colab secrets or environment.

See `EXPERIMENT_GUIDE.md` for detailed Colab and local setup instructions.

---

## Results (Preliminary, 100-dev WebQSP)

The type gate alone prunes ~1100 paths per question using pure ontology lookups.
Cosine similarity at τ=0.25 produced 84% empty tries (all paths rejected) — the
symbolic gates have no such failure mode because they are conservative by design.

Full results (Hits@1, F1, SIR decomposed by type/trajectory, average trie size
per step) for WebQSP and CWQ will be reported in Chapter 4.

---

## Relation to Existing Work

| Work | Constraint Basis | Type-Aware? | Threshold? | Encoder? |
|------|-----------------|-------------|------------|----------|
| GCR (Luo et al., 2025) | Graph topology | No | No | No |
| DoG (Li et al., 2025) | Graph topology | No | No | No |
| ToG (Sun et al., 2024) | Prompt-based | No | No | No |
| **DCA-Trie (this thesis)** | **Graph topology + KG ontology** | **Yes** | **No** | **No** |

DCA-Trie is the first constraint oracle to exploit the KG's ontological schema
directly, without encoding it through a learned representation.

---

## What the Lean Proof Covers

The formal proof (`FormalProof.lean`, 371 lines) defines:

- **Core types:** `Entity`, `Relation`, `Triplet`, `KGPath`, `KG`, `EntityType`, `TypeSet`
- **Symbolic gates:** `typeGate`, `rangeGate`, `admissible`
- **Axioms for:** BFS groundedness, trie lookup soundness, entity types, relation ranges, inferred answer types
- **Filter operations:** `applyTypeFilter`, `applyRangeFilter`, `applySymbolicFilter`
- **Monotonicity lemma:** symbolic filter ⊆ type filter ⊆ range filter ⊆ original set
- **v1/v2 faithfulness theorems:** both variants preserve KG grounding
- **FNR union bound:** `excluded_set.card ≤ type_fail_set.card + range_fail_set.card`

Build verified with `lake build` — 3298 jobs, zero errors.

---

## Contact

For questions, access requests, or collaboration:

**Bernard [Surname]**
[email]
Department of Computer Science and Engineering
University of Mines and Technology, Tarkwa
