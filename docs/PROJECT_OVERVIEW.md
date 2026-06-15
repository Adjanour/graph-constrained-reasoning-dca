# DCA-Trie: Symbolic TypeOracle for Faithful Knowledge Graph Reasoning

**Author:**

**Supervisor:** [Supervisor Name]

**Repository:**

---

## 1. The Problem

Large language models hallucinate on knowledge graph question answering (KGQA).
KG-constrained decoding methods (GCR, DoG) build a prefix tree of all structurally
valid paths and mask out invalid tokens at each decoding step. But for a 3-hop
question on Freebase, this trie contains up to **24,000 valid paths** — all
structurally sound, only one correct. The LLM must choose among thousands of
valid-but-irrelevant options using its parametric knowledge, the same mechanism
that produces hallucinations in the first place. The constraint oracle is too
permissive: structural validity is necessary but not sufficient. Embedding-based
attempts to tighten the oracle (cosine similarity, decomposed scoring) introduce
threshold dependence, non-determinism, and GPU overhead while still failing to
exploit the KG's own ontological metadata.

---

## 2. The Solution: Symbolic TypeOracle

The KG already stores entity types (`common.topic.notable_types`) and relation
ranges (`rdf-schema#range`) as RDF triples alongside domain facts. This
metadata directly answers "should this path be in the constraint set?" — it is
free signal that prior methods ignored. The Symbolic TypeOracle replaces all
embedding-based scoring with two deterministic gates that operate on this
ontological schema:

**Gate 1 — Answer Type Gate** (applied at the terminal hop only): checks whether
the terminal entity's type matches what the question asks for. If the question
asks "who?", only entities typed `Person` are admitted. Formally:
`types(e_term) ∩ answer_types(q) ≠ ∅`. This is O(1) set intersection.

**Gate 2 — Property Range Gate** (applied at every hop): checks whether the
relation's declared range is compatible with the tail entity's type. If a path
connects `people.person.place_of_birth` to an entity typed `Film`, it is blocked.
Formally: `types(tail_entity) ∩ range(relation) ≠ ∅`. Also O(1).

Both gates use a conservative fallback: when schema information is missing, the
path is admitted by default. This bounds the false negative rate. No embeddings,
no thresholds, no learned parameters. The oracle is a pure set-lookup over the
KG's own ontology — deterministic, interpretable, and GPU-free.

---

## 3. Three Approaches Comparison

| Property | GCR | Cosine DCA | Decomposed DCA | **Symbolic DCA** |
|----------|-----|------------|----------------|-------------------|
| Encoder needed | No | Yes (all-MiniLM) | Yes (all-MiniLM) | **No** |
| Threshold τ | No | Yes (tuned) | Yes (tuned) | **No** |
| GPU needed | Decode only | Encode + decode | Encode + decode | **Decode only** |
| Type awareness | None | Implicit in cosine | Implicit in components | **Explicit** |
| Range awareness | None | None | None | **Explicit** |
| Deterministic | Yes | No (float noise) | No (float noise) | **Yes** |
| Failure mode | Overly broad | Collapse at threshold | Collapse at threshold | **Conservative fallback** |

---

## 4. Formal Guarantees

All theorems are machine-checked in Lean 4 (`proof/formal_proof/FormalProof.lean`,
371 lines, 3298 jobs, zero errors) using mathlib v4.30.0-rc2.

**Theorem 1 (v1 Faithfulness):** Every token produced by DCA-Trie v1 is a
verified member of the knowledge graph. The symbolic filter is a subset of BFS,
the trie is built from the filtered set, so any admitted token traces back to a
valid KG triple.

**Theorem 2 (Monotonicity):** `P_filtered ⊆ P_range ⊆ P_GCR`. Every path
admitted by the symbolic oracle is also in GCR's path set — no false positives
are introduced by the pruning.

**Theorem 3 (FNR Union Bound):** `P(exclude | gold_path) ≤ P(type_fail) + P(range_fail)`
by the union bound. Each gate fails only when both the question provides a type
signal AND the entity has a known incompatible type — a conservative design that
bounds the false negative rate.

---

## 5. Results (100-Sample WebQSP Dev)

The type gate alone prunes approximately **1,100 paths per question** using pure
ontology lookups with no encoder overhead. Cosine similarity at τ=0.25 exhibited
threshold collapse: **84% of decoding attempts produced empty tries** (all paths
rejected), a catastrophic failure mode. The symbolic gates have no such failure
mode — they are conservative by design and never reject a path when schema
information is absent. Full results (Hits@1, F1, SIR decomposed by
type/trajectory) for WebQSP and CWQ are reported in Chapter 4.

---

## 6. Code Structure

```bash
graph-constrained-reasoning/
├── approach3_symbolic/          # Canonical implementation
│   ├── type_oracle.py           # Symbolic oracle (586 lines, stdlib only)
│   ├── algo_demo.py             # Self-contained algorithm demonstration
│   └── notebooks/               # End-to-end evaluation pipelines
├── src/
│   ├── graph_constrained_decoding.py  # Core decoding loop (45 lines)
│   ├── trie.py                        # Prefix tree construction (179 lines)
│   └── qa_prompt_builder.py           # Prompt + path formatting (537 lines)
├── proof/formal_proof/
│   └── FormalProof.lean         # Machine-checked Lean 4 proof
├── approach1_cosine/            # Cosine similarity baseline
└── approach2_decomposed/        # Decomposed product scoring
```

**Key entry point:** `approach3_symbolic/type_oracle.py` — the full symbolic
oracle (type map extraction, answer type inference, both gates) in a single
file with no dependencies beyond Python 3.10+.

---

## 7. Relation to Existing Work

| Work | Constraint Basis | Type-Aware? | Threshold? | Encoder? |
|------|-----------------|-------------|------------|----------|
| GCR (Luo et al., 2025) | Graph topology | No | No | No |
| DoG (Li et al., 2025) | Graph topology | No | No | No |
| ToG (Sun et al., 2024) | Prompt-based search | No | No | No |
| **DCA-Trie (this thesis)** | **Graph topology + KG ontology** | **Yes** | **No** | **No** |

DCA-Trie is the first constraint oracle to exploit the KG's ontological schema
directly at the token level, without encoding it through a learned
representation. Where GCR and DoG constrain by graph structure alone, DCA-Trie
adds a semantic layer: entity types and relation ranges that the KG already
provides as ground-truth metadata. This yields tighter constraint sets with zero
additional runtime cost beyond set membership checks.
