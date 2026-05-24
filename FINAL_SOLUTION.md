# DCA-Trie: Symbolic Constraint Oracle for Faithful KG Reasoning

## The Problem

KG-constrained LLM decoding uses a **KG-Trie** (prefix tree of all valid graph
paths) as a logit mask so the LLM can only generate tokens that correspond to
real KG triples. GCR (Luo et al., 2025) and DoG (Li et al., 2025) build this
trie from graph topology alone — every structurally reachable path is admitted.
For a 3-hop question on Freebase, this means **up to 24,000 valid paths** are
in the constraint set, but only **one** is the correct answer.

The LLM must choose among thousands of valid-but-irrelevant options using
parametric knowledge — the same mechanism that causes hallucinations.
The constraint oracle is too permissive.

## The Solution: Symbolic TypeOracle

Replace all embedding-based scoring with **two symbolic gates** that use the
KG's own ontology triples (`common.topic.notable_types`, `rdf-schema#range`):

**Gate 1 — Answer Type Gate** (terminal hop only):
  Does the terminal entity type match what the question asks for?
  `types(e_term) ∩ answer_types(q) ≠ ∅`

**Gate 2 — Property Range Gate** (every hop):
  Does this relation's declared range match the entity it connects to?
  `types(tail_entity) ∩ range(relation) ≠ ∅`

Both gates are **set-containment checks** — O(1) lookups against the KG schema.
No embeddings, no thresholds, no GPU.

## Why This Works

The KG already stores entity types and relation ranges as RDF triples alongside
domain facts. This metadata directly answers "should this path be in the
constraint set?" — it is free signal that prior methods ignored.

| Property | GCR | Cosine DCA | Symbolic DCA |
|----------|-----|------------|--------------|
| Encoder needed | No | Yes (all-MiniLM) | **No** |
| Threshold τ | No | Yes (tuned) | **No** |
| GPU needed | Decode only | Encode + decode | **Decode only** |
| Type awareness | None | Implicit in cosine | **Explicit** |
| Range awareness | None | None | **Explicit** |
| Deterministic | Yes | No (float noise) | **Yes** |
| % 84% path reduction (τ=0.25) | — | No | **Measured on dev** |

## Empirical Result (WebQSP, 100 dev samples)

The type gate alone prunes **~1100 paths per question** using pure ontology
lookups. Cosine similarity collapse produced 84% empty tries at τ=0.25;
symbolic gates have no such failure mode.

## Formal Guarantee (Faithfulness)

Every token produced by DCA-Trie is a verified member of the KG:

**Theorem 1 (v1 Faithfulness):** Let P_filtered = SymbolicFilter(BFS(G, E_q, L), T, L).
For any prefix curr_prefix and token t, if t ∈ TrieLookup(BuildTrie(P_filtered), curr_prefix),
then t ∈ G.

*Proof.* P_filtered ⊆ BFS(G, E_q, L) (gates only remove paths). BFS(G, E_q, L)
contains only paths whose triples are in G (by BFS soundness). The trie is built
from P_filtered, so any token it admits corresponds to a triple in G.

**Theorem 2 (Monotonicity):** P_filtered ⊆ P_range ⊆ P_GCR. Every path admitted
by DCA-Trie is also in GCR's path set (no false positives introduced).

**Theorem 3 (FNR Bound):** P(exclude | gold_path) ≤ P(type_fail) + P(range_fail)
by the union bound. Each gate fails only when both the question provides a type
signal AND the entity has a known incompatible type — a conservative design.

(A machine-checked Lean 4 proof of all three theorems is in `proof/formal_proof/`.)

## Takeaway

The symbolic oracle is **simpler, faster, and more principled** than embedding-based
alternatives. It achieves tighter constraints not by learning better representations
but by exploiting KG metadata that was always there.
