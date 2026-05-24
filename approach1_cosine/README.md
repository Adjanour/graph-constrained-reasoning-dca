# Approach 1: Cosine Similarity Path Scoring

**Status:** Initial design (superseded by Approaches 2 and 3).

**Thesis reference:** Original DCA-Trie design (before Chapter 3 revision).

## Idea

Score each candidate KG path by computing cosine similarity between:

- `E(path_str)` — a sentence-transformer embedding of the serialised path string
- `E(question)` — an embedding of the input question

If `cos(E(path_str), E(question)) ≥ τ`, the path is admitted into the trie.

## Algorithm

```
input: G, E_q, q, L, encoder all-MiniLM-L6-v2, threshold τ
  P ← BFS(G, E_q, L)
  u_q ← encoder.encode(q)
  for each path p in P:
    path_str ← serialize(p)
    u_p ← encoder.encode(path_str)
    score ← cos(u_q, u_p)
    if score ≥ τ:
      P_filtered ← P_filtered ∪ {p}
  T ← BuildTrie(P_filtered)
return T
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `notebooks/01_GCR_Baseline.ipynb` | GCR baseline (universal) |
| `notebooks/02_DCA_Trie_v1.ipynb` | **v1 — static cosine filtering** (single score per path) |
| `notebooks/03_DCA_Trie_v2.ipynb` | **v2 — step-wise cosine expansion** |
| `notebooks/04_SIR_Evaluation.ipynb` | SIR evaluation |

## Weaknesses

1. **One number for everything**: a single cosine score cannot distinguish *why* a path is relevant — type match, relation direction, or trajectory.
2. **Encoder cost**: every path requires a full encoder forward pass.
3. **Threshold sensitivity**: τ is dataset-dependent and must be tuned.
4. **Semantic collapse**: cosine similarity in a 384-dim space is a poor proxy for fine-grained KG reasoning.
