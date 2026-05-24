# Approach 3: Symbolic TypeOracle

**Status:** Current design (final approach used in Chapter 3 experiments).

**Thesis reference:** Chapter 3, §3.5–3.7 (TypeOracle, Algorithm 1, Algorithm 2).

## Idea

Replace all embedding computations with **pure ontology lookups**. The TypeOracle uses only the KG's own schema triples (`common.topic.notable_types`, `rdf-schema#domain`, `rdf-schema#range`) to decide whether a path is admissible.

No sentence transformer. No threshold. No GPU.

## Gates

### Gate 1: Answer Type Gate ($\rho_e$, terminal hop only)

$$\text{type\_gate}(e_h, q, h, L) = \begin{cases}
\text{True} & h < L \text{ (intermediate hop)} \\
\text{True} & \mathcal{T}(q) = \emptyset \text{ (no type inferred)} \\
\text{True} & \text{types}(e_h) = \emptyset \text{ (entity type unknown)} \\
\mathbf{1}[\text{types}(e_h) \cap \mathcal{T}(q) \neq \emptyset] & \text{otherwise}
\end{cases}$$

### Gate 2: Property Range Gate ($\rho_r$, all hops)

$$\text{range\_gate}(r, e') = \begin{cases}
\text{True} & r \notin \mathcal{R} \text{ (schema unknown)} \\
\text{True} & \text{range}(r) = \emptyset \\
\text{True} & \text{types}(e') = \emptyset \\
\mathbf{1}[\text{types}(e') \cap \text{range}(r) \neq \emptyset] & \text{otherwise}
\end{cases}$$

## Notebooks

| Notebook | Description |
|----------|-------------|
| `notebooks/01_GCR_Baseline.ipynb` | GCR baseline (universal) |
| `notebooks/02_DCA_Trie_v1.ipynb` | **v1 — static symbolic filtering** (type gate + range gate) |
| `notebooks/03_DCA_Trie_v2.ipynb` | **v2 — step-wise symbolic expansion** |
| `notebooks/04_SIR_Evaluation.ipynb` | SIR evaluation (decomposed type + trajectory) |
| `type_oracle.py` | Standalone symbolic oracle (stdlib only) |

## Key Advantages over Approaches 1 and 2

| Property | Approach 1 | Approach 2 | Approach 3 |
|----------|-----------|-----------|-----------|
| Encoder needed | Yes | Yes | **No** |
| Threshold $\tau$ | Yes | Yes | **No** |
| Type checking | Implicit | Hard gate | **Ontology-based** |
| Range checking | None | None | **Ontology-based** |
| Per-path cost | Encoder forward pass | Encoder forward pass | **O(1) set lookup** |
| Deterministic | No (float noise) | No (float noise) | **Yes** |
