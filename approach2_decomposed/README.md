# Approach 2: Decomposed Product Score

**Status:** Second design (intermediate between Approach 1 and 3).

**Thesis reference:** Chapter 3, §3.5–3.6 (Eq. 3.12–3.14).

## Idea

Replace the monolithic cosine score with a product of three interpretable components:

$$\text{score}(e_t, r, e' \mid q) = \rho_r(r, q) \cdot \rho_e(e', q) \cdot \rho_{\text{traj}}(r, e', q)$$

| Component | Meaning | Implementation |
|-----------|---------|----------------|
| $\rho_r(r, q)$ | Relational relevance | `cos(E(r), E(q_rel))` — how well relation `r` matches the question's relational intent |
| $\rho_e(e', q)$ | Hard type gate | `1[type(e') ∈ T(q, h)]` — binary check: is entity `e'` the right type? |
| $\rho_{\text{traj}}(r, e', q)$ | Trajectory relevance | `cos(E(r ‖ e'), E(q))` — does this (relation, entity) pair point in the right direction? |

## Notebooks

| Notebook | Description |
|----------|-------------|
| `notebooks/01_GCR_Baseline.ipynb` | GCR baseline (universal) |
| `notebooks/02_DCA_Trie_v1.ipynb` | **v1 — decomposed static filtering** ($\rho_r \cdot \rho_e \cdot \rho_{\text{traj}}$) |
| `notebooks/03_DCA_Trie_v2.ipynb` | **v2 — step-wise decomposed expansion** |
| `notebooks/04_SIR_Evaluation.ipynb` | SIR evaluation |

## Key Changes from Approach 1

1. **Entity masking** (`q_rel`): The question string has entity names replaced with `[MASK]` tokens so the encoder focuses on relational intent, not entity names.
2. **Hard type gate**: A strict binary filter ($\rho_e$) that prunes type-incompatible paths *before* any encoder call, saving compute.
3. **Product decomposition**: Three scores multiplied together, each measuring a different aspect of relevance.

## Weaknesses

1. **Still needs an encoder**: every relation and path fragment requires a forward pass (with caching).
2. **Threshold persists**: $\tau$ is still dataset-dependent.
3. **Type gate is hand-crafted**: answer type patterns must be written manually.
