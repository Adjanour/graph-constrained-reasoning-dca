# Structural Irrelevance Rate (SIR)

## What is SIR?

**SIR** measures the fraction of candidate paths that are *semantically irrelevant* to the question -- paths that exist in the knowledge graph but don't actually help answer the question.

When you run DFS from a question entity, you get thousands of candidate paths. Most are noise: they wander through random entities that have nothing to do with the answer. The TypeOracle's job is to identify and filter out these irrelevant paths before constrained decoding.

## Intuition

Consider the question: *"Who invented the telephone?"*

DFS from the entity `telephone` might produce paths like:

| Path | Relevant? |
|------|-----------|
| `telephone -> invented_by -> Alexander_Graham_Bell` | Yes |
| `telephone -> manufacturer -> AT&T` | Maybe |
| `telephone -> color -> black` | No |
| `telephone -> location -> New_York` | No |

The paths to `black` (type: `color`) and `New_York` (type: `city`) are semantically irrelevant -- they terminate at entity types that could never be the answer to "Who invented...?" The TypeOracle's type gate catches these.

## Mathematical Definition

Let $\mathcal{P}(q)$ denote the set of all candidate paths for question $q$, obtained via DFS up to $L$ hops.

### Irrelevance predicates

A path $p = (e_0, r_1, e_1, \ldots, r_k, e_k)$ is **type-irrelevant** if:

$$\text{irrel}_{\text{type}}(p, q) = \begin{cases} 1 & \text{if } k = L \text{ and } \text{types}(e_k) \cap \mathcal{T}(q) = \emptyset \\ 0 & \text{otherwise} \end{cases}$$

Where:
- $k = L$ means the path is at the terminal hop (the only hop where we check)
- $\text{types}(e_k)$ = Freebase types of the terminal entity
- $\mathcal{T}(q)$ = answer types inferred from the question

A path $p$ is **trajectory-irrelevant** if:

$$\text{irrel}_{\text{traj}}(p, q) = \begin{cases} 1 & \text{if } \exists\, i \in \{1, \ldots, k\}: \text{types}(e_i) \cap \text{range}(r_i) = \emptyset \\ 0 & \text{otherwise} \end{cases}$$

Where $\text{range}(r_i)$ = the declared range of relation $r_i$ in the KG schema.

### SIR decomposition

$$\text{SIR}^*_{\text{type}}(q) = \frac{|\{p \in \mathcal{P}(q) : \text{irrel}_{\text{type}}(p,q) = 1\}|}{|\mathcal{P}(q)|}$$

$$\text{SIR}^*_{\text{traj}}(q) = \frac{|\{p \in \mathcal{P}(q) : \text{irrel}_{\text{traj}}(p,q) = 1\}|}{|\mathcal{P}(q)|}$$

$$\text{SIR}^*(q) = \frac{|\{p \in \mathcal{P}(q) : \text{irrel}_{\text{type}}(p,q) \lor \text{irrel}_{\text{traj}}(p,q) = 1\}|}{|\mathcal{P}(q)|}$$

The overall $\text{SIR}^*$ is the union: a path is irrelevant if *either* gate flags it.

### False Negative Rate (FNR)

SIR measures pruning power. FNR measures safety:

$$\text{FNR}_{\text{type}}(q) = \frac{|\{p \in \mathcal{G}(q) : \text{irrel}_{\text{type}}(p,q) = 1\}|}{|\mathcal{G}(q)|}$$

$$\text{FNR}_{\text{traj}}(q) = \frac{|\{p \in \mathcal{G}(q) : \text{irrel}_{\text{traj}}(p,q) = 1\}|}{|\mathcal{G}(q)|}$$

Where $\mathcal{G}(q)$ = gold-truth paths (paths that actually lead to the correct answer).

## Empirical Results (WebQSP, 1,628 questions)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Total candidate paths | 4,102,833 | All DFS paths within $L=2$ |
| Paths after filtering | 3,509,451 | |
| **Paths pruned** | **593,382 (14.5%)** | Overall SIR |
| SIR_type | 10.6% | Type gate blocks ~436K paths |
| SIR_traj | 3.8% | Range gate blocks ~157K paths |
| Gold-truth paths | 14,829 | |
| **Type gate FNR** | **3.3%** | 490 gold paths incorrectly blocked |
| **Range gate FNR** | **2.9%** | 424 gold paths incorrectly blocked |

### Key observations

1. **SIR = 14.5%** means roughly 1 in 7 paths is irrelevant. The remaining 85.5% pass through -- the KG paths are mostly semantically coherent already.

2. **FNR ~3%** means the oracle is safe: it rarely drops gold paths. This is critical for maintaining recall.

3. **Type gate does 3x more work** (10.6% vs 3.8%). The most common failure mode is paths terminating at the wrong entity type, and a single Freebase type lookup catches most of these.

4. **Conservative fallback**: When metadata is absent (no types declared), the gate passes the path. This prevents catastrophic failure, unlike the cosine similarity approach ($\tau = 0.25$) which produced empty tries for 84% of queries.

## Why SIR matters for constrained decoding

The trie size directly affects decoding speed. With group-beam search (width $k=10$), each decoding step expands $k$ beams. A smaller trie means:
- Fewer valid tokens to consider at each step
- Faster `prefix_allowed_tokens_fn` evaluation
- Less memory for the trie structure

Reducing the trie by 14.5% translates to measurable speedup in Phase 2 (decoding), while the ~3% FNR means we retain almost all gold paths.

## Relationship to the paper

This metric is defined in:
- **Section 3.6** (SIR definition)
- **Table 3** (empirical results)
- **Theorem 2** (FNR union bound)

The SIR decomposition ($\text{SIR}^*_{\text{type}} + \text{SIR}^*_{\text{traj}}$) shows that the two gates are complementary: type gate catches terminal entity mismatches, range gate catches relation-entity incompatibilities along the path.
