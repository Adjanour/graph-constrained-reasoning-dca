# DCA-Trie: Research Notes & Open Questions

## The Problem

Large language models hallucinate. Graph-Constrained Reasoning (GCR) fixes this at the token level by masking logits so the LLM can only generate tokens that correspond to valid paths in a knowledge graph. This guarantees 100% structural faithfulness — every generated token matches a real KG triple.

But GCR's oracle is static and topology-only. It expands *all* paths within L hops of the question entities, regardless of relevance. For a 3-hop question, this can be 20,000+ valid paths. The LLM must choose among them using its parametric knowledge — the same mechanism that causes hallucinations in the first place.

## Our Proposed Solution: DCA-Trie

### What it does
Extend the oracle from `W_val(t) = f(G, E_q)` to `W_val(t) = f(G, E_q, q, y_<t)`. The constraint set shrinks as the generation progresses and changes based on the question.

### Decomposed scoring (Eq. 3.12)
```
score(e_t, r, e' | q, y_<t) = ρ_r(r, q) · ρ_e(e', q) · ρ_traj(r, e', q, y_<t)
```
- ρ_r = relational relevance: how well does relation r match the question?
- ρ_e = entity type gate: is entity e''s type compatible with the answer type?
- ρ_traj = trajectory relevance: does the full path (r, e') make sense given the question context?

### Variants
- **DCA-Trie v1**: static filtering at trie construction time (Algorithm 1)
- **DCA-Trie v2**: step-wise dynamic expansion with residual query vectors (Algorithm 2)

### SIR (Semantic Irrelevance Ratio)
A metric that measures oracle permissiveness independently of answer accuracy. Decomposed into SIR*_type (type blindness) and SIR*_traj (trajectory irrelevance).

## Current Implementation & Its Weakness

### What works
- The hard type gate (ρ_e) = symbolic ontology lookup. Prunes ~1100 paths/question. Fast, grounded, zero hallucination risk.
- The SIR decomposition cleanly separates type failures from trajectory failures.
- GCR's logit-masking architecture is correct — token-level constraints are the only way to guarantee faithfulness.

### What's wrong
ρ_r and ρ_traj are implemented as **cosine similarity of all-MiniLM-L6-v2 sentence embeddings**. This is the weak link:
1. Uses a separate 22M-parameter encoder — extra compute, extra dependency
2. Cosine similarity on MiniLM embeddings is a weak proxy for relational relevance
3. Product of sub-scores shrinks values rapidly: τ=0.25 was too aggressive (84% empty tries at the first test)
4. The sentence transformer is trying to solve a *semantic* problem that the LLM (8B params) already handles natively during generation

## The Core Insight (The Gap)

We realized the scoring architecture is in a no-man's land:

- **Not structural enough**: unlike the type gate (pure ontology lookup), the cosine scoring doesn't leverage any of the KG's symbolic constraints.
- **Not powerful enough**: unlike the LLM's own representations, MiniLM embeddings are too weak to judge relational relevance accurately.
- **Wrong tool for the job**: the KG's value is that it provides *ground-truth structural constraints*. We should be using structural tools (lookups, ontology rules) for the oracle, not probabilistic semantic approximations.

## What We Learned From the Literature

### Knowledge graphs are more than graphs
Freebase (and KGs generally) have a rich **ontological schema** layer:
- **Domains** → **Types** → **Properties** hierarchy (e.g., `/people/person`, `/film/film/director`)
- **Expected types** on property values (like RDFS range): each property constrains what type its object must be
- **Incompatibility rules**: some types are mutually exclusive
- **CVTs/Mediators**: compound value types for n-ary relations
- **Multi-valued properties by default**
- **Relation composition patterns**: which relation sequences are meaningful

This ontology is formalized in RDF triples — it's *machine-readable symbolic structure*, not embeddings.

### How the field processes KGs for LLMs (current paradigms)

| Approach | How it uses the KG | Token-level constraint? |
|---|---|---|
| **GCR / DoG** | Trie of all KG paths within L hops, used as logit mask | Yes |
| **DCA-Trie (current)** | Same + semantic filtering with sentence transformer | Yes |
| **GNN-RAG** (Mavromatis et al., 2025) | GNN scores node relevance, retrieves shortest paths as context | No |
| **REL-RAG** (2025) | Line graph transformation — relations become nodes, provably better generalization | No |
| **LKLR** (2024) | LLM generates logical queries, KG executes them iteratively | No |
| **ToG-2** (2025) | Alternates between graph retrieval and context retrieval | No |
| **ORT** (2025) | Reverse thinking on ontology *labels* (not entities) to prune paths | No |
| **AtlasKV** (2025) | KG triples → Q-K-V key-value data fused into LLM attention layers | No |
| **UltRAG** (2025) | Neural query executors for Wikidata-scale graphs | No |

Key observation: GCR-family methods are nearly unique in enforcing KG constraints **at the token level** via logit masking. Almost everything else feeds KG info as context and trusts the LLM not to hallucinate anyway.

### Open questions the field hasn't resolved
1. Can an oracle be both context-aware and purely symbolic? (No continuous scoring needed.)
2. Is the product decomposition of relevance scores the right factorization?
3. Where should semantic judgment live — in the oracle or in the LLM?

## Our Revised Direction

The KG ontology provides rich structural constraints that we're not using:

- **Type gate** (already doing): entity type compatibility — O(1), pure lookup
- **Property domain/range**: which relations can possibly start/end at which entity types
- **Relation composition patterns**: which multi-hop relation sequences are valid in the schema
- **Expected answer type inference from question**: more principled than our regex-based type oracle

The idea: make the oracle **entirely symbolic**. Use the KG's own structural constraints to prune paths — not continuous relevance scores from a separate embedding model. The LLM then does what GCR always had it do: choose the semantically correct path among the structurally valid ones, using its own attention mechanism.

This gives us:
- **Computational efficiency**: lookups instead of forward passes
- **Guaranteed correctness**: structural constraints are ground truth
- **Clean decomposition**: SIR*_type measures ontology-level permissiveness, SIR*_traj measures whatever semantic signal is left

## Remaining Questions for Deep Thinkers

1. Can an oracle that is purely structural (no learned semantic scoring) still provide meaningful constraint reduction, or does it inevitably degenerate to GCR's topology-only set?
2. What does "computationally sound" mean for a constraint oracle? How do we formally characterise the trade-off between oracle cost and generation cost?
3. Is the decomposition ρ_r · ρ_e · ρ_traj the right one, or should we think about relevance differently (e.g., additive, log-probability, or entirely different factorization)?
4. The product rule has an implicit independence assumption. Are ρ_r, ρ_e, and ρ_traj actually independent?
5. If we drop continuous scoring entirely, what replaces ρ_r and ρ_traj in the formalism? The paper currently positions DCA-Trie as "context-aware" — is that claim still valid with a purely symbolic oracle?
6. What is the relationship between SIR and computational cost? Can we prove that reducing SIR reduces expected decoding time?
7. GCR's approach puts constraints at the token level. Most other methods use context-level grounding. Is there a fundamental advantage to token-level that justifies its complexity, or should we be doing this differently?
8. Freebase has an unusually rich ontology. Would a purely structural oracle generalize to KGs with weaker schemas (e.g., Wikidata, domain-specific KGs)?

## Experimental Status

### Done
- 02_DCA_Trie_v1.ipynb: Algorithm 1 implemented, runs on 100 samples
- 03_DCA_Trie_v2.ipynb: Algorithm 2 implemented
- 04_SIR_Evaluation.ipynb: decomposed SIR computation
- GCR baseline: 89.0 Hits@1 on 100 samples of WebQSP
- DCA-Trie v1 (τ=0.25): 84% empty tries — τ far too aggressive with product scoring
- All notebooks aligned to K=5, group-beam, MAX_SAMPLES=100, MAX_NEW_TOKENS=256
- Flash-attn fix merged: auto-fallback to sdpa when flash-attn not installed

### Next experimental steps (pending revised scoring)
- Threshold calibration sweep for τ
- Re-run v1 with calibrated τ
- Re-run v2
- Compute SIR* across all systems
- Full dataset runs

## Key Technical Details

- **Model**: rmanluo/GCR-Meta-Llama-3.1-8B-Instruct (GCR's fine-tuned checkpoint)
- **Datasets**: WebQSP (4,737 questions, 1-2 hop), CWQ (~34,000 questions, up to 4 hop)
- **Knowledge graph**: Freebase (~40M entities, ~350M relations)
- **Hardware**: A100 40GB (Colab Pro / Colab Pro+)
- **Attention**: sdpa (default; flash-attn-2 optional when installed)
- **Transformers**: pinned at 4.44.2
- **Encoder (current)**: all-MiniLM-L6-v2 (sentence-transformers)
- **Output paths**: `results/GenPaths/RoG-webqsp/GCR-Meta-Llama-3.1-8B-Instruct/test/{tag}/`
