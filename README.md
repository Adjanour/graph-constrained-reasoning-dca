# DCA-Trie: Ontology-Guided Constraint Oracles for KG Reasoning

Extends [Graph-constrained Reasoning (GCR)](https://github.com/rmanluo/graph-constrained-reasoning) with tighter constraint oracles that exploit KG ontological metadata — entity types and relation ranges — to prune structurally valid but semantically irrelevant paths before LLM decoding.

## Problem

GCR builds a prefix tree of all structurally valid KG paths and masks invalid tokens during decoding. For a 3-hop question on Freebase, this yields up to **24,000 valid paths** — all structurally sound, only one correct. The LLM must choose among thousands of valid-but-irrelevant options using parametric knowledge, the same mechanism that produces hallucinations. Structural validity alone is insufficient.

## Solution

The KG already stores entity types and relation ranges as RDF triples alongside domain facts. The Symbolic TypeOracle replaces embedding-based scoring with two deterministic gates:

- **Answer Type Gate** — at the terminal hop, admits only entities whose type matches what the question asks for (e.g., "who?" admits `Person`-typed entities)
- **Property Range Gate** — at every hop, blocks paths where the relation's declared range is incompatible with the tail entity's type

No embeddings, no thresholds, no learned parameters. Pure set-lookup over the KG's own ontology. Deterministic, interpretable, and GPU-free.

## Approaches

| Directory | Approach | Key Idea |
|-----------|----------|----------|
| `approach1_cosine/` | Cosine similarity | `cos(E(path), E(question)) >= τ` — monolithic score |
| `approach2_decomposed/` | Decomposed product | `ρ_r · ρ_e · ρ_traj` — three components, one encoder |
| `approach3_symbolic/` | Symbolic TypeOracle | type gate + range gate — no encoder, no threshold |

## Project Structure

```
graph-constrained-reasoning/
├── approach3_symbolic/          # Canonical implementation
│   ├── type_oracle.py           # Symbolic oracle (stdlib only)
│   └── algo_demo.py             # Self-contained algorithm demo
├── notebooks/                   # Evaluation pipelines
├── src/                         # Original GCR source (unaltered)
├── workflow/                    # Original GCR entry points
├── scripts/                     # Original GCR shell scripts
├── proof/                       # Lean 4 formal proofs
└── resources/                   # Figures and assets
```

## Setup

Requires Python 3.12+, CUDA 12.1, and [Poetry](https://python-poetry.org/).

```bash
conda create -n gcr python=3.12 && conda activate gcr
poetry install
pip install flash-attn --no-build-isolation
```

## Quick Start

1. Build graph index: `bash scripts/build_graph_index.sh`
2. Run symbolic oracle demo: `python approach3_symbolic/algo_demo.py`
3. Run full evaluation notebooks in `notebooks/`

## Citation

```bibtex
@inproceedings{luo2025graph,
  title={Graph-constrained Reasoning: Faithful Reasoning on Knowledge Graphs with Large Language Models},
  author={Luo, Linhao and Zhao, Ziyi and Haffari, Gholamreza and Li, Yuan-Fang and Gong, Chen and Pan, Shirui},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year={2025}
}
```

Original GCR README preserved in [`src/ORIGINAL_README.md`](src/ORIGINAL_README.md).
