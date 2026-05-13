# Attributions

## Original GCR Framework

This repository builds upon the **Graph-Constrained Reasoning (GCR)** framework, originally developed by:

- **Linhao Luo** (Monash University)
- **Zhao Ziyi** (Monash University)
- **Gholamreza Haffari** (Monash University)
- **Yuan-Fang Li** (Monash University)
- **Chen Gong** (Singapore University of Technology and Design)
- **Shirui Pan** (Griffith University)

Their paper, *"Graph-constrained Reasoning: Faithful Reasoning on Knowledge Graphs with Large Language Models"*, was published at **ICML 2025**.

**Source repository:** [https://github.com/rmanluo/graph-constrained-reasoning](https://github.com/rmanluo/graph-constrained-reasoning)

**HuggingFace model collection:** [https://huggingface.co/collections/rmanluo/graph-constrained-reasoning-671052e5c808aa5e8c57501a](https://huggingface.co/collections/rmanluo/graph-constrained-reasoning-671052e5c808aa5e8c57501a)

**Citation:**
```bibtex
@inproceedings{luo2025graph,
  title={Graph-constrained Reasoning: Faithful Reasoning on Knowledge Graphs with Large Language Models},
  author={Luo, Linhao and Zhao, Ziyi and Haffari, Gholamreza and Li, Yuan-Fang and Gong, Chen and Pan, Shirui},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year={2025}
}
```

## Our Extensions

The code in this repository extends GCR with the following contributions:

| Contribution | Description | File(s) |
|---|---|---|
| **DCA-Trie v1** | Static semantic filtering of KG paths before trie construction using sentence transformer embeddings | `notebooks/02_DCA_Trie_v1.ipynb` |
| **DCA-Trie v2** | Step-wise dynamic trie expansion conditioned on question + partial generation state | `notebooks/03_DCA_Trie_v2.ipynb` |
| **Semantic Irrelevance Ratio (SIR)** | Metric measuring constraint oracle permissiveness independently of answer accuracy | `notebooks/04_SIR_Evaluation.ipynb` |
| **SIR analysis & ablation** | Per-hop SIR stratification, threshold sensitivity, false negative rate analysis | `notebooks/04_SIR_Evaluation.ipynb` |

## Repository Structure

- `src/` — Original GCR source code (unaltered)
- `workflow/` — Original GCR entry points
- `scripts/` — Original GCR shell scripts
- `notebooks/` — **Our work**: DCA-Trie implementation and evaluation notebooks
- `GCR_Colab.ipynb` — Colab demo notebook
- `EXPLAINER.md` — Comprehensive code explanation and learning material

## Datasets

- **WebQSP** (Yih et al., 2016)
- **ComplexWebQuestions** (Talmor and Berant, 2018)
- Pre-processed versions hosted on HuggingFace: [rmanluo/RoG-webqsp](https://huggingface.co/datasets/rmanluo/RoG-webqsp), [rmanluo/RoG-cwq](https://huggingface.co/datasets/rmanluo/RoG-cwq)

## License

This project is for academic research purposes. The original GCR code is used under the terms of its original license. Our extensions are provided for reproducibility of the results reported in the accompanying thesis.
