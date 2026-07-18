# GCR Paper: Experiment Configurations & Comparison

**Source**: Luo et al., "Graph-constrained Reasoning: Faithful Reasoning on Knowledge Graphs with Large Language Models" (ICML 2025)

---

## 1. GCR Paper Experiments

### 1.1 Models Tested

| Model | Parameters | VRAM | GPU Required | Status |
|-------|-----------|------|--------------|--------|
| Qwen2-0.5B-Instruct | 0.5B | ~1 GB | T4 ✅ | Released |
| Qwen2-1.5B-Instruct | 1.5B | ~3 GB | T4 ✅ | Released |
| Llama-2-7B-chat-hf | 7B | ~14 GB | A100 ✅ | Gated |
| **Llama-3.1-8B-Instruct** | **8B** | **~16 GB** | **A100 ✅** | **Gated, Best** |

**Best model**: `rmanluo/GCR-Meta-Llama-3.1-8B-Instruct`

### 1.2 Datasets

| Dataset | Questions | Hop Depth | Standard Test Set |
|---------|-----------|-----------|-------------------|
| **WebQSP** | 4,737 | 1-2 hops | Yes |
| **CWQ** (ComplexWebQuestions) | 34,689 | Up to 4 hops | Yes |

### 1.3 GCR Reported Results

| Dataset | Hits@1 | F1 | Structural Faithfulness |
|---------|--------|-----|------------------------|
| **WebQSP** | **92.6%** | - | 100% |
| **CWQ** | **75.8%** | - | 100% |

### 1.4 GCR Pipeline

```
Step 1: Graph-Constrained Decoding
  - KG-specialized LLM (Llama-3.1-8B) generates reasoning paths
  - Beam search constrained by KG-Trie
  - Output: Multiple KG-grounded reasoning paths + answer hypotheses

Step 2: Graph Inductive Reasoning  
  - General LLM (GPT-4, etc.) reasons over generated paths
  - Synthesizes final answer from hypotheses
```

**Key insight**: GCR uses a **two-stage pipeline** — constrained decoding + GPT reasoning.

---

## 2. Our Experiment Configurations

### 2.1 Models Tested

| Model | Parameters | VRAM | GPU Required | Status |
|-------|-----------|------|--------------|--------|
| **Llama-3.1-8B-Instruct** | **8B** | **~16 GB** | **RTX 4090** | **Used** |

**Model used**: `rmanluo/GCR-Meta-Llama-3.1-8B-Instruct` (same as GCR)

### 2.2 Datasets

| Dataset | Questions | Hop Depth | Status |
|---------|-----------|-----------|--------|
| **WebQSP** | 1,628 (test) | 1-2 hops | ✅ Complete |
| **CWQ** | 34,689 | Up to 4 hops | ❌ Not started |

### 2.3 Our Results

| Method | Hits@1 | Accuracy | F1 | Precision | Recall |
|--------|--------|----------|-----|-----------|--------|
| **GCR_Baseline** | **91.6%** | 77.7% | 66.2% | 66.5% | 77.7% |
| DCA_v1_Static | 86.4% | 72.2% | 61.6% | 62.1% | 72.2% |
| DCA_v2_Dynamic | 54.9% | 31.8% | 35.8% | 54.9% | 31.8% |

### 2.4 Our Pipeline

```
Step 1: Graph-Constrained Decoding (same as GCR)
  - KG-specialized LLM generates reasoning paths
  - Beam search constrained by KG-Trie
  - Added: TypeOracle filtering (v1) or dynamic expansion (v2)

Step 2: Answer Extraction (NO GPT)
  - Direct answer extraction from predicted paths
  - No separate reasoning LLM
```

**Key difference**: We do **NOT** use GPT for Step 2. Our pipeline is single-stage.

---

## 3. Configuration Comparison

### 3.1 Decoding Settings

| Parameter | GCR Paper | Our Setup |
|-----------|-----------|-----------|
| **Beam width (k)** | 5 | 10 |
| **Generation mode** | group-beam | group-beam |
| **Prompt mode** | zero-shot | zero-shot |
| **Max new tokens** | 3L+2 (8 for WebQSP) | 256 |
| **Index length (L)** | 2 (WebQSP), 4 (CWQ) | 2 |
| **Precision** | 16-bit | 16-bit |

### 3.2 Hardware

| Component | GCR Paper | Our Setup |
|-----------|-----------|-----------|
| **GPU** | NVIDIA A100 (40GB) | RTX 4090 (24GB) |
| **Attention** | flash_attention_2 | SDPA |
| **Encoder** | all-MiniLM-L6-v2 (CPU) | None (symbolic oracle) |

### 3.3 Key Differences

| Aspect | GCR Paper | Our Setup |
|--------|-----------|-----------|
| **Step 2 (Reasoning)** | GPT-4 / GPT-4o-mini | Direct extraction |
| **Beam width** | k=5 | k=10 |
| **Max tokens** | 8 (2L+2) | 256 |
| **Oracle** | None (structural only) | TypeOracle (semantic) |
| **Threshold tuning** | On validation set | None (symbolic) |

---

## 4. GPT Experiments in GCR Paper

### 4.1 GPT Usage

The GCR paper uses GPT in **Step 2 (Graph Inductive Reasoning)**:

```python
# GCR's two-stage pipeline
# Step 1: Constrained decoding with KG-specialized LLM
paths = kg_specialized_llm.generate(question, trie)

# Step 2: GPT reasons over paths (EXTRA COST)
answer = gpt_4o_mini.analyze(paths, question)
```

### 4.2 GCR's GPT Configuration

| Parameter | Value |
|-----------|-------|
| Model | GPT-4o-mini (or GPT-4) |
| Purpose | Path analysis + answer synthesis |
| Input | Top-K reasoning paths from Step 1 |
| Output | Final answer entity |

### 4.3 Our Setup (NO GPT)

```python
# Our single-stage pipeline
# Step 1: Constrained decoding with TypeOracle filtering
paths = kg_specialized_llm.generate(question, trie)

# Step 2: Direct extraction (NO GPT)
answer = extract_answer_from_paths(paths)
```

**Key difference**: We skip GPT entirely. This is:
- ✅ **Cheaper** (no API costs)
- ✅ **Faster** (no additional LLM call)
- ✅ **Simpler** (single-stage pipeline)
- ❌ **Potentially less accurate** (no GPT reasoning)

---

## 5. What We Didn't Reproduce

### 5.1 GPT Reasoning (Step 2)

**GCR paper**: Uses GPT-4o-mini to analyze top-K paths and synthesize answer.

**Our setup**: Direct extraction from predicted paths.

**Impact**: Our lower accuracy (91.6% vs 92.6%) may be partly due to missing GPT reasoning.

### 5.2 CWQ Dataset

**GCR paper**: Reports 75.8% Hits@1 on CWQ (4-hop questions).

**Our setup**: CWQ not yet evaluated.

**Status**: CWQ download interrupted, needs to be completed.

### 5.3 Beam Width k=5

**GCR paper**: Uses k=5 for constrained systems.

**Our setup**: Uses k=10 (more beams, more diverse paths).

**Impact**: Higher k may increase coverage but also increase noise.

### 5.4 Max New Tokens

**GCR paper**: 3L+2 = 8 tokens for WebQSP (L=2).

**Our setup**: 256 tokens (much longer).

**Impact**: Longer generation may produce more complete paths but also more noise.

---

## 6. Experiment Timeline

### GCR Paper (Luo et al., 2025)

| Phase | Description | Duration |
|-------|-------------|----------|
| Training | Fine-tune KG-specialized LLMs | Days |
| Evaluation | WebQSP + CWQ | Hours |
| GPT Analysis | Step 2 reasoning | Hours |

### Our Project

| Phase | Description | Duration | Status |
|-------|-------------|----------|--------|
| Setup | Fix decoding pipeline | 2 days | ✅ |
| WebQSP GCR_Baseline | Reproduce GCR | 2 hours | ✅ |
| WebQSP DCA_v1 | TypeOracle filtering | 2 hours | ✅ |
| WebQSP DCA_v2 | Dynamic expansion | 2 hours | ⚠️ Interrupted |
| CWQ (all methods) | 4-hop evaluation | 3-4 hours | ❌ Not started |
| ORT improvement | LLM-based type extraction | 10 min test | ❌ Not started |

---

## 7. Summary Table

| Aspect | GCR Paper | Our Project | Difference |
|--------|-----------|-------------|------------|
| **Model** | Llama-3.1-8B | Llama-3.1-8B | Same |
| **Dataset** | WebQSP + CWQ | WebQSP (partial) | Missing CWQ |
| **Hits@1 (WebQSP)** | 92.6% | 91.6% | -1.0% |
| **Hits@1 (CWQ)** | 75.8% | Not measured | Missing |
| **Step 2 (GPT)** | GPT-4o-mini | Direct extraction | No GPT |
| **Beam width** | k=5 | k=10 | Higher |
| **Max tokens** | 8 | 256 | Higher |
| **Oracle** | None | TypeOracle | Added |
| **GPU** | A100 (40GB) | RTX 4090 (24GB) | Smaller |
| **Structural Faithfulness** | 100% | 100% | Same |

---

## 8. Key Takeaways

### What We Reproduced
1. ✅ GCR baseline (91.6% vs 92.6% reported)
2. ✅ Structural faithfulness (100%)
3. ✅ Same model architecture (Llama-3.1-8B)
4. ✅ Same decoding pipeline (group-beam search)

### What We Didn't Reproduce
1. ❌ GPT reasoning (Step 2)
2. ❌ CWQ evaluation
3. ❌ Beam width k=5
4. ❌ Max tokens 3L+2

### What We Added
1. ✅ TypeOracle (semantic filtering)
2. ✅ DCA-Trie v1 (static filtering)
3. ✅ DCA-Trie v2 (dynamic expansion)
4. ✅ Formal oracle tightness framework
5. ✅ Non-monotone accuracy analysis

---

*Last updated: July 17, 2026*
