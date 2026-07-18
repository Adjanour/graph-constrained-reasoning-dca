# Evaluation Metrics Guide

## Overview

This document describes all available evaluation metrics for the DCA-Trie experiments. We have **12 metrics** across 4 categories, providing a comprehensive view of model performance.

---

## Metric Categories

```
Evaluation Metrics
├── Answer Quality (4 metrics)
│   ├── Accuracy (acc)
│   ├── Hits@1 (hit)
│   ├── F1 Score (f1)
│   └── Precision / Recall
│
├── Path Quality (3 metrics)
│   ├── Path F1
│   ├── Path Precision
│   └── Path Recall
│
├── Path-Answer Alignment (3 metrics)
│   ├── Path-Answer F1
│   ├── Path-Answer Precision
│   └── Path-Answer Recall
│
└── Oracle Quality (3 metrics)
    ├── Tightness
    ├── Gold-Path Retention (Recall)
    └── Path Reduction
```

---

## 1. Answer Quality Metrics

### 1.1 Accuracy (acc)

**Definition**: Fraction of ground-truth answers that appear in the prediction.

```python
acc = |predicted_answers ∩ ground_truth| / |ground_truth|
```

**Interpretation**:
- `acc = 1.0`: All ground-truth answers found
- `acc = 0.5`: Half of ground-truth answers found
- `acc = 0.0`: No ground-truth answers found

**Example**:
```
Ground Truth: ["Austria", "Switzerland", "Germany"]
Prediction: ["Austria", "Switzerland"]
Accuracy: 2/3 = 0.67
```

### 1.2 Hits@1 (hit)

**Definition**: Binary indicator — does at least one prediction match any ground-truth answer?

```python
hit = 1 if any(predicted ∩ ground_truth) else 0
```

**Interpretation**:
- `hit = 1`: At least one correct answer found
- `hit = 0`: No correct answers found

**Note**: This is what we currently report as "Hits@1" in our results.

### 1.3 F1 Score (f1)

**Definition**: Harmonic mean of precision and recall for answer matching.

```python
precision = |correct_predictions| / |total_predictions|
recall = |correct_predictions| / |ground_truth|
f1 = 2 * (precision * recall) / (precision + recall)
```

**Interpretation**:
- `f1 = 1.0`: Perfect precision and recall
- `f1 = 0.5`: Balance between precision and recall
- `f1 = 0.0`: No correct predictions

**Example**:
```
Ground Truth: ["Austria", "Switzerland", "Germany", "Liechtenstein"]
Prediction: ["Austria", "Switzerland", "Belgium"]
Precision: 2/3 = 0.67 (2 correct out of 3 predicted)
Recall: 2/4 = 0.50 (2 correct out of 4 ground truth)
F1: 2 * (0.67 * 0.50) / (0.67 + 0.50) = 0.57
```

### 1.4 Precision & Recall

**Precision**: Fraction of predictions that are correct.

```python
precision = |correct_predictions| / |total_predictions|
```

**Recall**: Fraction of ground-truth answers that are predicted.

```python
recall = |correct_predictions| / |ground_truth|
```

---

## 2. Path Quality Metrics

### 2.1 Path F1

**Definition**: F1 score between predicted reasoning paths and ground-truth paths.

```python
path_f1 = F1(predicted_paths, ground_truth_paths)
```

**Interpretation**: Measures how well the model's reasoning chains match the gold-standard paths.

### 2.2 Path Precision

**Definition**: Fraction of predicted paths that are valid ground-truth paths.

```python
path_precision = |correct_paths| / |total_predicted_paths|
```

**Interpretation**:
- High precision: Most predicted paths are valid
- Low precision: Many invalid/redundant paths

### 2.3 Path Recall

**Definition**: Fraction of ground-truth paths that are predicted.

```python
path_recall = |correct_paths| / |ground_truth_paths|
```

**Interpretation**:
- High recall: Most valid paths found
- Low recall: Missing many valid paths

---

## 3. Path-Answer Alignment Metrics

### 3.1 Path-Answer F1

**Definition**: F1 score between predicted paths and ground-truth answers (not paths).

```python
path_ans_f1 = F1(predicted_paths, ground_truth_answers)
```

**Interpretation**: Measures if the reasoning paths lead to correct answers, even if the paths themselves aren't in the gold standard.

### 3.2 Path-Answer Precision

**Definition**: Fraction of predicted paths that lead to correct answers.

```python
path_ans_precision = |paths_leading_to_correct_answer| / |total_paths|
```

### 3.3 Path-Answer Recall

**Definition**: Fraction of ground-truth answers that are reached by some predicted path.

```python
path_ans_recall = |answers_reached_by_paths| / |ground_truth_answers|
```

---

## 4. Oracle Quality Metrics

### 4.1 Tightness

**Definition**: Fraction of candidate paths excluded by the oracle.

```python
tightness = 1 - (|filtered_paths| / |all_paths|)
```

**Interpretation**:
- `tightness = 0`: Oracle admits everything (no filtering)
- `tightness = 1`: Oracle admits nothing (maximum filtering)

**Current Result**: DCA v1 has tightness = 0.145 (14.5% reduction)

### 4.2 Gold-Path Retention (Recall)

**Definition**: Fraction of gold-truth paths retained after filtering.

```python
gold_retention = |filtered_paths ∩ gold_paths| / |gold_paths|
```

**Interpretation**:
- `retention = 1.0`: All gold paths kept (safe)
- `retention < 1.0`: Some gold paths removed (risky)

**Note**: This measures oracle safety — we want high retention.

### 4.3 Path Reduction

**Definition**: Percentage of paths removed by the oracle.

```python
reduction = 1 - (|filtered_paths| / |all_paths|)
```

**Current Result**: DCA v1 reduces paths by 14.5%

---

## 5. Derived Metrics

### 5.1 Hits@k

**Definition**: Does any of the top-k predictions match a ground-truth answer?

```python
hits@k = 1 if any(top_k_predictions ∩ ground_truth) else 0
```

**Current Result**: GCR_Baseline Hits@10 = 91.3%

### 5.2 Answer Diversity

**Definition**: Number of unique answers in the top-k predictions.

```python
diversity = |unique_answers_in_top_k|
```

**Interpretation**: Higher diversity = model explores more answer candidates.

### 5.3 Path Efficiency

**Definition**: Ratio of useful paths (leading to correct answer) to total paths.

```python
efficiency = |paths_leading_to_correct_answer| / |total_paths|
```

---

## Running Evaluation

### Quick Evaluation (Current Method)

```python
from experiments.type_oracle_full.experiment import compute_hits

# Load predictions
preds = [json.loads(line) for line in open("predictions.jsonl")]

# Compute Hits@1
hits = compute_hits(preds)
print(f"Hits@1: {hits}/{len(preds)} ({hits/len(preds)*100:.1f}%)")
```

### Comprehensive Evaluation (All Metrics)

```python
from src.utils.qa_utils import eval_result, eval_path_result_w_ans

# Run full evaluation
eval_result("predictions.jsonl", cal_f1=True)

# Or with path analysis
eval_path_result_w_ans("predictions.jsonl", cal_f1=True)
```

### Evaluation Output

The evaluation generates:
1. `detailed_eval_result.jsonl`: Per-question metrics
2. `eval_result.txt`: Summary statistics

---

## Metric Comparison

| Metric | What it Measures | Range | Best For |
|--------|------------------|-------|----------|
| **Accuracy** | Answer coverage | [0, 1] | Multi-answer questions |
| **Hits@1** | Any correct answer | {0, 1} | Binary success/failure |
| **F1** | Precision-recall balance | [0, 1] | Balanced evaluation |
| **Precision** | Prediction correctness | [0, 1] | Minimizing false positives |
| **Recall** | Ground-truth coverage | [0, 1] | Minimizing false negatives |
| **Path F1** | Reasoning quality | [0, 1] | Path correctness |
| **Tightness** | Oracle strictness | [0, 1] | Oracle analysis |
| **Gold Retention** | Oracle safety | [0, 1] | Oracle reliability |

---

## Recommended Metrics for Paper

### Primary Metrics (Must Report)

1. **Hits@1**: Standard metric for KGQA
2. **F1**: Balanced measure of precision/recall
3. **Path Reduction**: Oracle efficiency

### Secondary Metrics (Should Report)

4. **Precision**: Important for constrained decoding
5. **Recall**: Important for oracle safety
6. **Tightness**: Oracle characterization

### Analysis Metrics (Nice to Have)

7. **Path F1**: Reasoning chain quality
8. **Gold Retention**: Oracle safety analysis
9. **Answer Diversity**: Model exploration

---

## Example Results Table

| Method | Hits@1 | F1 | Precision | Recall | Path Reduction | Tightness |
|--------|--------|----|-----------|----|----------------|-----------|
| GCR_Baseline | 80.9% | 85.2% | 88.1% | 82.5% | - | 0.000 |
| DCA_v1_Static | 75.9% | 80.1% | 85.3% | 75.6% | 14.5% | 0.145 |
| DCA_v2_Dynamic | 53.5% | 58.2% | 62.1% | 54.8% | TBD | TBD |

---

## Implementation Notes

### Current Limitations

1. **Ground-truth paths**: Not available in WebQSP/CWQ datasets
2. **Path extraction**: Requires parsing `# Reasoning Path:\n...` format
3. **Normalization**: Uses substring matching, not exact match

### Future Improvements

1. **Semantic similarity**: Use embeddings for answer matching
2. **Path validity checking**: Verify paths are valid KG paths
3. **Per-question analysis**: Breakdown by question difficulty

---

*Last updated: July 16, 2026*
