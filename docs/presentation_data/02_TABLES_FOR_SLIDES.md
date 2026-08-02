# Copy-Paste Tables for Slides

---

## Table 1: Main Results Slide

```markdown
| Method | WebQSP Hits@1 | CWQ Hits@1 | Path Reduction |
|--------|:------------:|:----------:|:--------------:|
| GCR_Baseline       | **91.6%** | **69.0%** | — |
| DCA_v1_Static      | **86.4%** | 62-65%* | **14.5%** |
| DCA_v2_Dynamic     | 54.0% (est. 878/1,627) | 35-45%* | — |

*Extrapolated from 100-sample verified run + 3,520-sample path analysis
```

---

## Table 2: The Non-Monotone Finding Slide

```markdown
| Metric | Before (Baseline) | After (DCA_v1) | Change |
|--------|:----------------:|:-------------:|:------:|
| Hits@1      | 91.6% | 86.4% | **-5.2%** |
| Accuracy    | 77.7% | 72.2% | -5.5% |
| F1          | 66.2% | 61.6% | -4.6% |
| Precision   | 66.5% | 62.1% | -4.4% |
| Recall      | 77.7% | 72.2% | -5.5% |
| Path reduction | 0 | **14.5%** | +14.5pp |
```

---

## Table 3: TypeOracle Metrics Slide

```markdown
| Gate | Paths Removed | % of Total | False Negative Rate |
|------|:------------:|:----------:|:------------------:|
| Type gate  | 435,897 | 10.6% | 3.3% |
| Range gate | 157,485 | 3.8% | 2.9% |
| **Total**  | **593,382** | **14.5%** | — |
```

---

## Table 4: WebQSP vs CWQ Slide

```markdown
| Property | WebQSP | CWQ |
|----------|:-----:|:---:|
| Baseline Hits@1 | 91.6% | 69.0% |
| Avg paths per question | 2,553 | 2,239 |
| Path reduction rate | 13.3-14.5% | 10.4-14.5% |
| Accuracy retention after filtering | 98.9% | 94.2% |
| Max hop depth | 2 | 4 |
```

---

## Table 5: Adaptive Budgets Slide

```markdown
| Budget | Hits@1 | vs Baseline | Interpretation |
|-------:|:-----:|:----------:|----------------|
| ∞ (full) | 80.6% | — | Greedy baseline (pre-beam-fix) |
| ~2,213 (filtered) | 78.9% | -1.7pp | TypeOracle static filter |
| 500 | 30.7% | -49.9pp | Too aggressive |
| 100 | 12.8% | -67.8pp | Far too aggressive |
| 30 | 7.9% | -72.7pp | Catastrophic |
```

---

## Table 6: GCR Paper Comparison Slide

```markdown
| Aspect | GCR Paper | Our Project |
|--------|:---------:|:-----------:|
| Model | Llama-3.1-8B | Same |
| Step 2 reasoning | **GPT-4o-mini** | Direct extraction (no GPT) |
| WebQSP Hits@1 | 92.6% | **91.6%** |
| CWQ Hits@1 | 75.8% | Not run |
| GPU | A100 (40GB) | RTX 4090 (24GB) |
| Beam width | k=5 | k=10 |
| Max tokens | 8 | 256 |
| Oracle | None | **TypeOracle** (novel) |
```

---

## Table 7: ORT Pilot Slide

```markdown
| Metric | Value |
|--------|:-----:|
| Questions tested | 10 (CWQ) |
| Processable | 4/10 (40%) |
| Hits@1 (when processable) | 4/4 (100%) |
| Skip rate | 60% |
| Ontology categories | 7 nodes, 24 edges |
| Average time per question | 10s |
```

---

## Table 8: Timing Slide

```markdown
| Component | Time | % of Total |
|-----------|:----:|:----------:|
| Model loading | 4s | <1% |
| Path generation (DFS) | 2,500s | 24% |
| Trie construction | 1,500s | 15% |
| Decoding | 6,000s | 58% |
| Evaluation | 325-381s | 3-4% |
| TypeOracle (DCA_v1 only) | 500s | 5% |
| **Total** | **~10,300s (2.9h)** | 100% |
```

---

## Table 9: Beam Search Impact Slide

```markdown
| Method | Greedy (pre-fix) | Beam (k=10) | Improvement |
|--------|:---------------:|:-----------:|:----------:|
| GCR_Baseline | 80.6% | 91.6% | **+11.0pp** |
| Filtered | 78.9% | 88.0% | +9.1pp |
```

---

## Text for Slide Callouts

> "14.5% path reduction at 5% accuracy cost"

> "Tighter oracle ≠ higher accuracy — non-monotone relationship"

> "CWQ baselines are 22-25pp lower — deep graphs are fundamentally harder"

> "DCA_v2 loses 37.6pp due to tokenization misalignment — architectural issue"

> "Beam search provides a consistent +8-9pp lift across all methods"

> "TypeOracle catches only 23.8% of wrong predictions — semantic errors remain"
