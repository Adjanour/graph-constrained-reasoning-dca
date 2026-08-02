# Slide Data Reference

## 1. TypeOracle SIR/FNR (Slide: "How much can we prune?")

| Metric | WebQSP (1,628) | CWQ (extrapolated) |
|--------|---------------|---------------------|
| **SIR** (path reduction) | **14.5%** | **10-14%** |
| SIR_type (type gate) | 10.6% | ~8% |
| SIR_traj (range gate) | 3.8% | ~3% |
| FNR_type | 3.3% | ~3-4% |
| FNR_range | 2.9% | ~2-3% |
| Total raw paths | 4,102,833 | — |
| Paths pruned | 593,382 | — |
| Type-blocked | 435,897 | — |
| Range-blocked | 157,485 | — |
| Avg paths/q (raw) | 2,553 | 2,239 |
| Avg paths/q (filtered) | 2,157 | ~1,920 |

---

## 2. Main Result: Hits@1 Accuracy (Slide: "Does it work?")

### WebQSP (1,627 test samples)

| Method | Hits@1 | Δ vs Baseline | Paths | Time |
|--------|--------|--------------|-------|------|
| GCR_Baseline | **91.6%** | — | 4,102,833 | 10,329s |
| DCA_v1_Static | **86.4%** | -5.2pp | 3,509,451 (-14.5%) | 10,385s |
| DCA_v2_Dynamic | **54.0%** (est. 878/1,627 full) | -37.6pp | 1,466/1,627 (90%) | ~10,000s (est. full) |

### CWQ (100 verified samples, +3,520 path-valid samples)

| Method | Hits@1 (100) | Δ vs Baseline | Extrapolated Full | Path reduction |
|--------|-------------|--------------|-------------------|----------------|
| GCR_Baseline | **69.0%** | — | **~69%** | — |
| Filtered (TypeOracle) | **65.0%** | -4.0pp | ~65% | 10.4% |
| DCA_v1_Static | — | — | **62-65%** (est.) | 14.5% (confirmed) |
| DCA_v2_Dynamic | — | — | **35-45%** (est.) | slower |

---

## 3. The Non-Monotone Finding (Slide: Key Insight)

| Metric | GCR_Baseline | DCA_v1_Static | % Change |
|--------|--------------|---------------|----------|
| Hits@1 | 91.6% | 86.4% | -5.2% |
| Accuracy | 77.7% | 72.2% | -5.5% |
| F1 | 66.2% | 61.6% | -4.6% |
| Precision | 66.5% | 62.1% | -4.4% |
| Recall | 77.7% | 72.2% | -5.5% |
| Tightness | 0.000 | **0.145** | +14.5pp |
| Path reduction | 0 | **14.5%** | — |

> **Tighter oracle ≠ higher accuracy.** 14.5% path reduction costs 5% accuracy.

---

## 4. Additional Methods (Slide: "What else did we try?")

### Adaptive Path Budgets (WebQSP, 1,627 samples, greedy)

| Method | Hits@1 | Budget | Notes |
|--------|--------|--------|-------|
| Baseline (greedy) | 80.6% | ∞ | Pre-beam-fix baseline |
| Validate (post-hoc) | 80.6% | ∞ | TypeOracle validation only |
| Filtered | 78.9% | 2,213 avg | Static pre-filtering |
| Label-plan | 78.8% | 2,213 avg | Ontology label paths |
| Adaptive 500 | **30.7%** | 500 | Too aggressive |
| Adaptive 100 | **12.8%** | 100 | Far too aggressive |
| Adaptive 30 | **7.9%** | 30 | Catastrophic |

### DCA v2 Smoke Tests (Qwen 2.5-3B, greedy, no GPU)

| Test | N | Hits@1 | Notes |
|------|---|--------|-------|
| v2_smoke_test | 3 | 0.0% | trust_remote_code error |
| v2_smoke_test2 | 3 | timed out | 120s timeout |
| v2_smoke_test3 | 2 | 0.0% | Works, no hits |
| v2_smoke_test4 | 3 | **33.3%** | First v2 success! |

### KGQA Baseline (Qwen 2.5-3B, no KG paths)

| Metric | Value |
|--------|-------|
| Hit rate | 60.0% |
| Accuracy | 29.2% |
| F1 | 37.3% |

---

## 5. GCR Paper Comparison (Slide: "How do we compare?")

| Aspect | GCR Paper (Luo et al.) | Our Project |
|--------|----------------------|-------------|
| Model | Llama-3.1-8B | Same |
| Step 2 reasoning | **GPT-4o-mini** | Direct extraction |
| WebQSP Hits@1 | **92.6%** | **91.6%** (no GPT) |
| CWQ Hits@1 | 75.8% | Not run (est. ~69%) |
| GPU | A100 (40GB) | RTX 4090 (24GB) |
| Beam width | k=5 | k=10 |
| Oracle | None | TypeOracle (+novel) |
| Structural faithfulness | 100% | 100% |

---

## 6. ORT-Style Ontology Reasoning (Slide: "Future direction")

| Metric | Value |
|--------|-------|
| Samples | 10 (CWQ) |
| Processable | 4/10 (40%) |
| Hits@1 (processable) | 4/4 (100%) |
| Skip rate | 60% |
| Avg time | 10s/q |
| Category graph | 7 nodes, 24 edges |

Problem: Category graph too coarse (7 categories) → 60% skip rate.
Solution: Finer granularity (15-20 categories).

---

## 7. Cross-Dataset Comparison (Slide: "WebQSP vs CWQ")

| Metric | WebQSP | CWQ | Ratio |
|--------|--------|-----|-------|
| Baseline Hits@1 | 91.6% | 69.0% | 0.75× |
| Avg paths/q | 2,553 | 2,239 | 0.88× |
| Path reduction | 13.3-14.5% | 10.4-14.5% | ~1× |
| Filtering cost | -1.1pp | -4.0pp | 3.6× |
| Retention rate | 98.9% | 94.2% | — |
| Questions | 1,628 | 34,689 | 21× |
| Hop depth | 1-2 | up to 4 | 2× |

---

## 8. Timing (Slide: "Runtime overhead")

| Experiment | Samples | Time | Rate | GPU |
|-----------|---------|------|------|-----|
| TypeOracle eval (full) | 1,628 | 322s | 5.1 q/s | A100 |
| GCR_Baseline (full) | 1,627 | 10,329s (2.9h) | 0.16 q/s | RTX 4090 |
| DCA_v1_Static (full) | 1,627 | 10,385s (2.9h) | 0.16 q/s | RTX 4090 |
| DCA_v2_Dynamic (54.0%, est. full) | 1,627 | ~10,000s (2.8h, est.) | 0.17 q/s | RTX 4090 |
| Beam 100q WebQSP | 100 | 671s | 6.71s/q | RTX 4090 |
| Beam 100q CWQ | 100 | 632s | 6.32s/q | RTX 4090 |

DCA_v1 has essentially zero time overhead vs baseline.

---

## 9. Key Takeaways for Slides

1. **GCR baseline reproduces well**: 91.6% vs published 92.6% (without GPT)
2. **TypeOracle prunes 14.5% of paths** with 3.3% FNR
3. **DCA_v1_Static is the best constrained method** (86.4%, zero overhead)
4. **Non-monotone relationship confirmed** — tighter oracle ≠ higher accuracy
5. **CWQ is harder**: 69% vs 92% baseline; filtering hurts more (-4pp vs -1pp)
6. **DCA_v2 is broken** (-37.6pp) — architectural issue, not tunable
7. **ORT direction promising** but needs finer category granularity
8. **Beam search adds +8-9pp** over greedy — critical configuration

---

*For the full consolidated analysis, see `../EXPERIMENT_RESULTS.md`*
