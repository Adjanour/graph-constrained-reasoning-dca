# One-Page Summary: All Key Numbers

## TypeOracle
- SIR = **14.5%** (path reduction)
- FNR_type = **3.3%**
- FNR_range = **2.9%**
- 593,382 paths pruned from 4,102,833

## WebQSP (1,627 samples)
- Baseline Hits@1: **91.6%**
- DCA_v1 Hits@1: **86.4%** (-5.2pp, -14.5% paths)
- DCA_v2 Hits@1: **54.0%** (791/1,466, est. 878/1,627 full, -37.6pp, broken)

## CWQ (verified: 100; path data: 3,520)
- Baseline Hits@1: **69.0%**
- Filtered Hits@1: **65.0%** (-4.0pp)
- DCA_v1 extrapolated: **62-65%** (path reduction confirmed at 14.5%)
- DCA_v2 extrapolated: **35-45%**

## Key Insight
> Tighter oracle ≠ higher accuracy. Path reduction and accuracy have a **non-monotone** relationship.

## Cross-Dataset
| | WebQSP | CWQ |
|---|---|---|
| Baseline | 91.6% | 69.0% |
| Retention after filter | 98.9% | 94.2% |
| Avg paths/q | 2,553 | 2,239 |
| Path reduction | 14.5% | 10-14% |

## Beam Search
- Beam (k=10): **91.6%** vs Greedy: **80.6%** = **+11pp**

## Timing (full WebQSP run)
- Total: **~10,300s (2.9h)** on RTX 4090
- DCA_v1 adds **zero overhead**

## Adaptive Budgets Fail
- Adaptive500: 30.7% (vs 80.6% baseline)
- Adaptive100: 12.8%
- Adaptive30: 7.9%

## ORT Pilot (CWQ, 10 samples)
- Processable: 4/10 (40%)
- Hits@1 when processable: 4/4 (100%)

## GCR Paper Comparison
- Paper WebQSP: 92.6% (uses GPT-4o-mini for step 2)
- Our WebQSP: 91.6% (no GPT, direct extraction)
- Paper CWQ: 75.8% (uses GPT)
- Our CWQ: not run (est. ~69%)
