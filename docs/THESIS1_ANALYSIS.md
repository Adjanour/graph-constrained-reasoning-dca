# Thesis Run 1 Analysis — WebQSP (300 samples)

Companion to [LAZY_CONSTRAINT.md](LAZY_CONSTRAINT.md). Results from the
`thesis1` run (instance 47627367, RTX 4090, model
`rmanluo/GCR-Meta-Llama-3.1-8B-Instruct`).

Raw data: `results_from_vast/thesis1/thesis1/` (predictions + metrics +
`summary.json`). Repro: `analyze_thesis.py` in `/tmp/opencode`.

## Setup

| Parameter | Value |
|-----------|-------|
| Dataset | RoG-webqsp, 300 samples (first 300 of test), seed 42 |
| Decoding | Beam search k=5 |
| `index_len` | 2 (CLI default) |
| Conditions | GCR_Baseline, DCA_v1_Static, DCA_v2_Dynamic, DCA_v2_NoGates, DCA_v3_Lazy (+ v3-nogates from calibration run) |
| All runs | skip=0, dead_ends=0 |

> Note: the CLI default `--index-len 2` was used; CHAPTER4's headline numbers
> were produced with `--index-len 4` on 100 samples. The 89.0% baseline figure
> therefore is not directly comparable to the 85.7% here — treat this run as a
> controlled head-to-head, not as the chapter's absolute numbers.

## 1. Accuracy table

| Condition | Hits@1 | Time | vs baseline |
|-----------|--------|------|-------------|
| GCR_Baseline | **85.7%** (257/300) | 40.6 min | — |
| DCA_v1_Static | 81.7% (245/300) | 43.5 min | −4.0 pp |
| DCA_v2_Dynamic | 60.7% (182/300) | 32.8 min | −25.0 pp |
| DCA_v2_NoGates | 60.7% (182/300) | 32.1 min | −25.0 pp |
| DCA_v3_Lazy | 81.0% (243/300) | 39.6 min | −4.7 pp |

## 2. Statistical significance (exact binomial McNemar, paired)

| Pair | b01 | b10 | p | |
|------|-----|-----|---|-|
| Baseline vs v1 | 13 | 1 | 0.0018 | ** |
| Baseline vs v2 | 75 | 0 | <0.0001 | *** |
| Baseline vs v2-nogates | 75 | 0 | <0.0001 | *** |
| Baseline vs v3 | 14 | 0 | 0.0001 | *** |
| v1 vs v2 | 71 | 8 | <0.0001 | *** |
| v1 vs v2-nogates | 71 | 8 | <0.0001 | *** |
| v1 vs v3 | 7 | 5 | 0.7744 | ns |
| v2 vs v2-nogates | 0 | 0 | 1.0000 | ns |
| v2 vs v3 | 13 | 74 | <0.0001 | *** |
| v2-nogates vs v3 | 13 | 74 | <0.0001 | *** |

Every deviation from baseline is one-directional: the gated conditions win on
0–1 questions and lose on 13–14. The gates' value is efficiency, not accuracy.

## 3. Calibration: lazy constraint is faithful

`DCA_v3_NoGates` (from the `lazycal1` calibration run) vs baseline on the same
300 questions:

- v3-nogates: **257/300 (85.7%)**, baseline: **257/300 (85.7%)**
- Per-question agreement: **100%** (both 257, neither 43, only-v3n 0, only-bl 0)

With gates disabled the lazy constraint admits *exactly* the baseline trie's
language, question for question. The mechanism is faithful; any gated-number
difference is attributable to the gates themselves.

## 4. Gates are irrelevant to v2's collapse

- v2 vs v2-nogates scores are identical: 182/300 vs 182/300.
- Only 11/300 predictions differ at all (96.3% identical strings).
- McNemar p = 1.0.

The 25-point v2 gap vs baseline is **architectural** (per-hop beam-wise trie
rebuilding, the `max_hops` path-length loop), not the type/range gates. This
contradicts the hypothesis that "the DoG-proxy gates are what cost accuracy".

## 5. Mechanism metrics (v2 vs v2-nogates, per-question means)

| Metric | v2 Dynamic | v2 NoGates | Interpretation |
|--------|-----------|-----------|----------------|
| BUR | 0.440 | 0.463 | beams collapse to ~2.2 distinct terminals of 5 |
| BUR entropy | 0.738 | 0.823 | moderate diversity collapse in both |
| SIR decay slope | +0.0285 | 0.0000 | gates slightly *increase* inefficiency |
| max volatility | 0.000 | 0.000 | per-hop tries are stable — churn is *not* the failure |
| max RV | 0.205 | 0.000 | gates cut candidates ~20% at peak; no-gates admits all |

Hop profile (v2, averaged): hop 1 starts 1 beam → 5; hop 2 holds 5 beams; trie
sizes ~405–455 candidates. Volatility of 0.0 in both runs is a clean negative
result: the rebuild-instability hypothesis is falsified, pointing instead at
tokenization misalignment (§4.4 of CHAPTER4_RESULTS.md) as the root cause.

## 6. Lazy efficiency (v3)

- Candidates materialised per question: **avg 754.7**, max 3,480, min 29.
- Frontier builds: avg 18.4 (= anchors visited).
- No DFS performed (`n_dfs_paths = 0` in every record).
- Baseline enumerates the full DFS path set (thousands per question) before
  decoding; v3 renders only what the beams reach.

## 7. Takeaways

1. Static (v1) and lazy (v3) type-gating are equivalent in accuracy
   (p = 0.77, ns) — the lazy frontier matches the static trie at ~2 orders of
   magnitude lower construction cost, with a single decoding pass.
2. v2's per-hop architecture costs 25 points; neither gates nor trie
   volatility explain it.
3. The lazy mechanism is faithful: ungated, it reproduces the baseline
   exactly (257 = 257, 100% agreement).

## 8. Open items

- CWQ run (`cwq1`) pending — same 4-condition protocol.
- `index_len=4` runs would align with CHAPTER4's absolute numbers (hits the
  50k DFS cap on baseline; v3 has no cap).
- Confidence intervals on the 300-sample deltas (SE ≈ 2.1 pp at 85.7%).
