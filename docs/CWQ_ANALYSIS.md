# CWQ Controlled 300-Question Analysis (cwq1)

Run: `experiments/type_oracle_full/main.py --method lazy-ablation --datasets RoG-cwq --max-samples 300 --run-name cwq1`, seed 42, model `rmanluo/GCR-Meta-Llama-3.1-8B-Instruct`, beam k=10, max_new_tokens=256, bf16, flash_attention_2.
Duration: 3.28h wall, ~$1.16 on Vast.ai (0.3533 $/h).
Results: `results_from_vast/cwq1/RoG-cwq/predictions_*.jsonl`.

## Accuracy (official eval, substring match, 300q, seed 42)

| Condition | Hits@1 | vs baseline |
|---|---|---|
| GCR_Baseline | 49.0% (147/300) | — |
| DCA_v1_Static | 43.7% (131/300) | **−5.3pp** |
| DCA_v3_Lazy | 48.0% (144/300) | −1.0pp |
| DCA_v3_NoGates | 49.0% (147/300) | 0.0pp (calibration) |

## Per-question agreement

- base vs v1: 236/300 identical (78.7%); both correct 126
- base vs v3: 270/300 (90.0%); both correct 140
- base vs v3nogates: **300/300 (100.0%)** — calibration exact; both correct 147
- v1 vs v3: 245/300 (81.7%); both correct 128

## McNemar test

- base vs v1: p=0.0017 ** (significant drop)
- base vs v3: p=0.3657 ns
- base vs v3nogates: p=1.0000
- v1 vs v3: p=0.0029 ** (significant)
- v3 vs v3nogates: p=0.3657

## Timing (300q, per question)

| Condition | Total | Mean/q | q/s |
|---|---|---|---|
| GCR_Baseline | 2891s | 9.64s | 0.104 |
| DCA_v1_Static | 3116s | 10.39s | 0.096 |
| DCA_v3_Lazy | 2750s | 9.17s | 0.109 |
| DCA_v3_NoGates | 2760s | 9.20s | 0.109 |

Graph size (CWQ): `n_paths_all` mean 2004, max 9866, min 15; `n_nodes` mean 1179 (max 1995); `n_dfs` = 2004 for gated conditions, 0 for lazy v3 (lazy does not enumerate full path set).

## SIR (v1 only; lazy/nogates don't enumerate full path set)

- v1: all_total=601,256, kept=521,944 → **SIR = 0.132** (13.2% path reduction), consistent with WebQSP's 0.138.
- Note: the earlier "SIR 0.168" figure from the 100q verification run was stale and is superseded by this 300q controlled measurement.

## Interpretation

- v1 static gating costs a significant −5.3pp (p=0.0017), replicating the WebQSP pattern on a harder dataset. CWQ graph is ~2x WebQSP (2004 vs ~1300 avg paths), so gate rejection windows are larger and the accuracy cost is slightly larger (−5.3 vs −4.0pp).
- v3 lazy is within noise of baseline (−1.0pp, p=0.37) and is the fastest condition (9.17s/q vs 9.64s baseline).
- v3-nogates ≡ baseline exactly (300/300 identical, p=1.0): the gates, not the decoding path, are the source of v1's accuracy cost. Same as WebQSP.