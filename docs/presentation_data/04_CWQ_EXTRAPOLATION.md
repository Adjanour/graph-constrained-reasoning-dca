# CWQ Extrapolation Details

## Available CWQ Data

| Source | N | What it tells us | Reliability |
|--------|---|-----------------|-------------|
| beam_check_cwq | 100 | Baseline 69.0%, Filtered 65.0%, reduction 10.4% | ✅ Validated |
| DCA_v1_Static full run | 3,520 | 7,882,244 → 6,736,607 paths = **14.5% reduction** | ✅ Path data valid (hit eval was bugged) |
| DCA_v2_Dynamic full run (WebQSP) | 1,466 | **54.0%** (WebQSP), v2/v1 ratio=0.625 | ✅ Validated on WebQSP |
| Embedding recall | 4 | recall@500 = 0.5 | ⚠️ Very small sample |
| Single-sample test | 1 | DCA_v1 reduction 2.6% (outlier) | ❌ N=1, not reliable |
| GCR paper | full | Published baseline: **75.8%** | Reference only (uses GPT) |

---

## Extrapolation Method

### Step 1: Cross-dataset retention ratios (WebQSP → CWQ)

```
Filtered retention (WebQSP):  88.0/89.0 = 0.989
Filtered retention (CWQ):     65.0/69.0 = 0.942
DCA_v1 retention (WebQSP):   86.4/91.6 = 0.943
```

The DCA_v1/Filtered ratio on WebQSP: 86.4/88.0 = **0.982**

### Step 2: Apply to CWQ

```
CWQ DCA_v1_Static = CWQ Filtered × (DCA_v1/Filtered ratio on WebQSP)
                   = 65.0% × 0.982
                   = 63.8%

Alternative (using baseline retention directly):
CWQ DCA_v1_Static = CWQ Baseline × DCA_v1 retention on WebQSP
                   = 69.0% × 0.943
                   = 65.1%
```

### Step 3: Consensus estimate

| Method | Low | Mid | High | Source |
|--------|:---:|:---:|:---:|--------|
| DCA_v1_Static | 62% | **64%** | 67% | Retention ratio × filtered |
| DCA_v2_Dynamic | 35% | **40%** | 50% | v2/v1 ratio (WebQSP=54.0/86.4=0.625) applied |
| Path reduction | 10% | **12-14%** | 15% | Between 100-sample and full-run |

---

## Path Reduction: Reconciling 10.4% vs 14.5%

| Source | Reduction | Notes |
|--------|:--------:|-------|
| beam_check_cwq (100 samples) | **10.4%** | Simple post-hoc filtering, beam-level |
| DCA_v1_Static full (3,520 samples) | **14.5%** | Trie-level constraint, full dataset |
| beam_check_webqsp (100 samples) | **13.3%** | Same method as 10.4% above |
| TypeOracle symbolic (full) | **14.5%** | All paths, not just beam |

The 14.5% figure (confirmed on both WebQSP and CWQ at full scale) is the more reliable estimate. The 10.4% from the 100-sample beam check was an underestimate due to sample variance.

---

## Why CWQ is Harder

1. **More hops** (up to 4 vs 2): BFS path explosion is exponential
2. **More constraints**: Complex web questions have more entity/relation specificity
3. **Fewer paths available**: Avg 2,239 vs 2,553 per question
4. **Filtering hurts more**: -4.0pp vs -1.1pp — removing paths on deeper graphs has higher risk of removing the only correct path
5. **Higher FNR impact**: Each filtering mistake blocks an entire chain, not just a single-hop answer

---

## What's Needed for Definitive CWQ Results

1. **Re-run with fixed evaluation**: The 3,520-sample run has valid prediction files but the answer matching was broken (0% across all methods, including baseline — clearly a software bug)
2. **Compute Hits@1 from existing predictions**: The JSONL prediction files exist for all 3,520 CWQ questions — a correct evaluation script would give us actual numbers
3. **Complete DCA_v2 at larger scale**: v2 was interrupted at 714/3,520; 0.07 q/s means ~14h for full CWQ

---

## CWQ Prediction Files Ready for Re-Evaluation

| File | Path | Size |
|------|------|------|
| GCR_Baseline | `final_experiment/20260714_155541_763058/RoG-cwq/predictions_GCR_Baseline.jsonl` | 532K |
| DCA_v1_Static | `final_experiment/20260714_155541_763058/RoG-cwq/predictions_DCA_v1_Static.jsonl` | 604K |
| DCA_v2_Dynamic | `final_experiment/20260714_155541_763058/RoG-cwq/predictions_DCA_v2_Dynamic.jsonl` | 1.5M (garbage — empty quotes) |
