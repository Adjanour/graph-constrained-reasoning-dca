# 4 Ideas Experiment — Vast.ai Runbook

**Script**: `experiments/type_oracle_full/experiment_4_ideas.py`
**Goal**: Test 4 hypotheses on N samples (10 dry-run → 100 full)

---

## 1. Push Latest Code

```bash
cd /home/bernard/research/projects/graph-constrained-reasoning
git add -A && git commit -m "4 ideas experiment script" && git push
```

---

## 2. SSH into Vast.ai

```bash
ssh -p 16354 root@ssh2.vast.ai
cd /workspace/graph-constrained-reasoning
git pull
source /venv/main/bin/activate
```

---

## 3. 10-Sample Dry Run (validates all conditions work)

Takes ~5 min on RTX 4090, ~15 min on A100.

### 3a. All 4 ideas with 8B model (GCR)

```bash
python experiments/type_oracle_full/experiment_4_ideas.py \
  --model-path rmanluo/GCR-Meta-Llama-3.1-8B-Instruct \
  --dataset RoG-webqsp \
  --max-samples 10 \
  --methods baseline,filtered,validate,adaptive30,adaptive100,label-plan \
  --output-dir results/4_ideas_dryrun_8b
```

### 3b. Small model comparison (Qwen2.5-3B)

```bash
python experiments/type_oracle_full/experiment_4_ideas.py \
  --model-path Qwen/Qwen2.5-3B-Instruct \
  --dataset RoG-webqsp \
  --max-samples 10 \
  --methods baseline,filtered \
  --output-dir results/4_ideas_dryrun_3b
```

---

## 4. 100-Sample Full Run

Takes ~45-60 min on RTX 4090.

### 4a. Main sweep (8B, all 4 ideas)

```bash
python experiments/type_oracle_full/experiment_4_ideas.py \
  --model-path rmanluo/GCR-Meta-Llama-3.1-8B-Instruct \
  --dataset RoG-webqsp \
  --max-samples 100 \
  --methods baseline,filtered,validate,adaptive30,adaptive100,adaptive500,label-plan \
  --output-dir results/4_ideas_full_run \
  --sample-timeout 180
```

### 4b. Small model comparison (3B, baseline vs filtered)

```bash
python experiments/type_oracle_full/experiment_4_ideas.py \
  --model-path Qwen/Qwen2.5-3B-Instruct \
  --dataset RoG-webqsp \
  --max-samples 100 \
  --methods baseline,filtered \
  --output-dir results/4_ideas_3b_comparison
```

---

## 5. Run in Background (Recommended)

```bash
# Run with nohup so it survives SSH disconnects
nohup python experiments/type_oracle_full/experiment_4_ideas.py \
  --model-path rmanluo/GCR-Meta-Llama-3.1-8B-Instruct \
  --dataset RoG-webqsp \
  --max-samples 100 \
  --methods baseline,filtered,validate,adaptive30,adaptive100,adaptive500,label-plan \
  --output-dir results/4_ideas_full_run \
  > /workspace/experiment_4ideas.log 2>&1 &

tail -f /workspace/experiment_4ideas.log
```

---

## 6. Monitor Log Output

When running, you'll see progress like:

```
2026-07-18 12:00:00 [INFO    ] Loading model: rmanluo/GCR-Meta-Llama-3.1-8B-Instruct
2026-07-18 12:00:01 [INFO    ] GPU available: True
2026-07-18 12:02:30 [INFO    ] Model loaded in 148.2s
...
2026-07-18 12:02:35 [INFO    ] ============================================================
2026-07-18 12:02:35 [INFO    ]   METHOD: baseline
2026-07-18 12:02:35 [INFO    ] ============================================================
2026-07-18 12:02:35 [INFO    ]   Samples: 100
2026-07-18 12:02:35 [INFO    ]   [baseline] 10/100 done | 63.5s | 0 skip 0 dead 0 timeout
2026-07-18 12:02:35 [INFO    ]   [baseline] 20/100 done | 127.0s | 0 skip 0 dead 0 timeout
...
2026-07-18 12:10:30 [INFO    ]   -> baseline: Hits@1=92/100 (92.0%) in 480.5s
```

Final summary looks like:

```
============================================================
                       FINAL RESULTS
============================================================
Method               N   Hits@1    Hit%     Time    Avg/q
------------------------------------------------------------
baseline            100      92   92.0%    480s    4.80s
filtered            100      87   87.0%    490s    4.90s
validate            100      92   92.0%    480s    4.80s
adaptive30          100      70   70.0%    120s    1.20s
adaptive100         100      82   82.0%    180s    1.80s
adaptive500         100      89   89.0%    300s    3.00s
label-plan          100      87   87.0%    485s    4.85s
============================================================

=== Validation Analysis (Idea 1) ===
  Wrong predictions: 8
  Caught by TypeOracle: 2 (25.0%)
  False positives (correct preds rejected): 1 (1.1%)
```

---

## 7. Retrieve Results

From your **local** machine (after run finishes):

```bash
# Use SSH details from Vast.ai dashboard
scp -P PORT -r root@IP:/workspace/graph-constrained-reasoning/results/4_ideas_full_run/ \
  ./results_from_vast/4_ideas_full_run/

# Also grab the log
scp -P PORT root@IP:/workspace/experiment_4ideas.log \
  ./results_from_vast/
```

---

## 8. Analyze Results

Check `summary.json` for each run:

```bash
cat results/4_ideas_full_run/summary.json
```

Fields of interest:

| Field | What it tells you |
|-------|-------------------|
| `baseline.hit_at_1` | GCR ceiling for this model/dataset |
| `filtered.hit_at_1` | Does TypeOracle help or hurt? |
| `validate._validate_analysis.catch_rate_pct` | What % of wrong preds are type-invalid? |
| `validate._validate_analysis.false_positive_rate_pct` | What % of correct preds get wrongly rejected? |
| `adaptive30/100/500.hit_at_1` | Accuracy at reduced path budgets |
| `adaptive30/100/500.avg_time_per_q` | Latency at reduced path budgets |
| `label-plan.hit_at_1` | Reverse/label-level approach effectiveness |

---

## Quick Reference (Everything in One Command)

```bash
# Dry run (10 samples) on Vast.ai
cd /workspace/graph-constrained-reasoning && \
source /venv/main/bin/activate && \
python experiments/type_oracle_full/experiment_4_ideas.py \
  --model-path rmanluo/GCR-Meta-Llama-3.1-8B-Instruct \
  --max-samples 10 \
  --methods baseline,filtered,validate,adaptive30,adaptive100,label-plan
```
