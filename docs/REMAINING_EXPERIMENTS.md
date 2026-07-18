# Remaining Experiments: Command Reference

## Status Summary

| Experiment | Status | Location | Notes |
|------------|--------|----------|-------|
| WebQSP GCR_Baseline | ✅ Complete | `run_full/` | 1,627 questions, 80.9% Hits@1 |
| WebQSP DCA_v1_Static | ✅ Complete | `run_full/` | 1,627 questions, 75.9% Hits@1 |
| WebQSP DCA_v2_Dynamic | ⚠️ Interrupted | `run_full/` | 1,466/1,628 questions, credit exhaustion |
| CWQ (all methods) | ❌ Not started | - | - |
| ORT improvement test | ❌ Not started | - | 50 samples only |

---

## 1. Complete WebQSP DCA_v2 Run

**Goal**: Finish remaining 162 questions (1,466 → 1,628)

```bash
# SSH into Vast.ai
ssh -p 16354 root@ssh2.vast.ai

# Navigate to project
cd /workspace/graph-constrained-reasoning

# Pull latest changes
git pull origin claude/fix-decoding-pipeline

# Run DCA_v2 on WebQSP (will resume from checkpoint)
python experiments/type_oracle_full/main.py \
  --datasets RoG-webqsp \
  --method v2 \
  --max-samples 999999 \
  --output-dir results/final_experiment/webqsp_v2
```

**Expected time**: ~30 minutes (162 questions × ~11s each)

---

## 2. Run CWQ Dataset (All Methods)

**Goal**: Evaluate on Complex WebQuestions (harder questions, 3-4 hops)

```bash
# SSH into Vast.ai
ssh -p 16354 root@ssh2.vast.ai

# Navigate to project
cd /workspace/graph-constrained-reasoning

# Run all methods on CWQ
python experiments/type_oracle_full/main.py \
  --datasets RoG-cwq \
  --max-samples 999999 \
  --output-dir results/final_experiment/cwq
```

**Note**: CWQ requires `index_len=4` for 4-hop questions. The script should auto-detect this, but verify in config.

**Expected time**: 3-4 hours (CWQ has more complex questions)

### CWQ Individual Methods (if needed)

```bash
# CWQ GCR_Baseline only
python experiments/type_oracle_full/main.py \
  --datasets RoG-cwq \
  --method baseline \
  --max-samples 999999 \
  --output-dir results/final_experiment/cwq_baseline

# CWQ DCA_v1 only
python experiments/type_oracle_full/main.py \
  --datasets RoG-cwq \
  --method v1 \
  --max-samples 999999 \
  --output-dir results/final_experiment/cwq_v1

# CWQ DCA_v2 only
python experiments/type_oracle_full/main.py \
  --datasets RoG-cwq \
  --method v2 \
  --max-samples 999999 \
  --output-dir results/final_experiment/cwq_v2
```

---

## 3. Test ORT Improvement (50 Samples)

**Goal**: Validate ORT-style improvements before full run

```bash
# SSH into Vast.ai
ssh -p 16354 root@ssh2.vast.ai

# Navigate to project
cd /workspace/graph-constrained-reasoning

# Pull latest changes (includes experiment_ort.py)
git pull origin claude/fix-decoding-pipeline

# Run ORT experiment with 50 samples
python experiments/type_oracle_full/experiment_ort.py \
  --max-samples 50 \
  --method ort-composed \
  --output-dir results/ort_experiment
```

**Expected time**: ~10 minutes (50 samples × ~12s each)

### Compare ORT vs TypeOracle (50 samples)

```bash
# Run TypeOracle baseline on same 50 samples for comparison
python experiments/type_oracle_full/main.py \
  --datasets RoG-webqsp \
  --method v1 \
  --max-samples 50 \
  --output-dir results/ort_experiment/baseline
```

---

## 4. Full ORT Experiment (if 50-sample test is promising)

```bash
# SSH into Vast.ai
ssh -p 16354 root@ssh2.vast.ai

# Navigate to project
cd /workspace/graph-constrained-reasoning

# Run ORT on full WebQSP
python experiments/type_oracle_full/experiment_ort.py \
  --max-samples 999999 \
  --method ort-composed \
  --output-dir results/ort_experiment/full_webqsp

# Run ORT on full CWQ
python experiments/type_oracle_full/experiment_ort.py \
  --datasets RoG-cwq \
  --max-samples 999999 \
  --method ort-composed \
  --output-dir results/ort_experiment/full_cwq
```

---

## Quick Reference: All Commands

### WebQSP (Complete remaining)

```bash
ssh -p 16354 root@ssh2.vast.ai "cd /workspace/graph-constrained-reasoning && \
  python experiments/type_oracle_full/main.py \
    --datasets RoG-webqsp \
    --method v2 \
    --max-samples 999999 \
    --output-dir results/final_experiment/webqsp_v2"
```

### CWQ (Full run)

```bash
ssh -p 16354 root@ssh2.vast.ai "cd /workspace/graph-constrained-reasoning && \
  python experiments/type_oracle_full/main.py \
    --datasets RoG-cwq \
    --max-samples 999999 \
    --output-dir results/final_experiment/cwq"
```

### ORT Test (50 samples)

```bash
ssh -p 16354 root@ssh2.vast.ai "cd /workspace/graph-constrained-reasoning && \
  git pull origin claude/fix-decoding-pipeline && \
  python experiments/type_oracle_full/experiment_ort.py \
    --max-samples 50 \
    --method ort-composed \
    --output-dir results/ort_experiment"
```

---

## Monitoring

### Check Progress

```bash
# Watch log file
tail -f results/final_experiment/cwq/run.log

# Check prediction count
wc -l results/final_experiment/cwq/RoG-cwq/predictions_*.jsonl

# Check for errors
grep -i error results/final_experiment/cwq/run.log
```

### Check Vast.ai Credits

```bash
# Check GPU usage
nvidia-smi

# Check disk space
df -h /workspace
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| Credit exhaustion | Vast.ai credits depleted | Add credits or stop run |
| CUDA out of memory | Model too large for GPU | Already using SDPA, should be fine |
| Timeout errors | Questions too complex | Increase `--sample-timeout-s` |
| Module not found | Wrong directory | Run from `/workspace/graph-constrained-reasoning` |

### Resume Interrupted Runs

The experiment script supports resuming from checkpoints. Simply re-run the same command and it will skip already-processed questions.

```bash
# Check what's been processed
wc -l results/final_experiment/webqsp_v2/RoG-webqsp/predictions_DCA_v2_Dynamic.jsonl

# Resume by re-running same command
python experiments/type_oracle_full/main.py \
  --datasets RoG-webqsp \
  --method v2 \
  --max-samples 999999 \
  --output-dir results/final_experiment/webqsp_v2
```

---

## Expected Results

### CWQ Dataset (estimates based on WebQSP)

| Method | Expected Hits@1 | Expected Hits@k |
|--------|-----------------|-----------------|
| GCR_Baseline | 75-80% | 85-90% |
| DCA_v1_Static | 70-75% | 80-85% |
| DCA_v2_Dynamic | 50-60% | 50-60% |

### ORT Improvement (target)

| Metric | TypeOracle | ORT Target |
|--------|------------|------------|
| Hits@1 | 75.9% | >75.9% |
| Path Reduction | 14.5% | >14.5% |

---

## Cost Estimate (Vast.ai)

| Experiment | Duration | Est. Cost |
|------------|----------|-----------|
| WebQSP DCA_v2 completion | 30 min | ~$0.50 |
| CWQ full run | 3-4 hours | ~$5-7 |
| ORT 50-sample test | 10 min | ~$0.25 |
| ORT full run | 2-3 hours | ~$3-5 |
| **Total** | ~6-8 hours | **~$9-13** |

---

*Last updated: July 16, 2026*
