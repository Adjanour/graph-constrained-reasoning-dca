# Vast.ai Experiment Commands

## 1. SSH + Pull (always first)

```bash
ssh -p PORT root@HOST
cd /workspace/graph-constrained-reasoning
git pull
source /venv/main/bin/activate
```

## 2. Dry Run (10 samples, verify everything works)

### v2 on CWQ
```bash
python experiments/type_oracle_full/main.py \
  --datasets RoG-cwq --method v2 --max-samples 10 \
  --output-dir results/cwq_dryrun
```

### All methods on CWQ
```bash
python experiments/type_oracle_full/main.py \
  --datasets RoG-cwq --method all --max-samples 10 \
  --output-dir results/cwq_dryrun_all
```

## 3. Full Runs (background, survive SSH disconnect)

### v2 on CWQ (~3.5K questions, 3-4 hrs)
```bash
nohup python experiments/type_oracle_full/main.py \
  --datasets RoG-cwq --method v2 --max-samples 999999 \
  --output-dir results/cwq_v2 \
  > /workspace/cwq_v2.log 2>&1 &

tail -f /workspace/cwq_v2.log
```

### All methods on CWQ (baseline + v1 + v2, ~8-10 hrs)
```bash
nohup python experiments/type_oracle_full/main.py \
  --datasets RoG-cwq --method all --max-samples 999999 \
  --output-dir results/cwq_all \
  > /workspace/cwq_all.log 2>&1 &

tail -f /workspace/cwq_all.log
```

### v2 on WebQSP (compare against old buggy 54.9%)
```bash
nohup python experiments/type_oracle_full/main.py \
  --datasets RoG-webqsp --method v2 --max-samples 999999 \
  --output-dir results/webqsp_v2_fixed \
  > /workspace/webqsp_v2_fixed.log 2>&1 &

tail -f /workspace/webqsp_v2_fixed.log
```

## 4. Retrieve Results

### From your machine (replace PORT + HOST with Vast.ai details)
```bash
scp -P PORT -r root@HOST:/workspace/graph-constrained-reasoning/results/ ./results_from_vast/
scp -P PORT root@HOST:/workspace/cwq_v2.log ./results_from_vast/
```

## 5. Automated Launch (local machine)

Full lifecycle: search → rent → setup → run → download → destroy:

```bash
cd /home/bernard/research/projects/graph-constrained-reasoning
source .venv/bin/activate

# Dry run on CWQ
bash scripts/run_vast.sh --max-samples 10 --datasets RoG-cwq --method all

# Full v2 on CWQ
bash scripts/run_vast.sh --datasets RoG-cwq --method v2 --max-samples 999999
```

## 6. Experiments (updated main)

The 4 ideas experiment script has been updated. Run it instead of `main.py`:

```bash
python experiments/type_oracle_full/experiment_4_ideas.py \
  --model-path rmanluo/GCR-Meta-Llama-3.1-8B-Instruct \
  --dataset RoG-webqsp --max-samples 100 \
  --methods baseline,filtered,validate,adaptive30,adaptive100,adaptive500,label-plan \
  --output-dir results/4_ideas_full_run
```
