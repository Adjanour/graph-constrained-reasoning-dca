# All Experiment Commands (Vast.ai)

```bash
ssh -p 11812 root@ssh8.vast.ai
cd /workspace/graph-constrained-reasoning
git pull
source /venv/main/bin/activate
```

---

## 1. 4 Ideas — WebQSP Full (all 7 methods, ~4 hrs)

```bash
python experiments/type_oracle_full/experiment_4_ideas.py \
  --model-path rmanluo/GCR-Meta-Llama-3.1-8B-Instruct \
  --dataset RoG-webqsp --max-samples 999999 \
  --methods filtered,validate,adaptive30,adaptive100,adaptive500,label-plan
```

## 2. 4 Ideas — CWQ Full (harder, 3-4 hop, ~8 hrs)

```bash
python experiments/type_oracle_full/experiment_4_ideas.py \
  --model-path rmanluo/GCR-Meta-Llama-3.1-8B-Instruct \
  --dataset RoG-cwq --max-samples 999999 \
  --methods baseline,filtered,validate,adaptive30,adaptive100,adaptive500,label-plan,v2
```

## 3. Small Model Comparison — Qwen 3B on WebQSP

Does filtering help small models more? (100 samples, ~10 min)

```bash
python experiments/type_oracle_full/experiment_4_ideas.py \
  --model-path Qwen/Qwen2.5-3B-Instruct \
  --dataset RoG-webqsp --max-samples 100 \
  --methods baseline,filtered
```

## 4. Adaptive Budget — WebQSP (tradeoff curve + v2 premium, ~4 hrs)

```bash
python experiments/type_oracle_full/experiment_adaptive_budget.py \
  --model-path rmanluo/GCR-Meta-Llama-3.1-8B-Instruct \
  --dataset RoG-webqsp --max-samples 999999 \
  --methods baseline,adaptive30,adaptive100,adaptive500,adaptive-budget,v2
```

## 5. Adaptive Budget — CWQ (does v2 shine on deep graphs? ~8 hrs)

```bash
python experiments/type_oracle_full/experiment_adaptive_budget.py \
  --model-path rmanluo/GCR-Meta-Llama-3.1-8B-Instruct \
  --dataset RoG-cwq --max-samples 999999 \
  --methods baseline,adaptive30,adaptive100,adaptive500,adaptive-budget,v2
```

---

## Retrieve Results (from your machine)

```bash
scp -P 11812 -r root@ssh8.vast.ai:/workspace/graph-constrained-reasoning/results/ ./vast_results/
```
