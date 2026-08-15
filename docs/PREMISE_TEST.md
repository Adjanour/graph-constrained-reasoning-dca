# E1/E2 Premise Test on Vast.ai

Run the premise test: does **filtering by relevance** (E1) or by **correct answer
types** (E2) improve rank-1 accuracy over the unfiltered GCR trie? This is the
causal test that separates "the gates hurt because they prune gold paths" from
"the gates hurt because filtering irrelevant paths is inherently bad."

- **Script**: `scripts/run_premise_test.sh`
- **Experiment entrypoint**: `experiments/type_oracle_full/main.py --method premise`
- **Branch**: `premise-test` (the script checks this branch out on the rented box)

---

## Prerequisites

1. `git`, `ssh`, `scp`, `jq`, and the `vastai` CLI
   (`uv pip install vastai && uv run vastai set api-key`).
2. Your Vast.ai API key configured, with credit on the account.
3. `HF_TOKEN` (a Hugging Face read token) exported if you use gated checkpoints
   (e.g. `meta-llama/Llama-3.1-8B-Instruct`):
   ```bash
   export HF_TOKEN=hf_xxx
   ```
4. GPU: none locally — the script rents one (default `RTX_4090`, 200 GB disk).

---

## 1. Push the branch (one-time)

The script runs `git fetch origin && git checkout premise-test` on the box, so the
branch must exist on `origin` first:

```bash
git checkout premise-test
git push -u origin premise-test
```

If `premise-test` doesn't exist locally yet, create it from the current code:

```bash
git checkout -b premise-test
git add experiments/type_oracle_full/experiment.py \
        experiments/type_oracle_full/main.py \
        scripts/run_premise_test.sh
git commit -m "feat: add E1/E2 premise test"
git push -u origin premise-test
```

> The branch carries only the experiment code. The offline analysis scripts in
> `workflow/` are not needed on the box and are not part of the branch.

---

## 2. Run the test

```bash
BRANCH=premise-test bash scripts/run_premise_test.sh \
  --models "gcr llama" --samples 300 --run-name premise
```

What happens:

1. Searches Vast.ai for an `RTX_4090` offer (add `--offer <id>` to pin one,
   `--gpu A100_40GB` / `--region eu` to change the search).
2. Rents the instance, uploads and runs `scripts/vast_boot.sh`
   (clone + `setup-env.sh` CUDA torch + flash-attn).
3. On the box: checks out `premise-test`, then runs
   `main.py --method premise --max-samples 300 --run-name premise
   --model-path <gcr-checkpoint> <llama-checkpoint>`.
4. Waits until the log contains `ALL MODELS DONE`.
5. Copies `results/` and `experiment.log` into
   `results_from_vast/premise/`.
6. Asks whether to destroy the instance.

### Model aliases

| Alias     | HF checkpoint                            | Notes                                    |
|-----------|------------------------------------------|------------------------------------------|
| `gcr`     | `rmanluo/GCR-Meta-Llama-3.1-8B-Instruct` | the fine-tuned GCR model (default)       |
| `llama`   | `meta-llama/Llama-3.1-8B-Instruct`       | base Llama 3.1 8B (gated, needs HF_TOKEN)|
| `mistral` | `mistralai/Mistral-7B-Instruct-v0.3`     |                                          |
| `qwen`    | `Qwen/Qwen2.5-7B-Instruct`               |                                          |

Any value that is not an alias is treated as a full HF checkpoint path and passed
through verbatim.

### Useful flags

```bash
--offer 44169006        # rent a specific GPU offer id
--gpu A100_40GB         # search for a different GPU
--region eu             # restrict search to European hosts
--max-hours 6           # runtime budget for the experiment
--search-only           # just list offers, do not rent
--run-name my-run       # name the results subdirectory (default: premise-test)
--samples 300           # questions per dataset (default 300)
```

Full usage: `bash scripts/run_premise_test.sh --help`.

---

## 3. The four conditions (per model)

`main.py --method premise` runs these on every dataset:

| Condition        | Trie filter                                          | What it answers                              |
|------------------|------------------------------------------------------|----------------------------------------------|
| `GCR_Baseline`   | unfiltered trie (control)                            | true rank-1 accuracy of the model            |
| `E1_GoldRelevant`| keep only paths whose **terminal is a gold answer entity** | ceiling if relevance filtering were perfect |
| `E2_GoldTypes`   | keep only paths whose terminal passes a gate over **gold-derived answer types** | does the *type* lever help when types are right (FNR ≈ 0) |

Gap to read from `summary.json`:

- **E1 vs Baseline** → the headroom of relevance filtering at decode time.
- **E2 vs Baseline** → the value of correct types (isolates the type lever from
  the range lever).
- A small E1 gain with a large deployed-gate loss (paper: T3 −6.3 pp) pins the
  blame on **FNR / type quality**, not on the idea of filtering.

---

## 4. Monitoring

The script polls and prints progress. To watch manually:

```bash
# from the script output you get the instance SSH endpoint; then:
ssh -p <PORT> root@<HOST> 'tail -f /workspace/experiment.log'
```

Per-question progress files live under
`/workspace/graph-constrained-reasoning/results/final_experiment/`.

---

## 5. Results

After the run, results are in `results_from_vast/premise/<model-slug>/` (the script
copies the box's `results/final_experiment/premise/` directory into
`results_from_vast/`):

```
results_from_vast/premise/
├── rmanluo__GCR-Meta-Llama-3.1-8B-Instruct/   # model slug: '/' -> '__'
│   ├── config.json
│   ├── summary.json                    # per-condition metrics
│   └── RoG-webqsp/ ...                 # per-dataset predictions
├── meta-llama__Llama-3.1-8B-Instruct/
│   └── ...
└── experiment.log
```

Each model's `summary.json` holds the per-condition metrics; compare
`GCR_Baseline` vs `E1_GoldRelevant` vs `E2_GoldTypes` within one model
(same 300 questions, paired).

---

## 6. Cleanup

The script asks to destroy the instance when done. If you declined:

```bash
uv run vastai destroy instance <INSTANCE_ID>
```

Billing is hourly and continues while the instance is up — destroy when finished.

---

## Troubleshooting

- **`vastai` not found** → `cd <repo> && uv pip install vastai && uv run vastai set api-key`.
- **No offers** → loosen the search: `--gpu A100_40GB`, drop the region filter,
  or run `--search-only` and pick an id with `--offer`.
- **HF auth fails for `llama`** → export `HF_TOKEN` and rerun; gated models need it.
- **Box setup times out** → the script prints the log path; run
  `ssh -p <PORT> root@<HOST> 'cat /workspace/vast_boot.log'`.
- **Experiment process dies early** → the script dumps the last 20 log lines and
  leaves the instance up for manual fixes.
- **Prediction files empty for E1 on many samples** → those questions have no
  gold-reachable path within the hop limit; E1 skips them (`n_empty_tries` in the
  summary). This is expected and is itself a finding (reachability ceiling).