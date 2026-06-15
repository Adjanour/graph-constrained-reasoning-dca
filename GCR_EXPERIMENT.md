# Graph-Constrained Reasoning (GCR) Experiment

This document describes the original GCR experiment pipeline, how the scripts work, and the core mechanisms that make it function.

---

## Two-Stage Pipeline

The experiment runs in two sequential stages, each with its own script and shell wrapper.

### Stage 1: Graph-Constrained Decoding

| | |
|---|---|
| **Script** | `workflow/predict_paths_and_answers.py` |
| **Shell wrapper** | `scripts/graph_constrained_decoding.sh` |
| **Model** | Fine-tuned lightweight LLM (e.g. Llama-3.1-8B) |
| **Output** | `results/GenPaths/{dataset}/{model}/{split}/predictions.jsonl` |

This is the core innovation. A fine-tuned lightweight LLM generates reasoning paths **constrained to the knowledge graph**.

**Step-by-step flow:**

1. **Load the KG** -- For each question, build a NetworkX graph from the `graph` field (a list of `(head, relation, tail)` triples).

2. **Enumerate valid paths** -- DFS from the question entities up to `index_path_length` (default 2 hops), collecting all possible paths.

   ```
   Justin Bieber -> people.person.parents -> Jeremy Bieber -> ...
   ```

3. **Build a Marisa Trie** -- Tokenize all valid paths and insert them into a prefix trie. This trie encodes *every valid token sequence* that corresponds to a real KG path.

4. **Constrained beam-search generation** -- The LLM generates text, but at each decoding step, a `prefix_allowed_tokens_fn` callback (`src/graph_constrained_decoding.py:29`) queries the trie. Only tokens that would produce a valid KG path are allowed. Generation is triggered inside `src/llms/graph_constrained_decoding_model.py:8` via HuggingFace's `model.generate(..., prefix_allowed_tokens_fn=gcr.allowed_tokens_fn)`.

5. **Output** -- Generated paths and predicted answers are saved as JSONL, with one record per question containing the predicted paths, ground-truth paths, and the answer.

**Key constraint mechanism** (`src/graph_constrained_decoding.py`):

- When the model is generating inside a `<PATH>...</PATH>` block, `allowed_tokens_fn` queries the trie for the next valid tokens.
- Outside path blocks (free text / answer generation), all tokens are allowed.
- This guarantees **zero hallucinated paths** -- every generated path is grounded in the KG.

---

### Stage 2: Graph Inductive Reasoning

| | |
|---|---|
| **Script** | `workflow/predict_final_answer.py` |
| **Shell wrapper** | `scripts/graph_inductive_reasoning.sh` |
| **Model** | General-purpose LLM (e.g. GPT-3.5-turbo) |
| **Input** | `predictions.jsonl` from Stage 1 |

A general-purpose LLM reads the reasoning paths from Stage 1 and produces the final answer. This stage requires no training -- it is pure prompting. It uses `PromptBuilder` (`src/qa_prompt_builder.py`) which formats the paths as context in the prompt.

---

## Overall Flow

```
KG + Question
    |
    v
[Stage 1: Graph-Constrained Decoding]
    |  Fine-tuned LLM generates paths constrained by KG trie
    |  Output: reasoning paths + candidate answers
    v
[Stage 2: Graph Inductive Reasoning]
    |  General LLM reasons over paths -> final answer
    v
Final Answer
```

---

## Shell Scripts

### `scripts/graph_constrained_decoding.sh`

Runs Stage 1 on both WebQSP and CWQ datasets with beam size 10, group-beam search, and flash attention.

```bash
MODEL_PATH=rmanluo/GCR-Meta-Llama-3.1-8B-Instruct
MODEL_NAME=$(basename "$MODEL_PATH")

python workflow/predict_paths_and_answers.py \
  --data_path rmanluo \
  --d {RoG-webqsp,RoG-cwq} \
  --split test \
  --index_path_length 2 \
  --model_name ${MODEL_NAME} \
  --model_path ${MODEL_PATH} \
  --k 10 \
  --prompt_mode zero-shot \
  --generation_mode group-beam \
  --attn_implementation flash_attention_2
```

### `scripts/graph_inductive_reasoning.sh`

Takes the `predictions.jsonl` from Stage 1 and feeds it to GPT-3.5-turbo for final answer prediction with 10 parallel threads.

```bash
MODEL_NAME=gpt-3.5-turbo
N_THREAD=10

REASONING_PATH="results/GenPaths/${DATA}/rmanluo/GCR-Meta-Llama-3.1-8B-Instruct/test/zero-shot-group-beam-k10/predictions.jsonl"

python workflow/predict_final_answer.py \
  --data_path rmanluo \
  --d {RoG-webqsp,RoG-cwq} \
  --split test \
  --model_name ${MODEL_NAME} \
  --reasoning_path ${REASONING_PATH} \
  --add_path True \
  -n ${N_THREAD}
```

---

## Additional Experiment: TypeOracle

| | |
|---|---|
| **Script** | `workflow/run_symbolic_experiment.py` |
| **Model** | TypeOracle (symbolic, no encoder) + optional Qwen2.5-3B via llama.cpp |

This is a separate experiment for evaluating a symbolic path-filtering oracle (TypeOracle) that prunes paths *before* they reach the LLM. It has two phases:

**Phase 1 -- SIR/FNR (CPU-only, full test set):**

- Evaluates TypeOracle gate pruning across the full WebQSP test set.
- Measures SIR (Semantic Irrelevance Ratio) -- what fraction of paths are pruned.
- Measures FNR (False Negative Rate) -- what fraction of gold paths are incorrectly removed.
- Uses two gates: a **range gate** (relation-tail compatibility) and a **type gate** (terminal entity type compatibility with the inferred answer type).

**Phase 2 -- Proxy answer generation (llama.cpp, subset):**

- Uses a small local model (Qwen2.5-3B Q4_K_M) with TypeOracle-filtered paths in the prompt to generate answers.
- Compares Hit@1 with and without TypeOracle filtering.
- Requires llama.cpp server running on port 8080.

**Usage:**

```bash
# Phase 1: SIR/FNR (CPU, full test set)
python workflow/run_symbolic_experiment.py --phase sir

# Phase 2: Proxy answers with llama.cpp (Intel GPU, subset)
python workflow/run_symbolic_experiment.py --phase proxy --n 10

# Both phases sequentially
python workflow/run_symbolic_experiment.py --phase all --n 10
```

---

## Key Source Files

| File | Purpose |
|------|---------|
| `src/graph_constrained_decoding.py` | The trie-based constraint callback for `model.generate()` |
| `src/llms/graph_constrained_decoding_model.py` | Wraps HuggingFace causal model with constrained generation |
| `src/qa_prompt_builder.py` | Builds prompts, enumerates KG paths, constructs the Marisa Trie |
| `src/utils/graph_utils.py` | Graph construction, DFS path enumeration, truth-path extraction |
| `src/trie.py` | Marisa Trie wrapper for fast prefix lookup |
| `workflow/predict_paths_and_answers.py` | Stage 1 entry point |
| `workflow/predict_final_answer.py` | Stage 2 entry point |
| `workflow/run_symbolic_experiment.py` | TypeOracle SIR/FNR + proxy experiment |
| `scripts/graph_constrained_decoding.sh` | Shell wrapper for Stage 1 |
| `scripts/graph_inductive_reasoning.sh` | Shell wrapper for Stage 2 |
