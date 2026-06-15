# TECHNICAL_REFERENCE

Technical reference for the Graph-Constrained Reasoning (GCR) codebase — ICML 2025.

---

## 1. Two-Stage Pipeline

### Stage 1: Graph-Constrained Decoding

| | |
|---|---|
| **Script** | `workflow/predict_paths_and_answers.py` |
| **Shell wrapper** | `scripts/graph_constrained_decoding.sh` |
| **Model** | Fine-tuned lightweight LLM (e.g. GCR-Meta-Llama-3.1-8B-Instruct) |
| **Output** | `results/GenPaths/{dataset}/{model}/{split}/predictions.jsonl` |

Enumerate KG paths via DFS, build a MarisaTrie of valid token sequences, run `model.generate()` with a `prefix_allowed_tokens_fn` callback that masks invalid tokens at every decoding step.

### Stage 2: Graph Inductive Reasoning

| | |
|---|---|
| **Script** | `workflow/predict_final_answer.py` |
| **Shell wrapper** | `scripts/graph_inductive_reasoning.sh` |
| **Model** | General-purpose LLM (e.g. GPT-3.5-turbo, or local via llama.cpp) |
| **Input** | `predictions.jsonl` from Stage 1 |

A general-purpose LLM reads the reasoning paths from Stage 1 and produces the final answer. Pure prompting, no training.

### ASCII Flow Diagram

```
KG + Question
    │
    ▼
[Stage 1: Graph-Constrained Decoding]
    │  DFS paths → MarisaTrie → constrained generate()
    │  Output: reasoning paths + candidate answers
    ▼
[Stage 2: Graph Inductive Reasoning]
    │  General LLM reasons over paths → final answer
    ▼
Final Answer
```

---

## 2. Model Loading

### Registry Pattern

`src/llms/__init__.py` (18 lines)

```python
registed_language_models = {
    'gpt': ChatGPT, 'others': HfCausalModel,
    'gcr': GraphConstrainedDecodingModel, 'proxy': LLMProxy,
}
def get_registed_model(model_name) -> BaseLanguageModel:
    for key, value in registed_language_models.items():
        if key in model_name.lower():
            return value
    return HfCausalModel
```

Substring match: `"rmanluo/GCR-Meta-Llama-3.1-8B-Instruct"` contains `"gcr"` → `GraphConstrainedDecodingModel`.

### HfCausalModel

`src/llms/base_hf_causal_model.py` (212 lines) — `prepare_for_inference()` loads:

1. **Tokenizer**: `AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)`
2. **Model config**: `AutoConfig.from_pretrained(model_path)` — overrides `_attn_implementation`
3. **Model weights**: `AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=...)`

### Attention Fallback

```python
attn = self.args.attn_implementation
if attn == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
    attn = "sdpa"
```

Silent fallback: flash_attention_2 → sdpa when `flash_attn` is missing. T4 (sm_75) requires `--attn_implementation sdpa`.

### Generation Modes

| Mode | Config |
|------|--------|
| `greedy` | `do_sample=False, num_return_sequences=1` |
| `beam` | `do_sample=False, num_beams=k, num_return_sequences=k` |
| `group-beam` | `do_sample=False, num_beams=k, num_beam_groups=k, diversity_penalty=1.0` |

GCR default: `group-beam` with `k=10`. Diversity penalty prevents beam collapse.

---

## 3. Trie Construction

### Path Enumeration

`src/utils/graph_utils.py:16-46` — `dfs(graph, start_node_list, max_length)`

- Builds a NetworkX DiGraph from the list of `(head, relation, tail)` triples
- DFS from question entities (e.g. `["Grand_Bahama"]`) up to `index_path_length` (default 2 hops)
- Returns list of path tuples: `[("Grand_Bahama", "location.location.containedby", "Bahamas"), ...]`

### Path-to-String

`src/utils/utils.py:34-44` — `path_to_string(path)`

```
("Grand_Bahama", "location.location.containedby", "Bahamas")
  ↓
"Grand_Bahama -> location.location.containedby -> Bahamas"
```

### Tokenization

`src/qa_prompt_builder.py:74-82` (in `JointReasoningPromptBuilder.get_graph_index()`):

```python
paths_list_str = [f"<PATH>{path_to_string(path)}</PATH>" for path in paths_list]
tokenized_paths = self.tokenizer(paths_list_str, padding=False, add_special_tokens=False).input_ids
tokenized_path_list = [ids + [self.tokenizer.eos_token_id] for ids in tokenized_paths]
return MarisaTrie(tokenized_path_list, max_token_id=len(self.tokenizer) + 1)
```

Each path becomes: `<PATH> entity -> relation -> entity </PATH> <eos>` as a sequence of token IDs.

### MarisaTrie

`src/trie.py:122-165`

- Wraps the `marisa_trie` C library — memory-optimized, disk-serializable prefix trie
- Maps token IDs → Unicode characters: `int2char[i] = chr(i)`, `char2int[chr(i)] = i`
- Builds C trie over character-mapped strings
- Caches first-branch tokens in `self.zero_iter` for O(1) empty-prefix lookup

`get(prefix_sequence)`: converts prefix to string → `trie.keys(prefix)` → extracts next character from each match → converts back to token IDs → returns list of valid next tokens.

---

## 4. Constrained Decoding

### GraphConstrainedDecoding Class

`src/graph_constrained_decoding.py` (45 lines)

Instantiated once per question:

```python
gcr = GraphConstrainedDecoding(
    tokenizer, trie,
    start_token_ids=tokenizer.convert_tokens_to_ids("<PATH>"),
    end_token_ids=tokenizer.convert_tokens_to_ids("</PATH>"),
    enable_constrained_by_default=False
)
```

### allowed_tokens_fn Callback

Called by HuggingFace `generate()` at **every decoding step** for **every beam**:

```python
def allowed_tokens_fn(self, batch_id, sent):
    if self.start_token is not None and self.end_token is not None:
        constrained_flag, L_input = self.check_constrained_flag(sent)
    else:
        L_input = self.L_input  # fallback: fixed input length

    if constrained_flag:
        allow_tokens = self.trie.get(sent.tolist()[L_input:])
        if len(allow_tokens) == 0:
            return self.all_tokens  # dead-end fallback
        return allow_tokens
    return self.all_tokens  # free generation
```

### State Machine: check_constrained_flag

```
Token sequence: "...question... <PATH> entity -> rel -> entity </PATH> answer"
                       ^                                            ^
              constrained_flag=False                     constrained_flag=False
                       └──── constrained_flag=True ────┘
```

Logic:

1. Find positions of `<PATH>` tokens in `sent`
2. If none found → `False` (no path started)
3. Take the **last** `<PATH>` position
4. Count `</PATH>` tokens **after** that position
5. If count == 0 → inside path → `True`; set `L_input = last <PATH>` position
6. If count > 0 → path closed → `False`

### The <PATH>...</PATH> Sentinel Mechanism

The model generates freely until it emits `<PATH>`. Between `<PATH>` and `</PATH>`, only tokens present in the KG trie are allowed. After `</PATH>`, free generation resumes for the answer. This gives the model autonomy for answer text while guaranteeing **zero hallucinated KG paths**.

### Full Flow for One Question

```bash
1. DFS enumerate paths → convert to strings → wrap in <PATH> tags
2. Tokenize each, append <eos> → build MarisaTrie
3. Format prompt with chat template
4. model.generate() with prefix_allowed_tokens_fn:
   a. Steps 1-N:   free generation (prompt tokens)
   b. Step N+1:    model generates <PATH>
   c. check_constrained_flag → True, L_input = <PATH> position
   d. Steps N+2..M: trie.get() limits tokens to valid KG prefixes
   e. Step M+1:    model generates </PATH>
   f. check_constrained_flag → False, free generation resumes
   g. Steps M+2..final: answer generated freely
5. Decode output, strip input, return prediction
```

---

## 5. TypeOracle Gates

`approach3_symbolic/type_oracle.py` (586 lines)

Symbolic path pruning using the Freebase ontology schema. All operations are O(1) set lookups — no embeddings, no forward passes.

### Construction: from_graph

```python
oracle = TypeOracle.from_graph(data["graph"])
```

Scans triples for type-defining relations (`common.topic.notable_types`, `freebase.type_hints.included_types`) → builds `_entity_types: Dict[str, FrozenSet[str]]`.

### range_gate(relation, tail_entity) → bool

Checks whether tail entity's type is compatible with the relation's declared range. Returns `True` (conservative) if: relation not in schema, entity has no types, or types intersect range.

```
relation = "film.film.country"  →  range = _LOCATION_TYPES
tail_entity = "United_States"   →  types = {Country}
etypes ∩ range = {Country}  →  True (admit)
```

### type_gate(entity_name, answer_types, hop, max_hop) → bool

Checks entity type at the **terminal hop only**. Intermediate hops always pass. Returns `True` (conservative) if: `hop < max_hop`, no answer_types, entity has no types, or types intersect answer_types.

### infer_answer_types(question) → FrozenSet[str]

Pattern-matches question words against `_QUESTION_PATTERNS`:

```
"country"  → _LOCATION_TYPES
"director" → _PERSON_TYPES
"who"      → _PERSON_TYPES
"film"     → _CREATIVE_WORK_TYPES
"when"     → _DATE_TYPES
...
```

Multiple patterns match → union of expected answer types. Empty set = unconstrained.

---

## 6. Key Source Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/trie.py` | 179 | `Trie` (Python), `MarisaTrie` (C-backed), dummy stubs |
| `src/graph_constrained_decoding.py` | 45 | `allowed_tokens_fn` callback |
| `src/llms/__init__.py` | 18 | Model registry and `get_registed_model()` |
| `src/llms/base_language_model.py` | 48 | Abstract base class |
| `src/llms/base_hf_causal_model.py` | 212 | Tokenizer, model loading, generation config |
| `src/llms/graph_constrained_decoding_model.py` | 32 | GCR model: injects trie constraint |
| `src/qa_prompt_builder.py` | 537 | Prompt construction, trie building |
| `src/utils/graph_utils.py` | 195 | Graph construction, DFS, truth-path extraction |
| `src/utils/utils.py` | 60 | `path_to_string()`, I/O utilities |
| `src/utils/qa_utils.py` | 577 | Evaluation metrics |
| `approach3_symbolic/type_oracle.py` | 586 | TypeOracle: symbolic path pruning |
| `workflow/predict_paths_and_answers.py` | 184 | Stage 1 entry point |
| `workflow/predict_final_answer.py` | 341 | Stage 2 entry point |
| `workflow/run_symbolic_experiment.py` | 390 | TypeOracle SIR/FNR + proxy experiment |

---

## 7. Shell Scripts

### scripts/graph_constrained_decoding.sh

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

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `--d` | `RoG-webqsp` | Dataset (also `RoG-cwq`) |
| `--index_path_length` | `2` | Max DFS hops |
| `--k` | `10` | Beam width |
| `--generation_mode` | `group-beam` | Diverse beam search |
| `--attn_implementation` | `flash_attention_2` | Use `sdpa` on T4 GPUs |

### scripts/graph_inductive_reasoning.sh

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

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `--reasoning_path` | — | Input predictions.jsonl from Stage 1 |
| `--add_path` | `True` | Include paths in prompt |
| `-n` | `10` | Parallel API threads |
