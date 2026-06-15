# GCR Pipeline: Model Loading, Trie Construction, and Constrained Decoding

This document traces the exact code paths for how a model is loaded, how the trie is built from knowledge graph paths, and how constrained decoding forces the LLM to generate only valid KG reasoning paths.

---

## 1. Model Loading

### 1.1 Model Registration

`src/llms/__init__.py`

When you call `get_registed_model("rmanluo/GCR-Meta-Llama-3.1-8B-Instruct")`, it does a substring match on the name:

```python
registed_language_models = {
    'gpt': ChatGPT,
    'others': HfCausalModel,
    'gcr': GraphConstrainedDecodingModel,
    'proxy': LLMProxy,
}
```

The string `"rmanluo/GCR-Meta-Llama-3.1-8B-Instruct"` contains `"gcr"`, so it returns `GraphConstrainedDecodingModel`. This class inherits from `HfCausalModel`, which inherits from `BaseLanguageModel`.

### 1.2 Argument Setup

`src/llms/base_hf_causal_model.py:42-81`

`HfCausalModel.add_args(parser)` registers all CLI arguments:

| Argument | Default | Purpose |
|----------|---------|---------|
| `model_path` | — | HuggingFace model ID or local path |
| `maximun_token` | 4096 | Max context window |
| `max_new_tokens` | 1024 | Max tokens to generate |
| `dtype` | `"bf16"` | Precision (`fp32`, `fp16`, `bf16`) |
| `quant` | `"none"` | Quantization (`4bit`, `8bit`) |
| `attn_implementation` | `"flash_attention_2"` | Attention backend |
| `generation_mode` | `"greedy"` | Decoding strategy |
| `k` | 1 | Beam width / number of sequences |
| `chat_model` | `"true"` | Whether to apply chat template |

In the notebook, the args are set programmatically:

```python
args.attn_implementation = ATTN_IMPL  # "sdpa" or "flash_attention_2"
args.generation_mode = GEN_MODE       # "group-beam"
args.k = K                            # 5
```

### 1.3 Loading the Model

`src/llms/base_hf_causal_model.py:84-145`

`prepare_for_inference()` does the heavy lifting:

**1.3a. Tokenizer** (line 86-88):

```python
self.tokenizer = AutoTokenizer.from_pretrained(
    self.args.model_path, token=HF_TOKEN, trust_remote_code=True
)
```

Loads the tokenizer (sentencepiece-based for Llama, tiktoken-based for Qwen). The `HF_TOKEN` authenticates with HuggingFace for gated models like Llama-3.1.

**1.3b. Model config override** (line 90-97):

```python
model_config = AutoConfig.from_pretrained(...)
attn = self.args.attn_implementation
if attn == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
    attn = "sdpa"
model_config._attn_implementation = attn
```

This is critical: the pretrained config may say `flash_attention_2`, but if flash-attn isn't installed, it silently falls back to `sdpa`. Without this, `from_pretrained` would crash.

**1.3c. Model weights** (line 98-107):

```python
self.model = AutoModelForCausalLM.from_pretrained(
    self.args.model_path,
    config=model_config,
    device_map="auto",        # automatically shards across GPUs
    torch_dtype=self.DTYPE.get(self.args.dtype, None),  # bf16/fp16/fp32
    ...
)
```

`device_map="auto"` uses `accelerate` to place the model on GPU (or split across multiple GPUs). For a 1.5B model on a T4, the entire model fits in VRAM. For 8B on A100, also fits (~16GB for bf16).

**1.3d. Generation config** (line 115-145):

```python
self.generation_cfg = GenerationConfig.from_pretrained(self.args.model_path)
```

Loads the model's default generation config, then overrides based on `generation_mode`:

- `"greedy"`: `do_sample=False, num_return_sequences=1`
- `"beam"`: `do_sample=False, num_beams=k, num_return_sequences=k`
- `"group-beam"`: `do_sample=False, num_beams=k, num_beam_groups=k, diversity_penalty=1.0`

Group-beam search is what GCR uses. With `k=5`, it creates 5 beam groups, each with 1 beam, and applies a diversity penalty of 1.0 to encourage different groups to explore different reasoning paths.

### 1.4 Prompt Formatting

`src/llms/base_hf_causal_model.py:147-153`

```python
def prepare_model_prompt(self, query):
    if self.args.chat_model:
        chat_query = [{"role": "user", "content": query}]
        return self.tokenizer.apply_chat_template(
            chat_query, tokenize=False, add_generation_prompt=True
        )
```

For chat models (Llama-3.1-Instruct, Qwen2.5-Instruct), this wraps the query in the model's chat template:

```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```

This tells the model "respond as an assistant" and the `add_generation_prompt=True` ensures it ends right before where the assistant's response would start.

### 1.5 Generation

`src/llms/graph_constrained_decoding_model.py`

`GraphConstrainedDecodingModel.generate_sentence()` overrides the base class:

```python
def generate_sentence(self, llm_input, trie, start_token_ids, end_token_ids, ...):
    inputs = self.tokenizer(llm_input, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs.input_ids.to(self.model.device)
    attention_mask = inputs.attention_mask.to(self.model.device)
    gcr = GraphConstrainedDecoding(self.tokenizer, trie, start_token_ids, end_token_ids, ...)
    res = self.model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=self.generation_cfg,
        prefix_allowed_tokens_fn=gcr.allowed_tokens_fn,  # <-- THE KEY LINE
        ...
    )
```

The `prefix_allowed_tokens_fn` parameter is what HuggingFace's `generate()` calls at **every decoding step** to get the set of allowed token IDs. This is how the trie constrains generation.

---

## 2. Trie Construction

### 2.1 Path Enumeration

`src/utils/graph_utils.py:16-46`

Given a question like "What country is the Grand Bahama Island in?", the graph is:

```
Grand Bahama -> location.location.containedby -> Bahamas
Grand Bahama -> location.location.contains -> ...
```

`dfs(graph, ["Grand Bahama"], max_length=2)` does a depth-first search from the question entities, collecting all paths up to `max_length` hops:

```python
def dfs(graph, start_node_list, max_length):
    path_lists = set()
    for start_node in start_node_list:
        dfs_visit(start_node, [])  # recursive DFS
    return list(path_lists)
```

Each path is a list of `(head, relation, tail)` tuples:

```
[("Grand Bahama", "location.location.containedby", "Bahamas")]
```

### 2.2 Path-to-String Conversion

`src/utils/utils.py:34-44`

```python
def path_to_string(path):
    result = ""
    for i, p in enumerate(path):
        if i == 0:
            h, r, t = p
            result += f"{h} -> {r} -> {t}"
        else:
            _, r, t = p
            result += f" -> {r} -> {t}"
    return result.strip()
```

Converts the path tuple to a human-readable string:

```
"Grand Bahama -> location.location.containedby -> Bahamas"
```

### 2.3 Tokenization

`src/qa_prompt_builder.py:74-82`

The `PathGenerationWithAnswerPromptBuilder.get_graph_index()` method (inherited from `JointReasoningPromptBuilder`) does:

```python
paths_list_str = [
    f"<PATH>{path_to_string(path)}</PATH>" for path in paths_list
]
tokenized_paths = self.tokenizer(
    paths_list_str, padding=False, add_special_tokens=False
).input_ids
tokenized_path_list = [
    ids + [self.tokenizer.eos_token_id] for ids in tokenized_paths
]
return MarisaTrie(tokenized_path_list, max_token_id=len(self.tokenizer) + 1)
```

So for a path like `"Grand Bahama -> location.location.containedby -> Bahamas"`, it becomes:

```
<PATH> Grand Bahama -> location . location . containedby -> Bahamas </PATH> <eos>
```

Each of these is a sequence of token IDs (integers). The `<eos>` is appended so the trie knows where each valid sequence ends.

### 2.4 MarisaTrie

`src/trie.py:122-165`

`MarisaTrie` wraps the `marisa_trie` C library (a highly optimized, memory-efficient prefix trie):

```python
class MarisaTrie:
    def __init__(self, sequences, cache_fist_branch=True, max_token_id=256001):
        # Map token IDs to characters (marisa_trie works with strings)
        self.int2char = [chr(i) for i in range(min(max_token_id, 55000))] + (...)
        self.char2int = {self.int2char[i]: i for i in range(max_token_id)}

        # Cache first-branch tokens for O(1) empty-prefix lookup
        if cache_fist_branch:
            self.zero_iter = list({sequence[0] for sequence in sequences})

        # Build the C trie
        self.trie = marisa_trie.Trie(
            "".join([self.int2char[i] for i in sequence]) for sequence in sequences
        )
```

The trick: token IDs are integers, but `marisa_trie` works with strings. So each token ID is mapped to a Unicode character via `int2char`. For example, token ID 12345 becomes `chr(12345)`. The entire path's token sequence becomes a string of characters, and `marisa_trie` builds a compact prefix tree over these strings.

### 2.5 Trie Lookup

`src/trie.py:150-162`

```python
def get(self, prefix_sequence):
    if self.cache_fist_branch and len(prefix_sequence) == 0:
        return self.zero_iter  # O(1) cache hit
    else:
        key = "".join([self.int2char[i] for i in prefix_sequence])
        return list({
            self.char2int[e[len(key)]]
            for e in self.trie.keys(key)
            if len(e) > len(key)
        })
```

Given a prefix (the tokens generated so far), it converts to a string and asks `marisa_trie` for all keys starting with that prefix. Then it extracts the **next character** from each matching key, converts back to token IDs, and returns them as the set of valid next tokens.

---

## 3. Constrained Decoding

### 3.1 The Callback Object

`src/graph_constrained_decoding.py`

`GraphConstrainedDecoding` is instantiated once per question:

```python
gcr = GraphConstrainedDecoding(
    tokenizer, trie,
    start_token_ids=tokenizer.convert_tokens_to_ids("<PATH>"),
    end_token_ids=tokenizer.convert_tokens_to_ids("</PATH>"),
    enable_constrained_by_default=False
)
```

### 3.2 The `allowed_tokens_fn`

This is called by HuggingFace's `generate()` at **every decoding step** for **every beam**:

```python
def allowed_tokens_fn(self, batch_id, sent):
    constrained_flag = self.constrained_flag

    if self.start_token is not None and self.end_token is not None:
        constrained_flag, L_input = self.check_constrained_flag(sent)
    else:
        if self.L_input is None:
            self.L_input = len(sent)
        L_input = self.L_input

    allow_tokens = self.all_tokens
    if constrained_flag:
        allow_tokens = self.trie.get(sent.tolist()[L_input:])
        if len(allow_tokens) == 0:
            return self.all_tokens  # fallback if dead end
    return allow_tokens
```

### 3.3 The State Machine

`check_constrained_flag` (line 14-28) tracks whether we're inside a `<PATH>...</PATH>` block:

```python
def check_constrained_flag(self, sent):
    matched_start_token = torch.where(sent == self.start_token)[0]
    if len(matched_start_token) == 0:
        return False, len(sent)           # no <PATH> seen yet

    last_start_tokens = torch.where(sent == self.start_token)[0][-1]
    end_token_number = len(torch.where(sent[last_start_tokens:] == self.end_token)[0])

    if end_token_number == 0:
        self.last_start_token = last_start_tokens
        return True, last_start_tokens    # inside <PATH>, constrain from here
    else:
        self.last_start_token = None
        return False, len(sent)           # </PATH> already closed
```

The model generates text in this order:

```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Question: ... 
Topic entities: ...
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

<PATH> Grand Bahama -> location.location.containedby -> Bahamas </PATH>
```

The state machine has three states:

1. **Before `<PATH>`**: `constrained_flag=False` -> allow all tokens (free generation)
2. **Inside `<PATH>` (no `</PATH>` yet)**: `constrained_flag=True` -> only allow tokens from trie
3. **After `</PATH>`**: `constrained_flag=False` -> allow all tokens again

### 3.4 The Full Flow for One Question

```
1. DFS enumerate all paths from question entities
   [("Grand Bahama", "location.location.containedby", "Bahamas"), ...]

2. Convert paths to strings
   ["Grand Bahama -> location.location.containedby -> Bahamas", ...]

3. Wrap in <PATH> tags
   ["<PATH>Grand Bahama -> location.location.containedby -> Bahamas</PATH>", ...]

4. Tokenize and append <eos>
   [[12345, 678, 910, ..., 2], [9999, 333, 444, ..., 2], ...]

5. Build MarisaTrie from all tokenized paths
   (compact C trie over character-mapped token sequences)

6. Format prompt with chat template
   "<|begin_of_text|>...user...\nQuestion: What country...\nTopic entities: Grand Bahama<|eot_id|>assistant\n"

7. HuggingFace generate() with prefix_allowed_tokens_fn:
   - Step 1-N: model generates prompt tokens freely
   - Step N+1: model generates <PATH> token
   - Step N+2: check_constrained_flag detects <PATH> without </PATH>
     -> constrained_flag = True, L_input = position of <PATH>
   - Step N+3..M: allowed_tokens_fn calls trie.get(sent[L_input:])
     -> only tokens that are valid prefixes of known paths
   - Step M+1: model generates </PATH> token
   - Step M+2: check_constrained_flag sees </PATH> after last <PATH>
     -> constrained_flag = False, free generation resumes
   - Step M+3..final: model generates answer freely

8. Decode output, strip input tokens, return prediction
```

---

## 4. Why This Works

The trie encodes the **entire space of valid KG paths** as a prefix tree. At each decoding step inside `<PATH>`, the model can only choose from tokens that appear in some valid path at that position. This means:

- The model **cannot hallucinate relations** that don't exist in the graph
- The model **cannot hallucinate entities** that aren't connected to the question entities
- The model **must follow actual KG edges** in sequence
- The model **can still choose** which path to follow (beam search explores multiple)
- The model **generates the answer freely** after `</PATH>`, using the paths it just produced as context

---

## 5. File Reference

| File | Purpose |
|------|---------|
| `src/llms/__init__.py` | Model registry and factory function |
| `src/llms/base_language_model.py` | Abstract base class defining the LLM interface |
| `src/llms/base_hf_causal_model.py` | Core HuggingFace causal LM: model loading, tokenization, generation config |
| `src/llms/graph_constrained_decoding_model.py` | GCR model wrapper: overrides `generate_sentence` to inject trie constraint |
| `src/graph_constrained_decoding.py` | Core constrained decoding callback: `allowed_tokens_fn` |
| `src/trie.py` | Trie data structures: `Trie` (pure Python), `MarisaTrie` (C-backed) |
| `src/qa_prompt_builder.py` | Prompt construction and trie building from KG paths |
| `src/utils/graph_utils.py` | Graph construction, DFS path enumeration, truth path extraction |
| `src/utils/utils.py` | `path_to_string` and other utilities |
