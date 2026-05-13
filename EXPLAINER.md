# Graph-Constrained Reasoning (GCR) — Full Code Explainer

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [The Big Picture](#2-the-big-picture)
3. [Pipeline Walkthrough](#3-pipeline-walkthrough)
4. [Python Concepts Used](#4-python-concepts-used)
5. [File-by-File Breakdown](#5-file-by-file-breakdown)
6. [The Challenge: Build It Yourself (Mini Version)](#6-the-challenge-build-it-yourself-mini-version)
7. [Running on Colab](#7-running-on-colab)

---

## 1. What Is This Project?

This is the code for **"Graph-constrained Reasoning: Faithful Reasoning on Knowledge Graphs with Large Language Models"** (ICML 2025).

**The problem:** LLMs hallucinate. When asked questions about facts in a knowledge graph (KG), they might make up relationships or entities.

**The solution:** Force the LLM to only generate text that corresponds to real paths in the KG. Do this by building a **trie** (prefix tree) of all valid KG paths, then using HuggingFace's `prefix_allowed_tokens_fn` hook to mask out any invalid tokens at every generation step.

**The guarantee:** Every reasoning path the LLM outputs actually exists in the KG. Zero hallucination on factual paths.

---

## 2. The Big Picture

```bash
┌───────────────────────────────────────────────────┐
│           Knowledge Graph                         │
│  (Christopher_Nolan) --directed--> (Inception)    │
│  (Inception) --won_award--> (Best_Picture)        │
│  (Christopher_Nolan) --directed--> (Interstellar) │
│  ...                                              │
└────────────┬──────────────────────────────────────┘
             │
             ▼ DFS from question entity (max 2 hops)
             │
┌──────────────────────────────────────────────────────────────────────────────┐
│  All valid paths:                                                            │
│  ["Christopher_Nolan -> directed -> Inception",                              │
│   "Christopher_Nolan -> directed -> Interstellar",                           │
│   "Christopher_Nolan -> directed -> Inception -> won_award -> Best_Picture", │
│   "Christopher_Nolan -> directed -> Interstellar -> won_award -> Oscar"]     │
└────────────┬─────────────────────────────────────────────────────────────────┘
             │
             ▼ Tokenize each path, insert into trie
             │
┌────────────────────────────────────────────┐
│  KG-Trie:                                  │
│  token_1 -> token_2 -> token_3 -> ...      │
│    (Christopher)  (->)    (directed)       │
│                  ╱    ╲                    │
│         (Inception)  (Interstellar)        │
│            │              │                │
│            ▼              ▼                │
│         (...)           (...)              │
└────────────┬───────────────────────────────┘
             │
             ▼ Passed as prefix_allowed_tokens_fn to model.generate()
             │
┌────────────────────────────────────────────┐
│  Decoding step 1:  "Christopher_Nolan ->"  │
│  → trie.get(["...tokens..."])              │
│  → returns ["directed"]                    │
│  → ALL OTHER TOKENS masked out             │
│                                            │
│  Decoding step 2:  "...directed ->"        │
│  → trie.get(["...tokens..."])              │
│  → returns ["Inception", "Interstellar"]   │
│  → everything else masked                  │
└────────────────────────────────────────────┘
```

---

## 3. Pipeline Walkthrough

### Step 1: Graph-Constrained Decoding

**Input:** A question + its topic entity + the KG subgraph

**What happens, line by line:**

1. **`qa_prompt_builder.py:GraphConstrainedPromptBuilder.get_graph_index()`**
   - Takes the question dict (from HuggingFace dataset)
   - Builds a NetworkX graph from the list of triples: `[(h,r,t), (h,r,t), ...]`
   - Runs DFS from the topic entity to find all paths up to `index_path_length` (usually 2)
   - Converts each path to a string like `"entity -> relation -> entity"`
   - Tokenizes each string with the LLM's tokenizer
   - Appends EOS token ID to each sequence
   - Builds a `MarisaTrie` from all token-ID sequences

2. **`graph_constrained_decoding.py:GraphConstrainedDecoding.allowed_tokens_fn()`**
   - This is the callback function passed to HuggingFace's `model.generate()`
   - It's called at EVERY decoding step with `(batch_id, generated_tokens_so_far)`
   - **Check if we're in constrained mode:** looks for `<PATH>` and `</PATH>` sentinel tokens in the generated sequence. Between `<PATH>` and `</PATH>` = constrained; outside = free generation
   - If constrained: extracts the tokens generated *since* the input prompt ended, calls `trie.get(...)` to ask "what tokens are valid next?"
   - Returns the list of allowed token IDs. HuggingFace masks out everything else

3. **`graph_constrained_decoding_model.py:GraphConstrainedDecodingModel.generate_sentence()`**
   - Tokenizes the prompt
   - Instantiates `GraphConstrainedDecoding` with the trie and sentinel tokens
   - Calls `self.model.generate(prefix_allowed_tokens_fn=gcr.allowed_tokens_fn, ...)`
   - Decodes the output sequences

4. **Result:** The model outputs paths like `<PATH>Christopher_Nolan -> directed -> Inception -> won_award -> Best_Picture</PATH> Best_Picture`

### Step 2: Graph Inductive Reasoning

**Input:** The generated paths + answer hypotheses from Step 1

**What happens:**

- A second LLM (typically GPT-4, GPT-3.5, or a local model) is given a prompt like:

  ```bash
  Based on the reasoning paths, please answer the given question.

  Reasoning Paths:
  Christopher_Nolan -> directed -> Inception -> won_award -> Best_Picture
  Christopher_Nolan -> directed -> Inception

  Question: Which film directed by Christopher Nolan won Best Picture?
  Answer:
  ```

- The LLM reads the paths and produces a final answer. No additional training needed.

### Training (Optional)

**Input:** Questions + shortest paths between question/answer entities

**What happens:**

- `build_shortest_path_index.py` extracts shortest paths connecting entities
- The model is fine-tuned with a supervised loss: given the question + `[PATH]` token, generate the correct path + answer
- At inference, the trie constraint kicks in to keep the model faithful to the KG

---

## 4. Python Concepts Used

### 4.1 Static Methods (`@staticmethod`)

In `trie.py`, methods like `_add_to_trie` and `_get_from_trie` are `@staticmethod`. This means they don't receive `self` — they're just regular functions that happen to live inside a class for organization.

```python
class Trie(object):
    @staticmethod
    def _add_to_trie(sequence, trie_dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])
```

You call them like `Trie._add_to_trie(...)` not `self._add_to_trie(...)`. They're pure functions — no side effects, just operate on passed-in data.

### 4.2 Recursion on Nested Dictionaries

The `Trie` class is a tree of nested Python dicts:

```python
# After inserting paths: [1,2,3] and [1,2,4]
trie_dict = {
    1: {
        2: {
            3: {},  # empty dict = end of a valid sequence
            4: {},  # empty dict = end of another valid sequence
        }
    }
}

# Query with prefix [1]:
# → Look up trie_dict[1] → get {2: {3: {}, 4: {}}}
# → Return keys: [2]

# Query with prefix [1, 2]:
# → Look up trie_dict[1][2] → get {3: {}, 4: {}}
# → Return keys: [3, 4]

# Query with prefix [1, 2, 3]:
# → Look up trie_dict[1][2][3] → get {}
# → Return empty list (no more tokens after this)
```

The recursion follows the chain: `dict[key]` → `dict[key][key]` → until you hit a non-existent key (invalid path) or an empty dict (valid path end).

### 4.3 Generators and `yield from`

```python
def __iter__(self):
    def _traverse(prefix_sequence, trie_dict):
        if trie_dict:  # not empty = has children
            for next_token in trie_dict:
                yield from _traverse(
                    prefix_sequence + [next_token],
                    trie_dict[next_token]
                )
        else:  # empty dict = leaf node = complete sequence
            yield prefix_sequence
    return _traverse([], self.trie_dict)
```

- `yield` produces one value from a generator
- `yield from` delegates to another generator (here: the recursive call)
- `__iter__` makes the class work with `for ... in trie` syntax
- This is a **recursive generator** — it lazily emits all paths stored in the trie without building a list

### 4.4 Dunder Methods (`__getitem__`, `__len__`, `__iter__`)

```python
def __getitem__(self, value):  # trie[prefix] works
    return self.get(value)

def __len__(self):              # len(trie) works
    return self.len

def __iter__(self):             # for item in trie works
    ...
```

These make your custom class behave like a built-in Python container (duck typing).

### 4.5 Closures (Inner Functions Capturing Outer Scope)

```python
def dfs(graph, start_node_list, max_length):
    path_lists = set()  # ← captured by dfs_visit

    def dfs_visit(node, path):  # ← closure
        if len(path) > max_length:
            return
        for neighbor in graph.neighbors(node):
            rel = graph[node][neighbor]["relation"]
            new_path = path + [(node, rel, neighbor)]
            if len(new_path) <= max_length:
                path_lists.add(tuple(new_path))  # ← mutates outer variable
            dfs_visit(neighbor, new_path)

    for start_node in start_node_list:
        dfs_visit(start_node, [])

    return list(path_lists)
```

`dfs_visit` references `path_lists` and `max_length` from the enclosing scope. Closures are Python's way of creating "private" state for a function.

### 4.6 Callbacks as Control Flow

```python
# In graph_constrained_decoding_model.py:
res = self.model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    generation_config=self.generation_cfg,
    prefix_allowed_tokens_fn=gcr.allowed_tokens_fn,  # ← CALLBACK
    return_dict_in_generate=True,
    pad_token_id=self.tokenizer.eos_token_id
)
```

HuggingFace calls `gcr.allowed_tokens_fn(batch_id, generated_so_far)` at every decoding step. This is a **callback** — you pass a function, and the library calls it back at specific points. It's the core mechanism that makes the whole GCR idea work.

The signature HuggingFace expects:

```python
def allowed_tokens_fn(batch_id: int, sent: torch.Tensor) -> List[int]:
    """Return list of valid token IDs for the next step."""
```

### 4.7 `@torch.inference_mode()`

```python
@torch.inference_mode()
def generate_sentence(self, llm_input, *args, **kwargs):
```

- Replaces the older `torch.no_grad()`
- Disables gradient computation AND autograd tracking
- ~20% faster inference than `no_grad()`
- Not just "don't track gradients" but also "don't even build the autograd graph"

### 4.8 String Formatting with `.format()`

```python
ZERO_SHOT_PROMPT = """Reasoning path is a sequence...
# Question:
{question}
# Topic entities:
{entities}
"""
# Later:
prompt = ZERO_SHOT_PROMPT.format(question="...", entities="...")
```

`str.format()` replaces `{placeholders}` with values. The **same** template is reused for every question.

### 4.9 `__init__.py` as a Registry Pattern

```python
# src/llms/__init__.py
registed_language_models = {
    'gpt': ChatGPT,
    'others': HfCausalModel,
    'gcr': GraphConstrainedDecodingModel,
}

def get_registed_model(model_name) -> BaseLanguageModel:
    for key, value in registed_language_models.items():
        if key in model_name.lower():
            return value  # returns the CLASS, not an instance
    return HfCausalModel
```

This is a **registry pattern**. You look up a class by a key present in the model name string. The caller then instantiates it:

```python
LLM = get_registed_model(args.model_name)
model = LLM(args)
```

### 4.10 `argparse` with Dynamic Args

```python
LLM = get_registed_model(args.model_name)
LLM.add_args(argparser)  # each model class adds its own args
args = argparser.parse_args()
```

The base model class defines common args via a `@staticmethod add_args(parser)`. Different model subclasses can add more. This is parse-twice pattern: first parse to get model name, then load the model class, then let it add its args, then re-parse.

### 4.11 Mapping Token IDs ↔ Characters for MarisaTrie

```python
class MarisaTrie(object):
    def __init__(self, sequences, max_token_id=256001):
        # Map every possible token ID to a unique character
        self.int2char = [chr(i) for i in range(max_token_id)]
        self.char2int = {chr(i): i for i in range(max_token_id)}
    
    def get(self, prefix_sequence):
        # Convert token IDs to string for marisa-trie lookup
        key = "".join([self.int2char[i] for i in prefix_sequence])
        # Find all keys starting with prefix
        # Extract the NEXT character after the prefix
        # Convert back to token IDs
        return [self.char2int[e[len(key)]] for e in self.trie.keys(key) if len(e) > len(key)]
```

The `marisa-trie` library works on **strings**, not lists of ints. So GCR maps every possible token ID to a unique Unicode character (via `chr()`). This works because the tokenizer vocabulary is ~32K-256K tokens, and Unicode has plenty of space.

### 4.12 Multiprocessing with `Pool.imap_unordered`

```python
with Pool(args.n) as p:
    for res in tqdm(p.imap_unordered(partial(process, K=args.K, ...), dataset)):
        results.append(res)
```

- `Pool(args.n)`: create a pool of `n` worker processes
- `partial(process, K=args.K)`: pre-fill arguments to `process`, returning a callable that takes one argument (the dataset sample)
- `p.imap_unordered(...)`: like `map` but yields results as they complete (not in order). Faster than `imap` for CPU-bound work.
- `tqdm(...)`: progress bar wrapper

---

## 5. File-by-File Breakdown

### `src/trie.py` (179 lines)

**Purpose:** Data structure for efficient prefix-based lookups of token sequences.

**Classes:**

| Class | Purpose | When to Use |
|-------|---------|-------------|
| `Trie` | Pure-Python nested-dict trie | Teaching/debugging, small vocabularies |
| `MarisaTrie` | Memory-optimized trie via `marisa-trie` C library | Production, large KGs with many paths |
| `DummyTrieMention` | Stub that always returns same tokens | Testing, ablation studies |
| `DummyTrieEntity` | Stub with hardcoded state machine | Testing, alignment experiments |

**`Trie._add_to_trie` algorithm:**

```
Input: [token_A, token_B, token_C]
trie_dict = {}
↓
Step 1: trie_dict[token_A] = {}
↓
Step 2: trie_dict[token_A][token_B] = {}
↓
Step 3: trie_dict[token_A][token_B][token_C] = {}
(Done — the final {} marks a complete path)
```

**`Trie._get_from_trie` algorithm:**

```
Input prefix: [token_A, token_B]
trie_dict = {A: {B: {C: {}, D: {}}, E: {F: {}}}}
↓
Step 1: Look up trie_dict[A] → {B: {C: {}, D: {}}, E: {F: {}}}
Step 2 prefix[0]==A, so recurse with prefix[1:] and trie_dict[A]
Step 3: prefix = [B], look up {B: {C: {}, D: {}}} → exists
Step 4: prefix empty! Return keys of current node: [C, D]
```

**`MarisaTrie` char mapping:**

```
Token IDs: [30582, 412, 9500, ...]
         ↓ chr() each
Chars:    '蜗' 'Ę' 'ɜ' ...
         ↓ join
String:   '蜗Ęɜ...'
         ↓ put in marisa_trie.Trie()
```

### `src/graph_constrained_decoding.py` (46 lines)

**Purpose:** The bridge between KG-Trie and HuggingFace's generation API.

**State machine:**

```
Token sequence: "...question... <PATH> entity -> rel -> entity </PATH> answer"
                         ^                                       ^
                constrained_flag=False                  constrained_flag=False
                         └──── constrained_flag=True ────┘
```

**`check_constrained_flag` logic:**

```
1. Find all positions of <PATH> token in generated sequence
2. If none → not constrained
3. Take the LAST <PATH> position
4. Count </PATH> tokens AFTER that position
5. If count == 0 → still inside a path → constrained
6. If count > 0 → path already closed → not constrained
```

### `src/llms/` directory — Model Zoo

| File | Class | What It Does |
|------|-------|-------------|
| `base_language_model.py` | `BaseLanguageModel` | Abstract base class. Defines interface: `generate_sentence()`, `token_len()`, `prepare_for_inference()` |
| `base_hf_causal_model.py` | `HfCausalModel` | Concrete implementation for any HuggingFace causal LM. Loads model, tokenizer, configures generation. |
| `graph_constrained_decoding_model.py` | `GraphConstrainedDecodingModel` | Extends `HfCausalModel`. Overrides `generate_sentence()` to inject `prefix_allowed_tokens_fn` with the KG-Trie. |
| `chatgpt.py` | `ChatGPT` | OpenAI API wrapper (GPT-3.5, GPT-4) |
| `llm_proxy.py` | `LLMProxy` | FastChat API wrapper for self-hosted models |
| `conv_prompt.py` | Conversation prompt templates | Handles model-specific chat formats |

### `src/utils/` directory — Utilities

| File | Key Functions | Purpose |
|------|---------------|---------|
| `graph_utils.py` | `build_graph()`, `dfs()`, `get_truth_paths()`, `bfs_with_rule()` | KG operations. Build NetworkX graphs, find paths, evaluate |
| `qa_utils.py` | `eval_path_result_w_ans()`, `eval_result()` | Evaluate path accuracy and answer accuracy |
| `utils.py` | `path_to_string()`, `load_jsonl()`, `list_to_string()` | I/O, string formatting |
| `align_utils.py` | Alignment-specific functions | Used for the spectoken variant (not core GCR) |
| `training_utils.py` | Training helpers | Used in the fine-tuning script |

### `src/qa_prompt_builder.py` (537 lines)

**Purpose:** Build prompts and construct KG-Tries from dataset samples.

**Class hierarchy:**

```
GraphConstrainedPromptBuilder
├── .get_graph_index()     → builds MarisaTrie from KG paths
├── .process_input()       → returns (prompt_string, ground_truth_paths, trie)
├── .format_input_with_template()  → fills in {question}, {entities}
│
├── PathGenerationPromptBuilder
│   └── Paths-only generation (not used in main pipeline)
│
├── JointReasoningPromptBuilder
│   ├── PATH_START_TOKEN = "<PATH>"
│   ├── PATH_END_TOKEN = "</PATH>"
│   └── Overrides get_graph_index() to wrap paths in <PATH>...</PATH>
│   │
│   └── PathGenerationWithAnswerPromptBuilder
│       └── Used in Step 1. Prompt: "generate paths to answer..."
│
├── RetrievalPromptBuilder
│   └── Builds separate tries for entities, relations, triples
│
PromptBuilder  (not a subclass — standalone)
    └── Used in Step 2. Formats predicted paths as context for the answering LLM.
```

**Prompt template loading:**

```python
def get_prompt_template(self, template_name):
    template_name = template_name.upper().replace("-", "_") + "_PROMPT"
    return self.__getattribute__(template_name)  # gets self.ZERO_SHOT_PROMPT etc.
```

This is a form of **reflection** — converting a string name to an attribute access. `"zero-shot"` → `"ZERO_SHOT_PROMPT"` → `self.ZERO_SHOT_PROMPT`.

### `workflow/` directory — Entry Points

| File | Command | What It Does |
|------|---------|-------------|
| `build_shortest_path_index.py` | `python workflow/build_shortest_path_index.py --d RoG-webqsp --split train` | Pre-compute shortest paths between Q/A entities for training |
| `build_graph_index.py` | `python workflow/build_graph_index.py --d RoG-webqsp --split test --K 2` | Pre-compute ALL paths up to K hops for evaluation |
| `finetune_kg_specialized_llm.py` | `accelerate launch workflow/finetune_kg_specialized_llm.py ...` | Fine-tune the KG-specialized LLM |
| `predict_paths_and_answers.py` | `python workflow/predict_paths_and_answers.py ...` | Step 1: Graph-constrained decoding |
| `predict_final_answer.py` | `python workflow/predict_final_answer.py ...` | Step 2: Inductive reasoning |

### `scripts/` directory — Shell Wrappers

Shell scripts that set environment variables and call workflow Python scripts. Example:

```bash
# scripts/graph_constrained_decoding.sh
MODEL_PATH=rmanluo/GCR-Meta-Llama-3.1-8B-Instruct
python workflow/predict_paths_and_answers.py \
  --data_path rmanluo \
  --d RoG-webqsp \
  --split test \
  --index_path_length 2 \
  --model_path ${MODEL_PATH} \
  --k 10 \
  --prompt_mode zero-shot \
  --generation_mode group-beam
```

---

## 6. The Challenge: Build It Yourself (Mini Version)

Write a Python program that implements the **core idea** of GCR — constrained decoding with a KG-Trie — **without using any LLM**. Just build the trie, simulate token-by-token generation, and show the constraint in action.

### Setup

```python
# A tiny "knowledge graph" as (head, relation, tail) triples
KG = [
    ("Christopher_Nolan", "directed", "Inception"),
    ("Christopher_Nolan", "directed", "Interstellar"),
    ("Christopher_Nolan", "directed", "The_Prestige"),
    ("Inception", "won_award", "Best_Picture"),
    ("Interstellar", "won_award", "Oscar"),
    ("The_Prestige", "nominated_for", "Oscar"),
    ("Best_Picture", "awarded_year", "2010"),
    ("Oscar", "awarded_year", "2014"),
    ("Emma_Thomas", "produced", "Inception"),
    ("Emma_Thomas", "produced", "Interstellar"),
]

# Simple "vocabulary" — entities and relations
VOCAB = sorted(set(
    word for triple in KG
    for word in [triple[0], triple[1], triple[2]]
))
TOKEN_IDS = {word: i for i, word in enumerate(VOCAB)}
NUM_TOKENS = len(VOCAB)

# Question
QUESTION = "Which film directed by Christopher Nolan won Best Picture?"
TOPIC_ENTITY = "Christopher_Nolan"
MAX_HOPS = 2
```

### Your Task

Implement these functions:

```python
def build_kg_trie(kg, start_entity, max_hops):
    """
    Build a nested-dict trie of all valid paths in the KG
    starting from start_entity, up to max_hops in length.

    A "path" is a sequence of tokens representing:
      entity -> relation -> entity -> relation -> entity ...

    Return: a trie_dict (nested dict) where keys are token IDs
            and {} marks the end of a valid path sequence.
    """
    # Step 1: Build adjacency list from KG
    # Step 2: DFS from start_entity, building token-ID sequences
    # Step 3: Insert each sequence into the trie
    pass


def query_trie(trie_dict, prefix_ids):
    """
    Given a prefix (list of token IDs), return all valid next token IDs.

    If prefix is empty, return all first tokens.
    If prefix is invalid (not in trie), return empty list.
    If prefix ends at a leaf ({}), return empty list (path complete).
    """
    pass


def simulate_constrained_decoding(trie_dict, seed_tokens=["Christopher_Nolan", "->", "directed", "->"]):
    """
    Simulate step-by-step generation:

    1. Start with seed_tokens (the path so far)
    2. At each step, query the trie for valid next tokens
    3. Print the allowed next tokens
    4. Choose one (or let user choose interactively)
    5. Continue until no more tokens are allowed or max_hops reached

    Show that ONLY KG-valid tokens are ever produced.
    """
    pass
```

### Expected Output

When you run `simulate_constrained_decoding()`, you should see:

```
Step 0:
  Seed tokens: ['Christopher_Nolan', '->', 'directed', '->']
  Token IDs: [0, 1, 2, 1]
  ────────────────────────────────────
  Allowed next tokens:
    - Inception (ID 3)
    - Interstellar (ID 4)
    - The_Prestige (ID 5)
  ────────────────────────────────────
  Invalid tokens that are BLOCKED: Emma_Thomas, produced,
  nominated_for, won_award, Best_Picture, Oscar, ...
  All 40000+ other token IDs in the LLM's vocabulary.

Step 1 (user picks "Inception"):
  Path so far: ['Christopher_Nolan', '->', 'directed', '->', 'Inception']
  ────────────────────────────────────
  Allowed next tokens:
    - won_award (ID 8)
  ────────────────────────────────────

Step 2:
  Path so far: ['Christopher_Nolan', '->', 'directed', '->', 'Inception', '->', 'won_award']
  ────────────────────────────────────
  Allowed next tokens:
    - Best_Picture (ID 6)
  ────────────────────────────────────

Step 3:
  Path so far: ['Christopher_Nolan', '->', 'directed', '->', 'Inception',
                '->', 'won_award', '->', 'Best_Picture']
  ────────────────────────────────────
  Allowed next tokens: [] (path complete — 2 hops reached)
```

### Extension: Add the `<PATH>` / `</PATH>` Sentinel Logic

Once you have the basic trie working, extend it to handle sentinel tokens:

```python
def check_if_constrained(tokens_so_far, path_start_id, path_end_id):
    """
    Return (constrained: bool, input_length: int)

    Same logic as GraphConstrainedDecoding.check_constrained_flag():
    - Find last <PATH> token
    - If none after it, and no </PATH> after it → constrained
    - If </PATH> found after last <PATH> → not constrained
    """
    pass
```

Then wrap it into an `allowed_tokens_fn`:

```python
def allowed_tokens_fn(tokens_so_far, trie_dict, path_start_id, path_end_id, all_token_ids):
    """
    Full GCR logic:
    - If constrained: return trie query result
    - If not constrained: return all_token_ids (free generation)
    - If trie returns empty (edge case): fall back to all_token_ids
    """
    pass
```

### Full Solution

If you get stuck, the solution is essentially a simplified version of what's in `src/trie.py` (the `Trie` class, ~96 lines) + `src/graph_constrained_decoding.py` (the `GraphConstrainedDecoding` class, ~46 lines). Your implementation should be about 80-100 lines total.

**Once you have this working, you understand the core of GCR.** The actual LLM is just a fancier way to fill in the blanks — the trie does all the heavy lifting for faithfulness.

---

## 7. Running on Colab

The file `GCR_Colab.ipynb` in the repository root is a ready-to-run Colab notebook.

**What it does:**

1. Installs dependencies
2. Loads the pre-trained GCR-Qwen2-0.5B-Instruct model (~1GB)
3. Runs graph-constrained decoding on 3 examples from RoG-webqsp
4. Optionally runs inductive reasoning with GPT-4o-mini (if you provide an OpenAI key)

**Model sizes:**

| Model | Size | Min VRAM | Colab Pro? |
|-------|------|----------|------------|
| GCR-Qwen2-0.5B-Instruct | ~1 GB | 4 GB | ✅ T4 free |
| GCR-Qwen2-1.5B-Instruct | ~3 GB | 6 GB | ✅ T4 free |
| GCR-Meta-Llama-3.1-8B-Instruct | ~16 GB | 20 GB | ✅ A100 (paid) |

**Note:** Flash Attention 2 requires compute capability >= 8.0 (A100, H100). T4 is sm_75. Use `--attn_implementation sdpa` on T4.
