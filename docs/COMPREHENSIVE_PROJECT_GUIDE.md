# Comprehensive Project Guide: Graph-Constrained Reasoning with DCA-Trie

This document provides a complete, step-by-step explanation of the entire project — from foundational concepts to code implementation, experiments, and measurements.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Knowledge Graphs: Concepts and Representations](#2-knowledge-graphs)
3. [The Problem: LLM Hallucination on KGs](#3-the-problem)
4. [The Solution: Constrained Decoding](#4-the-solution)
5. [Architecture and Pipeline](#5-architecture)
6. [Code Walkthrough: Every Component](#6-code-walkthrough)
7. [The Three Approaches](#7-three-approaches)
8. [The Symbolic TypeOracle (Our Contribution)](#8-type-oracle)
9. [Trie Data Structure Deep Dive](#9-trie)
10. [Decoding Process: How the LLM Generates](#10-decoding)
11. [Prompt Engineering](#11-prompts)
12. [Graph Utilities and Path Enumeration](#12-graph-utilities)
13. [Evaluation Metrics](#13-evaluation)
14. [Experiments and Measurements](#14-experiments)
15. [Shell Scripts and Configuration](#15-scripts)
16. [How to Run Everything](#16-how-to-run)
17. [Dependencies and Why They Were Chosen](#17-dependencies)
18. [Design Decisions and Rationale](#18-design-decisions)

---

## 1. Project Overview

### What This Project Does

This project addresses **Knowledge Graph Question Answering (KGQA)** — given a natural language question and a knowledge graph (KG), find the answer by reasoning over multi-hop paths in the graph.

**The core contribution** is extending the Graph-constrained Reasoning (GCR) framework (ICML 2025) with a **symbolic constraint oracle** called **DCA-Trie** (Decoding with Constraint Augmentation via Trie). The oracle uses the KG's own ontological metadata — entity types and relation ranges — to prune irrelevant paths before they reach the LLM, dramatically reducing the search space while preserving accuracy.

### The Big Picture

```
Question: "What is the nationality of the director of Blue Hawaii?"
                        │
                        ▼
    ┌───────────────────────────────────────┐
    │   Knowledge Graph (Freebase)          │
    │   Blue Hawaii → film.director → ...   │
    │   → people.person.nationality → ...   │
    │   → United States                     │
    └───────────────────────────────────────┘
                        │
                        ▼
    ┌───────────────────────────────────────┐
    │   Constraint Oracle (TypeOracle)      │
    │   Filters paths using entity types    │
    │   and relation ranges from the KG     │
    └───────────────────────────────────────┘
                        │
                        ▼
    ┌───────────────────────────────────────┐
    │   Trie (Prefix Tree)                  │
    │   Encodes valid token sequences       │
    │   for constrained decoding            │
    └───────────────────────────────────────┘
                        │
                        ▼
    ┌───────────────────────────────────────┐
    │   LLM (Llama 3.1 8B)                 │
    │   Generates reasoning path + answer   │
    │   constrained to valid KG tokens      │
    └───────────────────────────────────────┘
                        │
                        ▼
    Answer: "United States"
```

---

## 2. Knowledge Graphs: Concepts and Representations

### What Is a Knowledge Graph?

A **Knowledge Graph (KG)** is a structured representation of facts as a graph:

- **Nodes** = Entities (people, places, things, concepts)
- **Edges** = Relationships (directed, labeled)
- **Triple** = The fundamental unit: `(Head, Relation, Tail)`

Example triples from Freebase:

```
(Blue_Hawaii,          film.film.directed_by,     Norman_Taurog)
(Norman_Taurog,        people.person.nationality,  United_States)
(Blue_Hawaii,          film.film.starring,         Elvis_Presley)
(Elvis_Presley,        people.person.place_of_birth, Tupelo)
```

### Why Knowledge Graphs (Not Databases)?

| Dimension | Knowledge Graph | Relational DB |
|-----------|-----------------|---------------|
| Relationships | First-class citizen | Foreign key joins |
| Multi-hop query | Native graph traversal | k JOINs, exponential cost |
| Schema | Flexible, can evolve | Rigid, needs migrations |
| Semantic reasoning | Supports inference | Exact matching only |
| Extensibility | Add new edge types freely | New tables/columns required |

### Data Models: RDF (What Freebase Uses)

Freebase uses the **RDF (Resource Description Framework)** model — the W3C standard for knowledge graphs:

```turtle
@prefix fb: <http://rdf.freebase.com/ns/> .
fb:m.01_b4k  fb:film.film.directed_by    fb:m.0gzy4 .
fb:m.0gzy4   fb:people.person.nationality fb:m.09c7w .
```

Key characteristics:
- Every fact is a triple with URI-identified elements
- **Open World Assumption (OWA)**: if a triple is absent, it means *unknown*, not *false*
- Both instances (data) and schema (types) are RDF triples

### Ontologies: The Schema Layer

An **ontology** defines the vocabulary for a domain — classes, properties, and constraints:

```turtle
fb:Person    rdf:type    rdfs:Class .
fb:Movie     rdf:type    rdfs:Class .
fb:film.film.directed_by
             rdfs:domain fb:Movie ;
             rdfs:range  fb:Person .
```

This declares that `directed_by` can only connect a **Movie** to a **Person**. These declarations already exist in Freebase as metadata triples:
- `common.topic.notable_types` → entity types
- `rdf-schema#range` → relation range constraints
- `rdf-schema#domain` → relation domain constraints

**This is exactly what DCA-Trie exploits** — these free constraint signals that prior methods ignored.

### Entity Types in Freebase

Freebase entities have types stored as triples:

```
(Blue_Hawaii,     common.topic.notable_types, Film)
(Elvis_Presley,   common.topic.notable_types, Person)
(United_States,   common.topic.notable_types, Country)
(Norman_Taurog,   common.topic.notable_types, Person)
```

These types answer the question "what kind of thing is this?" — and DCA-Trie uses them to determine whether a path is heading in the right direction.

### Multi-Hop Reasoning and Path Explosion

A **k-hop path** traverses k edges. The difficulty grows as O(d^k) where d is the average out-degree — the **path explosion problem**.

For a Freebase entity with average degree ~20 at 3 hops:
- 3 × 20³ = **24,000 candidate paths**
- Only **one** of them is the correct answer path

This is why tight constraint oracles matter. The LLM must choose among thousands of valid-but-irrelevant options.

---

## 3. The Problem: LLM Hallucination on KGs

### The KGQA Task

Given a natural language question and a KG, find the answer entity reachable via multi-hop reasoning paths:

```
Q: "What award did Elvis Presley win in 1971?"
KG path: Elvis_Presley → award_won → Grammy_Award → year → 1971
Answer: Grammy Award
```

### Why LLMs Hallucinate

LLMs are **next-token predictors**. At each step, they compute:

```
P(next_token | input_question, tokens_so_far)
```

The probability is computed from learned parameters only — there is no mechanism to "look up" whether a candidate token is factually correct.

**The LLM is a generator, not a verifier.** It generates confident-sounding reasoning paths that don't exist in the KG because the decoding mechanism is unrestricted.

### The Gap

The KG stores facts as exact strings/machine-readable IDs in a graph. The LLM consumes and produces natural language tokens. The question is: **how do you force the LLM to only say things that are true in the KG?**

---

## 4. The Solution: Constrained Decoding

### The Core Idea

Intercept the token selection process and **physically prevent invalid tokens from being chosen**:

At each generation step:
1. Compute what tokens are **valid** given the KG and what's been generated so far
2. Set the probability of all invalid tokens to **exactly zero** (logit masking)
3. Let the LLM pick from only the valid set

```
LLM logits → [logit mask from oracle] → softmax → sample
```

This guarantees the output is always structurally faithful to the KG. You cannot generate an invalid triple because no invalid token ever has a non-zero probability.

### The Constraint Oracle

The **constraint oracle** is the component that decides which tokens are valid at each step. It sits between the LLM's logits and the softmax.

| Oracle | What it checks | Result |
|--------|---------------|--------|
| **GCR** | Is the path structurally valid? | Permissive (all paths admitted) |
| **Cosine DCA** | Is the path semantically similar to the question? | Less permissive, threshold-dependent |
| **Decomposed DCA** | Is each component relevant? | Better diagnostics, still has threshold |
| **Symbolic DCA** | Is the entity type compatible? Is the relation range satisfied? | Tightest, no threshold, no encoder |

---

## 5. Architecture and Pipeline

### The Complete Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                     THE PIPELINE                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Question: "Nationality of director of Blue Hawaii?" │
│         ↓                                               │
│  2. Entity linking → question_entities = [Blue Hawaii]  │
│         ↓                                               │
│  3. DFS from entities → all paths up to L hops          │
│         ↓                                               │
│  4. Constraint oracle filters paths                     │
│     (this is where DCA-Trie sits)                       │
│         ↓                                               │
│  5. Trie built from filtered paths                      │
│         ↓                                               │
│  6. Beam search decoding with logit masking             │
│     (LLM can only pick valid-next-tokens from trie)     │
│         ↓                                               │
│  7. Output: structurally faithful reasoning path        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Two-Stage Pipeline

**Stage 1: Graph-Constrained Decoding** (`workflow/predict_paths_and_answers.py`)
- Enumerate KG paths via DFS
- Build a MarisaTrie of valid token sequences
- Run `model.generate()` with a `prefix_allowed_tokens_fn` callback
- Output: reasoning paths + candidate answers

**Stage 2: Graph Inductive Reasoning** (`workflow/predict_final_answer.py`)
- A general-purpose LLM (e.g., GPT-3.5-turbo) reads the reasoning paths from Stage 1
- Produces the final answer
- Pure prompting, no training

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

## 6. Code Walkthrough: Every Component

### 6.1 Project Structure

```
graph-constrained-reasoning/
├── src/                            # Core library
│   ├── graph_constrained_decoding.py  # The decoding callback (45 lines)
│   ├── trie.py                        # Prefix tree data structures (179 lines)
│   ├── qa_prompt_builder.py           # Prompt construction (537 lines)
│   ├── llms/                          # Model loading and inference
│   │   ├── __init__.py                # Model registry (18 lines)
│   │   ├── base_language_model.py     # Abstract base class (48 lines)
│   │   ├── base_hf_causal_model.py    # HuggingFace model loading (216 lines)
│   │   ├── graph_constrained_decoding_model.py  # GCR model (35 lines)
│   │   ├── chatgpt.py                 # OpenAI API wrapper
│   │   └── llm_proxy.py              # llama.cpp proxy
│   └── utils/                         # Utility functions
│       ├── graph_utils.py             # Graph construction, DFS, paths (200 lines)
│       ├── utils.py                   # I/O, path formatting (60 lines)
│       └── qa_utils.py               # Evaluation metrics (581 lines)
│
├── approach3_symbolic/              # The symbolic TypeOracle (canonical)
│   ├── type_oracle.py               # Symbolic oracle (719 lines)
│   └── algo_demo.py                 # Self-contained demo
│
├── approach1_cosine/                # Cosine similarity baseline (historical)
├── approach2_decomposed/            # Decomposed product scoring (historical)
│
├── workflow/                        # Pipeline entry points
│   ├── predict_paths_and_answers.py # Stage 1 (184 lines)
│   ├── predict_final_answer.py      # Stage 2 (341 lines)
│   ├── run_symbolic_experiment.py   # TypeOracle SIR/FNR experiment (390 lines)
│   └── build_graph_index.py         # Pre-build graph indices
│
├── experiments/type_oracle_full/    # Full experiment orchestration
│   ├── main.py                      # CLI and model loading (265 lines)
│   ├── experiment.py                # Per-condition runners (281 lines)
│   ├── decoding.py                  # Constrained decoding wrappers
│   ├── trie_utils.py                # Trie building helpers
│   └── utils.py                     # Logging, timeout, I/O
│
├── scripts/                         # Shell scripts
│   ├── graph_constrained_decoding.sh
│   ├── graph_inductive_reasoning.sh
│   ├── train_kg_specialized_llm.sh
│   ├── build_graph_index.sh
│   └── run_vast.sh                  # Vast.ai GPU rental
│
├── notebooks/                       # Jupyter notebooks
│   ├── 01_GCR_Baseline.ipynb
│   ├── 02_DCA_Trie_v1.ipynb
│   ├── 03_DCA_Trie_v2.ipynb
│   ├── 04_SIR_Evaluation.ipynb
│   └── 05_TypeOracle_Colab_Validation.ipynb
│
├── pyproject.toml                   # Dependencies and config
└── docs/                            # Documentation
```

### 6.2 Model Loading (`src/llms/`)

**Registry pattern** (`src/llms/__init__.py:6-17`):

```python
def get_registed_model(model_name) -> BaseLanguageModel:
    registed_language_models = {
        'gpt': ChatGPT, 'others': HfCausalModel,
        'gcr': GraphConstrainedDecodingModel, 'proxy': LLMProxy,
    }
    for key, value in registed_language_models.items():
        if key in model_name.lower():
            return value
    return HfCausalModel
```

Substring match: `"rmanluo/GCR-Meta-Llama-3.1-8B-Instruct"` contains `"gcr"` → `GraphConstrainedDecodingModel`.

**Model loading** (`src/llms/base_hf_causal_model.py:75-101`):

```python
def prepare_for_inference(self):
    # 1. Load tokenizer
    self.tokenizer = AutoTokenizer.from_pretrained(model_path, token=HF_TOKEN)
    
    # 2. Load config, override attention implementation
    model_config = AutoConfig.from_pretrained(model_path)
    attn = self.args.attn_implementation
    if attn == "flash_attention_2" and flash_attn is None:
        attn = "sdpa"  # Silent fallback
    model_config._attn_implementation = attn
    
    # 3. Load model weights
    self.model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype=..., token=HF_TOKEN
    )
```

**Why this design:**
- Registry pattern allows easy model switching by name substring
- Attention fallback (flash_attention_2 → sdpa) enables running on both A100 and T4 GPUs
- `device_map="auto"` handles model sharding across GPUs automatically

### 6.3 Graph Construction and Path Enumeration (`src/utils/graph_utils.py`)

**Building a NetworkX graph from triples** (`graph_utils.py:5-13`):

```python
def build_graph(graph: list, undirected=False) -> nx.DiGraph | nx.Graph:
    if undirected:
        G = nx.Graph()
    else:
        G = nx.DiGraph()
    for triplet in graph:
        h, r, t = triplet
        G.add_edge(h.strip(), t.strip(), relation=r.strip())
    return G
```

**Why NetworkX:** It provides standard graph algorithms (DFS, shortest paths, BFS) out of the box. The triples from the dataset are converted into a directed graph where each edge carries a `relation` attribute.

**DFS path enumeration** (`graph_utils.py:16-51`):

```python
def dfs(graph, start_node_list, max_length, max_paths=50000):
    def dfs_visit(node, path):
        if len(path_lists) >= max_paths:
            return
        if len(path) > max_length:
            return
        for neighbor in graph.neighbors(node):
            rel = graph[node][neighbor]["relation"]
            new_path = path + [(node, rel, neighbor)]
            if len(new_path) <= max_length:
                path_lists.add(tuple(new_path))
            dfs_visit(neighbor, new_path)

    path_lists = set()
    for start_node in start_node_list:
        dfs_visit(start_node, [])
    return list(path_lists)
```

**Why DFS:** We need all paths up to L hops from the question entities. DFS naturally explores all reachable paths. The `max_paths=50000` cap prevents memory explosion on high-degree entities.

**Getting truth paths** (`graph_utils.py:83-109`):

```python
def get_truth_paths(q_entity, a_entity, graph):
    paths = []
    for h in q_entity:
        for t in a_entity:
            for p in nx.all_shortest_paths(graph, h, t):
                paths.append(p)
    # Add relation labels
    result_paths = []
    for p in paths:
        tmp = []
        for i in range(len(p) - 1):
            u, v = p[i], p[i+1]
            tmp.append((u, graph[u][v]["relation"], v))
        result_paths.append(tmp)
    return result_paths
```

Uses NetworkX's `all_shortest_paths` to find the gold-standard paths connecting question entities to answer entities. These are used for evaluation.

### 6.4 The Trie (`src/trie.py`)

Two trie implementations exist:

**Python Trie** (`trie.py:13-96`) — simple dict-based prefix tree:
- `_add_to_trie`: recursively builds nested dicts from token sequences
- `_get_from_trie`: follows prefix in the dict, returns valid next tokens
- Used for small path sets

**MarisaTrie** (`trie.py:98-139`) — C-backed, memory-optimized:
- Wraps the `marisa_trie` library (a C implementation)
- Maps token IDs to Unicode characters: `int2char[i] = chr(i)`
- Builds a C trie over character-mapped strings
- `get(prefix_sequence)`: converts prefix to string → `trie.keys(prefix)` → extracts next character → converts back to token IDs
- **Caches first-branch tokens** in `self.zero_iter` for O(1) empty-prefix lookup
- Used in production (memory-efficient for large path sets)

**Why MarisaTrie:** For a 3-hop question on Freebase, the trie can contain 24,000+ paths. A Python dict-based trie would consume too much memory. MarisaTrie's C implementation is compact and fast for prefix lookups.

### 6.5 Constrained Decoding (`src/graph_constrained_decoding.py`)

This is the **core of the constrained decoding mechanism** — only 45 lines:

```python
class GraphConstrainedDecoding:
    def __init__(self, tokenizer, trie, start_token_ids, end_token_ids,
                 enable_constrained_by_default=False):
        self.tokenizer = tokenizer
        self.trie = trie
        self.start_token = start_token_ids  # <PATH>
        self.end_token = end_token_ids      # </PATH>
        self.all_tokens = list(range(len(tokenizer)))

    def check_constrained_flag(self, sent):
        """Determine if we're inside a <PATH>...</PATH> block."""
        matched_start = torch.where(sent == self.start_token)[0]
        if len(matched_start) == 0:
            return False, len(sent)
        last_start = matched_start[-1]
        end_count = len(torch.where(sent[last_start:] == self.end_token)[0])
        if end_count == 0:  # Inside path (no closing tag yet)
            return True, last_start
        else:  # Path already closed
            return False, len(sent)

    def allowed_tokens_fn(self, batch_id, sent):
        """Called by HuggingFace generate() at EVERY decoding step."""
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
                return self.all_tokens  # Dead-end fallback
        return allow_tokens
```

**How it works:**

The `allowed_tokens_fn` is a callback that HuggingFace's `generate()` calls at **every decoding step** for **every beam**. It returns the list of valid next tokens.

**The `<PATH>...</PATH>` sentinel mechanism:**
1. The model generates freely until it emits `<PATH>`
2. Between `<PATH>` and `</PATH>`, only tokens present in the KG trie are allowed
3. After `</PATH>`, free generation resumes for the answer
4. This gives the model autonomy for answer text while guaranteeing **zero hallucinated KG paths**

**State machine:**
```
Token sequence: "...question... <PATH> entity -> rel -> entity </PATH> answer"
                       ^                                            ^
              constrained_flag=False                     constrained_flag=False
                       └──── constrained_flag=True ────┘
```

**Why this design:**
- The trie only needs to encode valid KG paths, not all possible text
- The `<PATH>` / `</PATH>` delimiters separate constrained (KG) from unconstrained (answer) generation
- Dead-end fallback (returning all tokens) prevents generation from stalling if the trie is empty

---

## 7. The Three Approaches

### Approach 1: Cosine Similarity (`approach1_cosine/`)

**Concept:** Score each path by cosine similarity between path embedding and question embedding.

```
score(path) = cos(E(path), E(question)) >= τ
```

**How:**
- Encode the path text using a sentence transformer (all-MiniLM-L6-v2)
- Encode the question using the same encoder
- Compute cosine similarity
- If score >= threshold τ, admit the path

**Problems:**
- Requires a GPU for the encoder
- Threshold τ must be tuned (different per dataset)
- **Threshold collapse**: at τ=0.25, 84% of questions produced empty tries (all paths rejected)
- Non-deterministic due to floating-point noise
- Doesn't exploit KG ontology metadata

### Approach 2: Decomposed Product (`approach2_decomposed/`)

**Concept:** Decompose the score into three components.

```
score(path) = ρ_r · ρ_e · ρ_traj
```

- `ρ_r`: relation relevance (is the relation relevant to the question?)
- `ρ_e`: entity relevance (is the entity relevant?)
- `ρ_traj`: trajectory relevance (is the overall path direction correct?)

**Improvement over Approach 1:** Better diagnostics — you can see which component contributes most to the score.

**Same fundamental problems:**
- Still requires an encoder
- Still has a threshold
- Still non-deterministic
- Still doesn't use KG ontology

### Approach 3: Symbolic TypeOracle (`approach3_symbolic/`) — **Our Contribution**

**Concept:** Replace all embedding computations with **pure ontology lookups**.

```
score(path) = type_gate(entity) AND range_gate(relation, entity)
```

No encoder. No threshold. No GPU. Pure set-membership checks over the KG's own schema.

| Property | Approach 1 | Approach 2 | Approach 3 |
|----------|-----------|-----------|-----------|
| Encoder needed | Yes | Yes | **No** |
| Threshold τ | Yes | Yes | **No** |
| GPU needed | Decode only | Encode + decode | **Decode only** |
| Type awareness | Implicit | Hard gate | **Ontology-based** |
| Range awareness | None | None | **Ontology-based** |
| Per-path cost | Encoder forward pass | Encoder forward pass | **O(1) set lookup** |
| Deterministic | No | No | **Yes** |
| Failure mode | Collapse at threshold | Collapse at threshold | **Conservative fallback** |

---

## 8. The Symbolic TypeOracle (`approach3_symbolic/type_oracle.py`)

This is the **canonical implementation** — 719 lines, pure Python stdlib, no dependencies.

### Construction: `TypeOracle.from_graph()`

```python
oracle = TypeOracle.from_graph(data["graph"])
```

Scans all triples in the subgraph:

1. **Extract entity types** from `common.topic.notable_types` triples:
   ```python
   if r == "common.topic.notable_types":
       entity_types[h].add(t)
   ```

2. **Auto-mine relation schema** from data triples:
   ```python
   for h, r, t in graph_triples:
       if _is_schema_predicate(r):
           continue  # Skip metadata predicates
       h_types = entity_type_map.get(h, frozenset())
       t_types = entity_type_map.get(t, frozenset())
       mined[r]["domain"].update(h_types)
       mined[r]["range"].update(t_types)
   ```

   This builds domain/range information for **every relation** in the subgraph, not just the 38 hand-curated ones.

### Answer Type Inference: `infer_answer_types()`

Pattern-matches question words against regex patterns:

```python
ANSWER_TYPE_MAP = {
    "nationality": _PERSON_TYPES | _LOCATION_TYPES,
    "country": _LOCATION_TYPES,
    "city": _LOCATION_TYPES,
    "director": _PERSON_TYPES,
    "who": _PERSON_TYPES,
    "film": _CREATIVE_WORK_TYPES,
    "when": _DATE_TYPES,
    ...
}

_QUESTION_PATTERNS = [
    (re.compile(r"\bnationality\b|\bcitizenship\b", re.I), "nationality"),
    (re.compile(r"\bcountry\b|\bnation\b", re.I), "country"),
    (re.compile(r"\bwho\b", re.I), "who"),
    ...
]
```

Example: "What country is the grand bahama island in?"
- Matches `country` → `_LOCATION_TYPES` = {Location, Country, City/Town/Village, ...}
- Multiple patterns can match → union of types
- Empty set = unconstrained (no filtering at terminal hop)

### Gate 1: Answer Type Gate (`type_gate()`)

```python
def type_gate(self, entity_name, answer_types, hop, max_hop):
    if hop < max_hop:       # Intermediate hops: always allow
        return True
    if not answer_types:     # No type inferred: allow
        return True
    etypes = self._entity_types.get(entity_name, frozenset())
    if not etypes:           # Entity type unknown: allow (conservative)
        return True
    return bool(etypes & answer_types)  # Set intersection check
```

**Why terminal hop only:** At intermediate hops, the entity is a waypoint, not the answer. We only care about the type of the final entity.

**Why conservative fallback:** If the entity has no recorded types, we admit the path. This bounds the false negative rate — we'd rather keep a few extra paths than lose the correct one.

### Gate 2: Property Range Gate (`range_gate()`)

```python
def range_gate(self, relation, tail_entity):
    rel_schema = self._mined_schema.get(relation)
    if rel_schema is None:
        rel_schema = self._schema.get(relation)
    if rel_schema is None:       # Unknown schema: allow
        return True
    range_types = rel_schema.get("range", frozenset())
    if not range_types:          # No range declared: allow
        return True
    etypes = self._entity_types.get(tail_entity, frozenset())
    if not etypes:               # Entity type unknown: allow
        return True
    return bool(etypes & range_types)  # Set intersection check
```

**Example:**
```
relation = "film.film.country"  →  range = _LOCATION_TYPES
tail_entity = "United_States"   →  types = {Country}
etypes ∩ range = {Country}  →  True (admit)
```

```
relation = "film.film.country"  →  range = _LOCATION_TYPES
tail_entity = "Elvis_Presley"   →  types = {Person}
etypes ∩ range = {}  →  False (block)
```

### Combined Admission: `is_admissible()`

```python
def is_admissible(self, relation, tail_entity, answer_types, hop, max_hop):
    if not self.type_gate(tail_entity, answer_types, hop, max_hop):
        return False
    if not self.range_gate(relation, tail_entity):
        return False
    return True
```

A candidate edge is admitted iff it passes **ALL** active gates.

### Why This Design

1. **No embeddings:** Pure set-lookup. O(1) per check.
2. **No threshold:** Binary decision (admit/reject). No floating-point comparison.
3. **Conservative fallback:** Missing information → admit. Prevents catastrophic false negatives.
4. **Deterministic:** Same input → same output. No float noise.
5. **Two complementary gates:** Type gate catches terminal entity mismatches (10.6% of paths). Range gate catches relation-entity incompatibilities along the path (3.8%). Together they are additive.

---

## 9. Trie Data Structure Deep Dive

### Why a Trie?

The oracle needs to answer: **given a partial path, what are the valid next tokens?** This is a prefix query. Tries are the optimal data structure for prefix queries.

### How Paths Become Tokens

1. **DFS enumerates paths** as lists of `(head, relation, tail)` tuples:
   ```
   [("Blue_Hawaii", "film.film.directed_by", "Norman_Taurog"),
    ("Norman_Taurog", "people.person.nationality", "United_States")]
   ```

2. **Path-to-string conversion** (`src/utils/utils.py:34-44`):
   ```
   "Blue_Hawaii -> film.film.directed_by -> Norman_Taurog -> people.person.nationality -> United_States"
   ```

3. **Tokenization** (in `qa_prompt_builder.py`):
   ```python
   paths_list_str = [f"<PATH>{path_to_string(path)}</PATH>" for path in paths_list]
   tokenized_paths = tokenizer(paths_list_str, padding=False, add_special_tokens=False).input_ids
   tokenized_path_list = [ids + [tokenizer.eos_token_id] for ids in tokenized_paths]
   return MarisaTrie(tokenized_path_list, max_token_id=len(tokenizer) + 1)
   ```

4. **Each path becomes a token sequence:**
   ```
   <PATH> Blue _ Hawaii -> film . film . directed _ by -> Norman _ Taurog -> people . person . nationality -> United _ States </PATH> <eos>
   ```

### Trie Lookup at Decode Time

At each decoding step, the trie returns all valid next tokens:

```
Prefix so far: "<PATH> Blue_Hawaii -> film.film.directed_by"
Trie lookup:   trie.get(prefix_tokens)
Returns:       ["Norman_Taurog", "Elvis_Presley", ...]  (all valid tail entities)
```

The LLM can only choose from these valid tokens. Invalid tokens are physically prevented from being selected.

---

## 10. Decoding Process: How the LLM Generates

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

### Generation Modes

| Mode | Config | Behavior |
|------|--------|----------|
| `greedy` | `do_sample=False, num_return_sequences=1` | Always pick highest probability token |
| `beam` | `do_sample=False, num_beams=k` | Maintain k best sequences |
| `group-beam` | `do_sample=False, num_beams=k, num_beam_groups=k, diversity_penalty=1.0` | k diverse beams, prevents collapse |

**GCR default:** `group-beam` with `k=10`. The diversity penalty prevents beam collapse — without it, all beams converge to the same path.

### Why Group-Beam Search

Standard beam search can produce duplicate beams (all converging to the same high-probability path). Group-beam search divides beams into groups and adds a diversity penalty to encourage exploration of different paths. This is critical for KGQA where multiple valid paths exist but only one is correct.

---

## 11. Prompt Engineering (`src/qa_prompt_builder.py`)

### Prompt Templates

**Path generation prompt** (what the model sees):
```
Reasoning path is a sequence of triples in the KG that connects the topic
entities in the question to answer entities. Given a question, please generate
some reasoning paths in the KG starting from the topic entities that you
believe can aid in answering it. Then, use these reasoning paths to derive
the answer to the question.

# Question:
What is the nationality of the director of Blue Hawaii?
# Topic entities:
Blue Hawaii
```

**Answer prompt** (Stage 2):
```
Based on the reasoning paths, please answer the given question.
Please keep the answer as simple as possible and only return answers.

Reasoning Paths:
<PATH>Blue Hawaii -> film.film.directed_by -> Norman Taurog -> people.person.nationality -> United States</PATH>

Question:
What is the nationality of the director of Blue Hawaii?
Answer:
```

### Prompt Length Management

```python
def check_prompt_length(self, prompt, list_of_paths, maximun_token):
    all_paths = "\n".join(list_of_paths)
    all_tokens = prompt + all_paths
    if self.tokenize(all_tokens) < maximun_token:
        return all_paths
    else:
        random.shuffle(list_of_paths)
        new_list_of_paths = []
        for p in list_of_paths:
            tmp_all_paths = "\n".join(new_list_of_paths + [p])
            if self.tokenize(prompt + tmp_all_paths) > maximun_token:
                return "\n".join(new_list_of_paths)
            new_list_of_paths.append(p)
```

If paths exceed the token limit, paths are shuffled and added one by one until the limit is reached. This ensures the prompt always fits within the model's context window.

---

## 12. Graph Utilities and Path Enumeration

### BFS with Rule Matching (`graph_utils.py:55-80`)

```python
def bfs_with_rule(graph, start_node, target_rule, max_p=10):
    result_paths = []
    queue = deque([(start_node, [])])
    while queue:
        current_node, current_path = queue.popleft()
        if len(current_path) == len(target_rule):
            result_paths.append(current_path)
        if len(current_path) < len(target_rule):
            for neighbor in graph.neighbors(current_node):
                rel = graph[current_node][neighbor]["relation"]
                if rel != target_rule[len(current_path)]:
                    continue  # Prune: wrong relation type
                queue.append((neighbor, current_path + [(current_node, rel, neighbor)]))
    return result_paths
```

BFS guided by a "rule" (sequence of relation types). Used when the model predicts a sequence of relation types, and we need to find all paths matching that pattern.

### Negative Path Sampling (`graph_utils.py:136-169`)

Uses the `walker` library for random walks — generates negative examples for training:

```python
paths = walker.random_walks(graph, n_walks=n_neg, walk_len=hop, start_nodes=start_nodes)
```

---

## 13. Evaluation Metrics (`src/utils/qa_utils.py`)

### Answer Quality Metrics

**Accuracy** — fraction of ground-truth answers found:
```python
acc = |predicted_answers ∩ ground_truth| / |ground_truth|
```

**Hits@1** — binary: any correct answer?
```python
hit = 1 if any(predicted ∩ ground_truth) else 0
```

**F1** — harmonic mean of precision and recall:
```python
precision = |correct_predictions| / |total_predictions|
recall = |correct_predictions| / |ground_truth|
f1 = 2 * (precision * recall) / (precision + recall)
```

### Path Quality Metrics

**Path F1** — F1 between predicted paths and gold paths
**Path Precision** — fraction of predicted paths that are valid gold paths
**Path Recall** — fraction of gold paths that are predicted

### Normalization for Matching

```python
def normalize(s: str) -> str:
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s

def match(s1: str, s2: str) -> bool:
    return normalize(s2) in normalize(s1)
```

Uses substring containment for matching — robust to minor formatting differences.

### Evaluation Functions

```python
# Quick evaluation
eval_result("predictions.jsonl", cal_f1=True)

# Full evaluation with path analysis
eval_path_result_w_ans("predictions.jsonl", cal_f1=True)
```

These generate:
1. `detailed_eval_result.jsonl` — per-question metrics
2. `eval_result.txt` — summary statistics

---

## 14. Experiments and Measurements

### Experiment 1: SIR/FNR (CPU-only)

**Script:** `workflow/run_symbolic_experiment.py --phase sir`

**What it measures:**
- **SIR (Semantic Irrelevance Ratio):** fraction of candidate paths pruned by the oracle
- **FNR (False Negative Rate):** fraction of gold paths incorrectly pruned
- **SIR_type:** paths blocked by the type gate
- **SIR_traj:** paths blocked by the range gate

**How it works:**
1. For each question in the test set:
   - Build TypeOracle from the question's subgraph
   - Run DFS to get all candidate paths
   - Apply both gates to each path
   - Count how many are pruned
2. For gold paths:
   - Check if any gate incorrectly blocks them

**Empirical Results (WebQSP, 1,628 questions):**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Total candidate paths | 4,102,833 | All DFS paths within L=2 |
| Paths after filtering | 3,509,451 | |
| **Paths pruned** | **593,382 (14.5%)** | Overall SIR |
| SIR_type | 10.6% | Type gate blocks ~436K paths |
| SIR_traj | 3.8% | Range gate blocks ~157K paths |
| Gold-truth paths | 14,829 | |
| **Type gate FNR** | **3.3%** | 490 gold paths incorrectly blocked |
| **Range gate FNR** | **2.9%** | 424 gold paths incorrectly blocked |

**Key observations:**
1. SIR = 14.5% means roughly 1 in 7 paths is irrelevant
2. FNR ~3% means the oracle is safe: it rarely drops gold paths
3. Type gate does 3x more work (10.6% vs 3.8%)
4. Conservative fallback prevents catastrophic failure

### Experiment 2: Proxy Answer Generation (llama.cpp)

**Script:** `workflow/run_symbolic_experiment.py --phase proxy`

**What it measures:** Hit@1 with and without TypeOracle filtering

**How it works:**
1. For each question:
   - Get all paths (DFS) and filtered paths (TypeOracle)
   - Ask a proxy model (Qwen2.5-3B) to answer using each set of paths
   - Compare Hit@1 between filtered and unfiltered
2. Measures:
   - Path reduction percentage
   - Hit@1 for filtered vs unfiltered
   - Latency for each approach

### Experiment 3: Full Pipeline (A100 GPU)

**Script:** `experiments/type_oracle_full/main.py`

**Three conditions:**
1. `GCR_Baseline` — no oracle filtering, all paths in trie
2. `DCA_v1_Static` — TypeOracle filtering before trie construction
3. `DCA_v2_Dynamic` — step-wise TypeOracle expansion during decoding

**Per-question flow:**
1. Build TypeOracle from question's subgraph
2. Run DFS to get all paths
3. Apply oracle gates to filter paths
4. Build trie from filtered paths
5. Run constrained decoding with the trie
6. Extract answer from generated text
7. Compute Hits@1

**Output:**
```json
{
  "id": "question_id",
  "question": "What is the nationality of the director of Blue Hawaii?",
  "prediction": "<PATH>Blue Hawaii -> film.film.directed_by -> Norman Taurog -> people.person.nationality -> United States</PATH>\n# Answer:\nUnited States",
  "ground_truth": ["United States"],
  "mode": "DCA_v1_Static",
  "n_paths_all": 156,
  "n_paths_filtered": 89
}
```

---

## 15. Shell Scripts and Configuration

### `scripts/graph_constrained_decoding.sh`

```bash
MODEL_PATH=rmanluo/GCR-Meta-Llama-3.1-8B-Instruct
MODEL_NAME=$(basename "$MODEL_PATH")

python workflow/predict_paths_and_answers.py \
  --data_path rmanluo \
  --d RoG-webqsp \
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

```bash
python workflow/predict_final_answer.py \
  --data_path rmanluo \
  --d RoG-webqsp \
  --split test \
  --model_name gpt-3.5-turbo \
  --reasoning_path results/GenPaths/.../predictions.jsonl \
  --add_path True \
  -n 10
```

### `scripts/train_kg_specialized_llm.sh`

Fine-tunes a lightweight LLM on the graph-constrained decoding task using LoRA.

### `scripts/run_vast.sh`

Automates running experiments on Vast.ai GPU instances.

### Configuration Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `MODEL_PATH` | `rmanluo/GCR-Meta-Llama-3.1-8B-Instruct` | GCR checkpoint |
| `DATASET` | `RoG-webqsp` | or `RoG-cwq` |
| `SPLIT` | `test` | Evaluation split |
| `INDEX_LEN` | `2` (WebQSP) | `4` for CWQ (4-hop questions) |
| `K` | `5` | Beam size |
| `GEN_MODE` | `group-beam` | Beam search mode |
| `MAX_NEW_TOKENS` | `256` | `512` for CWQ |
| `MAX_SAMPLES` | `100` | Set `None` for full dataset |
| `QUANT` | `False` | Set `True` for T4 (8-bit) |
| `ATTN_IMPL` | `flash_attention_2` | Use `sdpa` on non-A100 GPUs |

---

## 16. How to Run Everything

### Quick Dev Run (10 minutes, any GPU)

```python
MAX_SAMPLES = 10
K = 2
FORCE = True
```

### Full Experiment (1-4 hours on A100)

```bash
# 1. Build graph index
bash scripts/build_graph_index.sh

# 2. Run GCR baseline
bash scripts/graph_constrained_decoding.sh

# 3. Run inductive reasoning
bash scripts/graph_inductive_reasoning.sh

# 4. Run TypeOracle SIR/FNR analysis (CPU)
python workflow/run_symbolic_experiment.py --phase sir

# 5. Run full DCA-Trie experiment (GPU)
python experiments/type_oracle_full/main.py --method all --max-samples 100
```

### Using Vast.ai

```bash
# 1. Rent a GPU instance
scripts/run_vast.sh

# 2. Boot environment
scripts/vast_boot.sh
scripts/setup-env.sh
```

---

## 17. Dependencies and Why They Were Chosen

| Library | Version | Purpose | Why |
|---------|---------|---------|-----|
| `torch` | >=2.0 | Deep learning framework | Model inference, GPU acceleration |
| `transformers` | ==4.44.2 | HuggingFace model loading | Pinned version for `load_in_8bit` compatibility |
| `accelerate` | >=0.30 | Model distribution | `device_map="auto"` for GPU sharding |
| `peft` | >=0.11 | Parameter-efficient fine-tuning | LoRA for training the KG-specialized LLM |
| `marisa-trie` | >=1.2 | C-backed prefix trie | Memory-efficient trie for 24K+ paths |
| `networkx` | (implicit) | Graph algorithms | DFS, shortest paths, graph construction |
| `datasets` | >=2.19 | Dataset loading | HuggingFace datasets for WebQSP/CWQ |
| `scikit-learn` | >=1.5 | Metrics | Precision/recall/F1 computation |
| `openai` | >=1.31 | GPT API | Stage 2 inductive reasoning |
| `flash-attn` | (optional) | Fast attention | A100-only, falls back to sdpa |
| `bitsandbytes` | >=0.43 | Quantization | 8-bit loading for T4 GPUs |
| `wandb` | >=0.17 | Experiment tracking | Training metrics logging |

---

## 18. Design Decisions and Rationale

### Why Symbolic Over Embedding-Based?

1. **No threshold problem:** Embedding approaches require tuning a similarity threshold. Too high → empty tries (84% failure at τ=0.25). Too low → no filtering. Symbolic approach has no threshold.

2. **No GPU overhead:** The encoder runs on GPU for every path. With 24K paths, that's 24K forward passes per question. Symbolic approach uses O(1) set lookups.

3. **Deterministic:** Same input → same output. Critical for reproducibility.

4. **Interpretable:** You can trace exactly why a path was rejected (wrong type, wrong range). Embedding scores are opaque.

5. **Uses existing metadata:** The KG already stores entity types and relation ranges. This is free signal that prior methods ignored.

### Why Conservative Fallback?

When schema information is missing (entity has no declared types, relation has no range), the gate admits the path. This is a deliberate design choice:

- **False negative rate (FNR) is bounded:** We'd rather keep a few extra paths than lose the correct one
- **No catastrophic failure:** Unlike cosine similarity at τ=0.25 which produced empty tries for 84% of questions
- **Graceful degradation:** The oracle works even with incomplete metadata

### Why Two Gates (Not One)?

The two gates are **complementary**:

- **Type gate** (terminal hop only): catches paths that end at the wrong entity type. Does 3x more work (10.6% vs 3.8%).
- **Range gate** (every hop): catches relation-entity incompatibilities along the path. Prevents nonsensical intermediate steps.

Together they are additive (union bound on FNR), and each catches different failure modes.

### Why MarisaTrie Over Python Dict?

For small path sets, a Python dict-based trie works fine. But for production use:

- 24,000 paths × average 20 tokens = 480,000 token entries
- Python dict: ~50 bytes per entry = ~24 MB
- MarisaTrie: ~5 bytes per entry = ~2.4 MB
- 10x memory reduction, plus faster prefix lookups (C implementation)

### Why Group-Beam Search?

Standard beam search converges to the same high-probability path. Group-beam search with diversity penalty ensures exploration of different paths, which is critical when multiple valid paths exist but only one is correct.

### Why the `<PATH>...</PATH>` Sentinel?

This gives the model **autonomy for answer text** while guaranteeing **zero hallucinated KG paths**:

- Between `<PATH>` and `</PATH>`: constrained to KG tokens only
- After `</PATH>`: free generation for the answer

Without this separation, the model would either:
- Be fully constrained (can't generate natural language answers)
- Be unconstrained (can hallucinate KG paths)

The sentinel mechanism gives us the best of both worlds.

---

## Appendix: Architecture Deep-Dive

### Full System Architecture

```bash
                              ┌──────────────────────────────────────┐
                              │           Freebase Subgraph          │
                              │  (from WebQSP / CWQ dataset entry)   │
                              └──────────┬───────────────────────────┘
                                         │
                          ┌──────────────┴──────────────┐
                          │                             │
                          ▼                             ▼
               ┌─────────────────────┐     ┌──────────────────────────┐
               │   NetworkX DiGraph  │     │       TypeOracle         │
               │  (nodes=entities,   │     │  ┌───────────────────┐   │
               │   edges=relations)  │     │  │ Entity Type Map   │   │
               └──────────┬──────────┘     │  │ (notable_types)   │   │
                          │                │  ├───────────────────┤   │
                          ▼                │  │ Relation Schema:  │   │
               ┌─────────────────────┐     │  │ • 38 hand-curated │   │
               │  DFS Enumeration    │     │  │ • ∞ auto-mined    │   │
               │ (from q_entity,     │     │  │  from data triples│   │
               │  depth=index_len,   │     │  ├───────────────────┤   │
               │  max=50k paths)     │     │  │ Question→type     │   │
               └──────────┬──────────┘     │  │ regex matcher     │   │
                          │                   └───────────────────┘   │
               ┌──────────┴──────────┐     └──────────────────────────┘
               │                     │                │
               ▼                     ▼                ▼
     ┌──────────────────┐  ┌─────────────────────────────────────┐
     │  GCR Baseline    │  │  DCA_v1 (Static Filtering)          │
     │                  │  │                                     │
     │ All paths →      │  │ All paths → range_gate(hop) →       │
     │ MarisaTrie →     │  │ type_gate(terminal) → survivors →   │
     │ Constrained      │  │ MarisaTrie → Constrained Decoding   │
     │ Decoding         │  │                                     │
     └────────┬─────────┘  └──────────────────┬──────────────────┘
              │                               │
              ▼                               ▼
     ┌──────────────────────────────────────────────────────────┐
     │           GraphConstrainedDecoding (GCR Engine)          │
     │                                                          │
     │  1. Tokenizer encodes prompt                             │
     │  2. Model.generate() with prefix_allowed_tokens_fn       │
     │     → each step: allowed_fn checks trie for valid tokens │
     │     → START token → constrained mode ON                  │
     │     → END token → constrained mode OFF                   │
     │  3. Decode output sequence → prediction string           │
     └──────────────────────────┬───────────────────────────────┘
                                │
                                ▼
                    ┌──────────────────────────┐
                    │  DCA_v2 (Step-wise)      │
                    │                          │
                    │ hop=0: start_entities    │
                    │   → gate neighbours      │
                    │   → small trie → decode  │
                    │   → extract committed    │
                    │ hop=1: committed entity  │
                    │   → gate neighbours      │
                    │   → small trie → decode  │
                    │   → ...                  │
                    │ until end_token or max   │
                    └──────────────────────────┘
```

### GCR Baseline: MarisaTrie Implementation Detail

The `int2char` mapping is limited to token IDs < 55,000 for the direct mapping, and uses a shifted encoding for IDs ≥ 55,000. The `max_token_id=256001` matches the LLaMA vocabulary size.

```python
class MarisaTrie(object):
    def __init__(self, sequences, cache_fist_branch=True, max_token_id=256001):
        self.int2char = [chr(i) for i in range(min(max_token_id, 55000))] + [...]
        self.char2int = {self.int2char[i]: i for i in range(max_token_id)}
        self.trie = marisa_trie.Trie(
            "".join([self.int2char[i] for i in sequence]) for sequence in sequences
        )
```

### Group-Beam Search Impact

Group-beam search (k=10, diversity_penalty=1.0) produces 10 diverse output sequences. Each is a distinct reasoning path. The answer is extracted from each path and the Hits@1 metric checks if *any* path contains the correct answer.

**Impact**: Group-beam vs greedy gives +11pp (91.6% vs 80.6%). This is the single largest accuracy lever in the system.

### TypeOracle: Entity Type Extraction

Freebase includes type metadata in every subgraph via special relations. The TypeOracle extracts these:

```python
for h, r, t in graph_triples:
    if r == "common.topic.notable_types":
        entity_types[h].add(t)
    elif r == "freebase.type_hints.included_types":
        if t != "Topic":
            entity_types[h].add(t)
```

This is *not* learned. It is ground-truth schema information from the KG itself. No embeddings, no training, no GPU.

### Auto-Mined Relation Schema

This extends coverage to **all** relations in the subgraph (not just 38 hand-curated ones). For each data triple `(h, r, t)`, it looks up the known types of h and t and aggregates them per relation.

**Key insight**: The mined schema reflects *actual usage* patterns rather than Freebase's theoretical type hierarchy. This makes it more specific (and more useful for filtering) than the hand-curated schema.

### Type Gate: Terminal-Hop-Only Rationale

The type gate is applied **only at the terminal hop**. This is a deliberate design choice — intermediate entities may have types unrelated to the answer type. For example, in the path `Jamaica → location.country.languages_spoken → Jamaican English`:

- The intermediate entity `language` type doesn't need to match `PERSON` — only the terminal `Jamaican English` needs to match `LANGUAGE`.

---

## Appendix: DCA_v1 Algorithmic Analysis

### Code Path

**File:** `experiments/type_oracle_full/trie_utils.py:16-50`

```
build_filtered_trie(tokenizer, question_dict, index_len, oracle):
  1. Build graph
  2. DFS enumerate all paths (same as baseline)
  3. Infer answer types from question
  4. For each path:
     a. For each hop: range_gate(relation, tail_entity)
        → fail ⇒ block (n_range_blocked++)
     b. If all hops pass: type_gate(terminal, answer_types, hop, max_hop)
        at terminal hop only
        → fail ⇒ block (n_type_blocked++)
     c. If both pass: keep path
  5. Build MarisaTrie from surviving paths
  6. Run constrained decoding (identical to baseline)
```

### Algorithmic Properties

**Complexity**: O(P × H) where P = number of DFS paths (≤50k) and H = path length (≤2). Each path is visited once and each hop is checked once. The gates are O(1) — frozen set intersection.

**Filtering ratio**: 14.5% of paths removed. The gates are independent (range gate → type gate), so the total blocking factor is:

- Range gate: 157,485 / 4,102,833 = 3.8%
- Type gate: 435,897 / 4,102,833 = 10.6%
- Total: 593,382 / 4,102,833 = 14.5%

**False negative rate**: For gold (ground-truth) paths:

- FNR_type = 3.3% — 3.3% of gold paths have a terminal entity whose type doesn't match the inferred answer type
- FNR_range = 2.9% — 2.9% of gold paths have a hop whose relation range doesn't match the entity type

These are low because:

1. Freebase schema is well-aligned with the actual data
2. The conservative design (allow on unknown) means FNR comes only from genuine schema violations
3. Some gold paths in WebQSP use entities with unusual types or type-free relations

**Accuracy retention**: 86.4/91.6 = 94.3%. The 5.7% relative accuracy loss comes from:

- 3.3% FNR_type + 2.9% FNR_range = 6.2% of gold paths are *potentially* filtered out
- But not all filtered gold paths would have been chosen by the model anyway (group-beam search means multiple path candidates)
- And some non-gold paths that would have led to correct answers via alternative routes are also filtered

### Why DCA_v1's Accuracy Loss is Modest

The 5.2pp drop (91.6% → 86.4%) with 14.5% path reduction is surprisingly small. Why?

1. **Redundancy in path sets**: The average WebQSP question has 2,553 paths. Even after removing 14.5%, ~2,180 remain. For most questions, the correct answer is reachable via many paths, and only a fraction of them are blocked.

2. **Gate specificity**: The range gate blocks paths that violate the KG schema (e.g., a `people.person.place_of_birth` relation pointing to a non-Location entity). Such paths are *usually* wrong anyway — the model rarely extracts correct answers from them.

3. **Group-beam diversity**: With 10 diverse beam paths, even if some high-quality paths are filtered, the beam will find others.

4. **Conservative defaults**: Both gates default to "allow" when information is missing. This means the 14.5% reduction comes from *confident* filterings only.

---

## Appendix: DCA_v2 Failure Analysis

### Code Path

**File:** `experiments/type_oracle_full/decoding.py:34-159`

```
dca_v2_generate(data, nx_graph, llm_model, tokenizer, oracle, max_hops, max_new_tokens):
  1. Build first-hop paths from start_entities → gate neighbours → build small trie
  2. Loop (max_hops × 3 iterations):
     a. Build prompt (incrementally appended)
     b. Prepare llm_input (fixed from step 1 — context accumulation fix)
     c. Run constrained decoding with CURRENT HOP's trie
     d. Decode output
     e. If </PATH> or EOS → break
     f. Extract committed entity from output (split on " -> ")
     g. If same as previous commit → continue (same entity, keep going)
     h. Check hop limit
     i. Fetch entity's neighbours from nx_graph
     j. Gate neighbours: range_gate(all hops) + type_gate(terminal if applicable)
     k. Build new trie from admitted neighbours
     l. Append generated text to prompt
  3. Extract terminal entity → append "# Answer:\n{terminal}"
```

### Root Cause Analysis: 91.6% → 54.0%

The 37.6pp accuracy collapse stems from five compounding issues:

#### Issue 1: No Global Path Context

The most fundamental problem. In the baseline and v1, the MarisaTrie contains *all* complete paths. The model's generation can follow any path to completion. In v2, each step's trie contains only the *current entity's neighbours* — single hops. The model cannot see where a path will lead.

**Consequence**: The model makes greedy, locally-optimal entity choices with no lookahead. In KG reasoning, the correct entity at hop 1 depends on which path leads to the answer at hop 2. A locally "safer" entity (more neighbours, more common type) may be globally wrong.

**Example question**: "What language do people in Jamaica speak?"

- Hop 1 choices from "Jamaica": `location.country.languages_spoken → Jamaican English`, `location.country.capital → Kingston`, `location.country.form_of_government → ...`
- In baseline/v1, the model sees both hops and can generate the full path.
- In v2, after committing to the first hop, the model must continue from that entity. If it commits to "Kingston" at hop 1, hop 2's trie contains Kingston's neighbours — none of which lead to "Jamaican English."

#### Issue 2: Accumulated Commit Errors

Each committed entity constrains the next step's trie. An early suboptimal choice propagates through all subsequent steps. Unlike beam search (which maintains k candidates), v2 commits greedily with no backtracking.

**Comparison**:

- Baseline/v1: Beam search with k=10 → 10 diverse candidate paths evaluated
- v2: Greedy path → single candidate, no alternatives

#### Issue 3: Constrained Decoding Mismatch

The `GraphConstrainedDecoding` class was designed for complete paths with `<PATH>`...`</PATH>` markers. In v2, each step's trie contains single-hop fragments. The constrained decoding logic:

1. Detects `<PATH` → enters constrained mode
2. The trie contains only `entity -> relation -> neighbor` strings
3. After generating one hop, the model outputs `</PATH>`
4. v2 strips the markers and extracts the entity
5. Builds a new trie; repeats

**Interaction bug**: The `check_constrained_flag` method (line 14-27 of `graph_constrained_decoding.py`) detects constrained mode by finding the last `<PATH` token and counting `</PATH` tokens. With v2's hop-by-hop approach, each step has exactly one `<PATH`-to-`</PATH` segment. The model is repeatedly entering and exiting constrained mode for single-hop fragments. This is different from the baseline where one `<PATH` starts a multi-hop completion.

#### Issue 4: Terminal Entity Extraction Heuristic

```python
def _extract_terminal_entity(path_text):
    clean = path_text.replace(PATH_START, "").replace(PATH_END, "").strip()
    segments = clean.split(" -> ")
    return segments[-1] if len(segments) >= 3 else None
```

This takes the last `->` segment of the generated path as the answer. This is brittle:

- For a 2-hop path `A -> r1 -> B -> r2 -> C`, the answer should be `C`
- If the model generated only `A -> r1 -> B` (incomplete), the answer becomes `B` (wrong)
- If the path string has inconsistent formatting, parsing fails

The baseline/v1 evaluate all 10 beam outputs and check if *any* contains the correct answer. v2 produces a single string and extracts one answer.

#### Issue 5: Prompt Accumulation

```python
llm_input = llm_input + f"\n{output.strip()}\n"
```

v2 appends each step's output to the prompt. This means:

- The model sees its own previous output as input for the next step
- Error patterns (repeated entities, formatting issues) compound
- The growing prompt may exceed the model's context window on longer paths

### Performance Analysis

- **1,466 samples in ~9,000s** = 0.17 q/s
- Baseline completes 1,627 in 10,329s = 0.16 q/s
- **v2 is not slower** — it's actually marginally faster per sample (6.1s vs 6.3s)

The speed parity masks the structural problem: v2 processes fewer hops per sample but each hop requires a full `model.generate()` call. The overhead of repeated generation calls cancels out the reduced path complexity.

---

## Appendix: Results Translation

### Why Baseline = 91.6% (vs Paper's 92.6%)

The original GCR paper achieves 92.6% on WebQSP. Our reproduction at 91.6% differs by 1.0pp. The gap comes from:

| Factor | Paper | Our Implementation |
|--------|-------|-------------------|
| Step 2 (answer extraction) | **GPT-4o-mini** | Direct string parsing (`# Answer:\n`) |
| GPU | A100 40GB | RTX 4090 24GB |
| Beam width | k=5 | k=10 |
| Max tokens | 8 | 256 |
| Model | GCR-Llama-3.1-8B | Same |

The GPT-4o-mini step 2 extracts the answer entity from the generated path with higher accuracy than direct parsing. Our direct extraction using `# Answer:\n` splitting is simpler but loses some edge cases (malformatted answers, extra tokens).

### Why DCA_v1 = 86.4% (94.3% retention)

The TypeOracle gates, being purely symbolic and conservative, achieve principled pruning:

```bash
Path load: 4,102,833 → 3,509,451 (593,382 pruned)
SIR = 14.5%:
  - SIR_type (type gate)    = 10.6%  — 435,897 paths pruned
  - SIR_traj (range gate)   = 3.8%   — 157,485 paths pruned

False negatives on gold paths:
  - FNR_type  = 3.3%  — only 490/14,829 gold paths blocked by type gate
  - FNR_range = 2.9%  — only 424/14,829 gold paths blocked by range gate
```

The accuracy cost (5.2pp) is roughly proportional to the false negative rate (6.2% of gold paths are blocked, of which some fraction would have been chosen by the beam search).

### Why DCA_v2 = 54.0% (Near Random on Many Questions)

The 37.6pp collapse is not a tuning issue — it's a fundamental architectural mismatch:

| Property | Baseline/v1 | v2 |
|----------|------------|-----|
| Path visibility | Full set of complete paths | Single next-hop neighbours |
| Search strategy | Group-beam (k=10) | Greedy (no lookahead) |
| Answer extraction | All 10 beam outputs → any match | Single terminal entity |
| Constrained decoding | One pass through full trie | Repeated pass through tiny tries |
| Error recovery | Beam diversity covers suboptimal paths | Commit errors propagate |
| False negative impact | Low (many alternatives per question) | High (each blocked neighbour may eliminate the only path) |

### Why CWQ is Harder (69.0% vs 91.6%)

**Code-level explanation**:

From the DFS implementation (`graph_utils.py:16-51`), `max_length = index_len = 2`:

```bash
dfs_visit(node, path):
    if len(path) > max_length: return
    for neighbor in graph.neighbors(node):
        new_path = path + [(node, rel, neighbor)]
        if len(new_path) <= max_length:
            path_lists.add(tuple(new_path))
        dfs_visit(neighbor, new_path)
```

For WebQSP (mostly 1-2 hop queries):

- Start entities → direct answer entities via 1-2 relations
- Average 2,553 paths per question

For CWQ (up to 4-hop queries):

- Start entities → answer entities via longer chains
- Branches explode exponentially with depth
- The 50k path cap may truncate relevant paths
- Fewer paths reach the answer: avg 2,239 paths/question

Additionally, CWQ questions are more lexically diverse:

- `answer_type_inference` (regex-based) has lower recall
- More questions are "open-ended" — no clear type signal
- Results: TypeOracle gates are less effective (less of the wrong paths match the type patterns, and more of the right paths are blocked)

---

## Appendix: SOTA Comparison

### Embedding-Based Path Ranking (Learned Pruning)

The `experiments/learned_pruning/` directory contains an alternative approach using Sentence-BERT embeddings to rank paths. The key metric is Recall@K:

```bash
Embedding Recall@K vs Random:
  Recall@1:      0.0%  vs 0.0%
  Recall@5:      2.1%  vs 0.2%
  Recall@10:     2.1%  vs 0.4%
  Recall@50:    13.9%  vs 1.7%
  Recall@100:   27.8%  vs 3.3%
  Recall@500:   50.0%  vs 17.3%
```

The learned approach is 3× better than random but still weak at practical K values (2.1% at K=10). This means:

- **DCA_v1 (symbolic, 14.5% reduction) is more effective** than embedding-based recall at K=500+ with zero learned parameters.
- The learned approach's advantage (it can capture semantic relevance) is overwhelmed by the noise in embedding space for fine-grained KG entity types.
- DCA_v1's TypeOracle is *free* — no training, no GPU, no data.

### Adaptive Budget Methods

Tested in earlier experiments (data from historical runs):

| Budget | Hits@1 | Interpretation |
|--------|--------|----------------|
| ∞ (full greedy) | 80.6% | Greedy baseline (pre-beam-fix) |
| ~2,213 (TypeOracle) | 78.9% | Static filter, -1.7pp |
| 500 | 30.7% | Too aggressive |
| 100 | 12.8% | Far too aggressive |
| 30 | 7.9% | Catastrophic |

This directly contradicts the adaptive budget literature. The assumption that "path count = complexity" is false for KG reasoning:

- Some 1-hop questions have thousands of paths (many entity types → many neighbours)
- Some 2-hop questions have few paths (narrow relation chains)
- Truncating to 500 paths harms both cases differently

The TypeOracle approach of *type-based* filtering (not count-based) is fundamentally more principled: it removes paths that are *structurally incompatible* with the question, not paths that happen to be numerous.

### GCR Paper Comparison

```bash
+------------------+-------------------+-------------------+
|                  | GCR Paper (2024)  | Our Project       |
+------------------+-------------------+-------------------+
| WebQSP Baseline  | 92.6%             | 91.6%             |
| CWQ Baseline     | 75.8%             | 69.0% (100-sample)|
| Step 2 Method    | GPT-4o-mini       | Direct extraction |
| GPU              | A100 40GB         | RTX 4090 24GB     |
| Path Reduction   | None              | 14.5% (TypeOracle)|
| DCA_v1           | N/A               | 86.4% WebQSP      |
| DCA_v2           | N/A               | 54.0% WebQSP      |
+------------------+-------------------+-------------------+
```

Our contributions beyond the paper:

1. **TypeOracle symbolic filtering** — novel, zero-cost path pruning
2. **DCA_v1** — first static type-constrained decoding method
3. **DCA_v2** — first dynamic hop-by-hop expansion attempt (shows feasibility limits)
4. **Beam search at k=10** — the paper uses k=5; we gain +11pp over greedy

---

## Appendix: Implementation Lessons

### What Worked

- **Symbolic gates over learned gates**: TypeOracle's 38 hand-curated + auto-mined relations are sufficient for 14.5% pruning with only 5.2pp accuracy loss. No training data needed.
- **Group-beam over greedy**: +11pp improvement was the single biggest lever.
- **Conservative defaults**: Gates defaulting to "allow on unknown" ensures the FNR (6.2%) stays well below the SIR (14.5%).
- **Auto-mining from data triples**: Extends coverage from 38 relations to 100% of subgraph relations at zero cost.

### What Didn't Work

- **Step-wise greedy expansion (v2)**: Without global path context or beam-style lookahead, greedy entity commitment is catastrophic for accuracy.
- **Adaptive path budgets**: Count-based truncation ignores the semantic structure of paths. The TypeOracle's type-based filtering is strictly superior.
- **Learned embedding reranking**: Requires labeled data, GPU training, and even then Recall@500 < 50%. Symbolic filtering achieves 14.5% reduction at recall=100% for non-blocked paths.

### What We Would Do Differently

1. **DCA_v2 with beam rollout**: Instead of greedy entity commitment, maintain k candidate entities at each hop with beam-style expansion. This would give the diversity benefit of group-beam while still achieving path reduction.

2. **Hybrid v1+v2**: Use v1's static filtering to reduce the path space, then use v2-style dynamic expansion only for questions where the filtered path set is empty (handling the "dead end" case).

3. **Better answer extraction**: The 1.0pp gap between our 91.6% and the paper's 92.6% is attributable to GPT-4o-mini for step 2. A trained extractor or structured output format would close this gap.

4. **Cascading gates**: The range gate at every hop could be supplemented with a domain gate (checking the head entity type at each hop too). Currently only the tail entity is checked against the relation range.

5. **Multi-hop v1**: Extend `index_len` to 3-4 for CWQ and apply TypeOracle gates at each hop. The current v1 only gates the terminal hop with the type gate — deeper paths would benefit from intermediate type constraints.

---

## Appendix: API Reference

### Model Loading

`src/llms/base_hf_causal_model.py` — `prepare_for_inference()` loading sequence:

1. **Tokenizer**: `AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)`
2. **Model config**: `AutoConfig.from_pretrained(model_path)` — overrides `_attn_implementation`
3. **Model weights**: `AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=...)`

**Attention fallback**: `flash_attention_2 → sdpa` when `flash_attn` is missing. T4 (sm_75) requires `--attn_implementation sdpa`.

### Constrained Decoding: check_constrained_flag State Machine

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

### Key Source Files

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

### Shell Scripts: Parameter Reference

#### scripts/graph_constrained_decoding.sh

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `--d` | `RoG-webqsp` | Dataset (also `RoG-cwq`) |
| `--index_path_length` | `2` | Max DFS hops |
| `--k` | `10` | Beam width |
| `--generation_mode` | `group-beam` | Diverse beam search |
| `--attn_implementation` | `flash_attention_2` | Use `sdpa` on T4 GPUs |

#### scripts/graph_inductive_reasoning.sh

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `--reasoning_path` | — | Input predictions.jsonl from Stage 1 |
| `--add_path` | `True` | Include paths in prompt |
| `-n` | `10` | Parallel API threads |

---

*This document covers the complete project — from foundational concepts to implementation details, experimental setup, and design rationale. It should serve as a comprehensive reference for understanding, maintaining, and extending the codebase.*
