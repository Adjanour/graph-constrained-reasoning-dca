# TypeOracle Code Walkthrough

Detailed walkthrough of the TypeOracle implementation — data structures, algorithms, regex patterns, gate operations, and time complexity.

Source: `approach3_symbolic/type_oracle.py` (719 lines)

---

## 1. Architecture Overview

The TypeOracle has three layers:

```
Layer 1: Type Constants        Frozen sets of human-readable Freebase types
Layer 2: Schema                Relation → {domain, range} mappings (hand-curated + auto-mined)
Layer 3: Gates                 Boolean checks that filter candidate edges
```

The oracle is built once per question from the KG subgraph, then queried O(1) per candidate edge during path enumeration.

---

## 2. Data Structures

### 2.1 Frozen Sets for Type Groups

```python
_PERSON_TYPES: FrozenSet[str] = frozenset({
    "Person", "Deceased Person", "Politician", "Musical Artist",
    "Author", "Academic", "Film director", "Film actor", ...
})
```

**Why `frozenset`:**
- Immutable — can be used as dict values and in set intersections
- Hashable — can be stored in other sets/dicts
- O(1) membership test via hash table
- O(min(|A|,|B|)) intersection via `A & B`

**Type groups defined:** Person (16), Location (7), Organization (9), Creative Work (10), Date (1), Language (1), Award (2), Profession (1), Genre (6). Total: 53 unique types across 9 groups.

### 2.2 Answer Type Map

```python
ANSWER_TYPE_MAP: Dict[str, FrozenSet[str]] = {
    "nationality": _PERSON_TYPES | _LOCATION_TYPES,   # union
    "country": _LOCATION_TYPES,
    "who": _PERSON_TYPES,
    "film": _CREATIVE_WORK_TYPES,
    ...
}
```

Maps question keywords → expected answer type sets. The `|` operator creates a new frozenset (union) at module load time, so runtime lookups are O(1).

### 2.3 Relation Schema

```python
_RELATION_SCHEMA: Dict[str, Dict[str, FrozenSet[str]]] = {
    "people.person.nationality": {
        "domain": _PERSON_TYPES,
        "range": _LOCATION_TYPES,
    },
    ...
}
```

38 hand-curated relations with known domain/range constraints. Each value is a dict with two frozenset fields.

### 2.4 Entity Type Map (per-question)

```python
self._entity_types: Dict[str, FrozenSet[str]]
```

Built at runtime by `from_graph()`. Maps entity name → set of Freebase types. Example:

```python
{
    "Jamaica": frozenset({"Country", "Location"}),
    "Benjamin Franklin": frozenset({"Person", "Inventor", "Author"}),
    "m.0k8nh0b": frozenset(),  # Freebase ID — no types recorded
}
```

---

## 3. Construction: `from_graph()`

```python
@classmethod
def from_graph(cls, graph_triples: List[List[str]]) -> "TypeOracle":
```

### Algorithm

```
Input:  graph_triples = [(h, r, t), ...]  (raw KG triples)
Output: TypeOracle instance

1. entity_types ← empty dict of {entity → set of types}
2. FOR each (h, r, t) in graph_triples:
     IF r == "common.topic.notable_types":
       entity_types[h].add(t)
     ELIF r == "freebase.type_hints.included_types" AND t ≠ "Topic":
       entity_types[h].add(t)
     ELIF r == "freebase.type_profile.strict_included_types" AND t ≠ "Topic":
       entity_types[h].add(t)
3. Freeze each entity's type set → frozenset
4. mined ← mine_relation_schema(graph_triples, entity_types)
5. RETURN TypeOracle(entity_types, _RELATION_SCHEMA, mined)
```

### Time Complexity

Let `T` = number of triples, `E` = number of unique entities.

- Step 2: O(T) — single pass over triples
- Step 3: O(E) — freeze each entity's types
- Step 4: O(T) — single pass (see §4)
- **Total: O(T)**

For WebQSP: T ≈ 5,000-50,000 triples per question → construction takes <1ms.

---

## 4. Schema Mining: `_mine_relation_schema()`

```python
@staticmethod
def _mine_relation_schema(
    graph_triples: List[List[str]],
    entity_type_map: Dict[str, FrozenSet[str]],
) -> Dict[str, Dict[str, FrozenSet[str]]]:
```

### Algorithm

```
Input:  graph_triples, entity_type_map
Output: mined schema {relation → {domain: frozenset, range: frozenset}}

1. mined ← empty dict of {relation → {domain: set, range: set}}
2. FOR each (h, r, t) in graph_triples:
     IF r is a schema predicate (common.topic.*, type.*, etc.):
       SKIP
     h_types ← entity_type_map.get(h, empty)
     t_types ← entity_type_map.get(t, empty)
     IF h_types ≠ ∅:  mined[r].domain ∪= h_types
     IF t_types ≠ ∅:  mined[r].range  ∪= t_types
3. FOR each relation in mined:
     Freeze domain and range sets → frozenset
4. RETURN mined (only relations with non-empty domain or range)
```

### Why This Works

For a triple `(Jamaica, location.country.languages_spoken, Jamaican English)`:
- `Jamaica` has types `{Country, Location}` → mined schema records these as the **domain** of `location.country.languages_spoken`
- `Jamaican English` has types `{Human Language}` → mined schema records this as the **range**

After processing all triples, each relation's domain/range reflects the **actual usage patterns** in the subgraph — more specific than Freebase's theoretical type hierarchy.

### Schema Predicate Filtering

```python
_SCHEMA_PREDICATES: FrozenSet[str] = frozenset({
    "common.topic.notable_types",
    "freebase.type_hints.included_types",
    "rdf-schema#domain",
    "rdf-schema#range",
    "type.property.expected_type",
    ...
})

def _is_schema_predicate(r: str) -> bool:
    if r in _SCHEMA_PREDICATES: return True
    if r.startswith("type."): return True
    if r.startswith("base.") and "schema" in r: return True
    return False
```

Skips 17 explicit predicates + any `type.*` + `base.*schema*`. This prevents metadata triples from polluting the mined domain/range.

### Time Complexity

- Step 2: O(T) — single pass, O(1) set operations per triple
- Step 3: O(R) — R = number of unique relations (typically 50-200)
- **Total: O(T + R) = O(T)** since R ≤ T

---

## 5. Answer Type Inference

### 5.1 Regex-Based: `infer_answer_types()`

```python
def infer_answer_types(self, question: str) -> FrozenSet[str]:
    q = self._normalize_question(question)
    matched: Set[str] = set()
    for pattern, type_key in _QUESTION_PATTERNS:
        if pattern.search(q):
            matched.update(ANSWER_TYPE_MAP.get(type_key, frozenset()))
    return frozenset(matched)
```

### Question Patterns (18 regexes)

```python
_QUESTION_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bnationality\b|\bcitizenship\b", re.I), "nationality"),
    (re.compile(r"\bcountry\b|\bnation\b", re.I), "country"),
    (re.compile(r"\bwho\b", re.I), "who"),
    (re.compile(r"\blanguage\b|\bspoken\b|\bspeak\b", re.I), "language"),
    (re.compile(r"\bwhat.*profession|what.*do\b", re.I), "profession"),
    ...
]
```

**Pattern design:**
- `\b` word boundaries prevent partial matches (e.g., "speak" won't match "speaker")
- `re.I` case-insensitive
- Multi-word patterns use `.*` (e.g., `what.*profession`)
- Each pattern maps to a type key, which maps to a frozenset of types

### Normalization

```python
@staticmethod
def _normalize_question(question: str) -> str:
    q = re.sub(r'"[^"]+"', " [ent] ", question)
    q = re.sub(r"'[^']+'", " [ent] ", q)
    return q
```

Masks quoted entity mentions (e.g., `"Benjamin Franklin"`) to prevent matching on them. The entity name might contain words like "who" or "country" that would trigger false positives.

### Example

```
Question: "what language is spoken in jamaica"
After normalization: "what language is spoken in jamaica"
Matched patterns: \blanguage\b|\bspoken\b|\bspeak\b → "language"
Answer types: _LANGUAGE_TYPES = {"Human Language"}
```

### Time Complexity

- Normalization: O(Q) where Q = question length
- Pattern matching: O(P × Q) where P = number of patterns (18)
- **Total: O(P × Q) = O(Q)** since P is constant

### 5.2 KG-Based Fallback: `infer_answer_types_from_paths()`

```python
def infer_answer_types_from_paths(
    self, paths: List[List[Tuple[str, str, str]]]
) -> FrozenSet[str]:
    answer_types: Set[str] = set()
    for path in paths:
        terminal = path[-1][2]
        etypes = self._entity_types.get(terminal, frozenset())
        answer_types.update(etypes)
    return frozenset(answer_types)
```

Used when regex returns empty. Collects types of all terminal entities across all candidate paths.

**Time Complexity:** O(P × L) where P = number of paths, L = average path length

---

## 6. Gate Operations

### 6.1 Type Gate: `type_gate()`

```python
def type_gate(
    self, entity_name: str, answer_types: FrozenSet[str],
    hop: int, max_hop: int,
) -> bool:
    if hop < max_hop:          # intermediate hop → always allow
        return True
    if not answer_types:       # no answer types → unconstrained
        return True
    etypes = self._entity_types.get(entity_name, frozenset())
    if not etypes:             # unknown entity → conservative allow
        return True
    return bool(etypes & answer_types)  # set intersection
```

**Logic:**
- At intermediate hops: always True (we don't know the answer type yet)
- At terminal hop: True iff entity's types ∩ answer_types ≠ ∅
- Unknown entities: conservative True (don't filter what we don't know)

**Time Complexity:** O(min(|etypes|, |answer_types|)) for set intersection

### 6.2 Range Gate: `range_gate()`

```python
def range_gate(self, relation: str, tail_entity: str) -> bool:
    rel_schema = self._mined_schema.get(relation)   # check mined first
    if rel_schema is None:
        rel_schema = self._schema.get(relation)      # fall back to hand-curated
    if rel_schema is None:
        return True                                   # unknown relation → allow
    range_types = rel_schema.get("range", frozenset())
    if not range_types:
        return True
    etypes = self._entity_types.get(tail_entity, frozenset())
    if not etypes:
        return True
    return bool(etypes & range_types)
```

**Logic:**
1. Check auto-mined schema first (more specific, data-driven)
2. Fall back to hand-curated schema (authoritative, 38 relations)
3. If relation unknown: conservative True
4. If tail entity has no types: conservative True
5. Otherwise: True iff tail types ∩ range_types ≠ ∅

**Time Complexity:** O(1) dict lookups + O(min(|etypes|, |range_types|)) intersection

### 6.3 Combined Gate: `is_admissible()`

```python
def is_admissible(
    self, relation: str, tail_entity: str,
    answer_types: FrozenSet[str], hop: int, max_hop: int,
) -> bool:
    if not self.type_gate(tail_entity, answer_types, hop, max_hop):
        return False
    if not self.range_gate(relation, tail_entity):
        return False
    return True
```

Both gates must pass. This is the single entry point used during path filtering.

**Time Complexity:** O(1) + O(min(|etypes|, |range|)) = **O(1)** for typical entities

---

## 7. Path Filtering Pipeline

### 7.1 Static Filtering (v1)

In `trie_utils.py:build_filtered_trie()`:

```python
for p in all_paths:
    admit = True
    for _, rel, tail in p:           # check every hop
        if not oracle.range_gate(rel, tail):
            admit = False
            break
    if admit:
        terminal = p[-1][2]
        if not oracle.type_gate(terminal, ans_types, len(p), index_len):
            admit = False
    if admit:
        filtered.append(p)
```

**Algorithm:**
```
FOR each path p in all_paths:
  FOR each (head, rel, tail) in p:
    IF NOT range_gate(rel, tail): REJECT
  IF type_gate(terminal, ans_types, len(p), max_hop) is False: REJECT
  ACCEPT
```

**Time Complexity:** O(P × L × G) where:
- P = number of paths (typically 1,000-5,000)
- L = path length (2-4 hops)
- G = gate cost ≈ O(1)

**Total: O(P × L)** per question

### 7.2 Dynamic Filtering (v2)

In `decoding.py:_get_gated_paths()`:

```python
def _get_gated_paths(nx_graph, head_entity, oracle, answer_types, hop, max_hops):
    paths = []
    for neighbor in nx_graph.neighbors(head_entity):
        if _is_freebase_id(neighbor):     # skip opaque IDs
            continue
        rel = nx_graph[head_entity][neighbor]["relation"]
        if not oracle.range_gate(rel, neighbor):
            continue
        if hop >= max_hops and not oracle.type_gate(neighbor, answer_types, hop, max_hops):
            continue
        paths.append(f"{head_entity} -> {rel} -> {neighbor}")
    return paths
```

Called once per beam per hop. Much smaller search space (only 1-hop neighbors).

**Time Complexity:** O(D × G) where D = average degree of head entity

---

## 8. Freebase ID Detection

```python
_FB_ID_RE = re.compile(r"^[gm]\.\w+$")

def _is_freebase(name: str) -> bool:
    return bool(_FB_ID_RE.match(name))
```

Matches patterns like `m.0k8nh0b`, `g.12tb6gh4f`. These are opaque IDs that:
- Have no types recorded in the oracle
- The LLM can never generate them
- Must be filtered from trie paths

**Regex complexity:** O(N) where N = string length. Typically N < 20, so effectively O(1).

---

## 9. SIR Type Component

```python
def compute_type_irrelevance(
    paths, oracle, answer_types, max_hop
) -> float:
    n_irrelevant = 0
    for path in paths:
        terminal = path[-1]
        terminal_name = terminal if isinstance(terminal, str) else terminal.get("id", "")
        if not oracle.type_gate(terminal_name, answer_types, max_hop, max_hop):
            n_irrelevant += 1
    return n_irrelevant / len(paths)
```

**SIR*_type** = fraction of paths whose terminal entity type is **incompatible** with answer types. Higher SIR means more irrelevant paths to filter.

**Time Complexity:** O(P) where P = number of paths

---

## 10. Complete Complexity Summary

| Operation | Time Complexity | Notes |
|-----------|----------------|-------|
| `from_graph()` | O(T) | T = number of triples |
| `_mine_relation_schema()` | O(T) | Single pass over triples |
| `infer_answer_types()` | O(Q) | Q = question length, P=18 patterns |
| `infer_answer_types_from_paths()` | O(P × L) | P = paths, L = path length |
| `type_gate()` | O(1) | Set intersection, bounded by type count |
| `range_gate()` | O(1) | Dict lookup + set intersection |
| `is_admissible()` | O(1) | Two gate checks |
| Static filtering (v1) | O(P × L) | P = paths, L = path length |
| Dynamic filtering (v2) | O(D) | D = degree of head entity |

**Per-question total:** O(T + Q + P × L) where T ≈ 5,000-50,000, Q ≈ 50, P ≈ 1,000-5,000, L ≈ 2-4

---

## 11. Analysis Methodology

### How We Measured Filtering Effectiveness

1. **Path reduction ratio:** `1 - |filtered| / |all_paths|`
   - WebQSP (2-hop): 13.3% reduction
   - CWQ (4-hop): 10.4% reduction

2. **Accuracy cost:** Hits@1(filtered) - Hits@1(baseline)
   - WebQSP: -1.0pp
   - CWQ: -4.0pp

3. **SIR*_type measurement:** Fraction of paths with incompatible terminal types
   - Measured via `compute_type_irrelevance()`

### Why Filtering Is Marginal

The Freebase ontology is broad — most entities have multiple types, and most relations have broad ranges. For example:
- `people.person.nationality` has range = Location types (7 types)
- A person can have nationality → Country, City, US State, etc.
- Very few paths are actually pruned because the type constraints are loose

### Key Insight

The TypeOracle's value is not in massive path reduction, but in **guaranteeing semantic coherence** — every path in the filtered trie has type-compatible edges at every hop. This is a correctness constraint, not just a pruning heuristic.
