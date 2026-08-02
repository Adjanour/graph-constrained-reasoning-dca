# Mental Model: Graph-Constrained LLM Reasoning

A short, first-principles explanation of how knowledge graphs, LLMs, and constrained
decoding fit together — and where DCA-Trie fits in.

---

## 1. The Knowledge Graph (The Source of Truth)

A knowledge graph is a **directed labelled graph** of verified facts:

```
(Blue Hawaii) --[film.director]--> (Norman Taurog)
(Norman Taurog) --[people.person.nationality]--> (United States)
```

Each fact is a triple `(head, relation, tail)`. The triple set is the **only ground truth**
for anything knowledge-related. If a fact isn't in the KG, it's not verified.

A **multi-hop reasoning path** is a sequence of triples that chains through the graph:

```
Blue Hawaii --[film.director]--> Norman Taurog --[people.person.nationality]--> United States
```

This path answers "What is the nationality of the director of Blue Hawaii?".

**Core principle:** The KG is the sole authority. Nothing outside the triple set
is permitted in the final answer.

---

## 2. The LLM (The Reasoner)

An LLM is a next-token predictor. At each step, given all tokens so far, it
computes a probability distribution over every word in its vocabulary:

```python
P(next_token | input_question, tokens_so_far)
```

The LLM is good at:

- Understanding natural language questions
- Following multi-step instructions
- Generating fluent text

The LLM is bad at:

- **Verifying facts against a structured source**
- The generation probability is computed from learned parameters only,
  with no mechanism to "look up" whether a candidate token is factually correct

This is why LLMs hallucinate on KGQA tasks:
the model generates confident-sounding reasoning paths that don't exist in the KG.

**Core principle:** The LLM is a generator, not a verifier. It needs
an external constraint to stay grounded in the KG.

---

## 3. The Gap: Structured vs. Unstructured

The KG stores facts as exact strings/machine-readable IDs in a graph.
The LLM consumes and produces natural language tokens.

The question is: **how do you force the LLM to only say things that
are true in the KG?**

Option A (prompting): Put KG triples in the prompt and hope the LLM uses them.
This doesn't work reliably — the LLM can still hallucinate because
the decoding mechanism is unrestricted.

Option B (constrained decoding): Intercept the token selection process
and **physically prevent invalid tokens from being chosen**.

---

## 4. Constrained Decoding (The Bridge)

At each generation step, before the LLM picks a token:

1. Compute what tokens are **valid** given the KG and what's been generated so far.
2. Set the probability of all invalid tokens to **exactly zero** (logit masking).
3. Now let the LLM pick from only the valid set.

This guarantees the output is always structurally faithful to the KG.
You cannot generate an invalid triple because no invalid token ever
has a non-zero probability.

**The constraint oracle** is the component that decides which tokens
are valid at each step. It sits between the LLM's logits and the softmax:

```
LLM logits → [logit mask from oracle] → softmax → sample
```

---

## 5. The Trie (The Data Structure)

The oracle needs to answer: given a partial path, what are the valid
next tokens? This is a **prefix query**, and tries are the right
data structure.

A **KG-Trie** encodes every valid reasoning path (up to L hops) as a
set of token sequences. The trie is a tree where:

- Each node = a token ID (entity name, relation name, or formatting token)
- Each root-to-leaf path = a complete reasoning path string
- Children of a node = all valid next tokens from that position in the path

At decode time, the oracle follows the current partial path in the trie
and returns all children of the current node as the valid set.

**Limitation:** The trie only answers "what tokens extend this exact
prefix in my precomputed path set?" It does not know about KG semantics,
question intent, or reasoning state.

---

## 6. The Permissiveness Problem

The KG-Trie encodes every structurally reachable path within L hops.
For a question with three entities and average out-degree d ≈ 20,
a 3-hop question has up to 3 × 20³ = 24,000 valid paths.

But only **one** of them is the correct answer path.

This means at every decode step, the LLM chooses from thousands of
structurally valid but semantically irrelevant paths. The oracle is
too permissive — it admits every path in the graph, regardless of
whether it points in the right direction.

---

## 7. The Oracle Design Space

This is where your thesis lives. The constraint oracle can be
designed at different levels of tightness:

| Oracle | What it checks | Result |
|--------|---------------|--------|
| **GCR** | Is the path structurally valid? | Permissive (all paths admitted) |
| **Cosine DCA** | Is the path semantically similar to the question? | Less permissive, but threshold-dependent |
| **Decomposed DCA** | Is each component (relation, type, trajectory) relevant? | Better diagnostics, still has threshold |
| **Symbolic DCA** | Is the entity type compatible? Is the relation range satisfied? | Tightest, no threshold, no encoder |

The fundamental insight is:

**You can use the KG's own ontology to prune the search space,
without any learned component.**

The KG already declares entity types (`common.topic.notable_types`),
relation ranges (`rdf-schema#range`), and domain constraints.
These are free metadata that directly answer the question
"should this path be in the constraint set?"

---

## 8. The DCA-Trie Principle (Your Contribution)

```
Old oracle:  valid(t) = f(graph_structure, question_entities)
New oracle:  valid(t) = f(graph_structure, question_entities, question_text,
                          partial_generation, kg_ontology)
```

The new oracle uses **two symbolic gates** that are pure set-containment checks:

**Type gate** (terminal hop only):
  "Does the terminal entity's type match what the question asks for?"

- Question asks "who?" → only admit entities typed "Person"
- Question asks "where?" → only admit entities typed "Location"

**Range gate** (every hop):
  "Does this relation's declared range match the entity it connects to?"

- Relation `people.person.nationality` has range {"Country"}
- If a path connects it to an entity typed "Film", block it

Both gates are:

- **Deterministic**: same input always produces same output
- **Conservative**: admit by default when type info is missing
- **O(1)**: two set lookups per check, no floating point

---

## 9. "Reasoning" in LLMs vs. KG Reasoning

A common point of confusion: what does "reasoning" mean in both contexts?

### LLM "Reasoning" (Chain-of-Thought)

When an LLM "reasons", it generates intermediate tokens that decompose
a question into steps — e.g. "First find the director, then find their
nationality." This works because the LLM's training data contains millions
of examples of step-by-step explanations.

Critically, LLM reasoning is **pattern completion, not logical deduction.**
The LLM does not execute operations on a knowledge base. It generates
tokens that look like reasoning because that pattern was common in its
training text.

This is why LLM reasoning hallucinates: the pattern "first find X,
then find Y" can be fluently completed even when X or Y don't exist
in any knowledge graph.

### KG Reasoning

KG reasoning is **graph traversal**: start at the question entity,
walk to a neighbour via a relation, then to the next neighbour, and so on.
Each step is a concrete triple lookup (head, rel, tail) against the KG.

Multi-hop means the traversal goes through multiple edges:

```
hop 1:  Blue Hawaii ----film.director----> Norman Taurog
hop 2:  Norman Taurog ----nationality----> United States
```

The path *is* the reasoning chain. Each relation choice is a reasoning step.

### How They Combine in This System

The LLM's "reasoning" ability is used to **choose which relation to take
at each hop**. The LLM sees the question and the partial path so far,
then picks the next relation from the valid set provided by the trie.

The trie provides *structural validity* (only graph-existing triples).
The LLM provides *semantic selection* (which of the valid relations
is relevant to the question).

| Component | Role | Example |
|-----------|------|---------|
| **BFS** | Find all possible next entities | `film.director`, `film.starring`, `film.location`, ... |
| **Trie** | Encode valid paths for fast lookup | All 24,000 paths from question entities |
| **Oracle** | Filter to relevant paths | Only paths whose entities are the right type |
| **LLM** | Choose which valid path answers the question | "director" is the correct relation |

The innovation of constrained decoding is:
**The LLM chooses, but it can only choose from graph-valid options.**
It cannot hallucinate a non-existent triple because no invalid triple
ever reaches the softmax.

### Multi-hop Reasoning in Detail

For the question "What is the nationality of the director of Blue Hawaii?":

```
Step 1: LLM generates "Blue Hawaii" (question entity, always valid)
Step 2: LLM chooses relation from valid set →
        {film.director, film.starring, film.location, ...}
        LLM picks "film.director" (this is where reasoning happens)
Step 3: LLM generates entity "Norman Taurog" (only valid tail for that relation)
Step 4: LLM chooses next relation from now-expanded valid set →
        {people.person.nationality, people.person.spouse_s,
         people.person.date_of_birth, ...}
        LLM picks "people.person.nationality" (reasoning step 2)
Step 5: LLM generates "United States" (the answer)
```

At step 4, without an oracle, the LLM sees *every possible relation*
from Norman Taurog. With DCA-Trie's type gate, only relations whose
range includes "Country" are admitted. The LLM still makes the final
choice, but the choice set is narrowed by the KG's own schema.

**The reasoning is distributed:**

- The **graph** provides the possibilities (via BFS)
- The **oracle** narrows them (via ontology constraints)
- The **LLM** selects among them (via parametric knowledge)

---

## 10. The Complete Picture

```
┌─────────────────────────────────────────────────────────┐
│                     THE PIPELINE                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Question: "Nationality of director of Blue Hawaii?" │
│         ↓                                               │
│  2. Entity linking → question_entities = [Blue Hawaii]  │
│         ↓                                               │
│  3. BFS from entities → all paths up to L hops          │
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

The pipeline is modular. The constraint oracle is the only component
that changes between GCR, DCA-Trie v1, and DCA-Trie v2.
Everything else — entity linking, BFS, beam search, LLM — stays the same.

This isolation means: **any improvement in accuracy or efficiency
is attributable to the oracle design, not to model changes.**

---

## Common Confusions

### "Isn't the trie just a lookup table?"

Not exactly. A lookup table maps keys to values. A trie maps **prefixes to valid continuations**. The key difference is that a trie can efficiently answer "what are all valid next tokens given this partial sequence?" — which is exactly what you need at each decoding step.

A lookup table would need to store every complete path. A trie stores the structure, so shared prefixes are only stored once. For 24,000 paths with average length 20, that's 480,000 entries in a table vs ~50,000 nodes in a trie.

### "Why not just use the shortest path?"

Because the shortest path isn't always the right path. The question "What is the nationality of the director of Blue Hawaii?" requires a 2-hop path. But there might be multiple 2-hop paths, and the shortest path might not be the one that answers the question.

Also, the system generates **multiple paths** with beam search. The LLM picks the best one from the valid set. Using only the shortest path would remove the LLM's ability to choose.

### "Does the oracle guarantee the answer is correct?"

No. The oracle guarantees the answer is **structurally valid** — it exists in the KG. But the LLM still needs to choose the *right* path from among all valid paths.

The oracle reduces the search space from 24,000 paths to ~20,000 paths. The LLM then picks the best one from the reduced set. The guarantee is: whatever it picks, it's a real KG path. Not that it's the correct answer.

### "Why does the cosine similarity approach fail?"

The cosine approach uses `cos(E(path), E(question)) >= τ` to decide if a path is relevant. Two problems:

1. **Threshold collapse**: At τ=0.25, 84% of questions produced empty tries — all paths were rejected. The threshold was too aggressive. But lowering it reduces filtering power. There's no good middle ground.

2. **Wrong signal**: Cosine similarity measures text similarity, not semantic relevance. A path like "Blue Hawaii → film.film.genre → Comedy" is textually similar to the question but doesn't answer it.

The symbolic approach avoids both problems: no threshold to tune, and the signal (entity types) directly answers "is this path heading in the right direction?"

### "What's the difference between v1 and v2?"

**v1 (Static)**: Build the trie once from all filtered paths, then run constrained decoding. Simple, fast, one-shot.

**v2 (Dynamic)**: At each decoding step, expand the trie with new valid paths based on what's been generated so far. More complex, potentially more accurate, but slower.

v1 is the primary approach used in experiments. v2 is experimental.

### "Why group-beam search instead of standard beam search?"

Standard beam search can suffer from **beam collapse** — all k beams converge to the same high-probability sequence. For KGQA, this means all 10 beams might generate the same path, wasting compute.

Group-beam search divides beams into groups and adds a **diversity penalty** — beams in different groups are penalized for being similar. This forces exploration of different paths, which is critical when multiple valid paths exist.

---

## Design Rationale

### Why two oracle gates instead of one?

The two gates catch **different failure modes**:

**Type gate** (terminal hop): "Does the final entity have the right type for what the question asks?"
- Catches: paths that wander to random entities of the wrong type
- Example: "Who directed Blue Hawaii?" → path ends at a Film entity → blocked
- Does ~73% of the pruning work (10.6% vs 3.8%)

**Range gate** (every hop): "Does this relation's declared range match the entity it connects to?"
- Catches: nonsensical intermediate steps where a relation is applied to the wrong type
- Example: `film.film.country → Elvis_Presley` → blocked (Elvis is a Person, not a Country)
- Does ~27% of the pruning work

They're complementary. A path can pass the range gate (all intermediate steps make sense) but fail the type gate (ends at the wrong type). Or vice versa.

### Why only check types at the terminal hop?

At intermediate hops, the entity is a **waypoint**, not the answer. You're passing through it to get somewhere else. Its type doesn't matter for the final answer.

Example: "What is the nationality of the director of Blue Hawaii?"
- Hop 1: Blue Hawaii → directed_by → Norman Taurog (Person) — OK, he's a waypoint
- Hop 2: Norman Taurog → nationality → United States (Country) — this is the answer

At hop 1, we don't care that Norman Taurog is a Person. We care that at hop 2, the answer is a Country (matching "nationality").

If we checked types at every hop, we'd incorrectly block valid paths where intermediate entities happen to have "wrong" types.

### Why conservative fallback (admit when info is missing)?

Consider the alternative: **reject when info is missing**.

- If an entity has no declared types → reject it
- If a relation has no declared range → reject paths using it

This sounds safer. But in practice:
- Many entities in Freebase have incomplete type information
- Many relations have no explicit range declaration
- Rejecting on missing info would **kill recall** — you'd lose correct answers because the KG metadata is incomplete

The conservative approach (admit when info is missing) bounds the false negative rate at ~3%. The aggressive approach (reject on missing info) would have FNR > 20%.

**The design tradeoff:** We accept a few extra irrelevant paths (lower precision) to ensure we never lose the correct path (high recall). This is the right tradeoff because the LLM can handle some noise in the path set, but it can't recover from a missing correct path.

### Why not learn the oracle?

You could train a model to predict whether each path is relevant. But:

1. **You need training data** — gold-standard labels for "is this path relevant?" are expensive to annotate
2. **You need a GPU** — inference on the encoder adds latency and cost
3. **You need threshold tuning** — the model outputs a score, you need to decide where to cut
4. **It's non-deterministic** — float noise means slightly different results each run
5. **It doesn't generalize** — a model trained on WebQSP might not work on CWQ

The symbolic oracle:
1. Uses metadata already in the KG — no annotation needed
2. Runs on CPU — O(1) set lookups
3. Has no threshold — binary decision
4. Is deterministic — same input, same output
5. Generalizes — the KG schema is universal

---

## Deep-Dive Questions

### "Walk me through what happens for one question."

**Question:** "What is the nationality of the director of Blue Hawaii?"

**Step 1: Entity Linking**
The system identifies "Blue Hawaii" as a known Freebase entity.

**Step 2: DFS Path Enumeration**
Starting from Blue Hawaii, enumerate all paths up to 2 hops:
```
Blue Hawaii → film.film.directed_by → Norman Taurog
Blue Hawaii → film.film.starring → Elvis Presley
Blue Hawaii → film.film.country → United States
Blue Hawaii → film.film.language → English
Norman Taurog → people.person.nationality → United States
Norman Taurog → people.person.place_of_birth → New York City
Elvis Presley → people.person.nationality → United States
... (hundreds more)
```

**Step 3: TypeOracle Filtering**
- Question word "nationality" → answer types = {Person, Location, Country}
- **Range gate**: Check each intermediate hop
  - `film.film.directed_by → Norman Taurog`: range is {Person}, Norman is Person ✓
  - `film.film.country → United States`: range is {Country}, US is Country ✓
  - `film.film.starring → Elvis Presley`: range is {Person}, Elvis is Person ✓
- **Type gate**: Check terminal entity type
  - Terminal entity = United States, types = {Country}
  - Country ∩ {Person, Location, Country} ≠ ∅ ✓ → admitted
  - Terminal entity = Elvis Presley, types = {Person}
  - Person ∩ {Person, Location, Country} ≠ ∅ ✓ → admitted
  - Terminal entity = English, types = {Human Language}
  - Human Language ∩ {Person, Location, Country} = ∅ ✗ → **blocked**

**Step 4: Trie Construction**
Build a MarisaTrie from the filtered paths. Each path becomes a token sequence:
```
<PATH> Blue_Hawaii -> film.film.directed_by -> Norman_Taurog ->
       people.person.nationality -> United_States </PATH> <eos>
```

**Step 5: Constrained Decoding**
The LLM generates freely until it emits `<PATH>`. Then:
```
Step 1: LLM sees question, starts generating
Step 2: LLM emits <PATH> → constrained mode activates
Step 3: trie.get("<PATH>") → returns ["Blue_Hawaii"]
Step 4: LLM picks "Blue_Hawaii" (only option)
Step 5: trie.get("<PATH> Blue_Hawaii") → returns ["film.film.directed_by", "film.film.starring", ...]
Step 6: LLM picks "film.film.directed_by" (semantic choice)
Step 7: trie.get("... directed_by") → returns ["Norman_Taurog"]
Step 8: LLM picks "Norman_Taurog"
Step 9: trie.get("... Norman_Taurog") → returns ["people.person.nationality", ...]
Step 10: LLM picks "people.person.nationality"
Step 11: trie.get("... nationality") → returns ["United_States"]
Step 12: LLM picks "United_States"
Step 13: LLM emits </PATH> → free generation resumes
Step 14: LLM writes answer: "United States"
```

**Step 6: Answer Extraction**
Parse the output, extract "United States" as the predicted answer.

### "Why doesn't the oracle just pick the right path directly?"

Because the oracle doesn't "know" which path answers the question. It only knows which paths are **heading in the right direction** (right entity types, right relation ranges). The LLM is the one that knows which relation is semantically relevant to the question.

The oracle is a **structural filter**, not a semantic reasoner. It removes paths that are *structurally impossible* (wrong types, wrong ranges) but keeps paths that are *structurally possible but semantically irrelevant*. The LLM handles the semantic selection.

This is a deliberate division of labor:
- **Oracle**: Handles what it's good at (structural validity) — fast, deterministic, no GPU
- **LLM**: Handles what it's good at (semantic understanding) — slow, expensive, but necessary

### "What happens if the oracle is wrong?"

The oracle is designed to be **conservative** — it errs on the side of keeping paths, not rejecting them.

If the oracle rejects a path that should have been kept (false negative):
- The LLM can't choose that path (it's not in the trie)
- The correct answer might be lost
- This is measured by the **FNR** (False Negative Rate)

Empirically, FNR is ~3% — the oracle incorrectly blocks about 3 out of every 100 gold paths. This is the cost of the conservative design. An aggressive design would have higher FNR but more pruning.

If the oracle keeps a path that should have been rejected (false positive):
- The LLM has one more irrelevant path to choose from
- This adds noise but doesn't lose the correct answer
- This is measured by **1 - SIR** (the fraction of paths that pass through)

The design tradeoff: minimize FNR (don't lose correct paths) at the cost of some extra false positives (keep some irrelevant paths). The LLM can handle noise; it can't handle missing information.

### "Could this work without the LLM?"

Yes, partially. If you know the question entities and the answer type, you could:
1. DFS from question entities
2. Apply the type oracle to filter paths
3. Return all terminal entities that match the answer type

This is essentially what `direct_answer()` does in `predict_final_answer.py` when no model is available. It works for simple questions but fails for complex ones that require semantic understanding of which relation to follow.

The LLM adds the semantic reasoning capability — it can understand that "nationality" implies following the `people.person.nationality` relation, not `people.person.profession` or `people.person.gender`.

### "What's the relationship between SIR and decoding speed?"

Smaller trie → fewer valid tokens at each step → faster `prefix_allowed_tokens_fn` evaluation.

With group-beam search (k=10), each decoding step expands 10 beams. Each beam calls `trie.get()`. A 14.5% reduction in trie size means:
- 14.5% fewer tokens to consider per beam
- 14.5% less memory for the trie structure
- Measurable speedup in decoding

But the primary benefit isn't speed — it's **accuracy**. Fewer irrelevant paths means the LLM has an easier semantic selection problem.
