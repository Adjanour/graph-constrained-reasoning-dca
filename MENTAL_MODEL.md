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
