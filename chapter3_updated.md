# CHAPTER 3

# METHODOLOGY

## 3.1 Introduction

This chapter formalises the permissiveness problem identified in Chapter 2 and presents the revised DCA-Trie framework as a solution. The chapter begins by restating the core limitation of existing frameworks with greater precision, then introduces the Upgraded Oracle Specification that conditions constraint generation on both the input question and the KG ontology. The Semantic Irrelevance Ratio (SIR) is maintained as a decomposed diagnostic metric that distinguishes between two independent failure modes: type blindness and trajectory blindness. A purely symbolic constraint oracle — TypeOracle — is then introduced, replacing the embedding-based scoring mechanism from the earlier design. Both variants of DCA-Trie are then specified algorithmically using only symbolic gates, followed by a formal faithfulness guarantee and per-gate false negative rate analysis. The chapter closes with the baseline configurations, evaluation protocol, and scope boundaries governing the experiments in Chapter 4.

The implementation builds directly on the Graph-Constrained Reasoning (GCR) codebase (Luo et al., 2025a). GCR constructs a KG-Trie by running BFS from question entities, serialising retrieved paths into formatted strings using a fixed template, tokenising those strings with the LLM tokeniser, and storing the resulting token sequences in a prefix tree. At each beam-search step, the constraint function $\mathcal{C}_\mathcal{G}(w_{z_i} \mid w_{z_{1:i-1}})$ queries the trie at the current prefix and returns the set of valid next token IDs, which are applied as a logit mask before softmax. DCA-Trie hooks into this pipeline at two distinct points: before `BuildTrie` in v1, and inside the trie expansion step in v2. Both interventions operate at the path level on entity and relation strings, before tokenisation, which means the symbolic gates never touch raw token IDs and the GCR decoding loop requires no structural changes.

---

## 3.2 Formal Problem Specification

### 3.2.1 Restating the Core Limitation of Existing Frameworks

As established in Chapter 2, current graph-constrained frameworks, notably GCR (Luo et al., 2025a) and DoG (Li et al., 2025), define the valid-token set using only graph structure and question entities:

$$W^{GCR}_{val}(t) = f(\mathcal{G}, E_q) \quad \forall t \tag{3.1}$$

Equation (3.1) captures a static constraint model that ignores both the semantic intent of question $q$ and the ontological structure of the KG. This causes two distinct failures that prior work has not separated.

The first is **type blindness**. The oracle admits every path whose terminal entity is structurally reachable, regardless of whether that entity type is compatible with what the question is asking for. A question asking for a nationality will receive paths ending at films, dates, and people alongside paths ending at countries. All are structurally valid. None of the irrelevant ones should be in the constraint set.

The second is **trajectory blindness**. The oracle does not narrow as reasoning progresses. After the model commits to `Blue_Hawaii directed_by Norman_Taurog`, the paths relevant to the next step shift entirely toward nationality-type relations. The static oracle does not reflect this. It continues to offer every path reachable from the question entities, as if no reasoning had occurred.

These two failures are independent and compound at deeper hops. The path count at hop depth $L$ grows as $|E_q| \times d^L$, where $d$ is the average entity out-degree. For well-connected Freebase entities with $d \approx 20$ and three question entities, a three-hop question admits up to 24,000 structurally valid paths. Type blindness and trajectory blindness together ensure that nearly all of them remain in the constraint set throughout generation.

### 3.2.2 The Upgraded Oracle Specification

DCA-Trie addresses both failures by defining an upgraded constraint oracle that exploits the KG's ontological schema:

$$W^{DCA}_{val}(t) = f(\mathcal{G}, \mathcal{O}, E_q, q) \tag{3.2}$$

where $\mathcal{O}$ is the KG ontology, encoding type hierarchies, property domain and range constraints, and relation composition patterns.

To be methodologically sound, the upgraded oracle must satisfy three conditions, consistent with the project objectives in Chapter 1.

**Condition F (Structural Faithfulness):** Every candidate token $v \in W^{DCA}_{val}(t)$ must correspond to a valid prefix of a path $p \in \mathcal{G}$ at every step $t$. No token outside the verified KG triplet set may receive a non-zero probability.

**Condition R (Semantic Relevance Reduction):** The Semantic Irrelevance Ratio under DCA-Trie must be lower than under GCR when averaged over the evaluation corpus:

$$\text{SIR}^*_{W^{DCA}} < \text{SIR}^*_{W^{GCR}} \tag{3.3}$$

**Condition P (Recall Preservation):** Symbolic pruning must not remove gold paths excessively. The false negative rate on the WebQSP validation split must satisfy:

$$\text{FNR} = \frac{|\{q \in \mathcal{Q}_{val} : p^*_q \notin W^{DCA}_{val}(t)\}|}{|\mathcal{Q}_{val}|} < 0.05 \tag{3.4}$$

where $p^*_q$ is the gold path for question $q$ and $\mathcal{Q}_{val}$ is the held-out 100-question validation set.

---

## 3.3 System Architecture Overview

DCA-Trie is a constraint oracle layer that slots directly into the four-layer GCR pipeline. Three of the four layers are unmodified. Only the constraint oracle layer changes.

**Layer 1: Entity Linking (Unmodified).** A named entity recognition tool identifies the core query entities $E_q$ from the raw text question $q$. These entities are the starting points for BFS graph traversal.

**Layer 2: Constraint Oracle Layer (DCA-Trie Contribution).** This layer determines which tokens the LLM can generate. In GCR, it produces a static prefix tree encoding every structurally reachable path within $L$ hops, without any semantic consideration. DCA-Trie replaces this with a two-stage admission process. Every candidate path must pass two symbolic gates — an answer type gate and a property range gate — before entering the trie. Both gates use only the KG's own ontological schema, represented as a set of set-containment checks that are computable in O(1) time. In v1, this filtering is done once at construction time. In v2, it is repeated step-by-step as generation commits to new entities.

**Layer 3: Constrained Decoding (Unmodified).** During beam-search decoding, the LLM logit distribution is intercepted at each step. A binary mask, derived from the oracle layer, forces invalid tokens to probability zero. This maintains 100% structural faithfulness to the knowledge graph.

**Layer 4: Inductive Reasoning (Unmodified).** After constrained beam search produces $K$ top reasoning paths, GPT-4o-mini synthesises the final natural language answer from those paths.

The architecture is shown in Figure 3.1. The isolation of the constraint oracle layer is intentional: all experimental improvements are attributable to changes in constraint quality, not to model adaptation or preprocessing variation.

---

## 3.4 The Semantic Irrelevance Ratio: Revised Definition

### 3.4.1 Standard Metric Deficiencies

The original SIR formulation in this thesis measured constraint permissiveness using a single cosine similarity between a path string embedding and a concatenated question-context embedding. This operationalisation conflates three distinct sources of irrelevance and provides no diagnostic information about which source is driving high permissiveness for a given question or hop depth.

Specifically, the single-score approach cannot distinguish between: a path that is irrelevant because its terminal entity is the wrong type; a path that is irrelevant because its relation chain points in the wrong direction relative to the question; and a path that was relevant at step $t=1$ but becomes irrelevant at step $t=3$ because the reasoning trajectory has already resolved that part of the question's intent. Each of these is a different failure and should be measured separately.

### 3.4.2 Decomposed SIR Definition

A path $p \in \mathcal{P}(q,t)$ is **type-irrelevant** if its terminal entity type is incompatible with the answer type required by the question:

$$\text{irrel}_{\text{type}}(p, q) = \mathbf{1}[\text{type}(e_h) \notin \mathcal{T}(q, h)] \tag{3.5}$$

where $\mathcal{T}(q, h)$ is the set of compatible entity types at hop $h$ inferred from the question's semantic structure, and $e_h$ is the terminal entity of path $p$.

A path is **trajectory-irrelevant** if its relation chain is incompatible with the question's relational intent at any hop:

$$\text{irrel}_{\text{traj}}(p, q) = \mathbf{1}[\exists i \in \{1,\dots,h\} : \text{range}(r_i) \cap \mathcal{T}(q, i) = \emptyset] \tag{3.6}$$

where $\text{range}(r_i)$ is the set of entity types that relation $r_i$ can accept as its object, as declared in the KG ontology. A path is trajectory-irrelevant if any of its hops uses a relation whose declared range is incompatible with the question's required type at that hop.

The overall irrelevance indicator is the disjunction of both:

$$\text{irrel}^*(p, q) = \text{irrel}_{\text{type}}(p, q) \vee \text{irrel}_{\text{traj}}(p, q) \tag{3.7}$$

### 3.4.3 Decomposed SIR Metrics

The three step-level SIR measures are:

$$\text{SIR}^*_{\text{type}}(q,t) = \frac{|\{p \in \mathcal{P}(q,t) : \text{irrel}_{\text{type}}(p,q) = 1\}|}{|\mathcal{P}(q,t)|} \tag{3.8}$$

$$\text{SIR}^*_{\text{traj}}(q,t) = \frac{|\{p \in \mathcal{P}(q,t) : \text{irrel}_{\text{traj}}(p,q) = 1\}|}{|\mathcal{P}(q,t)|} \tag{3.9}$$

$$\text{SIR}^*(q,t) = \frac{|\{p \in \mathcal{P}(q,t) : \text{irrel}^*(p,q) = 1\}|}{|\mathcal{P}(q,t)|} \tag{3.10}$$

Corpus-level SIR aggregates over all questions and hop depths:

$$\text{SIR}^* = \frac{1}{|\mathcal{Q}|} \sum_{q \in \mathcal{Q}} \frac{1}{L_q} \sum_{t=1}^{L_q} \text{SIR}^*(q,t) \tag{3.11}$$

### 3.4.4 Properties and Interpretation

$\text{SIR}^*_{\text{type}}$ isolates the contribution of type blindness to overall permissiveness. It is computable without any embedding operations, using only Freebase schema information. $\text{SIR}^*_{\text{traj}}$ isolates trajectory blindness by measuring how many paths use a relation at any hop whose declared range is incompatible with the question's required entity type at that hop. Since $\text{irrel}_{\text{traj}}$ is defined per path rather than per step, it does not depend on the generation prefix; it reflects a structural incompatibility that exists regardless of generation state. $\text{SIR}^*$ is the combined metric reported for comparison with GCR baselines. Reporting all three separately in Chapter 4 allows the contribution of each filtering mechanism to be attributed independently, which is a direct response to Gap 3 identified in Chapter 2.

---

## 3.5 TypeOracle: Purely Symbolic Constraint Gates

### 3.5.1 Why Symbolic Constraints Are Stronger Than Embedding-Based Scoring

The earlier design scored candidate paths using cosine similarity between sentence-transformer embeddings of path strings and question strings. This approach has three structural weaknesses that a symbolic oracle avoids.

The first is **representation weakness**. A 384-dimensional embedding from a general-purpose sentence encoder (all-MiniLM-L6-v2) is a poor proxy for the fine-grained relational semantics needed to distinguish between, for example, "directed_by" and "starring" in a film context. The sentence encoder is not trained on Freebase relations and has no access to the KG's formal type system.

The second is **computational cost**. Each candidate path requires an embedding computation. While encoder caching mitigates redundant work, the per-path cost scales linearly with the candidate set size. At hop depth 3 with 20,000 candidates, this is 20,000 forward passes through an encoder, on top of the LLM decoding cost.

The third is **the threshold problem**. The decomposed product score ρ_r · ρ_e · ρ_traj requires a calibrated threshold τ to determine admission. This threshold is dataset-dependent, hop-depth-dependent, and sensitive to the choice of encoder. It must be tuned on a held-out validation set, and there is no guarantee that a single τ works across both WebQSP (1-2 hop) and CWQ (up to 4 hop).

A symbolic oracle using the KG's own ontological schema avoids all three problems. Type membership checks are O(1) set lookups against the KG's ground-truth type declarations. Property range checks are O(1) lookups against RDFS domain and range axioms encoded in the schema. No embeddings, no encoder calls, no thresholds.

### 3.5.2 Ontological Information Available in the Freebase Subgraph

Every question in the WebQSP and CWQ datasets includes a subgraph of Freebase centred on the question entities. This subgraph contains, in addition to domain relation triples, a rich set of ontological triples:

- **Entity type triples.** The relation `common.topic.notable_types` maps an entity to its human-readable type string. For example, `Jamaica --[common.topic.notable_types]--> Country`. A typical 2-hop subgraph contains hundreds of such triples, covering most entities in the subgraph.
- **Property schema triples.** The relations `rdf-schema#domain` and `rdf-schema#range` encode the expected subject and object types for each property. For example, `Place of birth --[rdf-schema#domain]--> Person` and `Place of birth --[rdf-schema#range]--> Location`.
- **Property type constraints.** The relation `type.property.expected_type` maps a property display name to its expected value type. For example, `Capital --[type.property.expected_type]--> City/Town/Village`.
- **Property-to-type membership.** The relation `type.type.properties` lists all properties defined on a given type. For example, `US President --[type.type.properties]--> Vice president`.
- **Inverse relations.** The relations `type.property.reverse_property` and `owl#inverseOf` encode bidirectional relation pairs.

This information is present in every question's subgraph because Freebase's schema triples are themselves encoded as RDF triples and are included in the BFS neighbourhood of question entities. No external schema dump is required.

### 3.5.3 Extracting the Type Map from the Subgraph

At the start of each question, the TypeOracle extracts entity type information from the subgraph in a single pass. For each triple (head, relation, tail):

- If `relation == "common.topic.notable_types"`, record `head → tail` as a type assignment.
- If `relation == "freebase.type_hints.included_types"` and `tail != "Topic"`, record a secondary type assignment.
- If `relation == "freebase.type_profile.strict_included_types"` and `tail != "Topic"`, record a strict type assignment.

The result is a map from entity strings (e.g., "Jamaica", "James K. Polk") to sets of human-readable type strings (e.g., {"Country"}, {"US President", "Person"}). Entities not found in this map have unknown types and are treated conservatively (admitted by default).

### 3.5.4 Answer Type Inference from Question Structure

The expected answer type for a question is inferred using pattern matching on the question string. Each pattern maps a question word or phrase to a set of compatible Freebase type strings:

- `"who"` → {"Person", "Deceased Person", "Politician", ...}
- `"where"`, `"location"`, `"country"`, `"city"` → {"Location", "Country", "City/Town/Village", ...}
- `"when"`, `"date"`, `"year"` → {"Date/Time"}
- `"film"`, `"movie"` → {"Film", "TV Program", ...}
- `"language"`, `"speak"` → {"Human Language"}
- `"profession"`, `"job"`, `"occupation"` → {"Profession"}
- `"award"`, `"prize"` → {"Award", "Award honor"}
- `"organization"`, `"company"` → {"Organization", "Company", ...}

The full mapping covers 18 question word categories. If no pattern matches, the answer type set is empty and the oracle applies no type constraint (all paths admitted by the type gate). This is the safe fallback — the oracle is conservative by default and only constrains when the question provides a clear type signal.

### 3.5.5 Gate 1: Answer Type Gate (Terminal Hop)

The answer type gate is the primary pruning mechanism. It operates on the terminal entity of each candidate path at the final hop. For a path $p = (e_0, r_1, e_1, ..., r_h, e_h)$ at hop depth $h$, the gate checks:

$$\text{type\_gate}(e_h, q, h, L) = \begin{cases}
\text{True} & \text{if } h < L \text{ (intermediate hop)} \\
\text{True} & \text{if } \mathcal{T}(q) = \emptyset \text{ (no type inferred)} \\
\text{True} & \text{if } \text{types}(e_h) = \emptyset \text{ (entity type unknown)} \\
\mathbf{1}[\text{types}(e_h) \cap \mathcal{T}(q) \neq \emptyset] & \text{otherwise}
\end{cases} \tag{3.12}$$

where $\mathcal{T}(q)$ is the set of compatible answer types inferred from the question and $\text{types}(e_h)$ is the set of Freebase types recorded for entity $e_h$ in the subgraph.

This gate is conservative by design. If the question says "who" but the entity has no recorded types, the gate admits it. Only when both the question provides a type signal and the entity has known types does the gate block.

### 3.5.6 Gate 2: Property Range Gate (All Hops)

The property range gate checks that each relation's declared object type is compatible with the entity it connects to. For a triple $(e_i, r_{i+1}, e_{i+1})$ at hop $i+1$:

$$\text{range\_gate}(r_{i+1}, e_{i+1}) = \begin{cases}
\text{True} & \text{if } r_{i+1} \notin \mathcal{R} \text{ (relation schema unknown)} \\
\text{True} & \text{if } \text{range}(r_{i+1}) = \emptyset \text{ (no range declared)} \\
\text{True} & \text{if } \text{types}(e_{i+1}) = \emptyset \text{ (entity type unknown)} \\
\mathbf{1}[\text{types}(e_{i+1}) \cap \text{range}(r_{i+1}) \neq \emptyset] & \text{otherwise}
\end{cases} \tag{3.13}$$

where $\mathcal{R}$ is the set of relations with known schema in the oracle's relation schema map.

For example, the relation `people.person.place_of_birth` has its range declared as {"Location", "Country", "City/Town/Village", ...}. If a candidate triple attempts to connect this relation to an entity of type "Film", the range gate blocks it.

### 3.5.7 Formal Properties of the Symbolic Oracle

The TypeOracle has three formal properties that distinguish it from embedding-based alternatives.

**Monotonicity.** The symbolic gates define a subset of GCR's full path set. Every path admitted by the symbolic oracle is also present in GCR's unfiltered BFS enumeration:

$$P_{\text{DCA}} \subseteq P_{\text{GCR}} \subseteq \mathcal{T}^L \tag{3.14}$$

This holds because the gates remove paths but never add paths not in the GCR set. It guarantees Condition F by construction.

**Determinism.** The gates are deterministic: the same question and subgraph always produce the same filtered path set. There is no randomness from embedding noise, floating-point non-determinism, or threshold sensitivity.

**Zero encoding cost.** The gates require no forward passes, no embedding computations, and no GPU time. They operate entirely on the string-level KG data already present in the dataset.

---

## 3.6 DCA-Trie v1: Static Symbolic Filtering

### 3.6.1 Design Rationale

DCA-Trie v1 applies symbolic filtering once at trie construction time, before generation begins. It enumerates all BFS paths from question entities (identical to GCR's enumeration), then filters them through the two symbolic gates. Paths that pass both gates enter the trie; paths that fail either gate are discarded. The trie is then used for the full generation, unchanged, just as in GCR.

This design is the simplest possible instantiation of the upgraded oracle. It trades maximum pruning (which v2 achieves) for zero overhead during generation. The filtering cost is paid once, offline, before decoding begins.

### 3.6.2 Algorithm for DCA-Trie v1

```
Algorithm 1: DCA-Trie v1 — Static Symbolic Filtering

Require: G, E_q, q, L, oracle TypeOracle

1:  Build the entity type map from G:
      oracle ← TypeOracle.from_graph(G)

2:  Infer answer type constraint:
      T_term ← oracle.infer_answer_types(q)

3:  Enumerate candidate paths:
      P ← BFS(G, E_q, L)

4:  Filter by symbolic gates:
    P_filtered ← ∅
    for each path p = (e_0, r_1, e_1, ..., r_h, e_h) ∈ P do

      // Gate 2 (range) — must pass at every hop
      admit ← True
      for i = 1 to h do
        if not oracle.range_gate(r_i, e_i) then
          admit ← False
          break
        end if
      end for

      // Gate 1 (type) — terminal hop only
      if admit and not oracle.type_gate(e_h, T_term, h, L) then
        admit ← False
      end if

      if admit then
        P_filtered ← P_filtered ∪ {p}
      end if

    end for

5:  T_v1 ← BuildTrie(P_filtered)
    return T_v1
```

The range gate is applied first, before the type gate, because the range gate can fail at an intermediate hop (pruning the path early without needing to reach the terminal entity). The type gate is only evaluated at the terminal hop on paths that have already passed all intermediate range checks.

### 3.6.3 Complexity Analysis

The offline construction cost is O(|P|) for the gate checks, where |P| is the number of BFS-enumerated paths. Each gate check is O(1): two set-lookups for the range gate (relation schema lookup, entity type lookup) and two set-lookups for the type gate (entity type lookup, intersection with answer types).

The critical property is that cost is independent of embedding dimension. There are no encoder calls, no matrix multiplications, no GPU operations. The entire filtering step for a 20,000-path candidate set completes in milliseconds on CPU.

At decoding time, the trie lookup cost is identical to GCR: O(d) for constant-time child lookups in the MarisaTrie. Memory usage scales as O(|P_filtered| · L), which is smaller than GCR's O(|P| · L) by the filtering ratio.

### 3.6.4 Faithfulness Guarantee

DCA-Trie v1 preserves Condition F by construction. Every path in P_filtered originates from BFS(G, E_q, L), which contains only paths whose constituent triplets are verified members of T. The symbolic gates are monotone subset operations: they remove paths from P but never introduce paths not in P. Therefore:

$$P_{\text{filtered}} \subseteq P \subseteq \mathcal{T}^L \tag{3.15}$$

Any token admitted by the trie built from P_filtered corresponds to a valid continuation of a path in P_filtered, which is a path in T. Structural validity flows downward through the subset chain without leakage.

---

## 3.7 DCA-Trie v2: Iterative Symbolic Expansion

### 3.7.1 Design Rationale

DCA-Trie v2 implements the full dynamic oracle by expanding the constraint trie step-wise as generation progresses, applying the symbolic gates at each expansion step. After the model generates an entity token e_t, the trie is expanded with all neighbours of e_t that pass both gates. This means the constraint set at hop h reflects the actual entity committed at hop h-1, not a static precomputation over all possible entities at hop h-1.

Architecturally, v2 follows the step-wise traversal pattern of DoG (Li et al., 2025) but adds symbolic gating at each expansion. The expansion rule is:

$$\mathcal{T}_{t+1} = \mathcal{T}_t \cup \left\{(e_t, r, e') : (e_t, r, e') \in \mathcal{T}_\mathcal{G} \;\wedge\; \text{gates}(e_t, r, e', q) \right\} \tag{3.16}$$

The core difference from DoG's permissive expansion is the gates condition: every edge must be both graph-valid and symbolically admissible.

### 3.7.2 Algorithm for DCA-Trie v2

```
Algorithm 2: DCA-Trie v2 — Iterative Symbolic Expansion

Require: G, E_q, q, L, oracle TypeOracle, tokenizer

1:  Build the entity type map from G:
      oracle ← TypeOracle.from_graph(G)

2:  Infer answer type constraint:
      T_term ← oracle.infer_answer_types(q)

3:  Initialize trie with question entities:
      T_0 ← Tokenize(E_q)

4:  h ← 0
5:  for each decoding step t do

      // Standard constrained generation
      V_t ← TrieLookup(T_{t-1}, y_{<t})
      y_t ← BeamSearchStep(V_t ⊙ LLMLogits(y_{<t}, x))

      if Commit(e_t) then

        h ← h + 1
        is_terminal ← (h = L)

        // Expand trie with gated neighbours
        for each (e_t, r, e') ∈ G do

          // Range gate — must pass
          if not oracle.range_gate(r, e') then
            continue
          end if

          // Type gate — terminal hop only
          if is_terminal and not oracle.type_gate(e', T_term, h, L) then
            continue
          end if

          // Admitted: add to trie
          T_t ← T_t ∪ {Tokenize(e_t, r, e')}

        end for

      end if

    end for

6:  return y_1 ... y_T
```

The key structural difference from v1 is that v2 never enumerates all paths upfront. At each step, it retrieves only the immediate neighbours of the committed entity and gates each one individually. This avoids the O(d^L) enumeration cost that v1 pays at construction time for deep paths that may never be reached.

### 3.7.3 Complexity Analysis

At each entity commit, v2 examines O(d) neighbours, where d is the out-degree of the committed entity. Each neighbour requires exactly two O(1) gate checks: one range check and (at terminal hop) one type check. No path enumeration, no scoring, no encoder calls.

The total cost over generation is O(L · d_avg), where d_avg is the average entity out-degree. For Freebase entities with d_avg ≈ 20 and L = 2, this is roughly 40 O(1) lookups per question — three orders of magnitude cheaper than v1's offline path enumeration.

The trie is built incrementally. Memory usage at step t is proportional to the number of admitted neighbours across all committed entities so far, which is typically O(L · d_admitted) where d_admitted ≤ d_avg.

### 3.7.4 Faithfulness Guarantee

DCA-Trie v2 preserves Condition F by induction over expansion steps. The initial trie T_0 contains only the question entities, which are trivially structurally grounded. At each expansion step, the condition (e_t, r, e') ∈ G is a hard membership check against the verified KG triplet set. The symbolic gates can only reduce the admitted set below G; they cannot add edges not in G. Therefore T_t ⊆ G at every step t, and by induction the accumulated trie remains structurally grounded throughout generation.

---

## 3.8 False Negative Rate Analysis

### 3.8.1 Per-Gate FNR Contribution

Under the symbolic oracle, a gold path p*_q is excluded when either gate fails on it. The type gate fails on p*_q when:

1. The answer type inference (Section 3.5.4) returns a type set that does not include the terminal entity's type, OR
2. The terminal entity has no recorded types in the subgraph but the inferred type set is non-empty.

The range gate fails on p*_q when:

1. A relation in p*_q has a known range declaration that excludes the tail entity's type, OR
2. A relation in p*_q has a known range declaration, and the tail entity has a known type that does not intersect with that range.

Both gates fail conservatively: when type information is missing for an entity, the gate defaults to admit.

### 3.8.2 Answer Type Inference Error

The primary source of FNR is answer type inference error. If the question asks "what did X do" and the correct answer entity has type "Profession", but the pattern matcher maps "what" + "do" to {"Profession"} correctly, the gate passes. If it incorrectly maps to {"Person"}, the gate blocks entities of type "Profession".

This is a deterministic lookup error, not a threshold-dependent one. It can be diagnosed by inspecting the pattern match for each FNR case and corrected by adding or adjusting patterns. The pattern set in Section 3.5.4 has been validated on 100 WebQSP validation questions to achieve FNR < 0.05.

### 3.8.3 Entity Type Availability

A secondary source of FNR is incomplete entity type coverage in the subgraph. Not every entity in the 2-hop BFS neighbourhood has a `common.topic.notable_types` triple. When the terminal entity has no recorded types and the answer type gate is non-empty, the gate defaults to admit (conservative behaviour). This means incomplete coverage does not increase FNR — it simply reduces the gate's pruning effectiveness.

For the range gate, the same conservative default applies: if the tail entity has no recorded types, the range check passes regardless of the relation's declared range. Incomplete coverage reduces pruning but does not risk false negatives.

### 3.8.4 No Compounding Risk

Unlike the product-score formulation (Section 3.8 of earlier design), the symbolic oracle has no compounding failure mechanism. Each gate is an independent set containment check. The failure events are:

$$P(\text{exclude} \mid p^*_q) = P(\text{type\_fail} \cup \text{range\_fail}) \leq P(\text{type\_fail}) + P(\text{range\_fail}) \tag{3.17}$$

by the union bound. The type gate fails only when both the question provides a type signal and the entity has a known type outside that signal. The range gate fails only when both the relation has a known range and the entity has a known type outside that range. These events are determined by the completeness of the subgraph's type information, which is independent of the LLM's parametric knowledge, the question's syntactic complexity, and the hop depth.

---

## 3.9 Baseline Systems

Four system configurations are evaluated.

**CoT Baseline.** The same Llama-3.1-8B backbone used in GCR (Dubey et al., 2024), run without any KG constraint oracle and prompted with standard chain-of-thought reasoning. This provides the unconstrained reference point.

**GCR.** The original Graph-Constrained Reasoning framework (Luo et al., 2025a) with static KG-Trie constraints $W_{val}(t) = f(\mathcal{G}, E_q)$. This is the primary constrained baseline. Reproduction targets WebQSP Hits@1 within 1-2% of the reported 92.6.

**DCA-Trie v1.** The static symbolic filtering variant described in Section 3.6, implemented as a preprocessing step in the GCR pipeline with no threshold or embedding model.

**DCA-Trie v2.** The iterative symbolic expansion variant described in Section 3.7, integrated into the decoding process.

All configurations use the same Llama-3.1-8B backbone, the same entity-linking pipeline for $E_q$, and matched decoding settings (beam width $k = 5$ for constrained systems). This isolates differences to the constraint oracle design rather than model or preprocessing variation.

---

## 3.10 Evaluation Protocol

### 3.10.1 Datasets

Evaluation is conducted on two standard Freebase-based KGQA datasets (Bollacker et al., 2008).

**WebQSP** (Yih et al., 2016): 4,737 questions, mostly requiring 1-2 hops. The standard test set is used for evaluation. A separate disjoint set of 100 questions is used only for validation.

**ComplexWebQuestions (CWQ)** (Talmor and Berant, 2018): 34,689 questions with higher compositional complexity and hop depths up to 4. This dataset is used in full for evaluation with no overlap with the validation split.

### 3.10.2 Evaluation Metrics

Six metrics are used across both datasets.

Hits@1 reports the proportion of examples where the top predicted answer matches the gold entity. F1 is the entity-level overlap score for settings with multiple acceptable answers, following common KGQA evaluation practice (Lan et al., 2022a). Structural Faithfulness Rate is the fraction of generated paths where all triplets are verified members of the underlying KG. $\text{SIR}^*$ is computed using Equations (3.8) through (3.11). $\text{SIR}^*_{\text{type}}$ and $\text{SIR}^*_{\text{traj}}$ are reported separately to attribute the contribution of each filtering component. Average Trie Size Per Step is the mean size of the valid token set $|V_t|$ across decoding steps, used as a direct efficiency indicator of constraint selectivity.

### 3.10.3 Hop-Depth Stratification

All major metrics are also reported by hop depth (1, 2, 3, and 4), consistent with prior KGQA analysis (Lan et al., 2022a). Permissiveness effects are expected to increase with hop depth. Aggregate results alone would conceal this behaviour. The decomposed SIR metrics are especially important to inspect by hop depth, since $\text{SIR}^*_{\text{type}}$ is hop-independent while $\text{SIR}^*_{\text{traj}}$ may vary with depth due to the range gate's cumulative effect.

### 3.10.4 Experimental Configuration

All experiments run on a single NVIDIA A100 (40GB). Llama-3.1-8B inference uses 16-bit precision through HuggingFace Transformers (Wolf et al., 2020). All constrained systems use beam width $k = 5$. The unconstrained CoT baseline uses $k = 1$ to match the original GCR setup. Maximum generation length is set to $3L + 2$, with $L = 2$ for WebQSP and $L = 4$ for CWQ. Unlike the earlier design, no sentence encoder is loaded or used at any point in the pipeline. The symbolic oracle operates entirely on string-level KG data and requires no GPU resources.

---

## 3.11 Scope Boundaries and Anticipated Limitations

**Single KG and domain.** Experiments are limited to Freebase-based KGQA. Generalisation to other KGs, such as Wikidata and ConceptNet, remains future work. The answer type oracle $\mathcal{T}(q, h)$ and the relation schema map are tuned to Freebase's ontological structure and would require re-specification for other knowledge bases.

**No model fine-tuning.** The Llama-3.1-8B backbone is not fine-tuned. Observed improvements reflect changes in constraint oracle design rather than parameter adaptation.

**Type oracle approximation.** The answer type inference procedure uses question structure heuristics rather than a trained classifier. Errors in type inference directly increase FNR. This is analysed in the ablation study in Chapter 4.

**Empirical faithfulness evidence.** Condition F is assessed empirically by checking generated paths against KG triplets, consistent with established practice (Luo et al., 2025a; Li et al., 2025). The formal proof of faithfulness under the symbolic oracle is provided as a proof sketch with stated axioms in this chapter.

**Limited relation schema coverage.** The relation schema map includes approximately 40 Freebase relations covering the most common relation types across people, film, location, organization, government, sports, music, award, and book domains. Relations outside this coverage are not constrained by the range gate. Expanding coverage to all 700+ relation types in the WebQSP subgraph is an engineering extension that can only improve pruning without increasing FNR risk.

**Entity type coverage in subgraphs.** The 2-hop BFS neighbourhood does not include type information for every entity. The type oracle treats entities with unknown types as admissible by default, reducing pruning effectiveness but protecting against false negatives.
