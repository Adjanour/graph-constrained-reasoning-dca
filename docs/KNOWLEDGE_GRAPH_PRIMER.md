# Knowledge Graph Primer for GCR/DCA-Trie

A trimmed reference covering only the concepts needed to understand this project.

---

## What Is a Knowledge Graph?

A **Knowledge Graph (KG)** is a structured representation of facts consisting of entities (nodes) and relationships (edges), where:

1. Data is **structured** in graph form
2. **Normalized** into small units (triples)
3. **Connected** via meaningful relationships
4. Typically **large**, **explicit**, and **declarative**

The core unit is the **triple**: `(Subject, Predicate, Object)`.

```
(Elvis_Presley,  starred_in,       Blue_Hawaii)
 └─ subject ──┘  └─ predicate ─┘  └─ object ─┘
```

**Why not just a relational database?**

| Dimension | Knowledge Graph | Relational Database |
|-----------|-----------------|-------------------|
| **Relationships** | First-class citizen, directly modeled | Foreign key joins |
| **Multi-hop query** | Native graph traversal | k JOINs, exponential cost |
| **Schema** | Flexible, can evolve | Rigid, needs migrations |
| **Semantic reasoning** | Supports inference | Exact matching only |
| **Extensibility** | Add new edge types freely | New tables/columns required |

**Major Knowledge Graphs:**

| KG | Scale | Domain | Access |
|----|-------|--------|--------|
| **Wikidata** | ~100M entities, ~1.5B statements | General | SPARQL endpoint, dump |
| **Freebase** (deprecated) | ~50M entities | General | Migrated to Wikidata |
| **DBpedia** | ~40M entities | General (from Wikipedia) | SPARQL, dumps |
| **YAGO** | ~10M entities | General | Download |
| **Google KG** | ~7B entities | General | API (limited) |
| **ConceptNet** | ~8M nodes | Commonsense | API, download |

---

## Data Models: RDF vs Labeled Property Graphs

There are two dominant data models. GCR/DCA-Trie targets **Freebase**, which uses the RDF model.

### RDF (Resource Description Framework)

The W3C standard. Every piece of data is a triple with URI-identified elements:

```turtle
@prefix ex: <http://example.org/> .
ex:Elvis_Presley ex:starred_in ex:Blue_Hawaii .
ex:Blue_Hawaii rdf:type ex:Movie .
ex:Elvis_Presley rdf:type ex:Person .
```

- Strict separation: instances vs. schema are both RDF
- No null values — missing info is simply absent (Open World Assumption)
- **Best for:** Linked Data, ontology-heavy systems, academic/research KGs

### Labeled Property Graphs (LPG)

The model used by Neo4j. Entities and relationships carry **properties** (key-value pairs):

```
Node: Elvis_Presley
  Labels: [Person, Musician]
  Properties: { birth_date: "1935-01-08" }

Edge: Elvis_Presley -[:STARRED_IN]-> Blue_Hawaii
  Properties: { role: "Chad Gates" }
```

- Properties on both nodes and edges (RDF requires reification for edge properties)
- Faster traversal for OLTP workloads
- **Best for:** Production applications, recommendation engines, fraud detection

### Quick Comparison

| Criterion | RDF | LPG |
|-----------|-----|-----|
| **Standardisation** | W3C standard | De facto (Cypher, GQL) |
| **Properties on edges** | Reification needed | Native |
| **Schema/ontology** | RDFS, OWL (built-in) | Application-level |
| **Reasoning** | Built-in (RDFS/OWL) | Manual |
| **Interoperability** | Excellent (Linked Data) | Limited |

---

## Ontologies and Schemas

An **ontology** defines the vocabulary for a domain: classes, properties, axioms, and inheritance. This is the metadata that TypeOracle exploits.

### Key Concepts

- **Classes**: Categories of entities (Person, Movie, Location)
- **Properties/Relations**: Types of relationships (starred_in, directed_by)
- **Domain/Range**: Constraints on where a relation can connect

### RDFS Example

```turtle
ex:Person rdf:type rdfs:Class .
ex:Movie  rdf:type rdfs:Class .
ex:starred_in rdfs:domain ex:Person ;
              rdfs:range  ex:Movie .
```

This declares that `starred_in` can only connect a Person to a Movie — the KG's own metadata about valid triples.

### Open World Assumption (OWA)

KGs operate under OWA: if a triple is *not present*, it does *not mean false*. It means unknown. (Opposite of relational databases' Closed World Assumption.)

### Relevance to DCA-Trie

In Freebase, relation range declarations already exist in the KG:
- `people.person.nationality` has range = {Country}
- `film.director` has range = {Person}

DCA-Trie exploits these declarations as **free constraint signals** — no learned encoder needed. The ontology's domain/range declarations answer "should this path be in the constraint set?" via simple set-containment checks.

---

## KGQA and Constrained Decoding

### The Task

Given a natural language question and a KG, find the answer entity reachable via multi-hop reasoning paths.

```
Q: "What award did Elvis Presley win in 1971?"
KG path: Elvis_Presley → award_won → Grammy_Award → year → 1971
Answer: Grammy Award
```

### Benchmarks

| Dataset | Questions | Hops | Source KG |
|---------|-----------|------|-----------|
| **WebQSP** | 4,737 | 1-2 | Freebase |
| **ComplexWebQuestions (CWQ)** | 34,689 | 2-4 | Freebase |
| **LC-QuAD 2.0** | 24,907 | Multiple | Wikidata |

### Constrained Decoding

The core idea: **intercept the token selection process and physically prevent invalid tokens from being chosen.**

At each generation step:
1. Compute what tokens are **valid** given the KG and what's been generated so far
2. Set the probability of all invalid tokens to **exactly zero** (logit masking)
3. Let the LLM pick from only the valid set

```
LLM logits → [logit mask from oracle] → softmax → sample
```

This guarantees the output is always structurally faithful to the KG.

### The GCR/DCA-Trie Approach

| System | Approach | Key Idea |
|--------|----------|----------|
| **GCR** (Luo et al., 2025) | Static KG trie, logit masking | 100% structural faithfulness |
| **DoG** (Li et al., 2025) | Step-wise trie expansion | Dynamic, entity-committed expansion |
| **DCA-Trie** | Ontology-aware pruning | Uses KG type + range to prune irrelevant paths |

### Evaluation Metrics

- **Hits@1**: Exact answer match
- **F1**: Token-level overlap between predicted and ground truth
- **Path Accuracy**: Fraction of generated paths that are KG-valid
- **SIR**: Semantic Irrelevance Ratio — fraction of valid-but-irrelevant paths remaining

---

## Multi-Hop Reasoning and Path Explosion

A **k-hop path** traverses k edges. The difficulty grows as O(d^k) where d is the average out-degree — the **path explosion problem**.

A Freebase entity with average degree ~20 at 3 hops yields up to 20^3 = 8,000 candidate paths. Only one of them is the correct answer path. This is why tight constraint oracles matter.
