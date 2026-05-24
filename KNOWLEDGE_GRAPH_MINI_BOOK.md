# Knowledge Graphs & Graph Theory: A Comprehensive Mini-Book

> A detailed guide covering graph theory fundamentals, knowledge graph design, construction, querying, machine learning integration, and LLM-era applications. With analogies, visual concepts, resource links, and hands-on references.

---

## Table of Contents

1. [Graph Theory: The Foundation](#1-graph-theory-the-foundation)
2. [What Is a Knowledge Graph?](#2-what-is-a-knowledge-graph)
3. [Data Models: RDF vs. Labeled Property Graphs](#3-data-models-rdf-vs-labeled-property-graphs)
4. [Ontologies, Schemas, and Semantics](#4-ontologies-schemas-and-semantics)
5. [Knowledge Graph Construction Pipeline](#5-knowledge-graph-construction-pipeline)
6. [Querying Knowledge Graphs](#6-querying-knowledge-graphs)
7. [Knowledge Graph Embeddings](#7-knowledge-graph-embeddings)
8. [Graph Neural Networks for KGs](#8-graph-neural-networks-for-kgs)
9. [Knowledge Graphs and LLMs: GraphRAG](#9-knowledge-graphs-and-llms-graphrag)
10. [Knowledge Graph Question Answering (KGQA)](#10-knowledge-graph-question-answering-kgqa)
11. [Storage and Production Infrastructure](#11-storage-and-production-infrastructure)
12. [Advanced Topics](#12-advanced-topics)
13. [Learning Roadmap](#13-learning-roadmap)
14. [Resource Index](#14-resource-index)

---

# 1. Graph Theory: The Foundation

## 1.1 What Is a Graph?

A **graph** is a mathematical structure $G = (V, E)$ consisting of a set of **vertices** (also called nodes) $V$ and a set of **edges** $E$ connecting pairs of vertices.

**Analogy:** A graph is a city map where intersections are vertices and roads are edges. The structure captures *connections*, not just individual points.

### Types of Graphs

| Type | Description | Example |
|------|-------------|---------|
| **Undirected** | Edges have no direction | Friendship graph (Alice-Bob = Bob-Alice) |
| **Directed** | Edges have a direction (arcs) | Web page links (A links to B, not necessarily B to A) |
| **Labeled** | Edges have type labels | KG triple: (Elvis, `starred_in`, Blue Hawaii) |
| **Weighted** | Edges have numeric weights | Road network (edge weight = distance) |
| **Multigraph** | Multiple edges between same vertices | Two people who are both colleagues and cousins |
| **Hypergraph** | Edges connect any number of vertices | A research paper with 5 co-authors |

### Key Graph Properties

- **Degree**: Number of edges incident to a vertex. In-degree (incoming) vs. out-degree (outgoing) for directed graphs.
- **Path**: A sequence of vertices where consecutive pairs are connected by edges. Path length = number of edges.
- **Connected component**: A maximal subgraph where any two vertices are connected by some path.
- **Cycle**: A path that starts and ends at the same vertex with no repeated vertices in between.
- **Tree**: A connected graph with no cycles. Every tree with $n$ vertices has exactly $n-1$ edges.
- **DAG** (Directed Acyclic Graph): A directed graph with no directed cycles. Used for dependency graphs, version histories.

## 1.2 Graph Representations

How do we store a graph in a computer?

| Representation | Space | Edge Query | Best For |
|----------------|-------|------------|----------|
| **Adjacency Matrix** $A_{ij} = 1$ if edge $(i,j)$ exists | $O(V^2)$ | $O(1)$ | Dense graphs, linear algebra operations |
| **Adjacency List** (per-vertex list of neighbours) | $O(V + E)$ | $O(\deg(v))$ | Sparse graphs (most real-world KGs) |
| **Edge List** (list of $(u,v)$ pairs) | $O(E)$ | $O(E)$ | Simple processing, graph file formats |

**Real-world note**: Knowledge graphs are almost always **sparse** (average degree $\ll V$) and **labeled** (edges have types), making adjacency lists the natural representation.

## 1.3 Graph Traversal Algorithms

| Algorithm | Strategy | Use Case |
|-----------|----------|----------|
| **BFS** (Breadth-First) | Explore level by level | Shortest path in unweighted graphs, KG neighbour discovery |
| **DFS** (Depth-First) | Explore one branch fully before backtracking | Path enumeration in KGs, cycle detection |
| **Dijkstra** | BFS with priority queue for weighted edges | Shortest path in weighted graphs |
| **A\*** | Dijkstra + heuristic | Goal-directed pathfinding |

**BFS for KG path enumeration** is fundamental: from a start entity, BFS traverses outgoing relations level by level up to a max depth, enumerating all reachable paths. This is how GCR and DCA-Trie build their constraint tries.

## 1.4 Important Graph Concepts for KGs

### Multi-Hop Reasoning
A **k-hop path** traverses $k$ edges. In KGs, the difficulty of multi-hop reasoning grows as $O(d^k)$ where $d$ is the average out-degree — the **path explosion problem**. A Freebase entity with average degree $\approx 20$ at 3 hops yields up to $20^3 = 8{,}000$ candidate paths.

### Graph Isomorphism
Two graphs are isomorphic if there is a vertex bijection preserving edges. **Weisfeiler-Lehman (WL) test** is a classic algorithm for graph isomorphism testing, and forms the theoretical basis for many GNN architectures (1-WL = GCN/GAT expressive power limit).

### Graph Laplacian
$L = D - A$ where $D$ is the degree matrix and $A$ is the adjacency matrix. Used in spectral graph theory, graph clustering (spectral clustering), and graph signal processing.

## 1.5 Resources

- **"Graph Theory" by Reinhard Diestel** — The standard graduate text. Free online: https://diestel-graph-theory.com/
- **"Networks, Crowds, and Markets" by Easley & Kleinberg** — Accessible, application-focused. https://www.cs.cornell.edu/home/kleinber/networks-book/
- **3Blue1Brown visual graph series**: https://youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab
- **Graph traversal visualizer**: https://visualgo.net/en/dfsbfs

---

# 2. What Is a Knowledge Graph?

## 2.1 Definition

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

**Analogy**: A knowledge graph is like a city map annotated with business directories. The map shows connections (roads = relationships), and the directories say what kind of business each location is (types). Together they let you answer "Find all Italian restaurants within 2 miles of this cinema."

## 2.2 Why Not Just a Relational Database?

| Dimension | Knowledge Graph | Relational Database |
|-----------|-----------------|-------------------|
| **Relationships** | First-class citizen, directly modeled | Foreign key joins |
| **Multi-hop query** | $O(d^k)$ traversal, native | $k$ JOINs, exponential cost |
| **Schema** | Flexible, can evolve | Rigid, needs migrations |
| **Semantic reasoning** | Supports inference | Exact matching only |
| **Extensibility** | Add new edge types freely | New tables/columns required |

## 2.3 Major Knowledge Graphs

| KG | Scale | Domain | Access |
|----|-------|--------|--------|
| **Wikidata** | ~100M entities, ~1.5B statements | General | SPARQL endpoint, dump |
| **Freebase** (deprecated) | ~50M entities | General | Migrated to Wikidata |
| **DBpedia** | ~40M entities | General (from Wikipedia) | SPARQL, dumps |
| **YAGO** | ~10M entities | General | Download |
| **Google KG** | ~7B entities | General | API (limited) |
| **NELL** | ~3M beliefs | General web | Download |
| **ConceptNet** | ~8M nodes | Commonsense | API, download |
| **WordNet** | ~155K words | Linguistic | Download |

## 2.4 Analogy: KGs vs. Vector Databases

A **vector database** stores chunks of text as embedding vectors and retrieves by similarity. A **knowledge graph** stores individual facts as structured triples and retrieves by graph traversal.

| Query | Vector DB | Knowledge Graph |
|-------|-----------|-----------------|
| "Find all companies funded by Sequoia" | Returns chunks *mentioning* Sequoia + funding | Returns exact entities via `FUNDED_BY` edge |
| "What integrations does Product A share with Product B?" | Needs multiple queries + manual intersection | Single Cypher query with `MATCH` |
| Completeness guarantee | "Most relevant" chunks (may miss) | All matching edges (if in graph) |

## 2.5 Resources

- **Aidan Hogan et al., "Knowledge Graphs" (book)**: https://kg-book.com/ — The definitive reference. Free online.
- **Tutorial by Markus Krötzsch (TU Dresden)**: https://iccl.inf.tu-dresden.de/web/Knowledge_Graphs/en — Full lecture series with videos.
- **DeepLearning.AI "Agentic Knowledge Graph Construction"**: https://www.deeplearning.ai/courses/agentic-knowledge-graph-construction/

---

# 3. Data Models: RDF vs. Labeled Property Graphs

There are two dominant data models for KGs. Choosing between them depends on whether you prioritise **interoperability** (RDF) or **performance** (Property Graph).

## 3.1 RDF (Resource Description Framework)

The W3C standard for representing graph data. Every piece of data is a triple:

```
<http://example.org/Elvis_Presley>
    <http://example.org/starred_in>
    <http://example.org/Blue_Hawaii> .
```

**Key characteristics:**
- Every triple element is a URI (or literal for values)
- Internationalised (IRI support)
- Strict separation: instances vs. schema are both RDF
- No null values — if information doesn't exist, triple simply isn't there (Open World Assumption)

**Serialization formats:** Turtle (`.ttl`), RDF/XML, JSON-LD, N-Triples.

```turtle
@prefix ex: <http://example.org/> .
ex:Elvis_Presley ex:starred_in ex:Blue_Hawaii .
ex:Blue_Hawaii rdf:type ex:Movie .
ex:Elvis_Presley rdf:type ex:Person .
```

**Best for:** Linked Data, cross-organisation data exchange, ontology-heavy systems, academic/research KGs (Wikidata, DBpedia).

### RDF-star (RDF*)

An extension allowing statements *about* statements — e.g., "Elvis starred in Blue Hawaii **according to source X**." This is critical for provenance, confidence scores, and temporal annotations without needing reification patterns.

## 3.2 Labeled Property Graphs (LPG)

The model used by Neo4j, TigerGraph, Amazon Neptune. Entities and relationships can carry **properties** (key-value pairs):

```
Node: Elvis_Presley
  Labels: [Person, Musician]
  Properties: { birth_date: "1935-01-08", birth_place: "Tupelo" }

Node: Blue_Hawaii
  Labels: [Movie]
  Properties: { release_year: 1961 }

Edge: Elvis_Presley -[:STARRED_IN]-> Blue_Hawaii
  Properties: { role: "Chad Gates" }
```

**Key characteristics:**
- Properties on both nodes and edges (RDF requires reification for edge properties)
- Labels for node types
- No global URI requirement
- Faster traversal for OLTP workloads

**Best for:** Production applications, recommendation engines, fraud detection, internal enterprise KGs.

## 3.3 RDF vs. LPG: Decision Table

| Criterion | RDF | LPG |
|-----------|-----|-----|
| **Standardisation** | W3C standard | De facto (Cypher, GQL) |
| **Properties on edges** | Reification needed | Native |
| **Schema/ontology** | RDFS, OWL (built-in) | Application-level |
| **Reasoning** | Built-in (RDFS/OWL reasoning) | Manual |
| **Query language** | SPARQL | Cypher, Gremlin, GQL |
| **Interoperability** | Excellent (Linked Data) | Limited |
| **Multi-hop traversal perf** | Slower (triple stores vary) | Fast (native graph) |
| **NLP/literature domain** | Common | Less common |

## 3.4 Query Language Comparison

| Task | SPARQL (RDF) | Cypher (Neo4j) |
|------|-------------|----------------|
| Find Elvis's movies | `SELECT ?m WHERE { ex:Elvis_Presley ex:starred_in ?m }` | `MATCH (:Person {name:'Elvis'})-[:STARRED_IN]->(m:Movie) RETURN m` |
| Two-hop query | `SELECT ?x WHERE { ex:Elvis_Presley ex:starred_in / ex:directed_by ?x }` | `MATCH (:Person {name:'Elvis'})-[:STARRED_IN]->()-[:DIRECTED_BY]->(d) RETURN d` |
| Property filter | `FILTER(?year > 1960)` | `WHERE m.release_year > 1960` |

## 3.5 Resources

- **W3C RDF Primer**: https://www.w3.org/TR/rdf11-primer/
- **SPARQL tutorial by Cambridge Semantics**: https://docs.cambridgesemantics.com/graphlakehouse/v3.2/userdoc/pdf/Graph-Lakehouse-2025.0-DBv32-Documentation.pdf (Learn SPARQL section)
- **Neo4j Graph Data Modeling course** (free): https://graphacademy.neo4j.com/courses/modeling-fundamentals/
- **RDF vs Property Graphs deep dive**: https://arunbaby.com/ml-system-design/0034-knowledge-graph-systems/ (Section 2)

---

# 4. Ontologies, Schemas, and Semantics

## 4.1 What Is an Ontology?

An **ontology** is a formal specification of a shared conceptualisation. In KG terms, it defines the **vocabulary** for describing a domain:

- **Classes**: Categories of entities (Person, Movie, Location)
- **Properties/Relations**: Types of relationships (starred_in, directed_by, born_in)
- **Axioms**: Rules and constraints (a Person cannot be a Movie; every Movie must have at least one director)
- **Inheritance**: Class hierarchies (Musician is a subclass of Person)

**Analogy:** An ontology is like the grammar of a language. It defines which sentences are valid, what the words mean, and how they can combine — independent of any particular utterance.

## 4.2 Ontology Languages

| Language | Layer | Expressivity | Use Case |
|----------|-------|-------------|----------|
| **RDFS** (RDF Schema) | Class/property hierarchies, domain/range | Simple | Basic vocabularies |
| **OWL 2** (Web Ontology Language) | Classes, restrictions, cardinality, disjointness, equality | Complex | Formal domain modeling |
| **SKOS** (Simple Knowledge Organization System) | Concept hierarchies, labels, relations | Moderate | Thesauri, taxonomies |
| **SHACL** (Shapes Constraint Language) | Validation constraints | Structural | Data quality validation |

### RDFS Example
```turtle
ex:Person rdf:type rdfs:Class .
ex:Movie  rdf:type rdfs:Class .
ex:starred_in rdfs:domain ex:Person ;
              rdfs:range  ex:Movie .
```

This declares that `starred_in` can only connect a Person to a Movie — the KG's own metadata about valid triples.

### OWL Reasoning

OWL enables **deductive reasoning** — inferring new facts from existing ones:

```turtle
ex:Musician rdfs:subClassOf ex:Person .
ex:Elvis_Presley rdf:type ex:Musician .
```

An OWL reasoner infers: `ex:Elvis_Presley rdf:type ex:Person`.

This can materialise new triples or answer queries over implicit knowledge without storing redundant data.

## 4.3 The Open World Assumption (OWA)

KGs operate under the **Open World Assumption**: if a triple is *not present*, it does *not mean false*. It means unknown. This is opposite to relational databases (Closed World Assumption: missing = false).

**Example:** If the KG has no `birth_date` for Elvis, OWA says "we don't know his birth date," not "he has no birth date."

## 4.4 Designing an Ontology: A Practical Process

1. **Define competency questions**: "What questions must this KG answer?"
2. **Identify key entities**: What are the nouns in your domain?
3. **Identify relationships**: What verbs connect them?
4. **Define class hierarchy**: Which entities share properties?
5. **Set constraints**: Cardinality, domain, range
6. **Add axioms**: Disjointness, equivalence, transitive properties

**Real-world example** (from Freebase DCA-Trie project):
- Entity types: Person, Location, Organization, Award, etc.
- Relations: `starred_in` has `range` = Movie, `people.born_in` has `range` = Location
- The range declaration is *already in the KG* — DCA-Trie exploits it as a free constraint signal.

## 4.5 Resources

- **OWL 2 Primer**: https://www.w3.org/TR/owl2-primer/
- **Protege ontology editor** (free): https://protege.stanford.edu/
- **RDF/OWL Pizza tutorial**: https://csiro-enviro-informatics.github.io/info-engineering/tutorials/tutorial-intro-to-rdf-and-owl.html
- **Ontology Design 101**: https://protege.stanford.edu/publications/ontology_development/ontology101-noy-mcguinness.html

---

# 5. Knowledge Graph Construction Pipeline

Building a KG from raw data involves several stages:

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  Source  │ → │  Entity  │ → │ Relation │ → │  Graph   │
│  Data    │   │  Extract │   │  Extract │   │  Storage │
└──────────┘   └──────────┘   └──────────┘   └──────────┘
                     │              │
                     ↓              ↓
                ┌──────────┐   ┌──────────┐
                │  Entity  │   │  Canon-  │
                │  Linking │   │  icalise │
                └──────────┘   └──────────┘
```

## 5.1 Source Data Types

| Source | Approach | Example Tooling |
|--------|----------|----------------|
| **Structured** (CSV, SQL) | Direct mapping via R2RML, SPARQL CONSTRUCT | Ontop, SPARQL CONSTRUCT |
| **Semi-structured** (JSON, XML) | Path extraction, schema mapping | XSLT, JSON-LD framing |
| **Unstructured** (text, PDF, web) | NLP pipeline (NER + RE) | spaCy, REBEL, LLMs |

## 5.2 Named Entity Recognition (NER)

Identifying entity mentions in text:

> "Elvis Presley was born in Tupelo, Mississippi."
> → `[Elvis Presley] (Person)`, `[Tupelo] (Location)`, `[Mississippi] (Location)`

**Approaches:**
- Rule-based (regex, gazetteer)
- Statistical (CRF)
- Neural (BERT-based, e.g., spaCy `en_core_web_trf`, GLiNER)

**Challenge:** Entity disambiguation — "Paris" could be the city in France or the character in Greek mythology.

## 5.3 Relation Extraction (RE)

Identifying relationships between entities:

> "Elvis Presley starred in Blue Hawaii."
> → `(Elvis_Presley, starred_in, Blue_Hawaii)`

**Approaches:**
- End-to-end models (REBEL, REBEL-large)
- Prompted LLM extraction with structured output schemas
- Bootstrapping from seed patterns (Snowball, DIPRE)

**LLM-based extraction** (modern approach):
```python
from pydantic import BaseModel
class Triple(BaseModel):
    subject: str
    predicate: str
    object: str

# Use instructor/outlines to get structured output
```

## 5.4 Entity Linking (Disambiguation)

Mapping surface forms to KG identifiers:

| Surface Form | Candidate KG Entity |
|-------------|-------------------|
| "Elvis" | Elvis_Presley (Q303) |
| "The King" | Elvis_Presley (Q303) OR Stephen_King (Q3981) |

**Approaches:**
- String similarity + context
- Entity embedding similarity
- Wikidata API queries

## 5.5 Knowledge Fusion and Deduplication

Multiple sources may describe the same entity differently:

| Source A | Source B | Merge |
|----------|----------|-------|
| "IBM" | "International Business Machines" | → `wikidata:Q37156` |
| "London, UK" | "London, England" | → `wikidata:Q84` |

**Techniques:**
- Blocking: hash entities into buckets, compare only within buckets
- Similarity scoring: attribute-level + structural
- Clustering: connected component over high-similarity pairs

## 5.6 End-to-End Pipeline (LLM-Based)

Modern KG construction uses LLMs for entity + relation extraction:

1. **Crawl** → scrape source documents
2. **Chunk** → split into manageable text segments
3. **Extract** → prompt an LLM with a Pydantic schema to output triples
4. **Resolve** → link surface forms to KG identifiers
5. **Load** → ingest into Neo4j or RDF store

**Tools:** KnowledgeSDK, LangChain, Haystack, LlamaIndex.

## 5.7 Resources

- **"How to Build a Knowledge Graph in 7 Steps"** (Neo4j): https://neo4j.com/blog/knowledge-graph/how-to-build-knowledge-graph/
- **Build KG from any website** (KnowledgeSDK): https://knowledgesdk.com/blog/knowledge-graph-from-website
- **PDF → RDF pipeline**: https://www.gooddata.ai/blog/from-reports-to-knowledge-rdf-knowledge-graph/
- **REBEL relation extraction**: https://huggingface.co/Babelscape/rebel-large
- **GLiNER NER**: https://github.com/urchade/GLiNER

---

# 6. Querying Knowledge Graphs

## 6.1 SPARQL (RDF Query Language)

SPARQL is the W3C standard for RDF data. Its structure resembles SQL but operates on graph patterns.

### Basic Pattern Matching

```sparql
PREFIX ex: <http://example.org/>
SELECT ?movie ?year
WHERE {
  ex:Elvis_Presley ex:starred_in ?movie .
  ?movie ex:release_year ?year .
  FILTER(?year > 1960)
}
```

### Property Paths (Multi-hop)

SPARQL 1.1 supports regular expression paths:

```sparql
# Two-hop query: Elvis → movie → director
SELECT ?director WHERE {
  ex:Elvis_Presley ex:starred_in / ex:directed_by ?director
}

# Variable length: find all reachable entities within 3 hops
SELECT ?x WHERE {
  ex:Elvis_Presley (ex:starred_in|ex:produced_by){1,3} ?x
}
```

### CONSTRUCT (Create RDF on the fly)

```sparql
CONSTRUCT { ?s ex:worked_with ?o }
WHERE {
  ?s ex:starred_in ?m .
  ?o ex:starred_in ?m .
  FILTER(?s != ?o)
}
```

Makes implicit co-star relationships explicit.

### SPARQL Query Forms

| Form | Returns | Example |
|------|---------|---------|
| `SELECT` | Table of variable bindings | Find all movies Elvis starred in |
| `CONSTRUCT` | RDF graph | Create co-star relationships |
| `ASK` | Boolean | "Did Elvis ever work with Tom Jones?" |
| `DESCRIBE` | RDF graph (depends on store) | Get all known facts about Elvis |

## 6.2 Cypher (Property Graph Query Language)

Neo4j's Cypher uses ASCII-art pattern syntax:

```cypher
// Find Elvis's movies
MATCH (p:Person {name: 'Elvis Presley'})-[:STARRED_IN]->(m:Movie)
RETURN m.title, m.release_year

// Two-hop with condition
MATCH (p:Person {name: 'Elvis Presley'})-[:STARRED_IN]->()-[:DIRECTED_BY]->(d:Person)
WHERE d.birth_year > 1930
RETURN d.name

// Variable-length path
MATCH path = (p:Person {name: 'Elvis Presley'})-[:STARRED_IN|PRODUCED_BY*1..3]->(x)
RETURN x.name, length(path) AS hops

// Shortest path
MATCH path = shortestPath(
  (p1:Person {name: 'Elvis Presley'})-[*]-(p2:Person {name: 'Tom Jones'})
)
RETURN path
```

## 6.3 GQL: The New ISO Standard

**GQL** (Graph Query Language, ISO 39075) is the first international standard for property graph queries, unifying Cypher and PGQL concepts. Released 2024, it will gradually replace Cypher as the standard PG query language. Major vendors (Neo4j, Oracle, Amazon Neptune) are adopting it.

## 6.4 Query Optimization Tips

- **Filter early**: Push `WHERE` clauses as deep as possible
- **Index on frequent query patterns**: Entity labels, property equality
- **Limit intermediate results**: Use `LIMIT` for exploratory queries
- **Profile queries**: Neo4j `PROFILE`, SPARQL endpoints provide query plans
- **Avoid Cartesian products**: Every `MATCH` that doesn't connect to prior variables explodes the result set

## 6.5 Resources

- **SPARQL 1.1 Overview**: https://www.w3.org/TR/sparql11-overview/
- **SPARQL by Example** (free course): https://www.cambridgesemantics.com/blog/semantic-university/learn-sparql/
- **Neo4j Cypher Refcard**: https://neo4j.com/docs/cypher-refcard/current/
- **Neo4j Graph Academy** (free courses): https://graphacademy.neo4j.com/
- **GQL standard**: https://www.gqlstandards.org/

---

# 7. Knowledge Graph Embeddings

## 7.1 Why Embed KGs?

KGs are symbolic and discrete. Machine learning needs continuous representations. **KG embeddings** map entities $e \in \mathcal{E}$ and relations $r \in \mathcal{R}$ to vectors in $\mathbb{R}^d$ such that the KG's structural patterns are preserved as geometric relationships.

**Analogy:** KG embeddings are like translating English sentences into vector space — `king - man + woman ≈ queen` (word2vec) is the same idea at the entity level.

## 7.2 Classic Models

### TransE (Bordes et al., 2013)
The simplest and most influential. For a triple $(h, r, t)$, learn embeddings such that:
$$\mathbf{h} + \mathbf{r} \approx \mathbf{t}$$

Score function: $f(h, r, t) = -\lVert \mathbf{h} + \mathbf{r} - \mathbf{t} \rVert$

**Limitation:** Cannot model symmetric relations, 1-to-N, N-to-1, N-to-N patterns.

### RotatE (Sun et al., 2019)
Embeddings in complex space. Relations as **rotations**:
$$\mathbf{t} = \mathbf{h} \circ \mathbf{r} \quad \text{(element-wise rotation)}$$

Can model symmetry, antisymmetry, inversion, and composition — all four relational patterns.

### DistMult (Yang et al., 2015)
Score: $f(h, r, t) = \mathbf{h}^\top \text{diag}(\mathbf{r}) \, \mathbf{t}$
Simple bilinear scoring. Cannot handle antisymmetric relations.

### ComplEx (Trouillon et al., 2016)
DistMult extended to complex space. Can handle antisymmetric relations via complex conjugate.

### Comparison Table

| Model | Space | Scalability | Relational Patterns Captured |
|-------|-------|-------------|------------------------------|
| TransE | $\mathbb{R}^d$ | Excellent | 1-to-1 only |
| DistMult | $\mathbb{R}^d$ | Excellent | Symmetric only |
| ComplEx | $\mathbb{C}^d$ | Good | Symmetric + Antisymmetric |
| RotatE | $\mathbb{C}^d$ | Good | All four patterns |
| ConvE | $\mathbb{R}^d$ | Moderate | All (via 2D convolution) |
| TuckER | $\mathbb{R}^d$ | Moderate | All (via Tucker decomposition) |

## 7.3 Training KG Embeddings

**Loss function:** Maximise score for true triples, minimise for corrupted triples (negative sampling).

```
For each true triple (h, r, t) in KG:
    Corrupt head: (h', r, t) where h' is random entity
    Corrupt tail: (h, r, t') where t' is random entity
    Loss = max(0, γ + f(corrupted) - f(true))
```

**Libraries:** PyKEEN, DGL-KE, OpenKE.

## 7.4 Link Prediction (KG Completion)

Given $(h, r, ?)$, predict the missing tail entity. This is the most common KG embedding task.

**Evaluation:**
- MRR (Mean Reciprocal Rank)
- Hits@K (K = 1, 3, 10)
- Rank of true entity among all candidates

**State-of-the-art** on FB15k-237, WN18RR: NBFNet (GNN-based), RotatE (for simpler patterns).

## 7.5 Resources

- **PyKEEN** (KG embedding library): https://github.com/pykeen/pykeen
- **"Knowledge Graph Embeddings" tutorial (KDD 2026)**: https://bdi-lab.github.io/kkg_kdd2026/
- **OpenKE** (THU toolkit): https://github.com/thunlp/OpenKE
- **RotatE paper + code**: https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding

---

# 8. Graph Neural Networks for KGs

## 8.1 What Is a GNN?

A **Graph Neural Network** learns node representations by **message passing**: each node aggregates information from its neighbours, transforms it, and updates its own representation.

```
Layer 0: h_v^(0) = x_v  (initial features)
Layer k: h_v^(k) = UPDATE( h_v^(k-1), AGGREGATE( { h_u^(k-1) : u ∈ N(v) } ) )
```

**Analogy:** A GNN is like a dinner party where each person (node) talks to their neighbours, then updates their own understanding based on what they heard. After enough rounds, everyone has a broad view of the conversation.

## 8.2 Key GNN Architectures

| Model | Aggregation | Key Idea |
|-------|-------------|----------|
| **GCN** (Kipf & Welling, 2017) | Mean + normalization | First-order spectral approximation |
| **GAT** (Veličković et al., 2018) | Attention-weighted mean | Learn which neighbours matter more |
| **GraphSAGE** (Hamilton et al., 2017) | Mean/LSTM/Pooling | Inductive (works on unseen nodes) |
| **GIN** (Xu et al., 2019) | Sum over MLP | As powerful as 1-WL test |
| **R-GCN** (Schlichtkrull et al., 2018) | Relation-type specific | Handles multiple relation types |

## 8.3 GNNs for Link Prediction

**NBFNet** (Zhu et al., 2021) formulates link prediction as a **label propagation** problem: run a GNN from the head entity with learned message functions for each relation type. State-of-the-art on multiple benchmarks.

**Key insight:** Unlike TransE (which learns a single vector per entity), NBFNet computes representations *contextually* — the same entity gets different representations depending on the query relation.

## 8.4 Inductive vs. Transductive

| Paradigm | Description | Example Model |
|----------|-------------|---------------|
| **Transductive** | Learns embeddings for a fixed set of entities | TransE, RotatE |
| **Inductive** | Can generalise to unseen entities at inference | GraphSAGE, NBFNet, GraIL |

Inductive methods are crucial for production KGs that grow over time.

## 8.5 KG Foundation Models

Recent work pre-trains a single model on diverse KGs and transfers to unseen KGs at inference:

- **ULTRA**: Pre-trains on relation graph structure, then fine-tunes relation representations for any new KG in one forward pass.
- **TRIX**: Pre-trained on multiple domains, zero-shot transfer.
- **KG-ICL**: Treats KG inference as in-context learning.

## 8.6 Resources

- **"Graph Neural Networks" book by Hamilton**: https://www.cs.mcgill.ca/~wlh/grl_book/
- **CS224W (Stanford)**: https://web.stanford.edu/class/cs224w/ — Full course with lectures
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/
- **DGL (Deep Graph Library)**: https://www.dgl.ai/
- **NBFNet paper**: https://arxiv.org/abs/2106.06935

---

# 9. Knowledge Graphs and LLMs: GraphRAG

## 9.1 The Problem: RAG's Flat Ceiling

Standard RAG retrieves text chunks via vector similarity. Limitations:
- Cannot answer multi-hop queries ("What companies use technology X that competitor Y also uses?")
- Cannot guarantee completeness
- Misses cross-references between chunks

## 9.2 What Is GraphRAG?

GraphRAG retrieves structured facts from a KG (not text chunks) as context for LLM generation. The pipeline:

```
Question → Entity Linking → Graph Traversal → Subgraph → Prompt → Answer
```

**Multi-hop example:**

```
Q: "Who is the CEO of the company that acquired GitHub?"
Steps:
1. Link "GitHub" → github_entity
2. Traverse ACQUIRED_BY → Microsoft
3. Traverse CEO_IS → Satya Nadella
4. Return {Satya Nadella, Microsoft, GitHub} as structured context
```

**Analogy:** RAG is like searching a library index to find relevant books. GraphRAG is like consulting a knowledgeable librarian who follows cross-references between books.

## 9.3 GraphRAG Architectures

| System | Approach | Year |
|--------|----------|------|
| **Microsoft GraphRAG** | Community detection → community summaries → global/local search | 2024 |
| **LightRAG** | Dual-level (entity + relation) retrieval | 2024 |
| **HippoRAG** | Personalized PageRank over KG, integrated with vector store | 2024 |
| **HippoRAG 2** | Improved integration of LLM for KG construction | 2025 |
| **GNN-RAG** | GNN to score node relevance, then retrieve shortest paths | 2025 |
| **GFM-RAG** | Graph Foundation Model for RAG | 2025 |

### Microsoft GraphRAG Deep Dive

1. **Build KG**: Extract entities and relationships from documents
2. **Community detection**: Leiden algorithm to find communities
3. **Community summarization**: LLM summarizes each community
4. **Query**: Local search (relevant entities) + Global search (community summaries)
5. **Answer synthesis**: Combine both into final answer

**Key advance:** Global understanding — answers questions requiring corpus-wide synthesis.

### LightRAG

1. **Dual-level indexing**: Entity-level + relation-level embeddings
2. **Incremental update**: Add new documents without full rebuild
3. **Efficient**: No iterative LLM calls for community summarization

## 9.4 KG-Augmented LLM Reasoning (Not Just RAG)

Beyond retrieval, KGs can **constrain** LLM generation:

- **GCR** (Luo et al., 2025): Token-level constraint via KG trie, 100% structural faithfulness
- **DoG** (Li et al., 2025): Step-wise dynamic trie expansion
- **DCA-Trie**: Ontology-aware context pruning using KG type/range metadata
- **RoG** (Luo et al., 2024): Planning-retrieval-reasoning with KG paths
- **ToG** (Sun et al., 2024): LLM as agent iteratively querying KG

## 9.5 Entity Linking for GraphRAG

The critical first step: mapping text mentions to KG entities.

| Tool | Approach |
|------|----------|
| **REL** (Radford et al.) | Neural ED with candidate generation |
| **GENRE** (De Cao et al.) | Autoregressive entity retrieval with trie |
| **Wikidata API** | Direct search, good coverage |
| **LLM-based** | Prompt to identify entities and map to KG IDs |

## 9.6 Resources

- **Microsoft GraphRAG**: https://github.com/microsoft/graphrag
- **LightRAG**: https://github.com/HKUDS/LightRAG
- **Neo4j GraphRAG Python**: https://neo4j.com/docs/neo4j-graphrag-python/current/
- **Agentic KG Construction (DeepLearning.AI)**: https://www.deeplearning.ai/courses/agentic-knowledge-graph-construction/
- **Ontology-Driven KG for GraphRAG** (deepsense.ai): https://deepsense.ai/resource/ontology-driven-knowledge-graph-for-graphrag/

---

# 10. Knowledge Graph Question Answering (KGQA)

## 10.1 The Task

Given a natural language question and a KG, find the answer entity (or entities) reachable via multi-hop reasoning paths.

```
Q: "What award did Elvis Presley win in 1971?"
KG path: Elvis_Presley → award_won → Grammy_Award → year → 1971
Answer: Grammy Award
```

## 10.2 KGQA Approaches

### Semantic Parsing
Translate question → logical form (SPARQL, λ-calculus) → execute against KG.

```
Q: "Who directed Titanic?"
→ SPARQL: SELECT ?d WHERE { :Titanic :directed_by ?d }
→ Execute against KG
→ Answer: James Cameron
```

Challenges: Coverage limited to KG schema, brittle to paraphrasing.

### Retrieval-Based
Embed question, retrieve relevant KG subgraph, reason over subgraph.

- **QA-GNN** (Yasunaga et al., 2021): Jointly encode question + KG subgraph with GNN
- **GNN-RAG** (Mavromatis & Karypis, 2025): GNN for relevance scoring, then LLM over retrieved paths
- **PullNet** (Sun et al., 2019): Iterative retrieval of relevant subgraph

### Constrained Decoding
Constrain LLM generation to produce only KG-valid paths:

- **GCR**: Build static trie of all paths → mask logits → guarantee 100% faithfulness
- **DoG**: Step-wise trie expansion based on the entity just committed
- **DCA-Trie**: Add KG ontology gates (type + range) to prune semantically irrelevant paths

## 10.3 Benchmarks

| Dataset | Questions | Hops | Source KG |
|---------|-----------|------|-----------|
| **WebQSP** | 4,737 | 1-2 | Freebase |
| **ComplexWebQuestions (CWQ)** | 34,689 | 2-4 | Freebase |
| **WebQuestions (WebQ)** | 5,810 | 1-2 | Freebase |
| **LC-QuAD 2.0** | 24,907 | Multiple | Wikidata |

## 10.4 Evaluation Metrics

- **Hits@1**: Exact answer match
- **F1**: Token-level overlap between predicted and ground truth
- **Path Accuracy**: For path-generating methods, fraction of generated paths that are KG-valid
- **SIR**: Semantic Irrelevance Ratio — fraction of valid-but-irrelevant paths that remain in constraint set

## 10.5 Resources

- **GCR paper** (ICML 2025): https://proceedings.mlr.press/v267/luo25t.html
- **DoG paper** (ACL 2025): https://aclanthology.org/2025.acl-long.1186/
- **GNN-RAG** (ACL 2025 Findings): https://aclanthology.org/2025.findings-acl.856/
- **WebQSP**: https://github.com/kelvinguu/webqsp
- **DCA-Trie project**: `/home/bernard/research/projects/graph-constrained-reasoning/` (your project)

---

# 11. Storage and Production Infrastructure

## 11.1 Graph Databases Compared

| Database | Model | Query Language | Best For |
|----------|-------|---------------|----------|
| **Neo4j** | LPG | Cypher, GQL | OLTP, most popular, ACID |
| **Amazon Neptune** | LPG + RDF | Gremlin, SPARQL | AWS-native, multi-model |
| **TigerGraph** | LPG | GSQL | Large-scale analytics |
| **ArangoDB** | Multi-model | AQL | Mixed graph + document |
| **Virtuoso** | RDF | SPARQL | Large RDF stores |
| **Stardog** | RDF | SPARQL, OWL reasoning | Enterprise reasoning |
| **GraphDB** | RDF | SPARQL, SHACL | Semantic web, OWL reasoning |
| **Apache Jena Fuseki** | RDF | SPARQL | Lightweight, research |
| **Dgraph** | LPG (custom) | DQL | Distributed, low latency |

## 11.2 Scaling Graph Databases

**Sharding strategies:**
- **Edge-cut partitioning**: Assign vertices to machines, edges that cross machines are remote
- **Vertex-cut partitioning**: Assign edges to machines, vertices replicated across partitions
- **METIS / ParMETIS**: Partition graphs to minimise edge cuts

**Distributed graph DBs:**
- Amazon Neptune (clustered, up to 128 TB)
- TigerGraph (MPP, up to hundreds of nodes)
- JanusGraph (HBase/Cassandra backend)

## 11.3 Caching for Graph Workloads

- **Neighbourhood cache**: Pre-fetch 2-hop neighbourhoods for frequently queried entities
- **Pattern cache**: Cache results of common query templates (parameterised)
- **Embedding cache**: Cache precomputed entity embeddings for ML workloads

## 11.4 When NOT to Use a Graph Database

- **Pure aggregation**: Sum, count, group-by are faster in relational/columnar
- **Trivial relationships**: Flat data with one join type
- **Full-text search**: Vector DB or Elasticsearch are better
- **High-volume simple lookups**: Key-value stores
- **Analytics-only workloads**: Data warehouse (no need for traversal)

## 11.5 Resources

- **Neo4j deployment guide**: https://neo4j.com/docs/operations-manual/current/
- **Building production KG (Brian Curry)**: https://medium.com/@brian-curry-research/building-a-production-grade-knowledge-graph-system-a-complete-guide-36434d85b987
- **Graph database comparison**: https://www.g2.com/categories/graph-databases

---

# 12. Advanced Topics

## 12.1 Temporal Knowledge Graphs

Entities and relationships change over time. Temporal KG add timestamps to triples:

```
(Elvis_Presley,  starred_in,  Blue_Hawaii,  [1960-01-01, 1962-12-31])
```

**Key tasks:** Time-aware link prediction, event forecasting, temporal reasoning.

**Models:** TComplEx, TeRo, T-GAP, ChronoR.

## 12.2 Hyper-Relational Knowledge Graphs

Beyond binary triples: statements can have **qualifiers**—additional properties that contextualise a relationship.

```
(Elvis_Presley,  won_award,  Grammy_Award,  {category: "Best Pop", year: 1975, city: "Los Angeles"})
```

Wikidata uses this extensively (e.g., population values with qualifiers for year, source, methodology). **Star-based** models (StarE, QUAD) embed hyper-relational facts.

## 12.3 Inductive KG Reasoning

Predict missing links in a **new KG** that was unseen during training — not just new triples between known entities. Required for real-world deployment where new entities appear daily.

**Key methods:**
- Rule-based: NeuralLP, DRUM
- Subgraph-based: GraIL, NBFNet
- Foundation model: ULTRA, KG-ICL

## 12.4 Neuro-Symbolic AI

The convergence of **neural learning** (embeddings, GNNs) with **symbolic reasoning** (logical rules, ontologies):

- **Differentiable rule learning**: Learn logical rules in continuous space, then extract symbolic rules
- **Constraint satisfaction**: Use KG constraints (type, range) to guide neural learning (DCA-Trie philosophy)
- **Abductive reasoning**: Infer best explanations for observations given KG background knowledge

## 12.5 Knowledge Graph Quality Assurance

| Dimension | Question | Method |
|-----------|----------|--------|
| **Completeness** | Are all facts present? | Rule mining, missing link prediction |
| **Correctness** | Are the facts true? | Human verification, cross-source validation |
| **Consistency** | Any logical contradictions? | SHACL/OWL reasoning, constraint checking |
| **Freshness** | How up-to-date? | Temporal decay weighting, refresh scheduling |

## 12.6 Resources

- **Temporal KG survey**: https://arxiv.org/abs/2303.07401
- **Wikidata qualifiers**: https://www.wikidata.org/wiki/Help:Qualifiers
- **SHACL specification**: https://www.w3.org/TR/shacl/
- **URA (Ultra) KG foundation model**: https://github.com/DeepGraphLearning/ULTRA
- **Neuro-Symbolic AI survey**: https://arxiv.org/abs/2205.11318

---

# 13. Learning Roadmap

A structured 9-week plan adapted from the Ontology & KG Cookbook.

## Phase 1: Foundations (Weeks 1-3)

| Week | Topic | Practice | Resources |
|------|-------|----------|-----------|
| 1 | Graph theory basics + RDF | Build a small RDF graph with `rdflib` | KG Book Ch.1-2, Diestel Ch.1-3 |
| 2 | SPARQL querying | Query Wikidata via its SPARQL endpoint | SPARQL by Example |
| 3 | Ontology design (RDFS/OWL) | Model a small domain in Protege | OWL 2 Primer, Protege tutorial |

## Phase 2: Construction (Weeks 4-5)

| Week | Topic | Practice | Resources |
|------|-------|----------|-----------|
| 4 | KG construction pipeline | Extract entities from text → build KG | spaCy, NER tutorial |
| 5 | Neo4j + Cypher | Load extracted KG, write 5 queries | Neo4j Graph Academy |

## Phase 3: LLM Integration (Weeks 6-7)

| Week | Topic | Practice | Resources |
|------|-------|----------|-----------|
| 6 | GraphRAG basics | Build GraphRAG pipeline with Neo4j | Neo4j GraphRAG library |
| 7 | Constrained decoding | Implement GCR-style trie for KG paths | GCR paper, DCA-Trie codebase |

## Phase 4: Advanced (Weeks 8-9)

| Week | Topic | Practice | Resources |
|------|-------|----------|-----------|
| 8 | KG embeddings | Train TransE/RotatE with PyKEEN | PyKEEN GitHub, KDD tutorial |
| 9 | GNNs for KGs | Implement NBFNet with PyG | PyG docs, CS224W lectures |

---

# 14. Resource Index

## Books

| Title | Author | Link |
|-------|--------|------|
| **Knowledge Graphs** | Hogan et al. | https://kg-book.com/ |
| **Graph Theory** | Diestel | https://diestel-graph-theory.com/ |
| **Graph Representation Learning** | Hamilton | https://www.cs.mcgill.ca/~wlh/grl_book/ |
| **Networks, Crowds, and Markets** | Easley & Kleinberg | https://www.cs.cornell.edu/home/kleinber/networks-book/ |

## Courses

| Course | Provider | Link |
|--------|----------|------|
| **Knowledge Graphs** (Krötzsch) | TU Dresden | https://iccl.inf.tu-dresden.de/web/Knowledge_Graphs/en |
| **CS224W: ML with Graphs** | Stanford | https://web.stanford.edu/class/cs224w/ |
| **Graph Data Modeling** | Neo4j | https://graphacademy.neo4j.com/courses/modeling-fundamentals/ |
| **Agentic KG Construction** | DeepLearning.AI | https://www.deeplearning.ai/courses/agentic-knowledge-graph-construction/ |
| **Knowledge Graphs for API Discovery** | DeepLearning.AI | https://learn.deeplearning.ai/courses/knowledge-graphs-for-ai-agent-api-discovery/ |
| **KDD 2026 KG Tutorial** | Whang et al. | https://bdi-lab.github.io/kkg_kdd2026/ |

## Software

| Tool | Purpose | Link |
|------|---------|------|
| **Neo4j** | Leading graph database | https://neo4j.com/ |
| **Protege** | Ontology editor | https://protege.stanford.edu/ |
| **rdflib** | Python RDF library | https://rdflib.readthedocs.io/ |
| **PyKEEN** | KG embeddings | https://github.com/pykeen/pykeen |
| **PyTorch Geometric** | GNNs | https://pytorch-geometric.readthedocs.io/ |
| **SPARQL wrapper** | SPARQL from Python | https://rdflib.readthedocs.io/ |
| **OpenKE** | KG embeddings (THU) | https://github.com/thunlp/OpenKE |
| **DGL** | Deep Graph Library | https://www.dgl.ai/ |
| **Microsoft GraphRAG** | LLM + KG | https://github.com/microsoft/graphrag |
| **LightRAG** | Lightweight GraphRAG | https://github.com/HKUDS/LightRAG |

## Datasets

| Dataset | Size | Domain | Access |
|---------|------|--------|--------|
| **Wikidata** | ~1.5B statements | General | https://query.wikidata.org/ |
| **DBpedia** | ~40M entities | Wikipedia | https://www.dbpedia.org/ |
| **YAGO** | ~10M entities | General | https://yago-knowledge.org/ |
| **ConceptNet** | ~8M nodes | Commonsense | https://conceptnet.io/ |
| **WebQSP** | 4,737 Q/A | KGQA | HuggingFace (rmanluo) |

---

> *"Knowledge graphs are the bridge connecting data and intelligence. They organize knowledge in a structured way, enabling machines to understand and reason about complex semantic relationships."*
