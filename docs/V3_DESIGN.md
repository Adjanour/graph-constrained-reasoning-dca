# Lazy Constraint Materialisation for Graph-Constrained Decoding

Design note for DCA-Trie v3. Companion to
[LAZY_CONSTRAINT.md](LAZY_CONSTRAINT.md), which covers usage.

---

## 1. Setting

Graph-constrained decoding masks the support of the next-token distribution so
that generation cannot leave the knowledge graph. Formally, a constraint is a
function

$$C : \Sigma^* \to 2^{V}$$

from a decoded prefix to the set of admissible next tokens over vocabulary
$V$. At each step the model's logits are masked to $C(y_{<t})$ before the
softmax, so the induced language $\mathcal{L}(C)$ is exactly the set of
sequences the decoder can emit.

GCR (Luo et al., ICML 2025) realises $C$ by enumerating every path up to $L$
hops, tokenising them, and compiling the result into a trie before decoding
begins. DoG (Li et al., ACL 2025) realises it by rebuilding a trie per beam per
reasoning step from that beam's head pool.

These differ along two axes that the literature treats as one: **what $C$
admits**, and **when $C$ is materialised**. Our v2 conflated them, and the
conflation is instructive.

## 2. What v2 established

v2 made $C$ state-dependent by restarting decoding at each hop: one
`generate()` call per hop, each constrained by a freshly built trie of 1-hop
continuations. It lost 21–37 points of Hits@1. The cause is not the
state-dependence; it is the three unrelated changes that restarting forced.

**Scoring.** The score of a beam became $\sum_h s_h$, where $s_h$ is
HuggingFace's `sequences_scores` for hop $h$ — a *length-normalised* mean
log-probability over that hop's completion. A sum of independently normalised
segment means is not the log-probability of the concatenated sequence, so
beams are not ranked by any coherent objective. Worse, each $s_h$ is computed
over a frontier-local support whose cardinality varies by two orders of
magnitude across entities; a frontier with three candidates concentrates
probability mass and scores higher than one with three hundred, independently
of whether the path is correct. Ranking therefore correlates with branching
factor. This is a defect of *rebuilding*, not of state-dependence: a single
pass scores all beams under one accumulation convention.

**Termination.** The static trie contains paths of every length $1 \le \ell
\le L$, so `</PATH>` is a legal token at every depth and the model selects
path length. v2's per-hop trie admits exactly one triple followed by
`</PATH>`, and the hop count is fixed by a Python loop with no early exit.
Every path is forced to length $L$. On WebQSP, where most gold paths are 1–2
hops and Chapter 4 ran $L=4$, this alone is a plausible majority of the deficit.

**Distributional match.** The checkpoint is fine-tuned to emit one contiguous
`<PATH>…</PATH>` block. v2's hop-2 prompt contains an unterminated `<PATH>`
followed by a restarted one, with the intermediate entity re-emitted as a new
head — a surface form absent from training.

The lesson generalises: *dynamism in the constraint must not be purchased with
changes to the decoding procedure.*

## 3. Design

v3 separates the two axes. It keeps GCR's single decoding pass verbatim — same
`generate()` call, same generation config, same prompt, same sentinels — and
changes only the representation of $C$: from a precompiled trie over an
enumerated path set to an object that materialises trie nodes on demand from
the graph.

Concretely, `LazyGraphConstraint` exposes the same interface as `MarisaTrie`
(a single `get(prefix) -> List[int]`) and is passed to an unmodified
`GraphConstrainedDecoding`. State is organised around two notions:

- An **anchor** is a confirmed position: the path text emitted so far, the
  current entity, the hop index. Anchors are created only at entity boundaries.
- A **frontier** is the set of continuations available from an anchor —
  gated outgoing edges, plus `</PATH>` when the current entity is an
  admissible terminal.

Frontiers are built lazily and memoised by anchor, so the number of
materialised frontiers is bounded by the number of distinct partial paths the
beams actually visit, rather than by the size of the graph.

Candidate token sequences are rendered from the anchor's own text
(`<PATH>` + anchor text + continuation) and never via a decode round-trip, so
the constraint never assumes $\text{tokenize}(\text{decode}(x)) = x$.

### 3.1 Frontiers are not prefix-free

The one non-obvious mechanism. Freebase entity names are not prefix-free in
token space: `Characters` and `Characters with this condition` are distinct
entities sharing a token prefix, as are `Spoken by` and its longer siblings.
Consequently, a prefix reaching the end of one candidate does **not** exhaust
the frontier containing it, and the natural implementation — advance to the
newly opened frontier when the current one returns no continuation — silently
drops both the longer sibling and the `</PATH>` token.

The constraint therefore computes

$$C(y_{<t}) = \bigcup_{a \in A(y_{<t})} F_a(y_{<t})$$

as a union over the *chain* of anchors preceding the prefix, rather than a
lookup against the deepest one. This is the price of lazy materialisation over
a non-prefix-free alphabet, and it is where both defects found during
development lived.

We note that this is the same phenomenon Chapter 4 attributes to v2 as
"tokenisation misalignment ... a fundamental architectural issue". It is real,
and it is a data-structure problem rather than an architectural limit.

### 3.2 Gate placement

v1 applies the TypeOracle gates to complete paths, before decoding, against
question-level answer types, with terminal-hood defined as depth $= L$. v3
applies the same two gates at the frontier, conditioned on the partial path:

- `range_gate(r, e)` admits an outgoing edge, unchanged;
- `type_gate(e, T, h, h)` decides whether the path may *stop* at $e$, the
  repeated hop argument forcing the gate's terminal branch.

The question the type gate answers thus shifts from "is this the right type at
depth $L$?" to "is this an admissible answer *here*?" — strictly more
information, and it restores model-chosen termination as a by-product.

## 4. Properties

**Language equivalence.** With gates disabled,
$\mathcal{L}(C_{\text{v3}}) = \mathcal{L}(C_{\text{static}})$. This is the
correctness condition: admitting less silently removes gold paths, admitting
more forfeits faithfulness, and neither is distinguishable from noise in an
accuracy number. Both inclusions are tested directly — exhaustively on a
synthetic graph, and on real WebQSP subgraphs by replaying every static
sequence through the constraint and by sampling descents back against the
static language.

It follows that `DCA_v3_NoGates` should reproduce the baseline's accuracy
within sampling error, which makes it a calibration run rather than an
ablation: any discrepancy localises to the mechanism before gated results can
be read.

**Cost.** Static materialisation is $O(|\Pi_L|)$ in the enumerated path set,
paid per question regardless of what the beams do; lazy materialisation is
$O(kLd)$ in beam width $k$, depth $L$, and mean out-degree $d$. Measured on
WebQSP (constraint construction only, CPU): 3–10× cheaper at $L=2$, 30–135× at
$L=4$.

**An artefact of the static path.** At $L=4$ every question sampled hit the
50,000-path cap in `graph_utils.dfs`, so the static trie at that depth is a
DFS-order truncation of $\Pi_L$ rather than $\Pi_L$ itself — a selection bias
of the kind Chapter 4 §4.8 already flags for the adaptive-budget study.
Chapter 4 reports $L=4$ for both datasets. Lazy materialisation has no
corresponding cap.

## 5. Position

The contribution is not a new constraint — with gates off, v3 admits precisely
what GCR admits — but a demonstration that constraint *dynamism* and
constraint *content* are separable, and that implementing dynamism inside the
decoding pass rather than around it preserves the properties v2 forfeited. GCR
pre-compiles; DoG rebuilds per beam per step and pays for it in latency; we are
not aware of prior work that expands a graph-backed constraint lazily within a
single beam search.

The negative result is thereby repositioned: v2 is not evidence that dynamic
constraint reconstruction fails, but evidence that reconstruction is the wrong
implementation of dynamism.

## 6. Open questions

No accuracy result yet exists; everything above is structural, and whether v3
recovers the baseline's 89.0% / 53.2% is the outstanding test. Per-call
constraint latency is 0.3–1.7 ms and scales with the anchor-chain union, which
is the first thing to profile if decoding slows. Cycle semantics follow
`graph_utils.dfs` (revisits permitted) because Freebase name collapsing creates
genuine self-loops; the alternative changes the admitted language and would
need its own ablation. Finally, v3 cannot use v1's
`infer_answer_types_from_paths` fallback, since that requires the enumerated
path set — a path-free answer-type estimator would close the last asymmetry
between the two conditions.
