# DCA-Trie v3: Lazy Constrained Decoding

How to run, test, and interpret the v3 condition (`DCA_v3_Lazy`).
For the design rationale and its relation to GCR, DoG, and v2, see
[V3_DESIGN.md](V3_DESIGN.md).

---

## 1. What v3 is

All three DCA conditions differ in one component only — the object that answers
"which tokens may come next?" — and in nothing else. Entity linking, the
prompt, the model, and the generation config are identical.

| | Constraint | Decoding calls | Path length chosen by |
|---|---|---|---|
| `GCR_Baseline` | Every DFS path, pre-compiled into one trie | 1 | model |
| `DCA_v1_Static` | Same, TypeOracle-filtered before compiling | 1 | model |
| `DCA_v2_Dynamic` | One trie per hop, rebuilt per beam | one per hop | `max_hops` loop |
| `DCA_v3_Lazy` | Graph-backed frontier, materialised on demand | 1 | model |

v3 keeps GCR's single decoding pass — so all beams are scored in one
probability space and the model still decides where to emit `</PATH>` — while
taking DoG's idea that the admissible set should depend on reasoning state. The
constraint is never materialised as a whole; only the frontier the beams
actually reach.

`LazyGraphConstraint` (`experiments/type_oracle_full/lazy_constraint.py`)
duck-types `MarisaTrie`: it exposes a single `get(prefix) -> List[int]`. It is
therefore passed to the *unmodified* `GraphConstrainedDecoding`, and
`run_lazy_decoding` is `run_constrained_decoding` with the trie swapped.

### Where the gates moved

v1 filters the whole path set up front against question-level answer types.
v3 applies the same two gates at the frontier, conditioned on the partial path:

- `range_gate(rel, tail)` admits an outgoing edge, as in v1.
- `type_gate(entity, answer_types, hop, hop)` decides whether the path may
  *stop* at the current entity. Passing `hop` as both arguments forces the
  gate's terminal branch, so the question becomes "is this a legal answer?"
  rather than v1's "is this the right type at exactly depth `index_len`?"

`</PATH>` is offered as a legal token at every entity that passes the type
gate, which is what keeps path-length selection with the model.

---

## 2. Running it

Same CLI as every other condition. See
`experiments/type_oracle_full/README.md` for the shared arguments.

```bash
# v3 alone
./.venv/bin/python experiments/type_oracle_full/main.py \
  --method v3 --datasets RoG-webqsp --max-samples 100

# the comparison that matters: baseline vs v1 vs v3
./.venv/bin/python experiments/type_oracle_full/main.py \
  --method lazy --datasets RoG-webqsp --max-samples 300

# adds the gates-off control
./.venv/bin/python experiments/type_oracle_full/main.py \
  --method lazy-ablation --datasets RoG-webqsp --max-samples 300
```

New `--method` values:

| Method | Conditions |
|---|---|
| `v3` | `DCA_v3_Lazy` |
| `v3-nogates` | `DCA_v3_NoGates` |
| `lazy` | `GCR_Baseline`, `DCA_v1_Static`, `DCA_v3_Lazy` |
| `lazy-ablation` | the above plus `DCA_v3_NoGates` |

Full runs and Vast.ai launches work unchanged:

```bash
./.venv/bin/python experiments/type_oracle_full/main.py \
  --method lazy --datasets RoG-webqsp --full --run-name lazy1

bash scripts/run_vast.sh --run-name lazy1 --datasets RoG-webqsp
```

### Run `v3-nogates` first

With gates disabled, v3's admitted language is provably identical to the
baseline trie's (see §4). So `DCA_v3_NoGates` should land on the baseline's
Hits@1 within sampling noise. If it does not, the difference is coming from the
constraint mechanism, not from the gates — and every gated number is
uninterpretable until that is resolved. Treat it as the calibration run, not an
optional ablation.

---

## 3. Reading the output

Predictions land in the usual place, in the same format as the baseline's
(`<PATH>…</PATH> # Answer: …`), so `compute_hits` scores all conditions
identically:

```
results/<run>/<dataset>/predictions_DCA_v3_Lazy.jsonl
```

Each v3 record carries a `lazy_stats` block:

| Field | Meaning |
|---|---|
| `candidates_materialised` | Edges rendered into the constraint for this question |
| `frontier_builds` | Frontiers built (one per distinct partial path reached) |
| `anchors_visited` | Confirmed entity boundaries crossed |

Compare `candidates_materialised` against the baseline's `n_paths_all` for the
same question — that ratio is the scalability claim. The condition summary
reports the per-question averages:

```
DCA_v3_Lazy  (lazy: 780.4 candidates over 312.1 frontiers per question)
```

v3 skips the DFS entirely (`PrepCache.build(..., enumerate_paths=False)`), so
`n_dfs_paths` in `graph_stats` is 0 for v3 rows by construction, and `timing_s`
is not charged for work the condition exists to avoid.

---

## 4. Testing

Nothing here loads model weights; the tokenizer and the WebQSP subgraphs are
enough. The suite takes ~15s on CPU.

```bash
./.venv/bin/python -m pytest tests/ -q
```

Tests skip cleanly if the tokenizer or dataset is not cached locally.

### What is being tested, and why

The property that matters is **language equivalence**: with gates disabled,
`LazyGraphConstraint` must admit exactly the token sequences the static
baseline trie admits. Admit fewer and gold paths silently become unreachable;
admit more and the faithfulness guarantee is gone. Neither failure shows up in
an accuracy number as anything but noise, so both inclusions are checked
directly.

| Test | Property |
|---|---|
| `test_toy_graph_language_is_exactly_the_static_trie` | Exhaustive equality on a small graph |
| `test_static_sequences_all_replay_through_lazy` | `static ⊆ lazy` — protects recall |
| `test_lazy_never_admits_a_path_outside_the_graph` | `lazy ⊆ static` — protects faithfulness |
| `test_gates_only_ever_remove_paths` | Gated language is a subset of ungated |
| `test_drops_into_graph_constrained_decoding` | Free → constrained → free, unmodified wrapper |
| `test_close_is_offered_before_the_hop_budget_is_spent` | Model, not the loop, ends the path |
| `test_no_dfs_is_performed` | `graph_utils.dfs` is never called |
| `test_marisa_trie_and_lazy_constraint_are_interchangeable` | Same `get` contract as the static trie |
| `tests/test_v3_runner.py` | Runner emits a baseline-shaped record; gates flag reaches the constraint |

### The failure mode these caught

Both bugs found while building v3 were the same phenomenon: **sibling entity
names whose token sequences are proper prefixes of each other.** Freebase has
an entity called `Characters` alongside `Characters with this condition`, and
`Spoken by` alongside longer siblings. A frontier that treats "reached the end
of a candidate" as "this frontier is exhausted" silently drops the longer
sibling *and* the `</PATH>` token. The fix is to union frontiers along the
anchor chain rather than switch between them (`lazy_constraint.py:134`).

This is the same class of problem `CHAPTER4_RESULTS.md` §4.4 attributes to v2
as "tokenization misalignment ... a fundamental architectural issue". It is
real, but it is a data-structure bug, not an architectural limit — which is why
the equivalence tests exist and why they run on real Freebase names rather than
a toy graph alone.

---

## 5. Benchmarking construction cost

```bash
./.venv/bin/python experiments/type_oracle_full/bench_lazy.py
./.venv/bin/python experiments/type_oracle_full/bench_lazy.py --index-len 4
```

Measures constraint construction only — no weights loaded. Static cost is paid
up front for every question whatever the beams do; lazy cost is measured over
`--beams` random descents, standing in for what beam search touches.

Measured on WebQSP `train`, 8B tokenizer, CPU:

| `index_len` | static paths | static tokens | static | lazy candidates | lazy | speedup |
|---|---|---|---|---|---|---|
| 2 | 1,001–4,774 | 30k–149k | 0.10–1.01s | 176–1,688 | 0.03–0.15s | 3–10× |
| 4 | 50,000 (capped) | 2.6M–3.3M | 3.2–11.3s | 176–2,490 | 0.02–0.26s | 30–135× |

**The `index_len=4` rows hit the 50,000-path cap in `graph_utils.dfs`.** At that
depth the static trie is a DFS-order truncation of the path set rather than the
whole of it — the same selection-bias problem `CHAPTER4_RESULTS.md` §4.8 flags
for the adaptive-budget study. Chapter 4 reports `index_len=4` for both
datasets; the current CLI default is 2, which does not truncate on the WebQSP
questions sampled. v3 has no such cap at any depth.

---

## 6. Known limitations

- **No accuracy result yet.** Everything verified so far is structural: the
  constraint admits the right language at the right cost. Whether v3 recovers
  the baseline's 89.0% (WebQSP) / 53.2% (CWQ) needs the GPU run. Run
  `v3-nogates` first (§2).
- **Per-call latency.** `get()` costs 0.3–1.7 ms depending on branching. At
  ~5 beams × ~60 tokens that is roughly 0.1–0.5 s/question against a ~6 s
  baseline, but it scales with the anchor-chain scan in
  `LazyGraphConstraint._anchor_chain`, which is the hotspot to profile first if
  decoding slows down.
- **Cycle handling differs by default.** `block_cycles=False` matches
  `graph_utils.dfs`, which permits revisits. Enabling it changes the admitted
  language rather than merely tightening it, because Freebase name collapsing
  creates genuine self-loops — so it would need its own ablation, not a flag
  flip.
- **`infer_answer_types_from_paths` fallback is unavailable to v3.** It needs
  the enumerated path set, which v3 does not build. When
  `infer_answer_types` returns empty, v3 runs with empty answer types, and the
  type gate admits everything (its documented conservative behaviour). v1 gets
  the fallback, so the two conditions can disagree on this subset of questions.

---

## 7. Files

| File | Role |
|---|---|
| `experiments/type_oracle_full/lazy_constraint.py` | `LazyGraphConstraint` |
| `experiments/type_oracle_full/decoding.py` | `run_lazy_decoding` |
| `experiments/type_oracle_full/experiment.py` | `_run_v3`, `LAZY_CONDITIONS` |
| `experiments/type_oracle_full/main.py` | `CONDITIONS_BY_METHOD` |
| `experiments/type_oracle_full/bench_lazy.py` | Construction-cost benchmark |
| `tests/test_lazy_constraint.py` | Language-equivalence and integration tests |
| `tests/test_v3_runner.py` | Runner wiring tests |
