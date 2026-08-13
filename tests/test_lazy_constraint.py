"""Correctness tests for the v3 lazy graph constraint.

The property that matters is *language equivalence*: with gates disabled,
``LazyGraphConstraint`` must admit exactly the token sequences the static
baseline trie admits.  If it admits fewer, gold paths become unreachable; if
it admits more, the faithfulness guarantee is gone.  Neither shows up in an
accuracy number as anything but noise, so it is checked directly.

Both inclusions are covered:

- ``static ⊆ lazy`` by replaying every static sequence through the constraint;
- ``lazy ⊆ static`` by walking the constraint and checking where it lands.

None of this loads model weights.
"""

import random

import pytest

import src.utils as graph_utils
from src.graph_constrained_decoding import GraphConstrainedDecoding
from src.trie import MarisaTrie

from approach3_symbolic.type_oracle import TypeOracle
from lazy_constraint import LazyGraphConstraint
from utils import PATH_END, PATH_START

MAX_HOPS = 2
N_WALKS = 200

TOY_TRIPLES = [
    ["Blue Hawaii", "film.film.directed_by", "Norman Taurog"],
    ["Blue Hawaii", "film.film.starring", "Elvis Presley"],
    ["Blue Hawaii", "film.film.country", "United States"],
    ["Norman Taurog", "people.person.nationality", "United States"],
    ["Norman Taurog", "people.person.place_of_birth", "Chicago"],
    ["Elvis Presley", "people.person.nationality", "United States"],
    ["Chicago", "location.location.containedby", "Illinois"],
]


def static_language(tokenizer, graph, start, max_hops):
    """Exactly what ``build_unfiltered_trie`` would encode, as a set."""
    paths = graph_utils.dfs(graph, start, max_hops)
    wrapped = [f"{PATH_START}{graph_utils.path_to_string(p)}{PATH_END}" for p in paths]
    ids = tokenizer(wrapped, padding=False, add_special_tokens=False).input_ids
    return {tuple(i + [tokenizer.eos_token_id]) for i in ids}


def lazy_constraint(tokenizer, question, max_hops=MAX_HOPS, gates_enabled=False):
    graph = graph_utils.build_graph(question["graph"], undirected=False)
    start = [e for e in question["q_entity"] if e in graph]
    oracle = TypeOracle.from_graph(question["graph"])
    answer_types = oracle.infer_answer_types(question["question"])
    constraint = LazyGraphConstraint(
        tokenizer, graph, start, oracle, answer_types, max_hops,
        gates_enabled=gates_enabled,
    )
    return graph, start, constraint


def walk(constraint, tokenizer, rng, n):
    """Random descents; every one must terminate through ``</PATH>``."""
    start_id = tokenizer.convert_tokens_to_ids(PATH_START)
    eos_id = tokenizer.eos_token_id
    out = []
    for _ in range(n):
        prefix = (start_id,)
        for _ in range(400):
            if prefix[-1] == eos_id:
                break
            allowed = constraint.get(list(prefix))
            assert allowed, f"dead end with no close offered: {prefix[-8:]}"
            prefix = prefix + (rng.choice(allowed),)
        else:
            pytest.fail("walk never terminated")
        out.append(prefix)
    return out


# ---------------------------------------------------------------------------
# Toy graph — exhaustive, so both inclusions are exact rather than sampled
# ---------------------------------------------------------------------------


def test_toy_graph_language_is_exactly_the_static_trie(tokenizer):
    question = {
        "graph": TOY_TRIPLES,
        "q_entity": ["Blue Hawaii"],
        "question": "What is the nationality of the director of Blue Hawaii?",
    }
    graph, start, constraint = lazy_constraint(tokenizer, question, max_hops=3)
    static = static_language(tokenizer, graph, start, 3)

    start_id = tokenizer.convert_tokens_to_ids(PATH_START)
    eos_id = tokenizer.eos_token_id

    lazy, stack = set(), [(start_id,)]
    while stack:
        prefix = stack.pop()
        if prefix[-1] == eos_id:
            lazy.add(prefix)
            continue
        assert len(prefix) < 400, "runaway prefix"
        allowed = constraint.get(list(prefix))
        assert allowed, f"dead end with no close offered: {prefix[-8:]}"
        stack.extend(prefix + (t,) for t in allowed)

    assert lazy == static


# ---------------------------------------------------------------------------
# Real Freebase subgraphs — too large to enumerate, so replay + sample
# ---------------------------------------------------------------------------


def test_static_sequences_all_replay_through_lazy(tokenizer, webqsp):
    """Every gold-reachable static path must survive the lazy constraint.

    This is the inclusion that protects recall.  It is also the one that
    catches token-prefix collisions between sibling entity names — Freebase
    has an entity called "Characters" alongside "Characters with this
    condition", and a frontier that stops at the shorter one silently drops
    both the longer sibling and the close token.
    """
    rng = random.Random(0)
    for question in webqsp:
        graph, start, constraint = lazy_constraint(tokenizer, question)
        if not start:
            continue
        static = static_language(tokenizer, graph, start, MAX_HOPS)

        for seq in rng.sample(sorted(static), min(200, len(static))):
            for i in range(1, len(seq)):
                allowed = constraint.get(list(seq[:i]))
                assert seq[i] in allowed, (
                    f"{question['id']}: blocked at token {i} of "
                    f"{tokenizer.decode(seq[:i + 1])!r}"
                )


def test_lazy_never_admits_a_path_outside_the_graph(tokenizer, webqsp):
    """The inclusion that protects faithfulness."""
    rng = random.Random(1)
    for question in webqsp:
        graph, start, constraint = lazy_constraint(tokenizer, question)
        if not start:
            continue
        static = static_language(tokenizer, graph, start, MAX_HOPS)

        for seq in walk(constraint, tokenizer, rng, N_WALKS):
            assert seq in static, (
                f"{question['id']}: admitted a non-graph path "
                f"{tokenizer.decode(seq)!r}"
            )


def test_gates_only_ever_remove_paths(tokenizer, webqsp):
    """Gated language must be a subset of ungated — gates prune, never add."""
    rng = random.Random(2)
    for question in webqsp:
        _, start, gated = lazy_constraint(tokenizer, question, gates_enabled=True)
        if not start:
            continue
        graph, _, _ = lazy_constraint(tokenizer, question)
        static = static_language(tokenizer, graph, start, MAX_HOPS)

        for seq in walk(gated, tokenizer, rng, N_WALKS):
            assert seq in static


# ---------------------------------------------------------------------------
# Integration with the decoding wrapper
# ---------------------------------------------------------------------------


def test_drops_into_graph_constrained_decoding(tokenizer, webqsp):
    """Free before ``<PATH>``, constrained inside, free again after."""
    torch = pytest.importorskip("torch")

    question = webqsp[0]
    _, start, constraint = lazy_constraint(tokenizer, question, gates_enabled=True)
    if not start:
        pytest.skip("no linked start entity")

    start_id = tokenizer.convert_tokens_to_ids(PATH_START)
    end_id = tokenizer.convert_tokens_to_ids(PATH_END)
    gcr = GraphConstrainedDecoding(
        tokenizer, constraint, start_id, end_id, enable_constrained_by_default=False
    )

    prompt = tokenizer(
        f"Question: {question['question']}\nReasoning path:", add_special_tokens=False
    ).input_ids
    sent = list(prompt)
    assert len(gcr.allowed_tokens_fn(0, torch.tensor(sent))) == len(tokenizer)

    sent.append(start_id)
    for _ in range(200):
        allowed = gcr.allowed_tokens_fn(0, torch.tensor(sent))
        assert len(allowed) != len(tokenizer), "fell back to unconstrained decoding"
        sent.append(allowed[0])
        if sent[-1] == end_id:
            break
    else:
        pytest.fail("path never closed")

    assert len(gcr.allowed_tokens_fn(0, torch.tensor(sent))) == len(tokenizer)

    decoded = tokenizer.decode(sent[len(prompt):])
    assert decoded.startswith(PATH_START) and decoded.endswith(PATH_END)


def test_close_is_offered_before_the_hop_budget_is_spent(tokenizer, webqsp):
    """The model, not the loop, decides where the path ends.

    v2 fixed path length at ``max_hops``; the static trie contains paths of
    every length up to L.  This asserts the lazy constraint keeps that
    property, which is the main thing v2's restart-per-hop design lost.
    """
    question = webqsp[0]
    _, start, constraint = lazy_constraint(tokenizer, question, max_hops=3)
    if not start:
        pytest.skip("no linked start entity")

    end_id = tokenizer.convert_tokens_to_ids(PATH_END)
    start_id = tokenizer.convert_tokens_to_ids(PATH_START)

    # Walk one full hop, then check </PATH> is available at the boundary.
    graph = graph_utils.build_graph(question["graph"], undirected=False)
    one_hop = next(
        f"{start[0]} -> {graph[start[0]][n]['relation']} -> {n}"
        for n in graph.neighbors(start[0])
    )
    prefix = tokenizer(
        f"{PATH_START}{one_hop}", add_special_tokens=False
    ).input_ids
    assert prefix[0] == start_id
    assert end_id in constraint.get(prefix), "cannot stop before the hop budget"


def test_no_dfs_is_performed(tokenizer, webqsp, monkeypatch):
    """The constraint must never enumerate the path set."""
    def explode(*args, **kwargs):
        pytest.fail("lazy constraint called graph_utils.dfs")

    question = webqsp[0]
    _, start, constraint = lazy_constraint(tokenizer, question, gates_enabled=True)
    if not start:
        pytest.skip("no linked start entity")

    monkeypatch.setattr(graph_utils, "dfs", explode)
    walk(constraint, tokenizer, random.Random(3), 20)

    stats = constraint.stats()
    assert stats["frontier_builds"] > 0
    assert stats["candidates_materialised"] > 0


def test_marisa_trie_and_lazy_constraint_are_interchangeable(tokenizer, webqsp):
    """Same ``get`` contract, so conditions differ only in the constraint."""
    question = webqsp[0]
    graph, start, constraint = lazy_constraint(tokenizer, question)
    if not start:
        pytest.skip("no linked start entity")

    static = static_language(tokenizer, graph, start, MAX_HOPS)
    reference = MarisaTrie(
        [list(s) for s in static], max_token_id=len(tokenizer) + 1
    )

    rng = random.Random(4)
    for seq in rng.sample(sorted(static), min(50, len(static))):
        for i in range(1, len(seq)):
            assert set(constraint.get(list(seq[:i]))) == set(
                reference.get(list(seq[:i]))
            ), f"diverged at token {i} of {tokenizer.decode(seq[:i])!r}"
