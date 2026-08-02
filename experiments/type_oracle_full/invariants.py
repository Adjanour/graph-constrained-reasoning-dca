"""
invariants.py — Executable assertions that verify code–math alignment.

Every theorem or invariant in the paper gets an assert here.
If code and math diverge, the assert fires that run — not at defense.

Usage
-----
    from invariants import assert_trie_utterable, assert_graph_membership

    assert_trie_utterable(tokenizer, trie, path_strings)
    assert_graph_membership(all_paths, nx_graph)
    assert filtered ⊆ all_paths
    assert all_admissible(filtered_paths, oracle, answer_types, max_hop)
"""

import re
from typing import List, Set

from utils import PATH_START, PATH_END, logger


# ---------------------------------------------------------------------------
# Freebase ID detection (for display, NOT for filtering)
# ---------------------------------------------------------------------------

_FB_ID_RE = re.compile(r"^[gm]\.\w+$")


# ---------------------------------------------------------------------------
# Assertion 1: Trie utterability
# ---------------------------------------------------------------------------

def assert_trie_utterable(tokenizer, trie, path_strings: List[str]) -> None:
    """Assert that every path in the trie decodes to a valid path string.

    This verifies that the tokenizer can round-trip every path:
      encode(path_string) → token_ids → decode → path_string

    If this fails, the trie contains token sequences that the LLM
    cannot meaningfully generate.
    """
    if not path_strings or trie is None:
        return

    wrapped = [f"{PATH_START}{s}{PATH_END}" for s in path_strings]
    for w in wrapped:
        token_ids = tokenizer.encode(w, add_special_tokens=False)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
        # Normalize whitespace for comparison
        w_norm = " ".join(w.split())
        d_norm = " ".join(decoded.split())
        assert w_norm == d_norm, (
            f"Trie utterability failed:\n"
            f"  original:  {w_norm!r}\n"
            f"  decoded:   {d_norm!r}"
        )

    logger.debug("Invariant: trie_utterable OK (%d paths)", len(path_strings))


# ---------------------------------------------------------------------------
# Assertion 2: Graph membership
# ---------------------------------------------------------------------------

def assert_graph_membership(all_paths: list, nx_graph) -> None:
    """Assert that every edge in every path exists in the graph.

    This verifies that DFS produced paths that are actually in the KG.
    If this fails, the path enumeration is hallucinating edges.
    """
    for path in all_paths:
        for head, rel, tail in path:
            assert nx_graph.has_edge(head, tail), (
                f"Graph membership failed: edge ({head}, {rel}, {tail}) "
                f"not in graph"
            )
            actual_rel = nx_graph[head][tail]["relation"]
            assert actual_rel == rel, (
                f"Graph membership failed: edge ({head}, {rel}, {tail}) "
                f"has relation {actual_rel!r} in graph"
            )

    logger.debug("Invariant: graph_membership OK (%d paths, %d edges checked)",
                 len(all_paths), sum(len(p) for p in all_paths))


# ---------------------------------------------------------------------------
# Assertion 3: Filtered ⊆ all_paths
# ---------------------------------------------------------------------------

def assert_filtered_subset(filtered_paths: list, all_paths: list) -> None:
    """Assert that filtered_paths is a subset of all_paths.

    This verifies the mathematical property that filtering only removes,
    never adds.  If this fails, the filter has a bug.
    """
    all_set = set(tuple(p) for p in all_paths)
    for p in filtered_paths:
        assert tuple(p) in all_set, (
            f"Filtered subset failed: path {p} not in all_paths"
        )

    logger.debug("Invariant: filtered_subset OK (%d ⊆ %d)",
                 len(filtered_paths), len(all_paths))


# ---------------------------------------------------------------------------
# Assertion 4: All admissible
# ---------------------------------------------------------------------------

def assert_all_admissible(
    filtered_paths: list,
    oracle,
    answer_types,
    max_hop: int,
) -> None:
    """Assert that every filtered path passes all TypeOracle gates.

    This verifies that the filtering logic is correct: every path that
    survives filtering is admissible.  If this fails, the gate logic
    has a bug.
    """
    for path in filtered_paths:
        for _, rel, tail in path:
            assert oracle.range_gate(rel, tail), (
                f"Admissibility failed: range_gate({rel!r}, {tail!r}) = False "
                f"on filtered path {path}"
            )
        terminal = path[-1][2]
        assert oracle.type_gate(terminal, answer_types, len(path), max_hop), (
            f"Admissibility failed: type_gate({terminal!r}, ...) = False "
            f"on filtered path {path}"
        )

    logger.debug("Invariant: all_admissible OK (%d paths)", len(filtered_paths))


# ---------------------------------------------------------------------------
# Assertion 5: No hallucinated entities
# ---------------------------------------------------------------------------

def assert_no_hallucinated_entities(all_paths: list, nx_graph) -> None:
    """Assert that every entity in every path is a node in the graph.

    This verifies that DFS did not invent entities.  If this fails,
    the path enumeration is hallucinating nodes.
    """
    graph_nodes = set(nx_graph.nodes())
    for path in all_paths:
        for head, _, tail in path:
            assert head in graph_nodes, (
                f"No hallucinated entities: {head!r} not in graph nodes"
            )
            assert tail in graph_nodes, (
                f"No hallucinated entities: {tail!r} not in graph nodes"
            )

    logger.debug("Invariant: no_hallucinated_entities OK (%d paths)",
                 len(all_paths))


# ---------------------------------------------------------------------------
# Run all invariants for a given condition
# ---------------------------------------------------------------------------

def check_all_invariants(
    tokenizer,
    trie,
    path_strings: List[str],
    all_paths: list,
    filtered_paths: list,
    nx_graph,
    oracle,
    answer_types,
    max_hop: int,
    cond_name: str,
) -> None:
    """Run all applicable invariants for a given condition.

    Call this after building the trie and before generation.
    Logs warnings instead of raising on failure (non-fatal).
    """
    checks = 0
    failures = 0

    def _run(name, fn, *args, **kwargs):
        nonlocal checks, failures
        checks += 1
        try:
            fn(*args, **kwargs)
        except AssertionError as e:
            failures += 1
            logger.error("INVARIANT [%s] %s FAILED: %s", cond_name, name, e)
        except Exception as e:
            failures += 1
            logger.error("INVARIANT [%s] %s ERROR: %s", cond_name, name, e)

    _run("trie_utterable", assert_trie_utterable, tokenizer, trie, path_strings)
    _run("graph_membership", assert_graph_membership, all_paths, nx_graph)
    _run("no_hallucinated_entities", assert_no_hallucinated_entities, all_paths, nx_graph)

    if filtered_paths is not None:
        _run("filtered_subset", assert_filtered_subset, filtered_paths, all_paths)
        _run("all_admissible", assert_all_admissible,
             filtered_paths, oracle, answer_types, max_hop)

    if failures:
        logger.warning("Invariant check: %d/%d FAILED for %s", failures, checks, cond_name)
    else:
        logger.debug("Invariant check: %d/%d passed for %s", checks, checks, cond_name)
