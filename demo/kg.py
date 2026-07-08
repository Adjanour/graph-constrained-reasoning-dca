"""Knowledge graph operations for the demo.

Builds NetworkX graphs from triples, runs DFS path enumeration,
and applies TypeOracle filtering.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import networkx as nx

# Add project root to path for TypeOracle import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from approach3_symbolic.type_oracle import TypeOracle


def build_graph(triples: List[Tuple[str, str, str]], undirected: bool = True) -> nx.DiGraph:
    """Build a graph from (subject, predicate, object) triples.

    Args:
        triples: List of (subject, predicate, object) triples.
        undirected: If True, add edges in both directions.

    Returns:
        NetworkX graph.
    """
    if undirected:
        G = nx.Graph()
    else:
        G = nx.DiGraph()
    for s, p, o in triples:
        G.add_edge(s.strip(), o.strip(), relation=p.strip())
        if undirected:
            G.add_edge(o.strip(), s.strip(), relation=p.strip())
    return G


def get_neighbors(G: nx.DiGraph, entity: str, max_hops: int = 2) -> List[Tuple[str, str, str]]:
    """Get all reachable triples from an entity within max_hops."""
    triples = []
    visited = set()
    queue = [(entity, 0)]

    while queue:
        current, depth = queue.pop(0)
        if depth >= max_hops or current in visited:
            continue
        visited.add(current)

        if current in G:
            for neighbor in G.neighbors(current):
                rel = G[current][neighbor]["relation"]
                triples.append((current, rel, neighbor))
                queue.append((neighbor, depth + 1))

    return triples


def enumerate_paths(
    G: nx.DiGraph,
    start_entities: List[str],
    max_length: int = 2,
) -> List[List[Tuple[str, str, str]]]:
    """Enumerate all paths from start entities up to max_length."""
    paths = set()

    def dfs(node: str, path: List[Tuple[str, str, str]]):
        if len(path) > max_length:
            return
        if path:
            paths.add(tuple(path))
        if node in G:
            for neighbor in G.neighbors(node):
                rel = G[node][neighbor]["relation"]
                new_path = path + [(node, rel, neighbor)]
                if len(new_path) <= max_length:
                    dfs(neighbor, new_path)

    for entity in start_entities:
        dfs(entity, [])

    return [list(p) for p in paths]


def filter_paths_with_oracle(
    paths: List[List[Tuple[str, str, str]]],
    oracle: TypeOracle,
    question: str,
    answer_types: set,
) -> List[List[Tuple[str, str, str]]]:
    """Filter paths using TypeOracle range and type gates."""
    filtered = []
    for path in paths:
        admit = True
        for _, rel, tail in path:
            if not oracle.range_gate(rel, tail):
                admit = False
                break
        if admit and path:
            terminal = path[-1][2]
            if not oracle.type_gate(terminal, answer_types, len(path), len(path)):
                admit = False
        if admit:
            filtered.append(path)
    return filtered


def format_path(path: List[Tuple[str, str, str]]) -> str:
    """Format a path as a human-readable string."""
    if not path:
        return ""
    parts = []
    for s, p, o in path:
        parts.append(f"{s} --[{p}]--> {o}")
    return "\n".join(parts)


def format_path_compact(path: List[Tuple[str, str, str]]) -> str:
    """Format a path as a compact string."""
    if not path:
        return ""
    triples = [f"({s}, {p}, {o})" for s, p, o in path]
    return " -> ".join(triples)
