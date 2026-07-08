"""DCA-Trie pipeline for the demo.

Implements the core DCA-Trie logic:
1. Build graph from KG triples
2. Initialize TypeOracle from graph schema
3. Enumerate paths via DFS
4. Filter paths using TypeOracle gates
5. Return filtered paths as the constrained search space
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from approach3_symbolic.type_oracle import TypeOracle
from .kg import (
    build_graph,
    enumerate_paths,
    filter_paths_with_oracle,
    format_path,
    format_path_compact,
)


class DCATrieResult:
    """Result from the DCA-Trie pipeline."""

    def __init__(
        self,
        question: str,
        all_paths: List[List[Tuple[str, str, str]]],
        filtered_paths: List[List[Tuple[str, str, str]]],
        answer_types: set,
        graph: nx.DiGraph,
    ):
        self.question = question
        self.all_paths = all_paths
        self.filtered_paths = filtered_paths
        self.answer_types = answer_types
        self.graph = graph

    @property
    def total_paths(self) -> int:
        return len(self.all_paths)

    @property
    def kept_paths(self) -> int:
        return len(self.filtered_paths)

    @property
    def removed_paths(self) -> int:
        return self.total_paths - self.kept_paths

    @property
    def sir(self) -> float:
        """Semantic Irrelevance Ratio."""
        if self.total_paths == 0:
            return 0.0
        return self.removed_paths / self.total_paths

    @property
    def best_path(self) -> List[Tuple[str, str, str]] | None:
        """Return the most relevant filtered path."""
        if not self.filtered_paths:
            return None
        # Return shortest path (most direct)
        return min(self.filtered_paths, key=len)

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Question: {self.question}",
            f"Answer types: {', '.join(str(t) for t in list(self.answer_types)[:5])}...",
            f"",
            f"Path enumeration:",
            f"  Total paths found: {self.total_paths}",
            f"  Paths after filtering: {self.kept_paths}",
            f"  Paths removed: {self.removed_paths}",
            f"  SIR: {self.sir:.1%}",
            f"",
            f"All paths:",
        ]
        for i, path in enumerate(self.all_paths):
            lines.append(f"  {i+1}. {format_path_compact(path)}")
        lines.append(f"")
        lines.append(f"Filtered paths (admissible):")
        for i, path in enumerate(self.filtered_paths):
            lines.append(f"  {i+1}. {format_path_compact(path)}")
        return "\n".join(lines)


def run_dca_trie(
    question: str,
    q_entity: List[str],
    graph_triples: List[Tuple[str, str, str]],
    max_hops: int = 2,
) -> DCATrieResult:
    """Run the DCA-Trie pipeline on a question.

    Args:
        question: Natural language question.
        q_entity: Topic entities in the question.
        graph_triples: KG subgraph as (subject, predicate, object) triples.
        max_hops: Maximum number of hops for path enumeration.

    Returns:
        DCATrieResult with all paths, filtered paths, and metrics.
    """
    # Step 1: Build graph (undirected for path enumeration)
    G = build_graph(graph_triples, undirected=True)

    # Step 2: Initialize TypeOracle
    oracle = TypeOracle.from_graph(graph_triples)

    # Step 3: Infer answer types
    answer_types = oracle.infer_answer_types(question)

    # Step 4: Enumerate all paths from topic entities
    all_paths = enumerate_paths(G, q_entity, max_length=max_hops)

    # Step 5: Filter paths using TypeOracle
    filtered_paths = filter_paths_with_oracle(all_paths, oracle, question, answer_types)

    return DCATrieResult(
        question=question,
        all_paths=all_paths,
        filtered_paths=filtered_paths,
        answer_types=answer_types,
        graph=G,
    )
