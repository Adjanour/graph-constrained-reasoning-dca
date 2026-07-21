"""
ontology_reasoner.py — ORT-style ontology reverse reasoning for DCA-Trie.

Addresses the O(E^L) path explosion problem by reversing over the
ontology label graph BEFORE entity-level enumeration.

Pipeline:
    1. Build label adjacency from TypeOracle's _RELATION_SCHEMA
    2. Reverse-reason from aim label → condition label via ontology
    3. Constrain entity-level DFS to follow only type-compatible entities
    4. Build trie from surviving paths

Usage:
    reasoner = OntologyReasoner(oracle)
    paths = reasoner.constrained_dfs(nx_graph, start_entities, index_len, aim_labels, condition_labels)
"""

import logging
from typing import Set, FrozenSet, Dict, List, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger("type_oracle")


class OntologyReasoner:
    """
    Builds a label-level ontology graph and uses it to prune entity-level search.

    The ontology adjacency is derived from the TypeOracle's _RELATION_SCHEMA:
    each relation's domain→range pair defines an edge between type labels.
    """

    def __init__(self, oracle):
        self._oracle = oracle
        self._label_adj: Dict[str, Set[str]] = defaultdict(set)
        self._relation_labels: Dict[str, str] = {}
        self._label_to_relations: Dict[str, Set[str]] = defaultdict(set)
        self._build_label_adjacency()

    def _build_label_adjacency(self):
        """Build undirected label graph from all relation schema entries."""
        schema = getattr(self._oracle, "_schema", {})
        if not schema:
            schema = getattr(self._oracle, "_RELATION_SCHEMA", {})
        if callable(schema) and not isinstance(schema, dict):
            schema = {}

        for rel_name, spec in schema.items():
            domain_types: FrozenSet[str] = spec.get("domain", frozenset())
            range_types: FrozenSet[str] = spec.get("range", frozenset())
            for d in domain_types:
                for r in range_types:
                    self._label_adj[d].add(r)
                    self._label_adj[r].add(d)
                    self._relation_labels[(d, r)] = rel_name
                    self._label_to_relations[d].add(rel_name)
                    self._label_to_relations[r].add(rel_name)

        logger.debug("Ontology: %d labels, %d edges",
                     len(self._label_adj),
                     sum(len(v) for v in self._label_adj.values()) // 2)

    def reverse_reason(self, aim_labels: FrozenSet[str],
                       condition_labels: FrozenSet[str],
                       max_depth: int = 4) -> List[List[str]]:
        """
        ORT-style reverse reasoning: find label paths from aim → condition.

        Builds a tree rooted at each aim label, expands via label adjacency,
        prunes to paths that contain condition labels, returns forward paths.

        Returns list of label paths [l1, l2, ..., ln] from condition → aim.
        """
        all_paths = set()
        for aim in aim_labels:
            self._expand_reverse(aim, aim, set(), condition_labels, 0, max_depth, all_paths, [aim])

        result = [list(reversed(p)) for p in all_paths]
        logger.debug("Reverse reasoning: %d aim labels → %d label paths",
                     len(aim_labels), len(result))
        return result

    def _expand_reverse(self, root_label, current_label, visited, condition_labels,
                        depth, max_depth, all_paths, current_path):
        """Recursive reverse expansion (ORT-style upFind)."""
        if current_label in condition_labels:
            all_paths.add(tuple(current_path))

        if depth >= max_depth:
            return

        visited = visited | {current_label}
        for neighbor in self._label_adj.get(current_label, set()):
            if neighbor not in visited:
                self._expand_reverse(
                    root_label, neighbor, visited, condition_labels,
                    depth + 1, max_depth, all_paths,
                    current_path + [neighbor]
                )

    def constrained_dfs(self, nx_graph, start_entities: List[str],
                        index_len: int,
                        aim_labels: FrozenSet[str],
                        condition_labels: Optional[Set[str]] = None) -> List[List[Tuple]]:
        """
        DFS constrained by label paths: only expand entities whose types
        match the expected label at each hop.

        This avoids enumerating all O(E^L) paths by pruning at expansion time.
        """
        label_paths = self.reverse_reason(aim_labels, frozenset(condition_labels or set()), index_len)

        if not label_paths:
            return []

        all_paths = set()
        for start in start_entities:
            if start not in nx_graph:
                continue
            start_types = self._oracle.get_types(start)

            for lp in label_paths:
                if condition_labels and start_types:
                    expected = lp[0] if lp else None
                    if expected and not (start_types & {expected}):
                        continue

                self._dfs_label_constrained(
                    nx_graph, start, [], lp, 0, index_len, all_paths
                )

        return list(all_paths)

    def _dfs_label_constrained(self, graph, node, path, label_path,
                               label_idx, max_len, all_paths):
        """DFS that only follows entities matching the current label path position."""
        if len(path) >= max_len:
            return

        for neighbor in graph.neighbors(node):
            rel = graph[node][neighbor]["relation"]
            neighbor_types = self._oracle.get_types(neighbor)

            if not neighbor_types:
                new_path = path + [(node, rel, neighbor)]
                if len(new_path) <= max_len:
                    all_paths.add(tuple(new_path))
                self._dfs_label_constrained(
                    graph, neighbor, new_path, label_path,
                    label_idx, max_len, all_paths
                )
                continue

            next_label = label_path[min(label_idx + 1, len(label_path) - 1)]
            if not (neighbor_types & {next_label}):
                continue

            new_path = path + [(node, rel, neighbor)]
            if len(new_path) <= max_len:
                all_paths.add(tuple(new_path))

            next_idx = min(label_idx + 1, len(label_path) - 1)
            self._dfs_label_constrained(
                graph, neighbor, new_path, label_path,
                next_idx, max_len, all_paths
            )

    def compute_path_reduction(self, all_paths_count: int,
                               constrained_count: int) -> float:
        if all_paths_count == 0:
            return 0.0
        return (1 - constrained_count / all_paths_count) * 100
