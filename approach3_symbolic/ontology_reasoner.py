"""
ontology_reasoner.py — ORT-style ontology reverse reasoning for DCA-Trie.

Addresses the O(E^L) path explosion problem by reversing over the
ontology label graph BEFORE entity-level enumeration.

Pipeline:
    1. Build directed label adjacency from TypeOracle's _RELATION_SCHEMA
    2. Reverse-reason from aim label → condition label via ontology (capped)
    3. Constrain entity-level DFS to follow only type-compatible entities
    4. Build trie from surviving paths
"""

import logging
from typing import Set, FrozenSet, Dict, List, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger("type_oracle")


class OntologyReasoner:
    """
    Builds a label-level ontology graph and uses it to prune entity-level search.

    The ontology adjacency is derived from the TypeOracle's _RELATION_SCHEMA.
    Unlike the full cross-product approach, we use a **type-set-level**
    adjacency to keep the label graph sparse and tractable.
    """

    def __init__(self, oracle, max_label_paths: int = 200):
        self._oracle = oracle
        self._max_label_paths = max_label_paths
        self._label_adj: Dict[str, Set[str]] = defaultdict(set)
        self._build_label_adjacency()

    def _build_label_adjacency(self):
        """Build sparse directed label adjacency from relation schema.

        Instead of cross-producting domain×range (which creates dense edges),
        we treat each distinct domain-range pair as a single abstract edge.
        This mimics ORT's hand-curated ontology graph.
        """
        schema = getattr(self._oracle, "_schema", {})
        if not schema:
            schema = getattr(self._oracle, "_RELATION_SCHEMA", {})
        if callable(schema) and not isinstance(schema, dict):
            schema = {}

        edges: Set[Tuple[str, str]] = set()
        for rel_name, spec in schema.items():
            domain_types: FrozenSet[str] = spec.get("domain", frozenset())
            range_types: FrozenSet[str] = spec.get("range", frozenset())
            if domain_types and range_types:
                d_repr = str(sorted(domain_types))
                r_repr = str(sorted(range_types))
                edges.add((d_repr, r_repr))
                for d in domain_types:
                    for r in range_types:
                        self._label_adj[d].add(r)

        logger.debug("Ontology: %d labels, %d edges (from %d relations)",
                     len(self._label_adj),
                     sum(len(v) for v in self._label_adj.values()),
                     len(schema))

    def reverse_reason(self, aim_labels: FrozenSet[str],
                       condition_labels: FrozenSet[str],
                       max_depth: int = 4) -> List[List[str]]:
        all_paths: Set[Tuple[str, ...]] = set()
        for aim in aim_labels:
            if len(all_paths) >= self._max_label_paths:
                break
            self._expand_reverse(aim, set(), condition_labels, 0,
                                 max_depth, all_paths, [aim])

        result = [list(reversed(p)) for p in all_paths]
        logger.debug("Reverse reasoning: %d aim labels → %d label paths (capped at %d)",
                     len(aim_labels), len(result), self._max_label_paths)
        return result[:self._max_label_paths]

    def _expand_reverse(self, current_label, visited, condition_labels,
                        depth, max_depth, all_paths, current_path):
        if len(all_paths) >= self._max_label_paths:
            return

        if current_label in condition_labels:
            all_paths.add(tuple(current_path))

        if depth >= max_depth:
            return

        visited = visited | {current_label}
        for neighbor in self._label_adj.get(current_label, set()):
            if len(all_paths) >= self._max_label_paths:
                return
            if neighbor not in visited:
                self._expand_reverse(
                    neighbor, visited, condition_labels,
                    depth + 1, max_depth, all_paths,
                    current_path + [neighbor]
                )

    def constrained_dfs(self, nx_graph, start_entities: List[str],
                        index_len: int,
                        aim_labels: FrozenSet[str],
                        condition_labels: Optional[Set[str]] = None
                        ) -> Tuple[List[List[Tuple]], List[List[str]]]:
        label_paths = self.reverse_reason(
            aim_labels, frozenset(condition_labels or set()), index_len
        )

        if not label_paths:
            return [], []

        all_paths: Set[Tuple] = set()
        applied_paths: Set[str] = set()
        for start in start_entities:
            if start not in nx_graph:
                continue

            for i, lp in enumerate(label_paths):
                if len(all_paths) >= 50000:
                    break
                tag = str(i)
                self._dfs_label_constrained(
                    nx_graph, start, [], lp, 0, index_len,
                    all_paths, tag, applied_paths
                )

        return list(all_paths), label_paths

    def _dfs_label_constrained(self, graph, node, path, label_path,
                               label_idx, max_len, all_paths,
                               path_tag, applied_paths):
        if len(all_paths) >= 50000:
            return
        if len(path) >= max_len:
            return

        for neighbor in graph.neighbors(node):
            if len(all_paths) >= 50000:
                return
            rel = graph[node][neighbor]["relation"]
            neighbor_types = self._oracle.get_types(neighbor)

            next_label = label_path[min(label_idx + 1, len(label_path) - 1)]
            if neighbor_types and not (neighbor_types & {next_label}):
                continue

            new_path = path + [(node, rel, neighbor)]
            if len(new_path) <= max_len:
                all_paths.add(tuple(new_path))

            next_idx = min(label_idx + 1, len(label_path) - 1)
            self._dfs_label_constrained(
                graph, neighbor, new_path, label_path,
                next_idx, max_len, all_paths, path_tag, applied_paths
            )
