"""
ontology_reasoner.py — ORT-style ontology reverse reasoning for DCA-Trie.

Addresses the O(E^L) path explosion problem by reversing over the
ontology label graph BEFORE entity-level enumeration.

Pipeline:
    1. Build category-level label adjacency from TypeOracle's _RELATION_SCHEMA
    2. Reverse-reason from aim category → condition category (capped)
    3. Constrain entity-level DFS to follow only category-compatible entities
"""

import logging
from typing import Set, FrozenSet, Dict, List, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger("type_oracle")

# Map granular Freebase types to broad categories (must match intent_parser.py)
_LABEL_TO_CATEGORY: Dict[str, str] = {
    "Person": "Person", "Deceased Person": "Person", "Politician": "Person",
    "Musical Artist": "Person", "Author": "Person", "Film director": "Person",
    "Film actor": "Person", "TV Actor": "Person", "Composer": "Person",
    "Singer": "Person", "Military Person": "Person", "Inventor": "Person",
    "Olympic athlete": "Person", "Professional Athlete": "Person",
    "Religious Leader": "Person",
    "Location": "Location", "Country": "Location", "City/Town/Village": "Location",
    "US State": "Location", "US County": "Location",
    "Administrative Division": "Location", "Venue": "Location",
    "Place of interment": "Location",
    "Organization": "Organization", "Company": "Organization",
    "Educational Institution": "Organization", "Government Agency": "Organization",
    "Sports Team": "Organization", "College/University": "Organization",
    "School": "Organization",
    "Film": "CreativeWork", "Book": "CreativeWork", "TV Program": "CreativeWork",
    "Written Work": "CreativeWork", "Musical Album": "CreativeWork",
    "Musical Recording": "CreativeWork", "Play": "CreativeWork",
    "TV Series": "CreativeWork", "TV Season": "CreativeWork",
    "Composition": "CreativeWork",
    "Film performance": "CreativeWork",
    "Award": "Award", "Award honor": "Award",
    "Ethnicity": "Miscellaneous",
    "Academic": "Organization",
    "Sports Facility": "Facility",
}

# Inverse: category → set of labels
_CATEGORY_TO_LABELS: Dict[str, Set[str]] = defaultdict(set)
for lbl, cat in _LABEL_TO_CATEGORY.items():
    _CATEGORY_TO_LABELS[cat].add(lbl)

KNOWN_CATEGORIES: Set[str] = set(_CATEGORY_TO_LABELS.keys())


class OntologyReasoner:
    """
    Builds a category-level ontology graph and uses it to prune entity-level search.
    
    The ontology adjacency is derived from TypeOracle's _RELATION_SCHEMA but
    aggregated to broad categories (12 vs 46 labels), keeping the graph sparse.
    """

    def __init__(self, oracle, max_label_paths: int = 200):
        self._oracle = oracle
        self._max_label_paths = max_label_paths
        self._cat_adj: Dict[str, Set[str]] = defaultdict(set)
        self._build_category_adjacency()

    def _get_categories(self, type_set: FrozenSet[str]) -> Set[str]:
        if not type_set:
            return {"Miscellaneous"}
        cats: Set[str] = set()
        for t in type_set:
            c = _LABEL_TO_CATEGORY.get(t)
            if c:
                cats.add(c)
        return cats or {"Miscellaneous"}

    def _build_category_adjacency(self):
        schema = getattr(self._oracle, "_schema", {})
        if not schema:
            schema = getattr(self._oracle, "_RELATION_SCHEMA", {})
        if callable(schema) and not isinstance(schema, dict):
            schema = {}

        for rel_name, spec in schema.items():
            domain_types: FrozenSet[str] = spec.get("domain", frozenset())
            range_types: FrozenSet[str] = spec.get("range", frozenset())
            if domain_types and range_types:
                domain_cats = self._get_categories(domain_types)
                range_cats = self._get_categories(range_types)
                for dc in domain_cats:
                    for rc in range_cats:
                        self._cat_adj[dc].add(rc)

        logger.debug("Category graph: %d categories, %d edges (from %d relations)",
                     len(self._cat_adj),
                     sum(len(v) for v in self._cat_adj.values()),
                     len(schema))

    def reverse_reason(self, aim_labels: FrozenSet[str],
                       condition_labels: FrozenSet[str],
                       max_depth: int = 4) -> List[List[str]]:
        aim_cats = self._get_categories(aim_labels)
        cond_cats = self._get_categories(condition_labels)

        if not aim_cats or not cond_cats:
            return []

        all_paths: Set[Tuple[str, ...]] = set()
        for aim in aim_cats:
            if len(all_paths) >= self._max_label_paths:
                break
            self._expand_reverse(aim, set(), cond_cats, 0,
                                 max_depth, all_paths, [aim])

        result = [list(reversed(p)) for p in all_paths]
        logger.debug("Reverse reason: %s → %s → %d category paths (capped %d)",
                     aim_cats, cond_cats, len(result), self._max_label_paths)
        return result[:self._max_label_paths]

    def _expand_reverse(self, current_cat, visited, cond_cats,
                        depth, max_depth, all_paths, current_path):
        if len(all_paths) >= self._max_label_paths:
            return

        if current_cat in cond_cats:
            all_paths.add(tuple(current_path))

        if depth >= max_depth:
            return

        visited = visited | {current_cat}
        for neighbor in self._cat_adj.get(current_cat, set()):
            if len(all_paths) >= self._max_label_paths:
                return
            if neighbor not in visited:
                self._expand_reverse(
                    neighbor, visited, cond_cats,
                    depth + 1, max_depth, all_paths,
                    current_path + [neighbor]
                )

    def constrained_dfs(self, nx_graph, start_entities: List[str],
                        index_len: int,
                        aim_labels: FrozenSet[str],
                        condition_labels: Optional[Set[str]] = None
                        ) -> Tuple[List[List[Tuple]], List[List[str]]]:
        cond_set = condition_labels or set()
        category_paths = self.reverse_reason(
            aim_labels, frozenset(cond_set), index_len
        )

        if not category_paths:
            logger.debug("constrained_dfs: no category paths found (aim=%s cond=%s)",
                         aim_labels, cond_set)
            return [], []

        all_paths: Set[Tuple] = set()
        for start in start_entities:
            if start not in nx_graph:
                continue

            for cp in category_paths:
                if len(all_paths) >= 50000:
                    break
                self._dfs_cat_constrained(
                    nx_graph, start, [], cp, 0, index_len,
                    all_paths
                )

        logger.debug("constrained_dfs: %d category paths → %d entity paths",
                     len(category_paths), len(all_paths))
        return list(all_paths), category_paths

    def _dfs_cat_constrained(self, graph, node, path, cat_path,
                             cat_idx, max_len, all_paths):
        if len(all_paths) >= 50000:
            return
        if len(path) >= max_len:
            return

        for neighbor in graph.neighbors(node):
            if len(all_paths) >= 50000:
                return
            neighbor_types = self._oracle.get_types(neighbor)
            neighbor_cats = self._get_categories(neighbor_types)

            next_cat = cat_path[min(cat_idx + 1, len(cat_path) - 1)]
            if neighbor_cats and next_cat not in neighbor_cats:
                continue

            new_path = path + [(node, graph[node][neighbor]["relation"], neighbor)]
            if len(new_path) <= max_len:
                all_paths.add(tuple(new_path))

            next_idx = min(cat_idx + 1, len(cat_path) - 1)
            self._dfs_cat_constrained(
                graph, neighbor, new_path, cat_path,
                next_idx, max_len, all_paths
            )
