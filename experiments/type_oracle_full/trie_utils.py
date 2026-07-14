"""
trie_utils.py — MarisaTrie builders for DCA-Trie experiments.

Provides three constructors:
- ``build_unfiltered_trie`` → all DFS paths (GCR baseline)
- ``build_filtered_trie``   → TypeOracle-gated paths (DCA v1 static)
- ``build_trie_from_strings`` → from raw path strings (DCA v2 dynamic)
"""

import src.utils as graph_utils
from src.trie import MarisaTrie

from utils import PATH_START, PATH_END


def build_filtered_trie(tokenizer, question_dict, index_len, oracle):
    """Build a MarisaTrie from TypeOracle-filtered paths (v1 static)."""
    g = graph_utils.build_graph(question_dict["graph"], undirected=False)
    entities = question_dict.get("q_entity", [])
    if not entities:
        return None, [], []

    all_paths = graph_utils.dfs(g, entities, index_len)
    ans_types = oracle.infer_answer_types(question_dict["question"])

    filtered = []
    for p in all_paths:
        admit = True
        for _, rel, tail in p:
            if not oracle.range_gate(rel, tail):
                admit = False
                break
        if admit:
            terminal = p[-1][2]
            if not oracle.type_gate(terminal, ans_types, len(p), index_len):
                admit = False
        if admit:
            filtered.append(p)

    filtered_str = [graph_utils.path_to_string(p) for p in filtered]
    if not filtered_str:
        return None, all_paths, filtered

    wrapped = [f"{PATH_START}{s}{PATH_END}" for s in filtered_str]
    tokenized = tokenizer(wrapped, padding=False, add_special_tokens=False).input_ids
    trie = MarisaTrie(tokenized, max_token_id=len(tokenizer) + 1)
    return trie, all_paths, filtered


def build_unfiltered_trie(tokenizer, question_dict, index_len):
    """Build a MarisaTrie from all DFS paths (GCR baseline)."""
    g = graph_utils.build_graph(question_dict["graph"], undirected=False)
    entities = question_dict.get("q_entity", [])
    if not entities:
        return None, []

    all_paths = graph_utils.dfs(g, entities, index_len)
    all_str = [graph_utils.path_to_string(p) for p in all_paths]
    if not all_str:
        return None, all_paths

    wrapped = [f"{PATH_START}{s}{PATH_END}" for s in all_str]
    tokenized = tokenizer(wrapped, padding=False, add_special_tokens=False).input_ids
    trie = MarisaTrie(tokenized, max_token_id=len(tokenizer) + 1)
    return trie, all_paths


def build_trie_from_strings(tokenizer, path_strings):
    """Build a MarisaTrie from raw path strings (for v2 iterative expansion)."""
    if not path_strings:
        return None
    wrapped = [f"{PATH_START}{s}{PATH_END}" for s in path_strings]
    tokenized = tokenizer(wrapped, padding=False, add_special_tokens=False).input_ids
    tokenized = [ids + [tokenizer.eos_token_id] for ids in tokenized]
    return MarisaTrie(tokenized, max_token_id=len(tokenizer) + 1)
