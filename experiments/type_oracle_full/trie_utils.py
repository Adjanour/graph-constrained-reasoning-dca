"""
trie_utils.py — MarisaTrie builders for DCA-Trie experiments.

Provides four constructors:
- ``build_unfiltered_trie`` → all DFS paths (GCR baseline)
- ``build_filtered_trie``   → TypeOracle-gated paths (DCA v1 static)
- ``build_trie_from_strings`` → from raw path strings (DCA v2 dynamic)
- ``build_dict_trie`` → Python dict trie for per-beam dynamic construction (DoG-style)
"""

from functools import reduce
from typing import Dict, List, Optional, Set

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
    if not ans_types:
        ans_types = oracle.infer_answer_types_from_paths(all_paths)

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
    tokenized = [ids + [tokenizer.eos_token_id] for ids in tokenized]
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
    tokenized = [ids + [tokenizer.eos_token_id] for ids in tokenized]
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


def build_trie_from_token_ids(tokenizer, token_ids):
    """Build a MarisaTrie from pre-tokenized ID sequences (with EOS already appended).

    This is the P4 optimization: tokenize once, build tries from the same
    token IDs across baseline/v1/v2.
    """
    if not token_ids:
        return None
    return MarisaTrie(token_ids, max_token_id=len(tokenizer) + 1)


def build_dict_trie(tokenizer, path_strings: List[str]) -> Optional[Dict[int, dict]]:
    """
    Build a Python dict trie from path strings (DoG-style dynamic construction).

    This is used for per-beam trie construction in v2 iterative decoding.
    The trie maps token IDs to child dicts, enabling O(1) lookup per token.

    Each complete path is added to the trie. The prefix lookup traverses
    the trie with the prefix and returns valid next tokens.

    Parameters
    ----------
    tokenizer : tokenizer
        HuggingFace tokenizer.
    path_strings : list of str
        Path strings like "entity1 -> relation -> entity2".

    Returns
    -------
    dict or None
        Python dict trie, or None if path_strings is empty.
    """
    if not path_strings:
        return None

    wrapped = [f"{PATH_START}{s}{PATH_END}" for s in path_strings]
    tokenized = tokenizer(wrapped, padding=False, add_special_tokens=False).input_ids
    tokenized = [ids + [tokenizer.eos_token_id] for ids in tokenized]

    trie = {}
    for token_seq in tokenized:
        node = trie
        for token in token_seq:
            if token not in node:
                node[token] = {}
            node = node[token]
    return trie


def dict_trie_get(trie: Dict[int, dict], prefix: List[int]) -> List[int]:
    """
    Get valid next tokens from a Python dict trie given a prefix.

    Parameters
    ----------
    trie : dict
        Python dict trie.
    prefix : list of int
        Token IDs so far.

    Returns
    -------
    list of int
        Valid next token IDs.
    """
    node = reduce(lambda d, k: d.get(k, {}), prefix, trie)
    return list(node.keys())
