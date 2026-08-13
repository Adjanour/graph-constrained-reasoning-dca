"""
bench_lazy.py — Constraint construction cost: static pre-compilation vs lazy.

No model weights are loaded; this measures constraint construction only, which
is where the two designs differ.  Static cost is paid up front for every
question whatever the beams do, so it is measured once.  Lazy cost is paid only
along the paths the beams walk, so it is measured over ``--beams`` random
descents.

    ./.venv/bin/python experiments/type_oracle_full/bench_lazy.py
    ./.venv/bin/python experiments/type_oracle_full/bench_lazy.py --index-len 4
"""

import argparse
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets import load_dataset
from transformers import AutoTokenizer

import src.utils as graph_utils
from src.trie import MarisaTrie

from approach3_symbolic.type_oracle import TypeOracle
from lazy_constraint import LazyGraphConstraint
from utils import PATH_END, PATH_START

DFS_CAP = 50000  # graph_utils.dfs max_paths


def static_cost(tokenizer, graph, start, max_hops):
    t0 = time.time()
    paths = graph_utils.dfs(graph, start, max_hops)
    wrapped = [
        f"{PATH_START}{graph_utils.path_to_string(p)}{PATH_END}" for p in paths
    ]
    ids = tokenizer(wrapped, padding=False, add_special_tokens=False).input_ids
    ids = [i + [tokenizer.eos_token_id] for i in ids]
    MarisaTrie(ids, max_token_id=len(tokenizer) + 1)
    return time.time() - t0, len(paths), sum(len(i) for i in ids)


def lazy_cost(constraint, start_id, eos_id, rng, n_walks):
    t0, calls = time.time(), 0
    for _ in range(n_walks):
        prefix = (start_id,)
        while prefix[-1] != eos_id and len(prefix) < 400:
            allowed = constraint.get(list(prefix))
            calls += 1
            if not allowed:
                raise SystemExit("lazy constraint hit a dead end")
            prefix = prefix + (rng.choice(allowed),)
    return time.time() - t0, calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rmanluo/RoG-webqsp")
    ap.add_argument("--split", default="train")
    ap.add_argument("--index-len", type=int, default=2)
    ap.add_argument("--beams", type=int, default=5)
    ap.add_argument("--n", type=int, default=8, help="questions to benchmark")
    ap.add_argument("--model-path", default="rmanluo/GCR-Meta-Llama-3.1-8B-Instruct")
    args = ap.parse_args()

    rng = random.Random(7)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    start_id = tokenizer.convert_tokens_to_ids(PATH_START)
    eos_id = tokenizer.eos_token_id
    ds = load_dataset(args.dataset, split=args.split)

    print(f"{args.dataset} {args.split} | index_len={args.index_len} | "
          f"lazy measured over {args.beams} descents\n")
    print(f"{'qid':<14} {'triples':>8} {'paths':>8} {'statTok':>10} {'statS':>7} "
          f"{'lazyCand':>9} {'lazyS':>7} {'us/call':>8} {'speedup':>8}")
    print("-" * 88)

    capped = 0
    for i in range(args.n):
        d = ds[i]
        graph = graph_utils.build_graph(d["graph"], undirected=False)
        start = [e for e in d["q_entity"] if e in graph]
        if not start:
            continue

        s_time, n_paths, n_tok = static_cost(
            tokenizer, graph, start, args.index_len
        )
        capped += n_paths >= DFS_CAP

        oracle = TypeOracle.from_graph(d["graph"])
        answer_types = oracle.infer_answer_types(d["question"])
        constraint = LazyGraphConstraint(
            tokenizer, graph, start, oracle, answer_types, args.index_len
        )
        l_time, calls = lazy_cost(constraint, start_id, eos_id, rng, args.beams)
        stats = constraint.stats()

        print(f"{d['id'][:14]:<14} {len(d['graph']):>8} "
              f"{n_paths:>7}{'*' if n_paths >= DFS_CAP else ' '} "
              f"{n_tok:>10} {s_time:>6.2f}s "
              f"{stats['candidates_materialised']:>9} {l_time:>6.2f}s "
              f"{1e6 * l_time / max(calls, 1):>8.0f} "
              f"{s_time / max(l_time, 1e-9):>7.1f}x")

    if capped:
        print(f"\n* {capped} question(s) hit the {DFS_CAP:,}-path DFS cap in "
              f"graph_utils.dfs — the static trie is a DFS-order truncation of "
              f"the path set at this index_len, not the whole of it.")


if __name__ == "__main__":
    main()
