"""
run_experiment.py — Evaluate learned path reranking for KGQA.

Pipeline:
  1. Load N questions from RoG-cwq
  2. For each question, run full DFS enumeration → all paths
  3. Label each path: terminal entity matches answer? (0/1)
  4. Use CrossEncoder (sentence-transformers) to score (question, path) pairs
  5. Measure recall@K: does any gold path appear in top-K scored paths?
  6. Compare: random ordering, semantic similarity, fine-tuned (if available)

Usage:
  python experiments/learned_pruning/run_experiment.py \
    --max-samples 50 --dataset RoG-cwq --index-len 4 \
    --k 5 10 50 100 500
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "type_oracle_full"))

import src.utils as graph_utils
from utils import logger, PATH_START, PATH_END


def extract_answer_from_path(path_str: str) -> str:
    """Extract the terminal entity from a path string."""
    segments = [s.strip() for s in path_str.split("->")]
    return segments[-1].strip() if segments else ""


def path_answers_question(path_str: str, answers: Set[str]) -> bool:
    terminal = extract_answer_from_path(path_str)
    if not terminal or not answers:
        return False
    tl = terminal.lower()
    return any(a.lower() == tl or a.lower() in tl or tl in a.lower()
               for a in answers)


def generate_paths(question_dict, index_len: int) -> Tuple[List[str], str]:
    """Generate all DFS paths for a question. Returns (path_strings, question_text)."""
    g = graph_utils.build_graph(question_dict["graph"], undirected=False)
    entities = question_dict.get("q_entity", [])
    q_text = question_dict.get("question", "")

    if not entities:
        return [], q_text

    all_paths = graph_utils.dfs(g, entities, index_len)
    path_strs = [graph_utils.path_to_string(p) for p in all_paths]
    return path_strs, q_text


def evaluate_recall(ranked: List[Tuple[int, str, float]],
                    gold_indices: Set[int],
                    ks: List[int]) -> dict:
    """Compute recall@K: does any gold path appear in top-K?"""
    results = {}
    for k in ks:
        topk_indices = {idx for idx, _, _ in ranked[:k]}
        hits = len(gold_indices & topk_indices)
        results[f"recall@{k}"] = hits / max(1, len(gold_indices))
    return results


def evaluate_random(paths: List[str], gold_indices: Set[int],
                    ks: List[int], n_trials: int = 100) -> dict:
    """Random baseline: average recall over random permutations."""
    import random
    results = {f"recall@{k}": 0.0 for k in ks}
    n_gold = len(gold_indices)
    if n_gold == 0:
        return results
    for _ in range(n_trials):
        shuffled = list(range(len(paths)))
        random.shuffle(shuffled)
        for k in ks:
            topk = set(shuffled[:k])
            hits = len(gold_indices & topk)
            results[f"recall@{k}"] += hits / n_gold
    for k in ks:
        results[f"recall@{k}"] = round(results[f"recall@{k}"] / n_trials, 4)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=30)
    parser.add_argument("--dataset", default="RoG-cwq")
    parser.add_argument("--split", default="test")
    parser.add_argument("--index-len", type=int, default=4)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 50, 100, 500])
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    from datasets import load_dataset
    from reranker import PathReranker

    ts = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"results/learned_pruning_{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_path = output_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(str(log_path)),
                  logging.StreamHandler(sys.stdout)],
    )

    logger.info("Loading dataset: %s/%s", args.dataset, args.split)
    dataset = load_dataset(f"rmanluo/{args.dataset}", split=args.split)
    n = min(args.max_samples, len(dataset))
    dataset = dataset.select(range(n))
    logger.info("Samples: %d", n)

    reranker = PathReranker(args.model)

    all_results = []
    total_paths = 0
    total_with_gold = 0
    cumulative_recall = {f"recall@{k}": 0.0 for k in args.k}
    cumulative_random = {f"recall@{k}": 0.0 for k in args.k}

    t_start = time.time()

    for idx, d in enumerate(dataset):
        qid = d["id"]
        gt = set(d.get("answer", []))
        if not gt:
            continue

        paths, q_text = generate_paths(d, args.index_len)
        if not paths:
            logger.debug("  [%d/%d] %s: no paths", idx + 1, n, qid)
            continue

        # Label gold paths
        gold_indices = {i for i, p in enumerate(paths)
                        if path_answers_question(p, gt)}
        n_gold = len(gold_indices)
        total_paths += len(paths)
        total_with_gold += 1 if n_gold > 0 else 0

        if n_gold == 0:
            logger.debug("  [%d/%d] %s: 0/%d gold paths (gt=%s)",
                         idx + 1, n, qid, len(paths), gt)
            continue

        # Score and rank
        t0 = time.time()
        ranked = reranker.rank(q_text, paths)
        t_elapsed = time.time() - t0

        recalls = evaluate_recall(ranked, gold_indices, args.k)
        random_recalls = evaluate_random(paths, gold_indices, args.k)

        for k in args.k:
            cumulative_recall[f"recall@{k}"] += recalls[f"recall@{k}"]
            cumulative_random[f"recall@{k}"] += random_recalls[f"recall@{k}"]

        result = {
            "id": qid,
            "question": q_text,
            "n_paths": len(paths),
            "n_gold": n_gold,
            "gold_pct": round(n_gold / max(1, len(paths)) * 100, 1),
            "scoring_time_s": round(t_elapsed, 2),
            **{f"recall@{k}": recalls[f"recall@{k}"] for k in args.k},
            **{f"random@{k}": random_recalls[f"recall@{k}"] for k in args.k},
        }
        all_results.append(result)

        if (idx + 1) % 5 == 0:
            avg_recall = {f"r@{k}": round(cumulative_recall[f"recall@{k}"] / max(1, len(all_results)), 3)
                          for k in args.k}
            avg_random = {f"rand@{k}": round(cumulative_random[f"recall@{k}"] / max(1, len(all_results)), 3)
                          for k in args.k}
            logger.info("  [%d/%d] %d questions | avg paths=%d | %s | %s",
                        idx + 1, n, len(all_results),
                        total_paths // max(1, len(all_results)),
                        avg_recall, avg_random)

    # Final metrics
    elapsed = time.time() - t_start
    n_q = len(all_results)
    logger.info("")
    logger.info("=" * 70)
    logger.info("  LEARNED PRUNING: ZERO-SHOT PATH RERANKING RESULTS")
    logger.info("=" * 70)
    logger.info("  Dataset: %s  Samples: %d  Index len: %d", args.dataset, n, args.index_len)
    logger.info("  Questions with gold paths: %d/%d", total_with_gold, n_q)
    logger.info("  Avg paths per question: %d", total_paths // max(1, n_q))
    logger.info("  Total time: %.1fs  Avg: %.2fs/q", elapsed, elapsed / max(1, n_q))
    logger.info("")
    logger.info("  %-15s %-12s %-12s %s", "Metrics", "Reranker", "Random", "Improvement")
    logger.info("  " + "-" * 55)
    for k in args.k:
        avg = cumulative_recall[f"recall@{k}"] / max(1, n_q)
        rnd = cumulative_random[f"recall@{k}"] / max(1, n_q)
        imp = avg - rnd
        logger.info("  %-15s %-12.3f %-12.3f %+.3f",
                    f"Recall@{k}", avg, rnd, imp)

    # Save results
    summary = {
        "config": {
            "dataset": args.dataset,
            "max_samples": n,
            "index_len": args.index_len,
            "model": args.model,
            "k_values": args.k,
        },
        "summary": {
            "n_questions": n_q,
            "questions_with_gold": total_with_gold,
            "avg_paths": total_paths // max(1, n_q),
            "total_time_s": round(elapsed, 1),
            "avg_time_per_q": round(elapsed / max(1, n_q), 2),
        },
        "metrics": {
            f"recall@{k}": round(cumulative_recall[f"recall@{k}"] / max(1, n_q), 4)
            for k in args.k
        },
        "random_baseline": {
            f"recall@{k}": round(cumulative_random[f"recall@{k}"] / max(1, n_q), 4)
            for k in args.k
        },
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "per_question.jsonl", "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    logger.info("")
    logger.info("Results: %s", output_dir)
    return summary


if __name__ == "__main__":
    main()
