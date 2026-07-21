"""
train_reranker.py — Fine-tune a bi-encoder path relevance scorer for KGQA.

Uses (question, gold_path) as positive pairs and (question, random_path) 
as negative pairs to train a sentence-transformer model.

Usage:
  # Generate training data from existing predictions + re-generate paths
  python experiments/learned_pruning/train_reranker.py \
    --train-samples 30 --dataset RoG-cwq --index-len 4 \
    --output-dir results/reranker_trained

  # Then evaluate
  python experiments/learned_pruning/run_experiment.py \
    --max-samples 20 --dataset RoG-cwq --index-len 4 \
    --k 1 5 10 50 100 \
    --model results/reranker_trained/model
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "type_oracle_full"))

import src.utils as graph_utils
from utils import logger

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def generate_paths(question_dict, index_len: int, max_paths: int = 0) -> Tuple[List[str], str, set]:
    g = graph_utils.build_graph(question_dict["graph"], undirected=False)
    entities = question_dict.get("q_entity", [])
    q_text = question_dict.get("question", "")
    gt = set(question_dict.get("answer", []))
    if not entities:
        return [], q_text, gt
    all_paths = graph_utils.dfs(g, entities, index_len, max_paths=max_paths if max_paths > 0 else 50000)
    path_strs = [graph_utils.path_to_string(p) for p in all_paths]
    return path_strs, q_text, gt


def extract_answer(path_str: str) -> str:
    segments = [s.strip() for s in path_str.split("->")]
    return segments[-1].strip() if segments else ""


def label_paths(paths: List[str], answers: set) -> List[int]:
    labels = []
    for p in paths:
        terminal = extract_answer(p).lower()
        is_gold = any(a.lower() == terminal or a.lower() in terminal or terminal in a.lower()
                      for a in answers)
        labels.append(1 if is_gold else 0)
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-samples", type=int, default=30)
    parser.add_argument("--dataset", default="RoG-cwq")
    parser.add_argument("--split", default="test")
    parser.add_argument("--index-len", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-neg-per-q", type=int, default=20,
                        help="Max negative samples per question")
    parser.add_argument("--max-paths", type=int, default=0,
                        help="Max paths per question (0 = unlimited / 50000)")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer, InputExample
    from sentence_transformers.sentence_transformer import losses

    ts = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"results/reranker_trained_{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset: %s/%s", args.dataset, args.split)
    dataset = load_dataset(f"rmanluo/{args.dataset}", split=args.split)
    n = min(args.train_samples, len(dataset))
    dataset = dataset.select(range(n))
    logger.info("Training samples: %d", n)

    # Generate training data
    train_examples = []
    stats = {"total": 0, "with_gold": 0, "total_pos": 0, "total_neg": 0}

    for idx, d in enumerate(dataset):
        paths, q_text, gt = generate_paths(d, args.index_len, max_paths=args.max_paths)
        if not paths or not gt:
            continue

        labels = label_paths(paths, gt)
        pos_indices = [i for i, l in enumerate(labels) if l == 1]
        neg_indices = [i for i, l in enumerate(labels) if l == 0]

        stats["total"] += 1
        stats["total_pos"] += len(pos_indices)

        for pi in pos_indices:
            train_examples.append(InputExample(
                texts=[q_text, paths[pi]], label=1.0
            ))

        # Sample negatives
        n_neg = min(args.max_neg_per_q, len(neg_indices))
        sampled_neg = random.sample(neg_indices, n_neg) if n_neg > 0 else []
        for ni in sampled_neg:
            train_examples.append(InputExample(
                texts=[q_text, paths[ni]], label=0.0
            ))

        if len(pos_indices) > 0:
            stats["with_gold"] += 1

        if (idx + 1) % 10 == 0:
            logger.info("  Generated %d examples from %d questions (pos=%d neg=%d)",
                        len(train_examples), stats["total"],
                        stats["total_pos"], len(train_examples) - stats["total_pos"])

    logger.info("Total training examples: %d (pos=%d neg=%d, questions=%d)",
                len(train_examples), stats["total_pos"],
                len(train_examples) - stats["total_pos"], stats["total"])

    if stats["total_pos"] < 5:
        logger.warning("Too few positive examples (%d). Training unlikely to work.",
                       stats["total_pos"])

    # Load model and train
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    logger.info("Loading model: %s", model_name)
    model = SentenceTransformer(model_name)

    logger.info("Training for %d epochs, batch_size=%d, lr=%f",
                args.epochs, args.batch_size, args.lr)
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        train_examples, shuffle=True, batch_size=args.batch_size
    )
    train_loss = losses.CosineSimilarityLoss(model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        evaluator=None,
        warmup_steps=int(len(train_dataloader) * args.epochs * 0.1),
        optimizer_params={"lr": args.lr},
        output_path=str(output_dir / "model"),
        show_progress_bar=True,
    )

    logger.info("Model saved to %s", output_dir / "model")
    logger.info("Done.")
    return output_dir / "model"


if __name__ == "__main__":
    main()
