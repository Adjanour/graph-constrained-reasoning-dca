"""
reranker.py — Cross-encoder path relevance scorer for KGQA.

Given a question and a set of KG paths, scores each path by
how likely it is to lead to the correct answer.

Two modes:
  - zero-shot: uses sentence-transformers cosine similarity
  - fine-tuned: trained on (question, path) → correct/incorrect pairs
"""

import logging
from typing import List, Tuple, Optional

logger = logging.getLogger("learned_pruning")


class PathReranker:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None
        self._cross_encoder = None

    def _lazy_load(self):
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder
        logger.info("Loading CrossEncoder: %s", self._model_name)
        self._cross_encoder = CrossEncoder(self._model_name)

    def score(self, question: str, paths: List[str]) -> List[float]:
        """Score each (question, path) pair. Higher = more relevant."""
        self._lazy_load()
        pairs = [(question, p) for p in paths]
        scores = self._cross_encoder.predict(pairs, show_progress_bar=False)
        return scores.tolist() if hasattr(scores, 'tolist') else list(scores)

    def rank(self, question: str, paths: List[str]) -> List[Tuple[int, str, float]]:
        """Return paths sorted by relevance score descending."""
        scores = self.score(question, paths)
        indexed = list(enumerate(zip(paths, scores)))
        indexed.sort(key=lambda x: x[1][1], reverse=True)
        return [(idx, p, s) for idx, (p, s) in indexed]


class PathData:
    """Prepare training/evaluation data from existing experiment runs."""

    @staticmethod
    def from_predictions(pred_path: str, graph_data=None):
        """Load predictions and extract (question, path, correct) triples."""
        import json
        triples = []
        with open(pred_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                q = rec.get("question", "")
                gt = set(rec.get("ground_truth", []))
                pred = rec.get("prediction", "")
                # Extract paths from prediction text
                paths = PathData._extract_paths(pred)
                for p in paths:
                    is_correct = PathData._path_answers_correct(p, gt)
                    triples.append((q, p, is_correct))
        return triples

    @staticmethod
    def _extract_paths(prediction) -> List[str]:
        """Extract individual path strings from model output."""
        if not prediction:
            return []
        if isinstance(prediction, list):
            items = prediction
        else:
            items = [prediction]
        paths = []
        for item in items:
            if "# Reasoning Path:\n" in item:
                parts = item.split("# Reasoning Path:\n")
                for part in parts[1:]:
                    path_line = part.split("\n")[0].strip()
                    if path_line:
                        paths.append(path_line)
            elif "# Reasoning Path:" in item:
                parts = item.split("# Reasoning Path:")
                for part in parts[1:]:
                    path_line = part.split("\n")[0].strip()
                    if path_line:
                        paths.append(path_line)
        return paths

    @staticmethod
    def _path_answers_correct(path_str: str, ground_truth: set) -> bool:
        """Check if the terminal entity in a path matches ground truth."""
        if not path_str or not ground_truth:
            return False
        segments = [s.strip() for s in path_str.split("->")]
        terminal = segments[-1].strip() if segments else ""
        if not terminal:
            return False
        terminal_lower = terminal.lower()
        return any(g.lower() == terminal_lower or g.lower() in terminal_lower
                   for g in ground_truth)
