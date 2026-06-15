"""
Approach 1: Cosine Similarity Path Scoring (Standalone Demo)

Scores each candidate KG path by computing the cosine similarity between
a sentence-transformer embedding of the serialised path string and an
embedding of the input question. Paths above threshold τ are admitted.

Usage:
    python core_algorithm.py

Dependencies: numpy, sklearn, sentence-transformers
    pip install numpy scikit-learn sentence-transformers
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def build_cosine_trie(
    question: str,
    paths: list[list[tuple[str, str, str]]],
    encoder,
    tau: float = 0.25,
) -> tuple[list | None, list, list[float]]:
    """DCA-Trie v0: cosine similarity gate.

    Args:
        question: Natural language question string.
        paths: List of candidate paths, where each path is a list of
               (head, relation, tail) triples.
        encoder: SentenceTransformer-like object with .encode() method.
        tau: Admission threshold (default 0.25).

    Returns:
        (trie_builder_data, kept_paths, scores)
        trie_builder_data is None if no paths pass the gate.
    """
    q_emb = encoder.encode(question, convert_to_numpy=True).reshape(1, -1)

    kept_paths: list[list[tuple[str, str, str]]] = []
    kept_scores: list[float] = []

    for p in paths:
        path_str = _serialize_path(p)
        p_emb = encoder.encode(path_str, convert_to_numpy=True).reshape(1, -1)
        score = float(cosine_similarity(q_emb, p_emb)[0, 0])

        if score >= tau:
            kept_paths.append(p)
            kept_scores.append(score)

    if not kept_paths:
        return None, [], []

    return kept_paths, kept_paths, kept_scores


def _serialize_path(path: list[tuple[str, str, str]]) -> str:
    """Convert a list of (head, rel, tail) triples to a single string."""
    return " ".join(f"{h} {r} {t}" for h, r, t in path)


# Demo
if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    sample_question = "What is the nationality of the director of Blue Hawaii?"

    sample_paths = [
        # Path A: correct gold path
        [("Blue Hawaii", "film.director", "Norman Taurog"),
         ("Norman Taurog", "people.person.nationality", "United States")],
        # Path B: wrong type (ends at a film, not a nationality)
        [("Blue Hawaii", "film.featured_film_location", "Hawaii")],
        # Path C: wrong direction
        [("Blue Hawaii", "film.director", "Norman Taurog"),
         ("Norman Taurog", "people.person.spouse_s", "Kathryn Pringle")],
    ]

    kept, paths_out, scores = build_cosine_trie(
        sample_question, sample_paths, encoder, tau=0.25
    )

    print(f"Question: {sample_question}")
    print(f"Paths evaluated: {len(sample_paths)}")
    print(f"Paths kept (τ=0.25): {len(paths_out)}\n")
    for p, s in zip(paths_out, scores):
        print(f"  score={s:.3f}  {_serialize_path(p)}")
