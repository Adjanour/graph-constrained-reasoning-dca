"""
Approach 2: Decomposed Product Score (Standalone Demo)

Implements Eq. 3.12: score = ρ_r · ρ_e · ρ_traj at construction time.

Components:
  ρ_r(r, q)     = cos(E(r), E(q_rel)) — entity-masked relational relevance
  ρ_e(e', q)    = 1[type(e') ∈ T(q,h)] — hard type gate (no encoder)
  ρ_traj(r,e',q)= cos(E(r ‖ e'), E(q)) — trajectory relevance

Usage:
    python core_algorithm.py

Dependencies: numpy, sklearn, sentence-transformers
    pip install numpy scikit-learn sentence-transformers
"""

import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ──────────────────────────────────────────────────────────────────────
# Answer Type Inference (hand-crafted patterns, §3.5.4)
# ──────────────────────────────────────────────────────────────────────

QUESTION_TYPE_PATTERNS: list[tuple[str, set[str]]] = [
    (r"\bwho\b", {"Person", "Deceased Person", "Politician"}),
    (r"\bwhere\b", {"Location", "Country", "City/Town/Village"}),
    (r"\bwhen\b|\bdate\b|\byear\b", {"Date/Time"}),
    (r"\bfilm\b|\bmovie\b", {"Film", "TV Program"}),
    (r"\blanguage\b|\bspeak\b", {"Human Language"}),
    (r"\bprofession\b|\bjob\b|\boccupation\b", {"Profession"}),
    (r"\baward\b|\bprize\b", {"Award", "Award honor"}),
    (r"\borganization\b|\bcompany\b", {"Organization", "Company"}),
    (r"\bcountry\b|\bnation\b", {"Country"}),
    (r"\bcity\b", {"City/Town/Village"}),
]


def infer_answer_types(question: str) -> set[str]:
    """Infer expected answer entity types from question text."""
    q_lower = question.lower()
    for pattern, types in QUESTION_TYPE_PATTERNS:
        if re.search(pattern, q_lower):
            return types
    return set()


def is_type_compatible(
    entity_types: set[str],
    answer_types: set[str],
    hop: int,
    is_terminal: bool,
) -> bool:
    """Hard type gate (ρ_e). Admits by default when info is missing."""
    if not is_terminal:
        return True
    if not answer_types:
        return True
    if not entity_types:
        return True
    return bool(entity_types & answer_types)


# ──────────────────────────────────────────────────────────────────────
# Entity Masking (§3.5.1)
# ──────────────────────────────────────────────────────────────────────


def mask_entities(question: str, entities: list[str]) -> str:
    """Replace entity names with [MASK] so encoder focuses on relations."""
    masked = question
    for e in sorted(entities, key=len, reverse=True):
        masked = masked.replace(e, "[MASK]")
    return masked


# ──────────────────────────────────────────────────────────────────────
# Entity Type Extraction
# ──────────────────────────────────────────────────────────────────────


def build_entity_type_map(
    graph_triples: list[tuple[str, str, str]],
) -> dict[str, set[str]]:
    """Extract entity types from graph triples (common.topic.notable_types)."""
    type_map: dict[str, set[str]] = {}
    for h, r, t in graph_triples:
        if r in ("common.topic.notable_types",):
            type_map.setdefault(h, set()).add(t)
    return type_map


# ──────────────────────────────────────────────────────────────────────
# DCA-Trie v1: Decomposed Static Filtering (Algorithm 1)
# ──────────────────────────────────────────────────────────────────────

_relation_emb_cache: dict[str, np.ndarray] = {}


def build_decomposed_trie(
    question: str,
    question_entities: list[str],
    paths: list[list[tuple[str, str, str]]],
    graph_triples: list[tuple[str, str, str]],
    encoder,
    tau: float = 0.25,
) -> tuple[list | None, list, list[float], int, int]:
    """Build DCA-Trie v1 with decomposed product score (Eq. 3.12).

    Returns:
        (trie_builder_data, kept_paths, scores, n_type_blocked, n_encoder_calls)
    """
    # Step 1: Relational intent vector via entity masking
    q_rel_str = mask_entities(question, question_entities)
    u_rel = encoder.encode(q_rel_str, convert_to_numpy=True)
    u_q = encoder.encode(question, convert_to_numpy=True)

    # Step 2: Answer type inference
    answer_types = infer_answer_types(question)
    type_map = build_entity_type_map(graph_triples)

    # Step 3: Filter paths
    kept_paths: list[list[tuple[str, str, str]]] = []
    kept_scores: list[float] = []
    n_type_blocked = 0
    n_encoder_calls = 2  # q_rel and q already encoded above

    for p in paths:
        h = len(p)
        e_terminal = p[-1][2]
        e_terminal_types = type_map.get(e_terminal, set())

        # Hard type gate (§3.5.5, Eq. 3.12 component ρ_e)
        if not is_type_compatible(e_terminal_types, answer_types, h, is_terminal=True):
            n_type_blocked += 1
            continue

        # Relational relevance ρ_r — product across hops
        rel_score = 1.0
        for i in range(h):
            r_i = p[i][1]
            if r_i not in _relation_emb_cache:
                _relation_emb_cache[r_i] = encoder.encode(r_i, convert_to_numpy=True)
            n_encoder_calls += 1
            r_emb = _relation_emb_cache[r_i].reshape(1, -1)
            rho_r = float(cosine_similarity(r_emb, u_rel.reshape(1, -1))[0, 0])
            rel_score *= rho_r

        # Trajectory relevance ρ_traj at terminal hop
        r_term = p[-1][1]
        rte_str = f"{r_term} {e_terminal}"
        rte_emb = encoder.encode(rte_str, convert_to_numpy=True)
        n_encoder_calls += 1
        rho_traj = float(
            cosine_similarity(rte_emb.reshape(1, -1), u_q.reshape(1, -1))[0, 0]
        )

        score = rel_score * rho_traj  # Eq. 3.12

        if score >= tau:
            kept_paths.append(p)
            kept_scores.append(score)

    if not kept_paths:
        return None, [], [], n_type_blocked, n_encoder_calls

    return kept_paths, kept_paths, kept_scores, n_type_blocked, n_encoder_calls


def serialize_path(path: list[tuple[str, str, str]]) -> str:
    return " ".join(f"{h} {r} {t}" for h, r, t in path)


# Demo
if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    question = "What is the nationality of the director of Blue Hawaii?"
    q_entities = ["Blue Hawaii"]

    graph_triples = [
        ("Blue Hawaii", "film.director", "Norman Taurog"),
        ("Norman Taurog", "people.person.nationality", "United States"),
        ("Norman Taurog", "people.person.spouse_s", "Kathryn Pringle"),
        ("Blue Hawaii", "film.featured_film_location", "Hawaii"),
        ("Norman Taurog", "common.topic.notable_types", "Person"),
        ("United States", "common.topic.notable_types", "Country"),
        ("Hawaii", "common.topic.notable_types", "Location"),
    ]

    paths = [
        [
            ("Blue Hawaii", "film.director", "Norman Taurog"),
            ("Norman Taurog", "people.person.nationality", "United States"),
        ],
        [("Blue Hawaii", "film.featured_film_location", "Hawaii")],
        [
            ("Blue Hawaii", "film.director", "Norman Taurog"),
            ("Norman Taurog", "people.person.spouse_s", "Kathryn Pringle"),
        ],
    ]

    result, kept, scores, n_blocked, n_enc = build_decomposed_trie(
        question, q_entities, paths, graph_triples, encoder, tau=0.25
    )

    print(f"Question: {question}")
    print(f"Answer type inferred: {infer_answer_types(question)}")
    print(f"Paths evaluated: {len(paths)}")
    print(f"Type-blocked: {n_blocked}")
    print(f"Encoder calls: {n_enc}")
    print(f"Paths kept: {len(kept)}\n")
    for p, s in zip(kept, scores):
        print(f"  score={s:.3f}  {serialize_path(p)}")
