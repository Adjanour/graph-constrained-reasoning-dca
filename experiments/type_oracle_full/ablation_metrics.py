"""
ablation_metrics.py — Diagnostic metrics for v2 ablation studies.

Four metrics that turn qualitative stories into measured mechanisms:

1. BUR (Beam Utilization Ratio) — diversity collapse among beams
2. SIR Trajectory — whether "dynamic" actually does anything per hop
3. Rebuild Volatility — trie instability at each rebuild
4. RV (Reduction Volume) — how much search space shrinks per hop

All metrics are computed per-question during v2 iterative decoding.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Per-hop snapshot (collected during v2 decoding)
# ---------------------------------------------------------------------------

@dataclass
class HopSnapshot:
    """Metrics captured at one hop during v2 iterative decoding."""
    hop: int
    n_allowed_paths: int          # paths surviving gates from head pool
    n_trie_paths: int             # paths inserted into trie (may differ if trie deduplicates)
    n_before_paths: int           # total 1-hop paths from head pool (before gating)
    tokens_before: Set[int]      # valid token set before this rebuild (from previous trie)
    tokens_after: Set[int]       # valid token set after this rebuild (current trie)
    beams_in: int                 # beams entering this hop
    beams_out: int                # beams surviving this hop
    terminal_entities: List[str]  # terminal entities of surviving beams


# ---------------------------------------------------------------------------
# Final metrics (returned per-question)
# ---------------------------------------------------------------------------

@dataclass
class AblationMetrics:
    """All ablation metrics for a single question's v2 decoding."""
    qid: str

    # Per-hop data
    hops: List[HopSnapshot] = field(default_factory=list)

    # BUR (Beam Utilization Ratio)
    bur: float = 0.0              # |distinct terminals| / |beams|
    bur_entropy: float = 0.0      # entropy of terminal entity distribution

    # SIR trajectory
    sir_curve: List[float] = field(default_factory=list)  # SIR at each hop
    sir_decay_slope: float = 0.0  # linear fit slope of SIR curve

    # Rebuild volatility
    volatility_curve: List[float] = field(default_factory=list)  # Jaccard dist per hop

    # RV (Reduction Volume)
    rv_curve: List[float] = field(default_factory=list)  # 1 - trie/before per hop

    # Per-question graph stats
    n_nodes: int = 0
    n_edges: int = 0
    n_fb_ids: int = 0
    avg_out_degree: float = 0.0
    n_dfs_paths: int = 0
    timing_s: float = 0.0

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "qid": self.qid,
            # BUR
            "bur": round(self.bur, 4),
            "bur_entropy": round(self.bur_entropy, 4),
            # SIR trajectory
            "sir_curve": [round(x, 4) for x in self.sir_curve],
            "sir_decay_slope": round(self.sir_decay_slope, 4),
            # Rebuild volatility
            "volatility_curve": [round(x, 4) for x in self.volatility_curve],
            # RV
            "rv_curve": [round(x, 4) for x in self.rv_curve],
            # Graph stats
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "n_fb_ids": self.n_fb_ids,
            "avg_out_degree": round(self.avg_out_degree, 2),
            "n_dfs_paths": self.n_dfs_paths,
            "timing_s": round(self.timing_s, 3),
            # Per-hop detail
            "hops": [
                {
                    "hop": h.hop,
                    "n_allowed": h.n_allowed_paths,
                    "n_trie": h.n_trie_paths,
                    "n_before": h.n_before_paths,
                    "beams_in": h.beams_in,
                    "beams_out": h.beams_out,
                    "n_terminals": len(h.terminal_entities),
                }
                for h in self.hops
            ],
        }


# ---------------------------------------------------------------------------
# Metric computation functions
# ---------------------------------------------------------------------------


def compute_bur(terminal_entities: List[str]) -> Tuple[float, float]:
    """Compute Beam Utilization Ratio and its entropy.

    BUR = |distinct terminals| / |total beams|
    Entropy = -sum(p * log2(p)) for each distinct terminal

    High BUR + low entropy = diversity collapse (beams agree)
    Low BUR + high entropy = healthy exploration
    """
    if not terminal_entities:
        return 0.0, 0.0

    total = len(terminal_entities)
    counts: Dict[str, int] = {}
    for e in terminal_entities:
        counts[e] = counts.get(e, 0) + 1

    bur = len(counts) / total

    # Entropy
    entropy = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            entropy -= p * math.log2(p)

    return bur, entropy


def compute_sir_trajectory(
    before_counts: List[int],
    after_counts: List[int],
) -> Tuple[List[float], float]:
    """Compute SIR curve and its decay slope.

    SIR(t) = 1 - after(t) / before(t)  (fraction removed at hop t)
    Decay slope = linear fit of SIR curve (negative = oracle sharpens with context)

    Returns (sir_curve, slope).
    """
    if not before_counts or not after_counts:
        return [], 0.0

    sir_curve = []
    for b, a in zip(before_counts, after_counts):
        if b > 0:
            sir_curve.append(1.0 - a / b)
        else:
            sir_curve.append(0.0)

    # Linear fit: SIR = slope * t + intercept
    n = len(sir_curve)
    if n < 2:
        return sir_curve, 0.0

    t_mean = (n - 1) / 2.0
    s_mean = sum(sir_curve) / n

    num = sum((t - t_mean) * (s - s_mean) for t, s in enumerate(sir_curve))
    den = sum((t - t_mean) ** 2 for t in range(n))

    slope = num / den if den > 0 else 0.0
    return sir_curve, slope


def compute_rebuild_volatility(
    tokens_before_list: List[Set[int]],
    tokens_after_list: List[Set[int]],
) -> List[float]:
    """Compute Jaccard distance of valid token sets between consecutive rebuilds.

    Volatility(t) = 1 - |tokens_before(t) ∩ tokens_after(t)| / |tokens_before(t) ∪ tokens_after(t)|

    High volatility = trie changed a lot → beams may be invalidated.
    """
    if len(tokens_before_list) < 2:
        return []

    volatility = []
    for i in range(1, len(tokens_before_list)):
        before = tokens_before_list[i - 1]
        after = tokens_after_list[i] if i < len(tokens_after_list) else tokens_before_list[i]

        if not before and not after:
            volatility.append(0.0)
            continue
        if not before or not after:
            volatility.append(1.0)
            continue

        intersection = before & after
        union = before | after
        jaccard_sim = len(intersection) / len(union) if union else 0.0
        volatility.append(1.0 - jaccard_sim)

    return volatility


def compute_rv(before_counts: List[int], trie_counts: List[int]) -> List[float]:
    """Compute Reduction Volume at each hop.

    RV(t) = 1 - trie_count(t) / before_count(t)

    High RV = lots of pruning (trie much smaller than search space)
    Low RV = little pruning (trie ≈ full search space)
    """
    rv = []
    for b, t in zip(before_counts, trie_counts):
        if b > 0:
            rv.append(1.0 - t / b)
        else:
            rv.append(0.0)
    return rv


# ---------------------------------------------------------------------------
# Graph statistics (per-question)
# ---------------------------------------------------------------------------

def compute_graph_stats(nx_graph, all_paths: list) -> dict:
    """Compute per-question graph statistics for logging."""
    import re
    fb_re = re.compile(r"^[gm]\.\w+$")

    n_nodes = len(nx_graph.nodes())
    n_edges = len(nx_graph.edges())
    n_fb = sum(1 for n in nx_graph.nodes() if fb_re.match(str(n)))

    # Average out-degree
    out_degrees = [len(list(nx_graph.neighbors(n))) for n in nx_graph.nodes()]
    avg_out = sum(out_degrees) / len(out_degrees) if out_degrees else 0.0

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_fb_ids": n_fb,
        "avg_out_degree": round(avg_out, 2),
        "n_dfs_paths": len(all_paths),
    }
