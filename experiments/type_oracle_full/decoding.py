"""
decoding.py — Constrained decoding strategies for DCA-Trie.

- ``run_constrained_decoding`` — baseline & v1 single-pass GCR
- ``dca_v2_generate`` — v2 iterative hop-by-hop trie expansion (DoG-style)

Design
------
v2 combines two orthogonal ideas:

1. **Topological constraint** (from DoG): the trie encodes *all* valid triples
   from the head pool — purely structural, no semantic filtering.

2. **Semantic constraint** (from TypeOracle): before inserting a triple into
   the trie, we apply ``range_gate`` and ``type_gate`` to prune type-incompatible
   edges.  This is strictly stronger than topological-only pruning.

The result is a *topologically-structured, semantically-pruned* trie that
constrains generation at each hop.

NOTE on Freebase IDs: The LLM CAN generate Freebase ID tokens (they are
in the trie from the training data).  We do NOT filter them — that would
destroy 11% of answer reachability (measured: 94%→89% after v2 filtering).
Instead, gates run on ALL neighbors regardless of whether they are named
entities or IDs.
"""

from dataclasses import dataclass, field
from typing import List, Set

import torch

from src.graph_constrained_decoding import GraphConstrainedDecoding
from trie_utils import build_trie_from_strings
from utils import PATH_START, PATH_END, logger


# ---------------------------------------------------------------------------
# Beam unit for v2 iterative decoding
# ---------------------------------------------------------------------------

@dataclass
class BeamUnit:
    """A single beam in v2 iterative decoding."""
    sequence: str
    head_pool: Set[str] = field(default_factory=set)
    score: float = 0.0
    step: int = 0


# ---------------------------------------------------------------------------
# Run constrained decoding (baseline / v1)
# ---------------------------------------------------------------------------

def run_constrained_decoding(model, input_builder, data, trie):
    """Run graph-constrained decoding for a single question (baseline / v1)."""
    input_query, ground_paths, _ = input_builder.process_input(data, return_tire=False)
    start_token_ids = model.tokenizer.convert_tokens_to_ids(input_builder.PATH_START_TOKEN)
    end_token_ids = model.tokenizer.convert_tokens_to_ids(input_builder.PATH_END_TOKEN)
    llm_input = model.prepare_model_prompt(input_query)
    logger.debug("Constrained decoding: prompt_len=%d trie_paths=%s",
                 len(llm_input), "yes" if trie else "NO TRIE")
    prediction = model.generate_sentence(
        llm_input,
        trie,
        start_token_ids=start_token_ids,
        end_token_ids=end_token_ids,
        enable_constrained_by_default=False,
    )
    logger.debug("Prediction type=%s value=%s", type(prediction).__name__,
                 repr(prediction)[:200] if prediction else "None")
    return prediction, ground_paths


# ---------------------------------------------------------------------------
# Helper: collect gated paths from a head entity
# ---------------------------------------------------------------------------

def _get_gated_paths(
    nx_graph,
    head_entity: str,
    oracle,
    answer_types,
    hop: int,
    max_hops: int,
    gates_enabled: bool = True,
) -> List[str]:
    """
    Collect 1-hop paths from head_entity.

    When gates_enabled=True (default): applies range_gate and type_gate.
    When gates_enabled=False: admits ALL neighbors (DoG-proxy ablation).
    """
    paths = []
    if head_entity not in nx_graph:
        return paths

    for neighbor in nx_graph.neighbors(head_entity):
        rel = nx_graph[head_entity][neighbor]["relation"]
        if gates_enabled:
            if not oracle.range_gate(rel, neighbor):
                continue
            if hop >= max_hops and not oracle.type_gate(neighbor, answer_types, hop, max_hops):
                continue
        paths.append(f"{head_entity} -> {rel} -> {neighbor}")

    return paths


# ---------------------------------------------------------------------------
# DCA-Trie v2: iterative hop-by-hop trie expansion (DoG-style)
# ---------------------------------------------------------------------------

def dca_v2_generate(
    data,
    nx_graph,
    llm_model,
    tokenizer,
    oracle,
    max_hops,
    max_new_tokens,
    input_builder,
    beam_size: int = 5,
    gates_enabled: bool = True,
    collect_metrics: bool = False,
):
    """
    DCA-Trie v2: iterative hop-by-hop trie expansion with beam search.

    At each hop we:
      1. Enumerate 1-hop triples from every entity in the beam's head pool
         (topological — DoG style).
      2. Prune type-incompatible triples via TypeOracle gates (semantic).
      3. Build a per-beam trie from the surviving triples.
      4. Generate beam_size hops with constrained decoding.
      5. Score each beam using the model's log-probabilities.
      6. Keep the top beam_size beams for the next hop.

    Parameters
    ----------
    gates_enabled : bool
        If True (default), apply TypeOracle gates.  If False, admit all
        neighbors (DoG-proxy ablation).
    collect_metrics : bool
        If True, return (prediction, metrics_dict) instead of just prediction.

    Returns
    -------
    prediction : str or None
    metrics : dict (only if collect_metrics=True)
    """
    from ablation_metrics import (
        HopSnapshot, AblationMetrics,
        compute_bur, compute_sir_trajectory,
        compute_rebuild_volatility, compute_rv,
    )

    question = data["question"]
    start_entities = data.get("q_entity", [])
    answer_types = oracle.infer_answer_types(question)
    start_id = tokenizer.convert_tokens_to_ids(PATH_START)
    end_id = tokenizer.convert_tokens_to_ids(PATH_END)

    # Build initial prompt (matches v1/baseline format)
    prompt, _, _ = input_builder.process_input(data, return_tire=False)
    answer_markers = ["Answer:", "A:", "answer:"]
    for marker in answer_markers:
        idx = prompt.rfind(marker)
        if idx != -1:
            prompt = prompt[:idx].rstrip()
            break

    # ── Metrics collection state ──
    metrics = AblationMetrics(qid=data["id"]) if collect_metrics else None
    prev_tokens: Set[int] = set()
    before_counts: List[int] = []
    after_counts: List[int] = []
    token_sets_before: List[Set[int]] = []
    token_sets_after: List[Set[int]] = []

    # ------------------------------------------------------------------
    # Phase 1: Initialize beams from first-hop gated paths
    # ------------------------------------------------------------------
    initial_beams = []
    for entity in start_entities:
        first_hop_paths = _get_gated_paths(
            nx_graph, entity, oracle, answer_types, 1, max_hops,
            gates_enabled=gates_enabled,
        )
        if first_hop_paths:
            initial_beams.append(
                BeamUnit(sequence=prompt, head_pool={entity}, score=0.0, step=0)
            )

    if not initial_beams:
        logger.warning("v2: No initial beams from first-hop paths")
        return (None, metrics.to_dict()) if collect_metrics else None

    # ------------------------------------------------------------------
    # Phase 2: Iterate hops with beam search
    # ------------------------------------------------------------------
    current_beams = initial_beams

    for hop in range(1, max_hops + 1):
        if not current_beams:
            break

        new_beams = []
        hop_beams_in = len(current_beams)

        for beam in current_beams:
            # ---- Step 1: topological enumeration + semantic pruning ----
            all_1hop: List[str] = []
            for head in beam.head_pool:
                neighbor_count = len(list(nx_graph.neighbors(head))) if head in nx_graph else 0
                paths = _get_gated_paths(
                    nx_graph, head, oracle, answer_types, hop, max_hops,
                    gates_enabled=gates_enabled,
                )
                all_1hop.extend(paths)

            if not all_1hop:
                continue

            # ---- Metrics: record before/after counts ----
            if collect_metrics:
                # Count total 1-hop before gating (recompute without gates)
                raw_count = 0
                for head in beam.head_pool:
                    if head in nx_graph:
                        raw_count += len(list(nx_graph.neighbors(head)))
                before_counts.append(raw_count)
                after_counts.append(len(all_1hop))

            # ---- Step 2: build per-beam trie ----
            trie = build_trie_from_strings(tokenizer, all_1hop)
            if trie is None:
                continue

            # ---- Metrics: record token set volatility ----
            if collect_metrics:
                current_tokens = set(trie.token_to_id.keys()) if hasattr(trie, 'token_to_id') else set()
                token_sets_before.append(prev_tokens.copy())
                token_sets_after.append(current_tokens)
                prev_tokens = current_tokens

            # ---- Step 3: build prompt for THIS hop ----
            hop_prompt = f"{beam.sequence}\n{PATH_START}"

            inputs = tokenizer(
                hop_prompt, return_tensors="pt", add_special_tokens=False
            )
            input_ids = inputs.input_ids.to(llm_model.model.device)
            attn_mask = inputs.attention_mask.to(llm_model.model.device)

            # ---- Step 4: beam search for ONE hop ----
            gcr = GraphConstrainedDecoding(
                tokenizer, trie, start_id, end_id,
                enable_constrained_by_default=True,
            )

            local_num_beams = min(beam_size, len(all_1hop))
            if local_num_beams < 2:
                local_num_beams = 1

            gen_cfg = llm_model.model.generation_config.__class__(
                num_beams=local_num_beams,
                num_return_sequences=local_num_beams,
                early_stopping=True if local_num_beams > 1 else False,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )

            res = llm_model.model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                generation_config=gen_cfg,
                prefix_allowed_tokens_fn=gcr.allowed_tokens_fn,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=end_id,
                trust_remote_code=True,
            )

            # ---- Step 5: parse outputs and score beams ----
            for seq_idx in range(len(res.sequences)):
                output_tokens = res.sequences[seq_idx][input_ids.shape[1]:]
                output_text = tokenizer.decode(output_tokens, skip_special_tokens=False)

                path_content = _extract_path_content(output_text)
                if path_content is None:
                    continue

                segments = [s.strip() for s in path_content.split(" -> ")]
                new_entity = segments[-1] if segments else None

                if new_entity is None or new_entity in beam.head_pool:
                    continue

                if beam.step == 0:
                    accumulated_path = path_content
                else:
                    old_path = beam.sequence.split(PATH_START)[-1] if PATH_START in beam.sequence else ""
                    accumulated_path = old_path + " -> " + " -> ".join(segments[1:])

                # Score from beam search
                if hasattr(res, 'sequences_scores') and res.sequences_scores is not None:
                    hop_score = res.sequences_scores[seq_idx].item()
                else:
                    hop_score = 0.0

                new_beams.append(
                    BeamUnit(
                        sequence=f"{prompt}\n{PATH_START}{accumulated_path}",
                        head_pool=beam.head_pool | {new_entity},
                        score=beam.score + hop_score,
                        step=hop,
                    )
                )

        # Keep top-beam_size beams
        current_beams = sorted(new_beams, key=lambda b: b.score, reverse=True)[:beam_size]

        # ---- Metrics: record hop snapshot ----
        if collect_metrics:
            terminals = []
            for b in current_beams:
                path_content = _extract_path_content(b.sequence)
                if path_content:
                    segs = [s.strip() for s in path_content.split(" -> ")]
                    if segs:
                        terminals.append(segs[-1])

            metrics.hops.append(HopSnapshot(
                hop=hop,
                n_allowed_paths=after_counts[-1] if after_counts else 0,
                n_trie_paths=len(all_1hop) if all_1hop else 0,
                n_before_paths=before_counts[-1] if before_counts else 0,
                tokens_before=token_sets_before[-1] if token_sets_before else set(),
                tokens_after=token_sets_after[-1] if token_sets_after else set(),
                beams_in=hop_beams_in,
                beams_out=len(current_beams),
                terminal_entities=terminals,
            ))

    # ------------------------------------------------------------------
    # Phase 3: extract answer from best beam
    # ------------------------------------------------------------------
    prediction = None
    if current_beams:
        best_beam = current_beams[0]
        path_content = _extract_path_content(best_beam.sequence)
        if path_content:
            prediction = f"{PATH_START}{path_content}{PATH_END}"

    # ---- Compute final metrics ----
    if collect_metrics:
        # BUR
        all_terminals = []
        for b in current_beams:
            pc = _extract_path_content(b.sequence)
            if pc:
                segs = [s.strip() for s in pc.split(" -> ")]
                if segs:
                    all_terminals.append(segs[-1])
        metrics.bur, metrics.bur_entropy = compute_bur(all_terminals)

        # SIR trajectory
        metrics.sir_curve, metrics.sir_decay_slope = compute_sir_trajectory(
            before_counts, after_counts
        )

        # Rebuild volatility
        metrics.volatility_curve = compute_rebuild_volatility(
            token_sets_before, token_sets_after
        )

        # RV
        metrics.rv_curve = compute_rv(before_counts, after_counts)

        return prediction, metrics.to_dict()

    return prediction


def _extract_path_content(text: str) -> str | None:
    """Extract path content between <PATH> and </PATH>."""
    start_idx = text.find(PATH_START)
    end_idx = text.find(PATH_END)

    if start_idx == -1:
        return None

    if end_idx != -1:
        content = text[start_idx + len(PATH_START):end_idx]
    else:
        content = text[start_idx + len(PATH_START):]

    content = content.strip()
    return content if content else None
