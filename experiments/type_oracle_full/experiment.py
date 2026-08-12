"""
experiment.py — Metrics, per-condition runners, and dataset-level orchestration.

- ``PrepCache`` — per-question precomputed graph, oracle, paths, tokenization
- ``compute_hits`` — Hits@1 evaluation metric
- ``_run_baseline`` / ``_run_v1`` / ``_run_v2`` — single-sample runners
- ``run_condition`` — loops over a dataset for one condition, with checkpoint/resume
"""

import json
import os
import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

import src.utils as graph_utils
from approach3_symbolic.type_oracle import TypeOracle
from src.utils.qa_utils import eval_hit, extract_topk_prediction, normalize

from decoding import dca_v2_generate, run_constrained_decoding
from invariants import check_all_invariants
from trie_utils import (
    build_filtered_trie,
    build_trie_from_token_ids,
    build_unfiltered_trie,
)
from utils import (
    PATH_END,
    PATH_START,
    TimeoutError,
    atomic_write_jsonl,
    load_preds,
    logger,
    safe_read_jsonl,
    timeout,
)


# ---------------------------------------------------------------------------
# PrepCache — per-question precomputed data shared across conditions
# ---------------------------------------------------------------------------


@dataclass
class PrepCache:
    """Per-question precomputed graph, oracle, paths, and tokenization.

    Computed once per question and reused across GCR_Baseline, DCA_v1_Static,
    and DCA_v2_Dynamic conditions.  Eliminates 3× redundant graph construction,
    DFS enumeration, oracle creation, and tokenization.
    """

    qid: str
    data: dict
    nx_graph: nx.DiGraph
    oracle: TypeOracle
    all_paths: list
    path_strings: List[str]
    tokenized: List[List[int]]  # list of token ID sequences (no EOS yet)

    @classmethod
    def build(cls, data: dict, index_len: int) -> "PrepCache":
        """Build a PrepCache from a single question's data dict."""
        qid = data["id"]
        nx_graph = graph_utils.build_graph(data["graph"], undirected=False)
        oracle = TypeOracle.from_graph(data["graph"])
        all_paths = graph_utils.dfs(nx_graph, data.get("q_entity", []), index_len)
        path_strings = [graph_utils.path_to_string(p) for p in all_paths]
        return cls(
            qid=qid,
            data=data,
            nx_graph=nx_graph,
            oracle=oracle,
            all_paths=all_paths,
            path_strings=path_strings,
            tokenized=[],  # filled lazily by _ensure_tokenized
        )

    def _ensure_tokenized(self, tokenizer) -> None:
        """Tokenize path_strings once (lazy — only when first needed)."""
        if self.tokenized:
            return
        if not self.path_strings:
            self.tokenized = []
            return
        wrapped = [f"{PATH_START}{s}{PATH_END}" for s in self.path_strings]
        self.tokenized = tokenizer(
            wrapped, padding=False, add_special_tokens=False
        ).input_ids

    def get_all_token_ids(self, tokenizer) -> List[List[int]]:
        """Return tokenized paths with EOS appended (for baseline/v1 trie)."""
        self._ensure_tokenized(tokenizer)
        eos = tokenizer.eos_token_id
        return [ids + [eos] for ids in self.tokenized]

    def get_filtered_token_ids(
        self, tokenizer, answer_types
    ) -> Tuple[List[List[int]], List[list]]:
        """Return filtered token IDs and filtered path tuples for v1.

        Filters all_paths through TypeOracle gates, then tokenizes only
        the surviving paths.
        """
        filtered_paths = []
        for p in self.all_paths:
            admit = True
            for _, rel, tail in p:
                if not self.oracle.range_gate(rel, tail):
                    admit = False
                    break
            if admit and p:
                terminal = p[-1][2]
                if not self.oracle.type_gate(
                    terminal, answer_types, len(p), len(p)
                ):
                    admit = False
            if admit:
                filtered_paths.append(p)

        if not filtered_paths:
            return [], filtered_paths

        filtered_strs = [graph_utils.path_to_string(p) for p in filtered_paths]
        wrapped = [f"{PATH_START}{s}{PATH_END}" for s in filtered_strs]
        tokenized = tokenizer(
            wrapped, padding=False, add_special_tokens=False
        ).input_ids
        eos = tokenizer.eos_token_id
        return [ids + [eos] for ids in tokenized], filtered_paths


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_hits(preds):
    """Compute Hits@1 using the repo's substring evaluator.

    Matches ``src.utils.qa_utils.eval_hit`` on the raw prediction string.
    """
    hits = 0
    for p in preds:
        prediction = p.get("prediction", "")
        answers = list(set(p.get("ground_truth", [])))
        if not answers:
            continue
        pred_str = prediction if isinstance(prediction, str) else " ".join(prediction)
        if eval_hit(pred_str, answers):
            hits += 1
    return hits


# ---------------------------------------------------------------------------
# Trace output — matches step_by_step.py format
# ---------------------------------------------------------------------------

import re as _re
_FB_ID_RE = _re.compile(r"^[gm]\.\w+$")


def trace_sample(data, prep, cond_name, result, answer_types=None):
    """Print per-sample trace output matching step_by_step.py format.

    This is the ``--trace`` mode: it prints exactly what the math says
    happens at each step, so ``Figure 3 in the paper'' and terminal
    output are the same artifact.
    """
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    RED = "\033[31m"
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    RESET = "\033[0m"

    def heading(text):
        print(f"\n{BOLD}{CYAN}{'━' * 72}{RESET}")
        print(f"{BOLD}{CYAN}  [{cond_name}] {text}{RESET}")
        print(f"{BOLD}{CYAN}{'━' * 72}{RESET}")

    def sub(text):
        print(f"\n  {BOLD}{YELLOW}▸ {text}{RESET}")
        print(f"{DIM}{'─' * 72}{RESET}")

    question = data["question"]
    entities = data.get("q_entity", [])
    answers = data.get("answer", [])
    g = prep.nx_graph
    oracle = prep.oracle

    fb_count = sum(1 for n in g.nodes() if _FB_ID_RE.match(n))

    # ── Step 1: Question ──
    heading("STEP 1: Question")
    print(f"  Question:  {BOLD}{question}{RESET}")
    print(f"  Entities:  {entities}")
    print(f"  Answer(s): {GREEN}{answers}{RESET}")

    # ── Step 2: KG Subgraph ──
    heading("STEP 2: Knowledge Graph Subgraph")
    print(f"  Nodes: {len(g.nodes())} ({len(g.nodes()) - fb_count} named, {fb_count} Freebase IDs)")
    print(f"  Edges: {len(g.edges())}")
    sub("Edges from question entity")
    for entity in entities:
        if entity not in g:
            continue
        for i, (nbr, edata) in enumerate(g[entity].items()):
            if i >= 8:
                print(f"    {DIM}... and {len(g[entity]) - 8} more{RESET}")
                break
            rel = edata["relation"]
            fb_tag = f" {RED}[FB-ID]{RESET}" if _FB_ID_RE.match(nbr) else ""
            print(f"    {entity}  →  {rel}  →  {nbr}{fb_tag}")

    # ── Step 3: TypeOracle ──
    heading("STEP 3: TypeOracle")
    if answer_types is None:
        answer_types = oracle.infer_answer_types(question)
        if not answer_types:
            answer_types = oracle.infer_answer_types_from_paths(prep.all_paths)
    print(f"  Answer types: {BOLD}{answer_types if answer_types else '(empty)'}{RESET}")
    print(f"  Entity types known: {len(oracle._entity_types)}")
    print(f"  Hand-curated schema: {len(oracle._schema)} relations")
    print(f"  Auto-mined schema:   {len(oracle._mined_schema)} relations")

    # ── Step 4: DFS Path Enumeration ──
    heading("STEP 4: DFS Path Enumeration (all paths)")
    print(f"  Total paths: {BOLD}{len(prep.all_paths)}{RESET}")
    sub("Sample paths")
    for p in prep.all_paths[:5]:
        print(f"    {graph_utils.path_to_string(p)}")
    if len(prep.all_paths) > 5:
        print(f"    {DIM}... and {len(prep.all_paths) - 5} more{RESET}")

    # ── Step 5: TypeOracle Filtering (v1) ──
    if cond_name in ("DCA_v1_Static", "GCR_Baseline"):
        heading("STEP 5: TypeOracle Filtering (v1 static)")
        filtered = []
        for p in prep.all_paths:
            admit = True
            for _, rel, tail in p:
                if not oracle.range_gate(rel, tail):
                    admit = False
                    break
            if admit and p:
                terminal = p[-1][2]
                if not oracle.type_gate(terminal, answer_types, len(p), len(p)):
                    admit = False
            if admit:
                filtered.append(p)
        reduction = (1 - len(filtered) / max(1, len(prep.all_paths))) * 100
        print(f"  Before: {len(prep.all_paths)}  →  After: {BOLD}{len(filtered)}{RESET}  ({reduction:.1f}% reduction)")

        removed = [p for p in prep.all_paths if p not in filtered]
        if removed:
            sub("Removed paths (failed TypeOracle)")
            for p in removed[:4]:
                s = graph_utils.path_to_string(p)
                terminal = p[-1][2]
                ttypes = oracle.get_types(terminal)
                print(f"    {s}")
                print(f"      terminal types={ttypes}, answer_types={answer_types}")

    # ── Step 6: Trie Construction ──
    heading("STEP 6: Trie Construction")
    n_paths = len(prep.all_paths)
    print(f"  Baseline trie: {n_paths} paths (all DFS paths)")
    if cond_name == "DCA_v1_Static":
        print(f"  Filtered trie: built from TypeOracle-filtered paths")
    elif cond_name == "DCA_v2_Dynamic":
        print(f"  V2 tries: per-hop, 1-hop paths from head pool")

    # ── Step 7: Prediction ──
    heading("STEP 7: Prediction")
    if result and result.get("prediction"):
        pred = result["prediction"]
        if isinstance(pred, list):
            pred = pred[0] if pred else ""
        path_part, _, answer_part = pred.partition("# Answer:")
        rp = path_part.replace("# Reasoning Path:", "").strip()
        print(f"  Reasoning path: {BOLD}{rp}{RESET}")
        print(f"  Answer:         {BOLD}{answer_part.strip()}{RESET}")
        correct = answer_part.strip().lower() in [a.lower() for a in answers]
        if correct:
            print(f"  {GREEN}✓ CORRECT{RESET}")
        else:
            print(f"  {RED}✗ WRONG — expected: {answers}{RESET}")
    else:
        print(f"  {RED}(no prediction){RESET}")

    print()


def _build_result_dict(qid, question, prediction_str, ground_truth, cond_name, *, extra=None):
    """Build a uniform result record (prediction always a string, never ``[]``)."""
    result = {
        "id": qid,
        "question": question,
        "prediction": prediction_str if prediction_str is not None else "",
        "ground_truth": ground_truth,
        "mode": cond_name,
    }
    if extra:
        result.update(extra)
    return result


# ---------------------------------------------------------------------------
# Per-condition single-sample runners
# ---------------------------------------------------------------------------


def _run_baseline(model, input_builder, data, qid, cond_name, prep, **_kwargs):
    """Run baseline GCR.  Returns (result_dict | None, trie_ok)."""
    token_ids = prep.get_all_token_ids(model.tokenizer)
    if not token_ids:
        logger.debug("Sample %s: no trie for baseline (no entities/paths)", qid)
        return None, False

    trie = build_trie_from_token_ids(model.tokenizer, token_ids)
    if trie is None:
        logger.debug("Sample %s: trie build failed for baseline", qid)
        return None, False

    check_all_invariants(
        tokenizer=model.tokenizer,
        trie=trie,
        path_strings=prep.path_strings,
        all_paths=prep.all_paths,
        filtered_paths=None,
        nx_graph=prep.nx_graph,
        oracle=prep.oracle,
        answer_types=frozenset(),
        max_hop=len(prep.all_paths[0]) if prep.all_paths else 2,
        cond_name=cond_name,
    )

    prediction, _ = run_constrained_decoding(model, input_builder, data, trie)
    result = _build_result_dict(
        qid, data["question"],
        prediction if prediction else "",
        data["answer"], cond_name,
        extra={"n_paths_all": len(prep.all_paths)},
    )
    return result, True


def _run_v1(model, input_builder, data, qid, cond_name, prep, **_kwargs):
    """Run v1 static type-oracle.  Returns (result_dict | None, trie_ok)."""
    answer_types = prep.oracle.infer_answer_types(data["question"])
    if not answer_types:
        answer_types = prep.oracle.infer_answer_types_from_paths(prep.all_paths)

    token_ids, filtered_paths = prep.get_filtered_token_ids(
        model.tokenizer, answer_types
    )
    if not token_ids:
        logger.debug("Sample %s: no trie for v1 (no entities/filtered paths)", qid)
        return None, False

    trie = build_trie_from_token_ids(model.tokenizer, token_ids)
    if trie is None:
        logger.debug("Sample %s: trie build failed for v1", qid)
        return None, False

    filtered_strs = [graph_utils.path_to_string(p) for p in filtered_paths]
    check_all_invariants(
        tokenizer=model.tokenizer,
        trie=trie,
        path_strings=filtered_strs,
        all_paths=prep.all_paths,
        filtered_paths=filtered_paths,
        nx_graph=prep.nx_graph,
        oracle=prep.oracle,
        answer_types=answer_types,
        max_hop=len(prep.all_paths[0]) if prep.all_paths else 2,
        cond_name=cond_name,
    )

    prediction, _ = run_constrained_decoding(model, input_builder, data, trie)
    result = _build_result_dict(
        qid, data["question"],
        prediction if prediction else "",
        data["answer"], cond_name,
        extra={
            "n_paths_all": len(prep.all_paths),
            "n_paths_filtered": len(filtered_paths),
        },
    )
    return result, True


def _run_v2(model, input_builder, data, qid, cond_name, prep, index_len, max_new_tokens, beam_size=5, **_kwargs):
    """Run v2 dynamic type-oracle.  Returns (result_dict | None, trie_ok)."""
    gates_enabled = _kwargs.get("gates_enabled", True)
    collect_metrics = _kwargs.get("collect_metrics", False)

    prediction = dca_v2_generate(
        data=data,
        nx_graph=prep.nx_graph,
        llm_model=model,
        tokenizer=model.tokenizer,
        oracle=prep.oracle,
        max_hops=index_len,
        max_new_tokens=max_new_tokens,
        input_builder=input_builder,
        beam_size=beam_size,
        gates_enabled=gates_enabled,
        collect_metrics=collect_metrics,
    )

    metrics_dict = None
    if collect_metrics:
        prediction, metrics_dict = prediction

    if prediction is None:
        logger.debug("Sample %s: v2 returned no prediction (dead end)", qid)
        return None, False

    extra = {}
    if metrics_dict:
        extra["ablation_metrics"] = metrics_dict

    result = _build_result_dict(qid, data["question"], prediction, data["answer"], cond_name, extra=extra)
    return result, True



# ---------------------------------------------------------------------------
# Dataset-level orchestration
# ---------------------------------------------------------------------------


def run_condition(
    model,
    input_builder,
    dataset,
    cond_name,
    ds_dir,
    force_rerun,
    index_len,
    max_new_tokens,
    sample_timeout_s,
    beam_size=5,
    trace=False,
    collect_metrics=False,
    gates_enabled=True,
):
    """Run a single condition and return a metrics dict."""
    pred_path = ds_dir / f"predictions_{cond_name}.jsonl"

    if force_rerun:
        pred_path.unlink(missing_ok=True)

    existing_records, processed_ids, has_partial = safe_read_jsonl(str(pred_path))

    if has_partial:
        logger.warning(
            "Truncated JSONL detected in %s – removing partial final line", pred_path
        )
        if existing_records:
            atomic_write_jsonl(str(pred_path), existing_records)
            processed_ids = {r["id"] for r in existing_records if "id" in r}
        else:
            pred_path.unlink(missing_ok=True)
            existing_records = []
            processed_ids = set()

    n_done = len(processed_ids)
    n_skipped = 0
    n_dead_ends = 0
    t0 = time.time()

    runners = {
        "GCR_Baseline": _run_baseline,
        "DCA_v1_Static": _run_v1,
        "DCA_v2_Dynamic": _run_v2,
        "DCA_v2_NoGates": _run_v2,
    }
    run_fn = runners.get(cond_name)
    if run_fn is None:
        logger.error("Unknown condition: %s", cond_name)
        return {"condition": cond_name, "n": 0, "hits": 0, "hit_at_1": 0.0,
                "time_s": 0, "n_dead_ends": 0, "n_skipped": 0}

    with open(pred_path, "a") as fout:
        for d in dataset:
            qid = d["id"]
            if qid in processed_ids:
                continue

            t_sample = time.time()
            prep = PrepCache.build(d, index_len)

            # Per-question graph stats
            from ablation_metrics import compute_graph_stats
            graph_stats = compute_graph_stats(prep.nx_graph, prep.all_paths)

            extra_kwargs = {}
            if cond_name == "DCA_v2_NoGates":
                extra_kwargs["gates_enabled"] = False
            if collect_metrics:
                extra_kwargs["collect_metrics"] = True

            try:
                with timeout(sample_timeout_s):
                    result, trie_ok = run_fn(
                        model, input_builder, d, qid, cond_name, prep,
                        index_len=index_len,
                        max_new_tokens=max_new_tokens,
                        beam_size=beam_size,
                        **extra_kwargs,
                    )
            except TimeoutError:
                logger.warning("Sample %s timed out after %ds", qid, sample_timeout_s)
                result = _build_result_dict(qid, d["question"], "", d["answer"], cond_name)
                trie_ok = True
            except Exception:
                logger.error("Unhandled error on sample %s:\n%s", qid, traceback.format_exc())
                result = _build_result_dict(qid, d["question"], "", d["answer"], cond_name)
                trie_ok = True

            if result is None:
                n_skipped += 1
                processed_ids.add(qid)
                continue

            if not trie_ok:
                n_dead_ends += 1

            # Attach per-question graph stats and timing
            sample_time = time.time() - t_sample
            result["graph_stats"] = graph_stats
            result["timing_s"] = round(sample_time, 3)

            if trace:
                trace_sample(d, prep, cond_name, result)

            fout.write(json.dumps(result) + "\n")
            fout.flush()
            os.fsync(fout.fileno())
            processed_ids.add(qid)
            n_done += 1

            if n_done % 10 == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                logger.info(
                    "[%s] %d/%d %.2f q/s | %.0fs | skip=%d dead=%d",
                    cond_name, n_done, len(dataset), rate, elapsed,
                    n_skipped, n_dead_ends,
                )

    elapsed = time.time() - t0

    preds = load_preds(str(pred_path))
    hits = compute_hits(preds)
    n = len(preds)

    path_info = {}
    if cond_name == "DCA_v1_Static" and n > 0:
        total_all = sum(p.get("n_paths_all", 0) for p in preds)
        total_filt = sum(p.get("n_paths_filtered", 0) for p in preds)
        path_info = {
            "total_paths_all": total_all,
            "total_paths_filtered": total_filt,
            "reduction_pct": round((1 - total_filt / max(1, total_all)) * 100, 1),
        }

    metrics = {
        "condition": cond_name,
        "n": n,
        "hits": hits,
        "hit_at_1": round(hits / max(1, n) * 100, 1),
        "time_s": round(elapsed, 1),
        "n_dead_ends": n_dead_ends,
        "n_skipped": n_skipped,
        **path_info,
    }

    logger.info(
        "%s: %d questions, Hits@1=%d/%d (%.1f%%), %.0fs, dead_ends=%d, skipped=%d",
        cond_name, n, hits, n, metrics["hit_at_1"], elapsed, n_dead_ends, n_skipped,
    )
    return metrics
