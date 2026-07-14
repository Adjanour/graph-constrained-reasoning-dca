"""
experiment.py — Metrics, per-condition runners, and dataset-level orchestration.

- ``compute_hits`` — Hits@1 evaluation metric
- ``_run_baseline`` / ``_run_v1`` / ``_run_v2`` — single-sample runners
- ``run_condition`` — loops over a dataset for one condition, with checkpoint/resume
"""

import json
import os
import time
import traceback

import src.utils as graph_utils
from approach3_symbolic.type_oracle import TypeOracle
from src.utils.qa_utils import extract_topk_prediction, normalize

from decoding import dca_v2_generate, run_constrained_decoding
from trie_utils import build_filtered_trie, build_unfiltered_trie
from utils import (
    TimeoutError,
    atomic_write_jsonl,
    load_preds,
    logger,
    safe_read_jsonl,
    timeout,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_hits(preds):
    """Compute Hits@1: exact normalized match between top-1 prediction and
    any ground-truth answer."""
    hits = 0
    for p in preds:
        prediction = p.get("prediction", "")
        pred_str = prediction if isinstance(prediction, str) else " ".join(prediction)
        top_preds = extract_topk_prediction(pred_str, -1)
        pred_normalized = normalize(" ".join(top_preds))
        answers = list(set(p.get("ground_truth", [])))
        for a in answers:
            if normalize(a) == pred_normalized:
                hits += 1
                break
    return hits


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


def _run_baseline(model, input_builder, data, qid, cond_name, _oracle, index_len):
    """Run baseline GCR.  Returns (result_dict | None, trie_ok)."""
    trie, all_paths = build_unfiltered_trie(model.tokenizer, data, index_len)
    if trie is None:
        logger.debug("Sample %s: no trie for baseline (no entities/paths)", qid)
        return None, False

    prediction, _ = run_constrained_decoding(model, input_builder, data, trie)
    result = _build_result_dict(
        qid, data["question"],
        prediction if prediction else "",
        data["answer"], cond_name,
        extra={"n_paths_all": len(all_paths)},
    )
    return result, True


def _run_v1(model, input_builder, data, qid, cond_name, oracle, index_len):
    """Run v1 static type-oracle.  Returns (result_dict | None, trie_ok)."""
    trie, all_paths, filtered = build_filtered_trie(model.tokenizer, data, index_len, oracle)
    if trie is None:
        logger.debug("Sample %s: no trie for v1 (no entities/filtered paths)", qid)
        return None, False

    prediction, _ = run_constrained_decoding(model, input_builder, data, trie)
    result = _build_result_dict(
        qid, data["question"],
        prediction if prediction else "",
        data["answer"], cond_name,
        extra={
            "n_paths_all": len(all_paths),
            "n_paths_filtered": len(filtered),
        },
    )
    return result, True


def _run_v2(model, input_builder, data, qid, cond_name, oracle, index_len, max_new_tokens):
    """Run v2 dynamic type-oracle.  Returns (result_dict | None, trie_ok)."""
    nx_graph = graph_utils.build_graph(data["graph"], undirected=False)
    prediction = dca_v2_generate(
        data=data,
        nx_graph=nx_graph,
        llm_model=model,
        tokenizer=model.tokenizer,
        oracle=oracle,
        max_hops=index_len,
        max_new_tokens=max_new_tokens,
        input_builder=input_builder,
    )
    if prediction is None:
        logger.debug("Sample %s: v2 returned no prediction (dead end)", qid)
        return None, False

    result = _build_result_dict(qid, data["question"], prediction, data["answer"], cond_name)
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

            oracle = TypeOracle.from_graph(d["graph"])

            try:
                with timeout(sample_timeout_s):
                    result, trie_ok = run_fn(
                        model, input_builder, d, qid, cond_name, oracle,
                        index_len=index_len,
                        max_new_tokens=max_new_tokens,
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
