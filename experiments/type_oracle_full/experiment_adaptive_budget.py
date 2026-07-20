"""
Adaptive Path Budget: per-question complexity estimation for KG-constrained decoding.

Hypothesis: Most KGQA questions are simple (1-2 hop, unambiguous entities) and
don't need the full path enumeration. By classifying each question's complexity
and allocating path budget accordingly, we can match ~97% of GCR's accuracy at
a fraction of the latency.

Methods:
- adaptive-budget: classify → pick budget (30/100/500)
- adaptive30: fixed 30 (baseline for easy)
- adaptive100: fixed 100 (baseline for medium)
- adaptive500: fixed 500 (baseline for hard)
- baseline: unlimited (GCR)

The output answers two questions:
  1. Does per-question budget allocation outperform any fixed budget?
  2. What is the accuracy-cost Pareto frontier for KG-constrained decoding?
"""
import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from collections import defaultdict

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
for p in [_PROJECT_ROOT, _SCRIPT_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import src.utils as graph_utils
from src.trie import MarisaTrie
from src.utils.qa_utils import eval_hit
from src.graph_constrained_decoding import GraphConstrainedDecoding
from approach3_symbolic.type_oracle import TypeOracle
from trie_utils import build_unfiltered_trie
from utils import TimeoutError, timeout, logger, PATH_START, PATH_END

SUPPRESS_HF_WARNINGS = True
if SUPPRESS_HF_WARNINGS:
    import transformers
    transformers.logging.set_verbosity_error()
    import logging as _logging
    _logging.getLogger("transformers").setLevel(_logging.ERROR)
    _logging.getLogger("datasets").setLevel(_logging.ERROR)


# Budget map
BUDGET_MAP = {
    "easy": 30,
    "medium": 100,
    "hard": 500,
}


# ---------------------------------------------------------------------------
# Question complexity classifier
# ---------------------------------------------------------------------------

def classify_question_complexity(data, oracle):
    """Bin a question into easy/medium/hard based on KG structure and ambiguity.

    Features used:
      - n_start_entities: how many topic entities (entity ambiguity)
      - n_answer_types: how many distinct answer types inferred (type ambiguity)
      - n_neighbors: total 1-hop neighbors from all start entities (KG density)

    Easy:   single entity, few types, sparse neighborhood
    Medium: moderate ambiguity or density
    Hard:   multiple entities or many types or dense neighborhood
    """
    entities = data.get("q_entity", []) or []
    n_start = len(entities)

    answer_types = oracle.infer_answer_types(data["question"])
    n_types = len(answer_types)

    # Count 1-hop neighbors (KG density near the start)
    g = graph_utils.build_graph(data["graph"], undirected=False)
    neighbors = set()
    for e in entities:
        for h, r, t in g:
            if h == e:
                neighbors.add(t)
    n_neighbors = len(neighbors)

    if n_start <= 1 and n_types <= 2 and n_neighbors <= 20:
        return "easy"
    elif n_start <= 3 and n_types <= 5 and n_neighbors <= 100:
        return "medium"
    else:
        return "hard"


# ---------------------------------------------------------------------------
# Per-question adaptive budget trie builder
# ---------------------------------------------------------------------------

def build_trie_with_budget(tokenizer, question_dict, index_len, max_paths):
    """Build trie from at most max_paths DFS paths (first N found)."""
    g = graph_utils.build_graph(question_dict["graph"], undirected=False)
    entities = question_dict.get("q_entity", [])
    if not entities:
        return None, []

    all_paths = graph_utils.dfs(g, entities, index_len)
    if not all_paths:
        return None, all_paths

    sampled = all_paths[:min(max_paths, len(all_paths))]
    sampled_str = [graph_utils.path_to_string(p) for p in sampled]
    wrapped = [f"{PATH_START}{s}{PATH_END}" for s in sampled_str]
    tokenized = tokenizer(wrapped, padding=False, add_special_tokens=False).input_ids
    tokenized = [ids + [tokenizer.eos_token_id] for ids in tokenized]
    trie = MarisaTrie(tokenized, max_token_id=len(tokenizer) + 1)
    return trie, all_paths


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def constrained_generate(model, input_builder, data, trie):
    """Run graph-constrained decoding on ANY model using prefix_allowed_tokens_fn."""
    input_query, ground_paths, _ = input_builder.process_input(data, return_tire=False)
    start_token_ids = model.tokenizer.convert_tokens_to_ids(input_builder.PATH_START_TOKEN)
    end_token_ids = model.tokenizer.convert_tokens_to_ids(input_builder.PATH_END_TOKEN)
    llm_input = model.prepare_model_prompt(input_query)

    gcr = GraphConstrainedDecoding(
        model.tokenizer, start_token_ids=start_token_ids, end_token_ids=end_token_ids
    )
    gcr.set_trie(trie)
    input_ids = model.tokenizer(llm_input, return_tensors="pt").to(model.model.device)

    gen_cfg = model.generation_cfg
    with gcr:
        res = model.model.generate(
            **input_ids,
            max_new_tokens=gen_cfg.get("max_new_tokens", 256),
            do_sample=gen_cfg.get("do_sample", False),
            num_beams=gen_cfg.get("num_beams", 1),
            num_return_sequences=gen_cfg.get("num_return_sequences", 1),
            temperature=gen_cfg.get("temperature"),
            top_p=gen_cfg.get("top_p"),
            top_k=gen_cfg.get("top_k"),
            eos_token_id=model.tokenizer.eos_token_id,
            pad_token_id=model.tokenizer.pad_token_id or model.tokenizer.eos_token_id,
            prefix_allowed_tokens_fn=gcr.allowed_tokens_fn,
            trust_remote_code=True,
        )

    if len(res.sequences) == 1:
        pred = model.tokenizer.decode(
            res.sequences[0][input_ids.input_ids.shape[1]:], skip_special_tokens=True
        )
    else:
        pred = [
            model.tokenizer.decode(r[input_ids.input_ids.shape[1]:], skip_special_tokens=True)
            for r in res.sequences
        ]
    return pred, ground_paths


# ---------------------------------------------------------------------------
# Result builder
# ---------------------------------------------------------------------------

def _make_result(qid, question, prediction_str, ground_truth, cond_name, *, extra=None):
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
# Metrics
# ---------------------------------------------------------------------------

def compute_hits(preds):
    hits = 0
    for p in preds:
        prediction = p.get("prediction", "")
        answers = list(set(p.get("ground_truth", [])))
        if not answers:
            continue
        predicted_answers = []
        items = prediction if isinstance(prediction, list) else [prediction]
        for item in items:
            if "# Answer:\n" in item:
                ans = item.split("# Answer:\n")[-1].strip()
                if ans:
                    predicted_answers.append(ans)
            elif "# Answer:" in item:
                ans = item.split("# Answer:")[-1].strip()
                if ans:
                    predicted_answers.append(ans)
        if not predicted_answers:
            continue
        pred_str = " ".join(predicted_answers)
        if eval_hit(pred_str, answers):
            hits += 1
    return hits


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def _run_baseline(model, input_builder, data, qid, cond_name, oracle, index_len, **kwargs):
    trie, all_paths = build_unfiltered_trie(model.tokenizer, data, index_len)
    if trie is None:
        return None, False
    prediction, _ = constrained_generate(model, input_builder, data, trie)
    result = _make_result(qid, data["question"],
                          prediction if prediction else "",
                          data["answer"], cond_name,
                          extra={"n_paths_total": len(all_paths)})
    return result, True


def _run_fixed_budget(model, input_builder, data, qid, cond_name, oracle, index_len,
                      max_paths=100, **kwargs):
    trie, all_paths = build_trie_with_budget(model.tokenizer, data, index_len, max_paths)
    if trie is None:
        return None, False
    prediction, _ = constrained_generate(model, input_builder, data, trie)
    result = _make_result(qid, data["question"],
                          prediction if prediction else "",
                          data["answer"], cond_name,
                          extra={"n_paths_total": len(all_paths),
                                 "budget": max_paths,
                                 "n_paths_used": min(max_paths, len(all_paths))})
    return result, True


def _run_adaptive_budget(model, input_builder, data, qid, cond_name, oracle, index_len, **kwargs):
    complexity = classify_question_complexity(data, oracle)
    budget = BUDGET_MAP[complexity]
    trie, all_paths = build_trie_with_budget(model.tokenizer, data, index_len, budget)
    if trie is None:
        return None, False
    prediction, _ = constrained_generate(model, input_builder, data, trie)
    result = _make_result(qid, data["question"],
                          prediction if prediction else "",
                          data["answer"], cond_name,
                          extra={"complexity": complexity,
                                 "budget": budget,
                                 "n_paths_total": len(all_paths),
                                 "n_paths_used": min(budget, len(all_paths))})
    return result, True


def _run_v2(model, input_builder, data, qid, cond_name, oracle, index_len, max_new_tokens=256, **kwargs):
    """v2: step-wise dynamic expansion (context-aware, never enumerates all paths)."""
    from decoding import dca_v2_generate
    nx_graph = graph_utils.build_graph(data["graph"], undirected=False)
    prediction = dca_v2_generate(
        data=data, nx_graph=nx_graph, llm_model=model,
        tokenizer=model.tokenizer, oracle=oracle,
        max_hops=index_len, max_new_tokens=max_new_tokens,
        input_builder=input_builder,
    )
    if prediction is None:
        return None, False
    result = _make_result(qid, data["question"], prediction, data["answer"], cond_name,
                          extra={"approach": "dynamic"})
    return result, True


# Registry
RUNNERS = {
    "baseline": _run_baseline,
    "adaptive30": lambda *a, **kw: _run_fixed_budget(*a, **kw, max_paths=30),
    "adaptive100": lambda *a, **kw: _run_fixed_budget(*a, **kw, max_paths=100),
    "adaptive500": lambda *a, **kw: _run_fixed_budget(*a, **kw, max_paths=500),
    "adaptive-budget": _run_adaptive_budget,
    "v2": _run_v2,
}


# ---------------------------------------------------------------------------
# Classifier analysis (post-hoc)
# ---------------------------------------------------------------------------

def analyse_classification(preds):
    """For the adaptive-budget method, show accuracy per complexity bin."""
    bins = defaultdict(lambda: {"total": 0, "hits": 0})
    for p in preds:
        c = p.get("complexity", "unknown")
        bins[c]["total"] += 1
        prediction = p.get("prediction", "")
        answers = list(set(p.get("ground_truth", [])))
        if not answers:
            continue
        predicted_answers = []
        items = prediction if isinstance(prediction, list) else [prediction]
        for item in items:
            if "# Answer:\n" in item:
                ans = item.split("# Answer:\n")[-1].strip()
                if ans:
                    predicted_answers.append(ans)
            elif "# Answer:" in item:
                ans = item.split("# Answer:")[-1].strip()
                if ans:
                    predicted_answers.append(ans)
        if not predicted_answers:
            continue
        pred_str = " ".join(predicted_answers)
        if eval_hit(pred_str, answers):
            bins[c]["hits"] += 1
    return bins


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(args):
    from datasets import load_dataset
    from src.llms import get_registed_model
    from src.qa_prompt_builder import PathGenerationWithAnswerPromptBuilder

    methods = [m.strip() for m in args.methods.split(",")]
    for m in methods:
        if m not in RUNNERS:
            print(f"Unknown method: {m}. Choices: {list(RUNNERS.keys())}")
            sys.exit(1)

    ts = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"results/adaptive_budget_{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "run.log"
    fmt = logging.Formatter("%(asctime)s [%(levelname)-7s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.info("Adaptive Budget Experiment — %s", ts)
    logger.info("Methods: %s", methods)

    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    LLM = get_registed_model(args.model_path)
    model_args = argparse.Namespace(
        model_name=args.model_path,
        max_seq_length=args.model_max_seq_length if hasattr(args, "model_max_seq_length") else 1536,
        max_new_tokens=args.generation_max_length if hasattr(args, "generation_max_length") else 256,
        top_p=args.nucleus_probability if hasattr(args, "nucleus_probability") else None,
        temperature=args.temperature if hasattr(args, "temperature") else None,
    )
    model = LLM(model_args)
    model.prepare_for_inference()

    cfg = model.generation_cfg
    if hasattr(cfg, "temperature"):
        cfg["temperature"] = None
    if hasattr(cfg, "top_p"):
        cfg["top_p"] = None
    if hasattr(cfg, "top_k"):
        cfg["top_k"] = None

    input_builder = PathGenerationWithAnswerPromptBuilder(
        model.tokenizer, "zero-shot", index_path_length=args.index_len
    )

    dataset = load_dataset(f"rmanluo/{args.dataset}", split=args.split)
    n_samples = min(args.max_samples, len(dataset))
    dataset = dataset.select(range(n_samples))
    logger.info("Dataset: %s/%s (%d samples)", args.dataset, args.split, n_samples)

    all_metrics = {}
    for method in methods:
        runner = RUNNERS[method]
        logger.info("")
        logger.info("=" * 60)
        logger.info("Running method: %s", method)
        logger.info("=" * 60)

        pred_path = output_dir / f"predictions_{method}.jsonl"
        processed_ids = set()
        if pred_path.exists():
            with open(pred_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            processed_ids.add(json.loads(line)["id"])
                        except (json.JSONDecodeError, KeyError):
                            pass
            logger.info("Resuming: %d already processed", len(processed_ids))

        n = 0
        n_skipped = 0
        n_dead_ends = 0
        n_timeouts = 0
        t0 = time.time()

        with open(pred_path, "a") as fout:
            for idx, d in enumerate(dataset):
                qid = d["id"]
                if qid in processed_ids:
                    continue

                oracle = TypeOracle.from_graph(d["graph"])

                try:
                    with timeout(args.sample_timeout or 120):
                        result, trie_ok = runner(
                            model, input_builder, d, qid, method, oracle,
                            index_len=args.index_len,
                            max_new_tokens=args.max_new_tokens,
                        )
                except TimeoutError:
                    result = _make_result(qid, d["question"], "", d["answer"], method)
                    trie_ok = True
                    n_timeouts += 1
                except Exception as e:
                    logger.warning("Error on qid=%s: %s", qid, e)
                    traceback.print_exc()
                    result = _make_result(qid, d["question"], "", d["answer"], method)
                    trie_ok = True

                if result is None or not trie_ok:
                    n_dead_ends += 1
                    continue

                fout.write(json.dumps(result) + "\n")
                n += 1
                processed_ids.add(qid)

                if (n + n_dead_ends) % 50 == 0:
                    logger.info("  %s: %d done (%d dead, %d timeout, %.1fs)",
                                method, n, n_dead_ends, n_timeouts, time.time() - t0)

        elapsed = time.time() - t0
        if n == 0:
            logger.warning("  No successful samples for %s", method)
            all_metrics[method] = {"condition": method, "n": 0, "hits": 0,
                                    "hit_at_1": 0.0, "time_s": elapsed,
                                    "avg_time_per_q": 0.0}
            continue

        preds = []
        with open(pred_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        preds.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        hits = compute_hits(preds)
        metrics = {
            "condition": method,
            "n": n,
            "hits": hits,
            "hit_at_1": round(hits / n * 100, 1),
            "time_s": round(elapsed),
            "avg_time_per_q": round(elapsed / max(1, n), 2),
            "n_skipped": n_skipped,
            "n_dead_ends": n_dead_ends,
            "n_timeouts": n_timeouts,
        }

        # Extra per-method stats
        if method == "adaptive-budget":
            bins = analyse_classification(preds)
            metrics["complexity_bins"] = {
                k: {"total": v["total"], "hits": v["hits"],
                    "hit_at_1": round(v["hits"] / max(1, v["total"]) * 100, 1)}
                for k, v in sorted(bins.items())
            }
            avg_budget = sum(p.get("budget", 0) for p in preds) / max(1, len(preds))
            metrics["avg_budget"] = round(avg_budget, 1)

        all_metrics[method] = metrics
        logger.info("  %s: %d samples, %d hits (%.1f%%), %.1fs total (%.2f/q)",
                    method, n, hits, metrics["hit_at_1"], elapsed, metrics["avg_time_per_q"])

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 80)
    logger.info("%s", "ADAPTIVE BUDGET EXPERIMENT — FINAL RESULTS".center(80))
    logger.info("=" * 80)

    header = f"{'Method':<20} {'N':>6} {'Hits@1':>8} {'Time':>8} {'Avg/q':>8} {'Budget':>8}"
    logger.info(header)
    logger.info("-" * 60)
    for method in methods:
        m = all_metrics[method]
        budget_col = ""
        if method == "adaptive-budget":
            budget_col = f"{m.get('avg_budget', ''):>8}"
        elif method == "baseline":
            budget_col = f"{'∞':>8}"
        elif method == "v2":
            budget_col = f"{'dynamic':>8}"
        else:
            budget_col = f"{method.replace('adaptive', ''):>8}"
        logger.info(f"{method:<20} {m['n']:>6} {m['hit_at_1']:>7.1f}% "
                     f"{m['time_s']:>7.0f}s {m['avg_time_per_q']:>7.2f}s {budget_col}")

    logger.info("-" * 60)
    if "baseline" in all_metrics:
        bl = all_metrics["baseline"]
        for method in methods:
            if method == "baseline":
                continue
            m = all_metrics[method]
            delta = m["hit_at_1"] - bl["hit_at_1"]
            latency_savings = (bl["time_s"] - m["time_s"]) / max(1, bl["time_s"]) * 100
            logger.info(f"  {method:<18} Δ = {delta:>+5.1f}pp, "
                         f"latency saved: {latency_savings:.0f}%")

    logger.info("")

    # ── Tradeoff curve ───────────────────────────────────────────────────
    all_runs = [m for m in methods if m in all_metrics]
    logger.info("%s", "ACCURACY-COST PARETO FRONTIER".center(80))
    logger.info("-" * 60)
    logger.info(f"{'Method':<20} {'Hits@1':>8} {'Avg/q (s)':>10} {'Paths':>8}")
    logger.info("-" * 60)
    for method in all_runs:
        m = all_metrics[method]
        paths = ""
        if method == "baseline":
            paths = "all"
        elif method == "adaptive-budget":
            paths = "auto"
        elif method == "v2":
            paths = "dynamic"
        else:
            paths = method.replace("adaptive", "")
        logger.info(f"{method:<20} {m['hit_at_1']:>7.1f}% {m['avg_time_per_q']:>9.2f}s {paths:>8}")
    logger.info("-" * 60)

    # ── Per-complexity breakdown ─────────────────────────────────────────
    if "adaptive-budget" in all_metrics:
        ab = all_metrics["adaptive-budget"]
        bins = ab.get("complexity_bins", {})
        if bins:
            logger.info("")
            logger.info("%s", "ADAPTIVE BUDGET: PER-COMPLEXITY BREAKDOWN".center(80))
            logger.info("-" * 60)
            logger.info(f"{'Complexity':<15} {'N':>6} {'Hits@1':>8} {'Budget':>8}")
            logger.info("-" * 60)
            for level in ["easy", "medium", "hard"]:
                b = bins.get(level, {"total": 0, "hit_at_1": 0})
                logger.info(f"{level:<15} {b['total']:>6} {b['hit_at_1']:>7.1f}% "
                             f"{BUDGET_MAP.get(level, 0):>8}")
            logger.info("-" * 60)

    logger.info("")
    logger.info("=" * 80)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Results: %s", output_dir)
    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive Budget Experiment")
    parser.add_argument("--model-path", default="rmanluo/GCR-Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset", default="RoG-webqsp")
    parser.add_argument("--split", default="test")
    parser.add_argument("--index-len", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--methods", default="baseline,adaptive30,adaptive100,adaptive500,adaptive-budget,v2",
                        help="Comma-separated: baseline,adaptive30,adaptive100,adaptive500,adaptive-budget,v2")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--sample-timeout", type=int, default=120)
    args = parser.parse_args()
    run_experiment(args)
