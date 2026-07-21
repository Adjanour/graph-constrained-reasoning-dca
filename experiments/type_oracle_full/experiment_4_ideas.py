"""
experiment_4_ideas.py — Rapid test of 4 improvement ideas on N samples (default 100).

Ideas:
  1. Validate (not filter) — post-hoc TypeOracle rejection of invalid predictions
  2. Small model + TypeOracle — does filtering help a smaller model more?
  3. Adaptive enumeration — limit DFS to max_paths per question (latency/accuracy tradeoff)
  4. Label-level planning — plan at type level, instantiate entity paths

Usage:
  # With Qwen2.5-3B (cached, ~6GB, CPU-friendly):
  python experiments/type_oracle_full/experiment_4_ideas.py \
    --model-path Qwen/Qwen2.5-3B-Instruct \
    --max-samples 100 \
    --methods baseline,filtered,adaptive30,adaptive100,label-plan

  # With GCR 8B (needs GPU):
  python experiments/type_oracle_full/experiment_4_ideas.py \
    --model-path rmanluo/GCR-Meta-Llama-3.1-8B-Instruct \
    --max-samples 10 \
    --methods baseline,filtered,label-plan

Metrics reported:
  - Hits@1 (any correct answer in top-K?)
  - Total time / avg time per question
  - Path count (all / after oracle / after label-plan)
  - For validation (Idea 1): what % of wrong preds would be caught
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
from approach3_symbolic.type_oracle import TypeOracle
from src.graph_constrained_decoding import GraphConstrainedDecoding
from trie_utils import build_unfiltered_trie, build_filtered_trie, build_trie_from_strings
from utils import TimeoutError, timeout, logger, PATH_START, PATH_END

SUPPRESS_HF_WARNINGS = True
if SUPPRESS_HF_WARNINGS:
    import transformers
    transformers.logging.set_verbosity_error()
    import logging as _logging
    _logging.getLogger("transformers").setLevel(_logging.ERROR)
    _logging.getLogger("datasets").setLevel(_logging.ERROR)


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------

def path_to_str(path):
    return graph_utils.path_to_string(path)


def constrained_generate(model, input_builder, data, trie):
    """Run graph-constrained decoding on ANY model using prefix_allowed_tokens_fn."""
    input_query, ground_paths, _ = input_builder.process_input(data, return_tire=False)
    start_token_ids = model.tokenizer.convert_tokens_to_ids(input_builder.PATH_START_TOKEN)
    end_token_ids = model.tokenizer.convert_tokens_to_ids(input_builder.PATH_END_TOKEN)
    llm_input = model.prepare_model_prompt(input_query)

    # Use GraphConstrainedDecoding on the raw model
    gcr = GraphConstrainedDecoding(
        model.tokenizer, trie,
        start_token_ids, end_token_ids,
        enable_constrained_by_default=False,
    )

    inputs = model.tokenizer(llm_input, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs.input_ids.to(model.model.device)
    attention_mask = inputs.attention_mask.to(model.model.device)

    try:
        res = model.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=model.generation_cfg,
            prefix_allowed_tokens_fn=gcr.allowed_tokens_fn,
            return_dict_in_generate=True,
            pad_token_id=model.tokenizer.eos_token_id,
            trust_remote_code=True,
        )
    except Exception as e:
        logger.error("Constrained generation failed: %s", e)
        return None, ground_paths

    if len(res.sequences) == 1:
        pred = model.tokenizer.decode(
            res.sequences[0][input_ids.shape[1]:], skip_special_tokens=True
        )
    else:
        pred = [
            model.tokenizer.decode(r[input_ids.shape[1]:], skip_special_tokens=True)
            for r in res.sequences
        ]
    return pred, ground_paths


def count_entity_types(subgraph):
    """Count how many entities have a given type in the subgraph."""
    type_counts = defaultdict(int)
    for h, r, t in subgraph:
        if r in ("common.topic.notable_types",
                 "freebase.type_hints.included_types",
                 "freebase.type_profile.strict_included_types"):
            if t != "Topic":
                type_counts[t] += 1
    return dict(type_counts)


# ---------------------------------------------------------------------------
# Idea 1: Validate (not filter)
# ---------------------------------------------------------------------------

def run_validate(model, input_builder, data, qid, cond_name, oracle, index_len, **kwargs):
    """
    Run GCR baseline (all paths), then validate predictions with TypeOracle.

    Returns the prediction plus a 'validation' field showing:
      - path_type_pass: did the predicted answer entity match answer types?
      - path_range_pass: did all relation hops pass range gate?
    """
    trie, all_paths = build_unfiltered_trie(model.tokenizer, data, index_len)
    if trie is None:
        return None, False

    prediction, _ = constrained_generate(model, input_builder, data, trie)
    if not prediction:
        return _make_result(qid, data["question"], "", data["answer"], cond_name,
                            extra={"n_paths_all": len(all_paths)}), True

    # Validate: extract answer paths from prediction and check against oracle
    answer_types = oracle.infer_answer_types(data["question"])
    path_type_pass = True
    path_range_pass = True
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

        # TypeOracle validation on paths in the output
        lines = item.split("\n")
        for line in lines:
            if "->" in line and not line.startswith("#"):
                segments = [s.strip() for s in line.split("->")]
                for i in range(1, len(segments) - 1, 2):
                    tail = segments[i + 1] if i + 1 < len(segments) else None
                    rel = segments[i]
                    if tail and rel:
                        if not oracle.range_gate(rel, tail):
                            path_range_pass = False
                if len(segments) >= 3:
                    final_entity = segments[-1].strip()
                    if not oracle.type_gate(final_entity, answer_types, index_len, index_len):
                        path_type_pass = False

    result = _make_result(qid, data["question"], prediction, data["answer"], cond_name,
                          extra={"n_paths_all": len(all_paths),
                                 "validation_type_pass": path_type_pass,
                                 "validation_range_pass": path_range_pass})
    return result, True


# ---------------------------------------------------------------------------
# Idea 2: Small model + TypeOracle (same as existing v1, just with any model)
# ---------------------------------------------------------------------------

def run_small_filtered(model, input_builder, data, qid, cond_name, oracle, index_len, **kwargs):
    """Same as DCA v1: TypeOracle filtered paths, constrained decoding."""
    return _run_generic_v1(model, input_builder, data, qid, cond_name, oracle, index_len)


def run_small_baseline(model, input_builder, data, qid, cond_name, oracle, index_len, **kwargs):
    """Same as GCR baseline: all paths, no filtering."""
    return _run_generic_baseline(model, input_builder, data, qid, cond_name, oracle, index_len)


# ---------------------------------------------------------------------------
# Idea 3: Adaptive enumeration
# ---------------------------------------------------------------------------

def build_adaptive_trie(tokenizer, question_dict, index_len, max_paths):
    """Build trie from at most max_paths DFS paths (first N found)."""
    g = graph_utils.build_graph(question_dict["graph"], undirected=False)
    entities = question_dict.get("q_entity", [])
    if not entities:
        return None, []

    all_paths = graph_utils.dfs(g, entities, index_len)
    if not all_paths:
        return None, all_paths

    # Take only the first max_paths
    sampled = all_paths[:min(max_paths, len(all_paths))]
    sampled_str = [path_to_str(p) for p in sampled]
    wrapped = [f"{PATH_START}{s}{PATH_END}" for s in sampled_str]
    tokenized = tokenizer(wrapped, padding=False, add_special_tokens=False).input_ids
    tokenized = [ids + [tokenizer.eos_token_id] for ids in tokenized]
    trie = MarisaTrie(tokenized, max_token_id=len(tokenizer) + 1)
    return trie, all_paths


def run_adaptive(model, input_builder, data, qid, cond_name, oracle, index_len, max_paths=100, **kwargs):
    """GCR baseline but with adaptive path limit."""
    trie, all_paths = build_adaptive_trie(model.tokenizer, data, index_len, max_paths)
    if trie is None:
        return None, False

    prediction, _ = constrained_generate(model, input_builder, data, trie)
    result = _make_result(qid, data["question"],
                          prediction if prediction else "",
                          data["answer"], cond_name,
                          extra={"n_paths_all": len(all_paths),
                                 "n_paths_used": min(max_paths, len(all_paths))})
    return result, True


# ---------------------------------------------------------------------------
# Idea 4: Label-level planning (reverse/ORT-style)
# ---------------------------------------------------------------------------

def build_label_plan_trie(tokenizer, question_dict, index_len, oracle):
    """
    Label-level path planning.

    1. Get all entity types present in the subgraph
    2. Infer answer types from question
    3. Find candidate answer entities (type match)
    4. Build paths from topic entities only to candidate answer entities
    5. This is like forward DFS but terminates early if entity type doesn't match
    """
    g = graph_utils.build_graph(question_dict["graph"], undirected=False)
    entities = question_dict.get("q_entity", [])
    if not entities:
        return None, []

    all_paths = graph_utils.dfs(g, entities, index_len)
    answer_types = oracle.infer_answer_types(question_dict["question"])
    if not answer_types:
        answer_types = oracle.infer_answer_types_from_paths(all_paths)

    # Filter paths where terminal entity matches answer type
    # (same as DCA v1 type gate, but we also filter intermediate hops by range gate)
    label_paths = []
    for p in all_paths:
        admit = True
        for i, (_, rel, tail) in enumerate(p):
            hop = i + 1
            if not oracle.range_gate(rel, tail):
                admit = False
                break
        if admit:
            terminal = p[-1][2]
            if not oracle.type_gate(terminal, answer_types, len(p), index_len):
                admit = False
        if admit:
            label_paths.append(p)

    if not label_paths:
        return None, all_paths

    label_str = [path_to_str(p) for p in label_paths]
    wrapped = [f"{PATH_START}{s}{PATH_END}" for s in label_str]
    tokenized = tokenizer(wrapped, padding=False, add_special_tokens=False).input_ids
    tokenized = [ids + [tokenizer.eos_token_id] for ids in tokenized]
    trie = MarisaTrie(tokenized, max_token_id=len(tokenizer) + 1)
    return trie, all_paths, label_paths


def run_label_plan(model, input_builder, data, qid, cond_name, oracle, index_len, **kwargs):
    """Label-level planning: only generate paths to type-compatible entities."""
    trie, all_paths, label_paths = build_label_plan_trie(model.tokenizer, data, index_len, oracle)
    if trie is None:
        return None, False

    prediction, _ = constrained_generate(model, input_builder, data, trie)
    result = _make_result(qid, data["question"],
                          prediction if prediction else "",
                          data["answer"], cond_name,
                          extra={"n_paths_all": len(all_paths),
                                 "n_paths_filtered": len(label_paths)})
    return result, True


# ---------------------------------------------------------------------------
# Generic runners (wrapping existing experiment.py functions)
# ---------------------------------------------------------------------------

def _run_generic_baseline(model, input_builder, data, qid, cond_name, oracle, index_len, **kwargs):
    trie, all_paths = build_unfiltered_trie(model.tokenizer, data, index_len)
    if trie is None:
        return None, False
    prediction, _ = constrained_generate(model, input_builder, data, trie)
    result = _make_result(qid, data["question"],
                          prediction if prediction else "",
                          data["answer"], cond_name,
                          extra={"n_paths_all": len(all_paths)})
    return result, True


def _run_generic_v1(model, input_builder, data, qid, cond_name, oracle, index_len, **kwargs):
    trie, all_paths, filtered = build_filtered_trie(model.tokenizer, data, index_len, oracle)
    if trie is None:
        return None, False
    prediction, _ = constrained_generate(model, input_builder, data, trie)
    result = _make_result(qid, data["question"],
                          prediction if prediction else "",
                          data["answer"], cond_name,
                          extra={"n_paths_all": len(all_paths),
                                 "n_paths_filtered": len(filtered)})
    return result, True


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
    """Hits@1: any extracted answer matches ground truth."""
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
# Analysis: Validation (Idea 1) post-hoc
# ---------------------------------------------------------------------------

def analysis_validate(preds, oracle_results):
    """
    From validation condition results: what fraction of wrong predictions
    would have been caught by TypeOracle?
    """
    total = len(preds)
    correct = 0
    wrong = 0
    caught_wrong = 0
    false_positives = 0  # correct preds that validation rejects

    for p in preds:
        gt = list(set(p.get("ground_truth", [])))
        pred_str = " ".join(_extract_answers(p.get("prediction", "")))
        is_correct = bool(gt and eval_hit(pred_str, gt))

        type_pass = p.get("validation_type_pass", True)
        range_pass = p.get("validation_range_pass", True)
        would_reject = not (type_pass and range_pass)

        if is_correct:
            correct += 1
            if would_reject:
                false_positives += 1
        else:
            wrong += 1
            if would_reject:
                caught_wrong += 1

    return {
        "total": total,
        "correct": correct,
        "wrong": wrong,
        "caught_wrong": caught_wrong,
        "catch_rate_pct": round(caught_wrong / max(1, wrong) * 100, 1),
        "false_positives": false_positives,
        "false_positive_rate_pct": round(false_positives / max(1, correct) * 100, 1),
    }


def _extract_answers(prediction):
    answers = []
    items = prediction if isinstance(prediction, list) else [prediction]
    for item in items:
        if "# Answer:\n" in item:
            ans = item.split("# Answer:\n")[-1].strip()
            if ans:
                answers.append(ans)
        elif "# Answer:" in item:
            ans = item.split("# Answer:")[-1].strip()
            if ans:
                answers.append(ans)
    return answers


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Idea 5: v2 step-wise decoding
# ---------------------------------------------------------------------------

def run_v2(model, input_builder, data, qid, cond_name, oracle, index_len, max_new_tokens=256, **kwargs):
    """v2 step-wise hop-by-hop trie expansion."""
    import src.utils as graph_utils
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
    result = _make_result(qid, data["question"], prediction, data["answer"], cond_name)
    return result, True


# ---------------------------------------------------------------------------
# Idea 6: Label-constrained ontology reverse reasoning
# ---------------------------------------------------------------------------

def build_label_reason_trie(tokenizer, question_dict, index_len, oracle):
    """
    ORT-style ontology reverse reasoning + constrained DFS.

    Instead of enumerating all O(E^L) paths and then filtering:
    1. Reverse-reason from aim label → condition label via ontology
    2. DFS constrained to only follow type-compatible entities
    3. This avoids creating the full path set entirely
    """
    from approach3_symbolic.ontology_reasoner import OntologyReasoner
    g = graph_utils.build_graph(question_dict["graph"], undirected=False)
    entities = question_dict.get("q_entity", [])
    if not entities:
        return None, [], []

    all_paths = graph_utils.dfs(g, entities, index_len)
    aim_labels = oracle.infer_answer_types(question_dict["question"])
    if not aim_labels:
        aim_labels = oracle.infer_answer_types_from_paths(all_paths)
    if not aim_labels:
        return None, all_paths, []

    reasoner = OntologyReasoner(oracle)
    condition_labels = set()
    for ent in entities:
        for t in oracle.get_types(ent):
            condition_labels.add(t)

    constrained, label_paths = reasoner.constrained_dfs(
        g, entities, index_len, aim_labels, condition_labels
    )

    if not constrained:
        return None, all_paths, []

    constrained_str = [graph_utils.path_to_string(p) for p in constrained]
    wrapped = [f"{PATH_START}{s}{PATH_END}" for s in constrained_str]
    tokenized = tokenizer(wrapped, padding=False, add_special_tokens=False).input_ids
    tokenized = [ids + [tokenizer.eos_token_id] for ids in tokenized]
    trie = MarisaTrie(tokenized, max_token_id=len(tokenizer) + 1)
    return trie, all_paths, constrained


def run_label_reason(model, input_builder, data, qid, cond_name, oracle, index_len, **kwargs):
    """ORT-style label-constrained DFS with reverse ontology reasoning."""
    trie, all_paths, constrained = build_label_reason_trie(
        model.tokenizer, data, index_len, oracle
    )
    if trie is None:
        return None, False
    prediction, _ = constrained_generate(model, input_builder, data, trie)
    result = _make_result(qid, data["question"],
                          prediction if prediction else "",
                          data["answer"], cond_name,
                          extra={"n_paths_all": len(all_paths),
                                 "n_paths_constrained": len(constrained)})
    return result, True


# ---------------------------------------------------------------------------
# Idea 7: Learned pruning — reranker path scoring + top-K trie
# ---------------------------------------------------------------------------

_RERANKER_CACHE = {}

def get_reranker(model_path):
    if model_path not in _RERANKER_CACHE:
        from experiments.learned_pruning.reranker import PathReranker
        _RERANKER_CACHE[model_path] = PathReranker(model_path)
    return _RERANKER_CACHE[model_path]


def build_reranker_prune_trie(tokenizer, question_dict, index_len, reranker, top_k):
    """Score all paths with reranker, keep only top-K."""
    g = graph_utils.build_graph(question_dict["graph"], undirected=False)
    entities = question_dict.get("q_entity", [])
    if not entities:
        return None, [], []

    all_paths = graph_utils.dfs(g, entities, index_len)
    if not all_paths:
        return None, all_paths, []

    path_strs = [path_to_str(p) for p in all_paths]
    q_text = question_dict.get("question", "")
    ranked = reranker.rank(q_text, path_strs)  # [(idx, path_str, score), ...]
    top_indices = {idx for idx, _, _ in ranked[:min(top_k, len(ranked))]}

    pruned_paths = [all_paths[i] for i in top_indices]
    pruned_str = [path_strs[i] for i in top_indices]
    wrapped = [f"{PATH_START}{s}{PATH_END}" for s in pruned_str]
    tokenized = tokenizer(wrapped, padding=False, add_special_tokens=False).input_ids
    tokenized = [ids + [tokenizer.eos_token_id] for ids in tokenized]
    trie = MarisaTrie(tokenized, max_token_id=len(tokenizer) + 1)
    return trie, all_paths, pruned_paths


def run_reranker_prune(model, input_builder, data, qid, cond_name, oracle, index_len, **kwargs):
    top_k = kwargs.get("top_k", 50)
    reranker_model = kwargs.get("reranker_model_path", None)
    if reranker_model is None:
        logger.error("reranker_model_path required for reranker method")
        return None, False
    reranker = get_reranker(reranker_model)
    trie, all_paths, pruned = build_reranker_prune_trie(
        model.tokenizer, data, index_len, reranker, top_k
    )
    if trie is None:
        return None, False
    prediction, _ = constrained_generate(model, input_builder, data, trie)
    result = _make_result(qid, data["question"],
                          prediction if prediction else "",
                          data["answer"], cond_name,
                          extra={"n_paths_all": len(all_paths),
                                 "n_paths_pruned": len(pruned),
                                 "reranker_top_k": top_k})
    return result, True


# Registry of runner functions
RUNNERS = {
    "baseline": _run_generic_baseline,
    "filtered": _run_generic_v1,
    "validate": run_validate,
    "adaptive30": lambda *a, **kw: run_adaptive(*a, **kw, max_paths=30),
    "adaptive100": lambda *a, **kw: run_adaptive(*a, **kw, max_paths=100),
    "adaptive500": lambda *a, **kw: run_adaptive(*a, **kw, max_paths=500),
    "label-plan": run_label_plan,
    "v2": run_v2,
    "label-reason": run_label_reason,
    "rerank10": lambda *a, **kw: run_reranker_prune(*a, **kw, top_k=10),
    "rerank50": lambda *a, **kw: run_reranker_prune(*a, **kw, top_k=50),
    "rerank100": lambda *a, **kw: run_reranker_prune(*a, **kw, top_k=100),
    "rerank500": lambda *a, **kw: run_reranker_prune(*a, **kw, top_k=500),
}


def run_experiment(args):
    from datasets import load_dataset
    from src.llms import get_registed_model
    from src.qa_prompt_builder import PathGenerationWithAnswerPromptBuilder

    methods = [m.strip() for m in args.methods.split(",")]
    for m in methods:
        if m not in RUNNERS:
            print(f"Unknown method: {m}. Choices: {list(RUNNERS.keys())}")
            sys.exit(1)

    # Setup output
    ts = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"results/4ideas_{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Logging
    log_path = output_dir / "run.log"
    fmt = logging.Formatter("%(asctime)s [%(levelname)-7s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(str(log_path))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root = logging.getLogger("type_oracle")
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.addHandler(fh)
    root.addHandler(ch)

    # Load model
    logger.info("Loading model: %s", args.model_path)
    has_gpu = False
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        pass
    logger.info("GPU available: %s", has_gpu)

    LLM = get_registed_model(args.model_path)
    model_args = argparse.Namespace(
        model_path=args.model_path, model_name=args.model_path,
        k=args.k, generation_mode=args.gen_mode,
        attn_implementation="sdpa",
        max_new_tokens=args.max_new_tokens,
        maximun_token=4096,
        dtype="fp16" if has_gpu else "fp32",
        quant="none",
        chat_model=True, use_assistant_model=False,
    )
    t0 = time.time()
    model = LLM(model_args)
    model.prepare_for_inference()
    model.generation_cfg.temperature = None
    model.generation_cfg.top_p = None
    model.generation_cfg.top_k = None
    if hasattr(model.model, "generation_config"):
        model.model.generation_config.temperature = None
        model.model.generation_config.top_p = None
        model.model.generation_config.top_k = None
    logger.info("Model loaded in %.1fs", time.time() - t0)

    input_builder = PathGenerationWithAnswerPromptBuilder(
        model.tokenizer, "zero-shot", index_path_length=args.index_len
    )

    # Load dataset
    logger.info("Loading dataset: %s (split=%s)", args.dataset, args.split)
    dataset = load_dataset(f"rmanluo/{args.dataset}", split=args.split)
    n_samples = min(args.max_samples, len(dataset))
    dataset = dataset.select(range(n_samples))
    logger.info("Samples: %d", n_samples)

    # Config
    config = {
        "model_path": args.model_path,
        "dataset": args.dataset,
        "split": args.split,
        "index_len": args.index_len,
        "k": args.k,
        "gen_mode": args.gen_mode,
        "max_new_tokens": args.max_new_tokens,
        "n_samples": n_samples,
        "methods": methods,
        "gpu": str(torch.cuda.get_device_name(0)) if has_gpu else "none",
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Run each method
    all_metrics = {}
    for method in methods:
        runner = RUNNERS[method]
        logger.info("=" * 60)
        logger.info("  METHOD: %s", method)
        logger.info("=" * 60)

        pred_path = output_dir / f"predictions_{method}.jsonl"
        processed_ids = set()
        if pred_path.exists():
            with open(pred_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if "id" in rec:
                            processed_ids.add(rec["id"])
                    except json.JSONDecodeError:
                        pass
            logger.info("  Resuming: %d already processed", len(processed_ids))

        n_skipped = 0
        n_dead_ends = 0
        n_timeouts = 0
        total_paths_all = 0
        total_paths_filtered = 0
        total_peak_mem_mb = 0.0
        has_gpu_mem = False
        try:
            import torch
            if torch.cuda.is_available():
                has_gpu_mem = True
        except ImportError:
            pass
        t_start = time.time()

        with open(pred_path, "a") as fout:
            for idx, d in enumerate(dataset):
                qid = d["id"]
                if qid in processed_ids:
                    continue

                if has_gpu_mem:
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    mem_before = torch.cuda.memory_allocated()

                oracle = TypeOracle.from_graph(d["graph"])

                try:
                    to = args.sample_timeout if args.sample_timeout > 0 else 120
                    with timeout(to):
                        result, trie_ok = runner(
                            model, input_builder, d, qid, method, oracle,
                            index_len=args.index_len,
                            max_new_tokens=args.max_new_tokens,
                            reranker_model_path=args.reranker_model_path,
                        )
                except TimeoutError:
                    logger.warning("  [%d/%d] %s timed out", idx + 1, n_samples, qid)
                    result = _make_result(qid, d["question"], "", d["answer"], method)
                    trie_ok = True
                    n_timeouts += 1
                except Exception as e:
                    logger.error("  [%d/%d] %s error: %s", idx + 1, n_samples, qid, e)
                    logger.debug(traceback.format_exc())
                    result = _make_result(qid, d["question"], "", d["answer"], method)
                    trie_ok = True

                if has_gpu_mem and result is not None:
                    torch.cuda.synchronize()
                    peak_mem = torch.cuda.max_memory_allocated()
                    peak_mb = (peak_mem - mem_before) / 1_000_000 if peak_mem > mem_before else 0
                    result["peak_memory_mb"] = round(peak_mb, 1)
                    total_peak_mem_mb += peak_mb

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

                total_paths_all += result.get("n_paths_all", 0)
                total_paths_filtered += result.get("n_paths_filtered", 0)

                if (idx + 1) % 10 == 0:
                    elapsed = time.time() - t_start
                    logger.info("  [%s] %d/%d done | %.1fs | %d skip %d dead %d timeout | peak mem: %.0f MB",
                                method, len(processed_ids), n_samples, elapsed,
                                n_skipped, n_dead_ends, n_timeouts, total_peak_mem_mb / max(1, len(processed_ids)))

        elapsed = time.time() - t_start
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
        n = len(preds)

        avg_peak_mem = round(total_peak_mem_mb / max(1, n), 1) if total_peak_mem_mb > 0 else 0
        metrics = {
            "condition": method,
            "n": n,
            "hits": hits,
            "hit_at_1": round(hits / max(1, n) * 100, 1),
            "time_s": round(elapsed, 1),
            "avg_time_per_q": round(elapsed / max(1, n), 2),
            "avg_peak_memory_mb": avg_peak_mem,
            "n_skipped": n_skipped,
            "n_dead_ends": n_dead_ends,
            "n_timeouts": n_timeouts,
        }
        if n > 0 and method in ("baseline", "filtered", "validate", "label-plan"):
            metrics["avg_paths_all"] = round(total_paths_all / n, 1)
        if n > 0 and method in ("filtered", "label-plan"):
            metrics["avg_paths_filtered"] = round(total_paths_filtered / n, 1)
            if total_paths_all > 0:
                metrics["reduction_pct"] = round(
                    (1 - total_paths_filtered / total_paths_all) * 100, 1
                )
        if n > 0 and method in ("adaptive30", "adaptive100", "adaptive500"):
            total_used = sum(p.get("n_paths_used", 0) for p in preds)
            metrics["avg_paths_used"] = round(total_used / n, 1)

        all_metrics[method] = metrics
        logger.info("  -> %s: Hits@1=%d/%d (%.1f%%) in %.1fs",
                    method, hits, n, metrics["hit_at_1"], elapsed)

    # Validation analysis (Idea 1)
    if "validate" in all_metrics and "validate" in methods:
        preds = []
        with open(output_dir / "predictions_validate.jsonl") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        preds.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        val_analysis = analysis_validate(preds, None)
        all_metrics["_validate_analysis"] = val_analysis
        logger.info("")
        logger.info("=== Validation Analysis (Idea 1) ===")
        logger.info("  Wrong predictions: %d", val_analysis["wrong"])
        logger.info("  Caught by TypeOracle: %d (%.1f%%)",
                     val_analysis["caught_wrong"], val_analysis["catch_rate_pct"])
        logger.info("  False positives (correct preds rejected): %d (%.1f%%)",
                     val_analysis["false_positives"], val_analysis["false_positive_rate_pct"])

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("%s", "FINAL RESULTS".center(80))
    logger.info("=" * 80)
    header = f"{'Method':<15} {'N':>6} {'Hits@1':>8} {'Hit%':>8} {'Time':>8} {'Avg/q':>8}"
    logger.info(header)
    logger.info("-" * 60)
    for method in methods:
        m = all_metrics[method]
        logger.info(f"{method:<15} {m['n']:>6} {m['hits']:>8} {m['hit_at_1']:>7.1f}% "
                     f"{m['time_s']:>7.0f}s {m['avg_time_per_q']:>7.2f}s")

    logger.info("=" * 80)

    # ── Tradeoff curve (baseline + adaptive methods) ─────────────────────
    has_adaptive = any(m.startswith("adaptive") for m in methods)
    if "baseline" in all_metrics and has_adaptive:
        logger.info("")
        logger.info("=" * 80)
        logger.info("%s", "TRADEOFF CURVE: Accuracy vs Latency at Different Path Budgets".center(80))
        logger.info("=" * 80)
        trade_header = f"{'Method':<15} {'Hits@1':>8} {'Δ vs baseline':>14} {'Time':>8} {'Avg/q':>8} {'Path Limit':>12}"
        logger.info(trade_header)
        logger.info("-" * 65)
        bl = all_metrics["baseline"]
        logger.info(f"{'baseline':<15} {bl['hit_at_1']:>7.1f}% {'—':>14} {bl['time_s']:>7.0f}s {bl['avg_time_per_q']:>7.2f}s {'∞':>12}")
        for adapt_name in ["adaptive30", "adaptive100", "adaptive500"]:
            if adapt_name in all_metrics:
                am = all_metrics[adapt_name]
                delta = am["hit_at_1"] - bl["hit_at_1"]
                limit = adapt_name.replace("adaptive", "")
                logger.info(f"{adapt_name:<15} {am['hit_at_1']:>7.1f}% {delta:>+13.1f}% {am['time_s']:>7.0f}s {am['avg_time_per_q']:>7.2f}s {limit:>12}")
        logger.info("-" * 65)

    # ── Filtering narrative (baseline + filtered + label-plan) ──────────
    for pair_name, label in [("filtered", "FILTERING NARRATIVE"), ("label-plan", "LABEL-LEVEL PLANNING")]:
        if pair_name in all_metrics and "baseline" in all_metrics:
            fm = all_metrics[pair_name]
            bl = all_metrics["baseline"]
            delta = fm["hit_at_1"] - bl["hit_at_1"]
            reduction = fm.get("reduction_pct", 0)
            logger.info("")
            logger.info("=" * 80)
            logger.info("%s", f"{label}".center(80))
            logger.info("=" * 80)
            logger.info(f"  {pair_name:<20} {fm['hit_at_1']:>7.1f}% vs baseline {bl['hit_at_1']:>7.1f}%  (Δ = {delta:>+5.1f}%)")
            if reduction:
                logger.info(f"  Path reduction: {reduction}%")
            if "avg_paths_all" in fm:
                logger.info(f"  Avg paths before: {fm['avg_paths_all']}")
            if "avg_paths_filtered" in fm:
                logger.info(f"  Avg paths after:  {fm['avg_paths_filtered']}")
            if "avg_paths_used" in fm:
                logger.info(f"  Avg paths used:   {fm['avg_paths_used']}")

    logger.info("")
    logger.info("=" * 80)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Results: %s", output_dir)
    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="4 Ideas experiment")
    parser.add_argument("--model-path", default="rmanluo/GCR-Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset", default="RoG-webqsp")
    parser.add_argument("--split", default="test")
    parser.add_argument("--index-len", type=int, default=2)
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--gen-mode", default="beam")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--methods", default="baseline,filtered,adaptive100,label-plan",
                        help="Comma-separated: baseline,filtered,validate,adaptive30,adaptive100,adaptive500,label-plan,v2")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--sample-timeout", type=int, default=120)
    parser.add_argument("--reranker-model-path", type=str, default=None,
                        help="Path to fine-tuned reranker model for learned pruning methods")
    args = parser.parse_args()
    run_experiment(args)
