#!/usr/bin/env python3
"""
run.py — TypeOracle + KG-Specialized LLM full experiment.

Runs the complete DCA-Trie pipeline:
  1. SIR/FNR evaluation (CPU-only) — measure pruning
  2. Graph-constrained decoding with TypeOracle-filtered trie
  3. Graph-constrained decoding with unfiltered trie (baseline)
  4. Head-to-head comparison: Hit@1, path reduction, timing

Usage:
    python experiments/type_oracle_full/run.py
    python experiments/type_oracle_full/run.py --max-samples 100
    python experiments/type_oracle_full/run.py --force-rerun
    python experiments/type_oracle_full/run.py --output-dir /path/to/results
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import torch
from datasets import load_dataset
from tqdm import tqdm

from src.llms import get_registed_model
from src.qa_prompt_builder import PathGenerationWithAnswerPromptBuilder
from src.trie import MarisaTrie
from src.utils.qa_utils import eval_path_result_w_ans, normalize, extract_topk_prediction
from approach3_symbolic.type_oracle import TypeOracle
import src.utils as utils


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_filtered_trie(tokenizer, question_dict, index_len, oracle):
    """Build a MarisaTrie from TypeOracle-filtered paths."""
    g = utils.build_graph(question_dict["graph"], undirected=False)
    entities = question_dict.get("q_entity", [])
    if not entities:
        return None, [], []

    all_paths = utils.dfs(g, entities, index_len)
    ans_types = oracle.infer_answer_types(question_dict["question"])

    filtered = []
    for p in all_paths:
        admit = True
        for _, rel, tail in p:
            if not oracle.range_gate(rel, tail):
                admit = False
                break
        if admit:
            terminal = p[-1][2]
            if not oracle.type_gate(terminal, ans_types, len(p), index_len):
                admit = False
        if admit:
            filtered.append(p)

    filtered_str = [utils.path_to_string(p) for p in filtered]
    if not filtered_str:
        return None, all_paths, filtered

    PATH_START = "<PATH>"
    PATH_END = "</PATH>"
    wrapped = [f"{PATH_START}{s}{PATH_END}" for s in filtered_str]

    tokenized = tokenizer(wrapped, padding=False, add_special_tokens=False).input_ids
    trie = MarisaTrie(tokenized, max_token_id=len(tokenizer) + 1)
    return trie, all_paths, filtered


def build_unfiltered_trie(tokenizer, question_dict, index_len):
    """Build a MarisaTrie from all DFS paths (no filtering)."""
    g = utils.build_graph(question_dict["graph"], undirected=False)
    entities = question_dict.get("q_entity", [])
    if not entities:
        return None, []

    all_paths = utils.dfs(g, entities, index_len)
    all_str = [utils.path_to_string(p) for p in all_paths]
    if not all_str:
        return None, all_paths

    PATH_START = "<PATH>"
    PATH_END = "</PATH>"
    wrapped = [f"{PATH_START}{s}{PATH_END}" for s in all_str]

    tokenized = tokenizer(wrapped, padding=False, add_special_tokens=False).input_ids
    trie = MarisaTrie(tokenized, max_token_id=len(tokenizer) + 1)
    return trie, all_paths


def run_constrained_decoding(model, input_builder, data, trie):
    """Run graph-constrained decoding for a single question."""
    input_query, ground_paths, _ = input_builder.process_input(data, return_tire=False)
    start_token_ids = model.tokenizer.convert_tokens_to_ids(input_builder.PATH_START_TOKEN)
    end_token_ids = model.tokenizer.convert_tokens_to_ids(input_builder.PATH_END_TOKEN)
    llm_input = model.prepare_model_prompt(input_query)
    prediction = model.generate_sentence(
        llm_input, trie,
        start_token_ids=start_token_ids,
        end_token_ids=end_token_ids,
        enable_constrained_by_default=False,
    )
    return prediction, ground_paths


def load_processed_ids(path):
    """Load already-processed question IDs from a JSONL file."""
    ids = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return ids


def compute_hits(preds):
    """Compute Hits@1 from a list of prediction dicts."""
    hits = 0
    for p in preds:
        prediction = p.get("prediction", "")
        answer = list(set(p.get("ground_truth", [])))
        if isinstance(prediction, list):
            pred_str = " ".join(prediction)
        else:
            pred_str = prediction
        top_preds = extract_topk_prediction(pred_str, -1)
        pred_joined = " ".join(top_preds)
        for a in answer:
            if normalize(a) in normalize(pred_joined):
                hits += 1
                break
    return hits


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TypeOracle full experiment")
    parser.add_argument("--model-path", default="rmanluo/GCR-Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--data-path", default="rmanluo")
    parser.add_argument("--dataset", default="RoG-webqsp", choices=["RoG-webqsp", "RoG-cwq"])
    parser.add_argument("--split", default="test", choices=["test", "validation"])
    parser.add_argument("--index-len", type=int, default=2)
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--gen-mode", default="group-beam", choices=["greedy", "group-beam", "beam"])
    parser.add_argument("--prompt-mode", default="zero-shot", choices=["zero-shot", "few-shot"])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None, help="Subset size (None = full)")
    parser.add_argument("--output-dir", type=str, default=None, help="Results directory")
    parser.add_argument("--force-rerun", action="store_true", help="Ignore checkpoints, start fresh")
    args = parser.parse_args()

    # ── Output directory ────────────────────────────────────────────────
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results") / "type_oracle_experiment" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Attention implementation ────────────────────────────────────────
    has_a100 = False
    flash_attn_installed = False
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        has_a100 = "A100" in gpu_name
        try:
            import flash_attn  # noqa: F401
            flash_attn_installed = True
        except ImportError:
            pass
    else:
        gpu_name = "None"

    attn_impl = "flash_attention_2" if (has_a100 and flash_attn_installed) else "sdpa"

    # ── Save config ─────────────────────────────────────────────────────
    config = {
        "model_path": args.model_path, "data_path": args.data_path,
        "dataset": args.dataset, "split": args.split, "index_len": args.index_len,
        "k": args.k, "gen_mode": args.gen_mode, "prompt_mode": args.prompt_mode,
        "max_new_tokens": args.max_new_tokens, "max_samples": args.max_samples,
        "attn_impl": attn_impl, "gpu": gpu_name, "output_dir": str(output_dir),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    for k, v in config.items():
        print(f"  {k:<20} {v}")
    print("=" * 60)

    # ── Load dataset ────────────────────────────────────────────────────
    dataset = load_dataset(f"{args.data_path}/{args.dataset}", split=args.split)
    print(f"Full test set: {len(dataset)} questions")
    if args.max_samples and args.max_samples < len(dataset):
        dataset = dataset.select(range(args.max_samples))
        print(f"Subsampled to {len(dataset)} questions")

    # ── Phase 1: SIR/FNR evaluation ────────────────────────────────────
    sir_path = output_dir / "sir_fnr_metrics.json"
    if sir_path.exists() and not args.force_rerun:
        print(f"\nLoading existing SIR/FNR from {sir_path}")
        with open(sir_path) as f:
            sir_metrics = json.load(f)
    else:
        print(f"\n{'=' * 60}")
        print("Phase 1: SIR/FNR Evaluation (CPU-only)")
        print(f"{'=' * 60}")

        total_before = total_after = 0
        total_range_blocked = total_type_blocked = 0
        n_range_fn = n_type_fn = 0
        total_gold = 0
        skipped = 0

        t0 = time.time()
        for i, d in enumerate(dataset):
            try:
                oracle = TypeOracle.from_graph(d["graph"])
                ans_types = oracle.infer_answer_types(d["question"])
                g = utils.build_graph(d["graph"], undirected=False)
                entities = d.get("q_entity", [])

                if entities:
                    paths_list = utils.dfs(g, entities, args.index_len)
                    kept = []
                    for p in paths_list:
                        admit = True
                        for _, rel, tail in p:
                            if not oracle.range_gate(rel, tail):
                                total_range_blocked += 1
                                admit = False
                                break
                        if admit:
                            terminal = p[-1][2]
                            if not oracle.type_gate(terminal, ans_types, len(p), args.index_len):
                                total_type_blocked += 1
                                admit = False
                        if admit:
                            kept.append(p)
                    total_before += len(paths_list)
                    total_after += len(kept)
                else:
                    skipped += 1

                truth_paths = utils.get_truth_paths(d["q_entity"], d["a_entity"], g)
                for p in truth_paths:
                    if not p:
                        continue
                    total_gold += 1
                    if any(not oracle.range_gate(rel, tail) for _, rel, tail in p):
                        n_range_fn += 1
                    if not oracle.type_gate(p[-1][2], ans_types, len(p), args.index_len):
                        n_type_fn += 1
            except Exception as e:
                print(f"  Skipping {i}: {e}")
                skipped += 1

            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(dataset)} ({time.time() - t0:.1f}s)")

        elapsed_sir = time.time() - t0
        pruned = total_before - total_after
        sir = pruned / max(1, total_before)

        sir_metrics = {
            "samples": len(dataset), "skipped": skipped,
            "total_paths_raw": total_before, "total_paths_filtered": total_after,
            "pruned": pruned, "sir": round(sir, 4),
            "sir_type": round(total_type_blocked / max(1, total_before), 4),
            "sir_traj": round(total_range_blocked / max(1, total_before), 4),
            "gold_paths": total_gold,
            "fnr_type": round(n_type_fn / max(1, total_gold), 4),
            "fnr_range": round(n_range_fn / max(1, total_gold), 4),
            "elapsed_s": round(elapsed_sir, 1),
        }

        print(f"\nSIR: {sir_metrics['sir']}  |  FNR_type: {sir_metrics['fnr_type']}  |  FNR_range: {sir_metrics['fnr_range']}")
        print(f"Time: {elapsed_sir:.1f}s")

        with open(sir_path, "w") as f:
            json.dump(sir_metrics, f, indent=2)

    # ── Load model ──────────────────────────────────────────────────────
    print(f"\nLoading {args.model_path}...")
    import argparse as _argparse
    LLM = get_registed_model(args.model_path)
    model_args = _argparse.Namespace(
        model_path=args.model_path, model_name=args.model_path,
        k=args.k, generation_mode=args.gen_mode,
        attn_implementation=attn_impl,
        max_new_tokens=args.max_new_tokens, maximun_token=4096,
    )
    t0 = time.time()
    model = LLM(model_args)
    model.prepare_for_inference()
    model.generation_cfg.temperature = None
    model.generation_cfg.top_p = None
    model.generation_cfg.top_k = None
    model.model.generation_config.temperature = None
    model.model.generation_config.top_p = None
    model.model.generation_config.top_k = None
    print(f"Loaded in {time.time() - t0:.1f}s")

    input_builder = PathGenerationWithAnswerPromptBuilder(
        model.tokenizer, args.prompt_mode, index_path_length=args.index_len
    )

    # ── Phase 2: TypeOracle-filtered decoding ──────────────────────────
    filtered_path = output_dir / "predictions_filtered.jsonl"
    processed_filtered = set() if args.force_rerun else load_processed_ids(filtered_path)
    fout_f = open(filtered_path, "a" if processed_filtered else "w")

    print(f"\n{'=' * 60}")
    print(f"Phase 2: TypeOracle-Filtered Decoding ({len(dataset)} questions)")
    print(f"{'=' * 60}")

    n_done = n_empty = 0
    t0 = time.time()

    for i, d in enumerate(dataset):
        qid = d["id"]
        if qid in processed_filtered:
            continue

        oracle = TypeOracle.from_graph(d["graph"])
        trie, all_paths, filtered = build_filtered_trie(
            model.tokenizer, d, args.index_len, oracle
        )

        if trie is None:
            result = {
                "id": qid, "question": d["question"],
                "prediction": [], "ground_truth": d["answer"],
                "ground_truth_paths": [],
                "n_paths_all": len(all_paths), "n_paths_filtered": 0,
                "mode": "typeoracle_filtered",
            }
            fout_f.write(json.dumps(result) + "\n")
            fout_f.flush()
            processed_filtered.add(qid)
            n_done += 1
            n_empty += 1
            continue

        try:
            prediction, ground_paths = run_constrained_decoding(model, input_builder, d, trie)
        except Exception as e:
            print(f"  [{i}] Error: {e}")
            prediction = None

        result = {
            "id": qid, "question": d["question"],
            "prediction": prediction or [], "ground_truth": d["answer"],
            "ground_truth_paths": ground_paths,
            "n_paths_all": len(all_paths), "n_paths_filtered": len(filtered),
            "mode": "typeoracle_filtered",
        }
        fout_f.write(json.dumps(result) + "\n")
        fout_f.flush()
        processed_filtered.add(qid)
        n_done += 1

        if n_done % 10 == 0:
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            print(f"  [{n_done}/{len(dataset)}] {rate:.2f} q/s | {elapsed:.0f}s")

    fout_f.close()
    elapsed_filtered = time.time() - t0
    print(f"Done: {n_done} questions ({n_empty} empty) in {elapsed_filtered:.1f}s")

    # ── Phase 3: Unfiltered baseline decoding ──────────────────────────
    unfiltered_path = output_dir / "predictions_unfiltered.jsonl"
    processed_unfiltered = set() if args.force_rerun else load_processed_ids(unfiltered_path)
    fout_u = open(unfiltered_path, "a" if processed_unfiltered else "w")

    print(f"\n{'=' * 60}")
    print(f"Phase 3: Unfiltered Baseline Decoding ({len(dataset)} questions)")
    print(f"{'=' * 60}")

    n_done_u = 0
    t0 = time.time()

    for i, d in enumerate(dataset):
        qid = d["id"]
        if qid in processed_unfiltered:
            continue

        trie, all_paths = build_unfiltered_trie(model.tokenizer, d, args.index_len)

        if trie is None:
            result = {
                "id": qid, "question": d["question"],
                "prediction": [], "ground_truth": d["answer"],
                "ground_truth_paths": [],
                "n_paths_all": 0, "mode": "unfiltered",
            }
            fout_u.write(json.dumps(result) + "\n")
            fout_u.flush()
            processed_unfiltered.add(qid)
            n_done_u += 1
            continue

        try:
            prediction, ground_paths = run_constrained_decoding(model, input_builder, d, trie)
        except Exception as e:
            print(f"  [{i}] Error: {e}")
            prediction = None

        result = {
            "id": qid, "question": d["question"],
            "prediction": prediction or [], "ground_truth": d["answer"],
            "ground_truth_paths": ground_paths,
            "n_paths_all": len(all_paths), "mode": "unfiltered",
        }
        fout_u.write(json.dumps(result) + "\n")
        fout_u.flush()
        processed_unfiltered.add(qid)
        n_done_u += 1

        if n_done_u % 10 == 0:
            elapsed = time.time() - t0
            rate = n_done_u / elapsed if elapsed > 0 else 0
            print(f"  [{n_done_u}/{len(dataset)}] {rate:.2f} q/s | {elapsed:.0f}s")

    fout_u.close()
    elapsed_unfiltered = time.time() - t0
    print(f"Done: {n_done_u} questions in {elapsed_unfiltered:.1f}s")

    # ── Phase 4: Comparison ────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Phase 4: Comparison")
    print(f"{'=' * 60}")

    def load_preds(path):
        results = []
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return results

    preds_f = load_preds(filtered_path)
    preds_u = load_preds(unfiltered_path)

    n_f = len(preds_f)
    n_u = len(preds_u)
    hits_f = compute_hits(preds_f)
    hits_u = compute_hits(preds_u)

    total_all = sum(p.get("n_paths_all", 0) for p in preds_f)
    total_filtered = sum(p.get("n_paths_filtered", 0) for p in preds_f)
    reduction = (1 - total_filtered / max(1, total_all)) * 100

    comparison = {
        "filtered": {
            "n": n_f, "hits": hits_f,
            "hit_at_1": round(hits_f / max(1, n_f) * 100, 1),
            "avg_paths": round(total_filtered / max(1, n_f), 1),
            "time_s": round(elapsed_filtered, 1),
        },
        "unfiltered": {
            "n": n_u, "hits": hits_u,
            "hit_at_1": round(hits_u / max(1, n_u) * 100, 1),
            "avg_paths": round(total_all / max(1, n_u), 1),
            "time_s": round(elapsed_unfiltered, 1),
        },
        "path_reduction_pct": round(reduction, 1),
        "sir_fnr_metrics": sir_metrics,
    }

    print(f"\n{'Metric':<28} {'Filtered':<12} {'Unfiltered':<12}")
    print("-" * 52)
    print(f"{'Questions':<28} {n_f:<12} {n_u:<12}")
    print(f"{'Hits@1':<28} {hits_f:<12} {hits_u:<12}")
    print(f"{'Hit@1 (%)':<28} {comparison['filtered']['hit_at_1']:<12} {comparison['unfiltered']['hit_at_1']:<12}")
    print(f"{'Avg paths/question':<28} {comparison['filtered']['avg_paths']:<12} {comparison['unfiltered']['avg_paths']:<12}")
    print(f"{'Time (s)':<28} {comparison['filtered']['time_s']:<12} {comparison['unfiltered']['time_s']:<12}")
    print(f"\nPath reduction: {reduction:.1f}%")

    with open(output_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
