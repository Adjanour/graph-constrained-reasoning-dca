#!/usr/bin/env python3
"""
run.py — DCA-Trie full experiment: GCR baseline vs v1 (static) vs v2 (dynamic).

Runs all three conditions on one or both datasets, with checkpoint/resume.

Usage:
    python experiments/type_oracle_full/run.py                          # both datasets, 50 samples
    python experiments/type_oracle_full/run.py --datasets RoG-webqsp    # one dataset
    python experiments/type_oracle_full/run.py --method v1              # v1 only
    python experiments/type_oracle_full/run.py --method all             # baseline + v1 + v2
    python experiments/type_oracle_full/run.py --max-samples 10
    python experiments/type_oracle_full/run.py --force-rerun
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import torch
from datasets import load_dataset
from tqdm import tqdm

from src.llms import get_registed_model
from src.qa_prompt_builder import PathGenerationWithAnswerPromptBuilder
from src.trie import MarisaTrie
from src.graph_constrained_decoding import GraphConstrainedDecoding
from src.utils.qa_utils import normalize, extract_topk_prediction
from approach3_symbolic.type_oracle import TypeOracle
import src.utils as utils


PATH_START = "<PATH>"
PATH_END = "</PATH>"


# ---------------------------------------------------------------------------
# Trie builders
# ---------------------------------------------------------------------------

def build_filtered_trie(tokenizer, question_dict, index_len, oracle):
    """Build a MarisaTrie from TypeOracle-filtered paths (v1 static)."""
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

    wrapped = [f"{PATH_START}{s}{PATH_END}" for s in filtered_str]
    tokenized = tokenizer(wrapped, padding=False, add_special_tokens=False).input_ids
    trie = MarisaTrie(tokenized, max_token_id=len(tokenizer) + 1)
    return trie, all_paths, filtered


def build_unfiltered_trie(tokenizer, question_dict, index_len):
    """Build a MarisaTrie from all DFS paths (GCR baseline)."""
    g = utils.build_graph(question_dict["graph"], undirected=False)
    entities = question_dict.get("q_entity", [])
    if not entities:
        return None, []

    all_paths = utils.dfs(g, entities, index_len)
    all_str = [utils.path_to_string(p) for p in all_paths]
    if not all_str:
        return None, all_paths

    wrapped = [f"{PATH_START}{s}{PATH_END}" for s in all_str]
    tokenized = tokenizer(wrapped, padding=False, add_special_tokens=False).input_ids
    trie = MarisaTrie(tokenized, max_token_id=len(tokenizer) + 1)
    return trie, all_paths


def build_trie_from_strings(tokenizer, path_strings):
    """Build a MarisaTrie from raw path strings (for v2 iterative expansion)."""
    if not path_strings:
        return None
    wrapped = [f"{PATH_START}{s}{PATH_END}" for s in path_strings]
    tokenized = tokenizer(wrapped, padding=False, add_special_tokens=False).input_ids
    tokenized = [ids + [tokenizer.eos_token_id] for ids in tokenized]
    return MarisaTrie(tokenized, max_token_id=len(tokenizer) + 1)


# ---------------------------------------------------------------------------
# Constrained decoding
# ---------------------------------------------------------------------------

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


def dca_v2_generate(question, start_entities, nx_graph, llm_model,
                    tokenizer, oracle, max_hops=2):
    """
    DCA-Trie v2: iterative hop-by-hop trie expansion (Algorithm 2).
    Start with first-hop gated neighbours, expand at each entity commit.
    """
    answer_types = oracle.infer_answer_types(question)
    start_id = tokenizer.convert_tokens_to_ids(PATH_START)
    end_id = tokenizer.convert_tokens_to_ids(PATH_END)

    first_hop_paths = []
    for entity in start_entities:
        if entity not in nx_graph:
            continue
        for neighbor in nx_graph.neighbors(entity):
            rel = nx_graph[entity][neighbor]["relation"]
            if not oracle.range_gate(rel, neighbor):
                continue
            first_hop_paths.append(f"{entity} -> {rel} -> {neighbor}")

    if not first_hop_paths:
        return None

    current_trie = build_trie_from_strings(tokenizer, first_hop_paths)
    if current_trie is None:
        return None

    prompt = (
        f"Reasoning path is a sequence of triples in the KG that connects the topic entities "
        f"to answer entities. Given the question, generate reasoning paths starting from "
        f"the topic entities to answer the question.\n\n"
        f"# Question:\n{question}\n"
        f"# Topic entities:\n{', '.join(start_entities)}\n"
    )

    output_text = ""
    committed_entity = start_entities[0] if start_entities else None
    hop = 0

    for step in range(max_hops * 3):
        llm_input = llm_model.prepare_model_prompt(prompt)
        gcr = GraphConstrainedDecoding(
            tokenizer, current_trie, start_id, end_id,
            enable_constrained_by_default=False
        )
        inputs = tokenizer(llm_input, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs.input_ids.to(llm_model.model.device)
        attn_mask = inputs.attention_mask.to(llm_model.model.device)

        res = llm_model.model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            generation_config=llm_model.generation_cfg,
            prefix_allowed_tokens_fn=gcr.allowed_tokens_fn,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=32,
        )

        output = tokenizer.decode(
            res.sequences[0][input_ids.shape[1]:], skip_special_tokens=True
        )
        output_text += output

        if PATH_END in output or tokenizer.eos_token in output:
            break

        new_entity = output.replace(PATH_END, "").strip().split(" -> ")[-1].strip() if output else None
        if new_entity is None or new_entity == committed_entity:
            continue
        committed_entity = new_entity
        hop += 1

        if hop >= max_hops:
            break

        is_terminal = (hop + 1 >= max_hops)
        new_paths = []
        if committed_entity in nx_graph:
            for neighbor in nx_graph.neighbors(committed_entity):
                rel = nx_graph[committed_entity][neighbor]["relation"]
                if not oracle.range_gate(rel, neighbor):
                    continue
                if is_terminal and not oracle.type_gate(
                    neighbor, answer_types, hop + 1, max_hops
                ):
                    continue
                new_paths.append(f"{committed_entity} -> {rel} -> {neighbor}")

        if new_paths:
            expanded_trie = build_trie_from_strings(tokenizer, new_paths)
            if expanded_trie is not None:
                current_trie = expanded_trie
                prompt = prompt + f"\n{output.strip()}\n"
        else:
            break

    return output_text


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_processed_ids(path):
    ids = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return ids


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


def compute_hits(preds):
    hits = 0
    for p in preds:
        prediction = p.get("prediction", "")
        answer = list(set(p.get("ground_truth", [])))
        pred_str = " ".join(prediction) if isinstance(prediction, list) else prediction
        top_preds = extract_topk_prediction(pred_str, -1)
        pred_joined = " ".join(top_preds)
        for a in answer:
            if normalize(a) in normalize(pred_joined):
                hits += 1
                break
    return hits


# ---------------------------------------------------------------------------
# Run one condition on one dataset
# ---------------------------------------------------------------------------

def run_condition(model, input_builder, dataset, cond_name, ds_dir, force_rerun):
    """Run a single condition and return metrics dict."""
    pred_path = ds_dir / f"predictions_{cond_name}.jsonl"
    processed = set() if force_rerun else load_processed_ids(str(pred_path))
    fout = open(pred_path, "a" if processed else "w")

    n_done = 0
    n_empty = 0
    n_dead_ends = 0
    t0 = time.time()

    for d in dataset:
        qid = d["id"]
        if qid in processed:
            continue

        oracle = TypeOracle.from_graph(d["graph"])

        if cond_name == "GCR_Baseline":
            trie, all_paths = build_unfiltered_trie(model.tokenizer, d, 2)
            if trie is None:
                result = {"id": qid, "question": d["question"],
                          "prediction": [], "ground_truth": d["answer"],
                          "n_paths_all": 0, "mode": cond_name}
                fout.write(json.dumps(result) + "\n")
                fout.flush()
                processed.add(qid)
                n_done += 1
                n_empty += 1
                continue
            try:
                prediction, ground_paths = run_constrained_decoding(model, input_builder, d, trie)
            except Exception:
                prediction = None
            result = {"id": qid, "question": d["question"],
                      "prediction": prediction or [], "ground_truth": d["answer"],
                      "n_paths_all": len(all_paths), "mode": cond_name}

        elif cond_name == "DCA_v1_Static":
            trie, all_paths, filtered = build_filtered_trie(model.tokenizer, d, 2, oracle)
            if trie is None:
                result = {"id": qid, "question": d["question"],
                          "prediction": [], "ground_truth": d["answer"],
                          "n_paths_all": len(all_paths) if all_paths else 0,
                          "n_paths_filtered": 0, "mode": cond_name}
                fout.write(json.dumps(result) + "\n")
                fout.flush()
                processed.add(qid)
                n_done += 1
                n_empty += 1
                continue
            try:
                prediction, ground_paths = run_constrained_decoding(model, input_builder, d, trie)
            except Exception:
                prediction = None
            result = {"id": qid, "question": d["question"],
                      "prediction": prediction or [], "ground_truth": d["answer"],
                      "n_paths_all": len(all_paths), "n_paths_filtered": len(filtered),
                      "mode": cond_name}

        elif cond_name == "DCA_v2_Dynamic":
            nx_graph = utils.build_graph(d["graph"], undirected=False)
            try:
                prediction = dca_v2_generate(
                    question=d["question"],
                    start_entities=d.get("q_entity", []),
                    nx_graph=nx_graph,
                    llm_model=model,
                    tokenizer=model.tokenizer,
                    oracle=oracle,
                    max_hops=2,
                )
                if prediction is None:
                    n_dead_ends += 1
            except Exception:
                prediction = None
            result = {"id": qid, "question": d["question"],
                      "prediction": prediction or [], "ground_truth": d["answer"],
                      "mode": cond_name}

        fout.write(json.dumps(result) + "\n")
        fout.flush()
        processed.add(qid)
        n_done += 1

        if n_done % 10 == 0:
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            print(f"    [{cond_name}] {n_done}/{len(dataset)} {rate:.2f} q/s | {elapsed:.0f}s")

    fout.close()
    elapsed = time.time() - t0

    preds = load_preds(str(pred_path))
    hits = compute_hits(preds)
    n = len(preds)

    # Path stats for v1
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
        "n": n, "hits": hits,
        "hit_at_1": round(hits / max(1, n) * 100, 1),
        "time_s": round(elapsed, 1),
        "n_dead_ends": n_dead_ends,
        **path_info,
    }

    print(f"    {cond_name}: {n} questions, Hits@1={hits}/{n} ({metrics['hit_at_1']}%), {elapsed:.0f}s")
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DCA-Trie full experiment")
    parser.add_argument("--model-path", default="rmanluo/GCR-Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--data-path", default="rmanluo")
    parser.add_argument("--datasets", nargs="+", default=["RoG-webqsp", "RoG-cwq"],
                        choices=["RoG-webqsp", "RoG-cwq"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--index-len", type=int, default=2)
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--gen-mode", default="group-beam", choices=["greedy", "group-beam", "beam"])
    parser.add_argument("--prompt-mode", default="zero-shot")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--method", default="all", choices=["baseline", "v1", "v2", "all"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    # Output directory
    if args.output_dir:
        output_base = Path(args.output_dir)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_base = Path("results") / "final_experiment" / timestamp
    output_base.mkdir(parents=True, exist_ok=True)

    # GPU / attention
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

    # Config
    config = {
        "model_path": args.model_path, "data_path": args.data_path,
        "datasets": args.datasets, "split": args.split, "index_len": args.index_len,
        "k": args.k, "gen_mode": args.gen_mode, "prompt_mode": args.prompt_mode,
        "max_new_tokens": args.max_new_tokens, "max_samples": args.max_samples,
        "method": args.method, "attn_impl": attn_impl, "gpu": gpu_name,
    }
    with open(output_base / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    for k, v in config.items():
        print(f"  {k:<20} {v}")
    print("=" * 60)

    # Load model
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

    # Conditions to run
    conditions = {
        "baseline": ["GCR_Baseline"],
        "v1": ["DCA_v1_Static"],
        "v2": ["DCA_v2_Dynamic"],
        "all": ["GCR_Baseline", "DCA_v1_Static", "DCA_v2_Dynamic"],
    }[args.method]

    # Run per dataset
    all_summary = {}

    for ds_name in args.datasets:
        print(f"\n{'#' * 60}")
        print(f"  DATASET: {ds_name}")
        print(f"{'#' * 60}")

        dataset = load_dataset(f"{args.data_path}/{ds_name}", split=args.split)
        if args.max_samples and args.max_samples < len(dataset):
            dataset = dataset.select(range(args.max_samples))
        print(f"  Samples: {len(dataset)}")

        ds_dir = output_base / ds_name
        ds_dir.mkdir(exist_ok=True)

        for cond in conditions:
            print(f"\n  Running {cond}...")
            metrics = run_condition(model, input_builder, dataset, cond, ds_dir, args.force_rerun)
            all_summary[(ds_name, cond)] = metrics

    # Final comparison table
    print(f"\n{'=' * 70}")
    print("FINAL RESULTS")
    print(f"{'=' * 70}")
    print(f"{'Dataset':<15} {'Condition':<20} {'N':<6} {'Hits@1':<8} {'Hit%':<8} {'Time':<8}")
    print("-" * 70)
    for (ds, cond), m in all_summary.items():
        print(f"{ds:<15} {cond:<20} {m['n']:<6} {m['hits']:<8} {m['hit_at_1']:<8} {m['time_s']:<8}")
        if "reduction_pct" in m:
            print(f"{'':>15} (paths: {m['total_paths_filtered']}/{m['total_paths_all']}, -{m['reduction_pct']}%)")
    print("=" * 70)

    # Save summary
    summary_out = {f"{ds}|{cond}": m for (ds, cond), m in all_summary.items()}
    with open(output_base / "summary.json", "w") as f:
        json.dump(summary_out, f, indent=2)

    print(f"\nResults saved to {output_base}")


if __name__ == "__main__":
    main()
