"""
predict_symbolic_dca_trie.py
============================
Symbolic DCA-Trie: KG-constrained path generation with ontology-based pruning.

Modes
-----
  --dca_mode v1   Static filtering (Algorithm 1):
                  Build all paths, filter with TypeOracle gates, then
                  run standard constrained decoding over survivors.

  --dca_mode v2   Step-wise expansion (Algorithm 2):
                  No upfront path enumeration. Expand trie dynamically
                  as each entity is committed; gate neighbours through
                  TypeOracle before adding to trie.

Key differences from the GCR baseline
  - No embeddings, no thresholds, no encoder calls
  - All pruning uses TypeOracle symbolic gates (pure set lookups)
  - Reports SIR (Semantic Irrelevance Ratio) decomposed into
    SIR_type (type gate) and SIR_traj (range gate)
"""

import argparse
import json
import os
import sys
from functools import partial
from multiprocessing import Pool
from typing import List, Optional, Tuple

import networkx as nx
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import utils
from src.graph_constrained_decoding import GraphConstrainedDecoding
from src.llms import get_registed_model
from src.qa_prompt_builder import PathGenerationWithAnswerPromptBuilder
from src.trie import MarisaTrie
from src.utils.qa_utils import eval_path_result_w_ans

from approach3_symbolic.type_oracle import TypeOracle

# ============================================================================
# Constants
# ============================================================================

DCA_V1 = "v1"
DCA_V2 = "v2"

# ============================================================================
# Helpers
# ============================================================================


def path_to_string(path: list) -> str:
    result = ""
    for i, p in enumerate(path):
        if i == 0:
            h, r, t = p
            result += f"{h} -> {r} -> {t}"
        else:
            _, r, t = p
            result += f" -> {r} -> {t}"
    return result.strip()


def build_trie_from_paths(path_strings: List[str], tokenizer, eos_token_id: int):
    if len(path_strings) == 0:
        return None
    tokenized = tokenizer(
        path_strings, padding=False, add_special_tokens=False
    ).input_ids
    tokenized = [ids + [eos_token_id] for ids in tokenized]
    return MarisaTrie(tokenized, max_token_id=len(tokenizer) + 1)


def build_symbolically_filtered_trie(
    question_dict,
    tokenizer,
    index_path_length: int,
    undirected: bool = False,
):
    """
    DCA-Trie v1: Static symbolic filtering (Algorithm 1).

    1. Build TypeOracle from graph triples
    2. Infer answer type constraint from question
    3. Enumerate candidate paths (DFS from q_entity, depth = index_path_length)
    4. Gate each path through range_gate (every hop) + type_gate (terminal hop)
    5. Build MarisaTrie from survivors

    Returns (trie, kept_paths, oracle, n_range_blocked, n_type_blocked, n_before)
    """
    oracle = TypeOracle.from_graph(question_dict["graph"])
    answer_types = oracle.infer_answer_types(question_dict["question"])

    entities = question_dict.get("q_entity", [])
    if not entities:
        return None, [], oracle, 0, 0, 0

    if "paths" in question_dict:
        paths_list = question_dict["paths"]
    else:
        g = utils.build_graph(question_dict["graph"], undirected)
        paths_list = utils.dfs(g, entities, index_path_length)

    n_before = len(paths_list)
    if n_before == 0:
        return None, [], oracle, 0, 0, 0

    kept = []
    n_range_blocked = 0
    n_type_blocked = 0

    for p in paths_list:
        admit = True

        for hop_idx, (_, relation, tail_entity) in enumerate(p):
            if not oracle.range_gate(relation, tail_entity):
                n_range_blocked += 1
                admit = False
                break

        if admit:
            terminal = p[-1][2]
            h = len(p)
            if not oracle.type_gate(terminal, answer_types, h, index_path_length):
                n_type_blocked += 1
                admit = False

        if admit:
            kept.append(p)

    kept_str = [path_to_string(p) for p in kept]
    trie = build_trie_from_paths(kept_str, tokenizer, tokenizer.eos_token_id)

    return trie, kept_str, oracle, n_range_blocked, n_type_blocked, n_before


# ============================================================================
# DCA-Trie v2: step-wise expansion generation
# ============================================================================


def extract_last_entity(output_text: str, end_token: str = "</PATH>") -> Optional[str]:
    cleaned = output_text.replace(end_token, "").strip()
    parts = cleaned.split(" -> ")
    return parts[-1].strip() if parts else None


def dca_v2_generate(
    question: str,
    start_entities: List[str],
    graph_triples: List[List[str]],
    nx_graph: nx.Graph,
    llm_model,
    tokenizer,
    oracle: TypeOracle,
    max_hops: int = 2,
    max_new_tokens: int = 32,
) -> Optional[str]:
    """
    DCA-Trie v2: Iterative symbolic expansion (Algorithm 2, 3.7.2).

    At each entity commit:
      1. Fetch neighbours of committed entity
      2. Gate each neighbour through range_gate + type_gate
      3. Build trie from admitted neighbours
      4. Constrained generation step
      5. Extract newly committed entity; repeat.

    Returns the full generated text, or None on failure.
    """
    answer_types = oracle.infer_answer_types(question)
    start_token = "<PATH>"
    end_token = "</PATH>"
    start_id = tokenizer.convert_tokens_to_ids(start_token)
    end_id = tokenizer.convert_tokens_to_ids(end_token)

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

    current_trie = build_trie_from_paths(
        first_hop_paths, tokenizer, tokenizer.eos_token_id
    )
    if current_trie is None:
        return None

    prompt = (
        "Reasoning path is a sequence of triples in the KG that connects the "
        "topic entities to answer entities. Given the question, generate "
        "reasoning paths starting from the topic entities to answer the question.\n\n"
        f"# Question:\n{question}\n"
        f"# Topic entities:\n{', '.join(start_entities)}\n"
    )

    output_text = ""
    committed = start_entities[0] if start_entities else None
    hop = 0

    for _ in range(max_hops * 3):
        llm_input = llm_model.prepare_model_prompt(prompt)
        gcr = GraphConstrainedDecoding(
            tokenizer,
            current_trie,
            start_id,
            end_id,
            enable_constrained_by_default=False,
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
            max_new_tokens=max_new_tokens,
        )

        output = tokenizer.decode(
            res.sequences[0][input_ids.shape[1] :], skip_special_tokens=True
        )
        output_text += output

        if end_token in output or tokenizer.eos_token_id in output:
            break

        new_entity = extract_last_entity(output, end_token)
        if new_entity is None or new_entity == committed:
            continue
        committed = new_entity
        hop += 1

        if hop >= max_hops:
            break

        is_terminal = hop + 1 >= max_hops
        new_paths = []

        if committed in nx_graph:
            for neighbor in nx_graph.neighbors(committed):
                rel = nx_graph[committed][neighbor]["relation"]

                if not oracle.range_gate(rel, neighbor):
                    continue

                if is_terminal and not oracle.type_gate(
                    neighbor, answer_types, hop + 1, max_hops
                ):
                    continue

                new_paths.append(f"{committed} -> {rel} -> {neighbor}")

        if new_paths:
            expanded = build_trie_from_paths(
                new_paths, tokenizer, tokenizer.eos_token_id
            )
            if expanded is not None:
                current_trie = expanded
                prompt = prompt + f"\n{output.strip()}\n"
        else:
            break

    return output_text


# ============================================================================
# SIR (Semantic Irrelevance Ratio) computation
# ============================================================================


def compute_sir_metrics(
    paths_before: int,
    paths_after: int,
    n_range_blocked: int,
    n_type_blocked: int,
    n_questions: int,
) -> dict:
    pruned = paths_before - paths_after
    sir = pruned / max(1, paths_before) if paths_before > 0 else 0.0
    sir_type = n_type_blocked / max(1, paths_before) if paths_before > 0 else 0.0
    sir_traj = n_range_blocked / max(1, paths_before) if paths_before > 0 else 0.0
    return {
        "sir": round(sir, 4),
        "sir_type": round(sir_type, 4),
        "sir_traj": round(sir_traj, 4),
        "pruned_total": pruned,
        "n_range_blocked": n_range_blocked,
        "n_type_blocked": n_type_blocked,
        "paths_before": paths_before,
        "paths_after": paths_after,
    }


def compute_false_negative_rates(
    dataset, index_path_length: int, undirected: bool = False
) -> dict:
    """
    Compute type-gate FNR and range-gate FNR on gold-truth paths.
    A gold path is a false negative if a symbolic gate would have
    excluded it during construction.
    """
    n_type_fn = 0
    n_range_fn = 0
    total_gold = 0

    for data in tqdm(dataset, desc="FNR analysis"):
        g = utils.build_graph(data["graph"], undirected)
        truth_paths = utils.get_truth_paths(data["q_entity"], data["a_entity"], g)
        if not truth_paths:
            continue
        total_gold += len(truth_paths)

        oracle = TypeOracle.from_graph(data["graph"])
        answer_types = oracle.infer_answer_types(data["question"])

        for p in truth_paths:
            range_fail = any(not oracle.range_gate(rel, tail) for _, rel, tail in p)
            if range_fail:
                n_range_fn += 1

            terminal = p[-1][2]
            type_fail = not oracle.type_gate(
                terminal, answer_types, len(p), index_path_length
            )
            if type_fail:
                n_type_fn += 1

    return {
        "fnr_type": round(n_type_fn / max(1, total_gold), 4),
        "fnr_range": round(n_range_fn / max(1, total_gold), 4),
        "n_type_fn": n_type_fn,
        "n_range_fn": n_range_fn,
        "total_gold_paths": total_gold,
    }


# ============================================================================
# Main prediction loop
# ============================================================================


def get_output_file(path: str, force: bool = False):
    if not os.path.exists(path) or force:
        return open(path, "w"), []
    with open(path, "r") as f:
        processed = []
        for line in f:
            try:
                processed.append(json.loads(line)["id"])
            except Exception:
                continue
    return open(path, "a"), processed


def predict_v1(
    data,
    processed_list,
    tokenizer,
    model,
    prompt_builder,
    index_path_length,
    undirected,
):
    qid = data["id"]
    if qid in processed_list:
        return None

    trie, kept_paths, oracle, n_range, n_type, n_before = (
        build_symbolically_filtered_trie(data, tokenizer, index_path_length, undirected)
    )
    if trie is None:
        return None

    input_query, ground_paths, _ = prompt_builder.process_input(data)

    start_ids = tokenizer.convert_tokens_to_ids(prompt_builder.PATH_START_TOKEN)
    end_ids = tokenizer.convert_tokens_to_ids(prompt_builder.PATH_END_TOKEN)

    llm_input = model.prepare_model_prompt(input_query)
    prediction = model.generate_sentence(
        llm_input,
        trie,
        start_token_ids=start_ids,
        end_token_ids=end_ids,
        enable_constrained_by_default=False,
    )

    return {
        "id": qid,
        "question": data["question"],
        "prediction": prediction or "",
        "ground_truth": data["answer"],
        "ground_truth_paths": ground_paths,
        "input": llm_input,
        "n_paths_before": n_before,
        "n_paths_after": len(kept_paths),
        "n_range_blocked": n_range,
        "n_type_blocked": n_type,
        "kept_paths_sample": kept_paths[:5],
    }


def main(args, LLM):
    # ── Load dataset ──
    input_file = os.path.join(args.data_path, args.d)
    dataset = load_dataset(input_file, split=args.split)
    print(f"Loaded {len(dataset)} samples from {args.d}/{args.split}")

    # ── Setup model ──
    model = LLM(args)
    model.prepare_for_inference()

    # ── DCA mode ──
    dca_mode = args.dca_mode
    print(f"DCA-Trie mode: {dca_mode}")

    # ── Output directory ──
    tag = f"DCA-{dca_mode}"
    data_name = args.d + "_undirected" if args.undirected else args.d
    post_fix = f"{args.prompt_mode}-{args.generation_mode}-k{args.k}-index_len{args.index_path_length}-{tag}"
    output_dir = os.path.join(
        args.predict_path, data_name, args.model_name, args.split, post_fix
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output: {output_dir}")

    with open(os.path.join(output_dir, "args.txt"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    fout, processed_list = get_output_file(
        os.path.join(output_dir, "predictions.jsonl"), force=args.force
    )

    # ── Accumulators for SIR ──
    total_before = 0
    total_after = 0
    total_range = 0
    total_type = 0
    total_empty = 0

    if dca_mode == DCA_V1:
        prompt_builder = PathGenerationWithAnswerPromptBuilder(
            model.tokenizer,
            args.prompt_mode,
            index_path_length=args.index_path_length,
            undirected=args.undirected,
            add_rule=args.add_rule,
        )

        for data in tqdm(dataset, desc=f"DCA-Trie {dca_mode}"):
            res = predict_v1(
                data,
                processed_list,
                model.tokenizer,
                model,
                prompt_builder,
                args.index_path_length,
                args.undirected,
            )
            if res is None:
                total_empty += 1
                continue

            total_before += res["n_paths_before"]
            total_after += res["n_paths_after"]
            total_range += res["n_range_blocked"]
            total_type += res["n_type_blocked"]

            fout.write(json.dumps(res) + "\n")
            fout.flush()

    elif dca_mode == DCA_V2:
        for data in tqdm(dataset, desc=f"DCA-Trie {dca_mode}"):
            qid = data["id"]
            if qid in processed_list:
                continue

            g = utils.build_graph(data["graph"], args.undirected)
            oracle = TypeOracle.from_graph(data["graph"])

            prediction = dca_v2_generate(
                question=data["question"],
                start_entities=data.get("q_entity", []),
                graph_triples=data["graph"],
                nx_graph=g,
                llm_model=model,
                tokenizer=model.tokenizer,
                oracle=oracle,
                max_hops=args.index_path_length,
            )

            truth_paths = utils.get_truth_paths(data["q_entity"], data["a_entity"], g)
            ground_paths = [path_to_string(p) for p in truth_paths]

            res = {
                "id": qid,
                "question": data["question"],
                "prediction": prediction or "",
                "ground_truth": data["answer"],
                "ground_truth_paths": ground_paths,
            }
            fout.write(json.dumps(res) + "\n")
            fout.flush()

    fout.close()

    # ── Print pruning statistics (v1 only) ──
    if dca_mode == DCA_V1 and total_before > 0:
        sir = (total_before - total_after) / total_before
        sir_type = total_type / total_before
        sir_traj = total_range / total_before
        print()
        print("=" * 60)
        print("DCA-Trie PRUNING STATISTICS")
        print("=" * 60)
        print(f"Questions processed:     {len(dataset) - total_empty}")
        print(f"Questions with empty KG: {total_empty}")
        print()
        print(f"Total paths (unfiltered): {total_before}")
        print(f"Total paths (filtered):   {total_after}")
        print(f"Pruned:                   {total_before - total_after}")
        print()
        print(f"SIR (overall):            {sir:.4f}")
        print(f"SIR_type (type gate):     {sir_type:.4f}")
        print(f"SIR_traj (range gate):    {sir_traj:.4f}")
        print(f"Range-gate blocked:       {total_range}")
        print(f"Type-gate blocked:        {total_type}")
        print("=" * 60)

        # ── FNR on gold paths ──
        print("\nComputing false-negative rates on gold paths...")
        fnr_dataset = load_dataset(input_file, split=args.split)
        if args.n_data is not None:
            fnr_dataset = fnr_dataset.select(range(min(args.n_data, len(fnr_dataset))))
        fnr = compute_false_negative_rates(
            fnr_dataset, args.index_path_length, args.undirected
        )
        print(f"Gold paths analysed:     {fnr['total_gold_paths']}")
        print(f"Type gate FNR:           {fnr['fnr_type']}  ({fnr['n_type_fn']})")
        print(f"Range gate FNR:          {fnr['fnr_range']}  ({fnr['n_range_fn']})")

    # ── Evaluate ──
    print("\nEvaluating...")
    eval_path_result_w_ans(os.path.join(output_dir, "predictions.jsonl"))


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Symbolic DCA-Trie: KG-constrained path generation with ontology pruning."
    )
    argparser.add_argument("--data_path", type=str, default="rmanluo")
    argparser.add_argument("--d", "-d", type=str, default="RoG-webqsp")
    argparser.add_argument("--split", type=str, default="test[:10]")
    argparser.add_argument("--index_path_length", type=int, default=2)
    argparser.add_argument("--predict_path", type=str, default="results/GenPaths")
    argparser.add_argument(
        "--model_name",
        type=str,
        help="model_name for save results and LLM registry lookup",
        default="gcr-Llama-2-7b-chat-hf",
    )
    argparser.add_argument(
        "--dca_mode",
        type=str,
        default="v1",
        choices=[DCA_V1, DCA_V2],
        help="v1 = static filtering (Algorithm 1), v2 = step-wise expansion (Algorithm 2)",
    )
    argparser.add_argument("--force", action="store_true")
    argparser.add_argument("--n", type=int, default=1, help="number of processes")
    argparser.add_argument(
        "--undirected", type=lambda x: str(x).lower() == "true", default=False
    )
    argparser.add_argument("--debug", action="store_true")
    argparser.add_argument(
        "--prompt_mode",
        type=str,
        default="zero-shot",
        choices=["zero-shot", "mcq-zero-shot", "few-shot"],
    )
    argparser.add_argument("--filter_empty", action="store_true")
    argparser.add_argument("--add_rule", action="store_true")
    argparser.add_argument(
        "--rule_path",
        type=str,
        default="results/gen_rule_path/webqsp_undirected/Llama-2-7b-chat-hf_align-spectoken-joint/test/predictions_3_False.jsonl",
    )
    argparser.add_argument("--prefix", type=str, default="")
    argparser.add_argument(
        "--n_data",
        type=int,
        default=None,
        help="number of samples for FNR analysis (default: all)",
    )

    args, _ = argparser.parse_known_args()

    LLM = get_registed_model(args.model_name)
    LLM.add_args(argparser)

    args = argparser.parse_args()
    main(args, LLM)
