"""
run_symbolic_experiment.py
==========================
TypeOracle symbolic DCA-Trie experiment: two phases.

Phase 1 — SIR/FNR (CPU only):
    Evaluate TypeOracle gate pruning across the full WebQSP test set.
    Measures SIR (Semantic Irrelevance Ratio) and false-negative rates
    on gold paths. No GPU required.

Phase 2 — Proxy answer generation (Intel GPU / llama.cpp):
    Uses the llama.cpp proxy model (Qwen2.5-3B Q4_K_M) with
    TypeOracle-filtered paths in the prompt to generate answers.
    Compares Hit@1 with and without TypeOracle filtering.
    Requires llama.cpp server running on port 8080.

Usage:
    # Phase 1: SIR/FNR (CPU, full test set)
    python workflow/run_symbolic_experiment.py --phase sir

    # Phase 2: Proxy answers with llama.cpp (Intel GPU, subset)
    # First start llama-server, then:
    python workflow/run_symbolic_experiment.py --phase proxy --n 10

    # Both phases sequentially:
    python workflow/run_symbolic_experiment.py --phase all --n 10

Full experiment (needs A100 40GB):
    See approach3_symbolic/EXPERIMENT_RESULTS.md §"Full Pipeline"
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import utils
from approach3_symbolic.type_oracle import TypeOracle


# ============================================================================
# Phase 1: SIR / FNR (CPU-only)
# ============================================================================


def run_sir_fnr(split: str = "test", index_path_length: int = 2):
    from datasets import load_dataset

    ds = load_dataset("rmanluo/RoG-webqsp", split=split)
    print(f"\n{'=' * 60}")
    print(f"Phase 1: SIR/FNR — {len(ds)} samples (CPU)")
    print(f"{'=' * 60}")

    total_before = total_after = total_range = total_type = 0
    n_type_fn = n_range_fn = total_gold = 0
    skipped_sir = skipped_fnr = 0

    for i, d in enumerate(ds):
        try:
            oracle = TypeOracle.from_graph(d["graph"])
            ans_types = oracle.infer_answer_types(d["question"])
            g = utils.build_graph(d["graph"], undirected=False)
            entities = d.get("q_entity", [])

            # SIR
            if entities:
                paths_list = utils.dfs(g, entities, index_path_length)
                kept = []
                for p in paths_list:
                    admit = True
                    for _, rel, tail in p:
                        if not oracle.range_gate(rel, tail):
                            total_range += 1
                            admit = False
                            break
                    if admit:
                        terminal = p[-1][2]
                        if not oracle.type_gate(
                            terminal, ans_types, len(p), index_path_length
                        ):
                            total_type += 1
                            admit = False
                    if admit:
                        kept.append(p)
                total_before += len(paths_list)
                total_after += len(kept)
            else:
                skipped_sir += 1

            # FNR on gold paths
            truth_paths = utils.get_truth_paths(d["q_entity"], d["a_entity"], g)
            for p in truth_paths:
                if not p:
                    continue
                total_gold += 1
                if any(not oracle.range_gate(rel, tail) for _, rel, tail in p):
                    n_range_fn += 1
                if not oracle.type_gate(p[-1][2], ans_types, len(p), index_path_length):
                    n_type_fn += 1

        except Exception as e:
            print(f"  Skipping sample {i}: {e}")
            skipped_fnr += 1
            continue

        if (i + 1) % 500 == 0:
            print(f"  processed {i + 1}/{len(ds)}...")

    pruned = total_before - total_after
    sir = pruned / max(1, total_before)
    metrics = {
        "samples": len(ds),
        "skipped_sir": skipped_sir,
        "skipped_fnr": skipped_fnr,
        "total_paths_raw": total_before,
        "total_paths_filtered": total_after,
        "pruned": pruned,
        "sir": round(sir, 4),
        "sir_type": round(total_type / max(1, total_before), 4),
        "sir_traj": round(total_range / max(1, total_before), 4),
        "n_range_blocked": total_range,
        "n_type_blocked": total_type,
        "gold_paths_analysed": total_gold,
        "fnr_type": round(n_type_fn / max(1, total_gold), 4),
        "fnr_range": round(n_range_fn / max(1, total_gold), 4),
        "n_type_fn": n_type_fn,
        "n_range_fn": n_range_fn,
    }

    print(f"\n{'=' * 60}")
    print(f"RESULTS — SIR/FNR")
    print(f"{'=' * 60}")
    print(f"Samples:               {metrics['samples']}")
    print(f"Total paths (raw):     {metrics['total_paths_raw']}")
    print(f"Total paths (filtered):{metrics['total_paths_filtered']}")
    print(f"Pruned:                {metrics['pruned']}")
    print(f"SIR (overall):         {metrics['sir']}")
    print(f"SIR_type (type gate):  {metrics['sir_type']}")
    print(f"SIR_traj (range gate): {metrics['sir_traj']}")
    print(f"Range-gate blocked:    {metrics['n_range_blocked']}")
    print(f"Type-gate blocked:     {metrics['n_type_blocked']}")
    print()
    print(f"Gold paths analysed:   {metrics['gold_paths_analysed']}")
    print(f"Type gate FNR:         {metrics['fnr_type']}  ({metrics['n_type_fn']})")
    print(f"Range gate FNR:        {metrics['fnr_range']}  ({metrics['n_range_fn']})")
    print(f"{'=' * 60}\n")

    return metrics


# ============================================================================
# Phase 2: Proxy model answer generation (llama.cpp / Intel GPU)
# ============================================================================


def check_llama_server(host="localhost", port=8080):
    import urllib.request
    import json

    try:
        url = f"http://{host}:{port}/v1/models"
        resp = urllib.request.urlopen(url, timeout=5)
        data = json.loads(resp.read())
        models = [m["id"] for m in data.get("data", [])]
        if models:
            print(f"  Connected to llama.cpp: {models[0]}")
            return True
    except Exception:
        pass
    return False


def paths_to_text(paths, max_paths=20):
    if not paths:
        return "No paths found."
    lines = []
    for i, p in enumerate(paths[:max_paths]):
        lines.append(f"  {i + 1}. {p}")
    if len(paths) > max_paths:
        lines.append(f"  ... and {len(paths) - max_paths} more")
    return "\n".join(lines)


def ask_model(client, question, path_context, model_id):
    prompt = (
        "Given the question and possible reasoning paths from a knowledge graph, "
        "determine the correct answer. Base your answer ONLY on the provided paths.\n\n"
        f"Question: {question}\n\n"
        f"Reasoning paths:\n{path_context}\n\n"
        "Answer:"
    )
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=64,
    )
    return resp.choices[0].message.content.strip()


def run_proxy_answer(n_questions: int = 10, split: str = "test", port: int = 8080):
    from datasets import load_dataset
    from openai import OpenAI

    if not check_llama_server(port=port):
        print("ERROR: llama.cpp server not running on port {port}.")
        print("  Start it with:")
        print(f"    llama-server --hf-repo Qwen/Qwen2.5-3B-Instruct-GGUF \\")
        print(f"      --hf-file qwen2.5-3b-instruct-q4_k_m.gguf -ngl 999 --port {port}")
        return None

    client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{port}/v1")
    model_id = "Qwen/Qwen2.5-3B-Instruct-GGUF"
    ds = load_dataset("rmanluo/RoG-webqsp", split=f"{split}[:{n_questions}]")

    print(f"\n{'=' * 60}")
    print(f"Phase 2: Proxy answer generation — {len(ds)} questions")
    print(f"  Server: http://localhost:{port}")
    print(f"  Model: {model_id}")
    print(f"{'=' * 60}")

    results = []
    hits_filtered = 0
    hits_unfiltered = 0

    for i, d in enumerate(ds):
        q = d["question"]
        gt = d["answer"]
        oracle = TypeOracle.from_graph(d["graph"])
        ans_types = oracle.infer_answer_types(q)
        g = utils.build_graph(d["graph"], undirected=False)
        entities = d.get("q_entity", [])

        all_paths = utils.dfs(g, entities, 2) if entities else []
        all_str = [utils.path_to_string(p) for p in all_paths]

        # TypeOracle filtering
        kept = []
        for p in all_paths:
            admit = True
            for _, rel, tail in p:
                if not oracle.range_gate(rel, tail):
                    admit = False
                    break
            if admit:
                if not oracle.type_gate(p[-1][2], ans_types, len(p), 2):
                    admit = False
            if admit:
                kept.append(p)
        kept_str = [utils.path_to_string(p) for p in kept]

        # Ask model with filtered paths
        ctx_filtered = paths_to_text(kept_str)
        t0 = time.time()
        pred_filtered = ask_model(client, q, ctx_filtered, model_id)
        dt_filtered = time.time() - t0
        hit_f = any(a.lower() in pred_filtered.lower() for a in gt)

        # Ask model with unfiltered paths (for comparison)
        ctx_all = paths_to_text(all_str)
        t0 = time.time()
        pred_all = ask_model(client, q, ctx_all, model_id)
        dt_all = time.time() - t0
        hit_a = any(a.lower() in pred_all.lower() for a in gt)

        hits_filtered += hit_f
        hits_unfiltered += hit_a

        results.append(
            {
                "id": d["id"],
                "question": q,
                "ground_truth": gt,
                "n_paths_all": len(all_str),
                "n_paths_filtered": len(kept_str),
                "n_range_blocked": sum(
                    1
                    for p in all_paths
                    if any(not oracle.range_gate(rel, tail) for _, rel, tail in p)
                ),
                "n_type_blocked": sum(
                    1
                    for p in all_paths
                    if all(oracle.range_gate(rel, tail) for _, rel, tail in p)
                    and not oracle.type_gate(p[-1][2], ans_types, len(p), 2)
                ),
                "pred_filtered": pred_filtered,
                "pred_unfiltered": pred_all,
                "hit_filtered": hit_f,
                "hit_unfiltered": hit_a,
                "time_filtered_s": round(dt_filtered, 2),
                "time_unfiltered_s": round(dt_all, 2),
            }
        )

        n = len(results)
        print(
            f"  [{i}] {q[:55]:55s}  "
            f"paths: {len(all_str):>4}→{len(kept_str):<4}  "
            f"Hit: F={int(hit_f)} U={int(hit_a)}  "
            f"({hits_filtered}/{n} vs {hits_unfiltered}/{n})"
        )

    n = len(results)
    print(f"\n{'=' * 60}")
    print(f"RESULTS — Proxy Answer Generation")
    print(f"{'=' * 60}")
    print(f"Questions:             {n}")
    print(f"Avg paths (raw):       {sum(r['n_paths_all'] for r in results) / n:.0f}")
    print(
        f"Avg paths (filtered):  {sum(r['n_paths_filtered'] for r in results) / n:.0f}"
    )
    print(
        f"Avg path reduction:    {(1 - sum(r['n_paths_filtered'] for r in results) / max(1, sum(r['n_paths_all'] for r in results))) * 100:.1f}%"
    )
    print()
    print(
        f"Unfiltered Hit@1:      {hits_unfiltered}/{n} = {hits_unfiltered / n * 100:.1f}%"
    )
    print(
        f"Filtered   Hit@1:      {hits_filtered}/{n} = {hits_filtered / n * 100:.1f}%"
    )
    print(f"{'=' * 60}\n")

    return results


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="TypeOracle symbolic DCA-Trie experiment"
    )
    parser.add_argument(
        "--phase",
        choices=["sir", "proxy", "all"],
        default="sir",
        help="sir = SIR/FNR (CPU), proxy = answer gen via llama.cpp, all = both",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of questions for proxy answer generation phase",
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Dataset split (default: test)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="llama.cpp server port (default: 8080)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/symbolic_experiment",
        help="Output directory",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    if args.phase in ("sir", "all"):
        sir_metrics = run_sir_fnr(split=args.split)
        with open(os.path.join(args.output, f"sir_results_{timestamp}.json"), "w") as f:
            json.dump(sir_metrics, f, indent=2)

    if args.phase in ("proxy", "all"):
        proxy_results = run_proxy_answer(
            n_questions=args.n,
            split=args.split,
            port=args.port,
        )
        if proxy_results is not None:
            path = os.path.join(args.output, f"proxy_results_{timestamp}.json")
            with open(path, "w") as f:
                json.dump(proxy_results, f, indent=2)
            print(f"Saved proxy results to {path}")

    print("Done.")


if __name__ == "__main__":
    main()
