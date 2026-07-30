#!/usr/bin/env python3
"""
step_by_step.py — Interactive demo of the DCA-Trie pipeline.

Two modes:
  --replay   (default)  Uses saved experiment data.  No GPU needed.
  --live                Loads the model and runs inference.  Needs GPU.

Runs each stage, prints results, and waits for ENTER before continuing.
Designed for live narration during a presentation.

Usage:
    uv run python demo/step_by_step.py                           # replay, random question
    uv run python demo/step_by_step.py --question-idx 0          # replay, specific question
    uv run python demo/step_by_step.py --live --method v2        # live v2 on GPU
    uv run python demo/step_by_step.py --live --method all       # live all methods
    uv run python demo/step_by_step.py --dataset RoG-cwq         # CWQ dataset
"""

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "experiments" / "type_oracle_full"))

import src.utils as graph_utils
from approach3_symbolic.type_oracle import TypeOracle
from experiments.type_oracle_full.utils import PATH_START, PATH_END

# ── terminal colours ────────────────────────────────────────────────────
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
RESET = "\033[0m"

import re
_FB_ID_RE = re.compile(r"^[gm]\.\w+$")


def _is_fb(name):
    return bool(_FB_ID_RE.match(name))


def heading(text):
    print(f"\n{BOLD}{CYAN}{'━' * 72}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'━' * 72}{RESET}")


def sub(text):
    print(f"\n  {BOLD}{YELLOW}▸ {text}{RESET}")
    print(f"{DIM}{'─' * 72}{RESET}")


def wait():
    try:
        input(f"\n{DIM}[press ENTER to continue]{RESET} ")
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)


def tag_fb(text):
    return f"{RED}[FB-ID]{RESET}" if _is_fb(text) else ""


# ── load saved predictions ──────────────────────────────────────────────

def load_preds(path):
    preds = {}
    p = Path(path)
    if not p.exists():
        return preds
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            preds[rec["id"]] = rec
    return preds


def parse_pred(rec):
    if rec is None:
        return {"reasoning_path": "(no prediction)", "answer": "?"}
    text = rec.get("prediction", "")
    if isinstance(text, list):
        text = text[0] if text else ""
    path_part, _, answer_part = text.partition("# Answer:")
    rp = path_part.replace("# Reasoning Path:", "").strip()
    return {"reasoning_path": rp, "answer": answer_part.strip()}


# ── REPLAY mode ─────────────────────────────────────────────────────────

def replay(data, pred_baseline, pred_v1, pred_v2, idx, oracle):
    """Run the demo using saved predictions — no GPU needed."""
    question = data["question"]
    entities = data.get("q_entity", [])
    answers = data.get("answer", [])
    g = graph_utils.build_graph(data["graph"], undirected=False)
    fb_count = sum(1 for n in g.nodes() if _is_fb(n))

    # ── Step 1 ──
    heading("STEP 1: Question")
    print(f"  Question:  {BOLD}{question}{RESET}")
    print(f"  Entities:  {entities}")
    print(f"  Answer(s): {GREEN}{answers}{RESET}")
    wait()

    # ── Step 2 ──
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
            print(f"    {entity}  →  {rel}  →  {nbr} {tag_fb(nbr)}")
    wait()

    # ── Step 3 ──
    heading("STEP 3: TypeOracle")
    ans_types = oracle.infer_answer_types(question)
    if not ans_types:
        all_paths = graph_utils.dfs(g, entities, 2)
        ans_types = oracle.infer_answer_types_from_paths(all_paths)
    print(f"  Answer types: {BOLD}{ans_types if ans_types else '(empty)'}{RESET}")
    print(f"  Entity types known: {len(oracle._entity_types)}")
    print(f"  Hand-curated schema: {len(oracle._schema)} relations")
    print(f"  Auto-mined schema:   {len(oracle._mined_schema)} relations")
    wait()

    # ── Step 4 ──
    heading("STEP 4: DFS Path Enumeration (all paths)")
    all_paths = graph_utils.dfs(g, entities, 2)
    print(f"  Total paths: {BOLD}{len(all_paths)}{RESET}")
    sub("Sample paths")
    for p in all_paths[:5]:
        print(f"    {graph_utils.path_to_string(p)}")
    if len(all_paths) > 5:
        print(f"    {DIM}... and {len(all_paths) - 5} more{RESET}")
    wait()

    # ── Step 5 ──
    heading("STEP 5: TypeOracle Filtering (v1 static)")
    filtered = []
    for p in all_paths:
        admit = True
        for _, rel, tail in p:
            if not oracle.range_gate(rel, tail):
                admit = False
                break
        if admit and p:
            terminal = p[-1][2]
            if not oracle.type_gate(terminal, ans_types, len(p), 2):
                admit = False
        if admit:
            filtered.append(p)
    reduction = (1 - len(filtered) / max(1, len(all_paths))) * 100
    print(f"  Before: {len(all_paths)}  →  After: {BOLD}{len(filtered)}{RESET}  ({reduction:.1f}% reduction)")

    removed = [p for p in all_paths if p not in filtered]
    if removed:
        sub("Removed paths (failed TypeOracle)")
        for p in removed[:4]:
            s = graph_utils.path_to_string(p)
            terminal = p[-1][2]
            ttypes = oracle.get_types(terminal)
            print(f"    {s}")
            print(f"      terminal types={ttypes}, answer_types={ans_types}")
    wait()

    # ── Step 6 ──
    heading("STEP 6: Trie Construction")
    print(f"  Baseline trie: {len(all_paths)} paths (all DFS paths)")
    print(f"  Filtered trie: {len(filtered)} paths (after TypeOracle)")
    print(f"  V2 tries:      per-hop, 1-hop paths from head pool")
    wait()

    # ── Step 7: predictions ──
    for label, pred in [
        ("GCR Baseline", pred_baseline),
        ("DCA-Trie v1 (static filtering)", pred_v1),
        ("DCA-Trie v2 (iterative beam search)", pred_v2),
    ]:
        heading(f"STEP 7: {label}")
        parsed = parse_pred(pred)
        print(f"  Reasoning path: {BOLD}{parsed['reasoning_path']}{RESET}")
        print(f"  Answer:         {BOLD}{parsed['answer']}{RESET}")

        if pred:
            correct = parsed["answer"].lower() in [a.lower() for a in answers]
            if correct:
                print(f"  {GREEN}✓ CORRECT{RESET}")
            else:
                print(f"  {RED}✗ WRONG — expected: {answers}{RESET}")
        wait()

    # ── summary ──
    heading("DEMO COMPLETE")
    print(f"  Question: {question}")
    print(f"  Answer:   {answers}\n")


# ── LIVE mode ───────────────────────────────────────────────────────────

def live(data, oracle, model, input_builder, method, beam_size, index_len):
    """Run the demo with the actual model — needs GPU."""
    import torch
    from experiments.type_oracle_full.trie_utils import (
        build_unfiltered_trie,
        build_filtered_trie,
    )
    from experiments.type_oracle_full.decoding import (
        run_constrained_decoding,
        _get_gated_paths,
    )

    question = data["question"]
    entities = data.get("q_entity", [])
    answers = data.get("answer", [])
    g = graph_utils.build_graph(data["graph"], undirected=False)
    fb_count = sum(1 for n in g.nodes() if _is_fb(n))

    # ── Step 1 ──
    heading("STEP 1: Question")
    print(f"  Question:  {BOLD}{question}{RESET}")
    print(f"  Entities:  {entities}")
    print(f"  Answer(s): {GREEN}{answers}{RESET}")
    wait()

    # ── Step 2 ──
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
            print(f"    {entity}  →  {rel}  →  {nbr} {tag_fb(nbr)}")
    wait()

    # ── Step 3 ──
    heading("STEP 3: TypeOracle")
    ans_types = oracle.infer_answer_types(question)
    if not ans_types:
        all_paths = graph_utils.dfs(g, entities, index_len)
        ans_types = oracle.infer_answer_types_from_paths(all_paths)
    print(f"  Answer types: {BOLD}{ans_types if ans_types else '(empty)'}{RESET}")
    wait()

    # ── Step 4 ──
    heading("STEP 4: DFS Path Enumeration")
    all_paths = graph_utils.dfs(g, entities, index_len)
    print(f"  Total paths: {BOLD}{len(all_paths)}{RESET}")
    sub("Sample paths")
    for p in all_paths[:5]:
        print(f"    {graph_utils.path_to_string(p)}")
    if len(all_paths) > 5:
        print(f"    {DIM}... and {len(all_paths) - 5} more{RESET}")
    wait()

    # ── Step 5 ──
    heading("STEP 5: TypeOracle Filtering")
    filtered = []
    for p in all_paths:
        admit = True
        for _, rel, tail in p:
            if not oracle.range_gate(rel, tail):
                admit = False
                break
        if admit and p:
            terminal = p[-1][2]
            if not oracle.type_gate(terminal, ans_types, len(p), index_len):
                admit = False
        if admit:
            filtered.append(p)
    reduction = (1 - len(filtered) / max(1, len(all_paths))) * 100
    print(f"  Before: {len(all_paths)}  →  After: {BOLD}{len(filtered)}{RESET}  ({reduction:.1f}% reduction)")
    wait()

    # ── Step 6: live decoding ──
    methods = []
    if method in ("baseline", "all"):
        methods.append("baseline")
    if method in ("v1", "all"):
        methods.append("v1")
    if method in ("v2", "all"):
        methods.append("v2")

    for m in methods:
        heading(f"STEP 6: Constrained Decoding — {m.upper()}")

        if m == "baseline":
            trie, _ = build_unfiltered_trie(model.tokenizer, data, index_len)
            print(f"  Method: single-pass, trie={len(all_paths)} paths, k=10 beam")
            sub("Running model...")
            prediction, _ = run_constrained_decoding(model, input_builder, data, trie)

        elif m == "v1":
            trie, _, _ = build_filtered_trie(model.tokenizer, data, index_len, oracle)
            print(f"  Method: single-pass, trie={len(filtered)} paths (filtered), k=10 beam")
            sub("Running model...")
            prediction, _ = run_constrained_decoding(model, input_builder, data, trie)

        elif m == "v2":
            from experiments.type_oracle_full.decoding import dca_v2_generate
            print(f"  Method: iterative hop-by-hop, beams={beam_size}, hops={index_len}")
            sub("Running model...")
            prediction = dca_v2_generate(
                data=data, nx_graph=g, llm_model=model,
                tokenizer=model.tokenizer, oracle=oracle,
                max_hops=index_len, max_new_tokens=256,
                input_builder=input_builder, beam_size=beam_size,
            )

        print(f"\n  {BOLD}Prediction:{RESET}")
        if prediction and PATH_START in prediction:
            path_str = prediction.split(PATH_START)[-1].split(PATH_END)[0]
            parts = [p.strip() for p in path_str.split(" -> ")]
            print(f"    {path_str}")
            terminal = parts[-1] if parts else "?"
            correct = terminal.lower() in [a.lower() for a in answers]
            if correct:
                print(f"  {GREEN}✓ CORRECT{RESET}")
            else:
                print(f"  {RED}✗ WRONG — expected: {answers}{RESET}")
        else:
            print(f"    {RED}(no prediction){RESET}")
        wait()

    heading("DEMO COMPLETE")
    print(f"  Question: {question}")
    print(f"  Answer:   {answers}\n")


# ── main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Step-by-step DCA-Trie demo")
    parser.add_argument("--dataset", default="RoG-webqsp", choices=["RoG-webqsp", "RoG-cwq"])
    parser.add_argument("--question-idx", type=int, default=None)
    parser.add_argument("--method", default="all", choices=["baseline", "v1", "v2", "all"])
    parser.add_argument("--index-len", type=int, default=2)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--live", action="store_true", help="Load model and run inference (needs GPU)")
    parser.add_argument(
        "--predictions-dir",
        type=str,
        default=None,
        help="Directory with predictions_*.jsonl (replay mode)",
    )
    args = parser.parse_args()

    from datasets import load_dataset

    ds = load_dataset(f"rmanluo/{args.dataset}", split="test")
    if args.question_idx is not None:
        idx = args.question_idx
    else:
        import random
        idx = random.randint(0, len(ds) - 1)
    data = ds[idx]

    oracle = TypeOracle.from_graph(data["graph"])

    # ── title screen ──
    print(f"""
{BOLD}{CYAN}{'━' * 72}
  DCA-Trie: Graph-Constrained Reasoning with Symbolic Type Gates
{'━' * 72}{RESET}
  Dataset:    {args.dataset}
  Question:   #{idx} of {len(ds)}
  Mode:       {'LIVE (GPU)' if args.live else 'REPLAY (saved data)'}
  Method:     {args.method}
  Index len:  {args.index_len} hops
{DIM}{'─' * 72}{RESET}
""")
    wait()

    if not args.live:
        # ── REPLAY mode ──
        if args.predictions_dir:
            pred_dir = Path(args.predictions_dir)
        else:
            # Try the final experiment results first, then ideas
            final = _PROJECT_ROOT / "results" / "final_experiment-20260723T011830Z-1-001" / "final_experiment" / "20260715_235359_763137" / args.dataset
            ideas = _PROJECT_ROOT / "results" / "ideas_webqsp_full"
            if final.exists():
                pred_dir = final
            else:
                pred_dir = ideas

        qid = data["id"]
        print(f"{DIM}Replay mode — using saved predictions from {pred_dir}{RESET}")
        print(f"{DIM}Question ID: {qid}{RESET}")

        pred_bl = load_preds(pred_dir / "predictions_GCR_Baseline.jsonl" if (pred_dir / "predictions_GCR_Baseline.jsonl").exists() else pred_dir / "predictions_baseline.jsonl")
        pred_v1 = load_preds(pred_dir / "predictions_DCA_v1_Static.jsonl" if (pred_dir / "predictions_DCA_v1_Static.jsonl").exists() else pred_dir / "predictions_filtered.jsonl")
        pred_v2 = load_preds(pred_dir / "predictions_DCA_v2_Dynamic.jsonl")

        replay(data, pred_bl.get(qid), pred_v1.get(qid), pred_v2.get(qid), idx, oracle)

    else:
        # ── LIVE mode ──
        import torch
        from src.llms import get_registed_model
        from src.qa_prompt_builder import PathGenerationWithAnswerPromptBuilder

        heading("LIVE MODE — Loading Model")
        model_path = "rmanluo/GCR-Meta-Llama-3.1-8B-Instruct"
        print(f"  Model: {BOLD}{model_path}{RESET}")

        LLM = get_registed_model(model_path)
        import argparse as _ap
        model_args = _ap.Namespace(
            model_path=model_path, model_name=model_path,
            k=10, generation_mode="group-beam",
            attn_implementation="sdpa", max_new_tokens=256,
            maximun_token=4096, dtype="bf16", quant="none",
            chat_model=True, use_assistant_model=False,
        )
        model = LLM(model_args)
        model.prepare_for_inference()
        model.generation_cfg.temperature = None
        model.generation_cfg.top_p = None
        model.generation_cfg.top_k = None
        print(f"  {GREEN}✓ Model loaded{RESET}")

        input_builder = PathGenerationWithAnswerPromptBuilder(
            model.tokenizer, "zero-shot", index_path_length=args.index_len
        )
        wait()

        live(data, oracle, model, input_builder, args.method, args.beam_size, args.index_len)


if __name__ == "__main__":
    main()
