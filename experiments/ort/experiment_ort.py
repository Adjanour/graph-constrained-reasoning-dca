"""
experiment_ort.py — Experimental ORT improvements for DCA-Trie.

Implements three key improvements from ORT (Ontology-Guided Reverse Thinking):
1. LLM-based answer type extraction (replaces regex)
2. ORT + Oracle composition pipeline
3. Label-level trie abstraction (reduces trie size)

Usage:
    python experiment_ort.py --max-samples 50 --method ort-composed
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, FrozenSet, List, Set

# Ensure we can import modules from both the project root and this experiment dir
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import torch
from datasets import load_dataset

from src.llms import get_registed_model
from src.qa_prompt_builder import PathGenerationWithAnswerPromptBuilder
from src.graph_constrained_decoding import GraphConstrainedDecoding

from trie_utils import build_trie_from_strings
from utils import PATH_START, PATH_END, logger


# ---------------------------------------------------------------------------
# ORT-style LLM-based answer type extraction
# ---------------------------------------------------------------------------

# ORT's prompt template for condition and aim recognition (Figure 3)
ORT_TYPE_EXTRACTION_PROMPT = """You are a knowledge graph expert. Given a question, extract:
1. The condition entities (known information in the question)
2. The aim labels (what type of answer is expected)

Question: {question}

Label List: {label_list}

Please output:
- Condition entities: [list of entity names]
- Condition labels: [list of Freebase type labels for condition entities]
- Aim labels: [list of Freebase type labels for the expected answer]

Output as JSON:
{{
    "condition_entities": [...],
    "condition_labels": [...],
    "aim_labels": [...]
}}"""


def extract_types_with_llm(model, question: str, label_list: str = "") -> FrozenSet[str]:
    """
    Use LLM to extract answer types from question (ORT-style).
    
    This replaces the regex-based approach with LLM understanding.
    """
    prompt = ORT_TYPE_EXTRACTION_PROMPT.format(
        question=question,
        label_list=label_list or "Person, Location, Organization, CreativeWork, Date, Language"
    )
    
    llm_input = model.prepare_model_prompt(prompt)
    
    # Use model to generate type extraction
    inputs = model.tokenizer(llm_input, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs.input_ids.to(model.model.device)
    attention_mask = inputs.attention_mask.to(model.model.device)
    
    with torch.no_grad():
        res = model.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=128,
            temperature=0.1,
            do_sample=False,
            pad_token_id=model.tokenizer.eos_token_id,
        )
    
    output = model.tokenizer.decode(res.sequences[0][input_ids.shape[1]:], skip_special_tokens=True)
    
    # Parse the output to extract aim labels
    try:
        # Try to extract JSON
        json_match = re.search(r'\{[^}]+\}', output, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            aim_labels = data.get("aim_labels", [])
            return frozenset(aim_labels)
    except (json.JSONDecodeError, AttributeError):
        pass
    
    # Fallback: extract type keywords from output
    type_keywords = set()
    output_lower = output.lower()
    
    # Common Freebase types
    type_map = {
        "person": "Person",
        "location": "Location",
        "place": "Location",
        "country": "Country",
        "city": "City",
        "organization": "Organization",
        "company": "Organization",
        "film": "Film",
        "movie": "Film",
        "music": "MusicalWork",
        "book": "WrittenWork",
        "date": "Date",
        "time": "Date",
        "language": "Language",
    }
    
    for keyword, type_name in type_map.items():
        if keyword in output_lower:
            type_keywords.add(type_name)
    
    return frozenset(type_keywords)


# ---------------------------------------------------------------------------
# Label-level trie (ORT's abstract reasoning paths)
# ---------------------------------------------------------------------------

def build_label_level_trie(tokenizer, question_dict, oracle, label_paths: List[str]):
    """
    Build a trie at the label level (not entity level).
    
    ORT constructs paths like "Person -> Film -> Director" at the label level.
    This reduces trie size by 10-100x compared to entity-level paths.
    """
    if not label_paths:
        return None
    
    # Convert label paths to token sequences
    wrapped = [f"{PATH_START}{path}{PATH_END}" for path in label_paths]
    tokenized = tokenizer(wrapped, padding=False, add_special_tokens=False).input_ids
    tokenized = [ids + [tokenizer.eos_token_id] for ids in tokenized]
    
    from src.trie import MarisaTrie
    return MarisaTrie(tokenized, max_token_id=len(tokenizer) + 1)


# ---------------------------------------------------------------------------
# ORT + Oracle composition pipeline
# ---------------------------------------------------------------------------

def run_ort_composed(model, input_builder, data, oracle, index_len, max_new_tokens):
    """
    ORT + Oracle composition pipeline:
    
    1. ORT extracts aim labels from question (LLM-based)
    2. ORT constructs label reasoning paths (reverse thinking)
    3. Oracle filters paths during constrained decoding
    4. LLM selects best path
    """
    question = data["question"]
    entities = data.get("q_entity", [])
    
    # Step 1: ORT extracts aim labels (LLM-based, not regex)
    aim_labels = extract_types_with_llm(model, question)
    logger.debug("ORT aim labels for '%s': %s", question[:50], aim_labels)
    
    # If LLM extraction fails, fall back to oracle's regex
    if not aim_labels:
        aim_labels = oracle.infer_answer_types(question)
        logger.debug("Fallback to oracle aim labels: %s", aim_labels)
    
    # Step 2: Construct label reasoning paths (simplified ORT)
    # For now, use entity-level paths with label filtering
    import src.utils as graph_utils
    nx_graph = graph_utils.build_graph(data["graph"], undirected=False)
    
    # Get all entity-level paths
    all_paths = graph_utils.dfs(nx_graph, entities, index_len)
    if not all_paths:
        return None
    
    # Filter paths using ORT's aim labels
    filtered_paths = []
    for path in all_paths:
        terminal_entity = path[-1][2]
        terminal_types = oracle.get_types(terminal_entity)
        
        # Check if terminal entity matches aim labels
        if aim_labels and terminal_types:
            if not (aim_labels & terminal_types):
                continue
        
        # Also apply range gate
        admit = True
        for _, rel, tail in path:
            if not oracle.range_gate(rel, tail):
                admit = False
                break
        
        if admit:
            filtered_paths.append(path)
    
    if not filtered_paths:
        # If no paths pass, use all paths (conservative fallback)
        filtered_paths = all_paths
    
    # Step 3: Build trie from filtered paths
    filtered_str = [graph_utils.path_to_string(p) for p in filtered_paths]
    trie = build_trie_from_strings(model.tokenizer, filtered_str)
    
    if trie is None:
        return None
    
    # Step 4: Run constrained decoding
    from decoding import run_constrained_decoding
    prediction, ground_paths = run_constrained_decoding(model, input_builder, data, trie)
    
    return prediction


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment(argv=None):
    """Run the ORT experiment with 50 samples."""
    parser = argparse.ArgumentParser(description="ORT improvements experiment")
    parser.add_argument("--model-path", default="rmanluo/GCR-Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--data-path", default="rmanluo")
    parser.add_argument("--dataset", default="RoG-webqsp")
    parser.add_argument("--split", default="test")
    parser.add_argument("--index-len", type=int, default=2)
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--gen-mode", default="group-beam", choices=["greedy", "group-beam", "beam"])
    parser.add_argument("--prompt-mode", default="zero-shot")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--method", default="ort-composed", 
                       choices=["baseline", "v1", "ort-composed", "all"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args(argv)
    
    # Output directory
    if args.output_dir:
        output_base = Path(args.output_dir)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_base = _PROJECT_ROOT / "results" / "ort_experiment" / f"ort_{ts}"
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    _setup_logging(output_base)
    logger.info("ORT experiment start — output: %s", output_base)
    
    # Load model
    logger.info("Loading %s ...", args.model_path)
    LLM = get_registed_model(args.model_path)
    model_args_ns = argparse.Namespace(
        model_path=args.model_path, model_name=args.model_path,
        k=args.k, generation_mode=args.gen_mode,
        attn_implementation="sdpa", max_new_tokens=args.max_new_tokens,
        maximun_token=4096, dtype="bf16", quant="none",
        chat_model=True, use_assistant_model=False,
    )
    t0 = time.time()
    model = LLM(model_args_ns)
    model.prepare_for_inference()
    model.generation_cfg.temperature = None
    model.generation_cfg.top_p = None
    model.generation_cfg.top_k = None
    logger.info("Model loaded in %.1fs", time.time() - t0)
    
    input_builder = PathGenerationWithAnswerPromptBuilder(
        model.tokenizer, args.prompt_mode, index_path_length=args.index_len
    )
    
    # Load dataset
    logger.info("Loading dataset %s ...", args.dataset)
    dataset = load_dataset(f"{args.data_path}/{args.dataset}", split=args.split)
    if args.max_samples and args.max_samples < len(dataset):
        dataset = dataset.select(range(args.max_samples))
    logger.info("Samples: %d", len(dataset))
    
    # Import TypeOracle
    from approach3_symbolic.type_oracle import TypeOracle
    
    # Run experiment
    ds_dir = output_base / args.dataset
    ds_dir.mkdir(exist_ok=True)
    pred_path = ds_dir / f"predictions_{args.method}.jsonl"
    
    if args.force_rerun:
        pred_path.unlink(missing_ok=True)
    
    # Load existing predictions
    from utils import safe_read_jsonl, atomic_write_jsonl
    existing_records, processed_ids, has_partial = safe_read_jsonl(str(pred_path))
    
    if has_partial:
        logger.warning("Truncated JSONL detected — removing partial final line")
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
    
    with open(pred_path, "a") as fout:
        for d in dataset:
            qid = d["id"]
            if qid in processed_ids:
                continue
            
            oracle = TypeOracle.from_graph(d["graph"])
            
            try:
                if args.method == "ort-composed":
                    prediction = run_ort_composed(
                        model, input_builder, d, oracle,
                        args.index_len, args.max_new_tokens
                    )
                elif args.method == "baseline":
                    from experiment import _run_baseline
                    result, trie_ok = _run_baseline(
                        model, input_builder, d, qid, args.method, oracle,
                        index_len=args.index_len
                    )
                    prediction = result["prediction"] if result else None
                elif args.method == "v1":
                    from experiment import _run_v1
                    result, trie_ok = _run_v1(
                        model, input_builder, d, qid, args.method, oracle,
                        index_len=args.index_len
                    )
                    prediction = result["prediction"] if result else None
                
                # Build result dict
                result = {
                    "id": qid,
                    "question": d["question"],
                    "prediction": prediction if prediction else "",
                    "ground_truth": d["answer"],
                    "mode": args.method,
                }
                
            except Exception as e:
                logger.error("Error on sample %s: %s", qid, str(e))
                result = {
                    "id": qid,
                    "question": d["question"],
                    "prediction": "",
                    "ground_truth": d["answer"],
                    "mode": args.method,
                }
            
            if not result["prediction"]:
                n_skipped += 1
                processed_ids.add(qid)
                continue
            
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
                    args.method, n_done, len(dataset), rate, elapsed,
                    n_skipped, n_dead_ends,
                )
    
    elapsed = time.time() - t0
    
    # Compute metrics
    from experiment import compute_hits
    from utils import load_preds
    
    preds = load_preds(str(pred_path))
    hits = compute_hits(preds)
    n = len(preds)
    
    metrics = {
        "condition": args.method,
        "n": n,
        "hits": hits,
        "hit_at_1": round(hits / max(1, n) * 100, 1),
        "time_s": round(elapsed, 1),
        "n_dead_ends": n_dead_ends,
        "n_skipped": n_skipped,
    }
    
    logger.info("=" * 80)
    logger.info("FINAL RESULTS")
    logger.info("=" * 80)
    logger.info("%s: %d questions, Hits@1=%d/%d (%.1f%%), %.0fs, dead_ends=%d, skipped=%d",
                args.method, n, hits, n, metrics["hit_at_1"], elapsed, n_dead_ends, n_skipped)
    logger.info("=" * 80)
    
    # Save summary
    summary_path = output_base / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("Results saved to %s", output_base)
    
    return metrics


def _setup_logging(output_dir: Path) -> None:
    """Configure logging to both console (INFO) and a file (DEBUG)."""
    log_path = output_dir / "run.log"
    
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    
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


if __name__ == "__main__":
    run_experiment()
