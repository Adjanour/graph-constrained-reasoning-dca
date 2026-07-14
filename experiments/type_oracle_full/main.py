"""
main.py — CLI, logging, model loading, and experiment orchestration.

Single entry point: ``main()``.  Call it directly or via ``run.py``.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

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

from experiment import run_condition
from utils import logger


# ---------------------------------------------------------------------------
# File-system lock (prevents two runs on the same output directory)
# ---------------------------------------------------------------------------


def _acquire_lock(output_dir: Path) -> bool:
    """Try to create a PID lock file.  Returns True on success."""
    lock_path = output_dir / ".run.lock"
    if lock_path.exists():
        try:
            old_pid = int(lock_path.read_text().strip())
            os.kill(old_pid, 0)
            logger.error(
                "Another run (PID %d) is already using %s. "
                "Remove %s to override, or use --force-rerun.",
                old_pid, output_dir, lock_path,
            )
            return False
        except (OSError, ValueError):
            pass  # Stale lock — overwrite
    lock_path.write_text(str(os.getpid()))
    return True


def _release_lock(output_dir: Path) -> None:
    """Remove the PID lock file if it belongs to us."""
    lock_path = output_dir / ".run.lock"
    try:
        if lock_path.exists() and lock_path.read_text().strip() == str(os.getpid()):
            lock_path.unlink()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


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



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(argv=None):
    """Parse args, load model, run experiment conditions, print summary."""
    parser = argparse.ArgumentParser(description="DCA-Trie full experiment")
    parser.add_argument("--model-path", default="rmanluo/GCR-Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--data-path", default="rmanluo")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["RoG-webqsp", "RoG-cwq"],
        choices=["RoG-webqsp", "RoG-cwq"],
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--index-len", type=int, default=2)
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument(
        "--gen-mode", default="group-beam", choices=["greedy", "group-beam", "beam"]
    )
    parser.add_argument("--prompt-mode", default="zero-shot")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--method", default="all", choices=["baseline", "v1", "v2", "all"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--force-rerun", action="store_true",
                        help="Overwrite existing results and ignore lock file")
    parser.add_argument(
        "--sample-timeout", type=int, default=120,
        help="Per-sample timeout in seconds (0 = no limit)",
    )
    args = parser.parse_args(argv)

    # Output directory
    if args.output_dir:
        output_base = Path(args.output_dir)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        us = f"{int(time.time() * 1_000_000) % 1_000_000:06d}"
        output_base = _PROJECT_ROOT / "results" / "final_experiment" / f"{ts}_{us}"
    output_base.mkdir(parents=True, exist_ok=True)

    # Lock file (prevent duplicate runs on rented GPU)
    if not _acquire_lock(output_base):
        sys.exit(1)
    try:
        _run(args, output_base)
    finally:
        _release_lock(output_base)



def _run(args, output_base):
    """Core experiment logic (called inside the lock)."""
    _setup_logging(output_base)
    logger.info("DCA-Trie experiment start — output: %s", output_base)

    # GPU / attention detection
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

    config = {
        "model_path": args.model_path, "data_path": args.data_path,
        "datasets": args.datasets, "split": args.split,
        "index_len": args.index_len, "k": args.k,
        "gen_mode": args.gen_mode, "prompt_mode": args.prompt_mode,
        "max_new_tokens": args.max_new_tokens, "max_samples": args.max_samples,
        "method": args.method, "attn_impl": attn_impl, "gpu": gpu_name,
        "sample_timeout_s": args.sample_timeout,
    }
    with open(output_base / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info("Configuration:")
    for k, v in config.items():
        logger.info("  %-22s %s", k, v)

    # ---- Load model ----
    logger.info("Loading %s ...", args.model_path)
    LLM = get_registed_model(args.model_path)
    model_args_ns = argparse.Namespace(
        model_path=args.model_path, model_name=args.model_path,
        k=args.k, generation_mode=args.gen_mode,
        attn_implementation=attn_impl, max_new_tokens=args.max_new_tokens,
        maximun_token=4096,
    )
    t0 = time.time()
    model = LLM(model_args_ns)
    model.prepare_for_inference()
    model.generation_cfg.temperature = None
    model.generation_cfg.top_p = None
    model.generation_cfg.top_k = None
    model.model.generation_config.temperature = None
    model.model.generation_config.top_p = None
    model.model.generation_config.top_k = None
    logger.info("Model loaded in %.1fs", time.time() - t0)

    input_builder = PathGenerationWithAnswerPromptBuilder(
        model.tokenizer, args.prompt_mode, index_path_length=args.index_len
    )

    conditions = {
        "baseline": ["GCR_Baseline"],
        "v1": ["DCA_v1_Static"],
        "v2": ["DCA_v2_Dynamic"],
        "all": ["GCR_Baseline", "DCA_v1_Static", "DCA_v2_Dynamic"],
    }[args.method]

    all_summary = {}
    for ds_name in args.datasets:
        logger.info("=" * 60)
        logger.info("  DATASET: %s", ds_name)
        logger.info("=" * 60)
        dataset = load_dataset(f"{args.data_path}/{ds_name}", split=args.split)
        if args.max_samples and args.max_samples < len(dataset):
            dataset = dataset.select(range(args.max_samples))
        logger.info("  Samples: %d", len(dataset))
        ds_dir = output_base / ds_name
        ds_dir.mkdir(exist_ok=True)
        for cond in conditions:
            logger.info("  Running %s ...", cond)
            metrics = run_condition(
                model=model, input_builder=input_builder,
                dataset=dataset, cond_name=cond, ds_dir=ds_dir,
                force_rerun=args.force_rerun, index_len=args.index_len,
                max_new_tokens=args.max_new_tokens,
                sample_timeout_s=args.sample_timeout,
            )
            all_summary[(ds_name, cond)] = metrics

    logger.info("=" * 80)
    logger.info("%s", "FINAL RESULTS".center(80))
    logger.info("=" * 80)
    logger.info("%-15s %-20s %6s %8s %8s %8s %8s %8s",
                "Dataset", "Condition", "N", "Hits@1", "Hit%", "Time", "DeadEnd", "Skip")
    logger.info("-" * 80)
    for (ds, cond), m in all_summary.items():
        logger.info("%-15s %-20s %6d %8d %7.1f%% %7.0fs %8d %8d",
                    ds, cond, m["n"], m["hits"], m["hit_at_1"],
                    m["time_s"], m["n_dead_ends"], m["n_skipped"])
        if "reduction_pct" in m:
            logger.info("%15s (paths: %d/%d, -%.1f%%)", "",
                        m["total_paths_filtered"], m["total_paths_all"], m["reduction_pct"])
    logger.info("=" * 80)

    summary_out = {f"{ds}|{cond}": m for (ds, cond), m in all_summary.items()}
    with open(output_base / "summary.json", "w") as f:
        json.dump(summary_out, f, indent=2)
    logger.info("Results saved to %s", output_base)


if __name__ == "__main__":
    run()

