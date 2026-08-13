"""
main.py — CLI, logging, model loading, and experiment orchestration.

Built for unattended runs on rented GPUs (Vast.ai): everything that can fail
cheaply is checked *before* the 16 GB model download, the run lock survives
re-rented instances, each condition is fault-isolated, and summary/status files
are written incrementally so a preempted box still leaves usable, resumable
state on disk.

Single entry point: ``run()``.  Call it directly or via ``uv run``.
"""

import argparse
import importlib.metadata
import json
import logging
import os
import platform
import random
import shutil
import signal
import socket
import subprocess
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


import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from experiment import run_condition  # noqa: E402

from src.llms import get_registed_model  # noqa: E402
from src.qa_prompt_builder import PathGenerationWithAnswerPromptBuilder  # noqa: E402
from utils import logger  # noqa: E402

# Resource thresholds

MIN_VRAM_GB = 17.0  # weights alone are ~16 GB
RECOMMENDED_VRAM_GB = 22.0  # + KV cache for k-way beam search at 4096 ctx
MODEL_DOWNLOAD_GB = 20.0  # weights + HF cache overhead
MIN_OUTPUT_DISK_GB = 2.0

CONDITIONS_BY_METHOD = {
    "baseline": ["GCR_Baseline"],
    "v1": ["DCA_v1_Static"],
    "v2": ["DCA_v2_Dynamic"],
    "v2-nogates": ["DCA_v2_NoGates"],
    "v3": ["DCA_v3_Lazy"],
    "v3-nogates": ["DCA_v3_NoGates"],
    "all": ["GCR_Baseline", "DCA_v1_Static", "DCA_v2_Dynamic"],
    "ablation": ["GCR_Baseline", "DCA_v1_Static", "DCA_v2_Dynamic", "DCA_v2_NoGates"],
    "lazy": ["GCR_Baseline", "DCA_v1_Static", "DCA_v3_Lazy"],
    "lazy-ablation": [
        "GCR_Baseline", "DCA_v1_Static", "DCA_v3_Lazy", "DCA_v3_NoGates",
    ],
}


# Atomic JSON writes


def _write_json(path: Path, payload) -> None:
    """Write *payload* as JSON to *path* via temp file + os.replace.

    A killed instance must never leave half a summary behind.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# Run lock


def _read_lock(lock_path: Path):
    """Parse a lock file.  Handles both the JSON and the legacy bare-PID format."""
    try:
        raw = lock_path.read_text().strip()
    except OSError:
        return None
    try:
        holder = json.loads(raw)
        if isinstance(holder, dict):
            return holder
    except json.JSONDecodeError:
        pass
    try:
        return {"pid": int(raw), "host": socket.gethostname(), "started": None}
    except ValueError:
        return None


def _lock_is_live(holder) -> bool:
    """True only if the lock belongs to a process still running on *this* host.

    A lock written on another host (a previous rented instance sharing a synced
    volume, or a container restart) is always stale — its PID means nothing here.
    """
    if not holder:
        return False
    if holder.get("host") != socket.gethostname():
        return False
    pid = holder.get("pid")
    if not isinstance(pid, int):
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _acquire_lock(output_dir: Path, force: bool = False) -> bool:
    """Claim the output directory so two runs cannot share it.  True on success."""
    lock_path = output_dir / ".run.lock"
    holder = _read_lock(lock_path) if lock_path.exists() else None

    if holder and _lock_is_live(holder):
        if not force:
            logger.error(
                "Another run (PID %s on %s) is already using %s. "
                "Remove %s to override, or pass --force-rerun.",
                holder.get("pid"),
                holder.get("host"),
                output_dir,
                lock_path,
            )
            return False
        logger.warning(
            "--force-rerun: breaking live lock held by PID %s on %s",
            holder.get("pid"),
            holder.get("host"),
        )
    elif holder:
        logger.warning(
            "Stale lock from PID %s on %s — taking over.", holder.get("pid"), holder.get("host")
        )

    lock_path.write_text(
        json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "started": time.time()})
    )
    return True


def _release_lock(output_dir: Path) -> None:
    """Remove the run lock if it belongs to us."""
    lock_path = output_dir / ".run.lock"
    try:
        if lock_path.exists():
            holder = _read_lock(lock_path)
            if holder and holder.get("pid") == os.getpid():
                lock_path.unlink()
    except Exception:
        pass


# Signals


def _install_signal_handlers() -> None:
    """Convert SIGTERM into KeyboardInterrupt so ``finally`` blocks still run.

    Stopping a Vast.ai instance sends SIGTERM; without this the lock is left
    behind and no partial status is written.

    SIGHUP is deliberately left alone: under ``nohup`` it is already ignored,
    and installing a handler would make an SSH disconnect kill the run.
    """

    def _handler(signum, _frame):
        raise KeyboardInterrupt(signal.Signals(signum).name)

    try:
        signal.signal(signal.SIGTERM, _handler)
    except (ValueError, OSError, AttributeError):
        pass


# Logging


def _setup_logging(output_dir: Path) -> None:
    """Configure logging to both console (INFO) and a file (DEBUG)."""
    log_path = output_dir / "run.log"

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(str(log_path))  # append — survives resume
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

    # `tail -f experiment.log` over SSH is the only progress view on a rented box
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass


# Preflight


def _gpu_info():
    """Return ``(name, total_vram_gb, (major, minor))``; name is None without CUDA."""
    if not torch.cuda.is_available():
        return None, 0.0, None
    props = torch.cuda.get_device_properties(0)
    return props.name, props.total_memory / 1e9, (props.major, props.minor)


def _hf_cache_dir() -> Path:
    """Resolve the HuggingFace hub cache the same way huggingface_hub does."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _model_is_cached(model_path: str) -> bool:
    """True if the checkpoint is a local dir or already in the hub cache."""
    if Path(model_path).exists():
        return True
    slug = "models--" + model_path.replace("/", "--")
    return (_hf_cache_dir() / slug).exists()


def _free_gb(path: Path) -> float:
    """Free space (GB) on the filesystem holding *path* (or its nearest parent)."""
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        return shutil.disk_usage(p).free / 1e9
    except OSError:
        return float("inf")


def _package_versions() -> dict:
    versions = {}
    for mod in ("torch", "transformers", "datasets", "networkx", "accelerate"):
        try:
            versions[mod] = importlib.metadata.version(mod)
        except importlib.metadata.PackageNotFoundError:
            versions[mod] = "not installed"
    return versions


def _git_provenance() -> dict:
    """Commit hash + dirty flag, so results can be traced back to code."""

    def _git(*cmd):
        try:
            out = subprocess.run(
                ["git", "-C", str(_PROJECT_ROOT), *cmd],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    commit = _git("rev-parse", "--short", "HEAD")
    status = _git("status", "--porcelain")
    return {"git_commit": commit, "git_dirty": bool(status) if status is not None else None}


def _min_vram_gb(dtype: str, quant: str) -> float:
    """Weights-only VRAM floor for an 8B checkpoint at this precision."""
    if quant == "4bit":
        return 6.0
    if quant == "8bit":
        return 9.0
    if dtype == "fp32":
        return 34.0
    return MIN_VRAM_GB


def _preflight(args, output_base, gpu_name, vram_gb, capability):
    """Log the environment and return a list of fatal problems (empty = go).

    Everything here runs before the model download so a misconfigured instance
    fails in seconds rather than after 10 minutes of paid GPU time.
    """
    hf_cache = _hf_cache_dir()
    cached = _model_is_cached(args.model_path)
    cache_free = _free_gb(hf_cache)
    out_free = _free_gb(output_base)
    min_vram = _min_vram_gb(args.dtype, args.quant)

    logger.info("Preflight:")
    logger.info("  host              %s", socket.gethostname())
    logger.info("  python            %s", platform.python_version())
    for mod, ver in _package_versions().items():
        logger.info("  %-17s %s", mod, ver)
    if gpu_name:
        logger.info(
            "  gpu               %s (%.1f GB, sm_%d%d, CUDA %s)",
            gpu_name,
            vram_gb,
            capability[0],
            capability[1],
            torch.version.cuda,
        )
    else:
        logger.info("  gpu               none (CPU only)")
    logger.info(
        "  hf cache          %s (%.1f GB free, model cached: %s)", hf_cache, cache_free, cached
    )
    logger.info("  output disk free  %.1f GB", out_free)
    logger.info("  HF_TOKEN set      %s", bool(os.environ.get("HF_TOKEN")))

    problems = []
    if not gpu_name:
        if args.allow_cpu:
            logger.warning("Running on CPU — expect this to be ~100x slower.")
        else:
            problems.append(
                "No CUDA device visible. On a rented GPU this usually means a driver/container "
                "mismatch — fix it before burning GPU-hours, or pass --allow-cpu."
            )
    elif vram_gb < min_vram:
        problems.append(
            f"GPU has {vram_gb:.1f} GB VRAM; the 8B checkpoint at dtype={args.dtype} "
            f"quant={args.quant} needs ~{min_vram:.0f} GB. Rent a bigger card, or use "
            f"--quant 8bit / --quant 4bit."
        )
    elif args.quant == "none" and vram_gb < RECOMMENDED_VRAM_GB:
        logger.warning(
            "Only %.1f GB VRAM — beam search with k=%d may OOM. Reduce -k or --beam-size if it does.",
            vram_gb,
            args.k,
        )

    if not cached and cache_free < MODEL_DOWNLOAD_GB:
        problems.append(
            f"Only {cache_free:.1f} GB free at {hf_cache} but the checkpoint needs "
            f"~{MODEL_DOWNLOAD_GB:.0f} GB. Point HF_HOME at the large volume "
            f"(e.g. export HF_HOME=/workspace/hf-cache)."
        )
    if out_free < MIN_OUTPUT_DISK_GB:
        problems.append(
            f"Only {out_free:.1f} GB free for results — predictions will fail to write."
        )

    return problems


# Output directory


def _default_base() -> Path:
    return _PROJECT_ROOT / "results" / "final_experiment"


def _resolve_output_dir(args) -> Path:
    """Pick the run directory, honouring --output-dir / --run-name / --resume."""
    if args.output_dir:
        return Path(args.output_dir)
    if args.run_name:
        return _default_base() / args.run_name
    if args.resume:
        base = _default_base()
        candidates = sorted(
            (d for d in base.glob("*") if d.is_dir()), key=lambda d: d.stat().st_mtime
        )
        if not candidates:
            sys.exit(f"--resume: no previous run directory found under {base}")
        chosen = candidates[-1]
        print(f"Resuming most recent run: {chosen}")
        return chosen
    ts = time.strftime("%Y%m%d_%H%M%S")
    us = f"{int(time.time() * 1_000_000) % 1_000_000:06d}"
    return _default_base() / f"{ts}_{us}"


# Dataset loading


def _load_dataset_with_retry(repo: str, split: str, attempts: int = 3):
    """Load a HF dataset, retrying with backoff — rented boxes have flaky egress."""
    for attempt in range(1, attempts + 1):
        try:
            return load_dataset(repo, split=split)
        except Exception as exc:
            if attempt == attempts:
                raise
            delay = 15 * attempt
            logger.warning(
                "load_dataset(%s) failed (attempt %d/%d): %s — retrying in %ds",
                repo,
                attempt,
                attempts,
                exc,
                delay,
            )
            time.sleep(delay)


# Main


def _build_parser() -> argparse.ArgumentParser:
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
        "--gen-mode", default="beam", choices=["greedy", "group-beam", "beam"]
    )
    parser.add_argument("--prompt-mode", default="zero-shot")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--beam-size", type=int, default=5, help="Beam size for v2 iterative decoding (default: 5)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=50, help="Questions per dataset (0 = all)"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run every question in the split (same as --max-samples 0)",
    )
    parser.add_argument("--method", default="all", choices=sorted(CONDITIONS_BY_METHOD))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Stable directory name under results/final_experiment (resumable across instances)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue the most recent run directory instead of creating a new one",
    )
    parser.add_argument(
        "--force-rerun", action="store_true", help="Overwrite existing results and ignore lock file"
    )
    parser.add_argument(
        "--sample-timeout",
        type=int,
        default=120,
        help="Per-sample timeout in seconds (0 = no limit)",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print per-sample trace output matching step_by_step.py format",
    )
    parser.add_argument(
        "--collect-metrics",
        action="store_true",
        help="Collect ablation metrics (BUR, SIR trajectory, volatility, RV)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    hw = parser.add_argument_group("hardware / rented-GPU options")
    hw.add_argument(
        "--attn-impl",
        default="auto",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
        help="auto = flash_attention_2 on sm_80+ when flash-attn is installed, else sdpa",
    )
    hw.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    hw.add_argument("--quant", default="none", choices=["none", "8bit", "4bit"])
    hw.add_argument("--allow-cpu", action="store_true", help="Proceed even if no GPU is visible")
    hw.add_argument("--no-preflight", action="store_true", help="Skip environment preflight checks")
    hw.add_argument(
        "--cost-per-hour",
        type=float,
        default=float(os.environ.get("VAST_COST_PER_HOUR", "0") or 0),
        help="Instance $/hr — logs a running cost estimate (env: VAST_COST_PER_HOUR)",
    )
    hw.add_argument(
        "--max-runtime-hours",
        type=float,
        default=0.0,
        help="Stop before starting a new condition once this budget is spent (0 = no limit)",
    )
    return parser


def run(argv=None):
    """Parse args, load model, run experiment conditions, print summary."""
    args = _build_parser().parse_args(argv)
    if args.full:
        args.max_samples = 0

    output_base = _resolve_output_dir(args)
    output_base.mkdir(parents=True, exist_ok=True)

    _setup_logging(output_base)
    _install_signal_handlers()

    gpu_name, vram_gb, capability = _gpu_info()
    if not args.no_preflight:
        problems = _preflight(args, output_base, gpu_name, vram_gb, capability)
        if problems:
            for p in problems:
                logger.error("PREFLIGHT: %s", p)
            logger.error("Aborting before model download. Override with --no-preflight.")
            sys.exit(2)

    # Lock file (prevent duplicate runs on rented GPU)
    if not _acquire_lock(output_base, force=args.force_rerun):
        sys.exit(1)

    exit_code = 0
    try:
        _run(args, output_base, gpu_name, vram_gb, capability)
    except KeyboardInterrupt:
        logger.warning(
            "Interrupted — partial predictions are checkpointed. Rerun with "
            "--output-dir %s (or --resume) to continue.",
            output_base,
        )
        _write_status(output_base, state="interrupted")
        exit_code = 130
    finally:
        _release_lock(output_base)
    if exit_code:
        sys.exit(exit_code)


def _resolve_attn_impl(requested: str, capability) -> str:
    """Pick an attention backend.

    flash-attn 2 runs on every Ampere-or-newer card (sm_80+: A100, A6000, 4090,
    L40S, H100), not just A100s — the previous name-match sent 4090 rentals down
    the slower sdpa path.  The model layer falls back to sdpa on its own if the
    package turns out to be missing.
    """
    if requested != "auto":
        return requested
    if capability is None or capability[0] < 8:
        return "sdpa"
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return "sdpa"
    return "flash_attention_2"


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_status(output_base: Path, **fields) -> None:
    """Heartbeat file — lets you check progress over SSH without parsing the log."""
    try:
        payload = {"updated": time.strftime("%Y-%m-%d %H:%M:%S"), **fields}
        _write_json(output_base / "status.json", payload)
    except OSError:
        pass


def _write_summary(output_base: Path, all_summary: dict) -> None:
    _write_json(
        output_base / "summary.json",
        {f"{ds}|{cond}": m for (ds, cond), m in all_summary.items()},
    )


def _cost_note(elapsed_s: float, cost_per_hour: float) -> str:
    if cost_per_hour <= 0:
        return ""
    return f" | ~${elapsed_s / 3600 * cost_per_hour:.2f} spent"


def _run(args, output_base, gpu_name, vram_gb, capability):
    """Core experiment logic (called inside the lock)."""
    logger.info("DCA-Trie experiment start — output: %s", output_base)

    _set_seeds(args.seed)
    logger.info("Random seed: %d", args.seed)

    attn_impl = _resolve_attn_impl(args.attn_impl, capability)

    config = {
        "model_path": args.model_path,
        "data_path": args.data_path,
        "datasets": args.datasets,
        "split": args.split,
        "index_len": args.index_len,
        "k": args.k,
        "gen_mode": args.gen_mode,
        "prompt_mode": args.prompt_mode,
        "max_new_tokens": args.max_new_tokens,
        "beam_size": args.beam_size,
        "max_samples": args.max_samples,
        "method": args.method,
        "seed": args.seed,
        "dtype": args.dtype,
        "quant": args.quant,
        "attn_impl": attn_impl,
        "gpu": gpu_name or "None",
        "gpu_vram_gb": round(vram_gb, 1),
        "sample_timeout_s": args.sample_timeout,
        "host": socket.gethostname(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(),
        **_package_versions(),
        **_git_provenance(),
    }
    _write_json(output_base / "config.json", config)

    logger.info("Configuration:")
    for k, v in config.items():
        logger.info("  %-22s %s", k, v)

    # -Load model ----
    logger.info("Loading %s ...", args.model_path)
    llm_cls = get_registed_model(args.model_path)
    model_args_ns = argparse.Namespace(
        model_path=args.model_path,
        model_name=args.model_path,
        k=args.k,
        generation_mode=args.gen_mode,
        attn_implementation=attn_impl,
        max_new_tokens=args.max_new_tokens,
        maximun_token=4096,
        dtype=args.dtype,
        quant=args.quant,
        chat_model=True,
        use_assistant_model=False,
    )
    t_load = time.time()
    model = llm_cls(model_args_ns)
    model.prepare_for_inference()
    model.generation_cfg.temperature = None
    model.generation_cfg.top_p = None
    model.generation_cfg.top_k = None
    model.model.generation_config.temperature = None
    model.model.generation_config.top_p = None
    model.model.generation_config.top_k = None
    logger.info("Model loaded in %.1fs", time.time() - t_load)
    if torch.cuda.is_available():
        logger.info("VRAM after load: %.1f / %.1f GB", torch.cuda.memory_allocated() / 1e9, vram_gb)

    input_builder = PathGenerationWithAnswerPromptBuilder(
        model.tokenizer, args.prompt_mode, index_path_length=args.index_len
    )

    conditions = CONDITIONS_BY_METHOD[args.method]

    t_start = time.time()
    budget_s = args.max_runtime_hours * 3600
    total_units = len(args.datasets) * len(conditions)
    done_units = 0
    all_summary = {}
    stopped_early = None

    _write_status(
        output_base,
        state="running",
        pid=os.getpid(),
        host=socket.gethostname(),
        total_units=total_units,
        completed_units=0,
    )

    for ds_name in args.datasets:
        if stopped_early:
            break
        logger.info("=" * 60)
        logger.info("  DATASET: %s", ds_name)
        logger.info("=" * 60)
        try:
            dataset = _load_dataset_with_retry(f"{args.data_path}/{ds_name}", args.split)
        except Exception as exc:
            logger.error(
                "Could not load %s/%s — skipping dataset: %s", args.data_path, ds_name, exc
            )
            total_units -= len(conditions)  # keep the ETA honest
            continue

        if args.max_samples and args.max_samples < len(dataset):
            dataset = dataset.select(range(args.max_samples))
        logger.info("  Samples: %d", len(dataset))
        ds_dir = output_base / ds_name
        ds_dir.mkdir(exist_ok=True)

        for cond in conditions:
            elapsed_total = time.time() - t_start
            if budget_s and elapsed_total > budget_s:
                stopped_early = "runtime budget exhausted"
                logger.warning(
                    "Runtime budget of %.2fh reached — stopping before %s/%s.%s",
                    args.max_runtime_hours,
                    ds_name,
                    cond,
                    _cost_note(elapsed_total, args.cost_per_hour),
                )
                break

            logger.info("  Running %s ...", cond)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            try:
                metrics = run_condition(
                    model=model,
                    input_builder=input_builder,
                    dataset=dataset,
                    cond_name=cond,
                    ds_dir=ds_dir,
                    force_rerun=args.force_rerun,
                    index_len=args.index_len,
                    max_new_tokens=args.max_new_tokens,
                    sample_timeout_s=args.sample_timeout,
                    beam_size=args.beam_size,
                    trace=args.trace,
                    collect_metrics=args.collect_metrics,
                )
            except KeyboardInterrupt:
                _write_summary(output_base, all_summary)
                raise
            except Exception:
                # One broken condition must not cost the whole rental.
                logger.exception(
                    "Condition %s on %s failed — continuing with the rest", cond, ds_name
                )
                metrics = {
                    "condition": cond,
                    "n": 0,
                    "hits": 0,
                    "hit_at_1": 0.0,
                    "time_s": 0,
                    "n_dead_ends": 0,
                    "n_skipped": 0,
                    "error": "see run.log",
                }

            if torch.cuda.is_available():
                metrics["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

            all_summary[(ds_name, cond)] = metrics
            done_units += 1

            # Persist after every condition: a preempted instance keeps its aggregates.
            _write_summary(output_base, all_summary)
            elapsed_total = time.time() - t_start
            eta_s = (elapsed_total / done_units) * (total_units - done_units) if done_units else 0
            logger.info(
                "  Progress %d/%d conditions | elapsed %.1fh | ETA %.1fh%s",
                done_units,
                total_units,
                elapsed_total / 3600,
                eta_s / 3600,
                _cost_note(elapsed_total, args.cost_per_hour),
            )
            _write_status(
                output_base,
                state="running",
                pid=os.getpid(),
                host=socket.gethostname(),
                total_units=total_units,
                completed_units=done_units,
                last_condition=f"{ds_name}|{cond}",
                elapsed_h=round(elapsed_total / 3600, 2),
                eta_h=round(eta_s / 3600, 2),
                est_cost_usd=round(elapsed_total / 3600 * args.cost_per_hour, 2)
                if args.cost_per_hour > 0
                else None,
            )

    logger.info("=" * 80)
    logger.info("%s", "FINAL RESULTS".center(80))
    logger.info("=" * 80)
    logger.info(
        "%-15s %-20s %6s %8s %8s %8s %8s %8s",
        "Dataset",
        "Condition",
        "N",
        "Hits@1",
        "Hit%",
        "Time",
        "DeadEnd",
        "Skip",
    )
    logger.info("-" * 80)
    for (ds, cond), m in all_summary.items():
        logger.info(
            "%-15s %-20s %6d %8d %7.1f%% %7.0fs %8d %8d",
            ds,
            cond,
            m["n"],
            m["hits"],
            m["hit_at_1"],
            m["time_s"],
            m["n_dead_ends"],
            m["n_skipped"],
        )
        if m.get("error"):
            logger.info("%15s (FAILED — %s)", "", m["error"])
        if "reduction_pct" in m:
            logger.info(
                "%15s (paths: %d/%d, -%.1f%%)",
                "",
                m["total_paths_filtered"],
                m["total_paths_all"],
                m["reduction_pct"],
            )
        if "avg_candidates_materialised" in m:
            logger.info(
                "%15s (lazy: %.1f candidates over %.1f frontiers per question)",
                "",
                m["avg_candidates_materialised"],
                m["avg_frontier_builds"],
            )
    logger.info("=" * 80)

    total_elapsed = time.time() - t_start
    logger.info(
        "Total wall time: %.2fh%s",
        total_elapsed / 3600,
        _cost_note(total_elapsed, args.cost_per_hour),
    )
    if stopped_early:
        logger.warning(
            "Run incomplete (%s) — rerun with the same output dir to finish.", stopped_early
        )

    _write_summary(output_base, all_summary)
    _write_status(
        output_base,
        state="stopped_early" if stopped_early else "done",
        total_units=total_units,
        completed_units=done_units,
        elapsed_h=round(total_elapsed / 3600, 2),
        est_cost_usd=round(total_elapsed / 3600 * args.cost_per_hour, 2)
        if args.cost_per_hour > 0
        else None,
    )
    logger.info("Results saved to %s", output_base)


if __name__ == "__main__":
    run()
