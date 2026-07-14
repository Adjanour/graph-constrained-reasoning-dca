"""
utils.py — Low-level utilities: timeout, JSONL I/O, atomic writes, constants.

No project-specific imports; safe to import from anywhere.
"""

import json
import logging
import os
import signal
from contextlib import contextmanager
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PATH_START = "<PATH>"
PATH_END = "</PATH>"

# ---------------------------------------------------------------------------
# Logger (shared across all experiment modules)
# ---------------------------------------------------------------------------

logger = logging.getLogger("type_oracle")

# ---------------------------------------------------------------------------
# Per-sample timeout
# ---------------------------------------------------------------------------


class TimeoutError(Exception):
    """Raised when a per-sample operation exceeds its time budget."""
    pass


@contextmanager
def timeout(seconds: int):
    """Context manager that raises TimeoutError if the block takes too long.

    Uses SIGALRM (Unix only).  No-op when ``seconds <= 0`` or on platforms
    without SIGALRM (Windows).
    """
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------


def safe_read_jsonl(path):
    """Read a JSONL file.

    Returns:
        records: list of parsed JSON dicts
        ids_set: set of ``record["id"]`` values
        has_incomplete_final_line: True if the last non-empty line failed to
            parse as JSON, indicating a truncated write.
    """
    records = []
    ids_set = set()
    has_partial = False
    if not os.path.exists(path):
        return records, ids_set, has_partial

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
                if "id" in rec:
                    ids_set.add(rec["id"])
            except json.JSONDecodeError:
                has_partial = True
    return records, ids_set, has_partial


def load_processed_ids(path):
    """Load set of already-processed question IDs from a JSONL file."""
    _, ids, _ = safe_read_jsonl(path)
    return ids


def load_preds(path):
    """Load all complete prediction records from a JSONL file."""
    records, _, _ = safe_read_jsonl(path)
    return records


def atomic_write_jsonl(path, records):
    """Write *records* to *path* atomically via temp-file + os.replace."""
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    os.replace(tmp, path)
