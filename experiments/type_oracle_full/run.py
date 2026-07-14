#!/usr/bin/env python3
"""
run.py — DCA-Trie full experiment entry point.

Usage:
    python experiments/type_oracle_full/run.py                          # both datasets, 50 samples
    python experiments/type_oracle_full/run.py --datasets RoG-webqsp    # one dataset
    python experiments/type_oracle_full/run.py --method v1              # v1 only
    python experiments/type_oracle_full/run.py --method all             # all three
    python experiments/type_oracle_full/run.py --max-samples 10
    python experiments/type_oracle_full/run.py --force-rerun
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_EXPERIMENT_DIR))

from main import run

if __name__ == "__main__":
    run()
