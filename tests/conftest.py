"""Shared fixtures.

The experiment modules import each other flatly (``from utils import ...``),
so ``experiments/type_oracle_full`` goes on the path alongside the repo root.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "type_oracle_full"))

MODEL = "rmanluo/GCR-Meta-Llama-3.1-8B-Instruct"


@pytest.fixture(scope="session")
def tokenizer():
    """Tokenizer only — no weights are downloaded or loaded."""
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(MODEL)
    except Exception as exc:
        pytest.skip(f"tokenizer unavailable ({exc}); run scripts/setup-env.sh first")


@pytest.fixture(scope="session")
def webqsp():
    """A few real WebQSP subgraphs, smallest first to keep runtime sane."""
    datasets = pytest.importorskip("datasets")
    try:
        ds = datasets.load_dataset("rmanluo/RoG-webqsp", split="train")
    except Exception as exc:
        pytest.skip(f"RoG-webqsp unavailable ({exc})")
    order = sorted((len(ds[i]["graph"]), i) for i in range(150))
    return [ds[i] for _, i in order[:4]]
