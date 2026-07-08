"""DCA-Trie Demo Configuration."""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = Path("/tmp/dca_trie_demo_cache")

# LLM
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# DCA-Trie
MAX_HOPS = 2
PATH_START = "<PATH>"
PATH_END = "</PATH>"

# Demo
SERVER_PORT = 7860
