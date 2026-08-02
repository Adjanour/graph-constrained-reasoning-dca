#!/usr/bin/env bash
# Run the DCA-Trie demo
set -e
cd "$(dirname "$0")"
uv run streamlit run demo/app.py "$@"
