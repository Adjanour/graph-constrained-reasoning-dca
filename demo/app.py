"""
DCA-Trie Demo — Step-by-step visualization of graph-constrained reasoning.

Run:
    streamlit run demo/app.py

Two modes:
- Pre-computed: loads saved results (no GPU needed)
- Live: runs the full pipeline on a GPU
"""

import json
import sys
from pathlib import Path

import streamlit as st
import networkx as nx
from pyvis.network import Network

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_DEMO_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _DEMO_DIR.parent
_DATA_DIR = _DEMO_DIR / "demo_data"

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DCA-Trie Demo",
    page_icon="🧠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .step-header {
        background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%);
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        margin: 10px 0;
        font-size: 1.1em;
        font-weight: bold;
    }
    .gate-pass { color: #2ecc71; font-weight: bold; }
    .gate-fail { color: #e74c3c; font-weight: bold; }
    .entity-tag {
        background: #3498db;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.85em;
        margin: 2px;
    }
    .relation-tag {
        background: #9b59b6;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.85em;
        margin: 2px;
    }
    .answer-box {
        background: #d5f5e3;
        border: 2px solid #27ae60;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        font-size: 1.2em;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load pre-computed data
# ---------------------------------------------------------------------------

@st.cache_data
def load_manifest():
    manifest_path = _DATA_DIR / "manifest.json"
    if not manifest_path.exists():
        return []
    with open(manifest_path) as f:
        return json.load(f)


@st.cache_data
def load_sample(filename):
    path = _DATA_DIR / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# KG visualization
# ---------------------------------------------------------------------------

def render_kg(kg_data, highlight_path=None):
    """Render KG as an interactive pyvis graph."""
    g = nx.DiGraph()
    for node in kg_data["nodes"]:
        color = "#e74c3c" if node["is_start"] else "#3498db"
        g.add_node(node["id"], label=node["id"], color=color, size=25)
    for edge in kg_data["edges"]:
        g.add_edge(edge["from"], edge["to"], label=edge["relation"])

    net = Network(height="400px", width="100%", directed=True, notebook=False)
    net.from_nx(g)
    net.set_options(json.dumps({
        "physics": {"stabilization": {"iterations": 100}},
        "edges": {"arrows": {"to": {"enabled": True}}, "font": {"size": 10}},
        "nodes": {"font": {"size": 14}},
    }))
    return net


# ---------------------------------------------------------------------------
# Step renderers
# ---------------------------------------------------------------------------

def step_question(data):
    st.markdown('<div class="step-header">📋 Step 1: The Question</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"### {data['question']}")
    with col2:
        if data.get("entities"):
            st.markdown("**Starting entities:**")
            for e in data["entities"]:
                st.markdown(f'<span class="entity-tag">{e}</span>', unsafe_allow_html=True)
        if data.get("answers"):
            st.markdown("**Ground truth:** " + ", ".join(data["answers"]))


def step_kg(kg_data):
    st.markdown('<div class="step-header">🕸️ Step 2: Knowledge Graph Subgraph</div>', unsafe_allow_html=True)
    st.markdown("The KG subgraph around the question entities. Red = start entities, blue = neighbours.")
    try:
        net = render_kg(kg_data)
        net.save_graph(str(_DEMO_DIR / "_temp_kg.html"))
        st.components.v1.html(open(_DEMO_DIR / "_temp_kg.html").read(), height=420, scrolling=True)
    except Exception as e:
        st.warning(f"Could not render interactive graph: {e}")
        st.json(kg_data)


def step_type_oracle(gated_data):
    st.markdown('<div class="step-header">🔬 Step 3: TypeOracle Semantic Gates</div>', unsafe_allow_html=True)

    at = gated_data.get("answer_types", [])
    if at:
        st.markdown(f"**Inferred answer types:** {', '.join(at)}")
    else:
        st.markdown("**Inferred answer types:** (none — using structural constraint only)")

    st.markdown(f"**Total paths found:** {gated_data['total_paths']}  |  "
                f"**Surviving after gates:** {gated_data['admitted_paths']}")

    admitted = [p for p in gated_data["paths"] if p["admitted"]]
    rejected = [p for p in gated_data["paths"] if not p["admitted"]]

    if admitted:
        st.markdown("#### ✅ Admitted paths (pass both gates)")
        for p in admitted:
            steps_html = ""
            for s in p["steps"]:
                range_sym = "✅" if s["range_gate"] else "❌"
                type_sym = ""
                if "type_gate" in s:
                    type_sym = " ✅" if s["type_gate"] else " ❌"
                steps_html += (
                    f'<span class="entity-tag">{s["head"]}</span> '
                    f'<span class="relation-tag">{s["relation"]}</span> '
                    f'<span class="entity-tag">{s["tail"]}</span> '
                    f'{range_sym}{type_sym} &nbsp;&nbsp; '
                )
            st.markdown(f"**{p['path']}**", unsafe_allow_html=True)

    if rejected:
        st.markdown("#### ❌ Rejected paths (fail a gate)")
        for p in rejected[:15]:  # cap at 15 for readability
            steps_html = ""
            for s in p["steps"]:
                range_sym = "✅" if s["range_gate"] else "❌"
                type_sym = ""
                if "type_gate" in s:
                    type_sym = " ✅" if s["type_gate"] else " ❌"
                steps_html += (
                    f'<span class="entity-tag">{s["head"]}</span> '
                    f'<span class="relation-tag">{s["relation"]}</span> '
                    f'<span class="entity-tag">{s["tail"]}</span> '
                    f'{range_sym}{type_sym} &nbsp;&nbsp; '
                )
            st.markdown(f"~~{p['path']}~~", unsafe_allow_html=True)
        if len(rejected) > 15:
            st.markdown(f"*... and {len(rejected) - 15} more rejected paths*")


def step_constrained_decoding(data):
    st.markdown('<div class="step-header">⚙️ Step 4: Constrained Decoding</div>', unsafe_allow_html=True)
    st.markdown("""
    The LLM generates a reasoning path **one hop at a time**. At each step:
    1. TypeOracle enumerates valid 1-hop paths from the current head entity
    2. Builds a trie from those paths
    3. LLM generates **only** tokens that follow the trie structure
    4. If no valid paths exist → dead end, beam dropped (no backtracking)
    """)

    # Show the hop-by-hop process
    hops = [
        {"hop": 1, "head": data["entities"][0] if data.get("entities") else "?",
         "action": "Enumerate gated paths → build trie → LLM generates one hop"},
    ]
    if data.get("gated_paths", {}).get("admitted_paths", 0) > 0:
        admitted = [p for p in data["gated_paths"]["paths"] if p["admitted"]]
        if admitted:
            first_path = admitted[0]["path"]
            segments = first_path.split(" -> ")
            for i in range(0, len(segments) - 1, 2):
                hop_num = i // 2 + 1
                head = segments[i]
                rel = segments[i + 1] if i + 1 < len(segments) else "?"
                tail = segments[i + 2] if i + 2 < len(segments) else "?"
                hops.append({
                    "hop": hop_num + 1,
                    "head": head,
                    "rel": rel,
                    "tail": tail,
                    "action": f"Generate: {head} → {rel} → {tail}",
                })

    for h in hops:
        if "rel" in h:
            st.markdown(
                f"**Hop {h['hop']}:** "
                f'<span class="entity-tag">{h["head"]}</span> '
                f'<span class="relation-tag">{h["rel"]}</span> '
                f'<span class="entity-tag">{h["tail"]}</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f"**Hop {h['hop']}:** {h['action']}")


def step_answer(data):
    st.markdown('<div class="step-header">🎯 Step 5: Final Answer</div>', unsafe_allow_html=True)
    # Extract answer from the admitted path
    admitted = [p for p in data.get("gated_paths", {}).get("paths", []) if p["admitted"]]
    if admitted:
        best_path = admitted[0]["path"]
        answer = best_path.split(" -> ")[-1]
        st.markdown(f'<div class="answer-box">Predicted answer: <strong>{answer}</strong></div>',
                    unsafe_allow_html=True)
        st.markdown(f"Ground truth: {', '.join(data.get('answers', []))}")
    else:
        st.markdown('<div class="answer-box">No valid path found (dead end)</div>',
                    unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.title("🧠 DCA-Trie: Graph-Constrained Reasoning Demo")
    st.markdown("Walk through how DCA-Trie constrains an LLM to reason over a knowledge graph, step by step.")

    manifest = load_manifest()
    if not manifest:
        st.error("No demo data found. Run `uv run demo/generate_demo_data.py` first.")
        return

    # Sidebar
    st.sidebar.header("Settings")
    mode = st.sidebar.radio("Mode", ["Pre-computed (no GPU)", "Live (GPU required)"])

    sample_idx = st.sidebar.selectbox(
        "Select a question",
        range(len(manifest)),
        format_func=lambda i: f"{manifest[i]['question'][:60]}..."
    )

    data = load_sample(manifest[sample_idx]["file"])
    if data is None:
        st.error("Could not load sample data.")
        return

    # Run steps
    step_question(data)
    step_kg(data["kg"])
    step_type_oracle(data["gated_paths"])
    step_constrained_decoding(data)
    step_answer(data)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Paths:** {data['gated_paths']['total_paths']} total, "
                        f"{data['gated_paths']['admitted_paths']} admitted")
    st.sidebar.markdown(f"**Answer types:** {', '.join(data['gated_paths']['answer_types']) or '(none)'}")


if __name__ == "__main__":
    main()
