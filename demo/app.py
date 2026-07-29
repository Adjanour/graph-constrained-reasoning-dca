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

def render_kg(kg_data, gated_paths=None, max_edges=40):
    """Render a readable 1-hop neighbourhood of the KG as an interactive graph.

    The full retrieved WebQSP subgraph can have thousands of edges, which is
    unreadable and slow to lay out with force-directed physics. We show only
    the direct (1-hop) neighbourhood of the start entities, capped at
    ``max_edges``, prioritising edges that lead to an entity reached by an
    admitted reasoning path so the highlighted structure matches the story
    told in the TypeOracle step.
    """
    start_ids = {n["id"] for n in kg_data["nodes"] if n["is_start"]}

    admitted_next_hop = set()
    if gated_paths:
        for p in gated_paths.get("paths", []):
            if p["admitted"] and p["steps"]:
                admitted_next_hop.add(p["steps"][0]["tail"])

    local_edges = [e for e in kg_data["edges"] if e["from"] in start_ids or e["to"] in start_ids]
    local_edges.sort(key=lambda e: 0 if (e["to"] if e["from"] in start_ids else e["from"]) in admitted_next_hop else 1)
    shown_edges = local_edges[:max_edges]

    g = nx.DiGraph()
    for sid in start_ids:
        g.add_node(sid, label=sid, color="#e74c3c", size=30)
    for e in shown_edges:
        for nid in (e["from"], e["to"]):
            if nid not in g:
                color = "#2ecc71" if nid in admitted_next_hop else "#3498db"
                g.add_node(nid, label=nid, color=color, size=18)
        g.add_edge(e["from"], e["to"], label=e["relation"])

    net = Network(height="420px", width="100%", directed=True, notebook=False)
    net.from_nx(g)
    net.set_options(json.dumps({
        "physics": {
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 60},
            "forceAtlas2Based": {"springLength": 120, "avoidOverlap": 0.5},
        },
        "edges": {
            "arrows": {"to": {"enabled": True}},
            "font": {"size": 10},
            "color": {"color": "#c8c8c8"},
            "smooth": False,
        },
        "nodes": {"font": {"size": 13}},
        "interaction": {"hover": True, "tooltipDelay": 100},
    }))
    return net, len(local_edges), len(shown_edges)


# ---------------------------------------------------------------------------
# Step renderers
# ---------------------------------------------------------------------------

def step_question(data):
    st.markdown('<div class="step-header">Step 1: The Question</div>', unsafe_allow_html=True)
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


def step_kg(data):
    st.markdown('<div class="step-header">Step 2: Knowledge Graph Subgraph</div>', unsafe_allow_html=True)
    kg_data = data["kg"]
    try:
        net, total_local, shown = render_kg(kg_data, data.get("gated_paths"))
        import streamlit.components.v1 as components
        components.html(net.generate_html(notebook=False), height=440, scrolling=True)
        st.caption(
            f"Showing {shown} of {total_local} direct relations from the start entity "
            f"(the full retrieved subgraph has {len(kg_data['edges'])} edges — too dense to "
            "render meaningfully). Red = start entity, green = reached by an admitted path, "
            "blue = other neighbours."
        )
    except Exception as e:
        st.warning(f"Could not render interactive graph: {e}")
        st.json(kg_data)


def _render_gated_path(p, struck_through=False):
    """Render one path plus its per-hop range/type gate verdicts."""
    steps_html = ""
    for s in p["steps"]:
        range_sym = '<span class="gate-pass">&#10003;</span>' if s["range_gate"] else '<span class="gate-fail">&#10007;</span>'
        type_sym = ""
        if "type_gate" in s:
            type_sym = (' <span class="gate-pass">&#10003;</span>' if s["type_gate"]
                        else ' <span class="gate-fail">&#10007;</span>')
        steps_html += (
            f'<span class="entity-tag">{s["head"]}</span> '
            f'<span class="relation-tag">{s["relation"]}</span> '
            f'<span class="entity-tag">{s["tail"]}</span> '
            f'{range_sym}{type_sym} &nbsp;&nbsp; '
        )
    path_line = f"~~{p['path']}~~" if struck_through else f"**{p['path']}**"
    st.markdown(path_line, unsafe_allow_html=True)
    st.markdown(steps_html, unsafe_allow_html=True)


def step_type_oracle(gated_data):
    st.markdown('<div class="step-header">Step 3: TypeOracle Semantic Gates</div>', unsafe_allow_html=True)

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
        st.markdown("#### Admitted paths (pass both gates)")
        for p in admitted:
            _render_gated_path(p)

    if rejected:
        st.markdown("#### Rejected paths (fail a gate)")
        for p in rejected[:15]:  # cap at 15 for readability
            _render_gated_path(p, struck_through=True)
        if len(rejected) > 15:
            st.markdown(f"*... and {len(rejected) - 15} more rejected paths*")


def step_constrained_decoding(data):
    st.markdown('<div class="step-header">Step 4: Constrained Decoding</div>', unsafe_allow_html=True)
    st.markdown("""
    The LLM generates a reasoning path **one hop at a time**. At each step:
    1. TypeOracle enumerates valid 1-hop paths from the current head entity
    2. Builds a trie from those paths
    3. LLM generates **only** tokens that follow the trie structure
    4. If no valid paths exist → dead end, beam dropped (no backtracking)
    """)

    dca = data.get("dca_trie_prediction")
    if not dca or not dca.get("reasoning_path"):
        st.info("No saved model run for this question — reasoning path unavailable in pre-computed mode.")
        return

    st.markdown("**Actual reasoning path chosen by the LLM (real saved model output, not a re-enumeration):**")
    segments = [s.strip() for s in dca["reasoning_path"].split(" -> ")]
    for i in range(0, len(segments) - 2, 2):
        hop_num = i // 2 + 1
        head, rel, tail = segments[i], segments[i + 1], segments[i + 2]
        st.markdown(
            f"**Hop {hop_num}:** "
            f'<span class="entity-tag">{head}</span> '
            f'<span class="relation-tag">{rel}</span> '
            f'<span class="entity-tag">{tail}</span>',
            unsafe_allow_html=True,
        )


def step_answer(data):
    st.markdown('<div class="step-header">Step 5: Final Answer</div>', unsafe_allow_html=True)

    gcr = data.get("gcr_prediction")
    dca = data.get("dca_trie_prediction")
    gt = ", ".join(data.get("answers", []))

    if not gcr and not dca:
        st.markdown('<div class="answer-box">No saved model prediction for this question.</div>',
                    unsafe_allow_html=True)
        return

    col1, col2 = st.columns(2)
    for col, label, pred in [(col1, "GCR Baseline", gcr), (col2, "DCA-Trie v1", dca)]:
        with col:
            st.markdown(f"**{label}**")
            if pred is None:
                st.markdown('<div class="answer-box">No saved prediction</div>', unsafe_allow_html=True)
                continue
            verdict_class = "gate-pass" if pred["correct"] else "gate-fail"
            verdict = "correct" if pred["correct"] else "incorrect"
            st.markdown(
                f'<div class="answer-box">{pred["answer"]}<br>'
                f'<span class="{verdict_class}">({verdict})</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown(f"**Ground truth:** {gt}")

    if gcr and dca and gcr["answer"] != dca["answer"]:
        st.caption(
            "GCR and DCA-Trie v1 disagree on this question — TypeOracle's filtering changed which "
            "reasoning path the LLM committed to. This is a concrete instance of the accuracy trade-off "
            "discussed in the results (see Chapter 4)."
        )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.title("DCA-Trie: Graph-Constrained Reasoning Demo")
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

    # ------------------------------------------------------------------
    # Step-by-step navigation state
    # ------------------------------------------------------------------
    STEPS = [
        ("Question", step_question, data),
        ("Knowledge Graph", step_kg, data),
        ("TypeOracle Gates", step_type_oracle, data["gated_paths"]),
        ("Constrained Decoding", step_constrained_decoding, data),
        ("Final Answer", step_answer, data),
    ]

    if st.session_state.get("_sample_idx") != sample_idx:
        st.session_state["_sample_idx"] = sample_idx
        st.session_state["_step"] = 0
    current = st.session_state.get("_step", 0)

    nav_prev, nav_progress, nav_next = st.columns([1, 4, 1])
    with nav_prev:
        if st.button("Back", disabled=current == 0, use_container_width=True):
            st.session_state["_step"] = current - 1
            st.rerun()
    with nav_next:
        if st.button("Next", disabled=current == len(STEPS) - 1, use_container_width=True, type="primary"):
            st.session_state["_step"] = current + 1
            st.rerun()
    with nav_progress:
        st.progress((current + 1) / len(STEPS))
        st.caption(f"Step {current + 1} of {len(STEPS)}: **{STEPS[current][0]}**")

    st.divider()

    # Render every step up to and including the current one, so earlier
    # steps stay visible for context while the story builds up live.
    for _, render, arg in STEPS[: current + 1]:
        render(arg)
        st.divider()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Paths:** {data['gated_paths']['total_paths']} total, "
                        f"{data['gated_paths']['admitted_paths']} admitted")
    st.sidebar.markdown(f"**Answer types:** {', '.join(data['gated_paths']['answer_types']) or '(none)'}")


if __name__ == "__main__":
    main()
