#!/usr/bin/env python3
"""DCA-Trie Demo: Compare Normal LLM vs RAG vs Constrained Decoding.

Usage:
    export GEMINI_API_KEY="your-key"
    python -m demo.app

Or without API key (shows DCA-Trie pipeline only):
    python -m demo.app
"""

from __future__ import annotations

import gradio as gr

from .config import GEMINI_API_KEY, SERVER_PORT
from .questions import CURATED_QUESTIONS
from .dca_trie import run_dca_trie
from .llm import call_normal_llm, call_rag_llm, call_path_guided_llm


def has_api_key() -> bool:
    """Check if Gemini API key is configured."""
    return bool(GEMINI_API_KEY)


def compare_approaches(question_idx: int) -> str:
    """Run all three approaches and return comparison.

    Args:
        question_idx: Index into CURATED_QUESTIONS.

    Returns:
        Markdown formatted comparison.
    """
    q_data = CURATED_QUESTIONS[question_idx]
    question = q_data["question"]
    q_entity = q_data["q_entity"]
    graph_triples = q_data["graph"]
    expected = q_data["answer"]

    # Run DCA-Trie pipeline (always works, no API needed)
    dca_result = run_dca_trie(question, q_entity, graph_triples)

    # Format DCA-Trie section
    dca_lines = [
        "**Step 1: Build graph from KG triples**",
        f"  Nodes: {dca_result.graph.number_of_nodes()}, Edges: {dca_result.graph.number_of_edges()}",
        "",
        "**Step 2: Initialize TypeOracle**",
        f"  Answer types inferred: {', '.join(str(t) for t in list(dca_result.answer_types)[:3])}...",
        "",
        "**Step 3: Enumerate paths (DFS)**",
        f"  Found {dca_result.total_paths} paths from topic entities",
        "",
        "**Step 4: Filter with TypeOracle gates**",
        f"  Range gate + type gate applied",
        f"  Paths kept: {dca_result.kept_paths}",
        f"  Paths removed: {dca_result.removed_paths}",
        f"  SIR: {dca_result.sir:.1%}",
        "",
        "**Step 5: Constrained generation**",
    ]

    if dca_result.best_path:
        from .kg import format_path_compact
        dca_lines.append(f"  Best path: {format_path_compact(dca_result.best_path)}")
    else:
        dca_lines.append("  No admissible paths found")

    dca_lines.append("")
    dca_lines.append("**All paths:**")
    for i, path in enumerate(dca_result.all_paths):
        from .kg import format_path_compact
        marker = "✓" if path in dca_result.filtered_paths else "✗"
        dca_lines.append(f"  {marker} {i+1}. {format_path_compact(path)}")

    dca_text = "\n".join(dca_lines)

    # LLM sections (only if API key available)
    if has_api_key():
        try:
            normal_result = call_normal_llm(question)
        except Exception as e:
            normal_result = f"Error: {e}"

        try:
            rag_result = call_rag_llm(question, graph_triples)
        except Exception as e:
            rag_result = f"Error: {e}"

        if dca_result.best_path:
            try:
                dca_gen_result = call_path_guided_llm(question, dca_result.best_path)
            except Exception as e:
                dca_gen_result = f"Error: {e}"
        else:
            dca_gen_result = "No path available for generation"
    else:
        normal_result = "(Set GEMINI_API_KEY to enable)"
        rag_result = "(Set GEMINI_API_KEY to enable)"
        dca_gen_result = "(Set GEMINI_API_KEY to enable)"

    # Format output
    output = f"""## Question
{question}

## Expected Answer
**{expected}**

---

### 1. Normal LLM (No Context)
{normal_result}

---

### 2. RAG (All KG Facts as Context)
{rag_result}

---

### 3. DCA-Trie (Constrained Decoding)

{dca_text}

**Generated answer (from best path):**
{dca_gen_result}
"""
    return output


def create_ui() -> gr.Blocks:
    """Create the Gradio UI."""
    with gr.Blocks(
        title="DCA-Trie: Dynamic Context-Aware Constrained Decoding",
    ) as demo:
        gr.Markdown(
            """
            # DCA-Trie: Dynamic Context-Aware Constrained Decoding

            Compare three approaches to knowledge graph question answering:

            | Approach | Description |
            |----------|-------------|
            | **Normal LLM** | No context, pure generation (may hallucinate) |
            | **RAG** | All KG facts injected as context |
            | **DCA-Trie** | TypeOracle-filtered paths → constrained generation |

            The DCA-Trie approach reduces the search space by filtering irrelevant paths
            before generation, ensuring structural faithfulness.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                question_dropdown = gr.Dropdown(
                    choices=[
                        (q["question"], i)
                        for i, q in enumerate(CURATED_QUESTIONS)
                    ],
                    label="Select Question",
                    value=0,
                )
                run_button = gr.Button(
                    "Run Comparison",
                    variant="primary",
                    size="lg",
                )

                gr.Markdown(
                    """
                    **Categories covered:**
                    - People & Positions
                    - Geography
                    - Literature & Arts
                    - Science
                    - Sports
                    - History
                    """
                )

            with gr.Column(scale=2):
                output = gr.Markdown(
                    label="Results",
                    value="Select a question and click 'Run Comparison'",
                )

        # Event handlers
        run_button.click(
            fn=compare_approaches,
            inputs=[question_dropdown],
            outputs=[output],
        )

        gr.Markdown(
            """
            ---
            ### How DCA-Trie works

            1. **Build graph** from KG triples
            2. **Initialize TypeOracle** with Freebase schema
            3. **Infer answer types** from question (e.g., "who" → Person)
            4. **Enumerate paths** via DFS from topic entities
            5. **Filter paths** using range gate + type gate
            6. **Generate** using constrained decoding on filtered trie

            **Key metric:** SIR (Semantic Irrelevance Ratio) = fraction of paths removed
            """
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=SERVER_PORT,
    )
