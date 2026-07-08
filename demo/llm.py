"""LLM interface for the demo.

Uses Google Gemini API for generation.
"""

from __future__ import annotations

import google.generativeai as genai

from .config import GEMINI_API_KEY, GEMINI_MODEL


def configure_llm():
    """Configure the Gemini API."""
    genai.configure(api_key=GEMINI_API_KEY)


def call_llm(prompt: str, system_instruction: str | None = None) -> str:
    """Call Gemini with a prompt.

    Args:
        prompt: The user prompt.
        system_instruction: Optional system instruction.

    Returns:
        Generated text.
    """
    configure_llm()
    model = genai.GenerativeModel(
        GEMINI_MODEL,
        system_instruction=system_instruction,
    )
    response = model.generate_content(prompt)
    return response.text


def call_normal_llm(question: str) -> str:
    """Call Gemini with no context (normal LLM baseline).

    Args:
        question: The question to answer.

    Returns:
        Generated answer.
    """
    prompt = f"Answer this question concisely in one sentence: {question}"
    return call_llm(prompt)


def call_rag_llm(question: str, kg_facts: list[tuple[str, str, str]]) -> str:
    """Call Gemini with KG facts as context (RAG baseline).

    Args:
        question: The question to answer.
        kg_facts: List of (subject, predicate, object) triples.

    Returns:
        Generated answer.
    """
    facts_str = "\n".join([f"- {s} {p} {o}" for s, p, o in kg_facts])
    prompt = (
        f"Based on the following knowledge graph facts, answer the question. "
        f"Only use information from the facts provided.\n\n"
        f"Facts:\n{facts_str}\n\n"
        f"Question: {question}\n\n"
        f"Answer concisely in one sentence:"
    )
    return call_llm(prompt)


def call_path_guided_llm(question: str, kg_path: list[tuple[str, str, str]]) -> str:
    """Call Gemini with a specific KG path as context (DCA-Trie output).

    Args:
        question: The question to answer.
        kg_path: The constrained KG path from DCA-Trie.

    Returns:
        Generated answer based on the path.
    """
    path_str = "\n".join([f"- {s} --[{p}]--> {o}" for s, p, o in kg_path])
    prompt = (
        f"Based on the following reasoning path from a knowledge graph, "
        f"answer the question. The path shows the exact relationships needed.\n\n"
        f"Reasoning path:\n{path_str}\n\n"
        f"Question: {question}\n\n"
        f"Answer concisely in one sentence:"
    )
    return call_llm(prompt)
