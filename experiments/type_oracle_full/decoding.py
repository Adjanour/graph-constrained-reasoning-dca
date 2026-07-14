"""
decoding.py — Constrained decoding strategies for DCA-Trie.

- ``run_constrained_decoding`` — baseline & v1 single-pass GCR
- ``dca_v2_generate`` — v2 iterative hop-by-hop trie expansion
"""

from src.graph_constrained_decoding import GraphConstrainedDecoding

from trie_utils import build_trie_from_strings
from utils import PATH_START, PATH_END, logger


def run_constrained_decoding(model, input_builder, data, trie):
    """Run graph-constrained decoding for a single question (baseline / v1)."""
    input_query, ground_paths, _ = input_builder.process_input(data, return_tire=False)
    start_token_ids = model.tokenizer.convert_tokens_to_ids(input_builder.PATH_START_TOKEN)
    end_token_ids = model.tokenizer.convert_tokens_to_ids(input_builder.PATH_END_TOKEN)
    llm_input = model.prepare_model_prompt(input_query)
    prediction = model.generate_sentence(
        llm_input,
        trie,
        start_token_ids=start_token_ids,
        end_token_ids=end_token_ids,
        enable_constrained_by_default=False,
    )
    return prediction, ground_paths


def dca_v2_generate(
    data,
    nx_graph,
    llm_model,
    tokenizer,
    oracle,
    max_hops,
    max_new_tokens,
    input_builder,
):
    """
    DCA-Trie v2: iterative hop-by-hop trie expansion (Algorithm 2).

    Start with first-hop gated neighbours, expand at each entity commit.
    Uses the same prompt builder as baseline/v1 for consistency, then
    strips the answer prompt suffix for iterative continuation.
    """
    question = data["question"]
    start_entities = data.get("q_entity", [])
    answer_types = oracle.infer_answer_types(question)
    start_id = tokenizer.convert_tokens_to_ids(PATH_START)
    end_id = tokenizer.convert_tokens_to_ids(PATH_END)

    first_hop_paths = []
    for entity in start_entities:
        if entity not in nx_graph:
            continue
        for neighbor in nx_graph.neighbors(entity):
            rel = nx_graph[entity][neighbor]["relation"]
            if not oracle.range_gate(rel, neighbor):
                continue
            first_hop_paths.append(f"{entity} -> {rel} -> {neighbor}")

    if not first_hop_paths:
        return None

    current_trie = build_trie_from_strings(tokenizer, first_hop_paths)
    if current_trie is None:
        return None

    # Build the initial prompt via the input_builder so it matches the format
    # used by baseline and v1.  Strip the answer-prompt suffix (everything
    # after the path-generation instruction) because v2 appends its own
    # hop-by-hop continuation.
    prompt, _, _ = input_builder.process_input(data, return_tire=False)
    answer_markers = ["Answer:", "A:", "answer:"]
    for marker in answer_markers:
        idx = prompt.rfind(marker)
        if idx != -1:
            prompt = prompt[:idx].rstrip()
            break

    output_text = ""
    committed_entity = start_entities[0] if start_entities else None
    hop = 0

    for _step in range(max_hops * 3):
        llm_input = llm_model.prepare_model_prompt(prompt)
        gcr = GraphConstrainedDecoding(
            tokenizer, current_trie, start_id, end_id, enable_constrained_by_default=False
        )
        inputs = tokenizer(llm_input, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs.input_ids.to(llm_model.model.device)
        attn_mask = inputs.attention_mask.to(llm_model.model.device)

        res = llm_model.model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            generation_config=llm_model.generation_cfg,
            prefix_allowed_tokens_fn=gcr.allowed_tokens_fn,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            trust_remote_code=True,
        )

        output = tokenizer.decode(res.sequences[0][input_ids.shape[1]:], skip_special_tokens=True)
        output_text += output

        if PATH_END in output or tokenizer.eos_token in output:
            break

        # Robust entity extraction: clean PATH markers, split on " -> ",
        # and take the last segment as the committed entity.
        clean_output = output.replace(PATH_END, "").replace(PATH_START, "").strip()
        if not clean_output:
            continue

        segments = [s.strip() for s in clean_output.split(" -> ")]
        new_entity = segments[-1] if segments else None

        if new_entity is None or new_entity == committed_entity:
            continue
        committed_entity = new_entity
        hop += 1

        if hop >= max_hops:
            break

        is_terminal = hop + 1 >= max_hops
        new_paths = []
        if committed_entity in nx_graph:
            for neighbor in nx_graph.neighbors(committed_entity):
                rel = nx_graph[committed_entity][neighbor]["relation"]
                if not oracle.range_gate(rel, neighbor):
                    continue
                if is_terminal and not oracle.type_gate(neighbor, answer_types, hop + 1, max_hops):
                    continue
                new_paths.append(f"{committed_entity} -> {rel} -> {neighbor}")

        if new_paths:
            expanded_trie = build_trie_from_strings(tokenizer, new_paths)
            if expanded_trie is not None:
                current_trie = expanded_trie
                prompt = prompt + f"\n{output.strip()}\n"
        else:
            break

    return output_text
