# DoG (Decoding on Graphs) Analysis — Key Insights for DCA-Trie v2

**Paper**: *Decoding on Graphs: A Principled Decoding Framework for Complex Reasoning on Knowledge Graphs*  
**Authors**: Yanbin Lin, Qi Wang, Yushan Zhu, Zili Wang, Shuaiqiang Wang, Dawei Yin  
**Venue**: ACL 2025 | **arXiv**: 2410.18415

---

## 1. Core Insight: Well-Formed Chain

DoG introduces the **well-formed chain** principle: every entity mentioned in a reasoning step must be either the question topic or a previously mentioned entity. This prevents the LLM from hallucinating entities that don't exist in the graph.

```
❌ Bad:  "1. < Albert Einstein -> field -> Physics >"
         (Physics is not in the graph, not a valid entity)

✅ Good: "1. < Albert Einstein -> nationality -> Germany >"
         "2. < Albert Einstein -> field -> Physics >"
         (Albert Einstein was mentioned in step 1, so it's valid in step 2)
```

---

## 2. DoG's Three-Phase Architecture

### Phase 1: Graph Construction
- Receive KG subgraph from GNN (triplets: `A -> B -> C`)
- Build adjacency index: `triples_of_ent[entity_id]` = all triples involving that entity

### Phase 2: Graph-Aware Constrained Decoding (per step)
For each reasoning step (up to `max_step`):

1. **Head Pool Selection** (per beam):  
   - Start with question entities as initial head pool  
   - After each step, update head pool: add both head AND tail of the generated triple  
   - This ensures all previously mentioned entities are available for next step

2. **Triple Generation** (constrained):  
   - For each entity in beam's head pool, collect all triples from `triples_of_ent[entity]`  
   - Build per-beam trie from allowed triples  
   - Generate with `ExtractiveGeneration` logits processor  
   - Output format: `step. < head -> relation -> tail >`

3. **Description Generation** (unconstrained):  
   - Append "This tells us" to the generated triple  
   - Generate free-form reasoning about the step  
   - This gives the LLM context for the next step

### Phase 3: Answer Generation
- When beam reaches terminal state (ends with `\n\n`) or exceeds `max_step`  
- Generate answer from the free-form reasoning

---

## 3. DoG vs DCA-Trie v2: Key Differences

| Aspect | DoG | DCA-Trie v2 (current) |
|--------|-----|----------------------|
| **Trie construction** | Dynamic per beam per step | Static per step (shared) |
| **Head pool** | Per-beam, tracks all mentioned entities | None — only tracks committed entity |
| **Beam diversity** | Multiple beams with different head pools | Single beam |
| **Pruning** | Before trie insertion (drop beam if no valid triples) | After generation (backtrack) |
| **Entity tracking** | Via head pool set | Via string parsing of output |
| **Two-phase gen** | Triple (constrained) + Description (unconstrained) | Single phase |
| **Exact matching** | `entity_text == entity` | Substring matching in paths |
| **Sentinel tokens** | `<` and `>` around triples | `<PATH>` and `</PATH>` around paths |

---

## 4. DoG's `ExtractiveGeneration` — The Key Innovation

```python
class ExtractiveGeneration(LogitsProcessor):
    def __call__(self, input_ids, scores):
        beam_prefixes = input_ids[:, self.input_start_len:].tolist()
        for i, prefix in enumerate(beam_prefixes):
            trie = self.tries[i // self.beam_size]  # Per-beam trie!
            options = self.valid_next_tokens(trie, prefix)
            mask = torch.zeros(scores[i].numel(), dtype=torch.bool)
            mask[options] = True
            scores[i][~mask] = float("-inf")
        return scores
```

**Key**: Each beam gets its own trie built from its head pool's triples. This is fundamentally different from our v2 which shares one trie across all beams.

---

## 5. DoG's `Graph_Processor.allowed_triples_processor` — How It Works

```python
def allowed_triples_processor(self, beams, prompt_len, step, beam_size):
    for beam in beams:
        allowed_triples = []
        for head in beam.head_pool:  # Per-beam head pool!
            for item in self.triples_of_ent[head]:
                if item not in generated_sequence:  # Skip already-used triples
                    allowed_triple = f"{step}. < {item} >"
                    allowed_triples.append(allowed_triple)
        # Build per-beam trie from allowed triples
        context_tokens.extend(allowed_triple_tokens)
    # Create ExtractiveGeneration with per-beam tries
    lp = ExtractiveGeneration(max(prompt_toks_len), batch_context_tokens, beam_size)
```

**Key**: The trie is rebuilt per beam at each step based on that beam's head pool. No dead-end backtracking needed.

---

## 6. What Our v2 Is Missing (and How to Fix It)

### Problem 1: No Per-Beam Head Pool
**DoG**: `beam.head_pool = {entity_id_1, entity_id_2, ...}` — set of entity IDs that can be heads  
**Our v2**: `committed_entity = "string_name"` — single string, no set

**Fix**: Track head_pool per beam as a set of entity names (matching DoG's pattern)

### Problem 2: Shared Trie Across Beams
**DoG**: Builds separate trie per beam from `triples_of_ent[head]` for each head in head_pool  
**Our v2**: `current_trie = build_trie_from_strings(tokenizer, new_paths)` — one trie for all

**Fix**: Build per-beam tries, or use a single trie that's the union of all beams' allowed paths

### Problem 3: No Backtracking — Dead-End Handling
**DoG**: If `len(allowed_triples) == 0`, skip that beam entirely (no backtracking)  
**Our v2**: `if not new_paths: break` — stops entirely

**Fix**: Continue with remaining beams, drop beams with no valid extensions

### Problem 4: String Parsing for Entity Extraction
**DoG**: Uses `ent_name2id` mapping and splits on `->` to get entity IDs  
**Our v2**: Splits on `->` and takes last segment as string

**Fix**: This is fine for our use case — DoG needs IDs for index lookup, we use string names

### Problem 5: No Two-Phase Generation
**DoG**: Triple (constrained) → Description (unconstrained) → repeat  
**Our v2**: Single generation per step

**Fix**: Optional — DoG's description phase helps with reasoning but isn't strictly necessary for correctness. Can add later.

---

## 7. Proposed v2 Rewrite Algorithm

```
FUNCTION dca_v2_generate(data, nx_graph, llm_model, tokenizer, oracle, max_hops):
    question = data["question"]
    start_entities = data.get("q_entity", [])
    answer_types = oracle.infer_answer_types(question)
    
    # Phase 1: Initialize beams
    beams = []
    for entity in start_entities:
        if entity not in nx_graph: continue
        head_pool = {entity}
        first_hop_paths = get_gated_paths(nx_graph, entity, oracle, answer_types, 1, max_hops)
        if first_hop_paths:
            beams.append(Beam(sequence=initial_prompt, head_pool=head_pool, paths=first_hop_paths))
    
    # Phase 2: Iterate hops
    for hop in range(2, max_hops + 1):
        new_beams = []
        for beam in beams:
            # Get allowed paths from beam's head_pool
            allowed_paths = []
            for head in beam.head_pool:
                if head not in nx_graph: continue
                for neighbor in nx_graph.neighbors(head):
                    rel = nx_graph[head][neighbor]["relation"]
                    if not oracle.range_gate(rel, neighbor): continue
                    if hop == max_hops and not oracle.type_gate(neighbor, answer_types, hop, max_hops): continue
                    allowed_paths.append(f"{head} -> {rel} -> {neighbor}")
            
            if not allowed_paths: continue  # Drop beam (DoG style)
            
            # Build per-beam trie
            trie = build_trie_from_strings(tokenizer, allowed_paths)
            
            # Generate with constrained decoding
            output = generate_with_trie(model, beam.sequence, trie)
            
            # Parse output, extract new entity, update head_pool
            new_entity = parse_entity(output)
            new_head_pool = beam.head_pool | {new_entity}
            new_beams.append(Beam(sequence=output, head_pool=new_head_pool))
        
        beams = sorted(new_beams, key=score)[:beam_size]
    
    # Phase 3: Generate answer from best beam
    return generate_answer(beams[0])
```

---

## 8. Key Implementation Details

### 8.1 Trie Format for v2
DoG uses a Python dict trie with token IDs as keys. Our MarisaTrie is more efficient but requires pre-building. For v2, we can either:
- Use Python dict trie (like DoG) for dynamic per-beam construction
- Use MarisaTrie but rebuild per beam (slightly slower but consistent with v1)

**Recommendation**: Use Python dict trie for flexibility, matching DoG's approach.

### 8.2 Prompt Format
DoG's format: `"step. < head -> relation -> tail >"`  
Our format: `"<PATH>head -> relation -> tail</PATH>"`

We should keep our format for consistency with v1/baseline, but the trie construction logic is the same.

### 8.3 Beam Management
- Start with `beam_size` beams (one per start entity, or duplicate if fewer)
- At each step, expand each beam, keep top-`beam_size` by score
- Drop beams with no valid extensions (DoG style)

### 8.4 Score Tracking
DoG tracks `sequence_scores` from `model.generate()`. We can use the same approach with `return_dict_in_generate=True`.

---

## 9. Files to Modify

| File | Change |
|------|--------|
| `experiments/type_oracle_full/decoding.py` | Rewrite `dca_v2_generate` with per-beam dynamic trie |
| `experiments/type_oracle_full/trie_utils.py` | Add `build_dict_trie_from_strings()` for dynamic construction |
| `src/graph_constrained_decoding.py` | Add `DynamicGraphConstrainedDecoding` class for per-beam trie |
| `experiments/type_oracle_full/experiment.py` | Update `_run_v2` to pass beam_size parameter |

---

## 10. Expected Improvements Over Current v2

1. **No dead-end backtracking**: Drop beams with no valid extensions instead of backtracking
2. **Per-beam diversity**: Different beams explore different paths through the graph
3. **TypeOracle integration**: Use `range_gate` and `type_gate` at each step (DoG doesn't have this — it's our advantage)
4. **Cleaner prompt handling**: No context accumulation issues
5. **Beam pruning**: Keep only top-`beam_size` beams at each step

---

## 11. DoG's Limitations (Where We Can Do Better)

1. **No type constraints**: DoG doesn't use answer type inference — we have `type_gate`
2. **No range constraints**: DoG doesn't check relation ranges — we have `range_gate`
3. **Exact matching only**: DoG uses `entity_text == entity` — we can use substring matching for fuzzy matching
4. **No `<PATH>` sentinel**: DoG's format is different from GCR — we maintain compatibility
5. **Incomplete GitHub**: DoG's repo is missing `Dog.py` and `utils.py` — we have full implementation

---

## 12. Implementation Status

### Completed
1. **DoG analysis document** (`docs/DOG_ANALYSIS.md`) — comprehensive comparison
2. **Dict trie builder** (`trie_utils.py::build_dict_trie`) — Python dict trie for dynamic construction
3. **Dict trie query** (`trie_utils.py::dict_trie_get`) — O(1) lookup per token
4. **BeamUnit dataclass** (`decoding.py`) — tracks sequence, head_pool, scores, step
5. **dca_v2_generate rewrite** (`decoding.py`) — DoG-style per-beam dynamic trie with TypeOracle gates
6. **_get_gated_paths helper** (`decoding.py`) — combines topological enumeration + semantic pruning
7. **_extract_path_content helper** (`decoding.py`) — extracts path between `<PATH>` and `</PATH>`
8. **_run_v2 update** (`experiment.py`) — accepts beam_size parameter

### Key Design Decisions
1. **Topological + Semantic**: Trie encodes all graph-structure-valid triples (DoG), then TypeOracle gates prune type-incompatible ones (our contribution).
2. **Prompt format**: Uses `<PATH>` / `</PATH>` consistently with v1/baseline (not DoG's `<` / `>`).
3. **EOS token**: Uses `</PATH>` as `eos_token_id` to stop generation at end of each hop.
4. **One hop per generate call**: Each `generate()` produces exactly one hop, stopping at `</PATH>`.
5. **Per-beam head pool**: Tracks all mentioned entities (DoG-style), not just the last committed entity.
6. **No dead-end backtracking**: Drop beams with no valid extensions (DoG style).
7. **Default beam_size=1**: Can be increased for beam search.

### Fixed Issues
1. ~~Prompt format mismatch (< vs <PATH>)~~ — Fixed: uses `<PATH>` in prompt
2. ~~Step number mismatch~~ — Fixed: removed step numbers from prompt
3. ~~EOS token issue~~ — Fixed: uses `</PATH>` as `eos_token_id`
4. ~~Incremental generation~~ — Fixed: one hop per `generate()` call

### How It Works (Step by Step)

```
For each hop (1 to max_hops):
  For each beam:
    1. Enumerate 1-hop triples from head_pool (topological)
    2. Prune via range_gate + type_gate (semantic)
    3. Build per-beam MarisaTrie from surviving triples
    4. Prompt: "<prompt so far>\n<PATH>"
    5. Generate with constrained decoding, stop at </PATH>
    6. Parse output: "<PATH>entity -> rel -> neighbour</PATH>"
    7. Update head_pool with new entity
    8. Accumulate path: "A -> R1 -> B -> R2 -> C"
  Keep top-beam_size beams
```

### Remaining Work
1. Run full experiment on WebQSP and CWQ with the Llama model
2. Compare results with baseline, v1, and GCR
3. Add beam search support (beam_size > 1) if needed

---

*Analysis completed: 2026-07-27*
*Status: v2 rewrite completed and tested, ready for full experiment*
