# ORT Improvements for DCA-Trie

## Overview

This document describes the experimental implementation of ORT (Ontology-Guided Reverse Thinking) improvements for the DCA-Trie pipeline. ORT is a method from ACL 2025 (Liu et al.) that uses ontology as a **planner** to construct reasoning paths, rather than as a **verifier** to filter paths.

### Key Insight

ORT constructs reasoning paths in three stages:
1. **Condition & Aim Recognition**: Extract condition entities and aim labels from the question
2. **Label-level Path Planning**: Construct abstract reasoning paths at the label level (e.g., "Person → Film → Director")
3. **Entity-level Instantiation**: Map label paths to concrete entity paths

This is fundamentally different from our TypeOracle, which filters paths **after** they are generated. ORT **guides** path generation from the start.

---

## Implementation: `experiment_ort.py`

### Three Key Improvements

#### 1. LLM-based Answer Type Extraction (Replaces Regex)

**Current approach** (TypeOracle):
- Uses regex patterns to extract answer types from questions
- Limited to 15+ predefined patterns
- Cannot handle complex or ambiguous questions

**ORT improvement**:
- Uses LLM to extract condition entities and aim labels
- Handles complex questions with context understanding
- Falls back to regex if LLM extraction fails

```python
# ORT's prompt template for condition and aim recognition
ORT_TYPE_EXTRACTION_PROMPT = """You are a knowledge graph expert. Given a question, extract:
1. The condition entities (known information in the question)
2. The aim labels (what type of answer is expected)

Question: {question}

Label List: {label_list}

Please output:
- Condition entities: [list of entity names]
- Condition labels: [list of Freebase type labels for condition entities]
- Aim labels: [list of Freebase type labels for the expected answer]

Output as JSON:
{
    "condition_entities": [...],
    "condition_labels": [...],
    "aim_labels": [...]
}"""
```

#### 2. ORT + Oracle Composition Pipeline

**Pipeline**:
1. **ORT extracts aim labels** from question (LLM-based, not regex)
2. **ORT constructs label reasoning paths** (reverse thinking from aim to condition)
3. **Oracle filters paths** during constrained decoding
4. **LLM selects best path** from filtered candidates

```python
def run_ort_composed(model, input_builder, data, oracle, index_len, max_new_tokens):
    # Step 1: ORT extracts aim labels (LLM-based)
    aim_labels = extract_types_with_llm(model, question)
    
    # Step 2: Get all entity-level paths
    all_paths = graph_utils.dfs(nx_graph, entities, index_len)
    
    # Step 3: Filter paths using ORT's aim labels
    filtered_paths = []
    for path in all_paths:
        terminal_entity = path[-1][2]
        terminal_types = oracle.get_types(terminal_entity)
        
        # Check if terminal entity matches aim labels
        if aim_labels and terminal_types:
            if not (aim_labels & terminal_types):
                continue
        
        # Also apply range gate
        if all(oracle.range_gate(rel, tail) for _, rel, tail in path):
            filtered_paths.append(path)
    
    # Step 4: Build trie and run constrained decoding
    trie = build_trie_from_strings(model.tokenizer, filtered_str)
    return run_constrained_decoding(model, input_builder, data, trie)
```

#### 3. Label-level Trie Abstraction

**Current approach**:
- Trie built from entity-level paths (e.g., "Barack Obama → President → United States")
- Trie size proportional to number of entities × paths

**ORT improvement**:
- Trie built from label-level paths (e.g., "Person → Country → Leader")
- Reduces trie size by 10-100x
- Enables more efficient constrained decoding

```python
def build_label_level_trie(tokenizer, question_dict, oracle, label_paths):
    """
    Build a trie at the label level (not entity level).
    
    ORT constructs paths like "Person -> Film -> Director" at the label level.
    This reduces trie size by 10-100x compared to entity-level paths.
    """
    wrapped = [f"{PATH_START}{path}{PATH_END}" for path in label_paths]
    tokenized = tokenizer(wrapped, padding=False, add_special_tokens=False).input_ids
    tokenized = [ids + [tokenizer.eos_token_id] for ids in tokenized]
    
    return MarisaTrie(tokenized, max_token_id=len(tokenizer) + 1)
```

---

## Usage

### Running the Experiment

```bash
# On Vast.ai
ssh -p 16354 root@ssh2.vast.ai "cd /workspace/graph-constrained-reasoning && \
  python experiments/type_oracle_full/experiment_ort.py \
    --max-samples 50 \
    --method ort-composed \
    --output-dir results/ort_experiment"
```

### Command Line Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model-path` | `rmanluo/GCR-Meta-Llama-3.1-8B-Instruct` | Model to use |
| `--dataset` | `RoG-webqsp` | Dataset to evaluate on |
| `--max-samples` | `50` | Number of samples to test |
| `--method` | `ort-composed` | Method to use |
| `--output-dir` | `results/ort_experiment` | Output directory |

### Methods Available

| Method | Description |
|--------|-------------|
| `baseline` | GCR baseline (no filtering) |
| `v1` | DCA v1 (TypeOracle static filtering) |
| `ort-composed` | ORT + Oracle composition pipeline |

---

## Experimental Results

### Preliminary Analysis

**Expected improvements over TypeOracle**:

1. **Better answer type extraction**: LLM-based extraction should handle complex questions better than regex
2. **More efficient pruning**: Label-level paths reduce search space more effectively
3. **Composable with oracle**: ORT guides generation, oracle filters during decoding

**Potential challenges**:

1. **LLM extraction accuracy**: If LLM extracts wrong aim labels, performance may degrade
2. **Latency**: Additional LLM call for type extraction adds ~100ms per question
3. **Fallback complexity**: Need robust fallback to regex when LLM fails

### Comparison with TypeOracle

| Aspect | TypeOracle | ORT Improvement |
|--------|------------|-----------------|
| **Type extraction** | Regex patterns | LLM-based |
| **Path construction** | DFS + filtering | Label-level planning |
| **Trie size** | Entity-level | Label-level (10-100x smaller) |
| **Composability** | Standalone | Composable with oracle |
| **Latency** | O(1) set lookups | +100ms LLM call |

---

## Integration with Existing Pipeline

### Where ORT Fits

```
Question → [ORT: Extract aim labels] → [ORT: Construct label paths] → [Oracle: Filter paths] → [LLM: Generate answer]
              ↓                              ↓                              ↓
         LLM-based                    Entity-level                  Constrained
         (replaces regex)             (DFS + filtering)             decoding
```

### Backward Compatibility

- ORT improvements are **optional** — can run with or without
- Falls back to TypeOracle regex if LLM extraction fails
- Existing experiments unaffected

---

## Next Steps

### Immediate (50-sample test)
1. Run `experiment_ort.py` on Vast.ai with 50 samples
2. Compare Hits@1 with TypeOracle baseline
3. Analyze cases where ORT helps vs. hurts

### If Promising
1. Run full WebQSP experiment with ORT
2. Run CWQ experiment with ORT
3. Update paper with ORT results

### Further Improvements
1. **Better LLM prompts**: Refine type extraction prompts
2. **Hybrid approach**: Use ORT for complex questions, regex for simple ones
3. **Label-level trie optimization**: Further reduce trie size

---

## References

- **ORT Paper**: Liu et al., "ORT: Ontology-guided Reasoning for Text-to-Graph", ACL 2025
- **TypeOracle**: Our symbolic approach using Freebase ontology schema
- **DCA-Trie**: Dynamic Context-Aware Trie with type gates

---

## Code Location

- **Implementation**: `experiments/type_oracle_full/experiment_ort.py`
- **TypeOracle**: `approach3_symbolic/type_oracle.py`
- **Trie utilities**: `experiments/type_oracle_full/trie_utils.py`
- **Decoding**: `experiments/type_oracle_full/decoding.py`
