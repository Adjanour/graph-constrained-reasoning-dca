# Presentation & Demo Plan

## Overview

Presentation flow: **Talk → Manim Visualization → Live UI Demo → Results**

1. Talk: What we set out to do, what the project is, how we did it
2. Manim animation: Animated walkthrough of KG-constrained decoding
3. Live UI demo: Side-by-side comparison (Normal LLM / RAG / DCA-Trie)
4. Results: Hits@1, F1, SIR comparison

---

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Animation | ManimCE + manim-slides | Pre-rendered, reliable, reveal.js export |
| UI Framework | Gradio | Purpose-built for ML demos, minimal code, clean layout |
| LLM Backend | Google Gemini API (free tier) | No credit card, 250 RPD, good quality |
| Alternative LLM | Ollama (local) | Fully offline, no API dependency |

### Gemini API Free Tier Limits

- Gemini 2.5 Flash: 10 RPM, 250,000 TPM, 250 RPD
- No credit card required
- More than enough for curated demo (5-6 questions × 3 methods = 18 requests)

---

## Components

### 1. Manim Animation (Highest Priority)

**Goal:** Visually explain how DCA-Trie works for the presentation

**Scenes to animate:**

| Scene | Content | Duration |
|-------|---------|----------|
| 1 | Knowledge graph as a graph (nodes + edges) | 10s |
 2 | GCR builds static trie — show all paths including irrelevant ones highlighted in red | 15s |
| 3 | SIR problem — show irrelevant paths wastes compute | 10s |
| 4 | DCA-Trie v2 — step-by-step pruning as tokens are generated, paths fade out | 20s |
| 5 | Final comparison — GCR trie size vs DCA-Trie trie size | 10s |

**Workflow:**
1. Write scenes as ManimCE `Slide` subclasses (manim-slides)
2. Render: `manim-slides render scenes.py SceneName`
3. Present: `manim-slides Scene1 Scene2 Scene3 Scene4 Scene5`
4. Export: reveal.js HTML for browser-based presentation

**Dependencies:**
- `manim` (ManimCE)
- `manim-slides`

**Status:** Not started

---

### 2. Gradio UI Demo

**Goal:** Live comparison showing constrained decoding beats normal LLM and RAG

**Layout:**
```
┌─────────────────────────────────────────────────────┐
│  DCA-Trie: Dynamic Context-Aware Constrained Decoding │
├─────────────────────────────────────────────────────┤
│  Question: [input]  [Run Comparison]                │
├─────────────────────────────────────────────────────┤
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐    │
│  │ Normal   │ │   RAG    │ │  DCA-Trie        │    │
│  │   LLM    │ │          │ │  Constrained     │    │
│  │          │ │          │ │                  │    │
│  │ answer   │ │ answer   │ │ answer + path    │    │
│  └──────────┘ └──────────┘ └──────────────────┘    │
│                                                     │
│  KG Path Visualization (optional)                   │
└─────────────────────────────────────────────────────┘
```

**Backend:**
- Normal LLM: Gemini API call, no context
- RAG: Retrieve top-K KG triples, inject into prompt
- DCA-Trie: Run constrained decoding pipeline (our implementation)

**Dependencies:**
- `gradio`
- `google-generativeai` (Gemini SDK)
- DCA-Trie pipeline code

**Status:** Not started

---

### 3. Curated Demo Questions

**Goal:** Pick 5-6 WebQSP questions that showcase the difference

**Criteria:**
- Multi-hop (2-3 hops) to show trie pruning benefit
- Questions where normal LLM hallucinates or gives wrong answer
- Questions where DCA-Trie path is clearly correct
- Mix of entity types (people, places, events)

**Source:** WebQSP test set, filtered by hop depth from notebook 06

**Status:** Not started

---

### 4. Full Evaluation (Objective v)

**Goal:** Complete Hits@1, F1 comparison for paper results section

**What's ready:**
- `experiments/type_oracle_full/run.py` — supports `--method all --datasets RoG-webqsp RoG-cwq`
- Notebook 05.5 — clean experiment notebook

**What's needed:**
- Run `python run.py --method all --datasets RoG-webqsp RoG-cwq --num_samples 50`
- Collect Hits@1, F1, structural faithfulness, trie size
- Stratify by hop depth

**Status:** Not started

---

### 5. Wire Up Backend

**Goal:** Connect Gradio UI to real pipelines

**Flow:**
1. User enters question in Gradio
2. Backend calls Gemini for Normal LLM
3. Backend calls Gemini with RAG context
4. Backend runs DCA-Trie constrained decoding
5. Display all three results side-by-side

**Status:** Blocked by #2, #3

---

## Objective Status (UMaT Template)

| # | Objective | Status |
|---|-----------|--------|
| i | Define & measure SIR | ✅ Done |
| ii | Semantic relevance scoring (pivoted to TypeOracle) | ✅ Done |
| iii | DCA-Trie v1 (static, FNR < 5%) | ✅ Done |
| iv | DCA-Trie v2 (dynamic) | ✅ Done |
| v | Evaluate vs GCR + CoT (Hits@1, F1, faithfulness, SIR) | 🔲 Pending |
| vi | Interactive prototype (UI demo) | 🔲 Pending |

---

## Timeline

| Day | Task |
|-----|------|
| 1 | Manim scenes 1-3 (KG, static trie, SIR problem) |
| 2 | Manim scenes 4-5 (DCA-Trie pruning, comparison) |
| 3 | Gradio UI scaffold + Gemini integration |
| 4 | Curated questions + wire up DCA-Trie pipeline |
| 5 | Full evaluation run (objective v) |
| 6 | Polish, test, rehearse |
| 7 | Presentation |

---

## Risks

| Risk | Mitigation |
|------|------------|
| Manim rendering slow | Use `-pql` (low quality) during dev, `-pqh` for final |
| Gemini API rate limits | Curated questions only, max 18 requests per demo |
| DCA-Trie pipeline errors | Test with fixed questions beforehand |
| Gradio layout issues | Keep UI simple, minimal customization |
