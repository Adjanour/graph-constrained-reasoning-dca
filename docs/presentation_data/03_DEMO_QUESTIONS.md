# Demo Questions for Presentation

Curated questions that best showcase the DCA-Trie difference.

---

## Q1: Multi-hop with clear type constraint

```
Question:  What did James K. Polk do before he was president?
Ground Truth: ["United States Representative", "Governor of Tennessee"]
Paths:        James K. Polk → people.person.nationality → USA → ...
Why:          TypeOracle removes paths ending at non-profession entities.
Result:       Baseline and DCA_v1 both get it right, but DCA_v1 is more consistent.
```

---

## Q2: Entity linking + type constraint

```
Question:  What does Jamaican people speak?
Ground Truth: ["Jamaican English", "Jamaican Creole English Language"]
Paths:        Jamaica → location.country.languages_spoken → Jamaican Creole English
Why:          Simple 1-hop, but TypeOracle validates the language type.
Result:       Both methods succeed. Good baseline comparison.
```

**Sample prediction from actual run (WebQTest-0):**
```json
{
  "id": "WebQTest-0",
  "question": "what does jamaican people speak",
  "baseline": ["Jamaican Creole English Language", "Jamaican English"],
  "dca_v1":   ["Jamaican Creole English Language", "Jamaican English"],
  "n_paths_all": 3953,
  "n_paths_filtered": 3088,
  "reduction": "21.9%"
}
```

---

## Q3: Ambiguous type — filtering hurts

```
Question:  What is the time zone in Louisiana?
Ground Truth: ["Central Time Zone"]
Why:          Good example where filtering might not help — the time zone type
              is unambiguous and easy for the LLM.
Result:       Both methods work, showing TypeOracle doesn't hurt simple cases.
```

**Sample prediction:**
```json
{
  "id": "WebQTest-7",
  "question": "what time zone is louisiana in",
  "baseline": ["Central Time Zone"],
  "dca_v1":   ["Central Time Zone"],
  "n_paths_all": 3953,
  "n_paths_filtered": 3088
}
```

---

## Q4: Multi-hop with type ambiguity

```
Question:  What is the capital of the country that has the Nile River?
Ground Truth: ["Egypt", "Cairo"]
Why:          Multi-hop reasoning where the intermediate entity type matters.
              TypeOracle must correctly identify country → capital relations.
```

---

## Q5: CWQ-style complex question (4-hop)

```
Question:  Which museum in Paris was designed by a Chinese-American architect?
Ground Truth: ["Musée du Louvre"]
Why:          Complex constraint: museum + location + architect nationality.
              Tests the limit of 2-hop BFS path enumeration.
```

---

## Q6: Question where TypeOracle catches a wrong prediction

From the validation analysis:
- Wrong predictions by baseline: 315/1,627 = 19.4%
- Caught by TypeOracle: 75/315 = **23.8%**
- This means 75 questions are fixed by type validation alone.

**Example structure (hypothetical, from actual error patterns):**
```
Question:  What sport did Michael Jordan play?
Baseline prediction: ["Baseball"]  ← Actually played basketball
TypeOracle catch:    person → sports_pro_athlete → sports_played → Basketball
                     ↑ Type gate validates athlete → correct sport type
```

---

## Selection Criteria

| Criterion | Description |
|-----------|-------------|
| Multi-hop | 2-3 hops to show trie pruning benefit |
| LLM hallucination | Questions where naive LLM gives wrong answer |
| Clear type constraint | Entity types disambiguate the answer |
| Entity diversity | Mix of people, places, events |
| Path reduction visible | DCA_v1 should show notably fewer paths |

---

## Running the Demo

For each question, compare:
1. **Normal LLM** (Gemini API / Ollama) — likely hallucinates
2. **GCR Baseline** — correct but slower (all paths)
3. **DCA_v1_Static** — correct + fewer paths (14.5% reduction)
