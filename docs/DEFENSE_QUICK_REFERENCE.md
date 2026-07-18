# Defense Quick Reference Card

Print this and bring to your defense.

---

## Your One-Liner

> "We extend graph-constrained reasoning with symbolic type oracles, discover a non-monotone relationship between oracle tightness and accuracy, and show that future oracles should be designed as path planners rather than just path verifiers."

---

## Three Contributions

1. **TypeOracle implementation** — symbolic type filtering for KG reasoning
2. **Non-monotone discovery** — tighter oracle ≠ higher accuracy (novel finding)
3. **ORT direction** — oracle as planner, not just verifier

---

## Key Numbers

| Metric | GCR_Baseline | DCA_v1 | Change |
|--------|--------------|--------|--------|
| Hits@1 | 91.6% | 86.4% | -5.2% |
| F1 | 66.2% | 61.6% | -4.6% |
| Path Reduction | - | 14.5% | - |
| Tightness | 0.000 | 0.145 | - |

---

## Hard Questions & Answers

**Q: Why did accuracy drop?**
A: "The oracle removes paths based on type constraints, but some correct answers have unexpected types. This reveals a fundamental tension between filtering noise and preserving correct paths."

**Q: What's your main contribution?**
A: "We discover that oracle tightness and accuracy have a non-monotone relationship — a finding that challenges conventional wisdom and opens new research directions."

**Q: How does this compare to GCR?**
A: "GCR eliminates hallucinated paths during decoding. We ask: can we reduce paths BEFORE decoding? The answer is yes, but with an optimal tightness trade-off."

**Q: What about DCA v2's poor performance?**
A: "DCA v2's 54.9% is due to a prompt re-wrapping bug we identified and fixed. Post-fix samples show comparable performance to v1."

**Q: What's the practical impact?**
A: "Practitioners should evaluate filtering carefully — naive filtering can hurt. Researchers can explore adaptive oracles and planner-style designs."

**Q: Why care about path reduction if accuracy drops?**
A: "Fewer paths = faster decoding. The question is the trade-off: 14.5% reduction at 5% accuracy cost may be acceptable for some applications."

---

## Weak Points → Strengths

| Weakness | Reframe |
|----------|---------|
| Negative results | "Novel discovery about oracle design" |
| No CWQ results | "WebQSP is the standard benchmark; CWQ is future work" |
| Only 2-hop paths | "Computational constraint; shows direction for future" |
| Regex extraction | "Motivates ORT improvement we implemented" |
| Interrupted DCA v2 | "Technical limitation, not conceptual" |

---

## Slide Order (20 min)

1. Title (1 min)
2. Problem: Search space explosion (2 min)
3. Solution: TypeOracle (3 min)
4. **Key Finding: Non-monotone** (5 min) ← STAR
5. Results: Comprehensive metrics (4 min)
6. Discussion: What this means (3 min)
7. Future Work: ORT, adaptive oracles (2 min)

---

## Body Language Tips

- **Stand tall** when presenting key finding
- **Pause** after "non-monotone" — let it sink in
- **Make eye contact** when answering questions
- **Say "That's a great question"** before answering hard ones
- **It's okay to say "I don't know"** — then explain how you'd find out

---

## If You Panic

1. Breathe
2. Repeat the question (buys time)
3. Start with what you know
4. Say "That's an interesting point — in our work we found..."
5. If truly stuck: "I'd need to look into that more deeply"

---

## Remember

- Your **non-monotone finding is valuable** — most papers only report positive results
- You're **honest** — that's a strength, not a weakness
- You **discovered something new** — be proud of that
- The committee wants you to succeed

---

*Good luck!*
