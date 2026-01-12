# Explanation Quality Evaluation

## What We're Evaluating

Twinkl generates explanations at two levels:
1. **Judge rationales**: Per-entry explanations for alignment scores
2. **Coach narratives**: Weekly digest summaries with evidence from journal history

This evaluation validates that explanations feel accurate and actionable to users.

---

## Explanation Sources

### Judge Rationales

For each alignment score, the Judge provides a rationale:

```json
{
  "Achievement": "+1",
  "rationale": "Entry shows prioritizing work performance (finishing deck for investor meeting) over personal plans. Clear demonstration of achievement-oriented behavior."
}
```

**Criteria for good rationales:**
- References specific details from the entry
- Explains *why* the score was assigned
- Ties behavior to the value dimension

### Coach Narratives (Future)

Weekly summaries that synthesize patterns:

```
"Your Benevolence score dropped this week. You mentioned helping others twice but
cancelled on a friend Saturday. This pattern has appeared in 3 of the last 4 weeks."
```

**Criteria for good narratives:**
- Cites specific evidence from journal entries
- Identifies patterns over time (not just single entries)
- Avoids prescriptive or judgmental language

---

## Evaluation Approach

### Primary: Likert Ratings (from PRD)

Show users their weekly digest and ask: **"Did this feel accurate?"**

| Rating | Meaning |
|--------|---------|
| 5 | Completely accurate — captures exactly what happened |
| 4 | Mostly accurate — minor misses but right overall |
| 3 | Somewhat accurate — got some things right |
| 2 | Mostly inaccurate — misses important context |
| 1 | Completely inaccurate — doesn't reflect my week |

### Secondary: Criteria-Based Scoring

For deeper analysis, rate explanations on three dimensions:

| Criterion | Question | Scale |
|-----------|----------|-------|
| **Correctness** | Does the explanation accurately reflect what happened? | 1-5 |
| **Specificity** | Does it reference concrete details, not vague generalities? | 1-5 |
| **Actionability** | Could the user take action based on this insight? | 1-5 |

---

## Evaluation Protocol

### For Synthetic Data (Automated)

1. Generate Judge rationales for 50-100 entries
2. Have LLM-as-meta-judge rate each rationale on correctness/specificity
3. Flag rationales that are:
   - Too generic ("This shows value alignment")
   - Factually wrong (misquotes entry content)
   - Circular ("Achievement score is +1 because of achievement behavior")

### For User Study (Manual)

1. **Sample size**: 5-10 users (from PRD)
2. **Duration**: 1-2 weeks of journaling
3. **Measurement points**:
   - After each weekly digest: "Did this feel accurate?" (5-point Likert)
   - Exit interview: Open-ended feedback on explanation quality

### Procedure

```
Day 1-7:     User journals normally
Day 7:       System generates weekly digest with explanations
             User rates: "Did this feel accurate?" [1-5]
Day 8-14:    Continue journaling
Day 14:      Second digest + rating
             Exit interview
```

---

## Success Criteria

| Metric | Target | Rationale |
|--------|--------|-----------|
| Mean Likert rating | ≥ 3.5/5 | Above neutral = generally useful |
| % ratings ≥ 4 | > 50% | Majority find it "mostly accurate" or better |
| Correctness (meta-judge) | > 80% | Rationales don't misquote or contradict entry |
| Specificity (meta-judge) | > 70% | Rationales reference concrete details |

---

## Known Limitations

1. **Subjectivity**: "Felt accurate" is inherently subjective
2. **Small sample**: 5-10 users limits statistical power
3. **Hawthorne effect**: Users may rate higher knowing researchers will see

**Mitigations:**
- Use consistent Likert anchors with behavioral definitions
- Collect qualitative feedback to contextualize ratings
- Compare ratings across different explanation types (Judge vs. Coach)

---

## Example Evaluation Output

After 10 users complete the study:

| User | Week 1 Rating | Week 2 Rating | Exit Feedback |
|------|---------------|---------------|---------------|
| U1 | 4 | 5 | "Spot on about the work-life thing" |
| U2 | 3 | 4 | "Got better in week 2" |
| U3 | 4 | 4 | "Useful but sometimes too vague" |
| ... | ... | ... | ... |

**Aggregate**:
- Mean: 3.8/5
- % ≥ 4: 65%
- Common feedback: "Helpful when specific, unhelpful when generic"

---

## References

- `docs/VIF/judge_implementation_spec.md` — Judge rationale format
- `docs/PRD.md` — Evaluation Strategy (Row 4: Explanation quality)
