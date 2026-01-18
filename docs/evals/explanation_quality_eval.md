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

### For Synthetic Data (Automated) — Tiered Evaluation

#### Tier 1: Automated Code Checks (No LLM)

Fast, objective checks that don't require LLM calls:

| Check | Description | Target |
|-------|-------------|--------|
| **Groundedness** | % of rationales with verifiable quotes (substring match in entry) | > 70% |
| **Non-circularity** | % that don't contain the value name itself | > 95% |
| **Length** | Flag too-short (<10 words) or too-long (>50 words) | 90% in range |

**Implementation:**
```python
def check_groundedness(rationale: str, entry_text: str) -> bool:
    """Check if rationale contains verifiable content from entry."""
    # Extract quoted phrases from rationale
    quotes = re.findall(r'"([^"]+)"', rationale)
    if not quotes:
        return False  # No quotes = not grounded
    return any(quote.lower() in entry_text.lower() for quote in quotes)

def check_non_circularity(rationale: str, value_name: str) -> bool:
    """Check if rationale avoids using the value name."""
    # Normalize: "self_direction" -> ["self", "direction"]
    terms = value_name.replace("_", " ").split()
    return not any(term.lower() in rationale.lower() for term in terms)
```

#### Tier 2: Meta-Judge Evaluation (LLM-Based)

For rationales that pass Tier 1, evaluate with LLM:

| Criterion | Question | Scale |
|-----------|----------|-------|
| **Correctness** | Does the rationale accurately reflect what happened in the entry? | 1-5 |
| **Specificity** | Does it reference concrete actions/statements, not vague generalities? | 1-5 |

**Meta-judge prompt structure:**
- Input: Entry text + Judge's rationale + score
- Task: Rate correctness and specificity on 1-5 scale
- Output: Scores + brief justification

**Flag for human review if:**
- Meta-judge correctness < 3
- Meta-judge specificity < 3
- Meta-judge expresses uncertainty

#### Tier 3: Human Calibration (Small Sample)

Validate meta-judge accuracy against human judgment:

1. Randomly sample 20-30 rationales
2. Human rates same criteria (correctness, specificity)
3. Calculate agreement with meta-judge (Cohen's κ)
4. Target: κ > 0.6 (substantial agreement)

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

## Evaluation Pipeline

```
Judge produces rationales for N entries
              ↓
┌─────────────────────────────────────┐
│  Tier 1: Automated Code Checks      │
│  - Groundedness (verifiable quotes) │
│  - Non-circularity (no value name)  │
│  - Length (10-50 words)             │
│  Output: Pass/Fail + metrics        │
└─────────────────────────────────────┘
              ↓
       (Passed Tier 1)
              ↓
┌─────────────────────────────────────┐
│  Tier 2: Meta-Judge Evaluation      │
│  - Correctness (1-5)                │
│  - Specificity (1-5)                │
│  Output: Scores + flags for review  │
└─────────────────────────────────────┘
              ↓
       (Flagged or sampled)
              ↓
┌─────────────────────────────────────┐
│  Tier 3: Human Calibration          │
│  - 20-30 rationales human-rated     │
│  - Compare to meta-judge            │
│  Output: Cohen's κ agreement        │
└─────────────────────────────────────┘
```

---

## Failure Modes to Detect

| Failure Mode | Example | Detection Method |
|--------------|---------|------------------|
| **Hallucinated quotes** | "Entry mentioned 'staying late'" when it didn't | Tier 1: Groundedness check |
| **Generic explanation** | "Shows alignment with this value" | Tier 1: Length check + Tier 2: Specificity |
| **Circular reasoning** | "Achievement +1 because of achievement behavior" | Tier 1: Non-circularity check |
| **Wrong attribution** | Confuses which value a behavior supports | Tier 2: Meta-judge correctness |
| **Over-inference** | Reads too much into vague entry | Tier 2: Meta-judge correctness |

---

## Success Criteria

| Metric | Target | Tier | Rationale |
|--------|--------|------|-----------|
| Groundedness (code) | > 70% | 1 | Rationales should quote or reference entry content |
| Non-circularity (code) | > 95% | 1 | Rationales shouldn't just restate value name |
| Length compliance | > 90% | 1 | Most rationales should be 10-50 words |
| Correctness (meta-judge) | Mean > 3.5/5 | 2 | Rationales should be factually accurate |
| Specificity (meta-judge) | Mean > 3.5/5 | 2 | Rationales should cite concrete details |
| Human-meta agreement | κ > 0.6 | 3 | Meta-judge should align with human judgment |
| Mean Likert rating (users) | ≥ 3.5/5 | User study | Above neutral = generally useful |
| % ratings ≥ 4 (users) | > 50% | User study | Majority find it "mostly accurate" or better |

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
