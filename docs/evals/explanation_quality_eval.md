# Explanation Quality Evaluation

## What We're Evaluating

Twinkl generates explanations at two levels:
1. **LLM-Judge rationales**: Per-Journal-Entry explanations for alignment scores
2. **Weekly Coach narratives**: Weekly Digest summaries with evidence from Journal Entry history

This evaluation validates that explanations feel accurate and actionable to users.

---

## Implementation Status

**Status:** 🟡 Partial

### What's Implemented
- Rationale generation working (1,594/1,651 Journal Entries have rationales in parquet)
- Rationale storage in [`logs/judge_labels/judge_labels.parquet`](../../logs/judge_labels/judge_labels.parquet)
- Rationale display UI in annotation tool ([`src/annotation_tool/components/modals.py`](../../src/annotation_tool/components/modals.py))
- LLM-Judge comparison view ([`src/annotation_tool/components/comparison_view.py`](../../src/annotation_tool/components/comparison_view.py))
- Weekly Coach prompt rendering plus programmatic narrative generation,
  validation, and persistence support in
  [`src/coach/weekly_digest.py`](../../src/coach/weekly_digest.py) and
  [`src/coach/runtime.py`](../../src/coach/runtime.py); the CLIs do not inject a
  live Weekly Coach LLM
- Tier 1 Weekly Coach narrative checks are implemented: groundedness via quoted substring matches, non-circularity via score-jargon avoidance, and length bounds via [`validate_weekly_digest_narrative()`](../../src/coach/weekly_digest.py)

### What's Missing
- **Tier 1 for LLM-Judge rationales:** No batch checker/report yet in `src/judge/`
- **Tier 1 reporting for Weekly Coach narratives:** Validation code exists, but there is no committed benchmark summary with pass rates across a Weekly Digest set
- **Tier 2:** Rationale-review LLM evaluation
- **Tier 3:** Human calibration protocol and κ calculation

### Blocking Dependencies
Tier 1 Weekly Coach checks are unblocked and implemented. Deeper end-to-end
explanation evaluation still depends on persisted decisions from the fixed
`gpt-5.6-luna` reasoning-effort-`low` Weekly Drift Reviewer, deterministic Drift
Detector output, and cited Journal Entry evidence (`twinkl-3sg`). VIF Critic
outputs belong to offline review and retraining.

### Implementation Scope

The implemented slice covers Weekly Digest construction, Weekly Coach prompt
rendering, programmatic narrative generation with an injected callable, and
Tier 1 narrative validation. The analogous batch checker for LLM-Judge rationales
remains planned, while Tier 2 (rationale-review LLM) and Tier 3 (human calibration) are
later validation phases.

### Next Steps
1. Add a batch Tier 1 checker for LLM-Judge rationales in `src/judge/` and run it over the existing 1,594 rationale-bearing rows
2. Run the existing Weekly Coach Tier 1 validation over a real Weekly Digest set and publish pass-rate summaries
3. Add provenance hooks from Weekly Drift Reviewer decisions and Drift Detector output to Weekly Coach evidence selection (`twinkl-3sg`)
4. *(Future phase)* Design a rationale-review LLM prompt for Tier 2 evaluation
5. *(Future phase)* Sample 20-30 explanations for Tier 3 human calibration

---

## Explanation Sources

### LLM-Judge Rationales

For each alignment score, the LLM-Judge provides a rationale:

```json
{
  "Achievement": "+1",
  "rationale": "Entry shows prioritizing work performance (finishing deck for investor meeting) over personal plans. Clear demonstration of achievement-oriented behavior."
}
```

**Criteria for good rationales:**
- References specific details from the Journal Entry
- Explains *why* the score was assigned
- Ties behavior to the value dimension

### Weekly Coach Narratives (Implemented, Experimental)

Weekly summaries that synthesize patterns:

```
"You wrote about cancelling on your friend after two weeks of saying you wanted
to make more room for the people close to you. What made this Saturday feel
different from the plan you had in mind?"
```

**Criteria for good narratives:**
- Cites specific evidence from Journal Entries
- Identifies patterns over time (not just one Journal Entry)
- Avoids prescriptive or judgmental language

The offline runtime path lives in `src/coach/weekly_digest.py` and
`src/coach/runtime.py`. The evaluation layer remains incomplete: Tier 1 checks
are implemented, while benchmark pass-rate reporting and user-study calibration
are pending.

---

## Evaluation Approach

### Primary: Likert Ratings (from PRD)

Show users their Weekly Digest and ask: **"Did this feel accurate?"**

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
| **Groundedness** | % of rationales with verifiable quotes (substring match in Journal Entry) | > 70% |
| **Non-circularity** | % that don't contain the value name itself | > 95% |
| **Length** | Flag too-short (<10 words) or too-long (>50 words) | 90% in range |

**Current code status:**
- Weekly Coach narratives: validated by `validate_weekly_digest_narrative()` inside [`src/coach/weekly_digest.py`](../../src/coach/weekly_digest.py)
- LLM-Judge rationales: still planned as a batch checker in `src/judge/`

**Reference implementation shape:**
```python
validation = validate_weekly_digest_narrative(digest, narrative)
results = {check.name: check.passed for check in validation.checks}
```

#### Tier 2: Rationale-Review LLM Evaluation

> **Implementation phase:** Future — not required for the initial Weekly Digest release.

For rationales that pass Tier 1, evaluate with LLM:

| Criterion | Question | Scale |
|-----------|----------|-------|
| **Correctness** | Does the rationale accurately reflect what happened in the Journal Entry? | 1-5 |
| **Specificity** | Does it reference concrete actions/statements, not vague generalities? | 1-5 |

**Rationale-review LLM prompt structure:**
- Input: Journal Entry text + LLM-Judge rationale + score
- Task: Rate correctness and specificity on 1-5 scale
- Output: Scores + brief justification

**Flag for human review if:**
- Rationale-review LLM correctness < 3
- Rationale-review LLM specificity < 3
- Rationale-review LLM expresses uncertainty

#### Tier 3: Human Calibration (Small Sample)

> **Implementation phase:** Future — designed for production validation.

Validate the rationale-review LLM against human judgment:

1. Randomly sample 20-30 rationales
2. Human rates same criteria (correctness, specificity)
3. Calculate agreement with the rationale-review LLM (Cohen's κ)
4. Target: κ > 0.6 (substantial agreement)

### For User Study (Manual)

1. **Sample size**: 5-10 users (from PRD)
2. **Duration**: 1-2 weeks of journaling
3. **Measurement points**:
   - After each Weekly Digest: "Did this feel accurate?" (5-point Likert)
   - Exit interview: Open-ended feedback on explanation quality

### Procedure

```
Day 1-7:     User journals normally
Day 7:       Weekly Coach generates Weekly Digest with explanations
             User rates: "Did this feel accurate?" [1-5]
Day 8-14:    Continue journaling
Day 14:      Second digest + rating
             Exit interview
```

---

## Evaluation Flow

```
LLM-Judge produces rationales for N Journal Entries
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
│  Tier 2: Rationale-Review LLM       │
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
│  - Compare to rationale-review LLM  │
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
| **Wrong attribution** | Confuses which value a behavior supports | Tier 2: rationale-review LLM correctness |
| **Over-inference** | Reads too much into a vague Journal Entry | Tier 2: rationale-review LLM correctness |

---

## Success Criteria

| Metric | Target | Tier | Phase | Rationale |
|--------|--------|------|-------|-----------|
| Groundedness (code) | > 70% | 1 | **Initial** | Rationales should quote or reference Journal Entry content |
| Non-circularity (code) | > 95% | 1 | **Initial** | Rationales shouldn't just restate value name |
| Length compliance | > 90% | 1 | **Initial** | Most rationales should be 10-50 words |
| Correctness (rationale-review LLM) | Mean > 3.5/5 | 2 | Future | Rationales should be factually accurate |
| Specificity (rationale-review LLM) | Mean > 3.5/5 | 2 | Future | Rationales should cite concrete details |
| Human-LLM agreement | κ > 0.6 | 3 | Future | The rationale-review LLM should align with human judgment |
| Mean Likert rating (users) | ≥ 3.5/5 | User study | Future | Above neutral = generally useful |
| % ratings ≥ 4 (users) | > 50% | User study | Future | Majority find it "mostly accurate" or better |

---

## Known Limitations

1. **Subjectivity**: "Felt accurate" is inherently subjective
2. **Small sample**: 5-10 users limits statistical power
3. **Hawthorne effect**: Users may rate higher knowing researchers will see

**Mitigations:**
- Use consistent Likert anchors with behavioral definitions
- Collect qualitative feedback to contextualize ratings
- Compare ratings across different explanation types (LLM-Judge vs. Weekly Coach)

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

- `docs/pipeline/judge_implementation_spec.md` — LLM-Judge rationale format
- `docs/prd.md` — Evaluation Strategy (Row 4: Explanation quality)
