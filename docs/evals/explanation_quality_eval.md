# Explanation Quality Evaluation

## What We're Evaluating

Twinkl generates explanations at two levels:
1. **LLM-Judge rationales**: Per-Journal-Entry explanations for alignment scores
2. **Coach Digest responses**: User-facing responses based on structured Weekly
   Drift Detection output

This evaluation validates that explanations feel accurate and actionable to users.

> **Runbook:** for the exact commands to run every Coach Digest response and
> Weekly Drift Detection output test and eval, see
> [`coach_narrative_test_and_eval_guide.md`](./coach_narrative_test_and_eval_guide.md).

---

## Implementation Status

**Status:** 🟡 Partial

### What's Implemented
- Rationale generation working (1,594/1,651 Journal Entries have rationales in parquet)
- Rationale storage in [`logs/judge_labels/judge_labels.parquet`](../../logs/judge_labels/judge_labels.parquet)
- Rationale display UI in annotation tool ([`src/annotation_tool/components/modals.py`](../../src/annotation_tool/components/modals.py))
- LLM-Judge comparison view ([`src/annotation_tool/components/comparison_view.py`](../../src/annotation_tool/components/comparison_view.py))
- Coach Digest prompt rendering plus programmatic response generation,
  validation, and persistence support in
  [`src/coach/weekly_digest.py`](../../src/coach/weekly_digest.py)
- The approved Weekly Drift Reviewer and Drift Detector runtime selects cited
  Journal Entry evidence for the Weekly Drift Detection output in
  [`src/coach/weekly_drift_runtime.py`](../../src/coach/weekly_drift_runtime.py)
- Automated Coach Digest response checks are implemented: groundedness via quoted substring matches, non-circularity via score-jargon avoidance, and length bounds via [`validate_weekly_digest_narrative()`](../../src/coach/weekly_digest.py)
- Automated reporting for Coach Digest responses: A batch runner
  ([`src/evals/coach_narrative_tier1.py`](../../src/evals/coach_narrative_tier1.py))
  runs `validate_weekly_digest_narrative()` over the persisted Weekly Drift Detection output
  parquet and reports per-check pass rates against targets. No current batch
  result is committed.
- AI review of Coach Digest responses: An AI evaluator
  ([`src/evals/coach_narrative_judge.py`](../../src/evals/coach_narrative_judge.py),
  prompt [`prompts/coach_narrative_judge.yaml`](../../prompts/coach_narrative_judge.yaml))
  scores correctness, specificity, non-prescriptive tone, and tension honesty
  against the same selected Coach Digest policy, Core Value phrases, goal
  context, Weekly Drift Detection findings, and cited Journal Entries used for
  Coach Digest generation.
  Scores are **AI evaluation, not human validation**. Human calibration remains
  future work.

### What's Missing
- **Coach Digest batch results:** No current automated-check or AI-review result is
  committed
- **Automated checks for LLM-Judge rationales:** No batch checker or report yet in `src/judge/`
- **AI review of LLM-Judge rationales:** No rationale-review evaluation
- **Human calibration:** No protocol or κ calculation for either explanation type

### Blocking Dependencies
Automated Coach Digest checks and approved-path evidence provenance are
implemented. Deeper end-to-end explanation evaluation still requires a fresh
final test and committed batch results. VIF Critic outputs belong to offline
review and retraining.

### Implementation Scope

The implemented slice covers Weekly Drift Detection output storage, Coach
Digest prompt rendering, programmatic response generation with an injected
callable, and automated response validation. The analogous batch checker for
LLM-Judge rationales remains planned. AI review of rationales and human
calibration are later validation phases.

### Next Steps
1. Add an automated batch checker for LLM-Judge rationales in `src/judge/` and run it over the existing 1,594 rationale-bearing rows
2. Run the automated Coach Digest batch evaluation and record its pass rates
3. Run the Coach Digest AI review and identify its scores as AI evaluation
4. *(Future phase)* Sample 20-30 explanations for human calibration

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

### Coach Digest Responses (Implemented, Experimental)

User responses that synthesize patterns:

```
"You wrote about cancelling on your friend after two weeks of saying you wanted
to make more room for the people close to you. What made this Saturday feel
different from the plan you had in mind?"
```

**Criteria for good responses:**
- Cites specific evidence from Journal Entries
- Identifies patterns over time (not just one Journal Entry)
- Avoids prescriptive or judgmental language

The approved path lives in `src/coach/weekly_drift_runtime.py` and
`src/coach/weekly_digest.py`. The evaluation layer remains incomplete. Automated
checks are implemented, while benchmark pass-rate reporting and user-study
calibration are pending.

---

## Evaluation Approach

### Primary: Likert Ratings (from PRD)

Show users their Coach Digest response and ask: **"Did this feel accurate?"**

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

#### Automated Code Checks (No LLM)

Fast, objective checks that don't require LLM calls:

| Check | Description | Target |
|-------|-------------|--------|
| **Groundedness** | % of rationales with verifiable quotes (substring match in Journal Entry) | > 70% |
| **Non-circularity** | % that don't contain the value name itself | > 95% |
| **Length** | Flag too-short (<25 words) or too-long (>180 words) | 90% in range |

**Current code status:**
- Coach Digest responses: validated by `validate_weekly_digest_narrative()` inside [`src/coach/weekly_digest.py`](../../src/coach/weekly_digest.py)
- LLM-Judge rationales: still planned as a batch checker in `src/judge/`

**Reference implementation shape:**
```python
validation = validate_weekly_digest_narrative(digest, narrative)
results = {check.name: check.passed for check in validation.checks}
```

#### AI Review of LLM-Judge Rationales

> **Implementation phase:** Future — not required for the initial Coach Digest
> response evaluation.

For rationales that pass the automated code checks, evaluate them with an LLM:

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

#### Human Calibration (Small Sample)

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
   - After each Coach Digest response: "Did this feel accurate?" (5-point Likert)
   - Exit interview: Open-ended feedback on explanation quality

### Procedure

```
Day 1-7:     User journals normally
Day 7:       Weekly Drift Detection stores structured output
             Coach Digest generates the user response
             User rates: "Did this feel accurate?" [1-5]
Day 8-14:    Continue journaling
Day 14:      Second Coach Digest response + rating
             Exit interview
```

---

## Evaluation Flow

```
LLM-Judge produces rationales for N Journal Entries
              ↓
┌─────────────────────────────────────┐
│  Automated Code Checks              │
│  - Groundedness (verifiable quotes) │
│  - Non-circularity (no value name)  │
│  - Length (25-180 words)            │
│  Output: Pass/Fail + metrics        │
└─────────────────────────────────────┘
              ↓
      (Passed code checks)
              ↓
┌─────────────────────────────────────┐
│  AI Review of Rationales            │
│  - Correctness (1-5)                │
│  - Specificity (1-5)                │
│  Output: Scores + flags for review  │
└─────────────────────────────────────┘
              ↓
       (Flagged or sampled)
              ↓
┌─────────────────────────────────────┐
│  Human Calibration                  │
│  - 20-30 rationales human-rated     │
│  - Compare to rationale-review LLM  │
│  Output: Cohen's κ agreement        │
└─────────────────────────────────────┘
```

---

## Failure Modes to Detect

| Failure Mode | Example | Detection Method |
|--------------|---------|------------------|
| **Hallucinated quotes** | "Entry mentioned 'staying late'" when it didn't | Automated groundedness check |
| **Generic explanation** | "Shows alignment with this value" | Automated length check and AI specificity review |
| **Circular reasoning** | "Achievement +1 because of achievement behavior" | Automated non-circularity check |
| **Wrong attribution** | Confuses which value a behavior supports | AI review of rationale correctness |
| **Over-inference** | Reads too much into a vague Journal Entry | AI review of rationale correctness |

---

## Success Criteria

| Metric | Target | Method | Phase | Rationale |
|--------|--------|--------|-------|-----------|
| Groundedness (code) | > 70% | Automated check | **Initial** | Rationales should quote or reference Journal Entry content |
| Non-circularity (code) | > 95% | Automated check | **Initial** | Rationales shouldn't just restate value name |
| Length compliance | > 90% | Automated check | **Initial** | Most narratives should be 25-180 words |
| Correctness (rationale-review LLM) | Mean > 3.5/5 | AI review | Future | Rationales should be factually accurate |
| Specificity (rationale-review LLM) | Mean > 3.5/5 | AI review | Future | Rationales should cite concrete details |
| Human-LLM agreement | κ > 0.6 | Human calibration | Future | The rationale-review LLM should align with human judgment |
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
- Compare ratings across different explanation types (LLM-Judge vs. Coach Digest)

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
