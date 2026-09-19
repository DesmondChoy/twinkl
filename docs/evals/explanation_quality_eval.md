# Explanation Quality Evaluation

## What We're Evaluating

Twinkl generates explanations at two levels:
1. **LLM-Judge rationales**: Per-Journal-Entry explanations for LLM-Judge VIF Labels
2. **Coach Digest responses**: User-facing responses based on structured Weekly
   Drift Detection output

This evaluation defines the evidence, language, and AI-review checks for those
two outputs. Human calibration and a user pilot are outside the capstone scope.

> **Runbook:** for the exact commands to run every Coach Digest response and
> Weekly Drift Detection output test and eval, see
> [`coach_narrative_test_and_eval_guide.md`](./coach_narrative_test_and_eval_guide.md).

---

## Implementation Status

**Status:** Done for the accepted Coach Digest evaluation workflow. Independent
provider selection and the 42-Drift/42-control study are complete capstone
deliverables. Human calibration and the user pilot are not done and closed
outside the capstone. The separate rationale checks and factual-attribution
follow-up remain not done.

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
- Coach Digest Validations are implemented: an exact evidence quotation in
  `weekly_mirror`, source matching for every quotation, nonempty fields,
  question form and a single generated question across the complete response,
  non-circularity via score-jargon avoidance, raw Schwartz value
  leakage, unsupported current-state claims, and a 180-word maximum via
  [`validate_weekly_digest_narrative()`](../../src/coach/weekly_digest.py)
  for ordinary Coach Digest prompt `4.7`. Verified source quotations are excluded
  from generated-question counts and unsupported-state-claim checks. There is no
  minimum word count.
  Historical receipts retain their recorded validation rules, including the
  original 25–180-word bounds.
- Coach Digest Validations batch reporting: A batch runner
  ([`src/evals/coach_digest_validations.py`](../../src/evals/coach_digest_validations.py))
  runs `validate_weekly_digest_narrative()` over the exact public-scenario
  sample manifest or the persisted Weekly Drift Detection output parquet and
  reports per-check pass rates against targets.
  Its default `--validation-policy recorded` reproduces saved validation checks
  or the recorded generation prompt's policy. `--validation-policy current`
  applies the current live checks in a separate assessment; neither mode
  rewrites the saved response or its validation receipt.
- Coach Digest Evals: An AI evaluator
  ([`src/evals/coach_narrative_judge.py`](../../src/evals/coach_narrative_judge.py),
  prompt [`prompts/coach_narrative_judge.yaml`](../../prompts/coach_narrative_judge.yaml))
  scores correctness, specificity, non-prescriptive tone, and tension honesty
  against the same selected Coach Digest policy, Core Value phrases, goal
  context, Weekly Drift Detection findings, and cited Journal Entries used for
  Coach Digest generation.
  Scores are **AI review, not human validation**. Human calibration is not done
  and is outside the capstone scope.
- Coach Digest Evals can select an OpenAI or Gemini evaluator independently
  from the Coach Digest generator through `--judge-provider` and
  `--judge-model`. Reports record both model identities and apply the
  same-model-review limitation only when they match.
- The deterministic Drift/control runner selects one target for each known
  development Drift and one matched control. The comparison report includes
  pass rates from Coach Digest Validations with Wilson intervals, means from
  Coach Digest Evals,
  known Drift state, input-history summaries, response modes, and match quality.
- Product callers discard responses that fail Coach Digest Validations. The
  Drift/control runner uses the evaluation-only
  `attach_failed_validation=True` option so the study can count those failures;
  evaluation code must check `digest.validation.all_passed` before treating an
  attached response as valid.

### Current Result Status

The saved Persona replay has 27 baseline responses from the
[9 September voice refresh and editorial repair](../../logs/experiments/reports/coach_voice_refresh_20260909/report.md),
using prompt `4.4`. The 22 weeks with accepted North Star Moment selections have
[paired Coach Digest responses](../north_star/demo_coach_comparison.md): 21 pairs
use `4.4`/`1.0`, while Lukas's week beginning 16 June uses `4.5`/`1.1`.
The five other weeks display their baseline response. Each receipt retains its
generation version and matching validation policy. These responses have AI
editorial review and code checks, without fresh Coach Digest Evals scores or
human validation.

The [13 September context comparison](../../logs/experiments/reports/coach_context_eval_20260913/report.md)
contains six fresh generations and nine fresh evaluator 3.1 assessments of one
synthetic Wei Jun week. Mean correctness was 4.33 with short generation excerpts
and 5.00 with complete generation context; both arms were judged against the
same complete sources. These scores failed the factual controls: the evaluator
gave the known incorrect response 5 with short evidence and 4 with complete
evidence, triggering no review flag. Separate AI source review found fewer
clear event/dialogue errors with complete context, but continuing unsupported
causal generalization. This small same-model development assessment is not a
clean factual pass, human validation, or a replacement for earlier receipts.

The [September replay refresh](../../logs/experiments/reports/demo_v4_run1_20260907/report.md)
contains five accepted responses for the preceding v4 Run 1 Persona roster's key weeks.
All passed Coach Digest Validations. Luna at reasoning effort `none`, using
prompt `4.2`, made seven generation calls, including retries for Noor and Wei
Jun. No new Coach Digest Evals or human review was performed, so these
responses have no new semantic-quality scores.

The [8 September completion](../../logs/experiments/reports/demo_coach_all_weeks_20260908/report.md)
preserves those five responses and 22 further responses for that roster.
All pass their recorded Coach Digest Validations. A selected
North Star Moment passage appears within a valid Coach Digest after its
question, preserving both components' raw output. The [integrated validation report](../../logs/experiments/reports/integrated_coach_validation_20260908/report.md)
records application checks and five unresolved AI editorial findings, including
a contradiction between the Coach's selected excerpts and the fuller NSM source.
These observations are not new Coach Digest Evals scores or human validation;
`twinkl-rklc.39` tracks the semantic follow-up.

The [replacement sample](../../logs/experiments/reports/coach_digest_sample_20260824/report.md)
preserves one key-week response for each Persona deployed in August. Its
evaluation manifest records the exact responses from those scenario bundles.
The current v4 Run 1 replay uses a revised Persona roster and accepts saved
Coach Digest responses only when their input hashes match the current Weekly
Drift Detection output. Historical scores do not transfer to new responses.
All five responses passed groundedness, non-circularity, raw value leakage,
current-state claims, and length checks. The [Coach Digest Evals](../../logs/experiments/reports/coach_digest_evals_20260824/report.md)
scored mean correctness `4.80`, specificity `5.00`, non-prescriptive tone
`5.00`, and tension honesty `4.60`. All reflective questions passed. No
evaluator call failed, and no response had a review flag.

Generation used seven Luna-none calls because Meera and Noor each needed one
validation-guided retry. The rejected raw outputs and API diagnostics are
preserved. The final AI review used five Luna-none calls. Across generation and
evaluation, the 12 calls used 16,547 input tokens and 1,696 output tokens. The
calculated published-rate cost was `$0.00607555`, and total request latency was
`33.707` seconds. This cost is not a billing export.

This committed result uses Luna-none for both generation and AI review. The
independent-provider options and Drift/control workflow have no committed paid
result. Their accepted capstone scope is the implemented evaluation workflow;
a paid independent study result is not required for closeout.

### Completion and Evidence Limits

| Item | Status | Evidence and scope |
|---|---|---|
| Coach Digest Validations and Coach Digest Evals | Done | Validation, AI scoring, batch reports, and saved historical results are implemented. |
| Independent-provider Coach Digest evaluation | Done | Provider and model selection, identity reporting, and the reproduction commands are implemented. No paid independent-provider result is committed or required for capstone closeout. |
| 42-Drift/42-control study | Done | Deterministic selection, generation, safe resume, and grouped reporting are implemented. No paid study result is committed or required for capstone closeout. |
| Automated checks for LLM-Judge rationales | Not done | No batch checker or report in `src/judge/`. |
| AI review of LLM-Judge rationales | Not done | No rationale-review evaluation. |
| Coach factual-attribution evaluation | Not done | `twinkl-z5nr` tracks source-event and causal-claim assessment; `twinkl-rklc.39` tracks the saved-output semantic follow-up. |
| Human calibration | Not done — closed | No human calibration result for either explanation type; outside capstone scope. |
| User pilot | Not done — closed | No real-user explanation-quality study; outside capstone scope. |

### Capstone Acceptance

Coach Digest Validations, Coach Digest Evals, the historical five-response replacement,
approved-path evidence provenance, independent-provider evaluation, and the
42-Drift/42-control workflow are complete within the accepted scope. The paid
independent study, human calibration, and user pilot are not closeout
requirements. Neither the historical five-response sample nor the 27-week
synthetic replay establishes product usefulness or human validity. VIF Critic
Predictions belong to offline research.

### Implementation Scope

The implemented slice covers Weekly Drift Detection output storage, Coach
Digest prompt rendering, programmatic response generation, automated response
validation, batch reporting, and Coach Digest Evals. The analogous batch
checker for LLM-Judge rationales remains planned. Coach Digest Evals support
provider separation, and the Drift/control workflow supports deterministic
selection, safe resume, evaluation-only failure measurement, and grouped
reporting. AI review of rationales is unimplemented. Human calibration and the
user pilot are closed outside the capstone scope.

### Remaining Implementation Work

1. Add an automated batch checker for LLM-Judge rationales in `src/judge/` and run it over the existing 1,594 rationale-bearing rows
2. Address the factual-attribution and causal-claim evaluation under
   `twinkl-z5nr` and the saved-output semantic findings under `twinkl-rklc.39`.

---

## Explanation Sources

### LLM-Judge Rationales

For each LLM-Judge VIF Label, the LLM-Judge provides a rationale:

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
`src/coach/weekly_digest.py`. The five historical August Persona responses
completed Coach Digest Validations and same-model Coach Digest Evals.
Independent-provider AI review and the 42-Drift/42-control study are done within
their accepted workflow scope, with no committed paid result. Human calibration
and the user pilot are not done and closed outside the capstone.

---

## Evaluation Approach

### User Study Design (Not Done — Closed)

The retained design is outside the capstone scope and has no collected user
ratings. Its proposed question is: **"Did this feel accurate?"**

| Rating | Meaning |
|--------|---------|
| 5 | Completely accurate — captures exactly what happened |
| 4 | Mostly accurate — minor misses but right overall |
| 3 | Somewhat accurate — got some things right |
| 2 | Mostly inaccurate — misses important context |
| 1 | Completely inaccurate — doesn't reflect my week |

### Coach Digest Evals

Coach Digest Evals use the exact structured Weekly Drift Detection output and
response to score four dimensions. The evaluator also decides whether the
reflective question is open and relevant.

| Criterion | Question | Scale |
|-----------|----------|-------|
| **Correctness** | Does the explanation accurately reflect what happened? | 1-5 |
| **Specificity** | Does it reference concrete details, not vague generalities? | 1-5 |
| **Non-prescriptive tone** | Does the response avoid commands, moral judgment, and diagnosis? | 1-5 |
| **Tension honesty** | Does the response preserve ambiguity and avoid unsupported progress claims? | 1-5 |

The target is a mean above `3.5` for each dimension. Any score below `3` is a
review flag. These scores are AI review, not human validation.

Evaluator prompt `3.1` accepts a grounded, open-ended question as a way to
preserve uncertainty under `more_reflection_needed`. An explicit statement of
the evidence limit is optional. The response must still leave Drift undecided
and avoid inventing decisions, motives, or outcomes. This matches ordinary
Coach Digest prompt `4.7`; historical evaluation receipts retain their rubric.

The source-context regression compares the same Wei Jun week using its saved
short excerpts and complete text for the same selected Journal Entries. Archived
Coach prompt `4.6`, evaluator prompt `3.1`, model settings, and Drift decisions
remain fixed. Both sets of
fresh responses are scored against the same complete-context input, so the
comparison measures factual accuracy against a common source. Separate controls
score the recorded event misattribution with short and complete context, and a
constructed accurate response with complete context. Expected control outcomes
are kept out of evaluator prompts. This focused comparison is a development
diagnostic; repeated samples of one week do not establish population accuracy.
See the [context comparison runbook](coach_narrative_test_and_eval_guide.md#source-context-comparison).

### Provider Separation and Drift/Control Comparison

**Status: Done** for the accepted capstone workflow. The commands remain
available for reproduction. No paid independent-provider or Drift/control
result is committed, and neither is a capstone closeout requirement.

`src.evals.coach_narrative_judge` accepts `--judge-provider {openai,gemini}`
and `--judge-model`. When the evaluator provider differs from the generator
provider, the report records that separation. Provider separation reduces one
known correlated-review risk, but it does not create human validation.

`scripts/experiments/run_coach_drift_control_eval.py` builds a deterministic
catalog of 42 known development Drifts and 42 matched controls from the current
committed inputs. A control has no known Drift for its Persona. It is not human
ground truth. The runner's default mode writes only the target catalog. Paid
Weekly Drift Detection and Coach Digest generation require `--execute`.

---

## Evaluation Protocol

### For Synthetic Data (Automated)

#### Automated Code Checks (No LLM)

Fast, objective checks that don't require LLM calls:

| Check | Description | Target |
|-------|-------------|--------|
| **Groundedness** | % of Coach Digest responses with an exact evidence quotation in `weekly_mirror` and source matches for every quotation | > 70% |
| **Non-circularity** | % of Coach Digest responses that avoid score and alignment jargon | > 95% |
| **Raw value leakage** | Response does not expose raw Schwartz value labels | Reported |
| **Current-state claims** | Response does not make an unsupported positive-change claim | Reported |
| **Generated question** | Exactly one generated question across all three fields, in question form in `reflective_question`; verified source quotations do not count as generated questions | Reported for prompt `4.7` |
| **Conversational voice** | Responses avoid calendar-recap openings and the narrow clinical finding phrases checked by code; quoted user wording is preserved | Reported from prompt `4.4` |
| **Natural reflection voice** | Responses avoid date-led openings and commentary about the writing process | Reported from prompt `4.5` |
| **Length** | Ordinary Coach Digest response has nonempty fields and at most 180 words, with no minimum word count | > 90% |

**Current code status:**
- Coach Digest responses: validated by `validate_weekly_digest_narrative()` inside [`src/coach/weekly_digest.py`](../../src/coach/weekly_digest.py)
- Ordinary generation and live display enable conversational-voice checks. Historical
  responses retain their original validation rules. These wording checks do not
  establish warmth, semantic correctness, or human benefit; the
  [9 September refresh](../../logs/experiments/reports/coach_voice_refresh_20260909/report.md)
  records the associated generation and separate AI editorial review.
- LLM-Judge rationales: still planned as a batch checker in `src/judge/`

**Reference implementation shape:**
```python
validation = validate_weekly_digest_narrative(digest, narrative)
results = {check.name: check.passed for check in validation.checks}
```

#### AI Review of LLM-Judge Rationales

> **Status:** Not done — separate from the accepted Coach Digest evaluation
> workflow.

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

#### Human Calibration (Not Done — Closed)

Human calibration is outside the capstone scope. The retained design below
does not represent collected ratings or a completed agreement calculation.

The proposed design compares the rationale-review LLM with human judgment:

1. Randomly sample 20-30 rationales
2. Human rates same criteria (correctness, specificity)
3. Calculate agreement with the rationale-review LLM (Cohen's κ)
4. Target: κ > 0.6 (substantial agreement)

### User Pilot Design (Not Done — Closed)

The user pilot is outside the capstone scope. These parameters and the procedure
below document an unexecuted design, not a remaining capstone requirement.

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

## LLM-Judge Rationale Evaluation Design

The flow below applies only to the unimplemented LLM-Judge rationale review. It
does not describe the implemented Coach Digest Validations or Coach Digest
Evals. Its human-calibration stage is closed outside the capstone scope.

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
│  Human Calibration (closed scope)    │
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
| **Generic explanation** | "Shows alignment with this value" | Automated jargon check and AI specificity review |
| **Circular reasoning** | "Achievement +1 because of achievement behavior" | Automated non-circularity check |
| **Wrong attribution** | Confuses which value a behavior supports | AI review of rationale correctness |
| **Over-inference** | Reads too much into a vague Journal Entry | AI review of rationale correctness |

---

## Success Criteria

| Metric | Target | Method | Phase | Rationale |
|--------|--------|--------|-------|-----------|
| Groundedness (code) | > 70% | Coach Digest Validations | **Current** | Responses should use quoted evidence from cited Journal Entries |
| Non-circularity (code) | > 95% | Coach Digest Validations | **Current** | Responses should avoid score and alignment jargon |
| Length compliance | > 90% | Coach Digest Validations | **Current** | Responses should have nonempty fields and at most 180 words; there is no minimum word count |
| Correctness, specificity, non-prescriptive tone, and tension honesty | Mean > 3.5/5 for each dimension | Coach Digest Evals | **Current** | Measures the response contract; remains AI review rather than human validation |
| Coach Digest Evals review flag | Any dimension < 3 | Coach Digest Evals | **Current** | Marks a low-scoring response for review; does not imply that human review occurred |
| Correctness (rationale-review LLM) | Mean > 3.5/5 | AI review | Not done | Rationales should be factually accurate |
| Specificity (rationale-review LLM) | Mean > 3.5/5 | AI review | Not done | Rationales should cite concrete details |
| Human-LLM agreement | κ > 0.6 | Human calibration | Not done — closed | Unmeasured; outside capstone scope |
| Mean Likert rating (users) | ≥ 3.5/5 | User study | Not done — closed | Unmeasured; outside capstone scope |
| % ratings ≥ 4 (users) | > 50% | User study | Not done — closed | Unmeasured; outside capstone scope |

---

## Known Limitations

1. **No human calibration or user pilot**: Both are not done and closed outside
   the capstone. No user ratings or human-agreement result is claimed.
2. **Study-design limits**: The unexecuted 5–10-user design would have limited
   statistical power. Perceived accuracy is subjective, and ratings could be
   influenced by participants knowing that researchers would see them.
3. **Same-model committed result**: Luna-none generated and evaluated the five
   historical August responses. Correlated errors can make those scores too
   favorable. The evaluator supports provider separation, but no paid cross-provider result is
   committed or required for capstone closeout.
4. **Synthetic sample**: The historical evaluation covers five selected
   responses. The current 27-week replay has code validation and qualitative
   AI editorial review, with five unresolved findings. Neither establishes a
   fresh final test, human validity, or user usefulness.
5. **Drift/control reference source**: Known Drifts and no-known-Drift controls
   come from AI-reviewed synthetic development evidence, not human ground truth.

**Considerations for the unexecuted user-study design:**

- Use consistent Likert anchors with behavioral definitions
- Collect qualitative feedback to contextualize ratings
- Compare ratings across different explanation types (LLM-Judge vs. Coach Digest)

---

## Illustrative User-Study Output (Not Collected)

The table and aggregates below are illustrative only. The closed user pilot
has no collected results.

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
