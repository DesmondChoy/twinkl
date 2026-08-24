# Coach Digest Response and Weekly Drift Detection — Test and Eval Guide

How to run every automated test and evaluation that covers the Weekly Drift Detection output and
the Coach Digest response (`weekly_mirror`, `tension_explanation`,
`reflective_question`). This is the operational companion to
[`explanation_quality_eval.md`](./explanation_quality_eval.md), which defines the
evaluation design; this file is the runbook.

All commands assume the virtual environment is active and you are at the repo
root:

```sh
source .venv/bin/activate        # Bash/Zsh
# source .venv/bin/activate.fish # Fish
```

---

## What covers what

| Layer | Kind | LLM calls? | Where |
| --- | --- | --- | --- |
| Weekly Drift Detection output builders + rendering | Unit tests | No (mocked) | `tests/coach/test_weekly_digest.py`, `tests/coach/test_runtime.py`, `tests/coach/test_weekly_drift_runtime.py` |
| Coach Digest Validations (groundedness, non_circularity, value_leakage, length) | Unit tests | No | `tests/coach/test_weekly_digest.py` |
| Coach Digest Validations batch report over a real Weekly Drift Detection output set | Eval | No | `src/evals/coach_digest_validations.py` |
| Coach Digest Evals (correctness, specificity, non-prescriptive tone, tension honesty) | Eval | **Yes (paid)** | `src/evals/coach_narrative_judge.py` |
| Drift against control comparison, split by evaluation arm | Eval | No | `src/evals/coach_drift_control_report.py` |
| Drift episode and control sample generation | Driver | **Yes (paid)** | `scripts/experiments/run_coach_drift_control_eval.py` |

Coach Digest Validations are mechanical code checks, not human validation.
Coach Digest Evals produce **AI evaluation, not human validation**. Future
human calibration of the AI review can use Cohen's κ.

---

## 1. Unit tests (no API calls)

These run offline; every LLM call is mocked with a fixture.

Run all Coach Digest and Weekly Drift Detection output unit tests:

```sh
uv run pytest tests/coach
```

Run only the Weekly Drift Detection output builder and automated response tests:

```sh
uv run pytest tests/coach/test_weekly_digest.py
```

Run only the automated response tests (groundedness, non_circularity,
value_leakage, length — both pass and fail paths):

```sh
uv run pytest tests/coach/test_weekly_digest.py -k validation
```

Run the evaluation unit tests for the batch report and AI review. Both use
mocked calls:

```sh
uv run pytest tests/evals/test_coach_digest_validations.py \
              tests/evals/test_coach_narrative_judge.py \
              tests/evals/test_coach_drift_control_report.py \
              tests/experiments/test_run_coach_drift_control_eval.py
```

Lint and type-check touched code when you change it:

```sh
uv run ruff check src/evals src/coach tests/coach tests/evals
uv run mypy src/evals            # when type behavior changed
```

---

## 2. Automated batch evaluation (no API calls)

Runs `validate_weekly_digest_narrative()` over every Coach Digest response in a persisted
Weekly Drift Detection output parquet and reports per-check pass rates against the targets in
[`explanation_quality_eval.md`](./explanation_quality_eval.md).

```sh
uv run python -m src.evals.coach_digest_validations \
  --parquet logs/exports/weekly_digests/weekly_digests.parquet \
  --out logs/experiments/reports/coach_digest_validations_20260727
```

- `--parquet` defaults to `logs/exports/weekly_digests/weekly_digests.parquet`.
- `--out` writes `metrics.json` and `report.md`; omit it to only print the
  summary.
- Rows with no response are skipped; unparseable responses are reported under
  `skipped_persona_weeks`.

Pass-rate targets: groundedness > 70%, non_circularity > 95%, length > 90%.
`value_leakage` has no published target and is reported for information.

---

## 3. Coach Digest Evals (paid API calls)

Scores Coach Digest responses on correctness, specificity, non-prescriptive tone, and
tension honesty, and flags whether the reflective question is open-ended. Judges
a fixed sample described by a manifest. **Makes paid calls and is gated behind
`--execute`.**

The evaluator uses the same factual contract as Coach Digest generation. This
contract includes the selected Coach Digest policy, user-facing Core Value
phrases, goal context, explicit Weekly Drift Detection findings, and cited
Journal Entries with dates, evidence roles, Core Value mappings, and excerpts.
It does not use the legacy
`top_tensions` field as a substitute for these facts.

### 3a. Build the sample (approved path)

The evaluator should score responses as the approved path actually produces them,
not leftover demo-tool outputs. Regenerate the fixed persona roster through
`run_weekly_drift_coach_cycle` and rebuild the manifest:

```sh
# Dry run — prints the plan, makes no calls:
.venv/bin/python scripts/coach/generate_approved_judge_sample.py
# OR
uv run python scripts/coach/generate_approved_judge_sample.py

# Real run — paid Weekly Drift Reviewer calls (one per week of history per
# persona) plus one Coach Digest response call per persona; overwrites their
# rows in the Weekly Drift Detection output parquet, then rebuilds the evaluation manifest:
.venv/bin/python scripts/coach/generate_approved_judge_sample.py --execute

# Smoke-test one persona first:
.venv/bin/python scripts/coach/generate_approved_judge_sample.py \
  --personas 7cc5cf92 --execute
```

The Coach Digest provider comes from `TWINKL_COACH_PROVIDER` (`openai` or
`gemini`; default `openai`). Set the matching API key in `.env`
(`OPENAI_API_KEY` and/or `GEMINI_API_KEY`). The Weekly Drift Reviewer uses
OpenAI.

### 3b. Run the AI review

```sh
# Dry run — prints the plan, makes no evaluator calls:
uv run python -m src.evals.coach_narrative_judge \
  --manifest logs/experiments/reports/coach_digest_validations_20260727/judge_sample_manifest.json

# Real run — paid evaluator calls; writes metrics.json, report.md, verdicts.json:
uv run python -m src.evals.coach_narrative_judge \
  --manifest logs/experiments/reports/coach_digest_validations_20260727/judge_sample_manifest.json \
  --out logs/experiments/reports/coach_digest_validations_20260727 \
  --judge-provider openai \
  --execute
```

Targets: mean > 3.5/5 per dimension. Any response scoring below 3 on any
dimension is flagged for human review. Report and doc lines label these scores
as AI evaluation, not human validation.

**Set `--judge-provider` to a provider other than the generator's.** Without it
the evaluator falls back to `TWINKL_COACH_PROVIDER`, the same setting that
selects the Coach Digest generator. One model then writes and scores the same
response, which raises the scores. The report records `judge_model` and
`generator_model`, and prints a self evaluation warning when they match.
`--judge-model` overrides the model id.

`verdicts.json` holds one record per response, keyed `persona_id:week_end`,
with each dimension score and the evaluator justification. Section 4 reads it,
and it is the only place to see why one response scored low.

---

## 4. Drift against control evaluation (paid API calls)

Sections 2 and 3 report one pooled number over a fixed Persona roster. That
roster has no Drift, so it measures the Weekly Drift Coach only on stable weeks.
This section covers the sample that includes Drift.

### 4a. Why the roster in section 3a is not enough

The five Personas in `DEFAULT_PERSONAS` (`7cc5cf92`, `0ad04582`, `20730018`,
`b7b942ab`, `61d7d490`) all have `has_drift = false` and `drift_count = 0` in
`complete_development_case_outcomes.parquet`. None appears in the 42 Drift
episodes. Use section 3a to reproduce the historical sample. Use this section to
measure Coach Digest behavior when Drift is present.

### 4b. Build the sample

The driver reads the Drift episodes and case outcomes, builds one target for
each Drift episode, and draws a matched control target for each one.

- **Drift arm.** One target per Drift episode, with `--end-date` set to the
  episode confirmation date. The runtime truncates history at that date and
  reports on the last week, so the confirmation week becomes the reported week.
- **Control arm.** Drawn only from Personas with no Drift episode in any Core
  Value. A Persona that drifts in another Core Value is not a clean control.
  Controls match on `historical_split` and history length bucket, then on
  reviewed week count. The control pool skews shorter than the Drift arm, so
  unmatched sampling would confound history length with Drift.

Target selection is a pure function of the two Parquet files, the wrangled
directory, and `--seed`. Always read `targets.json` before spending.

```sh
# Dry run — resolves and saves the targets, makes no calls:
uv run python scripts/experiments/run_coach_drift_control_eval.py --arm both

# Pilot one arm before the full run:
uv run python scripts/experiments/run_coach_drift_control_eval.py \
  --arm drift --limit 5 --execute

# Full run; --resume skips Persona weeks already in the Parquet:
uv run python scripts/experiments/run_coach_drift_control_eval.py \
  --arm drift --execute --resume
uv run python scripts/experiments/run_coach_drift_control_eval.py \
  --arm control --execute --resume
```

Output goes to a dedicated Parquet
(`logs/exports/weekly_digests/coach_drift_control_eval_20260823.parquet`), not
the shared `weekly_digests.parquet`. The shared file is overwritten by other
runs; an earlier evaluation lost its inputs that way.

The driver passes `attach_failed_validation=True` to
`run_weekly_drift_coach_cycle`. The runtime then records a Coach Narrative and
its validation verdict even when validation fails, because the failure rate is
the measurement. The default stays `False`, so the product paths in
`src.coach.runtime` and `src.demo.experience_service` keep dropping unvalidated
text. Check `digest.validation.all_passed` before you treat a recorded
narrative as deliverable.

### 4c. Compare the arms

Run section 2 and section 3b against the new Parquet and manifest, then:

```sh
uv run python -m src.evals.coach_drift_control_report \
  --manifest logs/experiments/reports/coach_narrative_drift_control_20260823/judge_sample_manifest.json \
  --verdicts logs/experiments/reports/coach_narrative_drift_control_20260823/tier2/verdicts.json \
  --out logs/experiments/reports/coach_narrative_drift_control_20260823/comparison
```

The report splits validation pass rates and evaluation scores by arm and by
`delivery_state`, and gives every rate a Wilson 95% confidence interval. At
about 42 responses per arm a difference of 15 points is not separable from
chance, so read the interval, not the rate alone.

Read three limits with every result:

1. The scores are mechanical checks and AI evaluation. Neither is human
   validation.
2. `tension_honesty` and `specificity` are expected to differ by arm. A control
   week holds no tension to describe, so a lower control score can be correct
   behavior.
3. Control weeks are weeks with **no detected Drift**, not weeks with no Drift.

### 4d. Pilot result, 2026-08-24

Ten Coach Digest responses: five Drift targets and five control targets.
Generator `gemini:gemini-3.5-flash`, evaluator `openai:gpt-5.4-mini`. The
evaluator differs from the generator, so these scores are not self evaluation.

Coach Digest Validations, Drift arm (five responses): groundedness 5/5,
non_circularity 5/5, value_leakage 5/5, length 4/5.

**Groundedness moved from 0% to 100%.** The earlier 0% came from digests with an
empty `evidence` list. `build_weekly_drift_reviewer_digest_from_entries` now
fills `evidence` from the last two window entries when no Conflict evidence
exists. The earlier responses predate that fallback.

Coach Digest Evals, all ten responses:

| Dimension | Mean | Target | Meets |
| --- | --- | --- | --- |
| correctness | 2.80 | ≥ 3.5 | No |
| specificity | 4.40 | ≥ 3.5 | Yes |
| non_prescriptive_tone | 4.60 | ≥ 3.5 | Yes |
| tension_honesty | 1.90 | ≥ 3.5 | No |

Seven of ten responses are flagged for human review.

Two findings need work before the full run:

- **`state_claims` splits by arm: Drift 5/5, control 2/5.** Control responses
  make Drift state claims the Weekly Drift Detection output does not support.
  This is the first signal the two arms have produced. Confirm the cause before
  spending on the remaining targets.
- **`tension_honesty` is 1.90.** A low control score (1.60) is partly expected,
  because a control week holds no tension. The Drift arm score (2.20) is not
  explained that way. Four of the five Drift responses come from `recovered`
  episodes and report `no_active_drift`, so they resemble controls. Only one
  response covered an `active` episode.

Sample shape to keep in mind: 32 of the 42 Drift episodes are `recovered` and
only 10 are `active`. A `recovered` episode produces a `no_active_drift`
response that reads much like a control. A two-way split therefore compares
mostly similar responses. Consider a three-way split, `active` against
`recovered` against control, and state the `active` count on the face of the
table.

---

## Provenance and honesty notes

- Record the parquet source, sample manifest, evaluator provider/model, and row/
  sample counts with every committed report.
- Do not treat AI evaluation scores as human validation. State the source
  wherever it affects the conclusion.
- Record the evaluator model and the generator model together. When one model
  writes and scores the same response, say so; the scores are too high.
- For a Drift against control run, commit `--seed` and `targets.json` with the
  report. A different seed draws a different control sample.
- A Coach Digest Validations pass rate taken from responses generated with the
  default attach gate is near 100% by construction, because a failed response is
  dropped before it reaches the Parquet. Compare such a rate only against
  another run with the same gate setting.
- The Weekly Drift Detection output parquet under `logs/exports/weekly_digests/` is a local,
  gitignored artifact; regenerate it via the approved path rather than assuming
  its rows reflect current behavior.

---

## References

- [`explanation_quality_eval.md`](./explanation_quality_eval.md) — evaluation
  design and status
- [`overview.md`](./overview.md) — where explanation quality sits in the VIF
  evaluation flow
- `src/coach/weekly_digest.py` — Weekly Drift Detection output builders,
  `validate_weekly_digest_narrative()`, and Coach Digest response generation
- `src/coach/weekly_drift_runtime.py` — Weekly Drift Detection and Coach Digest
  orchestration, and the `attach_failed_validation` gate
- `src/coach/llm_client.py` — provider adapters and `resolve_coach_model()`,
  which reports the `provider:model` id for a report
- `src/evals/coach_drift_control_report.py` — Drift against control comparison
- `scripts/experiments/run_coach_drift_control_eval.py` — Drift episode and
  control sample driver
