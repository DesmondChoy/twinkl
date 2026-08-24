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
| Coach Digest Validations (groundedness, non-circularity, raw value leakage, current-state claims, length) | Unit tests | No | `tests/coach/test_weekly_digest.py` |
| Coach Digest Validations batch report over a real Weekly Drift Detection output set | Eval | No | `src/evals/coach_digest_validations.py` |
| Coach Digest Evals (correctness, specificity, non-prescriptive tone, tension honesty) | Eval | **Yes (paid)** | `src/evals/coach_narrative_judge.py` |

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

Run only the automated response tests (groundedness, non-circularity, raw value
leakage, current-state claims, and length, with pass and fail paths):

```sh
uv run pytest tests/coach/test_weekly_digest.py -k validation
```

Run the evaluation unit tests for the batch report and AI review. Both use
mocked calls:

```sh
uv run pytest tests/evals/test_coach_digest_validations.py \
              tests/evals/test_coach_narrative_judge.py
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
  --out logs/experiments/reports/coach_digest_validations_20260824
```

- `--parquet` defaults to `logs/exports/weekly_digests/weekly_digests.parquet`.
- `--out` writes `metrics.json` and `report.md`; omit it to only print the
  summary.
- Rows with no response are skipped; unparseable responses are reported under
  `skipped_persona_weeks`.

Pass-rate targets: groundedness > 70%, non-circularity > 95%, length > 90%.
Raw value leakage and current-state claims have no published targets. The
report includes them for information.

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
not leftover demo-tool outputs. Regenerate an explicit Persona roster through
`run_weekly_drift_coach_cycle` and rebuild the manifest:

```sh
# Dry run — prints the plan, makes no calls:
.venv/bin/python scripts/coach/generate_approved_judge_sample.py \
  --personas 11de77e8
# OR
uv run python scripts/coach/generate_approved_judge_sample.py \
  --personas 11de77e8

# Real run — paid Weekly Drift Reviewer calls (one per week of history per
# persona) plus one Coach Digest response call per persona; overwrites their
# rows in the Weekly Drift Detection output parquet, then rebuilds the evaluation manifest:
.venv/bin/python scripts/coach/generate_approved_judge_sample.py \
  --personas 11de77e8 --execute

# Smoke-test one persona first:
.venv/bin/python scripts/coach/generate_approved_judge_sample.py \
  --personas 11de77e8 --execute

# Rebuild only Coach Digest responses from stored Weekly Drift Detection output:
.venv/bin/python scripts/coach/generate_approved_judge_sample.py \
  --personas 11de77e8 --reuse-weekly-drift-output --execute
```

The runner requires an explicit Persona roster. It has no default roster.

Each Coach Digest call writes a separate timestamped
`*.coach_diagnostic.json` file beside the Weekly Drift Detection output. A new
attempt creates a new file. The file keeps raw rejected output and names the
failed parse, schema, or Coach Digest Validation stage. For OpenAI calls, it
also keeps token usage, calculated published-rate cost, and request latency.
Rejected output does not enter the evaluation manifest.

The Coach Digest provider comes from `TWINKL_COACH_PROVIDER` (`openai` or
`gemini`; default `openai`). Set the matching API key in `.env`
(`OPENAI_API_KEY` and/or `GEMINI_API_KEY`). OpenAI Coach Digest calls use
`gpt-5.6-luna` at reasoning effort `none` by default. The Weekly Drift Reviewer
uses `gpt-5.6-luna` at reasoning effort `low`.

### 3b. Run the AI review

```sh
# Dry run — prints the plan, makes no evaluator calls:
uv run python -m src.evals.coach_narrative_judge \
  --manifest logs/experiments/reports/coach_digest_sample_20260824/judge_sample_manifest.json

# Real run — paid evaluator calls; writes metrics.json and report.md:
uv run python -m src.evals.coach_narrative_judge \
  --manifest logs/experiments/reports/coach_digest_sample_20260824/judge_sample_manifest.json \
  --out logs/experiments/reports/coach_digest_evals_20260824 \
  --execute
```

Targets: mean > 3.5/5 per dimension. Any response scoring below 3 on any
dimension is flagged for human review. Report and doc lines label these scores
as AI evaluation, not human validation. The report keeps each response score,
evaluator justification, token usage, calculated published-rate cost, and
request latency.

---

## Provenance and honesty notes

- Record the parquet source, sample manifest, evaluator provider/model, and row/
  sample counts with every committed report.
- Do not treat AI evaluation scores as human validation. State the source
  wherever it affects the conclusion.
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
  orchestration
