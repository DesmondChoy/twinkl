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
| Coach Digest Drift/control comparison | Study | **Yes (paid generation and AI review)** | `scripts/experiments/run_coach_drift_control_eval.py`, `src/evals/coach_drift_control_report.py` |

Coach Digest Validations are mechanical code checks, not human validation.
Coach Digest Evals produce **AI review, not human validation**. Future
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
              tests/evals/test_coach_narrative_judge.py \
              tests/evals/test_coach_drift_control_report.py \
              tests/experiments/test_run_coach_drift_control_eval.py
```

Lint and type-check touched code when you change it:

```sh
uv run ruff check \
  src/evals/coach_narrative_judge.py \
  src/evals/coach_drift_control_report.py \
  src/coach/llm_client.py src/coach/weekly_drift_runtime.py \
  scripts/experiments/run_coach_drift_control_eval.py \
  tests/coach/test_llm_client.py tests/coach/test_weekly_drift_runtime.py \
  tests/evals/test_coach_narrative_judge.py \
  tests/evals/test_coach_drift_control_report.py \
  tests/experiments/test_run_coach_drift_control_eval.py
uv run --with 'mypy==2.3.0' mypy --follow-imports=skip \
  src/evals/coach_narrative_judge.py \
  src/evals/coach_drift_control_report.py \
  src/coach/llm_client.py src/coach/weekly_drift_runtime.py \
  scripts/experiments/run_coach_drift_control_eval.py  # when type behavior changed
```

The import-isolated MyPy command supplies MyPy ephemerally and keeps this
focused check separate from known type errors in unrelated repository
dependencies.

---

## 2. Automated batch evaluation (no API calls)

Runs `validate_weekly_digest_narrative()` over the exact responses in the public
scenario sample manifest and reports per-check pass rates against the targets in
[`explanation_quality_eval.md`](./explanation_quality_eval.md).

```sh
uv run python -m src.evals.coach_digest_validations \
  --manifest logs/experiments/reports/coach_digest_sample_20260824/judge_sample_manifest.json \
  --out logs/experiments/reports/coach_digest_validations_20260824
```

- Use `--parquet` only for a separate persisted-output batch.
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

The evaluator must score the same responses that the React app shows. Read the
stored Weekly Drift Detection output from each scenario key week, generate the
five responses, rebuild the public bundles, and then build the manifest from
those bundles:

```sh
# Dry run — prints the plan, makes no calls:
.venv/bin/python scripts/coach/generate_approved_judge_sample.py \
  --personas 11de77e8 23d101f8 8f83c818 988d1a65 02fb94f3 \
  --reuse-scenario-key-weeks

# Real run — one paid Coach Digest call per Persona plus validation-guided
# retries. This command makes zero Weekly Drift Reviewer calls:
.venv/bin/python scripts/coach/generate_approved_judge_sample.py \
  --personas 11de77e8 23d101f8 8f83c818 988d1a65 02fb94f3 \
  --reuse-scenario-key-weeks --execute
```

The runner requires an explicit Persona roster. It has no default roster.

Each Coach Digest call writes a separate timestamped
`*.coach_diagnostic.json` file under the sample report directory. A new attempt
creates a new file. The file keeps raw rejected output and names the failed
parse, schema, or Coach Digest Validations stage. For OpenAI calls, it also
keeps token usage, calculated published-rate cost, request latency, and the
response ID. Rejected output does not enter the React fixtures or evaluation
manifest.

The Coach Digest provider comes from `TWINKL_COACH_PROVIDER` (`openai` or
`gemini`; default `openai`). Set the matching API key in `.env`
(`OPENAI_API_KEY` and/or `GEMINI_API_KEY`; Gemini also accepts
`GOOGLE_API_KEY`). OpenAI Coach Digest calls use
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

Use `--judge-provider openai` or `--judge-provider gemini` to select the
evaluator provider. Use `--judge-model` to select a model for that provider.
When the selected provider differs from `TWINKL_COACH_PROVIDER`, it uses its
provider default if `--judge-model` is absent. For example, this command
evaluates OpenAI-generated responses with Gemini:

```sh
uv run python -m src.evals.coach_narrative_judge \
  --manifest logs/experiments/reports/coach_digest_sample_20260824/judge_sample_manifest.json \
  --judge-provider gemini \
  --out logs/experiments/reports/coach_digest_evals_cross_provider \
  --execute
```

The output records both the generator model and evaluator model when the
manifest records the generator. It reports the same-model limit only when both
models match.

Targets: mean > 3.5/5 per dimension. Any response scoring below 3 on any
dimension is flagged for human review. Report and doc lines label these scores
as AI review, not human validation. The report keeps each response score,
evaluator justification, token usage, calculated published-rate cost, and
request latency.

---

## 4. Drift/control study

This study compares Coach Digest responses for known Drift records with matched
control targets. A control target comes from a Persona with no known Drift in
the complete development results. Matching first uses the same historical split
and Journal Entry count bucket (`<=6`, `7-9`, or `10-12`) and prefers the same
Core Value. If that pool is empty, the runner relaxes the historical split or
count bucket and records the result in `match_quality`. It selects the control
cutoff with the closest reviewed-week count. A control remains AI-reviewed
synthetic development evidence, not human ground truth.

First, build the deterministic target catalog. This command makes no provider
calls:

```sh
uv run python scripts/experiments/run_coach_drift_control_eval.py
```

Review `logs/experiments/reports/coach_digest_drift_control/targets.json`.
Then generate the missing Weekly Drift Detection and Coach Digest responses.
This command makes paid provider calls:

```sh
uv run python scripts/experiments/run_coach_drift_control_eval.py \
  --resume --execute
```

`--resume` keeps completed target IDs and does not repeat their paid calls. The
runner stops if generated outputs exist but the matching manifest is absent.
It also stops if a resumed run selects a different Coach Digest generator
model.

Product callers discard a response that fails Coach Digest Validations. The
study retains such a response through the evaluation-only
`attach_failed_validation=True` option so the comparison can measure the
failure. Evaluation code must check `digest.validation.all_passed` before it
treats the attached response as a valid Coach Digest response.

### Drift/control runner options

| Option | Default / behavior |
|---|---|
| `--episodes-parquet` | `logs/experiments/artifacts/twinkl_qtwz_complete_development_review_20260714/results/complete_development_drift_episodes.parquet` |
| `--case-outcomes-parquet` | `logs/experiments/artifacts/twinkl_qtwz_complete_development_review_20260714/results/complete_development_case_outcomes.parquet` |
| `--wrangled-dir` | `logs/wrangled` |
| `--parquet-path` | `logs/exports/weekly_digests/coach_digest_drift_control.parquet` |
| `--output-dir` | `logs/exports/weekly_drift_coach/drift_control` |
| `--manifest-out` | `logs/experiments/reports/coach_digest_drift_control/judge_sample_manifest.json` |
| `--targets-out` | `logs/experiments/reports/coach_digest_drift_control/targets.json` |
| `--group {drift,control,both}` | `both` |
| `--limit` | Unset; when present, limits the ordered Drift targets and their matched controls |
| `--seed` | `20260823` |
| `--resume` | Off; preserves compatible target and response records when enabled |
| `--execute` | Off; the default writes the deterministic target catalog without provider calls |

Run Coach Digest Evals with a provider that differs from the generator recorded
in the manifest. Use Gemini for OpenAI-generated responses, as shown below; use
OpenAI for Gemini-generated responses.

```sh
uv run python -m src.evals.coach_narrative_judge \
  --manifest logs/experiments/reports/coach_digest_drift_control/judge_sample_manifest.json \
  --judge-provider gemini \
  --out logs/experiments/reports/coach_digest_drift_control/evals \
  --execute
```

Build the comparison report. This command makes no provider calls:

```sh
uv run python -m src.evals.coach_drift_control_report \
  --manifest logs/experiments/reports/coach_digest_drift_control/judge_sample_manifest.json \
  --eval-metrics logs/experiments/reports/coach_digest_drift_control/evals/metrics.json \
  --out logs/experiments/reports/coach_digest_drift_control/comparison
```

The report compares pass rates from Coach Digest Validations and means from
Coach Digest Evals for Drift and control targets. It also reports the known Drift delivery
state and the input history for each target type. The known Drift records are
AI-reviewed synthetic development data. They are not human ground truth.

---

## Provenance and honesty notes

- Record the public bundle source, sample manifest, evaluator provider/model, and row/
  sample counts with every committed report.
- Record the Drift episode Parquet, case outcome Parquet, wrangled Journal Entry
  directory, source hashes, target seed, and target catalog for the
  Drift/control study.
- Do not treat AI review scores as human validation. State the source
  wherever it affects the conclusion.
- Build the deployed-Persona manifest from the rebuilt public scenario bundles.
  Do not evaluate a separate response copy.

---

## References

- [`explanation_quality_eval.md`](./explanation_quality_eval.md) — evaluation
  design and status
- [`overview.md`](./overview.md) — the user-facing evaluation path and separate
  VIF Critic (Offline) research path
- `src/coach/weekly_digest.py` — Weekly Drift Detection output builders,
  `validate_weekly_digest_narrative()`, and Coach Digest response generation
- `src/coach/weekly_drift_runtime.py` — Weekly Drift Detection and Coach Digest
  orchestration
