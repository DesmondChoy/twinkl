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
The [September manifest](../../logs/experiments/reports/demo_v4_run1_20260907/judge_sample_manifest.json)
preserves the five accepted 7 September key-week responses from the previous
Persona roster.
The [8 September completion](../../logs/experiments/reports/demo_coach_all_weeks_20260908/report.md)
added 22 validated responses for that roster. The
[Persona replacement run](../../logs/experiments/reports/demo_persona_replacement_20260908/report.md)
retains 17 compatible responses and adds ten for Lukas and Meera, covering all
27 current replay weeks. The command below checks only the historical
five-response sample; it is not a report of the current roster or all weeks.
Section 3a documents missing-week completion. The August manifest remains
available for reproducing the historical sample.

```sh
uv run python -m src.evals.coach_digest_validations \
  --manifest logs/experiments/reports/demo_v4_run1_20260907/judge_sample_manifest.json \
  --out logs/experiments/reports/demo_v4_run1_20260907/validations
```

- Use `--parquet` only for a separate persisted-output batch.
- `--out` writes `metrics.json` and `report.md`; omit it to only print the
  summary.
- Rows with no response are skipped; unparseable responses are reported under
  `skipped_persona_weeks`.

Verify all 27 exported responses, current input hashes, exact generation-source
event IDs, and preservation of compatible retained receipts without provider calls:

```sh
uv run pytest tests/demo/test_scenarios.py
```

The [integrated validation report](../../logs/experiments/reports/integrated_coach_validation_20260908/report.md)
records application checks and five unresolved AI editorial findings under
`twinkl-rklc.39`. A selected NSM passage now appears within a valid Coach Digest,
before its original question. Code checks do not resolve the reported semantic
issues and are not human validation or new Coach Digest Evals scores.

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

### 3a. Complete saved replay weeks and retain compatible responses

All 27 saved replay weeks now contain a validated Coach Digest. The completion
runner reads each week's exact saved `weekly_digest_built` input, preserves
compatible existing responses, and generates only missing weeks. It uses Luna
at reasoning effort `none`, prompt `4.2`, no SDK automatic retries, and at most
one validation-guided retry per case. It makes no Weekly Drift Reviewer or NSM
calls. `OPENAI_API_KEY` is read through the existing environment setup.

```sh
# Prepare or verify the saved plan; no provider calls:
uv run python -m scripts.coach.complete_scenario_coach

# Complete missing responses within authorized paid scope:
uv run python -m scripts.coach.complete_scenario_coach --execute

# Rebuild the public scenario bundles without provider calls:
uv run python -m scripts.export_demo_experiments
```

`--output` selects the checkpoint/report directory; the default is
`logs/experiments/reports/demo_persona_replacement_20260908/current`. Its saved
plan verifies all 27 current responses, with none missing. When the roster,
inputs, or generation policy change, select a new `--output` directory to
preserve earlier frozen plans. The historical all-week completion added 22
responses with 26 calls; the Persona replacement run retained 17 responses
exactly and added ten with 12 calls. Each generation attempt preserves its
input, complete prompt, raw response, usage, and validation result.
Completed checkpoints resume without new calls. Unknown interrupted attempts
stop for inspection; terminal failures do not gain retries. Existing responses
with changed inputs are rejected instead of silently overwritten. Ordinary
replay and verification need no generation.

The [7 September generator and report](../../logs/experiments/reports/demo_v4_run1_20260907/README.md)
document the historical five-key-week sample and its seven Luna-none calls.
`generate_approved_judge_sample.py --reuse-scenario-key-weeks` targets only
the current five key weeks and replaces its response fixture; do not use its default
fixture path to maintain the completed 27-week replay. The preserved September
manifest remains a historical five-response sample for the evaluator below. The August
manifest and its AI scores remain historical, separate evidence.

The completion runner does not create an all-week Coach Digest Evals manifest
or perform semantic judging. A future all-week AI evaluation must first build
a source-bound manifest of the exact displayed responses. Historical scores
must not be assigned to newly generated responses. The NSM live allowance is separate
from Coach generation: its pinned US$1 budget is shared across sessions and
restarts and is not spent by offline scenario export.

### 3b. Run the AI review

```sh
# Dry run — prints the plan, makes no evaluator calls:
uv run python -m src.evals.coach_narrative_judge \
  --manifest logs/experiments/reports/demo_v4_run1_20260907/judge_sample_manifest.json

# Real run — paid evaluator calls; writes metrics.json and report.md:
uv run python -m src.evals.coach_narrative_judge \
  --manifest logs/experiments/reports/demo_v4_run1_20260907/judge_sample_manifest.json \
  --out logs/experiments/reports/demo_v4_run1_20260907/evals \
  --execute
```

Use `--judge-provider openai` or `--judge-provider gemini` to select the
evaluator provider. Use `--judge-model` to select a model for that provider.
When the selected provider differs from `TWINKL_COACH_PROVIDER`, it uses its
provider default if `--judge-model` is absent. For example, this command
evaluates OpenAI-generated responses with Gemini:

```sh
uv run python -m src.evals.coach_narrative_judge \
  --manifest logs/experiments/reports/demo_v4_run1_20260907/judge_sample_manifest.json \
  --judge-provider gemini \
  --out logs/experiments/reports/demo_v4_run1_20260907/evals_cross_provider \
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
