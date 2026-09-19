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
| Coach Digest Validations (quotations, question form and count, non-circularity, raw value leakage, current-state claims, length, voice) | Unit tests | No | `tests/coach/test_weekly_digest.py`, `tests/coach/test_weekly_digest_qa_regressions.py` |
| Frozen generation, repair, selected comparison cases, and saved validation policies | Unit tests | No | `tests/coach/test_complete_scenario_coach.py`, `tests/coach/test_refresh_scenario_coach.py`, `tests/coach/test_compare_scenario_coach.py` |
| Provider attempt limits and failed response envelopes | Contract tests | No (local HTTP transport) | `tests/test_model_provider_boundaries.py`, `tests/nudge/test_runtime.py`, `tests/test_weekly_drift_reviewer.py` |
| Experience browser journeys | Browser smoke | No (controlled Python providers) | `frontend/onboarding/e2e/experience.spec.ts` |
| Coach Digest Validations batch report over a real Weekly Drift Detection output set | Eval | No | `src/evals/coach_digest_validations.py` |
| Coach Digest Evals (correctness, specificity, non-prescriptive tone, tension honesty) | Eval | **Yes (paid)** | `src/evals/coach_narrative_judge.py` |
| Coach Digest Drift/control comparison | Study | **Yes (paid generation and AI review)** | `scripts/experiments/run_coach_drift_control_eval.py`, `src/evals/coach_drift_control_report.py` |
| Coach source-context comparison and factual controls | Development experiment | **Yes (paid generation and AI review)** | `scripts/experiments/run_coach_context_eval.py` |

Coach Digest Validations are mechanical code checks, not human validation.
Coach Digest Evals produce **AI review, not human validation**. Human calibration
and the user pilot are **not done — closed**, outside the capstone scope.

Independent-provider Coach Digest evaluation and the 42-Drift/42-control study
are **Done** within their accepted workflow scope. Provider selection,
deterministic targets, safe resume, and reporting are implemented. No paid
independent study result is committed or required for capstone closeout. The
commands below remain available for reproduction and explicitly authorized
additional studies.

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

Run the source-quotation, generated-question, state-claim, and explicit retry
regressions:

```sh
uv run pytest tests/coach/test_weekly_digest_qa_regressions.py \
  tests/demo/test_coach_recovery.py
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

Ordinary Coach Digest generation uses prompt `4.7`. It requires an exact
quotation from supplied evidence in `weekly_mirror`, and every source quotation
must match supplied evidence. All three fields must be nonempty, with one
generated question across the complete response, in `reflective_question`, and
a maximum of 180 words; there is no minimum word count. Verified source
quotations are excluded from generated-question and unsupported-state-claim
checks. A source question can therefore appear inside a grounded reflection
without counting as an extra question addressed to the user.

Historical receipts retain their recorded validation policy. Prompt `4.6` keeps
its original quotation and question checks; `4.1`–`4.5` retain their historical
base checks and 25–180-word bounds, with voice checks appropriate to the saved
response. These checks do not establish semantic correctness, temporal
attribution, or non-prescriptive tone. Explicit Experience retries use validation
errors from the latest matching rejected attempt while preserving every attempt
in Inspect. They reuse the frozen Weekly Drift Detection output.
The displayed nudge uses the same question-form check. SDK automatic retries
are disabled: the nudge and
Coach Digest provider adapters make one attempt per call; the Weekly Drift
Reviewer owns its maximum of two attempts for transient failures. A refusal
or incomplete provider response cannot become a displayed response.

Run these boundary checks without network access:

```sh
uv run python -m pytest tests/test_model_provider_boundaries.py \
  tests/nudge/test_runtime.py tests/test_weekly_drift_reviewer.py \
  tests/evals/test_coach_behavioral_regressions.py
```

The [behavioral corpus](../../config/evals/coach_behavioral_regressions_v1.json)
contains eight paired examples, including invented quotations, advice, assumed
motives, uncompleted intentions, and instruction-like Journal Entries. These
are Codex-authored synthetic examples with author expectations stored outside
the model input. They are not sampled model outputs, measured evaluation
results, or human validation. The unit tests exercise structural failures and
complete source projection; semantic examples remain inputs for Coach Digest
Evals or a human review. The existing evaluator can preview the 16 responses:

```sh
uv run python -m src.evals.coach_narrative_judge \
  --manifest config/evals/coach_behavioral_regressions_v1.json
```

This is a dry run. Paid evaluation still requires `--execute`. This corpus does
not evaluate the combined Coach Digest and selected North Star Moment; that
review needs both components and their full source context.

For browser smoke, follow the Chromium setup and `npm run test:e2e` command in
the [frontend README](../../frontend/onboarding/README.md#checks). The two
narrow-screen journeys cover replay/Inspect/reload and manual
writing/closed-week review/Coach Digest retry/confirmed session deletion.

---

## 2. Automated batch evaluation (no API calls)

Runs `validate_weekly_digest_narrative()` over the exact responses in the public
scenario sample manifest and reports per-check pass rates against the targets in
[`explanation_quality_eval.md`](./explanation_quality_eval.md).
The [September manifest](../../logs/experiments/reports/demo_v4_run1_20260907/judge_sample_manifest.json)
preserves the five accepted 7 September key-week responses from the previous
Persona roster.
The [8 September completion](../../logs/experiments/reports/demo_coach_all_weeks_20260908/report.md)
contains 22 further validated responses for that roster. The
[Persona replacement run](../../logs/experiments/reports/demo_persona_replacement_20260908/report.md)
preserves 17 compatible responses and ten for Lukas and Meera. The current
27-week baseline fixture uses the [prompt-4.4 voice refresh and repair](../../logs/experiments/reports/coach_voice_refresh_20260909/report.md),
and the 22 eligible weeks display [saved comparison pairs](../north_star/demo_coach_comparison.md)
using `4.4`/`1.0` or the selected Lukas `4.5`/`1.1` pilot. The command below checks only the historical
five-response sample; it is not a report of the current roster or all weeks.
Section 3a documents missing-week completion. The August manifest remains
available for reproducing the historical sample.

```sh
uv run python -m src.evals.coach_digest_validations \
  --manifest logs/experiments/reports/demo_v4_run1_20260907/judge_sample_manifest.json \
  --validation-policy recorded
```

- Use `--parquet` only for a separate persisted-output batch.
- `--out` writes `metrics.json` and `report.md`; omit it to only print the
  summary.
- Rows with no response are skipped; unparseable responses are reported under
  `skipped_persona_weeks`.
- `--validation-policy recorded` reproduces saved checks or the recorded
  generation prompt's policy and voice rules. Saved checks take precedence over
  generation metadata. Records without either use an explicitly
  labelled historical base policy. It does not add today's rules to old
  receipts.
- `--validation-policy current` applies all current live response checks,
  including quote-aware question counts, question form, state-claim wording,
  and voice checks, and records the policy
  in the report. Use a separate `--out` directory for this assessment; preserve
  historical reports.
- Parquet re-evaluation restores prior-week comparisons, detailed Drift state,
  and saved validation checks so persistence does not change the factual basis
  of the verdict.
- `--signal-source` defaults to `weekly_drift_reviewer`; `--all-sources` includes
  compatibility outputs from other sources.

Verify all 27 exported responses, current input hashes, exact generation-source
event IDs, and preservation of compatible retained receipts without provider calls:

```sh
uv run pytest tests/demo/test_scenarios.py
```

The [integrated validation report](../../logs/experiments/reports/integrated_coach_validation_20260908/report.md)
records application checks and five unresolved AI editorial findings under
`twinkl-rklc.39`. A selected North Star Moment passage appears within a valid Coach Digest,
after its original question. Code checks do not resolve the reported semantic
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

### Source-context comparison

The bounded runner compares Wei Jun's saved short excerpts with complete
displayed text for the same selected Journal Entries. It keeps Coach prompt
`4.6`, evaluator prompt `3.1`, Luna at reasoning effort `none`, and Drift
decisions fixed. Three generations per input produce six fresh responses.
All six receive fresh evaluator scores against the same complete-context
digest, including responses that fail mechanical checks. A failed provider
response remains a recorded failure rather than a selected replacement.

Three additional assessments score the recorded incorrect event connection
with short and complete context, and a constructed accurate control with
complete context. The full run therefore permits at most 15 provider calls,
without automatic retries or repair generations. Source expectations and arm
labels stay outside evaluator input. Scores remain same-model AI review.

```sh
# Prepare the frozen requests without provider calls:
uv run python -m scripts.experiments.run_coach_context_eval \
  --out /tmp/coach-context-new-run

# Run the prepared comparison and fresh evaluator assessments:
uv run python -m scripts.experiments.run_coach_context_eval \
  --out /tmp/coach-context-new-run --execute
```

The runner preserves source hashes, exact requests, raw responses, usage, and
individual scores. Calls are appended to `receipts.jsonl`, one checksum-bearing
JSON record per call. `source_provenance.json` and `source.patch` reconstruct
the selected sources from a fixed Git revision, avoiding a copied source tree.
Resume verifies the source hashes and rejects corrupt or duplicate call records
before making a provider call; legacy separate call files remain readable.
It does not replace saved scenario responses or add an
evaluator call to the product runtime. The regression demonstrates whether
context changes this case; it is not a fresh final test or human validation.

Use a new output directory for changed code or prompts. The
[13 September run](../../logs/experiments/reports/coach_context_eval_20260913/report.md)
preserves all 15 completed calls. The known incorrect response scored 5 with
short evidence and 4 with complete evidence, so neither assessment triggered
the evaluator's review flag. High overall scores do not establish detection of
event-attribution errors.

### 3a. Complete saved replay weeks and retain compatible responses

All 27 saved replay weeks contain a validated Coach Digest. The completion
runner reads each week's exact saved `weekly_digest_built` input, preserves
compatible existing responses, and generates only missing weeks. It uses Luna
at reasoning effort `none`, ordinary Coach prompt `4.7`, no SDK automatic
retries, and at most one validation-guided retry per case. It makes no Weekly Drift Reviewer or NSM
calls. `OPENAI_API_KEY` is read through the existing environment setup.

```sh
# Prepare or verify the saved plan; no provider calls:
uv run python -m scripts.coach.complete_scenario_coach \
  --output logs/experiments/reports/coach_completion_run

# Complete missing responses within authorized paid scope:
uv run python -m scripts.coach.complete_scenario_coach \
  --output logs/experiments/reports/coach_completion_run --execute

# Rebuild the public scenario bundles without provider calls:
uv run python -m scripts.export_demo_experiments
```

| Option | Default / behavior |
| --- | --- |
| `--output` | `logs/experiments/reports/demo_persona_replacement_20260908/current`; checkpoint/report directory within the repository |
| `--execute` | Off; permits generation of missing responses and merges each accepted response into the active fixture |
| `--repair-requirements` | Optional JSON mapping from `scenario::week-start` keys to lists of repair instructions; frozen in the plan |

Use an unused `--output` directory for current inputs and policy. Historical
plans reject a changed roster, input hash, code hash, or generation policy.
The historical all-week completion records 22 responses with 26 calls; the
Persona replacement run records 17 retained responses and ten generated
responses with 12 calls. Each generation attempt preserves its
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
or perform semantic judging. An optional all-week AI evaluation requires
a source-bound manifest of the exact displayed responses. Historical scores
must not be assigned to newly generated responses. The NSM live allowance is separate
from Coach generation: its pinned US$1 budget is shared across sessions and
restarts and is not spent by offline scenario export.

### 3b. Refresh all saved baseline responses

The refresh runner stages a replacement for every saved baseline response using
ordinary Coach prompt `4.7` and Luna at reasoning effort `none`. Each case permits
at most four attempts with validation-guided repair and no SDK retries. The
frozen plan reserves US$0.018 per request against a US$5 total cap. Accepted
responses remain staged until `--apply` verifies complete validated coverage
and replaces `src/demo/coach_digest_responses.json`.

```sh
# Freeze inputs, prompts, policy, and the original response fixture:
uv run python -m scripts.coach.refresh_scenario_coach \
  --output logs/experiments/reports/coach_refresh_run

# Generate the staged responses within authorized paid scope:
uv run python -m scripts.coach.refresh_scenario_coach \
  --output logs/experiments/reports/coach_refresh_run --execute

# Apply validated coverage, then rebuild saved replay bundles without calls:
uv run python -m scripts.coach.refresh_scenario_coach \
  --output logs/experiments/reports/coach_refresh_run --apply
uv run python -m src.demo.scenarios
```

| Option | Default / behavior |
| --- | --- |
| `--output` | `logs/experiments/reports/coach_voice_refresh_20260909`; use an unused repository directory for a fresh run |
| `--execute` | Off; permits bounded paid generation |
| `--apply` | Off; installs complete validated responses without requiring provider calls |
| `--prior-run` | Optional compatible accepted run from which unselected responses are retained; requires a nonempty `--repair-requirements` mapping |
| `--repair-requirements` | With `--prior-run`, JSON mapping from `scenario::week-start` keys to nonempty lists of editorial repair instructions |

An editorial repair requires `--prior-run` and `--repair-requirements` together,
uses a different output directory, and preserves responses outside its selected
cases. Resume and apply use the same prior-run and repair
arguments as preparation. Frozen plans reject changed inputs or policies, and
an attempt without its diagnostic requires inspection. Scenario export changes
source hashes; a completed plan remains a receipt of its pre-export inputs.
The active fixture retains prompt `4.4` provenance from the recorded September
run until a separately authorized refresh is applied.

The [comparison runbook](../north_star/demo_coach_comparison.md#generation-and-application)
covers paired responses, including `--case` selection, `--prior-run`,
`--editorial-repairs`, and partial application. Those pairs use their own
versioned comparison prompt and validation policy.

### 3c. Run the AI review

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

Evaluator prompt `3.1` accepts a grounded, open-ended question as a way to
preserve uncertainty for `more_reflection_needed`. It does not require a
statement of the evidence limit. The response must still leave Drift undecided
and avoid inventing decisions, motives, or outcomes. This matches ordinary
Coach Digest prompt `4.7`. Keep any rerun under the revised rubric separate
from historical evaluation receipts.

---

## 4. Drift/control study

**Status: Done** for the accepted 42-Drift/42-control workflow. The paid study
result is absent from committed evidence and is not a capstone closeout
requirement. This section documents how to reproduce or execute the workflow.

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
