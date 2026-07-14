# Weekly Digest Generation

## Purpose

The Weekly Digest is the structured bridge between Drift Detector output and
the Weekly Coach. It does not produce a generic summary. It organizes the week
against Core Values, selects evidence, records the decision rationale, and
renders the prompt that the Weekly Coach can consume. The executable prototype
can still build a Weekly Digest from VIF Critic predictions or LLM-Judge labels;
that compatibility path is not the approved user-facing architecture.

Read this document with:

- [`docs/prd.md`](../prd.md) for product intent;
- [`docs/drift/trajectory_eda.md`](../drift/trajectory_eda.md) for the selected
  Drift definition;
- [`docs/evals/drift_detection_eval.md`](../evals/drift_detection_eval.md) for
  the evaluation contract;
- [`docs/vif/example.md`](../vif/example.md) for target Weekly Coach tone; and
- [`docs/evals/explanation_quality_eval.md`](../evals/explanation_quality_eval.md)
  for narrative checks.

---

## Two Executable Paths

### Standalone Digest CLI

The standalone command defaults to persisted single-pass LLM-Judge labels:

```sh
uv run python -m src.coach.weekly_digest --persona-id 0a2fe15c
```

With no explicit dates, it uses the latest available Journal Entry date and a
seven-day inclusive window. Pin a window with:

```sh
uv run python -m src.coach.weekly_digest \
  --persona-id 0a2fe15c \
  --start-date 2025-12-03 \
  --end-date 2025-12-09
```

Use a saved VIF Critic timeline instead of LLM-Judge labels with `--signals-path`:

```sh
uv run python -m src.coach.weekly_digest \
  --persona-id 0a2fe15c \
  --signals-path logs/exports/weekly_coach/0a2fe15c_vif_timeline.parquet
```

When `--signals-path` is supplied, it takes precedence over `--labels-path`.

Pass an upstream routing result with:

```sh
uv run python -m src.coach.weekly_digest \
  --persona-id 0a2fe15c \
  --drift-result-json path/to/drift_result.json
```

Force a schema mode for local testing with:

```sh
uv run python -m src.coach.weekly_digest \
  --persona-id 0a2fe15c \
  --response-mode high_uncertainty
```

If both `--response-mode` and `--drift-result-json` are present, the manual
response mode wins and upstream Drift reasons are not carried into the Weekly Digest.

#### Standalone Options

| Option | Default / behavior |
|---|---|
| `--persona-id` | Required; ID without the `persona_` filename prefix |
| `--start-date` | Unset |
| `--end-date` | Latest available date |
| `--response-mode` | Unset; manual testing override |
| `--drift-result-json` | Unset |
| `--labels-path` | `logs/judge_labels/judge_labels.parquet` |
| `--signals-path` | Unset; takes precedence over labels when supplied |
| `--wrangled-dir` | `logs/wrangled` |
| `--output-dir` | `logs/exports/weekly_digests` |
| `--parquet-path` | `logs/exports/weekly_digests/weekly_digests.parquet` |

### End-to-End Checkpoint Runtime

The full offline runtime currently predicts a VIF Critic timeline, aggregates
weekly signals, runs the existing prototype router, and builds the Weekly
Digest. This demonstrates the implemented prototype, not the approved Weekly
Drift Reviewer and Drift Detector path:

```sh
uv run python -m src.coach.runtime \
  --persona-id 0a2fe15c \
  --checkpoint-path logs/experiments/artifacts/.../selected_checkpoint.pt
```

Required options:

- `--persona-id`
- `--checkpoint-path`

Optional runtime controls:

| Option | Default |
|---|---|
| `--wrangled-dir` | `logs/wrangled` |
| `--config-path` | `config/vif.yaml` |
| `--output-dir` | `logs/exports/weekly_coach` |
| `--parquet-path` | `logs/exports/weekly_digests/weekly_digests.parquet` |
| `--start-date` / `--end-date` | Unset |
| `--n-mc-samples` | Unset; checkpoint/config value applies |
| `--batch-size` | `32` |
| `--device` | Unset; runtime selects the device |

The runtime writes:

```text
logs/exports/weekly_coach/
├── <persona_id>_vif_timeline.parquet
├── <persona_id>_vif_weekly.parquet
├── <persona_id>_<week_end>.drift.json
├── <persona_id>_<week_end>.json
├── <persona_id>_<week_end>.md
└── <persona_id>_<week_end>.prompt.txt
```

It also upserts the consolidated parquet at `--parquet-path`.

Both CLIs render and persist the Weekly Coach prompt. They do not call a live
Weekly Coach LLM. Programmatic callers can inject an asynchronous
`llm_complete` callable
into `run_weekly_coach_cycle()` or the lower-level generation functions to
populate `CoachNarrative` and `DigestValidation`.

---

## Demo Review UI

```sh
uv run shiny run src/demo_tool/app.py
```

The Shiny app uses the same runtime path, reuses cached persona/checkpoint
bundles, and exposes the structured digest and prompt. It does not generate a
live LLM narrative. See [`docs/demo/review_app.md`](../demo/review_app.md).

---

## Current Data Paths

### Standalone Path

1. Load persisted LLM-Judge labels, or a saved VIF Critic timeline from
   `--signals-path`.
2. Resolve the requested seven-day or explicit date window.
3. Read an upstream Drift result when supplied.
4. Load the persona profile and journal history from wrangled markdown.
5. Truncate the history at `week_end` to prevent future-Journal-Entry leakage.
6. Compute dimension summaries and select evidence.
7. Use a local fallback mode only when no upstream result or manual override is
   available.
8. Render and persist JSON, markdown, prompt, and consolidated parquet output.

### End-to-End Runtime Path

1. Reconstruct student-visible states from the wrangled timeline.
2. Run the frozen VIF Critic checkpoint with MC Dropout.
3. Persist per-Journal-Entry means and uncertainties.
4. Aggregate the timeline into a validated weekly frame.
5. Run the weekly crash/rut/evolution prototype router.
6. Pass the live VIF Critic predictions and structured routing result into the
   Weekly Digest builder.
7. Render and persist the runtime bundle.

`src/vif/weekly_schema.py` owns the weekly column contract between the runtime
producer and the prototype router. Missing required columns fail early
with a descriptive `ValueError`.

---

## Digest Contract

`WeeklyDigest` stores:

- persona ID, name, and date-window metadata;
- response mode, source, rationale, and optional upstream reasons;
- weekly aggregate metrics;
- Core Values;
- per-dimension summaries;
- representative evidence snippets;
- Journal Entry history capped at `week_end`;
- an optional `CoachNarrative`; and
- an optional `DigestValidation`.

The Weekly Digest is the canonical record passed to the Weekly Coach rather
than a transient prompt intermediary.

### Schema Modes

The schema accepts seven literal modes:

- `stable`
- `rut`
- `crash`
- `evolution`
- `high_uncertainty`
- `mixed_state`
- `background_strain`

The sources differ:

- The existing upstream runtime prototype can emit `stable`, `rut`, `crash`,
  `evolution`, and `high_uncertainty`.
- Automatic `evolution` routing is present in the prototype code even though
  the PRD does not adopt value evolution as v1 product behavior.
- Standalone fallback logic can emit `stable`, `rut`, `high_uncertainty`,
  `mixed_state`, and `background_strain`.
- Manual overrides can exercise any schema mode.

The schema is intentionally wider than the approved product contract so the
current prototype and prompt experiments remain inspectable.

These literal prototype modes are not the adopted v1 delivery vocabulary. The
Weekly Coach is intended to describe the delivery-time state as **active**,
**recovered**, **mixed**, or **uncertain**. The exact schema fields and mapping
from the current compatibility modes remain implementation work.

---

## Approved Drift Path Versus the Prototype Router

Drift is two consecutive Conflicts on the same Core Value:

- student-visible target: two adjacent Journal Entries visibly show a behavior
  or choice against the same Core Value;
- approved user-facing input: Weekly Drift Reviewer decisions made without VIF
  Critic predictions; and
- delivery: the Weekly Digest cites the supporting Journal Entries.

Each Core Value is evaluated independently. An aligned label for another Core
Value cannot cancel Drift, and simultaneous Drifts remain separate
value-specific records. The six-detector comparison's detector-vote count is
not the five-pass LLM-Judge reference.

The current runtime persists VIF Critic alignment means and uncertainties rather
than Weekly Drift Reviewer decisions or full class probabilities. The
[`twinkl-752.5`
reassessment](../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_752_5_reassessment.md)
used the 33-Drift known-development union and found no reliable benefit from
showing raw VIF Critic scores to the Weekly Drift Reviewer. VIF-Critic-triggered
early-plus-weekly review changed median delay but did not add Drift hits; its
timing benefit disappeared on the non-training subgroup. The offline VIF
Critic triggers targeted Drift-relevant opportunities better than random, but
that diagnostic made no reviewer calls and does not show that early review
works. No fresh final test exists.
The adopted architecture keeps VIF Critic predictions in offline comparison,
independent review, and retraining. The VIF Critic may later propose candidate
adjacent Conflict pairs only after predefined criteria and a fresh final test
support deployment approval. No VIF Critic or Drift Detector has deployment
approval, and the approved Drift Detector is not wired into the weekly runtime.
The former consensus-derived frozen benchmark is retired historical evidence.
The crash/rut/evolution output modes remain prototype compatibility values, not
the accepted v1 definition.

### Delivery-Time Recovery

The student-visible target records whether Drift occurred. Weekly Digest
wording reflects the state when the Weekly Coach is delivered.

For each value-specific Drift:

- **active**: its conflict run reaches the digest cutoff;
- **recovered**: a later non-Conflict decision closes the run before the cutoff;
- **uncertain**: a later Weekly Drift Reviewer abstention prevents a confident
  active-versus-recovered claim; and
- **mixed**: a Weekly Digest summary when relevant value-specific Drifts have
  different delivery states. It is not a fourth Drift type.

A sequence such as `-1, -1, +1, +1, +1` therefore remains a true Drift but is
described as recovered rather than active. Exact production
schema values and compatibility mapping from current modes remain
implementation work.

---

## Standalone Fallback Semantics

When no upstream result or manual override is supplied, the digest builder uses
conservative local heuristics for offline development:

- `high_uncertainty`: acute grief/distress markers make a value-specific
  critique unsafe;
- `mixed_state`: the week contains meaningful supportive and straining signals;
- `background_strain`: the week is positive overall but carries softer burden
  or transition cues;
- `rut`: a clearly negative weekly aggregate includes a Core Value
  among the main tensions; and
- `stable`: none of the preceding conditions apply.

These heuristics do not implement calibrated VIF Critic uncertainty or the
selected Drift Detector. They keep prompt and Weekly Digest work usable when
upstream results are absent.

---

## Prompt and Narrative Contract

The Weekly Coach prompt requires:

- reflective rather than prescriptive language;
- no score jargon, gamification, or judgmental framing;
- no micro-habit or action-plan output;
- quoted evidence where possible; and
- strict JSON fields:
  - `weekly_mirror`
  - `tension_explanation`
  - `reflective_question`

Tier 1 narrative validation checks:

- `groundedness`: at least one quoted phrase appears in journal history or
  selected evidence;
- `non_circularity`: the narrative avoids score/alignment jargon; and
- `length`: total narrative length remains within configured bounds.

These checks are narrow guardrails, not a complete explanation-quality claim.

---

## Safety and Selection Behavior

- Core Values receive priority when dimension summaries are equally
  informative.
- A dimension does not appear in both tension and strength sections by default.
- Backfilled Weekly Digests cannot read Journal Entries after `week_end`.
- Empty sections render `None clear this week` rather than inventing a tension.
- Acute grief/distress fallback favors presence over brittle value scoring.
- Mixed-state and background-strain fallbacks preserve nuance that a weekly
  mean can hide.

The acute-distress heuristic is lexical, and `mixed_state` /
`background_strain` are aggregate heuristics. They are safety scaffolding, not
learned routing policies.

---

## Remaining Work

1. Persist Weekly Drift Reviewer decisions and wire the deterministic
   two-Conflict Drift Detector into the Weekly Digest path.
2. Persist full VIF Critic class probabilities, uncertainty, checkpoint
   provenance, and input-contract version for offline comparison.
3. Add independent disagreement review, versioned retraining data, and
   model-blind controls. Weekly Drift Reviewer outputs must not automatically
   become LLM-Judge reference labels.
4. Under `twinkl-7vam`, predefine Drift recall, false Drift alert, coverage,
   abstention, stability, and any efficiency criteria. Select and freeze any
   VIF Critic candidate rule on development evidence only.
5. Build a fresh final test under `twinkl-pv6s`. Keep retraining cases out of it
   and score only after the checkpoint, rule, prompt, and criteria are frozen.
6. Add active, recovered, mixed, and uncertain Weekly Digest delivery wording.
   A Weekly Drift Reviewer abstention must emit no Drift claim, and coverage
   must be reported.
7. Add a production-facing Weekly Coach service entrypoint that injects the
   live LLM,
   validates the narrative, and persists the result.
8. Report Tier 1 pass rates over a batch and add Tier 2 meta-judge plus Tier 3
   human calibration.
9. Capture the user's perceived-accuracy rating and make it queryable.

---

## Implementation Reference

| Module | Role |
|---|---|
| `src/coach/weekly_digest.py` | Digest construction, fallback routing, prompt rendering, validation, and persistence |
| `src/coach/runtime.py` | Checkpoint-to-digest offline orchestration |
| `src/coach/mode_logic.py` | Standalone fallback response-mode logic |
| `src/coach/schemas.py` | Drift, digest, narrative, and validation schemas |
| `src/vif/runtime.py` | Per-Journal-Entry inference and weekly aggregation |
| `src/vif/weekly_schema.py` | Weekly frame column contract and validation |
| `src/vif/drift.py` | Existing weekly crash/rut/evolution prototype router |
| `prompts/weekly_digest_coach.yaml` | Weekly Coach prompt template |

---

## Verification

Targeted checks:

```sh
uv run pytest tests/coach/test_weekly_digest.py tests/vif/test_drift.py -q
```

The tests cover Weekly Digest construction, future-Journal-Entry isolation, upstream/manual
mode handling, safety fallbacks, prompt rendering, structured generation with a
fake LLM, persistence, weekly schema validation, and prototype router behavior.
