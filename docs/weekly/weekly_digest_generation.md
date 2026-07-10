# Weekly Digest Generation

## Purpose

The weekly digest is the structured bridge between VIF/Judge signals and the
narrative Coach. It does not produce a generic summary. It organizes the week
against declared values, selects evidence, records the routing rationale, and
renders the prompt that a Coach model can consume.

Read this document with:

- [`docs/prd.md`](../prd.md) for product intent;
- [`docs/drift/trajectory_eda.md`](../drift/trajectory_eda.md) for the selected
  sustained-conflict definition;
- [`docs/evals/drift_detection_eval.md`](../evals/drift_detection_eval.md) for
  the benchmark contract;
- [`docs/vif/example.md`](../vif/example.md) for target Coach tone; and
- [`docs/evals/explanation_quality_eval.md`](../evals/explanation_quality_eval.md)
  for narrative checks.

---

## Two Executable Paths

### Standalone Digest CLI

The standalone command defaults to persisted single-pass Judge labels:

```sh
uv run python -m src.coach.weekly_digest --persona-id 0a2fe15c
```

With no explicit dates, it uses the latest available entry date and a seven-day
inclusive window. Pin a window with:

```sh
uv run python -m src.coach.weekly_digest \
  --persona-id 0a2fe15c \
  --start-date 2025-12-03 \
  --end-date 2025-12-09
```

Use a saved Critic timeline instead of Judge labels with `--signals-path`:

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
response mode wins and upstream drift reasons are not carried into the digest.

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

The full offline runtime predicts a Critic timeline, aggregates weekly signals,
runs the existing prototype router, and builds the digest:

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

Both CLIs render and persist the Coach prompt. They do not call a live Coach
LLM. Programmatic callers can inject an asynchronous `llm_complete` callable
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

1. Load persisted Judge labels, or a saved Critic timeline from
   `--signals-path`.
2. Resolve the requested seven-day or explicit date window.
3. Read an upstream drift result when supplied.
4. Load the persona profile and journal history from wrangled markdown.
5. Truncate the history at `week_end` to prevent future-entry leakage.
6. Compute dimension summaries and select evidence.
7. Use a local fallback mode only when no upstream result or manual override is
   available.
8. Render and persist JSON, markdown, prompt, and consolidated parquet output.

### End-to-End Runtime Path

1. Reconstruct student-visible states from the wrangled timeline.
2. Run the frozen Critic checkpoint with MC Dropout.
3. Persist per-entry means and uncertainties.
4. Aggregate the timeline into a validated weekly frame.
5. Run the weekly crash/rut/evolution prototype router.
6. Pass the live Critic signals and structured routing result into the digest
   builder.
7. Render and persist the runtime bundle.

`src/vif/weekly_schema.py` owns the weekly column contract between the runtime
producer and the prototype drift consumer. Missing required columns fail early
with a descriptive `ValueError`.

---

## Digest Contract

`WeeklyDigest` stores:

- persona ID, name, and date-window metadata;
- response mode, source, rationale, and optional upstream reasons;
- weekly aggregate metrics;
- declared core values;
- per-dimension summaries;
- representative evidence snippets;
- journal history capped at `week_end`;
- an optional `CoachNarrative`; and
- an optional `DigestValidation`.

The digest is the canonical Coach-facing weekly artifact rather than a transient
prompt intermediary.

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

The schema is intentionally wider than the selected product contract so the
current prototype and prompt experiments remain inspectable.

---

## Drift v1 Versus the Prototype Router

The selected v1 construct is sustained conflict on a declared core/high-weight
value:

- strict reference: two consecutive consensus `-1` labels;
- runtime target: rolling soft `P(-1)` evidence under uncertainty gating; and
- delivery: the weekly digest cites the supporting entries.

The current runtime persists alignment means and uncertainties rather than
class probabilities, so the rolling-soft-evidence detector is not implemented.
The crash/rut/evolution output modes remain prototype compatibility values, not
the accepted v1 definition.

### Delivery-Time Recovery

Benchmark detection records whether a strict sustained-conflict episode
occurred. Digest wording reflects the state when the weekly Coach is delivered.

A sequence such as `-1, -1, +1, +1, +1` is still a true benchmark episode, but
the Coach should describe recovery rather than claim that drift is ongoing.
`recovered` is not currently a schema mode, so active/recovered/mixed delivery
semantics remain implementation work.

---

## Standalone Fallback Semantics

When no upstream result or manual override is supplied, the digest builder uses
conservative local heuristics for offline development:

- `high_uncertainty`: acute grief/distress markers make a value-specific
  critique unsafe;
- `mixed_state`: the week contains meaningful supportive and straining signals;
- `background_strain`: the week is positive overall but carries softer burden
  or transition cues;
- `rut`: a clearly negative weekly aggregate includes a declared core value
  among the main tensions; and
- `stable`: none of the preceding conditions apply.

These heuristics do not implement calibrated Critic uncertainty or the selected
v1 detector. They keep prompt and artifact work usable when upstream results are
absent.

---

## Prompt and Narrative Contract

The Coach prompt requires:

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

- Declared core values receive priority when dimension summaries are equally
  informative.
- A dimension does not appear in both tension and strength sections by default.
- Backfilled digests cannot read entries after `week_end`.
- Empty sections render `None clear this week` rather than inventing a tension.
- Acute grief/distress fallback favors presence over brittle value scoring.
- Mixed-state and background-strain fallbacks preserve nuance that a weekly
  mean can hide.

The acute-distress heuristic is lexical, and `mixed_state` /
`background_strain` are aggregate heuristics. They are safety scaffolding, not
learned routing policies.

---

## Remaining Work

1. Persist or reconstruct per-entry ordinal class probabilities, including
   `P(-1)`.
2. Implement rolling soft-evidence sustained-conflict detection.
3. Calibrate probability-mass and uncertainty thresholds against strict
   consensus-reference episodes.
4. Evaluate the MLP and LLM context arms on episode hit rate, false alarms,
   latency, and per-dimension behavior.
5. Add active, recovered, mixed, and uncertain digest-time wording semantics.
6. Add a production-facing Coach service entrypoint that injects the live LLM,
   validates the narrative, and persists the result.
7. Report Tier 1 pass rates over a batch and add Tier 2 meta-judge plus Tier 3
   human calibration.
8. Capture the user's perceived-accuracy rating and make it queryable.

---

## Implementation Reference

| Module | Role |
|---|---|
| `src/coach/weekly_digest.py` | Digest construction, fallback routing, prompt rendering, validation, and persistence |
| `src/coach/runtime.py` | Checkpoint-to-digest offline orchestration |
| `src/coach/mode_logic.py` | Standalone fallback response-mode logic |
| `src/coach/schemas.py` | Drift, digest, narrative, and validation schemas |
| `src/vif/runtime.py` | Per-entry inference and weekly aggregation |
| `src/vif/weekly_schema.py` | Weekly frame column contract and validation |
| `src/vif/drift.py` | Existing weekly crash/rut/evolution prototype router |
| `prompts/weekly_digest_coach.yaml` | Coach prompt template |

---

## Verification

Targeted checks:

```sh
uv run pytest tests/coach/test_weekly_digest.py tests/vif/test_drift.py -q
```

The tests cover digest construction, future-entry isolation, upstream/manual
mode handling, safety fallbacks, prompt rendering, structured generation with a
fake LLM, persistence, weekly schema validation, and prototype drift behavior.
