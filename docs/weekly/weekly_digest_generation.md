# Weekly Digest Generation

## Purpose

This document is the implementation-facing reference for Twinkl's weekly digest and Coach flow.
It replaces the previous placeholder state and should be read together with:

- `docs/prd.md` for product intent
- `docs/weekly/comments.md` for scope decisions and historical commentary
- `docs/vif/example.md` for target Coach tone
- `docs/evals/explanation_quality_eval.md` for Tier 1 explanation checks

The weekly digest is the bridge between the numeric VIF/Judge layer and the narrative Coach layer.
Its job is not to summarize the week generically. Its job is to mirror the week back against declared values, cite evidence, and hand structured context to the Coach prompt.


## How To Run

The examples below use `uv run` so they pick up the project environment
directly. Activating `.venv` manually also works.

### Basic CLI

Generate a digest for one persona using the latest available 7-day window:

```sh
uv run python -m src.coach.weekly_digest --persona-id 0a2fe15c
```

Generate a digest for an explicit date window:

```sh
uv run python -m src.coach.weekly_digest \
  --persona-id 0a2fe15c \
  --start-date 2025-12-03 \
  --end-date 2025-12-09
```

### Using Upstream Drift Output

Preferred long-term usage is to pass drift detection output into the digest generator.

Example `drift_result.json`:

```json
{
  "response_mode": "background_strain",
  "rationale": "Upstream drift detector found a positive week with softer transition strain.",
  "reasons": ["supportive_week", "transition_burden", "no_clear_negative_core_value"],
  "source": "drift_detector"
}
```

Run the digest generator with that upstream drift result:

```sh
uv run python -m src.coach.weekly_digest \
  --persona-id 0ad04582 \
  --start-date 2025-12-26 \
  --end-date 2026-01-01 \
  --drift-result-json drift_result.json
```

### Temporary Fallback / Manual Override

If no `--drift-result-json` is provided, the digest generator uses temporary local fallback heuristics for offline development.

You can also force a mode manually for testing:

```sh
uv run python -m src.coach.weekly_digest \
  --persona-id 0a2fe15c \
  --start-date 2025-12-03 \
  --end-date 2025-12-09 \
  --response-mode high_uncertainty
```

Manual override is for testing only.
Normal pipeline usage should prefer upstream drift output.

The CLI accepts `evolution` as a schema-compatible manual override, but the
active runtime does not automatically route weeks into `evolution`.

### End-to-End Runtime CLI

Run the full local checkpoint -> weekly signals -> drift -> digest flow:

```sh
uv run python -m src.coach.runtime \
  --persona-id 0a2fe15c \
  --checkpoint-path logs/experiments/artifacts/.../selected_checkpoint.pt
```

Useful overrides for the runtime CLI:

- `--config-path` to point at a non-default VIF config
- `--start-date` and `--end-date` to pin the weekly window
- `--n-mc-samples` to override checkpoint/default MC Dropout sampling
- `--batch-size` to control inference batch size
- `--device` to force CPU or CUDA
- `--output-dir` to change the runtime artifact folder
- `--parquet-path` to change the consolidated weekly digest parquet path

The runtime CLI writes:

- `vif_timeline.parquet`
- `vif_weekly.parquet`
- `<persona>_<week_end>.drift.json`
- `<persona>_<week_end>.json`
- `<persona>_<week_end>.md`
- `<persona>_<week_end>.prompt.txt`
- the upserted consolidated parquet at `--parquet-path`

### Demo Review UI

The Shiny review UI sits on top of the same runtime path:

```sh
uv run shiny run src/demo_tool/app.py
```

It loads cached artifact bundles when available, runs the live checkpoint flow
on demand, and compares six detector families against either judge labels or
Critic predictions. See [`docs/demo/review_app.md`](../demo/review_app.md) for
the full workflow.

### Output Files

By default the command writes:

- `logs/exports/weekly_digests/<persona>_<week_end>.json`
- `logs/exports/weekly_digests/<persona>_<week_end>.md`
- `logs/exports/weekly_digests/<persona>_<week_end>.prompt.txt`
- `logs/exports/weekly_digests/weekly_digests.parquet`

You can override input/output locations with:

- `--labels-path`
- `--signals-path`
- `--wrangled-dir`
- `--output-dir`
- `--parquet-path`


## Scope For The Current POC

The current POC does not need RAG.
The synthetic corpus is small enough that the Coach can use full-context prompting over the entire journal history.

The current POC also does not have production-ready drift detection.
That means:

- crash/rut/high-uncertainty formulas remain a downstream dependency
- the weekly digest generator is designed to consume drift output from upstream
- the current weekly digest implementation still includes conservative fallback mode assignment only as a temporary development stub
- the prompt and output schema are ready for richer trigger wiring later

## Current Implementation

### Implemented In Code

Primary code lives in:

- `src/coach/weekly_digest.py`
- `src/coach/runtime.py`
- `src/coach/mode_logic.py`
- `src/coach/schemas.py`
- `src/vif/runtime.py`
- `src/vif/drift.py`
- `prompts/weekly_digest_coach.yaml`

An experimental `src/vif/evolution.py` module also exists in the repo, but evolution gating is currently undecided and is not part of the committed weekly Coach flow described in this document.

The current implementation now supports:

1. Building a structured weekly digest from `logs/judge_labels/judge_labels.parquet`
2. Loading wrangled journal history for the persona from `logs/wrangled/persona_<id>.md`
3. Accepting an upstream drift-detection result as an explicit input contract
4. Carrying declared core values into the digest payload
5. Selecting focus tensions/strengths with core-value-aware ranking
6. Selecting representative evidence snippets from the scored week
7. Truncating journal history at the digest `week_end` so backfilled digests do not leak future entries
8. Preventing overlapping tension/strength dimensions in the same digest by default
9. Using conservative fallback response-mode heuristics only when upstream drift output is unavailable
10. Detecting acute distress / grief markers and falling back to `high_uncertainty` instead of forcing brittle value-level critiques
11. Detecting mixed-signal weeks and positive-weeks-with-burden through a dedicated fallback mode router
12. Rendering empty tension/strength sections explicitly as `None clear this week`
13. Rendering a full-context Coach prompt with:
   - time-safe journal history
   - declared core values
   - value elaborations from `config/schwartz_values.yaml`
   - dimension summaries
   - evidence snippets
   - response mode metadata
14. Generating structured Coach output through an injected LLM callable
15. Running Tier 1 automated narrative checks:
   - groundedness via quoted substring matches
   - non-circularity via score-jargon avoidance
   - length bounds
16. Persisting each weekly digest into a consolidated parquet artifact for future longitudinal analysis
17. Exporting per-run JSON, markdown, and prompt artifacts
18. Rebuilding per-entry VIF signal timelines from a trained Critic checkpoint at runtime
19. Aggregating those signals into weekly VIF summaries for drift detection
20. Running uncertainty-gated crash/rut-style detection experiments and emitting a structured `DriftDetectionResult`
21. Executing the full offline runtime path through `src/coach/runtime.py`
22. Loading the persisted runtime bundle into the demo review UI, including cached run reuse and detector-source switching

### Current Digest Data Model

`WeeklyDigest` now carries:

- persona and window metadata
- response mode, mode source, and mode rationale
- optional drift reasons from upstream detection
- weekly aggregate metrics
- declared core values
- per-dimension summaries
- representative evidence snippets
- time-safe journal history capped at the digest end date
- optional `CoachNarrative`
- optional `DigestValidation`

This is intentionally more than a prompt intermediary.
It is the canonical Coach-facing weekly artifact.

## Current Architecture

### Inputs

Current inputs are offline artifacts:

- judge labels parquet
- wrangled persona markdown
- Schwartz value elaborations config
- optional upstream drift detection result

### Flow

1. Load persona labels for the requested week.
2. Resolve the digest window.
3. Accept upstream drift detection output for the same week when available.
4. Load the full persona profile and journal history from the wrangled file.
5. Truncate journal history to entries on or before the digest `week_end`.
6. Compute per-dimension weekly means and class proportions.
7. Rank top tensions and strengths, with declared core values treated as first-class context and overlaps suppressed.
8. Select representative evidence rows.
9. If no upstream drift result is provided, use a temporary fallback mode heuristic for offline development only.
10. Render the Coach prompt using time-safe history plus structured summaries.
11. Optionally call the LLM with the structured Coach JSON schema.
12. Optionally run Tier 1 validation.
13. Persist both per-run exports and a consolidated parquet record.

### Response Modes

Supported schema modes:

- `stable`
- `rut`
- `crash`
- `high_uncertainty`
- `mixed_state`
- `background_strain`

Current implementation status:

- all modes are schema-supported as upstream drift outputs
- the active upstream-facing runtime contract is `stable`, `rut`, `crash`, and `high_uncertainty`
- fallback-only local inference still exists for `stable`, `rut`, `high_uncertainty`, `mixed_state`, and `background_strain` when no upstream drift result is supplied
- `mixed_state` and `background_strain` remain heuristic-only local modes
- `high_uncertainty` can now come from real aggregated Critic uncertainty in the runtime path
- `evolution` remains available for schema/manual override compatibility, but it is not an automatically-routed mode in the active runtime

`evolution` remains an undecided idea rather than an active runtime commitment for the Coach flow.

This is deliberate.
The digest contract is designed for full trigger wiring from drift detection. The in-module heuristics are temporary scaffolding for offline development and prompt testing.

### Temporary Fallback Semantics

When no upstream drift-detection result is supplied, the temporary fallback mode logic behaves as follows:

- `high_uncertainty`
  - triggered when the weekly journal window contains acute grief/distress markers such as death, screaming, crying, or explicitly unbearable situations
  - intended to prevent the Coach from making brittle value-specific critiques in weeks where a presence-oriented reflection is safer
- `mixed_state`
  - triggered when the week contains both supportive and straining signals without a single dominant direction
  - intended to recover nuance when weekly means hide within-week polarity shifts
- `background_strain`
  - triggered when the week trends positive overall but softer burden or transition cues suggest the user is carrying strain in the background
  - intended to avoid flattening nuanced weeks into `stable`
- `rut`
  - triggered only when the weekly aggregate is clearly negative and a declared core value appears among the main tensions
- `stable`
  - default when neither of the above conditions are met

This is still only a stopgap for the POC.
It should be removed or bypassed once the Drift Detection Engine is implemented upstream.

## Prompt Contract

The Coach prompt has been updated to match the intended behavior more closely.

### Current Requirements Enforced In Prompt

- reflective, not prescriptive
- no score jargon
- no gamification
- no judgmental language
- no micro-habit or action-plan output
- use quoted evidence where possible
- return strict JSON with:
  - `weekly_mirror`
  - `tension_explanation`
  - `reflective_question`

### Important Change

The old prompt asked for a `micro-anchor action`.
That has been removed because it conflicted with the reflective-only Coach design in `docs/weekly/comments.md` and `docs/vif/example.md`.

## Validation

Tier 1 validation is now implemented for Coach narratives.

### Checks

- `groundedness`
  - requires at least one quoted phrase that appears in journal history or evidence
- `non_circularity`
  - fails if the narrative falls back to score/alignment jargon instead of reflective language
- `length`
  - checks total narrative word count against configured bounds in code

### Notes

This validation is intentionally simple and conservative.
It is a first-pass guardrail, not a claim that explanation quality is solved.

## Recent Output Review

Two generated markdown digests were used as sanity checks after the latest semantic cleanup:

- `logs/exports/weekly_digests/0a2fe15c_2025-12-09.md`
- `logs/exports/weekly_digests/0ad04582_2026-01-01.md`

### What Improved

- grief-heavy weeks no longer surface obviously misleading tensions like `Hedonism` as the primary weekly takeaway
- the acute hospital-loss week now routes to `high_uncertainty`, which is much closer to the intended Coach behavior
- the same dimension no longer appears in both `Tensions` and `Strengths` by default
- empty tension sections are now rendered explicitly as `None clear this week`
- backfilled digests no longer leak future entries into the prompt/history context
- weeks with positive overall alignment but softer burden can now route to `background_strain` instead of being flattened to `stable`
- weeks with within-week polarity shifts can now route to `mixed_state` even when simple weekly means would hide that pattern

### What Is Still Not Ideal

- the current logic is intentionally conservative, so some weeks now render `None clear this week` even when a human might describe a subtler mixed-state tension
- evidence selection is still driven by label aggregates rather than richer narrative patterning
- the acute-distress heuristic is lexical and narrow; it is useful as a stopgap but not a real uncertainty model
- `background_strain` evidence can currently render with `dims=none`, which is acceptable for debugging but not ideal as a final user-facing abstraction

### Current Interpretation

At this stage, the markdowns are useful as structured debug artifacts and prompt inputs.
They are materially better than the earlier outputs, but they are still not final user-facing Coach results.

## Persistence

The weekly digest flow now writes:

- `logs/exports/weekly_digests/<persona>_<week_end>.json`
- `logs/exports/weekly_digests/<persona>_<week_end>.md`
- `logs/exports/weekly_digests/<persona>_<week_end>.prompt.txt`
- `logs/exports/weekly_digests/weekly_digests.parquet`

The parquet row stores structured fields as JSON strings where needed.
This is enough for POC-scale querying and later anomaly or pattern analysis.

## Current Status

### Working

- structured weekly digest building
- explicit drift-result input contract
- runtime VIF inference from trained checkpoints
- uncertainty-gated crash/rut-style drift detection experiments
- end-to-end offline weekly Coach runtime entrypoint
- full-history Coach prompt rendering
- structured Coach output contract
- Tier 1 validation helpers
- parquet persistence
- targeted Coach tests

### Still Missing

- learned/calibrated routing for `mixed_state` and `background_strain`
- live production LLM orchestration for the Coach path
- user-facing rating capture for "Did this feel accurate?"
- Tier 2 meta-judge evaluation
- Tier 3 human calibration

## Known Constraints And Tradeoffs

### 1. Current Weekly Signals Are Label-Based

The digest currently builds from judge labels parquet, not live Critic outputs.
This is acceptable for the current offline POC slice but not the final online architecture.

### 2. Response Modes Are Conservative

Without upstream drift detection and uncertainty wiring, the digest currently falls back to a safe reflective mode unless the weekly aggregate clearly shows strain on declared core values.
The newer acute-distress fallback improves safety, but it is still heuristic rather than model-based.
The new `mixed_state` and `background_strain` modes reduce false flattening, but they are also heuristic categorizations rather than learned or calibrated decisions.

### 3. Tier 1 Groundedness Expects Quotes

This biases generation toward quoted evidence.
That is acceptable for now because the bigger risk is ungrounded reflective prose.

### 4. Core Values Matter More Than Raw Extremes

Focus-dimension ranking now prefers declared core values when useful.
This is intentional because a strongly negative non-core dimension is not always the most meaningful Coach topic.

### 5. Conservative Suppression Is Better Than Confident Misframing

The current implementation now prefers `None clear this week` over inventing a dubious tension.
This is a deliberate tradeoff after observing misleading outputs such as grief-heavy weeks being framed primarily through `Hedonism`.
The downside is lower recall for subtle tensions.

### 6. Mixed-State Semantics Are Better Than Mean-Only Semantics

Some weeks contain genuine within-week polarity shifts that disappear when scores are averaged.
The new `mixed_state` fallback partially corrects for this by looking beyond mean-only summaries, but it is still a heuristic layer over judge-label artifacts.

## Requirements For The Full Coach Flow

To finish the whole digest Coach flow cleanly, the next implementation stage should satisfy all of the following:

1. Calibrate the new drift-detector thresholds against synthetic scenario sweeps and held-out personas.
2. Distinguish clearly between:
   - weekly descriptive summary
   - triggered critique or acknowledgement mode
   - uncertainty-driven presence mode
3. Extend the Coach runtime from offline artifact generation to production-facing orchestration.
4. Store generated Coach narratives and validation outcomes in a queryable way that supports evaluation.
5. Add scenario-based tests for:
   - stable alignment
   - rut
   - crash
   - high uncertainty
6. Add user-study instrumentation for perceived accuracy.

If the project later decides to revisit evolution gating, that should be handled as a fresh scope decision rather than assumed as part of this flow.

## Recommended Next Steps

### P1

- Harden the live Critic-output and uncertainty wiring against a wider batch of weekly scenarios.
- Calibrate and simplify the existing drift-detector module, then keep the digest contract narrow and explicit.
- Add scenario tests that verify the correct mode for representative weeks.
- Add scenario fixtures specifically for:
  - grief / bereavement
  - caregiving overload
  - mixed positive week with background strain
  - true stable alignment with no strong tension
  - within-week polarity flips on a declared core value

### P2

- Add a Coach service entrypoint that executes:
  - build digest
  - generate narrative
  - validate narrative
  - persist output
- Add a small evaluation script that reports Tier 1 pass rates over generated weekly digests.
- Add qualitative review tooling that compares:
  - top tensions/strengths
  - selected evidence
  - final Coach narrative
  across a batch of personas
- Improve mode-aware evidence selection so:
  - `high_uncertainty` weeks surface presence-oriented evidence without brittle value labels
  - `background_strain` weeks surface meaningful strain snippets without `dims=none` when possible
  - `mixed_state` weeks show both sides of the polarity clearly

### P3

- Add user feedback capture for digest accuracy.
- Design the Tier 2 meta-judge prompt and evaluation loop.
- Explore a richer tension-ranking layer that blends:
  - declared core values
  - within-week salience
  - recency
  - contradiction/mixed-signal detection
  - uncertainty gates

## TODO

- [ ] Reduce or remove fallback `response_mode` inference once upstream drift output is trusted
- [ ] Calibrate Critic uncertainty thresholds for `high_uncertainty`
- [ ] Support true `crash` detection over multi-week trajectories
- [ ] Add a production-facing Coach service entrypoint beyond the current offline runtime and CLI exports
- [ ] Run Tier 1 validation over a larger batch of digest generations
- [ ] Add scenario fixtures that cover grief/acute-life-event handling conservatively
- [ ] Add scenario fixtures for subtle mixed-state weeks so the digest does not collapse too often to `None clear this week`
- [ ] Decide whether occasional acknowledgement in `stable` mode should be rate-limited upstream
- [ ] Improve evidence selection so a week with no clear tension can still surface important emotionally salient context without pretending it is a scored misalignment
- [ ] Decide whether `high_uncertainty` weeks should show positive strengths at all, or switch to a different markdown/prompt layout entirely
- [ ] Improve `background_strain` evidence so it can map to more meaningful soft-focus dimensions instead of rendering `dims=none`
- [ ] Decide whether `mixed_state` and `background_strain` should become first-class prompt templates rather than only response-mode labels

## Verification

Current targeted verification:

- `pytest tests/coach/test_weekly_digest.py`

That test file currently covers:

- digest construction
- no future-entry leakage past the digest end date
- explicit upstream drift-result override
- acute-distress fallback to `high_uncertainty`
- mixed-state fallback for within-week polarity shifts
- background-strain fallback for softer positive-but-burdened weeks
- suppression of overlapping tension/strength dimensions
- prompt rendering
- structured Coach generation with an injected fake LLM
- Tier 1 validation
- parquet persistence
