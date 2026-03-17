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

## Scope For The Current POC

The current POC does not need RAG.
The synthetic corpus is small enough that the Coach can use full-context prompting over the entire journal history.

The current POC also does not have production-ready drift detection.
That means:

- crash/rut/high-uncertainty formulas remain a downstream dependency
- the current weekly digest implementation uses a conservative fallback mode assignment
- the prompt and output schema are ready for richer trigger wiring later

## Current Implementation

### Implemented In Code

Primary code lives in:

- `src/coach/weekly_digest.py`
- `src/coach/schemas.py`
- `prompts/weekly_digest_coach.yaml`

The current implementation now supports:

1. Building a structured weekly digest from `logs/judge_labels/judge_labels.parquet`
2. Loading the full wrangled journal history for the persona from `logs/wrangled/persona_<id>.md`
3. Carrying declared core values into the digest payload
4. Selecting focus tensions/strengths with core-value-aware ranking
5. Selecting representative evidence snippets from the scored week
6. Assigning a conservative response mode for the Coach (`stable` or `rut` today)
7. Rendering a full-context Coach prompt with:
   - full journal history
   - declared core values
   - value elaborations from `config/schwartz_values.yaml`
   - dimension summaries
   - evidence snippets
   - response mode metadata
8. Generating structured Coach output through an injected LLM callable
9. Running Tier 1 automated narrative checks:
   - groundedness via quoted substring matches
   - non-circularity via score-jargon avoidance
   - length bounds
10. Persisting each weekly digest into a consolidated parquet artifact for future longitudinal analysis
11. Exporting per-run JSON, markdown, and prompt artifacts

### Current Digest Data Model

`WeeklyDigest` now carries:

- persona and window metadata
- response mode, mode source, and mode rationale
- weekly aggregate metrics
- declared core values
- per-dimension summaries
- representative evidence snippets
- full journal history
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

### Flow

1. Load persona labels for the requested week.
2. Resolve the digest window.
3. Load the full persona profile and journal history from the wrangled file.
4. Compute per-dimension weekly means and class proportions.
5. Rank top tensions and strengths, with declared core values treated as first-class context.
6. Select representative misaligned and aligned evidence rows.
7. Assign a fallback response mode.
8. Render the Coach prompt using full history plus structured summaries.
9. Optionally call the LLM with the structured Coach JSON schema.
10. Optionally run Tier 1 validation.
11. Persist both per-run exports and a consolidated parquet record.

### Response Modes

Supported schema modes:

- `stable`
- `rut`
- `crash`
- `high_uncertainty`

Current implementation status:

- `stable`: implemented
- `rut`: implemented via fallback heuristic
- `crash`: schema-ready, not inferred yet
- `high_uncertainty`: schema-ready, not inferred yet

This is deliberate.
The digest contract is ready for full trigger wiring without pretending the current Critic-side logic already exists.

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
- full-history Coach prompt rendering
- structured Coach output contract
- Tier 1 validation helpers
- parquet persistence
- targeted Coach tests

### Still Missing

- real Critic `predict_with_uncertainty()` integration
- true crash/rut detection from trajectory formulas
- true high-uncertainty routing
- live production LLM orchestration for the Coach path
- user-facing rating capture for "Did this feel accurate?"
- Tier 2 meta-judge evaluation
- Tier 3 human calibration

## Known Constraints And Tradeoffs

### 1. Current Weekly Signals Are Label-Based

The digest currently builds from judge labels parquet, not live Critic outputs.
This is acceptable for the current offline POC slice but not the final online architecture.

### 2. Response Modes Are Conservative

Without drift detection and uncertainty wiring, the digest defaults to a safe reflective mode unless the weekly aggregate clearly shows strain on declared core values.

### 3. Tier 1 Groundedness Expects Quotes

This biases generation toward quoted evidence.
That is acceptable for now because the bigger risk is ungrounded reflective prose.

### 4. Core Values Matter More Than Raw Extremes

Focus-dimension ranking now prefers declared core values when useful.
This is intentional because a strongly negative non-core dimension is not always the most meaningful Coach topic.

## Requirements For The Full Coach Flow

To finish the whole digest Coach flow cleanly, the next implementation stage should satisfy all of the following:

1. Replace fallback mode assignment with actual drift-detector outputs.
2. Feed real Critic means and uncertainties into the digest.
3. Distinguish clearly between:
   - weekly descriptive summary
   - triggered critique or acknowledgement mode
   - uncertainty-driven presence mode
4. Add a proper Coach orchestration entrypoint under `src/coach/`.
5. Store generated Coach narratives and validation outcomes in a queryable way that supports evaluation.
6. Add scenario-based tests for:
   - stable alignment
   - rut
   - crash
   - high uncertainty
7. Add user-study instrumentation for perceived accuracy.

## Recommended Next Steps

### P1

- Wire the weekly digest to live Critic outputs and uncertainty vectors.
- Implement drift detector logic as a separate module and pass its mode into the digest.
- Add scenario tests that verify the correct mode for representative weeks.

### P2

- Add a Coach service entrypoint that executes:
  - build digest
  - generate narrative
  - validate narrative
  - persist output
- Add a small evaluation script that reports Tier 1 pass rates over generated weekly digests.

### P3

- Add user feedback capture for digest accuracy.
- Design the Tier 2 meta-judge prompt and evaluation loop.

## TODO

- [ ] Replace fallback `response_mode` inference with drift-detector output
- [ ] Integrate Critic uncertainty and support `high_uncertainty`
- [ ] Support true `crash` detection over multi-week trajectories
- [ ] Add a real Coach runtime entrypoint instead of only library helpers and CLI exports
- [ ] Run Tier 1 validation over a larger batch of digest generations
- [ ] Add scenario fixtures that cover grief/acute-life-event handling conservatively
- [ ] Decide whether occasional acknowledgement in `stable` mode should be rate-limited upstream

## Verification

Current targeted verification:

- `pytest tests/coach/test_weekly_digest.py`

That test file currently covers:

- digest construction
- prompt rendering
- structured Coach generation with an injected fake LLM
- Tier 1 validation
- parquet persistence
