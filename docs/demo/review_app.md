# Runtime Demo Review App

This document describes the Shiny review app used to inspect Twinkl's local
end-to-end runtime on top of existing wrangled personas and saved VIF Critic
checkpoints.

This app executes the experimental VIF Critic-to-Weekly-Digest path. The
separate [Drift Inspection App](weekly_drift_review_app.md) is read-only and
compares frozen Weekly Drift Reviewer Runs without executing the VIF Critic
runtime or making model or provider API calls.

The app is a review and debugging interface for the current POC. It does not
change product scope. It makes the existing runtime files easier to inspect
and compare.

## What It Does

The app lets you:

- browse a persona's profile, Core Values, and full Journal Entry timeline
- inspect nudges and follow-up responses inline with the originating Journal Entry
- choose a local VIF Critic checkpoint from the experiment archive or `models/vif/`
- run the full local checkpoint -> weekly signals -> Drift -> Weekly Digest path
- reload cached results for a previously-run persona/checkpoint pair
- compare six rule-based detectors against either LLM-Judge labels or VIF Critic predictions
- inspect an Overview plus the resulting per-Journal-Entry signals, weekly aggregates, Drift JSON, Weekly Digest, and detector comparison in one place

## How To Run

The standard entrypoint uses Shiny's app runner:

```sh
uv run shiny run src/demo_tool/app.py
```

Open `http://127.0.0.1:8000`.

To launch the same app directly with Python:

```sh
uv run python src/demo_tool/app.py
```

Open `http://127.0.0.1:8001`.

If another process already uses port `8001`, the direct Python launcher exits
instead of terminating that process. Use the Shiny runner or free the port
before retrying.

## Inputs

### Required local data

- `logs/wrangled/persona_*.md`
- local VIF Critic checkpoints discoverable under:
  - `logs/experiments/artifacts`
  - `models/vif`
  - `logs/experiments`

### Detector sources

The detector comparison panel supports two input sources:

- `Judge labels` (LLM-Judge labels)
  - reads `logs/judge_labels/judge_labels.parquet`
  - uses persisted single-pass labels, not the five-pass consensus reference
  - works without running the VIF Critic first
- `Critic predictions` (VIF Critic predictions)
  - reads continuous per-Journal-Entry mean predictions from a cached or freshly completed runtime timeline
  - requires a checkpoint-backed runtime bundle

Neither source supplies rolling ordinal `P(-1)` evidence, so the app does not
run the selected Drift Detector.

## UI Flow

### Controls

The persistent, collapsible left rail controls:

- persona selection
- checkpoint selection
- detector input source (`Judge labels` or `Critic predictions`)
- catalog refresh
- live runtime execution

### Results canvas

The main canvas shows:

- persona metadata
- checkpoint summary
- collapsible bio
- Core Value badges
- the full journal timeline with nudge and response threads
- six result tabs:
  - `Overview` — Drift status, triggered dimensions, strengths, and tensions
  - `Per-entry critic` — per-Journal-Entry alignment means, uncertainties, strongest alignments, and detector badges
  - `Weekly signals` — weekly aggregate table from the runtime timeline
  - `Drift` — structured Drift JSON plus triggered dimensions
  - `Weekly digest` — rendered Weekly Digest markdown and summary badges
  - `Detector comparison` — chart, table, and fired/silent chips for all six detector families


## Detector Comparison

The app renders all six rule-based detectors from the notebook evaluation path:

- `Baseline`
- `EMA`
- `CUSUM`
- `Cosine`
- `Control Chart`
- `KL Div`

The chart overlays detector alerts on top of the alignment trajectories. The
table shows per-Journal-Entry detector votes plus a detector-vote count for each step.
That count is agreement among the six exploratory methods; it is not the
five-pass LLM-Judge consensus parquet and is not benchmark ground truth.

The comparison remains a diagnostic view. Drift requires two adjacent Journal
Entries that visibly show a behavior or choice against the same Core Value.
[`twinkl-v8pb`](../evals/drift_v1_student_visible_target.md) completed
the student-visible target and locked final test review. It is not wired into
this review app or the Weekly Coach runtime because the development score was
weak and one final test case remained unresolved; the old consensus-derived frozen
benchmark is retired historical evidence.

## Generated Outputs

Each persona/checkpoint pair maps to a stable output directory:

```text
logs/exports/demo_tool_runs/<persona_id>/<checkpoint-stem>-<hash>/
```

That directory stores:

- `<persona_id>_vif_timeline.parquet`
- `<persona_id>_vif_weekly.parquet`
- `<persona_id>_<week_end>.drift.json`
- `<persona_id>_<week_end>.json`
- `<persona_id>_<week_end>.md`
- `<persona_id>_<week_end>.prompt.txt`

The consolidated Weekly Digest parquet remains separate at the configured
`parquet_path` (default: `logs/exports/weekly_digests/weekly_digests.parquet`).

## Implementation Reference

Key modules:

- `src/demo_tool/app.py`
- `src/demo_tool/data_loader.py`
- `src/demo_tool/runtime_bridge.py`
- `src/demo_tool/multi_drift.py`
- `src/demo_tool/state.py`

The app uses `src.coach.runtime.run_weekly_coach_cycle()` for live execution,
then reads the persisted files back into the UI. It renders the Weekly Digest
and Weekly Coach prompt but does not inject a live Weekly Coach LLM or generate
a live Weekly Coach output.
