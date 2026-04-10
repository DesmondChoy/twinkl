# Demo Review App

This document describes the Shiny review app used to inspect Twinkl's local
end-to-end runtime on top of existing wrangled personas and saved Critic
checkpoints.

The app is a review and debugging surface for the current POC. It does not
change product scope. It makes the existing runtime artifacts easier to inspect
and compare.

## What It Does

The app lets you:

- browse a persona's profile, core values, and full journal timeline
- inspect nudges and follow-up responses inline with the originating entry
- choose a local checkpoint from the experiment archive or `models/vif/`
- run the full local checkpoint -> weekly signals -> drift -> digest path
- reload cached results for a previously-run persona/checkpoint pair
- compare six rule-based drift detectors against either judge labels or Critic predictions
- inspect the resulting per-entry signals, weekly aggregates, drift payload, and weekly digest in one place

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

## Inputs

### Required local data

- `logs/wrangled/persona_*.md`
- local Critic checkpoints discoverable under:
  - `logs/experiments/artifacts`
  - `models/vif`
  - `logs/experiments`

### Detector sources

The detector comparison panel supports two input surfaces:

- `Judge labels`
  - reads `logs/judge_labels/judge_labels.parquet`
  - works without running the Critic first
- `Critic predictions`
  - reads the per-entry runtime timeline from a cached or freshly completed run
  - requires a checkpoint-backed runtime bundle

## UI Flow

### Controls

The left panel controls:

- persona selection
- checkpoint selection
- detector input source (`Judge labels` or `Critic predictions`)
- catalog refresh
- live runtime execution

### Center panel

The center panel shows:

- persona metadata
- collapsible bio
- core-value badges
- the full journal timeline with nudge and response threads

### Results panel

The right panel exposes five tabs:

- `Per-entry critic` — per-entry alignment means, uncertainties, strongest alignments, and detector badges
- `Weekly signals` — weekly aggregate table from the runtime timeline
- `Drift` — structured drift payload plus triggered dimensions
- `Weekly digest` — rendered markdown digest and summary badges
- `Detector comparison` — chart and table for all six detector families

## Detector Comparison

The app renders all six rule-based detectors from the notebook evaluation path:

- `Baseline`
- `EMA`
- `CUSUM`
- `Cosine`
- `Control Chart`
- `KL Div`

The chart overlays detector alerts on top of the alignment trajectories. The
table shows per-entry detector votes plus a consensus count for each step.

## Generated Artifacts

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

The consolidated weekly digest parquet remains separate at the configured
`parquet_path` (default: `logs/exports/weekly_digests/weekly_digests.parquet`).

## Implementation Reference

Key modules:

- `src/demo_tool/app.py`
- `src/demo_tool/data_loader.py`
- `src/demo_tool/runtime_bridge.py`
- `src/demo_tool/multi_drift.py`
- `src/demo_tool/state.py`

The app uses `src.coach.runtime.run_weekly_coach_cycle()` for live execution,
then reads the persisted artifacts back into the UI.
