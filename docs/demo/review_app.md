# Twinkl Demo App

This document describes the user-facing Shiny demo app for the approved
capstone POC architecture: onboarding with Core Values, Journal Entry input,
Weekly Drift Reviewer Decisions, and the deterministic Drift Detector. There is
no VIF Critic input on this path; the VIF Critic remains offline for review and
retraining.

The separate [Drift Inspection App](weekly_drift_review_app.md) is the
read-only reviewer surface: it compares frozen Weekly Drift Reviewer Runs
without making model or provider API calls. This demo app is the interactive
counterpart and does make paid model calls.

## What It Does

- Onboarding: the user completes the published SVBWS values assessment (11
  balanced-incomplete-block-design groups of six cards, tap-to-pick Most/
  Least), then structured goal selection, yielding 1-2 Core Values (only Core
  Values can produce Drift) and an optional display name.
- Demo shortcut: instead of onboarding manually, the user can preload a
  synthetic persona from `logs/wrangled/`. That fills in the persona's Core
  Values (capped at 2) and its Journal Entries, including displayed nudges and
  responses.
- Journal: the user writes dated Journal Entries in the app. Entries must stay
  in date order. Saving an entry may surface a nudge (a short follow-up
  question) before the entry is finalized.
- Weekly review: runs automatically whenever the journal changes (a new entry
  saves, one is removed, a demo persona loads) — no button press needed. The
  app groups Journal Entries into calendar weeks and calls the Weekly Drift
  Reviewer once per week with cumulative history. Each Journal Entry gets a
  Weekly Drift Reviewer Decision of Conflict, Not Conflict, or Abstain per
  Core Value. On failure (e.g. a quota error), a Retry action appears
  contextually instead of a permanent control.
- Drift: the Drift Detector applies the two-consecutive-Conflict rule per Core
  Value; the app shows an ambient per-Core-Value Drift snapshot (stable,
  active, recovered, uncertain) without the underlying rationale.
- Weekly Digest: refreshes automatically whenever the weekly review produces a
  new outcome, with a Refresh button for an on-demand re-run. Produces the
  Weekly Coach reflection when a provider key is configured.
- Internal/debug (QA only): a collapsed section at the bottom of the journal
  page holds the raw Drift rationale (entry indices, timestamps, evidence
  quotes) and the Weekly Drift Reviewer run audit trail — not part of the
  product surface, isolated so it can be deleted cleanly.

## Model Calls and Cost

The Weekly Drift Reviewer contract is fixed at `gpt-5.6-luna` with reasoning
effort `low`. Running the weekly review makes one paid call per persona-week.
Weekly receipts are cached in the session by prompt hash, so re-running after
new entries only pays for new or changed weeks. Nothing is persisted to disk;
closing the session discards the journal and its receipts.

`OPENAI_API_KEY` must be set in `.env`. Without it the app loads and accepts
Journal Entries but refuses to run the weekly review.

## How To Run

```sh
uv run shiny run src/demo_tool/app.py
```

Open `http://127.0.0.1:8000`.

To launch the same app directly with Python on port `8001`:

```sh
uv run python src/demo_tool/app.py
```

## Inputs

- `.env` with `OPENAI_API_KEY` (required for the weekly review)
- `logs/wrangled/persona_*.md` (optional; only needed for the demo persona
  shortcut)

## UI Flow

1. Onboarding: the SVBWS values assessment, goal selection, a Core Value
   summary, and a first Journal Entry handoff — or load a demo persona
   instead.
2. Journal stage: a persistent left rail shows identity only (name, Core
   Value badges, "Start over") and a live status banner; there is no manual
   "run review" control. The main canvas leads with the entry composer, then
   the user's own journal, then an ambient Drift snapshot, then the Weekly
   Digest. A collapsed "Internal / debug (QA only)" section at the bottom
   holds the raw Drift rationale and the Weekly Drift Reviewer run table.
3. The weekly review and the Weekly Digest both refresh automatically as
   entries change; a Retry action appears in the status banner only after a
   failure, and a Refresh button lives in the Weekly Digest panel for an
   on-demand re-run.

## Implementation Reference

Key modules:

- `src/demo_tool/app.py` — Shiny interface and session flow
- `src/demo_tool/runtime_bridge.py` — journal review path:
  `build_weekly_drift_reviewer_request` → `OpenAIWeeklyDriftReviewer` →
  `detect_drift`, with the per-week receipt cache
- `src/demo_tool/data_loader.py` — synthetic persona catalog for the demo
  shortcut
- `src/demo_tool/state.py` — session state (stage, Core Values, entries,
  review outcome)

The file-based runtime for wrangled personas remains
`src.coach.weekly_drift_runtime.run_weekly_drift_coach_cycle()`; this app uses
the same Weekly Drift Reviewer and Drift Detector components over in-session
input instead of persona files. `src/demo_tool/multi_drift.py` is retained for
exploratory scripts but is no longer wired into the app.
