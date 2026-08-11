# Twinkl End-to-End Architecture

This document shows the current capstone assessment architecture. The
[PRD](../prd.md) is authoritative for product scope. See the
[VIF Critic architecture](../vif/current_system_architecture.mmd) for detailed
offline training and inference data flow.

![Twinkl end-to-end architecture](e2e_architecture.png)

## Current Status

The Experience and Inspect React App is implemented. Railway serves the React
app and the same-origin Python API for capstone assessment. This deployment is
not deployment approval.

The browser keeps resumable client state. The Python Experience service keeps
session state and trace events in memory. Production authentication,
multi-user storage, and a background schedule are outside the capstone scope.

## Experience Path

The React app owns onboarding, Experience, Inspect, and saved Persona replay.
Onboarding creates the confirmed Profile and its Core Values. Experience accepts
manual Journal Entries and shows nudges, Weekly Drift Detection results, and
Coach Digest responses.

The React app sends versioned requests to the same-origin Python API. The Python
Experience service validates the Profile, processes Journal Entries, selects
closed weeks, stores results, and creates trace events for Inspect.

For each eligible Journal Entry, a deterministic rule first decides whether to
suppress a nudge. If a model call is allowed, one `gpt-5.6-luna` call with
reasoning effort `none` decides whether to show a nudge and can create its
question. A displayed nudge must be answered or skipped before its week is
eligible for review.

Saving a Journal Entry does not review an open week. A Monday-to-Sunday week is
eligible only after Sunday. The React assessment clock starts this work in the
capstone app. A production background schedule is not implemented.

## Weekly Drift Detection

The Weekly Drift Reviewer receives cumulative displayed Journal Entry history
and the Profile Core Values. Its fixed contract is `gpt-5.6-luna` with reasoning
effort `low`. It does not receive VIF Critic Predictions.

The Weekly Drift Reviewer returns one Weekly Drift Reviewer Decision for each
current Journal Entry and Core Value. Invalid, refused, or failed responses
produce Abstain decisions. This fail-closed behavior prevents an unsupported
Drift claim.

The Drift Detector applies one deterministic rule: two consecutive Conflicts
for the same Core Value form Drift. It stores Active Drift, No Active Drift, or
Insufficient Evidence as the current state for each Core Value. It stores each
confirmed past Drift as a Historical Drift Record.

Weekly Drift Detection stores structured output with Core Values, cited Journal
Entries, and Drift state. The Coach Digest runs after every stored result,
including No Active Drift. Coach Digest Validations check cited text,
restricted terms, and response length. If no valid response is available, the
Weekly Drift Detection output remains available.

Inspect reads the same Profile, Journal Entries, Weekly Drift Reviewer
Decisions, Drift state, Weekly Drift Detection output, and trace events as
Experience. Inspect shows the source and model contract for saved and live work.
It does not expose provider credentials.

## Saved Persona Replay

Saved Persona replay loads committed scenario bundles into the shared React
session. It uses saved Weekly Drift Reviewer Decisions by default. The app marks
each result as saved or live and verifies its recorded source data.

Saved Persona replay is an assessment input. It is not the only Experience
input because manual Journal Entries use the same React and Python contract.

## Offline Research

Synthetic persona generation creates coherent Journal Entries for offline
research. Wrangling removes generation metadata before the LLM-Judge creates
LLM-Judge VIF Labels. These labels train and evaluate the VIF Critic (Offline).

The VIF Critic (Offline) produces ten-value VIF Critic Predictions with
uncertainty for experiment reports. It does not produce user-facing Drift. The
historical crash, rut, and evolution compatibility path is deprecated and is
not shown in the diagram.

Human annotations and multi-pass LLM-Judge work remain evaluation evidence.
AI-reviewed synthetic development evidence is not human validation or
deployment approval.

## Detailed References

- [Experience and Inspect React App](../demo/experience_inspect_app.md)
- [Onboarding Specification](../onboarding/onboarding_spec.md)
- [Weekly Drift Detection](../weekly/weekly_drift_detection.md)
- [VIF Critic Capstone Scope](../vif/05_capstone_scope_decision.md)
- [VIF Critic Training](../vif/03_model_training.md)
- [Coach Digest Validations and Evals](../evals/coach_narrative_test_and_eval_guide.md)
