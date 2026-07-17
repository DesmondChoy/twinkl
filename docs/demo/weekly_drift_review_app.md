# Drift Inspection App

This read-only Python Shiny app compares Runs 1–3 for three frozen Weekly Drift
Reviewer setups without merging Runs or calculating a majority vote.

The desktop review flow has three stages:

1. Read the first-page At a glance section and shared How it works contract,
   then filter persona/Core Value cases by known Drift status and Core Value.
2. Choose a matching persona from the filtered summary list. Inspection badges
   use all three Runs of Luna at reasoning low. They identify missed known Drift,
   false Drift alerts, Run variability, experiment setup disagreement, unresolved
   because of Abstain, invalid Weekly Drift Reviewer responses, and Uncertain
   LLM-Judge Conflict Labels; one case can have several badges.
3. Read the persona scoreboard, then inspect that persona's Journal Entries,
   LLM-Judge Conflict Labels, and preserved Runs.

The interface is desktop-only and requires a viewport at least 1120 pixels
wide. Narrower screens use horizontal scrolling.

At a glance appears on the first page before any filter is applied. Dataset
shows how 204 synthetic personas become 292 review cases, each pairing one
persona with one Core Value. The data includes 36 cases with known Drift, 256
with no known Drift, and 42 known Drifts.
Its collapsed detail adds Journal Entry, persona-week, label, timing, historical
VIF Critic training provenance, and Core Value counts. LLMs used briefly names
the role of each Weekly Drift Reviewer and LLM-Judge model and links to its
official model page. Results reports known Drifts found, Drift recall, false
Drift alerts, and Drift precision for all three Weekly Drift Reviewer setups
and Runs, plus median Drift recall for each setup. Coverage remains in the
model-selection disclosure rather than the summary table.

The persona scoreboard then states which Runs found or missed known Drift and
names the exact known Drift and Drift alert spans. The Journal Entry table keeps
the LLM-Judge Conflict Label in a dedicated column so the label and each Weekly
Drift Reviewer Decision can be compared directly. Biographies remain collapsed
because they were not Weekly Drift Reviewer input.

The Filter screen defines what does and does not count as Conflict before it
states the two-consecutive-Conflict Drift rule. Journal Entry comparisons show
Weekly Drift Reviewer Decisions and cited evidence without uncalibrated
confidence tiers or raw reason codes. Abstain receives a plain-English
explanation.

Runs 1–3 repeat the same frozen setup on the same input and are never merged;
their disagreement is shown as Run variability. Luna at reasoning low is
identified as the fixed Weekly Drift Reviewer model contract, with its
selection hierarchy and development uncertainty disclosed beside the corpus
overview.

A Drift alert counts as a hit when it is confirmed between the known Drift's
first Journal Entry and two Journal Entries after its end. This affects
scoring, not the Drift definition. The Journal Entry comparison initially
shows the first known Drift, or the first Drift alert when no known Drift
exists; Full timeline remains available.

A cross-week Drift spans two review weeks, so its second Conflict is not
assessed until the next weekly review. Across these development Runs,
cross-week detection was about four days slower than same-week detection.

The universal Weekly Drift Reviewer input contract appears once on Filter as a
two-part, step-controlled animation. Synthetic development first shows the
isolated LLM-Judge Conflict Label lane, the Weekly Drift Reviewer input
boundary, and the later development comparison. Intended deployed flow shows new
Journal Entries, the verified weekly cutoff, Weekly Drift Reviewer Decisions,
and the deterministic Drift Detector rule. It is explicitly marked as not yet
deployment-approved. The complete static contract and LLM-Judge Conflict Label
provenance remain available below the animation, while Journal Entries keeps
only persona-specific verified weekly cutoffs in a collapsed evidence drawer.

The sequence plays once when it enters the viewport. Previous, next, direct-step,
pause, replay, and two-part controls remain available. Reduced-motion
preferences disable autoplay and spatial motion without hiding any explanation.

The app makes no model or provider API calls. It reads committed research files
only and keeps invalid responses fail-closed.

## Local launch

```sh
uv run shiny run --host 127.0.0.1 --port 8000 --no-dev-mode \
  src/drift_review_app/app.py
```

Open `http://127.0.0.1:8000`; the main review page opens immediately.

`--host` sets the bind address, `--port` sets the local port, and
`--no-dev-mode` disables development reload behavior. The `drift-review-app`
entry in `.claude/launch.json` runs this exact command.

## Local Docker launch

```sh
docker build -f Dockerfile.review_app -t twinkl-drift-review .
docker run --rm -p 8000:8000 -e PORT=8000 twinkl-drift-review
```

Open `http://127.0.0.1:8000`. The container includes only the app code,
required parsers, registered configurations, and frozen review inputs.

## Weekly Drift Reviewer input boundary

The app reconstructs the exact boundary recorded in each frozen weekly prompt:

- The Weekly Drift Reviewer receives all declared Core Values and cumulative
  displayed Journal Entries through the review week. The current week's Journal
  Entries are marked for assessment.
- Displayed nudge and response text is included when present.
- The Weekly Drift Reviewer does not receive the persona biography, later
  Journal Entries, AI-reviewed LLM-Judge Conflict Labels, known Drift, VIF
  Critic predictions, or another setup's decisions.

The loader verifies prompt hashes, weekly cutoffs, Journal Entry text, declared
Core Values, and the empty VIF Critic input block before the page renders. The
Journal Entries screen exposes each persona-specific cutoff in a collapsed
evidence drawer. This demonstrates the intended inference-time boundary; it is
not deployment approval.

## Railway launch

`railway.json` selects `Dockerfile.review_app`. The image installs only Shiny,
Polars, and PyYAML, then copies the app plus the frozen review inputs. It needs
no database or persistent volume.

1. Create a Railway service from this repository.
2. Deploy from the connected branch or run `railway up`.
3. Generate a public domain in Railway after the deployment becomes healthy.

Railway supplies `PORT`; the container binds Shiny to `0.0.0.0:$PORT`.

## Frozen inputs

The app also verifies joins, setup identities, requested and resolved model
identifiers, counts, and per-Run aggregate parity before rendering research
data. Reasoning effort is verified from each frozen manifest and registered
configuration because individual response receipts do not record it.

Inputs are read from:

- `logs/wrangled/persona_*.md`
- `logs/experiments/artifacts/twinkl_52zz_model_comparison_20260714/`
- `logs/experiments/artifacts/twinkl_52zz_luna_low_20260714/`
- `logs/experiments/artifacts/twinkl_qtwz_complete_development_review_20260714/results/`
- `config/evals/twinkl_52zz_model_comparison_v1.yaml`
- `config/evals/twinkl_52zz_luna_low_v1.yaml`

These are AI-reviewed synthetic development inputs, not human validation or a
fresh final test. The app does not provide deployment approval or product
runtime wiring. Those limits do not reopen the fixed Luna-low model choice.

The LLM-Judge Conflict Labels were produced by two isolated `gpt-5.6-sol`
lanes at reasoning effort `xhigh`, with disagreement-only adjudication. Four
earlier Uncertain labels were separately reviewed with `claude-opus-4-8` at
reasoning effort `high`. Known Drift is derived from these AI-reviewed labels;
it is not ground truth or human validation.
