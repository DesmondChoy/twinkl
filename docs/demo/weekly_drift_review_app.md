# Drift Inspection App

This read-only Python Shiny app inspects the synthetic corpus and its Drift
characteristics. Report prompt names v0, v1, and v2 identify the preserved
experiments; the latest GPT-6 Luna and historical results remain available as
context for examining individual cases. The interface shows one selected setup
and Run rather than a model comparison dashboard.

The review flow has three stages:

1. Select a prompt, preserved reviewer setup, and Run. Use the existing known
   Drift and Core Value filters, or leave them on all cases and all values.
   Inspect the clickable corpus, Core Value, and Drift breakdowns.
2. Select a count to open its exact contributing Personas, Journal Entries,
   cases, or Drift spans. The list states its unit and filter scope and provides
   pagination so every contributing record is accessible.
3. Open the relevant Core Value case or Journal Entry. The inspector opens on
   the selected entry or Drift span and preserved Run, with reference labels,
   decisions, cited evidence, and verified input cutoffs.

Select the top-left **Drift inspection app** title from any view to return to
the landing page with the selected prompt, reviewer setup, Run, and filters.

Counts describe actual source records. A Persona can contribute more than one
Core Value case; one case can contain several Drifts. Deduplicated Persona and
Journal Entry views retain their associated Core Value choices. Reference Drift
counts, same-week and cross-week spans, and selected-Run found, missed, false,
covered, unresolved, invalid, and Abstain breakdowns lead to their source
records. A median across Runs is not presented as an inspectable cohort.

The default setup is GPT-6 Luna at reasoning effort `low` with report prompt v2.
Selecting another preserved setup changes the decisions under inspection while
retaining the same synthetic corpus and AI-reviewed reference labels. The
Core Value definition study remains detection-only: it exposes detected spans
and decisions without claiming an accuracy evaluation.

Desktop and laptop screens are the primary review target. Narrower screens
stack the controls and keep wide result tables in scrollable sections.
The report's Figure 10 matching diagram remains in the collapsed methods
explanation. Static model comparison figures, cost tables, and model-selection
summaries are excluded from the inspection interface.

## Prompt versions and experiments

The visible names follow the submitted report. Original receipts and hashes
retain their archived version identifiers:

| Report prompt | Archived prompt | Change | Frozen experiments |
|---|---|---|---|
| v0 | `2.0` | Cumulative Journal Entries, Core Value names, clear behavior or choices, and exact cited text | GPT-5.4 mini at `none`; GPT-5.6 Luna at `none`, `low`, `medium`, `high`, and `xhigh` |
| v1 | `3.0` | Tightened confidence, reason, and exact-quote rules; low-confidence decisions become Abstain | Definitions-off GPT-5.6 Luna at `low` in the paired Core Value definition study |
| v2 | `4.0` | Adds each selected Core Value's definition and core motivation | Definitions-on study; matched GPT-5.6 Luna and GPT-6 Luna comparisons at `none`, `low`, `medium`, `high`, and `xhigh` |

Across these studies, the displayed Journal Entries, verified weekly cutoffs,
Core Values, two-consecutive-Conflict rule, and three-Run protocol are preserved.
The accuracy-evaluated experiments use the same AI-reviewed reference labels
and matching window. Prompt versions change the review instructions, so a cross-version
result is not an isolated model comparison. The setup selector identifies the
study as well as the model and reasoning effort.
Reviewer options group GPT-5.6 Luna before GPT-6 Luna and order each model's
reasoning effort as `none`, `low`, `medium`, `high`, then `xhigh`. Separate
definition-study results remain beside the other `low` options.

The case scoreboard states which preserved Runs of the selected setup found
or missed known Drift and names the exact known Drift and Drift alert spans. The Journal Entry trajectory
stacks one date-aligned plot per Run and uses the same circular point shape in
all three plots. Green points match a resolved LLM-Judge Conflict Label; red
pulsing points mark any different Weekly Drift Reviewer Decision, including
Abstain; amber hollow points mark invalid responses or unresolved LLM-Judge
Conflict Labels that cannot be compared. Reduced-motion preferences replace the
pulse with a strong static red halo. For accuracy-evaluated setups, false Drift
alerts and missed known Drift are annotated at the relevant Journal Entry, with
Drift-level annotations spanning the applicable pair. The Core Value definition
study shows detected spans without accuracy classifications. Invalid responses
are annotated for every setup. The inspector opens the selected
Journal Entry, LLM-Judge Conflict Label, and Weekly Drift Reviewer Decision;
selecting another plot point changes the Run. The exhaustive comparison table
remains available in a collapsed disclosure.
Biographies remain collapsed because they were not Weekly Drift Reviewer input.

The Filter screen defines what does and does not count as Conflict before it
states the two-consecutive-Conflict Drift rule. Journal Entry comparisons show
Weekly Drift Reviewer Decisions and cited evidence without uncalibrated
confidence tiers or raw reason codes. Abstain receives a plain-English
explanation.

Runs 1–3 repeat the same frozen setup on the same input and are never merged.
The landing breakdown uses the selected individual Run; the case trajectory
keeps the three Runs available for inspecting variability. GPT-6 Luna at
reasoning effort `low` uses report v2 (archived `4.0`). Historical setups retain
their original inputs and results.

A Drift alert counts as a hit when it is confirmed between the known Drift's
first Journal Entry and two Journal Entries after its end. This affects
scoring, not the Drift definition. Opening a whole case shows its Full timeline.
A contributor link opens its relevant Journal Entry, Drift span, or review week;
the period control can return to the Full timeline or isolate another span.

A cross-week Drift spans two review weeks, so its second Conflict is not
assessed until the next weekly review. In the historical v0 development Runs,
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
configuration. The historical higher-reasoning archive also records it in
individual response receipts, which the loader checks.

Inputs are read from:

- `logs/wrangled/persona_*.md`
- `logs/experiments/artifacts/twinkl_52zz_model_comparison_20260714/`
- `logs/experiments/artifacts/twinkl_52zz_luna_low_20260714/`
- `logs/experiments/artifacts/twinkl_ck3w_luna_higher_reasoning_20260809/`
- `logs/experiments/artifacts/twinkl_j3k7_core_value_definitions_20260907/`
- `logs/experiments/artifacts/twinkl_q6pt_luna_model_20260923/`
- `logs/experiments/artifacts/twinkl_gv3x_luna_higher_20260923/`
- `logs/experiments/artifacts/twinkl_qtwz_complete_development_review_20260714/results/`
- `config/evals/twinkl_52zz_model_comparison_v1.yaml`
- `config/evals/twinkl_52zz_luna_low_v1.yaml`
- The corresponding `twinkl_ck3w`, `twinkl_q6pt`, and `twinkl_gv3x` evaluation configurations

The report matching diagram and extraction provenance are packaged under
`src/drift_review_app/static/report/`. Their metadata records the submitted
PDF's SHA-256, source pages, crop bounds, rendering command, and image hashes.
The deployed app does not need access to the original PDF in Downloads.

These are AI-reviewed synthetic development inputs, not human validation or a
fresh final test. The app does not provide deployment approval or product
runtime wiring. Every setup retains its original prompt, model, study, and
individual Runs, including the historical GPT-5.6 Luna selection evidence.

The LLM-Judge Conflict Labels were produced by two isolated `gpt-5.6-sol`
lanes at reasoning effort `xhigh`, with disagreement-only adjudication. Four
earlier Uncertain labels were separately reviewed with `claude-opus-4-8` at
reasoning effort `high`. Known Drift is derived from these AI-reviewed labels;
it is not ground truth or human validation.
