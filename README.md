# Twinkl

Twinkl is an "inner compass" that helps users compare daily behavior with
confirmed Core Values. Unlike traditional journaling apps that summarize moods and
topics, Twinkl starts with a confirmed Profile, reviews Journal Entries over
time, and shows evidence when behavior conflicts with Core Values. The current
POC keeps the Profile fixed. Profile evolution is future work.

> **Project status:** Twinkl is under active development as an academic capstone. Status markers in each section indicate capability maturity:
> ✅ Complete · 🧪 Experimental · ⚠️ Partial · 📋 Specified · ❌ Not Started
>
> See [Known Gaps](#known-gaps) for the summary and [Implementation Status](docs/prd.md#implementation-status) for the full breakdown.

## Architecture

[![Twinkl end-to-end architecture](docs/architecture/e2e_architecture.png)](docs/architecture/e2e_architecture.md)

## Documentation Guide

- [`docs/prd.md`](docs/prd.md) — authoritative product intent and implementation status
- [`docs/canonical_nouns.md`](docs/canonical_nouns.md) — required Twinkl product terms and evidence-language rules
- [`docs/architecture/e2e_architecture.md`](docs/architecture/e2e_architecture.md) — current Experience, Weekly Drift Detection, and offline research paths
- [`docs/demo/experience_inspect_app.md`](docs/demo/experience_inspect_app.md) and [`frontend/onboarding/README.md`](frontend/onboarding/README.md) — Experience and Inspect behavior, local launch, checks, and assessment deployment
- [`docs/vif/01_concepts_and_roadmap.md`](docs/vif/01_concepts_and_roadmap.md), [`docs/vif/02_system_architecture.md`](docs/vif/02_system_architecture.md), [`docs/vif/03_model_training.md`](docs/vif/03_model_training.md), [`docs/vif/04_uncertainty_logic.md`](docs/vif/04_uncertainty_logic.md) — VIF design, runtime, training, and uncertainty logic
- [`docs/pipeline/pipeline_specs.md`](docs/pipeline/pipeline_specs.md), [`docs/pipeline/data_schema.md`](docs/pipeline/data_schema.md), [`docs/pipeline/consensus_rejudging_instructions.md`](docs/pipeline/consensus_rejudging_instructions.md) — data generation, label datasets, and consensus diagnostics
- [`docs/drift/trajectory_eda.md`](docs/drift/trajectory_eda.md) — historical Drift-definition analysis comparing five-pass LLM-Judge consensus with persisted labels
- [`docs/evals/drift_v1_student_visible_target.md`](docs/evals/drift_v1_student_visible_target.md) — historical five-Drift development result and withheld former final-test score
- [`docs/weekly/weekly_drift_detection.md`](docs/weekly/weekly_drift_detection.md) — Weekly Drift Detection and Coach Digest contracts and runtime CLI
- [`docs/evals/overview.md`](docs/evals/overview.md) and [`docs/evals/coach_narrative_test_and_eval_guide.md`](docs/evals/coach_narrative_test_and_eval_guide.md) — evaluation status, offline checks, paid Coach Digest Evals, and the Drift/control study
- [`docs/demo/weekly_drift_review_app.md`](docs/demo/weekly_drift_review_app.md) — read-only Drift inspection of the frozen Weekly Drift Reviewer comparison Runs
- [`docs/demo/review_app.md`](docs/demo/review_app.md) — deprecated Runtime Demo Review App for the VIF Critic (Offline) compatibility path
- [`docs/capstone_report/capstone_project_report.md`](docs/capstone_report/capstone_project_report.md) and [`docs/capstone_report/capstone_project_report.pdf`](docs/capstone_report/capstone_project_report.pdf) — maintained Phase 2 Technical Paper source and rendered PDF
- [`docs/future_work/README.md`](docs/future_work/README.md) — exploratory directions, including OpenClaw integration research

## Onboarding — 🧪 Experimental React POC

The React app in [`frontend/onboarding/`](frontend/onboarding/) implements the published 11-group, six-object balanced SVBWS design. People tap or drag visually neutral cards into Most and Least boxes before a label-free Core Value summary and first Journal Entry handoff. Group and card order are randomized, raw BWS results remain separate from the ten-value Profile transformation, and there is no midpoint result or unsupported confidence field. This is a research-grounded pilot instrument, not a validated Twinkl instrument.

```sh
cd frontend/onboarding
npm install
npm run dev
```

The POC stores resumable progress and its confirmed Profile in the browser. The
manual Experience synchronizes the confirmed Profile and browser-held
interaction state with the in-memory Python boundary. A separate host can also
persist the Profile exposed by the callback or browser event, and the batch
runtime accepts saved Profile JSON with `--profile-path`. Production
multi-user storage and generalized persistence are outside the time-boxed
capstone.

## Experience and Inspect React App — 🚧 In Progress

The React app also contains the manual Experience, saved Persona replay, and
Inspect. Manual Experience submits Journal Entries to the versioned Python
boundary, supports displayed nudge reply and skip actions, reviews only closed
Monday-through-Sunday weeks, and keeps the Weekly Drift Detection result when
the Coach Digest cannot return a valid response. Inspect reads the same Profile,
Journal Entries, Weekly Drift Reviewer Decisions, Drift state, Coach Digest
response, and trace events.

Saved Persona replay is deterministic and does not require a provider key. The
browser requests the scenario catalog and bundles with `cache: no-store`, then
verifies each bundle against its catalogued SHA-256 hash. Inspect presents both
completed and reused Coach Digest events as available responses. Refused,
invalid, and failed events remain unavailable.

Run the Python boundary from the repository root:

```sh
uv run uvicorn src.demo.api:app --port 8000
```

Run the React development server in a second terminal:

```sh
cd frontend/onboarding
npm install
npm run dev
```

React checks are available through `npm test`, `npm run typecheck`, and
`npm run build`. Regenerate the shared JSON Schema and canonical fixture with
`uv run python -m src.demo.export_contract_schema` after a contract change.
See the [Experience and Inspect guide](docs/demo/experience_inspect_app.md) for
the six operations, assessment deployment, data boundary, and verification
workflow.

The [public assessment](https://onboarding-production-1dd2.up.railway.app/)
serves the React app and same-origin Python boundary. It allows anonymous access
for capstone assessment and can make paid provider calls during manual use. It
is not deployment approval and provides no production authentication,
multi-user storage, or service-level commitment.

## VIF Critic (Offline) — ✅ Complete Capstone Research

The VIF Critic (Offline) training and evaluation stack is complete for the time-boxed
capstone. It compares
Journal Entries with a ten-dimensional value profile. The current user-facing
path uses Weekly Drift Detection instead. No further VIF Critic (Offline) work is
planned.

Key properties:
- **Vector-valued**: Tracks multiple life dimensions simultaneously, preserving trade-offs (e.g., "work goals crushed, but sleep suffered")
- **Uncertainty-aware**: Reports MC Dropout uncertainty for offline analysis
- **Current-entry by default**: The active VIF Critic (Offline) configuration uses `window_size: 1`; trajectory experiments remain diagnostic

The VIF Critic (Offline) research stack includes ordinal MLP heads with MC Dropout, a BNN baseline, config-driven frozen encoders with `nomic-embed-text-v1.5` as the active default, corrected-split experiment logging, checkpoint discovery, recall-first checkpoint selection, raw output export, runtime timeline reconstruction, and weekly aggregation. Its former crash/rut/evolution response routing is deprecated compatibility code. The 69-run / 133-config archive keeps `run_019`-`run_021` Balanced Softmax as the historical corrected-split reference. See [`logs/experiments/index.md`](logs/experiments/index.md) for the live board.

**Current Drift contract:** Drift is two consecutive Conflicts for the same Core Value. Weekly Drift Detection uses the internal Weekly Drift Reviewer and Drift Detector. It stores structured output with Core Values, cited Journal Entries, and Drift state. The fixed model contract is `gpt-5.6-luna` with reasoning effort `low`. The frozen development Runs are AI-reviewed synthetic evidence. They are not human validation, a fresh final test, or deployment approval. The completed VIF Critic (Offline) remains outside this path. See the [`twinkl-52zz` report](logs/experiments/reports/experiment_review_2026-07-14_twinkl_52zz_luna_low.md), [`docs/architecture/drift_detection.md`](docs/architecture/drift_detection.md), and [`docs/evals/drift_detection_eval.md`](docs/evals/drift_detection_eval.md).

**Current runtime behavior:** `src.coach.weekly_drift_runtime` runs Weekly Drift Detection. It reads Journal Entries, imports Core Values and Coach Digest context from a confirmed Profile when supplied, calls the internal Weekly Drift Reviewer, persists versioned JSON receipts, and applies the internal Drift Detector across week boundaries. It stores structured output and renders one of three Coach Digest policies: Drift detected, no current Drift, or more reflection needed. When no Profile is supplied, synthetic personas retain their deterministic `core_values` compatibility path. `src.coach.runtime` and `src.vif.drift` are deprecated compatibility paths for the former VIF Critic (Offline) crash/rut/evolution runtime. See `docs/vif/`, [`docs/weekly/weekly_drift_detection.md`](docs/weekly/weekly_drift_detection.md), and [`docs/demo/review_app.md`](docs/demo/review_app.md).

### Automated Experiment Logging & Review

The experiment log tracks VIF Critic (Offline) training runs. The canonical frontier driver, `scripts/experiments/critic_training_v4_review.py`, writes metadata, configurations, model capacity, selection traces, alternate checkpoints, and evaluation metrics to `logs/experiments/`.

An AI **experiment-review skill** acts as an autonomous data science partner to process these runs. Rather than mechanically tuning hyperparameters, it synthesizes results to provide research-backed insights and hypotheses.

**To trigger it:** Point any capable LLM at `.claude/skills/experiment-review/SKILL.md` and ask it to read the skill and run it via the instructions.

**What it does:** 
- **Intelligent Backfilling**: Reads `git` logs and configuration diffs to reconstruct the rationale for past runs, automatically backfilling missing provenance and observations.
- **Data Science Partner**: Synthesizes interacting variables (e.g., encoder choice vs model capacity) to form hypotheses about the model's fundamental understanding of the task.
- **Research Colleague**: Actively browses the web for state-of-the-art literature to validate its recommendations for next-step experiments.
- **Reporting**: Produces a structured analysis of metric trade-offs (e.g., hedging vs minority recall), logs compact circumplex summaries, and maintains a leaderboard of the best models.

### LLM Context Baseline

The frozen-holdout baseline in `scripts/experiments/llm_critic_baseline.py` compares small OpenAI models under three explicit context setups: `student_visible`, `human_context`, and the upper-bound-only `full_judge_context`. On the 221-row test split, the `human_context` setup improves QWK and mean minority recall over `run_020`, while the VIF Critic (Offline) retains higher `recall_-1` and lower hedging. The LLM is useful for LLM-Judge target repair or inference fallback diagnostics, not as a drop-in VIF Critic (Offline) replacement. See [`docs/vif/03_model_training.md`](docs/vif/03_model_training.md) for the CLI and interpretation.

## Synthetic Data Generation — ✅ Complete

Bootstraps training data for value tagging and reward modeling. Generates realistic, longitudinal Journal Entries from synthetic personas with known Schwartz value profiles.

Key features:
- Personas with 1-2 assigned values expressed through concrete life details (not labels)
- Longitudinal Journal Entries that exhibit value tensions, Conflicts, and ambiguity
- Parallel asynchronous generation workflow
- Configurable tone, verbosity, and reflection mode per entry
- ✅ Two-way conversational journaling with displayed nudge classification,
  generation, reply, skip, safe failure, and retry behavior

The displayed nudge interaction is complete for the capstone POC. It is a
product interaction, not a method for improving the VIF Critic (Offline) or Weekly Drift
Detection. A future external pilot can measure response rate, continued
journaling, and perceived relevance.

See `docs/pipeline/pipeline_specs.md` for implementation details.

### Current Dataset

| Metric | Value |
|--------|-------|
| Personas | 204 |
| Journal Entries | 1,651 |
| Avg entries/persona | 8.1 |
| Entries with generated nudges | 1,028 (62.3%) |

**Demographics:** 6 cultures, 9 professions, and 5 standard age brackets.

**Schwartz Value Distribution** (personas can have 1-2 values):
| Value | Personas | % |
|-------|----------|---|
| Power | 37 | 18% |
| Security | 36 | 18% |
| Hedonism | 33 | 16% |
| Universalism | 32 | 16% |
| Conformity | 28 | 14% |
| Tradition | 28 | 14% |
| Achievement | 25 | 12% |
| Benevolence | 25 | 12% |
| Self-Direction | 24 | 12% |
| Stimulation | 24 | 12% |

**LLM-Judge VIF Label Distribution** (16,510 per-dimension labels across 1,651 Journal Entries):
| Label | Count | % |
|-------|-------|---|
| -1 | 1,165 | 7.1% |
| 0 | 12,535 | 75.9% |
| +1 | 2,810 | 17.0% |

Most generated nudges still use the standard three-category taxonomy (`tension_surfacing`, `elaboration`, `clarification`); a small number of older one-off labels remain in legacy raw artifacts.

See [`docs/pipeline/data_schema.md`](docs/pipeline/data_schema.md) for parquet schemas and query examples.

## LLM-Judge Labeling Workflow — ✅ Complete

Creates LLM-Judge VIF Labels for synthetic Journal Entries across the 10
Schwartz value dimensions for VIF Critic (Offline) training. The workflow uses
Claude Code subagents for parallel, consistent labeling.

**Workflow:**
```
Registry Check → Auto-Wrangle → Parallel Labeling (subagents) → Validation → Consolidation
```

**Usage:** Run `/judge` in Claude Code to execute the full workflow. The skill:
1. Checks the registry for pending work (`logs/registry/personas.parquet`)
2. Auto-wrangles raw synthetic data if needed (`logs/synthetic_data/` → `logs/wrangled/`)
3. Spawns parallel subagents — one per persona — each creating LLM-Judge VIF Labels for all Journal Entries
4. Validates JSON output against Pydantic models
5. Consolidates labels to `logs/judge_labels/judge_labels.parquet`

**LLM-Judge VIF Labels:** Each Journal Entry receives a 10-dimensional vector
with values `{-1, 0, +1}`. `-1` means Conflict, `0` is neutral, and `+1` is the
positive class for one Schwartz value. **Rationales** explain each non-zero
LLM-Judge VIF Label. Most Journal Entries have 1-3 non-zero LLM-Judge VIF
Labels.

**Data outputs:** See [`docs/pipeline/data_schema.md`](docs/pipeline/data_schema.md) for parquet file schemas, example Polars queries, and analytics guidance.

**Consensus reference labels:** [`logs/judge_labels/consensus_labels.parquet`](logs/judge_labels/consensus_labels.parquet) stores the five-pass LLM-Judge resolver output, confidence tiers, agreement counts, and label-change flags. It remains diagnostic rather than the mainline VIF Critic (Offline) training target. For Drift v1, a strict Conflict is `alignment_<value> == -1`; the resolver first chooses neutral versus non-neutral, then polarity among non-neutral votes. The agreement fields are confidence metadata, not full class distributions; actual `P(-1)`, `P(0)`, and `P(+1)` targets require the per-pass LLM-Judge vote files. This LLM-Judge reference is distinct from the six-detector comparison's detector-vote count. The orchestration guide lives in [`docs/pipeline/consensus_rejudging_instructions.md`](docs/pipeline/consensus_rejudging_instructions.md), and the stability-first report lives in [`logs/exports/twinkl_754/consensus_rejudging_report.md`](logs/exports/twinkl_754/consensus_rejudging_report.md). It is label provenance and diagnostic evidence only: it must not be used as a Drift target, threshold-selection input, or final test set.

**Key files:**
- `.claude/commands/judge.md` — Skill entry point
- `.claude/skills/judge/orchestration.md` — Detailed workflow
- `.claude/skills/judge/annotation_guide.md` — Scorability heuristics and calibration examples
- `.claude/skills/judge/rubric.md` — Schwartz value reference for scoring

**Primary Generation Method:**
- `docs/pipeline/claude_gen_instructions.md` — Instructions for Claude Code to generate synthetic data using parallel subagents

**Experimentation Scripts/Modules** (for prompt iteration and testing):
- `src/synthetic/generation.py` — One-way generation primitives (context, date sampling, banned-term guards)
- `src/synthetic/batch_preparation.py` — Baseline snapshots and frozen-holdout manifests for targeted data-lift experiments
- `src/synthetic/batch_verification.py` — Raw-batch acceptance checks and spot-check export generation
- `src/nudge/decision.py` + `src/nudge/generation.py` — Two-way conversational nudging logic
- `scripts/journalling/generation_sanity_check.py` — Quick local sanity checks
- `scripts/journalling/twinkl_681_5_freeze_baseline.py` / `scripts/journalling/twinkl_691_2_prepare_batch.py` — Example baseline-freeze wrappers
- `scripts/journalling/twinkl_681_5_verify_batch.py` / `scripts/journalling/twinkl_691_2_verify_batch.py` — Example targeted-batch verification wrappers
- `scripts/journalling/twinkl_754_prepare_consensus.py` / `twinkl_754_validate_results.py` / `twinkl_754_merge_pass_results.py` / `twinkl_754_summarize_consensus.py` — Consensus rerun bundle preparation, validation, merge, and stability-first reporting

## Human Annotation Tool — ✅ Complete

Measures overlap between project-team annotations and LLM-Judge VIF Labels across 10 Schwartz value dimensions. Annotators score the shared subset before seeing the LLM-Judge VIF Labels; the tool then computes Cohen's κ and Fleiss' κ. This bounded sample does not establish independent human validation or label accuracy.

**Run the tool:**
```sh
uv run shiny run src/annotation_tool/app.py
```

Open `http://127.0.0.1:8000` in your browser.

**Features:**
- Displays persona context (name, age, profession, culture, Core Values, collapsible bio)
- Shows Journal Entries with nudge/response threading
- 10-value LLM-Judge VIF Label grid with `-1` (Conflict), `0` (neutral), and
  `+1`, with CSS tooltips for Schwartz value definitions
- Progress tracking per annotator
- Annotations persisted to `logs/annotations/<annotator>.parquet`
- **Analysis & Metrics panel** — Computes inter-annotator agreement (Cohen's κ, Fleiss' κ)
- **Export functionality** — CSV, Parquet, and Markdown report formats
- **Comparison view** — Inline display of project-team annotations and LLM-Judge VIF Labels for review

**Key files:**
- `src/annotation_tool/app.py` — Main Shiny application
- `src/annotation_tool/data_loader.py` — Loads entries from wrangled files
- `src/annotation_tool/annotation_store.py` — Persists annotations with file locking
- `src/annotation_tool/agreement_metrics.py` — Kappa calculations and export
- `src/annotation_tool/components/` — Modular UI components (scoring grid, comparison view, analysis)
- `src/annotation_tool/state.py` — Centralized state management
- `docs/pipeline/annotation_tool_plan.md` — Full implementation plan

## Drift Inspection App — ✅ Complete

The read-only Python Shiny app compares Runs 1–3 for three frozen Weekly Drift
Reviewer setups: `gpt-5.4-mini` at reasoning effort `none`, `gpt-5.6-luna` at
reasoning effort `none`, and `gpt-5.6-luna` at reasoning effort `low`. It shows
complete development results, persona-level outcomes, Journal Entries,
AI-reviewed LLM-Judge Conflict Labels, Weekly Drift Reviewer Decisions, and Run
variability without merging Runs or calculating a majority vote.
The first two setups are historical comparisons; `gpt-5.6-luna` at reasoning
effort `low` is the fixed Weekly Drift Reviewer model contract.

**Run the app:**

```sh
uv run shiny run --host 127.0.0.1 --port 8000 --no-dev-mode \
  src/drift_review_app/app.py
```

Open `http://127.0.0.1:8000`. The `drift-review-app` entry in
`.claude/launch.json` runs the same command.

**Features:**

- filters for known Drift status and Core Value before persona selection
- complete development summaries for known Drift hits, false Drift alerts,
  coverage, and all preserved Runs
- persona scoreboards with exact known Drift and Drift alert spans
- side-by-side Journal Entries, LLM-Judge Conflict Labels, Weekly Drift
  Reviewer Decisions, cited evidence, and verified weekly cutoffs
- fail-closed checks for prompt hashes, setup identities, model identifiers,
  reasoning effort, joins, counts, and aggregate parity
- no model or provider API calls; all inputs are committed research files

`railway.json` deploys the app with `Dockerfile.review_app`; `railway up` starts
a Railway deployment from the repository. The container uses Railway's `PORT`
and needs no database or persistent volume.

See [`docs/demo/weekly_drift_review_app.md`](docs/demo/weekly_drift_review_app.md)
for the review contract, input boundary, launch options, and frozen files.

**Key files:**

- `src/drift_review_app/app.py` — Shiny interface
- `src/drift_review_app/data.py` — frozen-input loading and validation
- `src/drift_rules.py` — shared deterministic Drift rules
- `Dockerfile.review_app`, `requirements-review-app.txt`, and `railway.json` —
  deployment boundary

## Runtime Demo Review App — 🧪 Deprecated Experimental Compatibility

A sibling Shiny app for inspecting the deprecated VIF Critic (Offline) crash/rut/evolution compatibility path. It remains available for historical demonstrations but does not execute the approved Weekly Drift Reviewer and Drift Detector runtime.

**Run the app:**
```sh
uv run shiny run src/demo_tool/app.py
```

Open `http://127.0.0.1:8000` in your browser when running via `shiny run`.

To launch the same app directly with Python:
```sh
uv run python src/demo_tool/app.py
```

Open `http://127.0.0.1:8001` when running the file directly.

**Features:**
- Persona browser with full timeline, nudges, responses, and collapsible bio
- Checkpoint catalog sourced from `logs/experiments/artifacts`, `models/vif`, and `logs/experiments`
- Cached output loading for previously run persona/checkpoint pairs
- End-to-end runtime execution via `src.coach.runtime.run_weekly_coach_cycle`
- Detector input source toggle between **LLM-Judge VIF Labels** and **VIF Critic Predictions**
- Detector comparison across **Baseline**, **EMA**, **CUSUM**, **Cosine**, **Control Chart**, and **KL Div**, with per-Journal Entry detector-vote counts (not the five-pass LLM-Judge reference)
- A six-tab result canvas. Its old `Weekly Digest` tab name is a compatibility label.
- A live Coach Digest response with `weekly_mirror`, `tension_explanation`, and `reflective_question` sections

**Coach Digest response:** `src/coach/llm_client.py` builds the provider-backed
callable that `src/demo_tool/runtime_bridge.py` injects into
`run_weekly_coach_cycle`. `TWINKL_COACH_PROVIDER` selects `openai` (default) or
`gemini`; `TWINKL_COACH_MODEL` overrides the per-provider default model
(`gpt-5.6-luna` or `gemini-3.5-flash`). OpenAI calls use reasoning effort
`none`. When the selected provider's API key is absent, the provider is
unrecognised, or the request fails, the app keeps its structured output without
a response. The app stays runnable offline. The `src.coach.runtime` and
`src.coach.weekly_digest` CLIs do not call a live Coach Digest LLM. They render
and persist the prompt only.

**Generated files:** The app writes persona/checkpoint-specific runtime bundles under `logs/exports/demo_tool_runs/<persona_id>/<checkpoint-stem>-<hash>/`.

See [`docs/demo/review_app.md`](docs/demo/review_app.md) for the full workflow
and file layout. This app is distinct from the read-only
[`Drift Inspection App`](docs/demo/weekly_drift_review_app.md), which compares
frozen Weekly Drift Reviewer Runs and does not execute the VIF Critic (Offline) runtime.

**Key files:**
- `src/demo_tool/app.py` — Main Shiny demo application
- `src/demo_tool/data_loader.py` — Persona catalog and chronological timeline loading
- `src/demo_tool/runtime_bridge.py` — Checkpoint discovery and Coach Digest runtime wrapper
- `src/coach/llm_client.py` — Provider-backed Coach Digest response adapters (Gemini, OpenAI)
- `src/demo_tool/multi_drift.py` — Multi-detector comparison bundle for LLM-Judge VIF Label and VIF Critic Prediction views
- `src/demo_tool/state.py` — Centralized reactive UI state

## Evaluation Workflow — ⚠️ Partial

The evaluations follow two connected paths:

- **User-facing path:** Weekly Drift Detection checks whether the fixed Weekly
  Drift Reviewer and Drift Detector find Drift without unacceptable false
  Drift alerts. Coach Digest evaluation checks the cited evidence, response
  contract, and future user-perceived accuracy.
- **Offline research path:** LLM-Judge validation measures label consistency and
  project-team agreement. Value Modeling records how well the VIF Critic
  (Offline) recovers Conflict. The VIF Critic (Offline) remains outside the
  user-facing path.

See [`docs/evals/overview.md`](docs/evals/overview.md) for the full evaluation workflow and current status.

Coach Digest Evals accept `--judge-provider {openai,gemini}` and
`--judge-model`, so the evaluator can use a different provider from the Coach
Digest generator. The deterministic Drift/control study selects one target for
each of the 42 known development Drifts and 42 matched controls under the
current committed inputs. Its default command writes the target catalog without
provider calls. The paid generation, cross-provider AI review, and comparison
report have no committed result; the five-response result remains same-model AI
review, not human validation.

## Embedding Explorer — ✅ Complete

An interactive 3D visualization that lets you explore the VIF Critic (Offline) embedding space. By projecting high-dimensional hidden-layer activations and SBERT text embeddings into 3D via PCA and t-SNE, the explorer shows how the model organizes Journal Entries — whether Journal Entries with similar value profiles cluster together, how prediction errors distribute across the space, and where the model is most uncertain.

This is useful for building intuition about the VIF Critic (Offline): do Conflict predictions occupy distinct regions? Are hard dimensions (stimulation, hedonism) scattered differently than easy ones? Does the hidden-layer structure differ meaningfully from the raw text embeddings?

**Generate and open:**
```sh
uv run python -m src.vif.extract_embeddings \
  --checkpoint logs/experiments/artifacts/.../BalancedSoftmax/selected_checkpoint.pt
```

**Features:**
- 4 projection spaces: Hidden Layer / SBERT Embedding × PCA / t-SNE
- 5 color modes: Data Split, Prediction, LLM-Judge VIF Label (shown as
  `Ground Truth` in the compatibility UI), Persona, Uncertainty
- Per-dimension filtering across all 10 Schwartz values
- Click-to-inspect: view Journal Entry text, VIF Critic Predictions versus
  LLM-Judge VIF Labels, and uncertainty per Journal Entry
- Persona trajectory lines (toggle-able) showing temporal progression through embedding space
- Adjustable bloom glow, auto-rotation, full orbit controls

**Output:** Self-contained HTML file (`viz/embedding_explorer.html`, ~3MB) with embedded Three.js and all 1,651 data points. No server required.

The CLI accepts `--output` for a different HTML path, `--perplexity` for the
t-SNE setting (default `30.0`), and `--no-browser` to suppress automatic browser
opening.

**Key files:**
- `src/vif/extract_embeddings.py` — Extraction script and HTML template
- `viz/embedding_explorer.html` — Generated visualization (gitignored)

## Known Gaps

| Capability | Status | Note |
|---|---|---|
| Onboarding (SVBWS Values Assessment) | 🧪 Experimental | The React POC implements the complete local, user-facing flow and a versioned Profile. Manual Experience synchronizes the confirmed Profile with the in-memory Python boundary. Production multi-user storage and generalized persistence remain outside the capstone. |
| Coach Digest validation depth | ⚠️ Partial | The same five accepted key-week responses appear in the Persona replay fixtures and the [evaluation manifest](logs/experiments/reports/coach_digest_sample_20260824/judge_sample_manifest.json). All five passed all Coach Digest Validations. Coach Digest Evals scored mean correctness `4.80`, specificity `5.00`, non-prescriptive tone `5.00`, and tension honesty `4.60`; all reflective questions passed, with no failed verdicts or review flags. These scores are same-model AI review, not human validation. Cross-provider evaluator options and the deterministic 42-Drift/42-control study are implemented, but no paid independent-provider result is committed. Future human calibration of the AI review remains incomplete. |
| Experience and Inspect completion | 🚧 In Progress | The shared app, five saved Persona replays, manual Journal Entries, displayed nudges, Weekly Drift Detection, Coach Digest, Inspect, privacy notice, confirmed session deletion, and release checks are implemented. Coach Digest feedback, longitudinal Core Value history, and the final professor walkthrough evidence remain open. |
| Displayed nudge user evidence | ⚠️ Not collected | Displayed nudge implementation is complete. A future external pilot can measure response rate, continued journaling, and perceived relevance. Saved replays and regression tests do not establish those user outcomes. |
| Embedding Explorer | ✅ Complete | Interactive 3D visualization of VIF Critic (Offline) embeddings |
| Drift Detector validation and deployment approval | ⚠️ Not claimed | The deterministic Drift Detector and Luna-low Weekly Drift Reviewer runtime are complete and wired for the capstone POC, with versioned receipts and fail-closed abstention. The evidence is AI-reviewed synthetic development evidence; no fresh final test was run, so no deployment approval is claimed. |
| Journaling anomaly radar | ❌ Not Started | Cadence/gap detection beyond the current prototype-router tooling |
| Goal-aligned inspiration feed | ❌ Not Started | External API integration |

For the full breakdown, see the [Implementation Status](docs/prd.md#implementation-status) table in prd.md.

## Common Commands

Examples below use `uv run` so they pick up the project environment directly. Activating `.venv` manually also works.

- Launch the annotation tool: `uv run shiny run src/annotation_tool/app.py`
- Launch the Experience and Inspect Python boundary: `uv run uvicorn src.demo.api:app --port 8000`
- Launch the React development server from `frontend/onboarding/`: `npm run dev`
- Run the React checks from `frontend/onboarding/`: `npm test`, `npm run typecheck`, and `npm run build`
- Regenerate the Experience and Inspect contracts: `uv run python -m src.demo.export_contract_schema`
- Launch the Drift Inspection App: `uv run shiny run --host 127.0.0.1 --port 8000 --no-dev-mode src/drift_review_app/app.py`
- Deploy the Drift Inspection App to Railway: `railway up`
- Launch the Runtime Demo Review App: `uv run shiny run src/demo_tool/app.py`
- Run the Runtime Demo Review App directly on port `8001`: `uv run python src/demo_tool/app.py`
- Run the paid Weekly Drift Detection path: `uv run python -m src.coach.weekly_drift_runtime --persona-id 0a2fe15c --execute`
- Reproduce the deprecated VIF Critic (Offline) compatibility path: `uv run python -m src.coach.runtime --persona-id 0a2fe15c --checkpoint-path logs/experiments/artifacts/.../selected_checkpoint.pt`
- Build compatibility Weekly Drift Detection output from persisted LLM-Judge VIF Labels: `uv run python -m src.coach.weekly_digest --persona-id 0a2fe15c`
- Build compatibility Weekly Drift Detection output from saved VIF Critic Predictions: `uv run python -m src.coach.weekly_digest --persona-id 0a2fe15c --signals-path logs/exports/weekly_coach/0a2fe15c_vif_timeline.parquet`
- Train the VIF Critic (Offline) with CLI overrides and LR-finder export: `uv run python -m src.vif.train --grad-clip 1.0 --lr-find-output-path logs/exports/lr_find.png`
- Run the BNN baseline: `uv run python -m src.vif.train_bnn --epochs 10 --batch-size 16`
- Generate the embedding explorer without auto-opening a browser: `uv run python -m src.vif.extract_embeddings --checkpoint logs/experiments/artifacts/.../selected_checkpoint.pt --no-browser`
- Prepare a deterministic consensus pilot bundle: `uv run python scripts/journalling/twinkl_754_prepare_consensus.py --pilot-size 50 --pilot-hard-dimensions security,hedonism,stimulation`
- Reproduce the default consensus-label Drift EDA with runtime-compatible week bins: `uv run python scripts/drift/trajectory_eda.py`
- Compare persisted LLM-Judge VIF Labels with week bins anchored to the first Journal Entry: `uv run python scripts/drift/trajectory_eda.py --labels judge --week-mode persona_anchor`
- Estimate the LLM context baseline cost without making API calls: `uv run python scripts/experiments/llm_critic_baseline.py estimate --split test --context-arms student_visible human_context`
- Re-score the frozen Weekly Drift Reviewer model comparison: `uv run python -m scripts.experiments.compare_twinkl_52zz_models score`
- Re-score the Luna higher-reasoning comparison: `uv run python -m scripts.experiments.compare_twinkl_ck3w_luna_higher_reasoning score`
- Replay recall-aware checkpoint selection from saved traces without retraining: `uv run python scripts/experiments/replay_recall_aware_checkpoint_selection.py`
- Build the deterministic Coach Digest Drift/control target catalog without provider calls: `uv run python scripts/experiments/run_coach_drift_control_eval.py`
- Dry-run cross-provider Coach Digest Evals over the committed five-response manifest: `uv run python -m src.evals.coach_narrative_judge --manifest logs/experiments/reports/coach_digest_sample_20260824/judge_sample_manifest.json --judge-provider gemini`
- After paid Drift/control generation and Coach Digest Evals, build the saved comparison report: `uv run python -m src.evals.coach_drift_control_report --manifest logs/experiments/reports/coach_digest_drift_control/judge_sample_manifest.json --eval-metrics logs/experiments/reports/coach_digest_drift_control/evals/metrics.json --out logs/experiments/reports/coach_digest_drift_control/comparison`
- Regenerate the capstone report figures: `MPLCONFIGDIR=/tmp/twinkl-matplotlib uv run python scripts/capstone/generate_report_figures.py`
- Render the capstone report PDF: `quarto render docs/capstone_report/capstone_project_report.md --to pdf`

The Drift EDA accepts `--labels {consensus,judge}` (default: `consensus`) and `--week-mode {runtime,persona_anchor}` (default: `runtime`). The LLM baseline exposes `estimate`, `run`, `score`, and `report`; `run` writes dry-run records unless `--execute` is supplied.

The `twinkl-52zz` model-comparison runner exposes `prepare`, `estimate`, `run`,
and `score`; `run` requires `--execute` and accepts
`--model-key {all,gpt_5_4_mini,gpt_5_6_luna}`. The Luna reasoning-effort runner
exposes `prepare`, `smoke`, `run`, and `score`; its paid `smoke` and `run`
commands require `--execute`. Both runners accept `--root` and `--config`.
Those global options must precede the subcommand.

The Drift/control runner accepts source and output overrides through
`--episodes-parquet`, `--case-outcomes-parquet`, `--wrangled-dir`,
`--parquet-path`, `--output-dir`, `--manifest-out`, and `--targets-out`.
`--group {drift,control,both}`, `--limit`, and `--seed` control selection.
`--resume` preserves completed target IDs, and `--execute` authorizes paid
Weekly Drift Reviewer and Coach Digest calls. See the
[Coach Digest test and eval guide](docs/evals/coach_narrative_test_and_eval_guide.md)
before using either paid command.

# Setup

This repo uses `uv` and `pyproject.toml` for dependency management.

1. Install `uv`:
   ```sh
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   (Or see https://docs.astral.sh/uv/getting-started/installation/ for other methods)

2. Create the virtual environment:
   ```sh
   uv venv
   ```
3. Activate it when you want an interactive shell (Fish shell preferred in this repo):
   ```sh
   source .venv/bin/activate.fish
   ```
   Bash/Zsh fallback:
   ```sh
   source .venv/bin/activate
   ```
4. For provider-backed commands, create a `.env` file in the project root with
   the key for each selected provider:
   ```sh
   OPENAI_API_KEY=your-api-key-here
   GEMINI_API_KEY=your-gemini-api-key-here
   ```

   Gemini also accepts `GOOGLE_API_KEY`. `TWINKL_COACH_PROVIDER` selects
   `openai` or `gemini` for Coach Digest generation, and
   `TWINKL_COACH_MODEL` overrides that provider's default. Coach Digest Evals
   can select an independent evaluator with `--judge-provider` and
   `--judge-model`.

   The Drift Inspection App reads committed files and does not require an API
   key.

## Installing dependencies

Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`.

- Install everything from the lockfile:
  ```sh
  uv sync
  ```
- Install the development group with pytest, pytest-asyncio, and Ruff:
  ```sh
  uv sync --group dev
  ```

## Running tests

Install the dev dependencies first:

```sh
uv sync --group dev
```

Run the full pytest suite:

```sh
uv run pytest
```

Run Ruff on the Python files touched by a change:

```sh
uv run ruff check path/to/changed_file.py tests/path/to/changed_test.py
```

Run the pinned MyPy version on Python files whose typed behavior changed:

```sh
uv run --with 'mypy==2.3.0' mypy path/to/changed_file.py
```

The repository still contains historical notebook and test lint debt, so a
repo-wide `uv run ruff check .` is diagnostic rather than a clean gate.

Run the deterministic local end-to-end smoke pipeline only:

```sh
uv run pytest tests/e2e -q
```

This smoke test exercises the offline path `synthetic_data -> wrangled markdown -> consolidated LLM-Judge VIF Labels -> VIF Critic (Offline) training` using tiny local fixtures and a mock text encoder, so it does not require live LLM calls.

## Adding a dependency

Use `uv add` to both install into the environment and record it in
`pyproject.toml`:

```sh
uv add <package>
```

Pin an exact version if desired:

```sh
uv add "<package>==<version>"
```

After adding, `uv` updates `uv.lock` automatically.

## Exporting requirements.txt (optional)

Only needed for legacy tooling or platforms that require it:

```sh
uv export -o requirements.txt
```
