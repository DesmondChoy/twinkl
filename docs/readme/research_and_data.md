# Research and Data

[Overview](../../README.md) | [Onboarding and Experience](onboarding_and_experience.md) | [Review Apps](review_apps.md) | **Research and Data** | [Status and Setup](status_and_setup.md)

---

## Weekly Drift Detection and North Star Moment

Weekly Drift Detection reviews Journal Entries against the confirmed Profile's
Core Values using `gpt-5.6-luna` with reasoning effort `low`. Prompt `4.0`
includes only the selected Core Values' `definition` and `core_motivation`
fields from `config/schwartz_values.yaml`: Schwartz-based definitions plus
project-specific elaborations. This context belongs to trusted instructions;
Journal Entry text remains separate user-controlled input. The deterministic
Drift Detector identifies two consecutive Conflicts for the same Core Value.

The [Core Value definitions comparison](../../logs/experiments/reports/experiment_review_2026-09-07_twinkl_j3k7_core_value_definitions.md)
contains three Runs per prompt across 204 Personas and 951 observed weeks.
The v3 baseline detects 22, 22, and 21 Drifts; v4 detects 19, 21, and 18.
The report compares Core Value totals, individual Drift boundaries and
terminations, changed entry decisions, and variation between Runs. These
counts establish differences in detections, not better or worse accuracy.
The historical Luna-low recall and false-alert results use prompt v2 and
remain a separate development result.

North Star Moment supplies at most one exact quotation of a supportive action
beneath the Coach Digest. Active Drift uses writing from before onset; No
Active Drift prefers current-week encouragement, then a historical reminder.
Insufficient Evidence produces no card. Eligible Journal Entries and user
nudge responses require ownership and availability evidence; AI-written nudges
cannot supply the quotation. Full eligible history is the application method,
using Luna-low source review and deterministic quotation, chronology, and
selection checks. Live execution requires a finalized integration budget and
fails closed when that budget is unavailable.

The [targeted NSM comparison](../../logs/experiments/reports/north_star_v4_run1_20260907/report.md)
uses saved Weekly Drift v4 Run 1 outputs for 501 weeks from 105 Personas.
It reassesses both methods on 38 affected weeks and retains observations for
463 unchanged weeks, preserving the 81/24 development/final Persona split.

| Partition | Method | Card precision | Opportunity recall |
|---|---|---:|---:|
| Development | Nomic top-three retrieval | 58.58% | 55.35% |
| Development | Full eligible history | 75.54% | 74.62% |
| Qualified final | Nomic top-three retrieval | 57.65% | 56.98% |
| Qualified final | Full eligible history | 82.35% | 81.40% |

These are AI assessments of synthetic writing with paired metric exclusions,
not human validation or an untouched final test. The final histories have prior
upstream research exposure, and the comparison retains earlier observations.
It does not establish real-user benefit, Weekly Drift accuracy, or deployment
approval. The five saved Persona replays preserve full-history selections and
no-card outcomes across 27 weeks. See the [feature specification](../north_star/north_star_moment.md),
[experiment methodology](../north_star/nsm_experiment_methodology.md), and
[research commands](status_and_setup.md#weekly-drift-and-north-star-moment-experiments).

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

The VIF Critic (Offline) research stack includes ordinal MLP heads with MC Dropout, a BNN baseline, config-driven frozen encoders with `nomic-embed-text-v1.5` as the active default, corrected-split experiment logging, checkpoint discovery, recall-first checkpoint selection, raw output export, runtime timeline reconstruction, and weekly aggregation. Its former crash/rut/evolution response routing is deprecated compatibility code. The 69-run / 133-config archive keeps `run_019`-`run_021` Balanced Softmax as the historical corrected-split reference. See [`logs/experiments/index.md`](../../logs/experiments/index.md) for the live board.

**Current Drift contract:** Drift is two consecutive Conflicts for the same Core Value. Weekly Drift Detection uses the internal Weekly Drift Reviewer and Drift Detector. It stores structured output with Core Values, cited Journal Entries, and Drift state. The fixed model contract is `gpt-5.6-luna` with reasoning effort `low`. The frozen development Runs are AI-reviewed synthetic evidence. They are not human validation, a fresh final test, or deployment approval. The completed VIF Critic (Offline) remains outside this path. See the [`twinkl-52zz` report](../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_52zz_luna_low.md), [`docs/architecture/drift_detection.md`](../../docs/architecture/drift_detection.md), and [`docs/evals/drift_detection_eval.md`](../../docs/evals/drift_detection_eval.md).

**Current runtime behavior:** `src.coach.weekly_drift_runtime` runs Weekly Drift Detection. It reads Journal Entries, imports Core Values and Coach Digest context from a confirmed Profile when supplied, calls the internal Weekly Drift Reviewer, persists versioned JSON receipts, and applies the internal Drift Detector across week boundaries. It stores structured output and renders one of three Coach Digest policies: Drift detected, no current Drift, or more reflection needed. When no Profile is supplied, synthetic personas retain their deterministic `core_values` compatibility path. `src.coach.runtime` and `src.vif.drift` are deprecated compatibility paths for the former VIF Critic (Offline) crash/rut/evolution runtime. See `docs/vif/`, [`docs/weekly/weekly_drift_detection.md`](../../docs/weekly/weekly_drift_detection.md), and [`docs/demo/review_app.md`](../../docs/demo/review_app.md).

### Automated Experiment Logging & Review

The experiment log tracks VIF Critic (Offline) training runs. The canonical frontier driver, `scripts/experiments/critic_training_v4_review.py`, writes metadata, configurations, model capacity, selection traces, alternate checkpoints, and evaluation metrics to `logs/experiments/`.

The **experiment-review skill** compares VIF Critic (Offline) runs relevant to
the requested question. It uses the PRD and delegated specifications for current
scope and decisions, then checks run records and saved results for evidence.

**To use it:** Ask the assistant to read the
[experiment-review skill](../../.claude/skills/experiment-review/SKILL.md) and
review the specified runs or research question.

**What it does:**

- Compares compatible data splits, labels, inputs, and metric definitions,
  with summaries across model seeds when available.
- Explains metric trade-offs and checks for confounding changes before making
  causal claims. Keeps inferred explanations separate from recorded rationale.
- Investigates relevant saved results and consults primary research when
  recommendations depend on current methods.
- Reports findings, uncertainty, and next steps appropriate to current capstone
  scope. A review leaves run records and the experiment index unchanged;
  saving a report or updating documentation follows the authorized task scope.

### LLM Context Baseline

The frozen-holdout baseline in `scripts/experiments/llm_critic_baseline.py` compares small OpenAI models under three explicit context setups: `student_visible`, `human_context`, and the upper-bound-only `full_judge_context`. On the 221-row test split, the `human_context` setup improves QWK and mean minority recall over `run_020`, while the VIF Critic (Offline) retains higher `recall_-1` and lower hedging. The LLM is useful for LLM-Judge target repair or inference fallback diagnostics, not as a drop-in VIF Critic (Offline) replacement. See [`docs/vif/03_model_training.md`](../../docs/vif/03_model_training.md) for the CLI and interpretation.

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

See [`docs/pipeline/data_schema.md`](../../docs/pipeline/data_schema.md) for parquet schemas and query examples.

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

**Data outputs:** See [`docs/pipeline/data_schema.md`](../../docs/pipeline/data_schema.md) for parquet file schemas, example Polars queries, and analytics guidance.

**Consensus reference labels:** [`logs/judge_labels/consensus_labels.parquet`](../../logs/judge_labels/consensus_labels.parquet) stores the five-pass LLM-Judge resolver output, confidence tiers, agreement counts, and label-change flags. It remains diagnostic rather than the mainline VIF Critic (Offline) training target. For Drift v1, a strict Conflict is `alignment_<value> == -1`; the resolver first chooses neutral versus non-neutral, then polarity among non-neutral votes. The agreement fields are confidence metadata, not full class distributions; actual `P(-1)`, `P(0)`, and `P(+1)` targets require the per-pass LLM-Judge vote files. This LLM-Judge reference is distinct from the six-detector comparison's detector-vote count. The orchestration guide lives in [`docs/pipeline/consensus_rejudging_instructions.md`](../../docs/pipeline/consensus_rejudging_instructions.md), and the stability-first report lives in [`logs/exports/twinkl_754/consensus_rejudging_report.md`](../../logs/exports/twinkl_754/consensus_rejudging_report.md). It is label provenance and diagnostic evidence only: it must not be used as a Drift target, threshold-selection input, or final test set.

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

## Evaluation Workflow — ⚠️ Partial

The evaluations follow two connected paths:

- **User-facing path:** Weekly Drift Detection checks whether the fixed Weekly
  Drift Reviewer and Drift Detector find Drift without unacceptable false
  Drift alerts. Coach Digest evaluation checks the cited evidence, response
  contract, and future user-perceived accuracy. North Star Moment evaluation
  compares exact quotations and omissions against shared AI assessments of
  eligible writing, with Card precision and Opportunity recall as primary metrics.
- **Offline research path:** LLM-Judge validation measures label consistency and
  project-team agreement. Value Modeling records how well the VIF Critic
  (Offline) recovers Conflict. The VIF Critic (Offline) remains outside the
  user-facing path.

See [`docs/evals/overview.md`](../../docs/evals/overview.md) for the full evaluation workflow and current status.

Coach Digest Evals accept `--judge-provider {openai,gemini}` and
`--judge-model`, so the evaluator can use a different provider from the Coach
Digest generator. The deterministic Drift/control study selects one target for
each of the 42 known development Drifts and 42 matched controls under the
current committed inputs. Its default command writes the target catalog without
provider calls. The paid generation, cross-provider AI review, and comparison
report have no committed result. The August five-response result remains
same-model AI review, not human validation. The five current saved Persona
responses use Coach Digest prompt `4.2` and Luna-none, pass Coach Digest
Validations, and match their current Weekly Drift inputs. They have no
corresponding Coach Digest Evals result; the August scores do not describe them.

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
