# VIF – System Architecture, State, and Runtime Flow

This document is the canonical technical overview for the live VIF stack and
its training/runtime data contracts.

For training details, see [Reward Modeling & Training](03_model_training.md).
For uncertainty and Drift review logic, see
[Uncertainty and Drift Review Logic](04_uncertainty_logic.md).
For the adopted capstone scope and metric hierarchy, see
[VIF Capstone Scope and Evaluation Decision](05_capstone_scope_decision.md).

---

## 1. Current POC Shape

The current VIF implementation is intentionally narrow:

- **Modality**: text only
- **Training target**: immediate per-dimension alignment labels in `{-1, 0, +1}`
- **Encoder**: frozen sentence encoder configured in `config/vif.yaml`
- **State**: text window + time gaps + 10-dim value-profile weights
- **Runtime output**: per-Journal-Entry alignment means and uncertainties, plus weekly aggregates
- **Primary capstone role**: recover visible Conflict (`-1`) evidence; retain
  ternary outputs for diagnostics and non-gating positive context
- **Current executable downstream use**: the approved Weekly Drift Reviewer and
  Drift Detector path into the Weekly Digest, plus a deprecated VIF Critic
  crash/rut/evolution compatibility path
- **Fixed Weekly Drift Reviewer**: `gpt-5.6-luna` with reasoning effort `low`,
  without VIF Critic input
- **Approved user-facing runtime**: Weekly Drift Reviewer Decisions followed by
  the deterministic rule that two consecutive Conflicts for the same Core
  Value form Drift
- **Completed VIF Critic research**: training, evaluation, raw output export,
  and timeline inference are implemented for offline reproduction; a
  generalized review-and-retrain loop and candidate confirmation are not
  planned for the time-boxed capstone

> **Note:** VIF Critic encoder names, embedding dimensions, and default window
> sizes change over time. Treat `config/vif.yaml` as the source of truth for
> those runtime values. The fixed Weekly Drift Reviewer contract is a separate
> product decision.

The approved capstone POC runtime persists versioned Weekly Drift Reviewer
Decisions, fails closed to Abstain, applies the Drift Detector across week
boundaries, and feeds the Weekly Digest. VIF Critic training supports the
versioned recall-first checkpoint-selection policy, and the runtime still uses
synthetic `core_values` rather than persisted onboarding `top_values`. The fixed
model choice and implementation do not grant deployment approval.

## Current Diagram

The current implementation diagram is maintained in two synced forms. It shows
the approved capstone POC path beside the deprecated compatibility path; the
adopted staged architecture is documented in
[VIF Critic Role in Drift Detection](../architecture/drift_detection.md).

- Mermaid source: [`current_system_architecture.mmd`](current_system_architecture.mmd)
- Rendered SVG: [`current_system_architecture.svg`](current_system_architecture.svg)

![Current VIF architecture](current_system_architecture.svg)

## Publication Figures

For report and presentation use, there are two simplified variants of the same
architecture:

- Compact report/slide figure:
  - Mermaid source: [`publication_system_architecture.mmd`](publication_system_architecture.mmd)
  - Rendered SVG: [`publication_system_architecture.svg`](publication_system_architecture.svg)
- Slightly more detailed appendix figure:
  - Mermaid source: [`appendix_system_architecture.mmd`](appendix_system_architecture.mmd)
  - Rendered SVG: [`appendix_system_architecture.svg`](appendix_system_architecture.svg)

![Compact publication VIF architecture](publication_system_architecture.svg)

---

## 2. Inputs and State Representation

### 2.1 Raw Inputs

For a user or synthetic persona `u` at step `t`, the VIF consumes:

- Journal Entry text `T_{u,t}`
- time gap features derived from Journal Entry dates
- user profile weights `w_u` over the 10 Schwartz dimensions

Audio, prosody, and physiological signals remain out of scope for the capstone
POC.

### 2.2 State Definition

For a configured window size `N`:

$$
s_{u,t} = \text{Concat}\Big[
\phi_{\text{text}}(T_{u,t}),
\phi_{\text{text}}(T_{u,t-1}), \dots,
\phi_{\text{text}}(T_{u,t-N+1}),
\Delta t_{u,t}, \dots, \Delta t_{u,t-N+2},
w_u
\Big]
$$

Where:

- `\phi_text(T)` is the frozen sentence embedding. `T` is built by the runtime
  `concatenate_entry_text` helper from `initial_entry`, `nudge_text`, and
  `response_text`.
- `\Delta t` are normalized time-gap features
- `w_u` is the normalized 10-dim value profile

For synthetic personas, the runtime assigns equal mass to each Core Value and
falls back to a uniform ten-dimensional vector if no Core Value
matches. The graded BWS profile is specified in the onboarding contract but is
not wired into this state path.

The state excludes label-derived history statistics because they create
train/serve skew.

### 2.3 Current Default vs General Form

The architecture is written generically in terms of `N`, but the current config
defaults to `window_size: 1` because larger windows inflated the parameter
budget and overfit at current data scale.

That means the live default state is effectively:

$$
s_{u,t} = \text{Concat}\big[\phi_{\text{text}}(T_{u,t}), w_u\big]
$$

with no time-gap terms when `N = 1`.

The active default therefore does not contain the Journal Entry date, persona
name, age, profession, culture, biography, or earlier Journal Entries. Any
audit that proposes a
replacement distillation target must review a faithful representation of the
runtime-formatted session and the numeric profile vector, without those hidden
fields. The fail-closed Security workflow is specified in
[Security Distillation Target Contract](security_target_contract.md).

`twinkl-749` added a config-gated compact-history diagnostic without changing
that default. The tested mode is Nomic-specific: with `history_pooling: mean`,
the state keeps the full current
embedding and appends the normalized leading 64 dimensions of the mean of up to
three strictly prior embeddings plus a normalized prior-count feature. The
first Journal Entry receives zeros for that channel. Dataset and runtime both
select prior Journal Entries by chronological position inside the same persona;
future Journal Entries and label-derived data are excluded. Seed-11 `run_069`
regressed on the
overall package, so the mode remains diagnostic. See the
[design note](compact_history_ablation.md) and
[experiment review](../../logs/experiments/reports/experiment_review_2026-07-11_twinkl_749_compact_history.md).

### 2.4 Missing History Handling

When `N > 1` and earlier Journal Entries are unavailable:

- missing embeddings are zero-padded
- missing time gaps are zero-filled

This keeps the state dimension fixed while allowing early-timeline inference.
Compact-history mode instead averages only real prior Journal Entries and emits
a zero summary plus zero count at cold start; padding is never included in the
mean.

---

## 3. Training Data Workflow

### 3.1 Logical Data Objects

The training workflow turns synthetic journals plus LLM-Judge labels into fixed
state/target rows:

- **Persona**: profile, Core Values, and narrative context
- **Journal Entry**: text, date, and per-Journal-Entry metadata
- **JudgeLabel**: per-dimension LLM-Judge labels in `{-1, 0, +1}`
- **StateTargetSample**: the flattened state vector paired with the target vector

The mainline VIF Critic trains on persisted single-pass LLM-Judge labels in
`judge_labels.parquet`. The five-pass LLM-Judge table is diagnostic retraining
data. Drift v1 uses it strictly: each qualifying reference Conflict requires
`alignment_<value> == -1`. It is not the
mainline training target or the six-detector comparison's detector vote.

### 3.2 Journal Entry Text Used by the VIF Critic

The runtime and dataset layers build VIF input text from the wrangled Journal
Entry components:

- `initial_entry`
- `nudge_text`
- `response_text`

This is concatenated into one text field before encoding so the VIF Critic sees
the same Journal Entry representation in both training and inference.

### 3.3 State Construction Procedure

The concrete state-construction path is:

1. Load wrangled Journal Entries and consolidated LLM-Judge labels.
2. Join them on `(persona_id, t_index)` with integrity checks.
3. Precompute or cache sentence embeddings for Journal Entry text.
4. Build each state vector from:
   - the current Journal Entry and `N-1` previous Journal Entries
   - normalized time gaps between those Journal Entries
   - the persona's 10-dim normalized value profile
5. Emit one training sample per labeled Journal Entry.

### 3.4 Splits and Holdouts

Evaluation splits are by persona, not by individual Journal Entry.

- default split: 70/15/15 by persona
- holdout selection: best-effort sign-stratified validation/test partitions
- optional experiment mode: fixed validation/test holdout manifests for
  before/after retrains

This is the regime used by the current frontier experiment archive.

---

## 4. Runtime Inference Flow

### 4.1 VIF Critic Runtime Path

The runtime path rebuilds the same state definition used in training:

1. Load a trained checkpoint and recover runtime-relevant config metadata.
2. Recreate the text encoder and `StateEncoder`.
3. Rebuild one state vector per wrangled Journal Entry in a timeline.
4. Run uncertainty-aware VIF Critic inference.
5. Persist the currently implemented per-Journal-Entry alignment means,
   uncertainties, and weekly aggregates.

The runtime parquet files do not currently persist ordinal class probabilities
or checkpoint provenance sufficient for a generalized review-and-retrain loop.
This does not block the user-facing path. The unplanned review-and-retrain idea
would need versioned `P(-1)`, `P(0)`, `P(+1)`, uncertainty, and input-contract
records before offline comparison and candidate evaluation were reproducible.

The bridge from checkpoint -> timeline parquet -> weekly VIF parquet is
implemented in `src/vif/runtime.py`.

### 4.2 Weekly Aggregation

Per-Journal-Entry outputs are aggregated into weekly tables containing:

- per-dimension mean alignment
- per-dimension mean uncertainty
- profile weights
- profile-weighted overall mean alignment
- profile-weighted overall uncertainty

These weekly parquet files are the inputs for Drift experiments and Weekly Coach
generation.

`src/vif/weekly_schema.py` owns the ordered column contract between
`aggregate_timeline_by_week()` and `detect_weekly_drift()`. It defines the
per-dimension alignment, uncertainty, and profile-weight names and raises a
`ValueError` naming any required columns missing at the consumer boundary.

### 4.3 Drift Routing: Deprecated Compatibility and Approved Runtime

The deprecated compatibility router in `src/vif/drift.py` consumes weekly means and
uncertainties and can emit `stable`, `crash`, `rut`, `evolution`, or
`high_uncertainty`. It invokes the experimental evolution classifier
automatically when no precomputed result is supplied. This is the route used by
the deprecated `src.coach.runtime` entry point and Runtime Demo Review App.

The approved capstone POC runtime is narrower: the Weekly Drift Reviewer is fixed
at `gpt-5.6-luna` with reasoning effort `low` and reads Journal Entries and Core
Values without VIF Critic predictions. The deterministic Drift Detector
declares Drift after two consecutive Conflicts for the same Core Value and
feeds its delivery state to the Weekly Digest. The former
consensus-derived frozen benchmark is retired historical evidence; it does not
implement or justify the active target. [`twinkl-v8pb`](../evals/drift_v1_student_visible_target.md)
is historical. [`twinkl-752.4`](../../logs/experiments/reports/experiment_review_2026-07-13_twinkl_752_4_legacy_drift_review.md)
found 31 episodes across 26 Drift trajectories. Three overlap the earlier five;
adding the two omitted prior episodes produces the 33-episode /
28-Drift-trajectory known-development union. Four reviewed episodes came from
the former final-test split; include them in the primary development analysis
and report provenance subgroups separately. `twinkl-752.5` found 9/33 median
Drifts with weekly review without VIF Critic input, 7/33 with raw VIF Critic
input, and 9/33 with VIF-Critic-triggered early-plus-weekly review. The staged
architecture therefore keeps raw predictions and scheduling out of the
user-facing path. The later [`twinkl-qtwz` complete
review](../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_qtwz_complete_development_review.md)
expanded the development data to 292 resolved cases with 42 Drifts across 36
Drift trajectories; it did not rerun or change the `twinkl-752.5` results. The
production connection remains absent, and the existing weekly router remains a
compatibility prototype.
See
[`docs/drift/trajectory_eda.md`](../drift/trajectory_eda.md) and
[`docs/evals/drift_detection_eval.md`](../evals/drift_detection_eval.md).

---

## 5. Decision and Learning Separation

The architecture separates user-facing Drift decisions from VIF Critic
measurement and improvement.

### 5.1 VIF Critic

The VIF Critic:

- reads only the structured state described above
- uses only the configured state; optional history is strictly prior-only
- outputs numeric alignment estimates plus uncertainty
- exports VIF Critic Predictions and uncertainty for offline diagnostics
- does not implement the unplanned generalized independent-review and
  retraining loop
- does not send predictions to the Weekly Drift Reviewer or make a user-facing
  Drift claim

### 5.2 Weekly Drift Reviewer and Drift Detector

The Weekly Drift Reviewer:

- uses `gpt-5.6-luna` with reasoning effort `low`
- reads Journal Entries and Core Values without VIF Critic predictions
- decides Conflict, Not Conflict, or Abstain

The deterministic Drift Detector declares Drift only after two consecutive
Weekly Drift Reviewer Conflicts for the same Core Value.

### 5.3 Weekly Coach

The Weekly Coach:

- receives Drift-specific state from the Drift Detector rather than VIF Critic
  predictions
- reads the user's full journal history at current POC scale
- turns the Weekly Digest into reflective, evidence-based language

This keeps VIF Critic prediction generation auditable, prevents it from biasing
the Weekly Drift Reviewer, and lets the Weekly Coach speak in richer,
evidence-based language.

---

## 6. Implementation Reference

Key files for the architecture described here:

| Module | Role |
|--------|------|
| `src/vif/state_encoder.py` | Builds fixed-length state vectors |
| `src/vif/dataset.py` | Loads labels and Journal Entries, joins them, and manages persona splits |
| `src/vif/encoders.py` | Creates the configured sentence encoder |
| `src/vif/runtime.py` | Rebuilds states from history and emits runtime parquet files |
| `src/vif/weekly_schema.py` | Defines and validates the weekly signal-frame contract |
| `src/weekly_drift_reviewer.py` | Implements the frozen Weekly Drift Reviewer contract, caller, validation, and receipts |
| `src/drift_detector.py` | Implements the deterministic Drift Detector and delivery states |
| `src/coach/weekly_drift_runtime.py` | Orchestrates the approved Weekly Drift Reviewer to Weekly Digest path |
| `src/vif/drift.py` | Implements the deprecated weekly crash/rut/evolution compatibility router |
| `src/vif/evolution.py` | Supplies the prototype's automatic evolution classification |
| `src/vif/holdout.py` | Loads fixed holdout manifests for experiment reruns |
| `src/coach/runtime.py` | Orchestrates the deprecated VIF Critic compatibility path |
| `src/demo_tool/multi_drift.py` | Compares six exploratory detector families |

---

## 7. Future Extensions

The current architecture leaves room for later work without changing the core
spine:

- larger context windows when justified by data scale
- richer profile conditioning
- multimodal inputs
- retrieval once journal histories outgrow the context window
