# Alignment and Drift Detection Evaluation

## Evaluation Contract

Twinkl evaluates one v1 definition of drift:

> A sustained conflict episode occurs when two adjacent journal entries clearly
> show the writer making a behavior or choice against the same declared core
> value.

The earlier consensus-derived frozen benchmark is [retired historical
evidence](../archive/evals/retired_wq9p_drift_benchmark_2026-07-11.md). The
five-pass Judge consensus table remains label provenance and a diagnostic, but
it is not a drift target, threshold-selection input, or promotion surface. This
is distinct from the six-detector comparison's vote count.

The current layers of the contract are deliberately different:

| Layer | Contract |
|---|---|
| Student-visible target | An entry is negative only when the full displayed runtime text clearly shows a behavior or choice against a declared core value. `twinkl-v8pb` completed the paired review. |
| Development population | The original fixed validation personas defined the target and selected one fixed `run_020` threshold: probability 0.8 and uncertainty 1.010153. |
| Promotion population | A separate 24-case population was locked before review and threshold selection. One 19-entry case remained unresolved, so its promotion score was deliberately not performed. |
| Production runtime | Selected scorer and detector are not approved or wired. |
| User delivery | The weekly Coach digest cites the relevant entries and uses active, recovered, mixed, or uncertain wording without score jargon; exact schema implementation is pending. |

### Adopted metric hierarchy (`twinkl-752`)

- The product decision unit is the sustained-conflict **episode**, not an
  isolated entry or aggregate QWK.
- Future product evaluation prioritizes episode recall. Before deployment, its
  operating point must also satisfy a conservative episode precision or
  user-facing false-alert constraint.
- The acceptable false-alert tolerance is deliberately deferred until after
  recall-focused model development. No new numerical precision floor is active.
- Entry-level `recall_-1` is the primary development proxy because an episode
  cannot be recovered when its component negative entries are missed.
- QWK and `+1` recall are diagnostics. Positive evidence cannot trigger or
  cancel a conflict episode.
- An uncertain or abstaining scorer produces no drift claim. Coverage,
  abstention, and true episodes suppressed by uncertainty must be reported.

See the adopted [VIF scope decision](../vif/05_capstone_scope_decision.md).

The runtime detector is soft because the current Critic often hedges a true
conflict toward neutral. Weekly delivery remains a product cadence rather than
a requirement that the evidence itself be grouped into multi-week averages.

Single-entry dip alerts, crash/rut taxonomies, fade/dormancy, peripheral-value
rise, onboarding-gap messaging, value-evolution gating, and multi-week low-mean
rules are outside the v1 evaluation contract.

The empirical basis is
[`docs/drift/trajectory_eda.md`](../drift/trajectory_eda.md).

---

## Implementation Status

**Status:** 🟡 Partial

### Implemented and Measured

- The five-pass Judge consensus table is available at
  `logs/judge_labels/consensus_labels.parquet`. For an executable strict
  conflict check, require `alignment_<value> == -1`.
- The trajectory EDA covers 204 personas and 1,651 entries with
  runtime-compatible weekly bins.
- The selected reference definition identifies 40 of 204 personas (19.6%);
  the persisted single-pass label comparison identifies 49 of 204 (24.0%).
- The conflict-heavy-week candidate table contains 106 core-dimension weeks
  across 71 personas at `-1` density `>= 0.5` and at least two entries.
- Per-entry MC Dropout means and uncertainties are emitted by the Critic
  runtime.
- `src/vif/weekly_schema.py` defines and validates the weekly-frame contract
  shared by `aggregate_timeline_by_week()` and `detect_weekly_drift()`.
- `src/coach/runtime.py` runs the offline checkpoint-to-digest path and writes
  timeline, weekly, drift, digest, markdown, and prompt artifacts.
- The demo review app compares six exploratory rule-based detector families
  against Judge labels or Critic means.
- `twinkl-v8pb` completed its full-runtime-text review. Development had 42
  cases / 335 entries, with 41/42 case agreement (0.97619) and 324/335
  entry agreement (0.96716). At the fixed threshold, `run_020` found 1/5
  reference episodes (precision 1.0, recall 0.2, F1 0.3333, false-positive
  rate 0.0).
- The locked promotion review had 24 cases / 191 entries, with 23/24 case
  agreement (0.95833) and 180/191 entry agreement (0.94241). case_023 was
  unresolved across 19 entries, so the promotion score was deliberately not
  performed rather than scoring only the agreed cases.
- The retained author-designed trajectories are a capability probe only. They
  are not a target, a threshold-selection input, or a promotion surface.

### Current Prototype Boundary

`src/vif/drift.py` implements an experimental weekly router with literal output
modes `stable`, `crash`, `rut`, `evolution`, and `high_uncertainty`. It also
invokes the experimental evolution classifier automatically. That router is a
working prototype and remains useful for end-to-end UI and schema testing, but
it is not the selected sustained-conflict detector.

The six-detector comparison in `src/demo_tool/multi_drift.py` is another
exploratory surface. Its per-entry vote count is detector agreement, not the
five-pass Judge reference.

### Retired consensus-derived benchmark (historical)

`twinkl-wq9p` was retired because its frozen consensus episodes were not a fair
student-visible promotion surface. Its former scripts, artifacts, report, and
dedicated tests are not active repository surfaces. The historical audit is
useful only for explaining that retirement; it must not be rerun, tuned, scored,
or used to promote a scorer. See the [retirement record](../archive/evals/retired_wq9p_drift_benchmark_2026-07-11.md).

The full evidence is in the [student-visible target record](drift_v1_student_visible_target.md), the [development threshold receipt](../../logs/experiments/artifacts/drift_target_twinkl_v8pb_20260711/development/thresholds.json), and the [promotion no-score record](../../logs/experiments/artifacts/drift_target_twinkl_v8pb_20260711/promotion/promotion_no_score.json). No fallback score was taken from the retired benchmark.

### Still Missing for Product v1

- Per-entry `P(-1)` persistence in the runtime artifact surface
- A fresh, independently resolved promotion surface; the reviewed population
  cannot be reduced to its agreed cases after disagreement was observed
- A scorer and calibrated operating point that pass a future fair
  decision-level promotion check
- Production integration of the selected soft-evidence detector
- Coach-language checks for active, recovered, mixed, and uncertain states at
  digest time

---

## Why This Definition Fits the Current Data

The observed timelines are short: the median persona has five active runtime
weeks. Multi-week low-period machinery therefore consumes most of the available
history and is too sparse for a robust capstone benchmark.

Single-entry dips are common but noisy. On declared core dimensions, 84.4% of
dip events recover to `>= 0` within two entries. Requiring two consecutive
conflicts removes most spike noise without imposing calendar logic that the
dataset cannot support.

Core-value gating is also load-bearing. Nearly half of all persona-by-dimension
trajectories are all-neutral, while only 1.0% of declared core trajectories are
all-neutral. An ungated benchmark would be dominated by dimensions the persona
did not declare as important.

---

## Student-Visible Target Outcome (2026-07-11)

The [student-visible target](drift_v1_student_visible_target.md) labels an
entry negative only when its displayed text clearly shows the writer making a
behavior or choice against a declared core value. Frustration, guilt, wishes,
outside constraints without a voluntary choice, biography, and ambiguous prose
do not qualify on their own.

Two immediately adjacent negative entries for the same value form one
sustained-conflict episode. A non-negative or uncertain entry breaks the run.
Later entries can describe whether an already-recorded episode is active,
recovered, or uncertain; they do not change whether the earlier pair occurred.

The target used two deliberately separate populations. The development review
used the original fixed validation personas; the locked promotion review used
24 registry personas added after the original model split. Both reviews used
the full text the runtime state encoder receives: journal entry, displayed
nudge, and displayed response.

The development result is diagnostic, not a promotion pass: 42 cases / 335
entries, 41/42 case agreement, 324/335 entry agreement, and `run_020` detecting
1/5 reference episodes at the fixed probability 0.8 / uncertainty 1.010153
threshold. The promotion result is a valid no-go outcome: 23/24 case agreement
and 180/191 entry agreement, but case_023 remained unresolved across 19
entries. No promotion score was run. The old consensus event table remains
retired historical evidence and must not be materialized or used for either
population.

---

## Runtime Detector Target

For dimension `j` at entry `t`, let `p^-_{t,j}` be the Critic probability of
class `-1` and `u_{t,j}` be its uncertainty estimate. A v1 detector accumulates
recent negative evidence only when:

- the dimension is a declared core value;
- uncertainty is below the calibrated ceiling; and
- the recent `P(-1)` mass passes a persistence threshold.

Profile weights may calibrate evidence or thresholds among declared core
values, but they do not make an undeclared value eligible for v1 drift.

The exact rolling function and thresholds are evaluation parameters, not part
of the student-visible label definition. Candidate forms include a two-entry
mean, an exponentially weighted sum, or a small cumulative evidence score.
These forms are not automatically equivalent: a pair mean can pass because of
one very strong entry even when the other lacks adequate evidence. The runtime
rule must either enforce a per-entry evidence condition or demonstrate that its
chosen soft rule preserves the canonical two-entry event. That choice remains
open.

Hard argmax predictions are not the runtime contract. Requiring two predicted
`-1` classes would be brittle at the current `recall_-1` frontier.

---

## Evaluation Protocol

### Phase 1: Student-Visible Target Development

1. Review the fixed development population using only the declared core value
   and displayed journal trajectory.
2. Reconcile the paired reviews into a versioned target while preserving the
   original Judge labels as provenance rather than overwriting them.
3. Choose at most one detector threshold from that development target.
4. Record uncertainty, rationale, and unresolved-case handling separately from
   the main sustained-conflict decision.

### Phase 2: Locked Promotion Review

1. Keep the promotion population locked before its review and before threshold
   selection.
2. Review it under the same student-visible rule and reconcile only after both
   review responses are recorded.
3. Treat an unresolved promotion case as a block on a promotion claim.

### Phase 3: One Fair Scorer Comparison

1. Run the allowed scorer once against the locked promotion target after the
   development threshold is fixed.
2. Report episode hits, false positives, and delivery-state handling without
   changing the target or threshold after seeing those results.
3. Keep author-designed trajectories separate as a capability probe; they can
   never substitute for the locked promotion population.

`twinkl-v8pb` completed Phases 1 and 2 under this protocol. Phase 3 stopped
before scoring: the unresolved promotion case makes the target invalid for a
fair score, and scoring only the agreed cases would be cherry-picking.

---

## Metrics and Targets

### Primary Event Metrics

| Metric | Status | Meaning |
|---|---|---|
| Development episode recall (`run_020`) | 1/5 (0.2) | The fixed development threshold found one of five reviewed reference episodes |
| Development precision / false-positive rate (`run_020`) | 1.0 / 0.0 | The single predicted development episode was correct, but four reference episodes were missed |
| Development F1 (`run_020`) | 0.3333 | Balances the perfect precision with low recall |
| Promotion episode metrics | Deliberately not scored | case_023 remained unresolved across 19 entries; the target is not valid for a fair score |
| First-alert latency | Not scored | It requires a resolved promotion target |
| Author-designed capability recall | Capability-only diagnostic | Whether the scorer can find deliberately clear episodes; never a scorer-promotion gate |

A historical development-only operating point exists (probability 0.8,
uncertainty 1.010153), but it is not the newly adopted recall-first policy and
no numerical promotion threshold is active. The retained
author-designed controls remain capability-only diagnostics and cannot
substitute for a fresh, resolved locked promotion population.

### Required Slices

- Per Schwartz value dimension
- Declared-core-value rank or profile-weight band
- Episode length and severity
- Review confidence and disagreement state
- Active, recovered, mixed, and uncertain digest-time cases
- Allowed scorer and input-contract version

### Uncertainty Validation

Uncertainty gating must be evaluated on the `-1` class specifically. Global
calibration can look acceptable while the minority class is poorly calibrated.
Report:

- error rate by uncertainty decile;
- retained episode recall at each uncertainty ceiling;
- false-positive reduction from gating; and
- the number of true episodes suppressed by high uncertainty;
- abstention count and coverage; and
- the number of false user-facing claims per non-drift trajectory or week.

---

## Reproduction

Run the historical consensus analysis with runtime-compatible week bins:

```sh
uv run python scripts/drift/trajectory_eda.py
```

Compare persisted labels and first-entry-anchored bins:

```sh
uv run python scripts/drift/trajectory_eda.py \
  --labels judge \
  --week-mode persona_anchor
```

Options:

- `--labels {consensus,judge}`; default: `consensus`
- `--week-mode {runtime,persona_anchor}`; default: `runtime`

Generated evidence lives under `docs/drift/figures/` and
`docs/drift/tables/`.

These commands reproduce historical EDA only. They must not be used to build a
drift target, select a threshold, or promote a scorer. There is deliberately no
active command for the retired `twinkl-wq9p` benchmark. The completed
student-visible review and its blocked promotion result are described in
[`drift_v1_student_visible_target.md`](drift_v1_student_visible_target.md).

---

## Limitations

1. Consensus labels are stored Judge provenance, not human ground truth. The
   former frozen consensus benchmark is retired and cannot support a target,
   threshold, or promotion claim.
2. The incumbent `run_020` checkpoint predates the student-visible target. It
   was evaluated only on the completed development target; the unresolved
   promotion target deliberately received no score.
3. Core-gated per-dimension denominators are small and uneven.
4. Five personas have only two entries, so their temporal evidence is limited
   to one possible adjacent pair.
5. The current synthetic corpus contains volatility more readily than clean,
   gradual arcs; it cannot validate fade or value-evolution claims.
6. Weekly Coach delivery can lag reference confirmation, so event matching must
   distinguish detector latency from delivery cadence.

---

## Implementation References

| File | Role |
|---|---|
| [`docs/drift/trajectory_eda.md`](../drift/trajectory_eda.md) | Empirical definition analysis and prevalence results |
| [`scripts/drift/trajectory_eda.py`](../../scripts/drift/trajectory_eda.py) | Reproducible trajectory analysis |
| [`src/vif/runtime.py`](../../src/vif/runtime.py) | Per-entry inference and weekly aggregation |
| [`src/vif/weekly_schema.py`](../../src/vif/weekly_schema.py) | Weekly producer/consumer column contract |
| [`src/vif/drift.py`](../../src/vif/drift.py) | Existing experimental weekly router |
| [`src/coach/runtime.py`](../../src/coach/runtime.py) | Offline checkpoint-to-digest orchestration |
| [`src/demo_tool/multi_drift.py`](../../src/demo_tool/multi_drift.py) | Six-detector exploratory comparison |
| [`scripts/experiments/llm_critic_baseline.py`](../../scripts/experiments/llm_critic_baseline.py) | Student-visible and history-context comparison arms |
| [`drift_v1_student_visible_target.md`](drift_v1_student_visible_target.md) | Completed target rule, review boundary, development result, and blocked promotion result |
| [`../../config/evals/drift_v1_student_visible_v1.yaml`](../../config/evals/drift_v1_student_visible_v1.yaml) | Locks the development and promotion populations before review or threshold selection |
| [`../../src/vif/drift_target.py`](../../src/vif/drift_target.py) | Student-visible review packets, reconciliation, and target materialization |
| [`../../scripts/experiments/build_v8pb_student_visible_target.py`](../../scripts/experiments/build_v8pb_student_visible_target.py) | Builds the development or promotion review packet |
| [`../../scripts/experiments/materialize_v8pb_student_visible_target.py`](../../scripts/experiments/materialize_v8pb_student_visible_target.py) | Materializes a reviewed student-visible target variant |
| [`../../config/evals/drift_v1_author_designed_capability.yaml`](../../config/evals/drift_v1_author_designed_capability.yaml) | Author-designed capability probe; never a promotion surface |
| [`logs/experiments/reports/experiment_review_20260702_twinkl_w2mu_frozen_context_gap.md`](../../logs/experiments/reports/experiment_review_20260702_twinkl_w2mu_frozen_context_gap.md) | Frozen test-split LLM context results |
| [`../archive/evals/retired_wq9p_drift_benchmark_2026-07-11.md`](../archive/evals/retired_wq9p_drift_benchmark_2026-07-11.md) | Retired benchmark record; do not rerun, score, tune, or promote from it |
