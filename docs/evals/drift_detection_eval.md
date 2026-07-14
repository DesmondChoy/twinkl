# Alignment and Drift Detection Evaluation

## Evaluation Contract

Twinkl evaluates one v1 definition of Drift:

> Drift occurs when two consecutive Journal Entries each clearly show the
> writer making a behavior or choice against the same Core Value.

The earlier consensus-derived frozen benchmark is [retired historical
evidence](../archive/evals/retired_wq9p_drift_benchmark_2026-07-11.md). The
five-pass LLM-Judge consensus table remains label provenance and a diagnostic,
but it is not a Drift target, threshold-selection input, or final test set. This
is distinct from the six-detector comparison's vote count.

The current layers of the contract are deliberately different:

| Layer | Contract |
|---|---|
| Student-visible target | A Journal Entry is a Conflict only when the full displayed text clearly shows a behavior or choice against a Core Value. `twinkl-752.4` reviewed every candidate discoverable from either legacy label source plus matched controls. |
| Development set | The known-development union contains 33 Drift episodes across 28 Drift trajectories: all 31 episodes from the `twinkl-752.4` cohort plus two prior episodes omitted by its candidate mining. Four reviewed episodes came from the former final-test split; include them in the primary development analysis and report provenance subgroups separately. Four `twinkl-752.4` trajectories remain uncertain. The fixed `run_020` threshold is historical development evidence. |
| Final test set | None is active. The former 24-person `twinkl-v8pb` final-test cohort became development-only when its cases were opened for the full review. `twinkl-pv6s` owns a fresh final test. |
| Production runtime | The VIF Critic and Drift Detector are not approved or wired. |
| User delivery | The Weekly Digest cites the relevant Journal Entries and uses active, recovered, mixed, or uncertain wording without score jargon; exact schema implementation is pending. |

### Adopted metric hierarchy (`twinkl-752`)

- The product decision unit is **Drift**, not an
  isolated Journal Entry or aggregate QWK.
- Future product evaluation prioritizes Drift recall. Before deployment, its
  operating point must also satisfy conservative Drift precision or a
  user-facing false Drift alert constraint.
- The acceptable false-alert tolerance is deliberately deferred until after
  recall-focused model development. No new numerical precision floor is active.
- Entry-level `recall_-1` is the primary development proxy because Drift
  cannot be recovered when either component Conflict is missed.
- QWK and `+1` recall are diagnostics. Positive evidence cannot trigger or
  cancel Drift.
- An uncertain or abstaining VIF Critic produces no Drift claim. Coverage,
  abstention, and true Drifts suppressed by uncertainty must be reported.

See the adopted [VIF scope decision](../vif/05_capstone_scope_decision.md).

The runtime Drift Detector target uses soft probabilities because the current
VIF Critic often hedges a true Conflict toward neutral. Weekly delivery remains
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

- The five-pass LLM-Judge consensus table is available at
  `logs/judge_labels/consensus_labels.parquet`. For an executable strict
  conflict check, require `alignment_<value> == -1`.
- The trajectory EDA covers 204 personas and 1,651 Journal Entries with
  runtime-compatible weekly bins.
- The selected reference definition identifies 40 of 204 personas (19.6%);
  the persisted single-pass label comparison identifies 49 of 204 (24.0%).
- The conflict-heavy-week table contains 106 Core-Value weeks
  across 71 personas at `-1` density `>= 0.5` and at least two Journal Entries.
- Per-entry MC Dropout means and uncertainties are emitted by the VIF Critic
  runtime.
- `src/vif/weekly_schema.py` defines and validates the weekly-frame contract
  shared by `aggregate_timeline_by_week()` and `detect_weekly_drift()`.
- `src/coach/runtime.py` runs the offline checkpoint-to-digest path and writes
  timeline, weekly, Drift, Weekly Digest, markdown, and prompt files.
- The demo review app compares six exploratory rule-based detector families
  against LLM-Judge labels or VIF Critic means.
- `twinkl-v8pb` completed its full-runtime-text review. Development had 42
  cases / 335 Journal Entries, with 41/42 case agreement (0.97619) and 324/335
  Journal Entry agreement (0.96716). At the fixed threshold, `run_020` found 1/5
  reference Drifts (precision 1.0, recall 0.2, F1 0.3333, false-positive
  rate 0.0).
- `twinkl-752.4` froze 52 legacy-discoverable candidate trajectories and 52
  unique-person matched legacy-negative controls: 104 trajectories and 874
  entry/Core-Value decisions. Two packet-only Codex lanes agreed on 849/874
  entries (97.1%); a third reviewer resolved 21 of 25 disagreements. The final
  target resolves 100/104 trajectories and contains 31 maximal Drift episodes
  across 26 Drift trajectories. Four of those episodes came from the former
  final-test split, which is a provenance subgroup rather than an evaluation
  exclusion.
- Only three of the earlier five development episodes occur in the
  `twinkl-752.4` cohort. Adding the omitted `3a3b15e4:tradition` and
  `7adc5866:benevolence` episodes produces the 33-episode / 28-Drift-trajectory
  known-development union for `twinkl-752.5`. The frozen `twinkl-752.4`
  artifacts remain a correct cohort receipt.
- Among resolved development cases, 22/43 legacy candidates and 1/42 controls
  contain Drift. This is candidate confirmation and one legacy-miner miss, not
  prevalence or a Weekly Drift Reviewer false-alert estimate. Controls sample
  only part of the legacy-negative pool, so other missed Drifts may remain
  outside the cohort.
- `twinkl-752.1` compared the Weekly Drift Reviewer without VIF Critic input
  against the Weekly Drift Reviewer with fixed `run_020` predictions across 41
  resolved development trajectories. Without VIF Critic input, median Drift
  recall was 0.40 with one false Drift alert and 0.756 coverage; with VIF
  Critic input, those results were 0.20, zero, and 0.732. Ten genuinely invalid outputs
  remained fail-closed; 69 raw receipts rejected by an over-strict optional-
  quote validator were recovered under the registered prompt contract.
- The `twinkl-752.1` Drift-recall comparison contained only five episodes:
  median recall 0.20 versus 0.40 represents one versus two detected episodes.
  Its conditional rejection of raw VIF Critic input must be reassessed on the
  33-episode union in `twinkl-752.5`.
- `twinkl-752.3` repeated complete adjacent Journal Entry pairs, including
  week-boundary pairs, supplied a versioned Core Value rubric, and requested
  explicit Drift decisions. Median Drift recall fell to 0.20, median false Drift
  alerts rose to 5, and neither cross-week reference Drift was recovered. The
  tested prompt differences did not materially limit the earlier result.
- Future AI-reviewed reference-label work must use the same versioned rubric in
  `config/evals/drift_v1_conflict_rubric_v1.yaml`; the completed reference labels
  remain unchanged.
- The former locked review had 24 cases / 191 Journal Entries, with 23/24 case
  agreement (0.95833) and 180/191 Journal Entry agreement (0.94241). Its score
  was correctly withheld. Because `twinkl-752.4` opened the old population for
  development reference work, it is no longer eligible as a final test.
- The retained author-designed trajectories are a capability probe only. They
  are not a target, a threshold-selection input, or a final test set.

### Current Prototype Boundary

`src/vif/drift.py` implements an experimental weekly router with literal output
modes `stable`, `crash`, `rut`, `evolution`, and `high_uncertainty`. It also
invokes the experimental evolution classifier automatically. That router is a
working prototype and remains useful for end-to-end UI and schema testing, but
it is not the selected Drift Detector.

The six-detector comparison in `src/demo_tool/multi_drift.py` is another
exploratory comparison. Its per-entry vote count is detector agreement, not the
five-pass LLM-Judge reference.

### Retired consensus-derived benchmark (historical)

`twinkl-wq9p` was retired because its frozen consensus Drifts were not a fair
student-visible final test set. Its former scripts, files, report, and
dedicated tests are not active repository surfaces. The historical audit is
useful only for explaining that retirement; it must not be rerun, tuned, scored,
or used to approve a VIF Critic for deployment. See the [retirement record](../archive/evals/retired_wq9p_drift_benchmark_2026-07-11.md).

The historical `twinkl-v8pb` evidence remains in the [student-visible target
record](drift_v1_student_visible_target.md), the [development threshold
receipt](../../logs/experiments/artifacts/drift_target_twinkl_v8pb_20260711/development/thresholds.json),
and the [final-test no-score
record](../../logs/experiments/artifacts/drift_target_twinkl_v8pb_20260711/promotion/promotion_no_score.json).
The reviewed cohort and known-development union correction are in the
[`twinkl-752.4`
report](../../logs/experiments/reports/experiment_review_2026-07-13_twinkl_752_4_legacy_drift_review.md).
No fallback score was taken from the retired benchmark.

### Still Missing for Product v1

- Per-entry `P(-1)` persistence in the runtime output
- A fresh, independently resolved final test set under `twinkl-pv6s`
- A VIF Critic and calibrated operating point that pass a future fair
  decision-level deployment-approval check
- Production integration of the selected soft-evidence detector
- Weekly Coach language checks for active, recovered, mixed, and uncertain states at
  digest time

---

## Why This Definition Fits the Current Data

The observed timelines are short: the median persona has five active runtime
weeks. Multi-week low-period machinery therefore consumes most of the available
history and is too sparse for a robust capstone benchmark.

Single-entry dips are common but noisy. On Core Values, 84.4% of
dip events recover to `>= 0` within two Journal Entries. Requiring two consecutive
conflicts removes most spike noise without imposing calendar logic that the
dataset cannot support.

Core-value gating is also load-bearing. Nearly half of all persona-by-dimension
trajectories are all-neutral, while only 1.0% of Core-Value trajectories are
all-neutral. An ungated benchmark would be dominated by dimensions the persona
did not declare as important.

---

## Student-Visible Target Outcome (2026-07-11)

This section preserves the historical `twinkl-v8pb` result. Its former final-
test population is development-only after `twinkl-752.4` and cannot support a
future deployment claim.

The [student-visible target](drift_v1_student_visible_target.md) labels an
Journal Entry as a Conflict only when its displayed text clearly shows the writer making a
behavior or choice against a Core Value. Frustration, guilt, wishes,
outside constraints without a voluntary choice, biography, and ambiguous prose
do not qualify on their own.

Two consecutive Conflicts for the same Core Value form one Drift. A
non-Conflict or uncertain Journal Entry breaks the run. Later Journal Entries
can describe whether an already-recorded Drift is active,
recovered, or uncertain; they do not change whether the earlier pair occurred.

The target used two deliberately separate sets. The development review
used the original fixed validation personas; the locked final-test review used
24 registry personas added after the original model split. Both reviews used
the full text the runtime state encoder receives: Journal Entry, displayed
nudge, and displayed response.

The development result is diagnostic, not deployment approval: 42 cases / 335
Journal Entries, 41/42 case agreement, 324/335 Journal Entry agreement, and `run_020` detecting
1/5 reference Drifts at the fixed probability 0.8 / uncertainty 1.010153
threshold. The final-test result is a valid no-go outcome: 23/24 case agreement
and 180/191 Journal Entry agreement, but case_023 remained unresolved across 19
Journal Entries. No deployment-approval score was run. The old consensus event table remains
retired historical evidence and must not be materialized or used for either
set.

---

## Runtime Detector Target

For dimension `j` at Journal Entry `t`, let `p^-_{t,j}` be the VIF Critic probability of
class `-1` and `u_{t,j}` be its uncertainty estimate. A v1 detector accumulates
recent Conflict evidence only when:

- the dimension is a Core Value;
- uncertainty is below the calibrated ceiling; and
- the recent `P(-1)` mass passes a persistence threshold.

Profile weights may calibrate evidence or thresholds among Core Values, but
they do not make another value eligible for v1 Drift.

The exact rolling function and thresholds are evaluation parameters, not part
of the student-visible label definition. Options include a two-entry
mean, an exponentially weighted sum, or a small cumulative evidence score.
These forms are not automatically equivalent: a pair mean can pass because of
one very strong Journal Entry even when the other lacks adequate evidence. The runtime
rule must either enforce a per-entry evidence condition or demonstrate that its
chosen soft rule preserves the canonical two-entry event. That choice remains
open.

Hard argmax predictions are not the runtime contract. Requiring two predicted
`-1` classes would be brittle at the current `recall_-1` frontier.

---

## Evaluation Protocol

### Phase 1: Student-Visible Target Development

1. Review the fixed development set using only the Core Value
   and displayed journal trajectory.
2. Reconcile the paired reviews into a versioned target while preserving the
   original LLM-Judge labels as provenance rather than overwriting them.
3. Choose at most one detector threshold from that development target.
4. Record uncertainty, rationale, and unresolved-case handling separately from
   the main Drift decision.

### Phase 2: Locked Final-Test Review

1. Keep the final test set locked before its review and before threshold
   selection.
2. Review it under the same student-visible rule and reconcile only after both
   review responses are recorded.
3. Treat an unresolved final-test case as a block on deployment approval.

### Phase 3: One Fair VIF Critic Comparison

1. Run the allowed VIF Critic once against the locked final-test target after the
   development threshold is fixed.
2. Report Drift hits, false positives, and delivery-state handling without
   changing the target or threshold after seeing those results.
3. Keep author-designed trajectories separate as a capability probe; they can
   never substitute for the locked final test set.

`twinkl-v8pb` completed Phases 1 and 2 under this historical protocol. Phase 3
stopped before scoring: the unresolved case made the target invalid for a fair
score. `twinkl-752.4` later reclassified that population as development-only;
the next final test must be fresh.

---

## Metrics and Targets

### Primary Event Metrics

| Metric | Status | Meaning |
|---|---|---|
| Development Drift recall (`run_020`) | 1/5 (0.2) | The fixed development threshold found one of five reviewed Drifts |
| Reviewed cohort (`twinkl-752.4`) | 31 episodes across 26 resolved Drift trajectories | All cases are development-only; four episodes retain former-final-test provenance for subgroup reporting |
| Known-development union (`twinkl-752.5` input) | 33 episodes across 28 Drift trajectories | Primary reassessment surface; adds two prior episodes omitted by candidate mining |
| Legacy candidate confirmation (`twinkl-752.4`) | 22/43 (51.2%) | Resolved development candidates only; selection-biased diagnostic |
| Matched-control Drift rate (`twinkl-752.4`) | 1/42 (2.4%) | One legacy-miner miss among resolved development controls; not a false-alert rate |
| Development precision / false-positive rate (`run_020`) | 1.0 / 0.0 | The single predicted development Drift was correct, but four reference Drifts were missed |
| Development F1 (`run_020`) | 0.3333 | Balances the perfect precision with low recall |
| Weekly Drift Reviewer without VIF Critic input (`twinkl-752.1`) | Median Drift recall 0.40 / 1 false Drift alert / coverage 0.756 | Historical five-episode result over three repeats; raw-input reassessment pending |
| Weekly Drift Reviewer with VIF Critic input (`twinkl-752.1`) | Median Drift recall 0.20 / 0 false Drift alerts / coverage 0.732 | Historical five-episode result; one detected episode versus two without the input |
| Aligned Weekly Drift Reviewer (`twinkl-752.3`) | Median Drift recall 0.20 / 5 false Drift alerts / coverage 0.829 | More complete but less precise; neither cross-week reference Drift was recovered |
| Final-test Drift metrics | No active final test | The old cohort is development-only; `twinkl-pv6s` must build a fresh surface |
| First-alert latency | Not scored | It requires a resolved final-test target |
| Author-designed capability recall | Capability-only diagnostic | Whether the VIF Critic can find deliberately clear Drifts; never a deployment-approval gate |

A historical development-only operating point exists (probability 0.8,
uncertainty 1.010153), but it is not the newly adopted recall-first policy and
no numerical deployment threshold is active. The retained
author-designed controls remain capability-only diagnostics and cannot
substitute for a fresh, resolved locked final test set.

### Required Slices

- Per Schwartz value dimension
- Core-Value rank or profile-weight band
- Drift length and severity
- Review confidence and disagreement state
- Active, recovered, mixed, and uncertain digest-time cases
- Allowed VIF Critic and input-contract version

### Uncertainty Validation

Uncertainty gating must be evaluated on the `-1` class specifically. Global
calibration can look acceptable while the minority class is poorly calibrated.
Report:

- error rate by uncertainty decile;
- retained Drift recall at each uncertainty ceiling;
- false-positive reduction from gating; and
- the number of true Drifts suppressed by high uncertainty;
- abstention count and coverage; and
- the number of false user-facing claims per trajectory or week without Drift.

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
Drift target, select a threshold, or approve a VIF Critic for deployment. There is deliberately no
active command for the retired `twinkl-wq9p` benchmark. The completed
historical student-visible review is described in
[`drift_v1_student_visible_target.md`](drift_v1_student_visible_target.md). The
reviewed cohort and the correction that forms the 33-episode known-development
union are described in the [`twinkl-752.4`
report](../../logs/experiments/reports/experiment_review_2026-07-13_twinkl_752_4_legacy_drift_review.md).

---

## Limitations

1. Consensus labels are stored LLM-Judge provenance, not human ground truth. The
   former frozen consensus benchmark is retired and cannot support a target,
   threshold, or deployment claim.
2. The incumbent `run_020` checkpoint predates the student-visible target. It
   was evaluated only on the completed development target; the unresolved
   final-test target deliberately received no score.
3. Core-gated per-dimension denominators are small and uneven.
4. Five personas have only two Journal Entries, so their temporal evidence is limited
   to one possible adjacent pair.
5. The current synthetic corpus contains volatility more readily than clean,
   gradual arcs; it cannot validate fade or value-evolution claims.
6. Weekly Coach delivery can lag reference confirmation, so Drift matching must
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
| [`drift_v1_student_visible_target.md`](drift_v1_student_visible_target.md) | Completed target rule, review boundary, development result, and blocked final-test result |
| [`../../config/evals/drift_v1_student_visible_v1.yaml`](../../config/evals/drift_v1_student_visible_v1.yaml) | Locks the development and final-test sets before review or threshold selection |
| [`../../src/vif/drift_target.py`](../../src/vif/drift_target.py) | Student-visible review packets, reconciliation, and target materialization |
| [`../../scripts/experiments/build_v8pb_student_visible_target.py`](../../scripts/experiments/build_v8pb_student_visible_target.py) | Builds the development or final-test review input file |
| [`../../scripts/experiments/materialize_v8pb_student_visible_target.py`](../../scripts/experiments/materialize_v8pb_student_visible_target.py) | Materializes a reviewed student-visible target variant |
| [`../../config/evals/drift_v1_author_designed_capability.yaml`](../../config/evals/drift_v1_author_designed_capability.yaml) | Author-designed capability probe; never a final test set |
| [`../../config/evals/twinkl_752_4_legacy_drift_review_v1.yaml`](../../config/evals/twinkl_752_4_legacy_drift_review_v1.yaml) | Expanded legacy-discoverable candidate and matched-control review contract |
| [`../../src/vif/drift_candidate_review.py`](../../src/vif/drift_candidate_review.py) | Deterministic selection, blind review, adjudication, and episode derivation |
| [`../../scripts/experiments/review_twinkl_752_4_legacy_drift_candidates.py`](../../scripts/experiments/review_twinkl_752_4_legacy_drift_candidates.py) | Freezes packets and materializes the reviewed cohort |
| [`../../logs/experiments/reports/experiment_review_2026-07-13_twinkl_752_4_legacy_drift_review.md`](../../logs/experiments/reports/experiment_review_2026-07-13_twinkl_752_4_legacy_drift_review.md) | Full result, limitations, and `twinkl-752.5` handoff |
| [`logs/experiments/reports/experiment_review_20260702_twinkl_w2mu_frozen_context_gap.md`](../../logs/experiments/reports/experiment_review_20260702_twinkl_w2mu_frozen_context_gap.md) | Frozen test-split LLM context results |
| [`../archive/evals/retired_wq9p_drift_benchmark_2026-07-11.md`](../archive/evals/retired_wq9p_drift_benchmark_2026-07-11.md) | Retired benchmark record; do not rerun, score, tune, or promote from it |
