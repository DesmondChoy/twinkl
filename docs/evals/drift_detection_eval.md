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
| Development set | The complete development review contains 42 Drifts across 36 Drift trajectories in 292 resolved cases. `twinkl-qtwz` added nine Drifts across eight Drift trajectories from the 186 cases outside the earlier 106-case union. The earlier `twinkl-752.5` study used 33 Drifts across 28 Drift trajectories; keep its reported metrics bound to that input. Historical provenance must be reported as a subgroup because all nine newly found Drifts came from training-seen Journal Entries. The fixed `run_020` threshold is historical development evidence. |
| Final test set | None is active. The former 24-person `twinkl-v8pb` final-test cohort became development-only when its cases were opened for the full review. `twinkl-pv6s` owns a fresh final test. |
| Weekly Drift Reviewer | The model contract is fixed at `gpt-5.6-luna` with reasoning effort `low`, without VIF Critic input. The fixed model choice is distinct from final-test validation and deployment approval. |
| Approved architecture | Weekly Drift Reviewer decisions feed the deterministic two-Conflict Drift Detector. The VIF Critic supplies stored predictions for independent review and retraining; candidate confirmation is outside the remaining capstone scope. |
| Production runtime | The executable runtime still uses the crash/rut/evolution prototype. The approved Weekly Drift Reviewer and Drift Detector path is not wired or deployment-approved. |
| User delivery | The Weekly Digest cites the relevant Journal Entries and uses active, recovered, mixed, or uncertain wording without score jargon; exact schema implementation is pending. |

### Adopted metric hierarchy (`twinkl-752`, updated under `twinkl-52zz`)

- The product decision unit is **Drift**, not an
  isolated Journal Entry or aggregate QWK.
- The fixed Luna-low model contract was selected by prioritizing Drift recall
  first and false Drift alerts second.
- Coverage and abstention are diagnostic metrics. They must be reported because
  they expose fail-closed behavior, but they do not gate development selection.
- `twinkl-7vam` must fix the minimum Drift recall, acceptable false Drift alert
  tolerance, stability, and any efficiency requirement before the fresh final
  test is scored. It must also predefine coverage and abstention reporting.
- Entry-level `recall_-1` is the primary development proxy because Drift
  cannot be recovered when either component Conflict is missed.
- QWK and `+1` recall are diagnostics. Positive evidence cannot trigger or
  cancel Drift.
- An uncertain or abstaining Weekly Drift Reviewer produces no Drift claim.
  Coverage, abstention, and true Drifts suppressed by abstention must be
  reported.

See the adopted [VIF scope decision](../vif/05_capstone_scope_decision.md).

The approved Drift Detector uses two consecutive Weekly Drift Reviewer
Conflicts for the same Core Value. VIF Critic probabilities and uncertainty
remain in the offline review-and-retrain path and cannot produce or suppress a
user-facing Drift claim.

Single-entry dip alerts, crash/rut taxonomies, fade/dormancy, peripheral-value
rise, onboarding-gap messaging, value-evolution gating, and multi-week low-mean
rules are outside the v1 evaluation contract.

The empirical basis is
[`docs/drift/trajectory_eda.md`](../drift/trajectory_eda.md).

### Why the VIF Critic does not directly define Drift

A natural alternative is to skip the Weekly Drift Reviewer and treat two
consecutive VIF Critic `-1` Predictions as a Drift, since a VIF Critic
Prediction of `-1` and a Weekly Drift Reviewer Conflict describe the same
event. That is not adopted. The reason is not a competing definition of
Conflict; it is label precision, auditability, and evaluation independence.

- **Precision.** On the matched `twinkl-752.1` Journal Entries, the
  `run_019`-`run_021` family reached macro `recall_-1` of `0.530` to `0.607`
  but `-1` precision of only `0.262` to `0.327`
  ([architecture/drift_detection.md](../architecture/drift_detection.md)). Raw
  VIF Critic Predictions recover candidate Conflicts well but are not a safe
  standalone Drift rule.
- **Measured, not assumed.** Supplying VIF Critic Predictions to the Weekly
  Drift Reviewer was tested and did not help. See the `twinkl-752.1` and
  `twinkl-752.5` rows in [Metrics and Targets](#metrics-and-targets): raw input
  left Drift recall inconclusive, lowered coverage, and raised median false
  Drift alerts.
- **Evaluation independence.** The VIF Critic is trained on LLM-Judge VIF
  Labels. Deriving the Drift target from VIF Critic Predictions would create a
  self-confirming loop, so the Weekly Drift Reviewer decides Conflict, Not
  Conflict, or Abstain without seeing VIF Critic Predictions.

The rejected direct-authority options — no raw VIF Critic Predictions in the
Weekly Drift Reviewer prompt, and no VIF Critic veto, confirmation, or direct
Drift decision — are recorded in
[architecture/drift_detection.md](../architecture/drift_detection.md). No
candidate-confirmation exception is included in the remaining capstone scope.

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
- A blind fourth review through `claude -p --model opus` resolved the four
  remaining trajectories with three Conflict and one non-Conflict label. None
  formed an adjacent Conflict pair, so the reviewed cohort is now 104/104
  resolved with the same 31 Drift episodes across 26 Drift trajectories. The
  known-development union is 106/106 resolved.
- Only three of the earlier five development episodes occur in the
  `twinkl-752.4` cohort. Adding the omitted `3a3b15e4:tradition` and
  `7adc5866:benevolence` episodes produces the 33-episode / 28-Drift-trajectory
  known-development union for `twinkl-752.5`. The frozen `twinkl-752.4`
  artifacts remain a correct cohort receipt.
- `twinkl-qtwz` reviewed the exact 186-case complement with two independent
  LLM-Judge lanes and disagreement-only adjudication. It found nine additional
  Drifts across eight Drift trajectories. The complete development analysis is
  292/292 resolved case-level outcomes, 2,377 Journal Entry/Core Value
  combinations, and 42 Drifts across 36 Drift trajectories. Two immutable
  historical LLM-Judge Conflict Labels remain null inside a case with a frozen
  resolved Drift outcome.
- `twinkl-52zz` compares three frozen Runs for each Weekly Drift Reviewer setup
  on the complete development data. The fixed model contract, `gpt-5.6-luna`
  at reasoning effort `low`, found a median 23/42 known Drifts, produced 4 false
  Drift alerts, and had `0.637` coverage. The read-only
  [Drift Inspection App](../demo/weekly_drift_review_app.md) exposes the complete
  results, persona-level outcomes, Journal Entries, AI-reviewed LLM-Judge
  Conflict Labels, Weekly Drift Reviewer Decisions, cited evidence, and
  verified weekly cutoffs without making model or provider API calls.
- All nine newly found Drifts have historical training provenance. This does
  not invalidate their LLM-Judge Conflict Labels, but VIF Critic results on
  those Journal Entries are in-sample and must be reported separately.
- Among resolved development cases, 22/44 legacy candidates and 1/44 controls
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
- `twinkl-752.5` reassessed the exact raw-input comparison on the 33-Drift
  union. Weekly review without VIF Critic input found a median 9/33 Drifts;
  raw input found 7/33. The paired recall delta was `-0.061` with 95%
  trajectory-bootstrap interval `[-0.158, 0.033]`, so the earlier conditional
  rejection is inconclusive. Raw input also lowered median coverage from
  `0.670` to `0.594` and raised median false Drift alerts from 0 to 3.
- VIF-Critic-triggered early-plus-weekly review found the same median 9/33
  Drifts as weekly-only review. Median delay moved from 5 to 1 day, with one
  added median false Drift alert and 57 added reviewer calls. The apparent
  timing benefit disappeared on the non-training subgroup. The zero-call
  placement diagnostic found 7/19 Drift-relevant triggers versus a random
  median of 1/19, which is targeting evidence rather than evidence that early
  review improves Drift detection.
- `twinkl-752.3` repeated complete adjacent Journal Entry pairs, including
  week-boundary pairs, supplied a versioned Core Value rubric, and requested
  explicit Drift decisions. Median Drift recall fell to 0.20, median false Drift
  alerts rose to 5, and neither cross-week reference Drift was recovered. The
  tested prompt differences did not materially limit the earlier result.
- Future AI-reviewed reference-label work must use the same versioned rubric in
  `config/evals/drift_v1_conflict_rubric_v1.yaml`. The Opus follow-up complied
  and wrote a provenance-preserving revised table instead of overwriting the
  frozen `twinkl-752.4` artifacts.
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
it is not the approved Drift Detector.

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
The four-label Opus follow-up is in the
[`twinkl-752.5` resolution
report](../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_752_5_opus_null_resolution.md).
The raw-input and scheduling results are in the
[`twinkl-752.5` reassessment
report](../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_752_5_reassessment.md).
No fallback score was taken from the retired benchmark.

### Still Missing for Product v1

- Weekly Drift Reviewer decision persistence and deterministic Drift Detector
  wiring
- Full VIF Critic probability, uncertainty, and checkpoint-provenance storage
- Independent disagreement review and versioned retraining data
- Predefined deployment-approval criteria under `twinkl-7vam`
- A fresh, independently resolved final test set under `twinkl-pv6s`
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

The [student-visible target](drift_v1_student_visible_target.md) labels a
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

## Approved Detector and Deferred VIF Critic Idea

The current user-facing target is deterministic:

1. The Weekly Drift Reviewer reads Journal Entries and Core Values without VIF
   Critic predictions.
2. It decides Conflict, non-Conflict, or abstention for each relevant Journal
   Entry.
3. The Drift Detector declares one Drift when two consecutive Weekly Drift
   Reviewer Conflicts concern the same Core Value.

The VIF Critic remains required in a separate review-and-retrain path. Its
versioned probabilities and uncertainty are compared offline with Weekly Drift
Reviewer decisions. Disagreement and uncertain cases may receive independent
LLM-Judge or human review, but Weekly Drift Reviewer outputs must not
automatically become training labels.

VIF Critic candidate confirmation is outside the remaining capstone scope. Its
development evidence remains historical context, not an active implementation
or evaluation branch. Revisiting it requires a new scope decision and fresh
evaluation; the fixed Weekly Drift Reviewer must still see Journal Entry text
and Core Values rather than VIF Critic Predictions.

---

## Evaluation Protocol

### Phase 1: Development Evaluation

1. Review the fixed development set using only the Core Value
   and displayed journal trajectory.
2. Reconcile the paired reviews into a versioned target while preserving the
   original LLM-Judge labels as provenance rather than overwriting them.
3. Evaluate the weekly-only path against the predefined deployment-approval
   criteria.
4. Record uncertainty, rationale, and unresolved-case handling separately from
   the main Drift decision.

### Phase 2: Freeze Before Final Test

1. Keep `gpt-5.6-luna` and reasoning effort `low` fixed, then freeze the Weekly
   Drift Reviewer prompt, response schema, deterministic Drift Detector, and
   deployment-approval criteria.
2. Keep the fresh final test locked and unscored while those choices are made.
3. Keep retraining and development cases out of the final test.

### Phase 3: Fresh Final Test and Deployment Approval

1. Resolve every final-test label independently without VIF Critic predictions
   or expected outcomes.
2. Score the frozen weekly-only path once.
3. Report Drift recall, false Drift alerts, coverage, abstention, stability,
   hard Core Value slices, and any claimed LLM-call or cost reduction without
   changing the criteria.
4. Treat an unresolved case or failed criterion as a block on deployment
   approval. Author-designed trajectories remain capability probes only.

Historically, `twinkl-v8pb` locked and reviewed a proposed final test but stopped
before scoring because one case remained unresolved. `twinkl-752.4` later made
that population development-only. It cannot satisfy the fresh-final-test phase
above.

---

## Metrics and Targets

### Primary Event Metrics

| Metric | Status | Meaning |
|---|---|---|
| Development Drift recall (`run_020`) | 1/5 (0.2) | The fixed development threshold found one of five reviewed Drifts |
| Reviewed cohort (`twinkl-752.4`) | 31 episodes across 26 Drift trajectories; 104/104 resolved | All cases are development-only; four episodes retain former-final-test provenance for subgroup reporting |
| Known-development union (`twinkl-752.5` input) | 33 episodes across 28 Drift trajectories; 106/106 resolved | Primary reassessment set; adds two prior episodes omitted by candidate mining |
| Complete development review (`twinkl-qtwz`) | 42 Drifts across 36 Drift trajectories; 292/292 case-level outcomes resolved | Frozen input used by `twinkl-52zz`; the nine newly found Drifts all have historical training provenance |
| `gpt-5.4-mini` Weekly Drift Reviewer (`twinkl-52zz`) | 7/42 median Drift hits / recall 0.167 / precision 0.583 / 5 false Drift alerts / coverage 0.740 | Complete-development baseline over three repeats |
| `gpt-5.6-luna` Weekly Drift Reviewer (`twinkl-52zz`) | 20/42 median Drift hits / recall 0.476 / precision 0.606 / 13 false Drift alerts / coverage 0.777 | Frozen reasoning-effort-`none` baseline for the follow-up; superseded by the fixed reasoning-effort-`low` contract |
| `gpt-5.6-luna` reasoning-effort-`low` follow-up (`twinkl-52zz`) | 23/42 median Drift hits / recall 0.548 / precision 0.852 / 4 false Drift alerts / coverage 0.637 | Fixed Weekly Drift Reviewer model contract under the approved hierarchy: Drift recall first, false Drift alerts second, and coverage diagnostic; the study stopped before `medium`; final-test validation and deployment approval remain pending |
| Legacy candidate confirmation (`twinkl-752.4`) | 22/44 (50.0%) | Resolved development candidates only; selection-biased diagnostic |
| Matched-control Drift rate (`twinkl-752.4`) | 1/44 (2.3%) | One legacy-miner miss among resolved development controls; not a false-alert rate |
| Development precision / false-positive rate (`run_020`) | 1.0 / 0.0 | The single predicted development Drift was correct, but four reference Drifts were missed |
| Development F1 (`run_020`) | 0.3333 | Balances the perfect precision with low recall |
| Weekly review without VIF Critic input (`twinkl-752.5`) | 9/33 median Drift hits / recall 0.273 / precision 1.0 / 0 false Drift alerts / coverage 0.670 | Historical 106-case development-union result over three repeats; superseded by the complete development review for `twinkl-52zz` |
| Weekly review with raw VIF Critic input (`twinkl-752.5`) | 7/33 median Drift hits / recall 0.212 / precision 0.70 / 3 false Drift alerts / coverage 0.594 | Paired recall interval crosses zero; old rejection is inconclusive |
| VIF-Critic-triggered early-plus-weekly review (`twinkl-752.5`) | 9/33 median Drift hits / recall 0.273 / precision 0.90 / 1 false Drift alert / coverage 0.670 | No recall gain; median delay 1 versus 5 days; review-again only |
| Weekly Drift Reviewer without VIF Critic input (`twinkl-752.1`) | Median Drift recall 0.40 / 1 false Drift alert / coverage 0.756 | Superseded five-episode comparison |
| Weekly Drift Reviewer with VIF Critic input (`twinkl-752.1`) | Median Drift recall 0.20 / 0 false Drift alerts / coverage 0.732 | Superseded five-episode comparison |
| Aligned Weekly Drift Reviewer (`twinkl-752.3`) | Median Drift recall 0.20 / 5 false Drift alerts / coverage 0.829 | More complete but less precise; neither cross-week reference Drift was recovered |
| Final-test Drift metrics | No active final test | The old cohort is development-only; `twinkl-pv6s` must build a fresh final test set |
| Development first-alert latency (`twinkl-752.5`) | Median 5 days weekly-only / 3 days raw-input / 1 day scheduled | Selection-biased development timing; no final-test latency is available |
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
- VIF Critic checkpoint and input-contract version

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
The blind fourth-review labels are described in the
[`twinkl-752.5` resolution
report](../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_752_5_opus_null_resolution.md).
The raw-input, scheduling, and trigger-placement results are described in the
[`twinkl-752.5` reassessment
report](../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_752_5_reassessment.md).
The complete 292-case development-data contract is described in the
[`twinkl-qtwz` review
report](../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_qtwz_complete_development_review.md).

Re-score the committed Weekly Drift Reviewer responses without API calls:

```sh
uv run python -m scripts.experiments.compare_twinkl_52zz_models score
uv run python -m scripts.experiments.compare_twinkl_52zz_luna_reasoning score
```

The model-comparison runner exposes `prepare`, `estimate`, `run`, and `score`;
`run` requires `--execute` and accepts
`--model-key {all,gpt_5_4_mini,gpt_5_6_luna}`. The Luna reasoning-effort runner
exposes `prepare`, `smoke`, `run`, and `score`; `smoke` and `run` require
`--execute`. Both runners accept `--root` and `--config`. The experiment reports
record the frozen commands and inputs; paid execution is unnecessary for
re-scoring the committed responses.

---

## Limitations

1. Consensus labels are stored LLM-Judge provenance, not human ground truth. The
   former frozen consensus benchmark is retired and cannot support a target,
   threshold, or deployment claim.
2. The `twinkl-qtwz` reviews used separate contexts but the same model. They
   are AI-reviewed evidence, not human validation, and correlated errors may
   remain.
3. The incumbent `run_020` checkpoint predates the student-visible target. The
   former final-test data is now development-only, and no active final test has
   been opened.
4. Core-gated per-dimension denominators are small and uneven.
5. Five personas have only two Journal Entries, so their temporal evidence is limited
   to one possible adjacent pair.
6. The current synthetic corpus contains volatility more readily than clean,
   gradual arcs; it cannot validate fade or value-evolution claims.
7. Weekly Coach delivery can lag reference confirmation, so Drift matching must
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
| [`scripts/experiments/reassess_twinkl_752_5.py`](../../scripts/experiments/reassess_twinkl_752_5.py) | Frozen union, raw-input, scheduling, and placement reassessment |
| [`config/evals/twinkl_752_5_reassessment_v1.yaml`](../../config/evals/twinkl_752_5_reassessment_v1.yaml) | Preregistered setup, trigger, bootstrap, and API contract |
| [`drift_v1_student_visible_target.md`](drift_v1_student_visible_target.md) | Completed target rule, development result, and withheld former final-test score |
| [`../../config/evals/drift_v1_student_visible_v1.yaml`](../../config/evals/drift_v1_student_visible_v1.yaml) | Locks the development and final-test sets before review or threshold selection |
| [`../../src/vif/drift_target.py`](../../src/vif/drift_target.py) | Student-visible review packets, reconciliation, and target materialization |
| [`../../scripts/experiments/build_v8pb_student_visible_target.py`](../../scripts/experiments/build_v8pb_student_visible_target.py) | Builds the development or final-test review input file |
| [`../../scripts/experiments/materialize_v8pb_student_visible_target.py`](../../scripts/experiments/materialize_v8pb_student_visible_target.py) | Materializes a reviewed student-visible target variant |
| [`../../config/evals/drift_v1_author_designed_capability.yaml`](../../config/evals/drift_v1_author_designed_capability.yaml) | Author-designed capability probe; never a final test set |
| [`../../config/evals/twinkl_752_4_legacy_drift_review_v1.yaml`](../../config/evals/twinkl_752_4_legacy_drift_review_v1.yaml) | Expanded legacy-discoverable candidate and matched-control review contract |
| [`../../src/vif/drift_candidate_review.py`](../../src/vif/drift_candidate_review.py) | Deterministic selection, blind review, adjudication, and episode derivation |
| [`../../scripts/experiments/review_twinkl_752_4_legacy_drift_candidates.py`](../../scripts/experiments/review_twinkl_752_4_legacy_drift_candidates.py) | Freezes packets and materializes the reviewed cohort |
| [`../../logs/experiments/reports/experiment_review_2026-07-13_twinkl_752_4_legacy_drift_review.md`](../../logs/experiments/reports/experiment_review_2026-07-13_twinkl_752_4_legacy_drift_review.md) | Full result, limitations, and `twinkl-752.5` handoff |
| [`../../config/evals/twinkl_qtwz_complete_development_review_v1.yaml`](../../config/evals/twinkl_qtwz_complete_development_review_v1.yaml) | Complete development review contract |
| [`../../scripts/experiments/review_twinkl_qtwz_remaining_development.py`](../../scripts/experiments/review_twinkl_qtwz_remaining_development.py) | Freezes and validates the 186-case complement |
| [`../../scripts/experiments/reconcile_twinkl_qtwz_review.py`](../../scripts/experiments/reconcile_twinkl_qtwz_review.py) | Reconciles labels and derives the complete development analysis |
| [`../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_qtwz_complete_development_review.md`](../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_qtwz_complete_development_review.md) | Complete 292-case result, provenance, cost, and limitations |
| [`../../config/evals/twinkl_52zz_model_comparison_v1.yaml`](../../config/evals/twinkl_52zz_model_comparison_v1.yaml) | Frozen `gpt-5.4-mini` and Luna reasoning-effort-`none` comparison contract |
| [`../../scripts/experiments/compare_twinkl_52zz_models.py`](../../scripts/experiments/compare_twinkl_52zz_models.py) | Model-comparison preparation, estimate, execution, and scoring workflow |
| [`../../config/evals/twinkl_52zz_luna_low_v1.yaml`](../../config/evals/twinkl_52zz_luna_low_v1.yaml) | Preregistered Luna reasoning-effort-`low` protocol and historical selection gate |
| [`../../scripts/experiments/compare_twinkl_52zz_luna_reasoning.py`](../../scripts/experiments/compare_twinkl_52zz_luna_reasoning.py) | Luna reasoning-effort execution, receipt, and scoring workflow |
| [`../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_52zz_luna_low.md`](../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_52zz_luna_low.md) | Complete Luna reasoning-effort result and evidence behind the fixed model contract |
| [`../../src/drift_review_app/data.py`](../../src/drift_review_app/data.py) | Frozen-input verification and result loading for the Drift Inspection App |
| [`../demo/weekly_drift_review_app.md`](../demo/weekly_drift_review_app.md) | Drift Inspection App contract, launch commands, and input boundary |
| [`../../scripts/experiments/resolve_twinkl_752_5_null_cases.py`](../../scripts/experiments/resolve_twinkl_752_5_null_cases.py) | Freezes and materializes the blind Opus follow-up |
| [`../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_752_5_opus_null_resolution.md`](../../logs/experiments/reports/experiment_review_2026-07-14_twinkl_752_5_opus_null_resolution.md) | Four resolved Conflict labels, revised counts, and limits |
| [`logs/experiments/reports/experiment_review_20260702_twinkl_w2mu_frozen_context_gap.md`](../../logs/experiments/reports/experiment_review_20260702_twinkl_w2mu_frozen_context_gap.md) | Frozen test-split LLM context results |
| [`../archive/evals/retired_wq9p_drift_benchmark_2026-07-11.md`](../archive/evals/retired_wq9p_drift_benchmark_2026-07-11.md) | Retired benchmark record; do not rerun, score, tune, or grant deployment approval from it |
