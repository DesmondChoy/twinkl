# Alignment and Drift Detection Evaluation

## Evaluation Contract

Twinkl evaluates one v1 definition of drift:

> A sustained conflict episode occurs when the same declared core value
> receives a qualifying conflict label on two adjacent journal entries.

For the strict reference, a conflict qualifies when the existing five-pass
Judge consensus resolver stores `-1` for that value. The resolver first decides
whether most passes are non-neutral, then selects the majority polarity among
those non-neutral votes; a polarity tie resolves to `0` with `no_majority`
confidence. Drift consumes the resolved label rather than rerunning vote
aggregation. This is distinct from the six-detector comparison's vote count.

The three layers of the contract are deliberately different:

| Layer | Contract |
|---|---|
| Reference labels | Stored five-pass Judge consensus `-1` labels for the same declared core value on two adjacent entries |
| Offline detector benchmark | Two-entry mean `P(-1)` with declared-core gating and a maximum uncertainty ceiling; implemented and evaluated |
| Production runtime | Selected scorer and detector are not approved or wired |
| User delivery | The weekly Coach digest cites the relevant entries and uses active, recovered, mixed, or uncertain wording without score jargon; exact schema implementation is pending |

The reference definition is strict and auditable. The runtime detector is soft
because the current Critic often hedges a true `-1` toward neutral. Weekly
delivery remains a product cadence rather than a requirement that the evidence
itself be grouped into multi-week averages.

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
- `src/vif/drift_benchmark.py` materializes strict reference episodes, retains
  every adjacent decision window, detects predicted episodes, matches them
  one-to-one, and reports episode and false-positive metrics.
- `scripts/experiments/drift_trigger_benchmark.py` calibrates one global
  two-entry `P(-1)` plus uncertainty rule on frozen validation personas and
  compares the incumbent MLP, persisted-label sibling, consensus-trained MLPs,
  and both LLM context arms.
- The locked designed holdout contains 10 reference episodes across all 10
  Schwartz dimensions plus 10 matched control trajectories. It was fixed
  before scoring and is explicitly author-designed rather than human ground
  truth.

### Current Prototype Boundary

`src/vif/drift.py` implements an experimental weekly router with literal output
modes `stable`, `crash`, `rut`, `evolution`, and `high_uncertainty`. It also
invokes the experimental evolution classifier automatically. That router is a
working prototype and remains useful for end-to-end UI and schema testing, but
it is not the selected sustained-conflict detector.

The six-detector comparison in `src/demo_tool/multi_drift.py` is another
exploratory surface. Its per-entry vote count is detector agreement, not the
five-pass Judge reference.

### Benchmark Result (`twinkl-wq9p`)

The frozen consensus split contains 52 strict episodes overall, but only six
on validation personas and five on test personas. Threshold selection on that
small validation surface produced conservative operating points:

- `run_020` predicted no frozen-test episode and detected 1/10 designed
  episodes;
- the two evaluated consensus-trained MLP variants detected 2/10 designed
  episodes, so hard-consensus retraining did not close the recall gap;
- both `gpt-5.4-mini` context arms detected 10/10 deliberately explicit
  designed episodes with no false alarms, but detected 0/5 consensus-derived
  frozen-test episodes.

That cross-set disagreement is the decision. The LLM result proves that clear,
observable sustained conflict is within the scorer's capability; it does not
validate the consensus-derived episodes or justify a production cascade. No
scorer is promotion-ready until the frozen reference episodes and designed
cases receive human review under the declared input contract. Full evidence is
in the [`twinkl-wq9p` report](../../logs/experiments/reports/experiment_review_2026-07-10_twinkl_wq9p.md).

### Still Missing for Product v1

- Per-entry `P(-1)` persistence in the runtime artifact surface
- Human review of the frozen-versus-designed benchmark disagreement
- A scorer and calibrated operating point that pass both target-validity and
  decision-level promotion checks
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

## Reference Event Construction

Construct the reference independently for each
`(persona_id, declared_core_value)`:

1. Use the declared-core set stored with the profile. Do not infer eligibility
   from the graded weight vector.
2. Sort the persona's entries by `t_index`, then date. Adjacent entries count
   even when they occur on the same day or cross a calendar-week boundary. V1
   has no maximum elapsed-time threshold.
3. For the value being evaluated, mark an entry as a qualifying conflict when
   `alignment_<value> == -1` in `consensus_labels.parquet`. Missing consensus
   rows are non-qualifying and must not be skipped to join conflicts on either
   side. A reference builder should report malformed or incomplete input.
4. Ignore every other value dimension when evaluating this value's run. An
   aligned second core value cannot cancel the conflict, and a conflict on a
   different value cannot complete the pair.
5. On the first qualifying conflict, remember a candidate onset. On the second
   adjacent qualifying conflict, confirm one episode whose `onset_entry` is the
   first entry and whose `confirmation_entry` is the second.
6. Extend that episode through further adjacent qualifying conflicts. Store the
   final qualifying conflict as `end_entry`. If the run reaches the end of the
   observed timeline, keep the episode open and active.
7. Any non-qualifying or missing entry breaks the run. A later qualifying pair
   starts a new episode. A broken run is not automatically a recovered episode:
   a stored `0` with `no_majority` confidence or missing evidence supports
   **uncertain**, while a resolved neutral or aligned result can support
   **recovered** at delivery time.
8. Keep each value's episodes as separate records. A persona may have multiple
   or simultaneous value-specific episodes.

These examples follow directly from the rules:

| Value sequence | Reference result |
|---|---|
| `-1, -1` | One episode; onset at the first entry, confirmed at the second |
| `-1, -1, -1` | One extended episode, not two overlapping episodes |
| `-1, 0, -1` | No episode |
| `-1, -1, 0, -1, -1` | Two episodes |
| Core A `-1, -1`; Core B `+1, +1` | One episode on Core A; Core B does not cancel it |
| Core A `-1, -1`; Core B `-1, -1` | Two simultaneous value-specific episodes |

This event table is the reference surface for detector evaluation. The first
observable reference trigger is `confirmation_entry`, so detector latency is
measured from confirmation rather than the candidate onset. A weekly digest is
considered timely when it surfaces an active episode during the delivery week
or within the allowed latency window.

`consensus_agreement_*` is confidence metadata, not a second eligibility gate
and not the full class distribution. Soft target probabilities require the
per-pass vote files from the consensus rerun bundle.

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
of the reference-label definition. Candidate forms include a two-entry mean,
an exponentially weighted sum, or a small cumulative evidence score. They all
produce the same output concept: active sustained conflict on a named value.

Hard argmax predictions are not the runtime contract. Requiring two predicted
`-1` classes would be brittle at the current `recall_-1` frontier.

---

## Evaluation Protocol

### Phase 1: Reference and Tuning Split

1. Reproduce the consensus trajectory EDA.
2. Materialize sustained-conflict reference episodes.
3. Split by persona so no person's entries appear in both tuning and evaluation
   sets.
4. Use label-derived conflict-heavy weeks and reference episodes for threshold
   tuning and error analysis only.
5. Stratify reports by value dimension because event prevalence is highly
   asymmetric.

### Phase 2: Critic in the Loop

1. Run a frozen Critic checkpoint over the same persona timelines.
2. Persist `P(-1)` and uncertainty per entry and dimension.
3. Run candidate soft-evidence policies without reading evaluation labels.
4. Match predicted episodes to reference episodes by persona, value, and onset
   tolerance.
5. Compare the active MLP frontier and the LLM context arms at this decision
   layer.

### Phase 3: Isolated Designed Holdout

1. Create trajectories with explicitly scripted sustained-conflict episodes and
   matched non-conflict controls.
2. Keep the scripted set isolated from threshold selection and prompt tuning.
3. Include easy, subtle, recovery, and volatile cases.
4. Report event metrics and Coach evidence quality on this holdout once.

The label-derived candidates are adequate for tuning. They are not independent
validation because the same Judge regime defines the reference. The designed
holdout is isolated from tuning, but it is author-designed and not yet
human-reviewed, so it is a capability probe rather than a final benchmark.

---

## Metrics and Targets

### Primary Event Metrics

| Metric | Target | Meaning |
|---|---:|---|
| Episode hit rate / recall | `>= 80%` | At least 8 of 10 held-out sustained-conflict episodes are detected |
| Precision | `> 60%` | Most surfaced episodes match a reference conflict |
| Event F1 | `> 0.5` | Precision and recall remain jointly useful |
| False-positive rate | `< 20%` | Non-conflict windows rarely produce alerts |
| First-alert latency | `<= 2 entries` | The detector reacts soon after `confirmation_entry`, when the reference episode first becomes observable |

### Required Slices

- Per Schwartz value dimension
- Declared-core-value rank or profile-weight band
- Episode length and severity
- Consensus agreement tier
- Active, recovered, mixed, and uncertain digest-time cases
- MLP versus LLM context arm

### Uncertainty Validation

Uncertainty gating must be evaluated on the `-1` class specifically. Global
calibration can look acceptable while the minority class is poorly calibrated.
Report:

- error rate by uncertainty decile;
- retained episode recall at each uncertainty ceiling;
- false-positive reduction from gating; and
- the number of true episodes suppressed by high uncertainty.

---

## Reproduction

Run the default consensus analysis with runtime-compatible week bins:

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

Run the decision-level benchmark from the locked fixture and cached scorer
artifacts:

```sh
uv run python scripts/experiments/drift_trigger_benchmark.py
```

Use `--execute-llm` only when the locked designed-holdout LLM response cache is
absent and fresh API inference is intended. Benchmark artifacts live under
`logs/experiments/artifacts/drift_trigger_benchmark_twinkl_wq9p_20260710/`.

---

## Limitations

1. Consensus labels are a more stable Judge reference, not human ground truth.
2. The incumbent `run_020` checkpoint was trained on persisted single-pass
   labels, so its consensus evaluation mixes model error with label-regime
   shift. The consensus-trained comparison arms reduce that mismatch but still
   miss most designed episodes.
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
| [`src/vif/drift_benchmark.py`](../../src/vif/drift_benchmark.py) | Strict reference builder, pair decisions, detector, matching, and event metrics |
| [`scripts/experiments/drift_trigger_benchmark.py`](../../scripts/experiments/drift_trigger_benchmark.py) | Frozen and designed-holdout benchmark orchestration |
| [`config/evals/drift_v1_designed_holdout.yaml`](../../config/evals/drift_v1_designed_holdout.yaml) | Locked 10-episode plus 10-control designed holdout |
| [`logs/experiments/reports/experiment_review_20260702_twinkl_w2mu_frozen_context_gap.md`](../../logs/experiments/reports/experiment_review_20260702_twinkl_w2mu_frozen_context_gap.md) | Frozen test-split LLM context results |
| [`logs/experiments/reports/experiment_review_2026-07-10_twinkl_wq9p.md`](../../logs/experiments/reports/experiment_review_2026-07-10_twinkl_wq9p.md) | Decision-level benchmark result and promotion recommendation |
