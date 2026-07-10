# Alignment and Drift Detection Evaluation

## Evaluation Contract

Twinkl evaluates one v1 definition of drift:

> A sustained conflict episode occurs when a declared core or high-weight value
> receives two consecutive consensus `-1` reference labels.

The three layers of the contract are deliberately different:

| Layer | Contract |
|---|---|
| Reference labels | Two consecutive consensus `-1` labels on the same declared core/high-weight value |
| Runtime detector | Rolling soft `P(-1)` evidence under uncertainty gating |
| User delivery | The weekly Coach digest cites the relevant entries and reflects the conflict without score jargon |

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

- The five-pass consensus reference table is available at
  `logs/judge_labels/consensus_labels.parquet`.
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

### Current Prototype Boundary

`src/vif/drift.py` implements an experimental weekly router with literal output
modes `stable`, `crash`, `rut`, `evolution`, and `high_uncertainty`. It also
invokes the experimental evolution classifier automatically. That router is a
working prototype and remains useful for end-to-end UI and schema testing, but
it is not the selected sustained-conflict detector.

The six-detector comparison in `src/demo_tool/multi_drift.py` is another
exploratory surface. Its per-entry vote count is detector agreement, not the
five-pass Judge consensus reference.

### Missing for v1

- Per-entry `P(-1)` persistence in the runtime artifact surface
- A rolling soft-evidence detector aligned with the sustained-conflict reference
- Uncertainty and probability-mass thresholds calibrated on held-out data
- An event-matching harness that compares predicted episodes with reference
  episodes by persona, value dimension, and time window
- A scripted held-out episode set that is independent of the Judge labels used
  to choose and tune the definition
- End-to-end hit rate, precision, recall, F1, false-positive rate, and alert
  latency reporting
- Coach-language checks that distinguish active conflict, recovery, and stable
  weeks at digest time

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

For each persona and declared core/high-weight value:

1. Sort entries by `t_index` and date.
2. Read the consensus label for that value on each entry.
3. Start an episode when two consecutive labels are `-1`.
4. Extend the episode while consecutive `-1` labels continue.
5. End the episode when the label returns to `0` or `+1`.
6. Keep persona ID, value dimension, onset entry, end entry, dates, and the
   supporting journal rows.

This event table is the reference surface for detector evaluation. A weekly
digest is considered timely when it surfaces an active episode during the
delivery week or within the allowed latency window.

`consensus_agreement_*` fields can weight event confidence. They are not the
full class distribution. Soft target probabilities require the per-pass vote
files from the consensus rerun bundle.

---

## Runtime Detector Target

For dimension `j` at entry `t`, let `p^-_{t,j}` be the Critic probability of
class `-1` and `u_{t,j}` be its uncertainty estimate. A v1 detector accumulates
recent negative evidence only when:

- the dimension is declared core or its profile weight passes the importance
  floor;
- uncertainty is below the calibrated ceiling; and
- the recent `P(-1)` mass passes a persistence threshold.

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

### Phase 3: Unbiased Scripted Holdout

1. Create trajectories with explicitly scripted sustained-conflict episodes and
   matched non-conflict controls.
2. Keep the scripted set isolated from threshold selection and prompt tuning.
3. Include easy, subtle, recovery, and volatile cases.
4. Report event metrics and Coach evidence quality on this holdout once.

The label-derived candidates are adequate for tuning. They are not a final
unbiased benchmark because the same Judge regime defines the reference.

---

## Metrics and Targets

### Primary Event Metrics

| Metric | Target | Meaning |
|---|---:|---|
| Episode hit rate / recall | `>= 80%` | At least 8 of 10 held-out sustained-conflict episodes are detected |
| Precision | `> 60%` | Most surfaced episodes match a reference conflict |
| Event F1 | `> 0.5` | Precision and recall remain jointly useful |
| False-positive rate | `< 20%` | Non-conflict windows rarely produce alerts |
| First-alert latency | `<= 2 entries` | The detector reacts soon after reference onset |

### Required Slices

- Per Schwartz value dimension
- Core-value rank or profile-weight band
- Episode length and severity
- Consensus agreement tier
- Active conflict versus recovered-by-digest-time cases
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

---

## Limitations

1. Consensus labels are a more stable Judge reference, not human ground truth.
2. Current checkpoints were trained on persisted single-pass labels, so
   consensus evaluation mixes model error with label-regime shift.
3. Core-gated per-dimension denominators are small and uneven.
4. Five personas have at most two entries and cannot express a two-step episode.
5. The current synthetic corpus contains volatility more readily than clean,
   gradual arcs; it cannot validate fade or value-evolution claims.
6. Weekly Coach delivery can lag entry-level onset, so event matching must
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
| [`logs/experiments/reports/experiment_review_20260702_twinkl_w2mu_frozen_context_gap.md`](../../logs/experiments/reports/experiment_review_20260702_twinkl_w2mu_frozen_context_gap.md) | Frozen test-split LLM context results |
