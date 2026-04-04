# Alignment & Drift Detection Evaluation

## What We're Evaluating

The VIF detects when a user's behavior drifts from their declared values. This evaluation validates that the dual-trigger system (Sudden Crash + Chronic Rut) correctly identifies misalignment episodes.

For the current project scope, this evaluation is about crash/rut-style drift detection only. A separate evolution-gating concept exists in the docs, but it is currently undecided and should be treated as an idea rather than part of the active implementation or evaluation plan.

---

## Implementation Status

**Status:** 🟡 Partial

### What's Implemented
- Evaluation specification complete (this document)
- Conceptual design documented in [`docs/vif/04_uncertainty_logic.md`](../vif/04_uncertainty_logic.md)
- Trigger formulas defined (Crash: V_{t-1} - V_t > δ, Rut: sustained low)
- Experiment archive: 50 run IDs / 114 persisted configs, with corrected-split frontier runs and later diagnostics tracked in [`logs/experiments/index.md`](../../logs/experiments/index.md)
- MC Dropout uncertainty estimation: [`src/vif/critic.py:predict_with_uncertainty()`](../../src/vif/critic.py) and [`src/vif/eval.py:evaluate_with_uncertainty()`](../../src/vif/eval.py)
- Uncertainty-gated crash/rut-style routing experiments in [`src/vif/drift.py`](../../src/vif/drift.py)
- Weekly runtime bridge from Critic checkpoint -> timeline -> weekly signals in [`src/vif/runtime.py`](../../src/vif/runtime.py)
- Full offline Coach path from VIF output -> drift result -> weekly digest artifact in [`src/coach/runtime.py`](../../src/coach/runtime.py)
- Unit coverage for drift behavior in [`tests/vif/test_drift.py`](../../tests/vif/test_drift.py)
- Calibration and circumplex summaries implemented and tracked per run in the experiment index and run YAMLs

### What's Missing
- Crisis injection test data generation
- Threshold calibration against a labeled weekly benchmark
- Hit rate / precision / recall reporting on injected timelines
- A project decision on whether evolution gating belongs in scope at all

### Blocking Dependencies
The active corrected-split default (`run_019`-`run_021` BalancedSoftmax) improved misalignment sensitivity, but the frontier still sits at median QWK **0.362** with unresolved `Security`/circumplex trade-offs. That is not yet strong enough for reliable automated drift triggers. Until per-value Critic accuracy improves further, crash/rut detection will inherit noisy alignment scores and produce unreliable alerts.

Separately, evolution gating should not be treated as a dependency for the current drift evaluation. It remains an undecided idea rather than an active implementation commitment.

### Next Steps
1. Improve upstream Critic reliability through the current frontier follow-ups (`twinkl-748`, `twinkl-749`, `twinkl-751`, and the gated `twinkl-750` path)
2. Generate synthetic crisis-injection timelines with explicit ground-truth crisis weeks
3. Run the implemented `predict_persona_timeline()` -> `aggregate_timeline_by_week()` -> `detect_weekly_drift()` path end to end on those timelines
4. Tune crash/rut thresholds on the injected benchmark
5. Report hit rate, precision, recall, F1, and false-positive rate on the calibrated benchmark

---

## Drift Detection System Overview

### Two Trigger Types

| Trigger | Definition | Formula |
|---------|------------|---------|
| **Sudden Crash** | Sharp negative change in alignment | V_{t-1} - V_t > δ_j |
| **Chronic Rut** | Sustained low values over time | V_t < τ_low for ≥ C_min consecutive weeks |

### Uncertainty Gating

Both triggers require **low uncertainty** (σ < ε_j) to fire. High variance suppresses critiques to avoid false alarms on ambiguous or OOD inputs.

### Possible Future Extension: Evolution Gating

A separate idea documented in [`docs/evolution/01_value_evolution.md`](../evolution/01_value_evolution.md) is to classify recent divergence as STABLE, EVOLUTION, or DRIFT before evaluating crash/rut triggers. If the project chooses to revisit that idea later, the intended goal would be to suppress false-positive rut alerts when a user's priorities have genuinely changed. It is not part of the current drift-evaluation contract.

---

## Ground Truth: Synthetic Crisis Injection

### Methodology

1. **Baseline period**: Generate 4-6 weeks of "aligned" entries for each synthetic persona
2. **Crisis injection**: At week N, introduce systematic misalignment:
   - For "Benevolence-first" persona: inject work-obsession entries that neglect relationships
   - For "Achievement-first" persona: inject entries showing stagnation or avoidance
3. **Label**: Mark injected weeks as ground-truth "crisis weeks"

### Example Crisis Scenarios

| Persona Values | Crisis Injection | Expected Detection |
|---------------|------------------|-------------------|
| Benevolence > Achievement | Persona cancels on friends, works late repeatedly | Benevolence rut or crash |
| Security > Stimulation | Persona takes risky financial decisions | Security crash |
| Self-Direction > Conformity | Persona defers all decisions to others | Self-Direction rut |
| Achievement > Benevolence | Persona gradually prioritizes Benevolence over career after having a child | Current eval: do not count as a crash/rut hit unless the active crash/rut detector fires; future evolution-gating idea would treat this as a non-drift value shift |
| Self-Direction high | Persona oscillates +1/−1 on Self-Direction week-to-week | Current eval: stress-test crash/rut robustness on volatile timelines |
| Hedonism > Security | Persona steadily reduces hedonistic activities over 6+ weeks | Current eval: treat as a boundary-case timeline; future evolution-gating idea would likely route it away from rut alerts |

---

## Metrics

### Primary: Hit Rate (from PRD)

```
Hit Rate = (# of ground-truth crisis weeks correctly flagged) / (# of total crisis weeks)
```

**Target: ≥ 8/10 (80%)**

### Secondary: Per-Value Precision & Recall

For each value dimension j:

```
Precision_j = TP_j / (TP_j + FP_j)  # When we flag, are we right?
Recall_j = TP_j / (TP_j + FN_j)     # Do we catch all crises?
F1_j = 2 * (Precision_j * Recall_j) / (Precision_j + Recall_j)
```

### Tertiary: False Positive Rate

```
FPR = (# of non-crisis weeks incorrectly flagged) / (# of total non-crisis weeks)
```

**Target: < 20%** (users shouldn't be bombarded with false alarms)

---

## Evaluation Protocol

### Step 1: Generate Test Timelines

For each of 3-5 synthetic personas:
- Generate 8-10 weeks of journal entries
- Inject 2-3 crisis weeks at known positions
- Ensure crisis severity varies (some obvious, some subtle)

### Step 2: Run Drift Detection

```python
for persona in test_personas:
    timeline_df, _meta = predict_persona_timeline(
        persona_id=persona.persona_id,
        checkpoint_path=checkpoint_path,
        wrangled_dir=wrangled_dir,
    )
    weekly_df = aggregate_timeline_by_week(timeline_df)

    for week_end in persona.week_ends:
        result = detect_weekly_drift(
            weekly_df,
            target_week_end=week_end,
        )
        if result.trigger_type in {"crash", "rut"}:
            detections.append((persona.persona_id, week_end, result.trigger_type))
```

### Step 3: Compute Metrics

```python
# Ground truth crisis weeks
ground_truth = set(persona.crisis_weeks for persona in test_personas)

# Detected weeks
detected = set(detections)

# Metrics
hits = len(ground_truth & detected)
hit_rate = hits / len(ground_truth)
precision = hits / len(detected) if detected else 0
recall = hit_rate
fpr = len(detected - ground_truth) / len(all_weeks - ground_truth)
```

---

## Success Criteria

| Metric | Target | Rationale |
|--------|--------|-----------|
| Hit Rate | ≥ 80% (8/10) | From PRD evaluation strategy |
| Precision | > 60% | Minimize false alarms |
| F1 per value | > 0.5 | Balanced detection |
| FPR | < 20% | User experience |

---

## Threshold Tuning

The following parameters need tuning on synthetic data:

| Parameter | Symbol | Description | Starting Value |
|-----------|--------|-------------|----------------|
| Crash threshold | δ_j | Minimum drop to trigger crash | 0.5 |
| Rut threshold | τ_low | Value below which counts as "low" | -0.4 |
| Rut duration | C_min | Consecutive weeks needed | 3 |
| Uncertainty ceiling | ε_j | Maximum uncertainty to allow critique | 0.3 |

**Tuning approach**: Grid search over synthetic personas to maximize F1 while keeping FPR acceptable.

---

## Uncertainty Validation

### Goal
Verify that MC Dropout uncertainty correlates with prediction errors.

### Method
1. Run 50 forward passes per entry
2. Compute variance σ² for each prediction
3. Compute actual error |predicted - ground_truth|
4. Calculate Pearson correlation between σ² and error

**Expected**: Positive correlation (r > 0.3). Higher uncertainty should predict larger errors.

### Bimodal Scenario Test
- Create entries with conflicting signals (e.g., "worked 100 hours but spent quality time with family")
- Verify these produce high variance (model is "confused")
- Confirm system avoids issuing confident critiques on such entries

---

## Known Limitations

1. **Synthetic crisis injection is artificial**: Real drift may be more subtle and gradual
2. **Profile-weighted detection**: Requires accurate value profiles; errors in w_u propagate
3. **Cold start**: New users have no history for EMA calculation
4. **Evolution handling is undecided**: A separate evolution-gating idea exists, but it is not part of the current committed drift-evaluation path.

**Mitigations:**
- Vary crisis severity in test set (include subtle cases)
- Test with perturbed value profiles to assess robustness
- Define fallback rules for cold-start period

---

## References

- `docs/vif/04_uncertainty_logic.md` — Uncertainty, drift formulas, and trigger logic
- `docs/evolution/01_value_evolution.md` — Concept note for a possible future evolution-vs-drift filter
- `docs/prd.md` — Evaluation Strategy (Row 3: Drift detection)
