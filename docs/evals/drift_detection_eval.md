# Alignment & Drift Detection Evaluation

## What We're Evaluating

The VIF detects when a user's behavior drifts from their declared values. This evaluation validates that the dual-trigger system (Sudden Crash + Chronic Rut) correctly identifies misalignment episodes.

---

## Implementation Status

**Status:** 🔴 Blocked

### What's Implemented
- Evaluation specification complete (this document)
- Conceptual design documented in [`docs/VIF/VIF_04_Uncertainty_Logic.md`](../VIF/VIF_04_Uncertainty_Logic.md)
- Trigger formulas defined (Crash: V_{t-1} - V_t > δ, Rut: sustained low)

### What's Missing
- All drift detection code (crash/rut trigger implementation)
- MC Dropout for uncertainty estimation
- Crisis injection test data generation
- Hit rate / precision / recall metric calculation

### Blocking Dependencies
Requires trained Critic with MC Dropout uncertainty — both the base model and uncertainty mechanism are prerequisites that don't exist yet.

### Next Steps
1. Complete Value Modeling Eval first (train Critic)
2. Add MC Dropout to Critic architecture
3. Implement dual-trigger detection in `src/vif/`
4. Generate synthetic crisis injection test data
5. Run evaluation on injected timelines

---

## Drift Detection System Overview

### Two Trigger Types

| Trigger | Definition | Formula |
|---------|------------|---------|
| **Sudden Crash** | Sharp negative change in alignment | V_{t-1} - V_t > δ_j |
| **Chronic Rut** | Sustained low values over time | V_t < τ_low for ≥ C_min consecutive weeks |

### Uncertainty Gating

Both triggers require **low uncertainty** (σ < ε_j) to fire. High variance suppresses critiques to avoid false alarms on ambiguous or OOD inputs.

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
    for week in persona.weeks:
        # Get Critic predictions with uncertainty
        alignment, uncertainty = critic.predict_with_uncertainty(week.entries)

        # Apply dual-trigger rules
        crash_triggered = check_crash(alignment, prev_alignment, delta_threshold)
        rut_triggered = check_rut(alignment, consecutive_low_count, tau_low)
        uncertainty_ok = uncertainty < epsilon

        # Log detection
        if (crash_triggered or rut_triggered) and uncertainty_ok:
            detections.append(week)
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

**Mitigations:**
- Vary crisis severity in test set (include subtle cases)
- Test with perturbed value profiles to assess robustness
- Define fallback rules for cold-start period

---

## References

- `docs/VIF/VIF_04_Uncertainty_Logic.md` — Dual-trigger rules and MC Dropout
- `docs/VIF/VIF_06_Profile_Conditioned_Drift_and_Encoder.md` — Drift formulas
- `docs/PRD.md` — Evaluation Strategy (Row 3: Drift detection)
