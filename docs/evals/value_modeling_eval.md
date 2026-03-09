# Value & Identity Modeling Evaluation

## What We're Evaluating

The VIF (Value Identity Function) maps journal entries to a 10-dimensional Schwartz value vector. This evaluation validates that the learned mapping captures the user's true value priorities.

---

## Implementation Status

**Status:** 🟡 In Progress (as of 2026-03-09)

### What's Implemented
- Evaluation specification complete (this document)
- Judge training data: 1 651 labeled entries across 204 personas in [`logs/judge_labels/judge_labels.parquet`](../../logs/judge_labels/judge_labels.parquet)
- Ground truth value orderings embedded in persona bios
- Critic architecture: MLP ordinal ([`src/vif/critic_ordinal.py`](../../src/vif/critic_ordinal.py)) and BNN ([`src/vif/critic_bnn.py`](../../src/vif/critic_bnn.py))
- Training pipeline spans 27 run IDs / 91 persisted run configs, including CORAL, CORN, EMD, CDW-CE, SoftOrdinal, BalancedSoftmax, LDAM-DRW, and legacy weighted-MSE baselines ([`logs/experiments/index.md`](../../logs/experiments/index.md))
- Text and state encoders: nomic-embed-text-v1.5, MiniLM ([`src/vif/encoders.py`](../../src/vif/encoders.py))
- Evaluation metrics: QWK, Spearman ρ, MAE, calibration, per-dimension recall, raw ordinal exports, and compact circumplex diagnostics ([`src/vif/eval.py`](../../src/vif/eval.py))
- Current corrected-split default: `run_019`-`run_021` BalancedSoftmax — median QWK **0.362**, median `recall_-1` **0.313**, median minority recall **0.448**, median hedging **0.621**, median calibration **0.713**
- Post-lift rebaseline `run_025`-`run_027` is logged and reviewable, but it did not replace the incumbent frontier because `Security` and circumplex structure regressed

### What's Missing
- **Corrected-split frontier still below target**: The active default has better minority recovery than the old pre-split baselines, but median QWK remains 0.362 and still sits below the moderate target range.
- **Hard dimensions remain unresolved**: `Hedonism` and especially `Security` still lag, and the latest regenerated targeted batch improved some local behavior without producing a cleaner overall frontier.
- **Circumplex structure is now measured, but not yet optimized**: recent reruns can improve one metric family while worsening opposite-pair violations or adjacent-pair support.
- Persona-level aggregation protocol (aggregate per-entry scores into persona-level value profile for Top-K accuracy)
- Formal held-out evaluation against declared value orderings (Spearman ρ > 0.7 target)

### Next Steps
1. Improve `Security`/`Hedonism` behavior without giving back corrected-split QWK or circumplex structure
2. Continue class-imbalance and decision-boundary work, but evaluate on the fixed-holdout targeted-lift regime rather than fresh reshuffles
3. Implement persona-level score aggregation across entries
4. Compute Spearman ρ between aggregated profiles and declared orderings
5. Report Top-K accuracy on synthetic personas

---

## Ground Truth

**Synthetic personas with known value profiles:**
- Each persona has a declared value ordering (e.g., Benevolence > Achievement > Self-Direction)
- This ordering is embedded in their bio and reflected in generated journal entries
- The declared ordering serves as ground truth for evaluation

**Why synthetic data works here:**
- Real users don't have objective value orderings we can measure
- Synthetic personas let us control the ground truth perfectly
- If the model captures synthetic personas correctly, it validates the mapping mechanism

---

## Metrics

Evaluation operates at two levels: **entry-level** metrics assess whether the Critic can correctly classify individual journal entries, while **persona-level** metrics assess whether aggregated entry scores recover the persona's declared value profile. Entry-level metrics are the current training bottleneck and the focus of ongoing experimentation.

### Entry-Level Metrics (Current Training Focus)

#### Quadratic Weighted Kappa (QWK)

QWK is the primary entry-level metric for ordinal classification. It measures agreement between predicted and true classes {-1, 0, +1}, adjusted for chance, with quadratic penalty for larger ordinal distances (predicting +1 when truth is -1 is penalised more than predicting 0).

**Why QWK over accuracy:** With severe class imbalance (neutral class is 60.5–88.3% per dimension), a model that predicts 0 for everything achieves high accuracy but zero QWK. QWK exposes this pathology.

**Interpretation:**
- QWK > 0.6: Good (substantial agreement beyond chance)
- QWK 0.4 – 0.6: Moderate (model captures some ordinal structure)
- QWK 0.2 – 0.4: Fair (better than chance but unreliable)
- QWK < 0.2: Poor (near-chance or degenerate predictions)

**Current corrected-split default:** median QWK 0.362 for `run_019`-`run_021` BalancedSoftmax. A historical pre-split high of 0.434 (`run_010` CORN) is kept in the experiment index for context only and is not the active evaluation regime.

#### Minority Recall

Minority recall measures the model's ability to detect non-neutral entries — the {-1, +1} classes that represent value misalignment and alignment. This is the product-critical metric: Twinkl's purpose is to surface tensions between behavior and values. A model that cannot detect -1 (misalignment) entries cannot fulfil this purpose, regardless of its QWK.

```
Minority Recall = mean of per-dimension recall for -1 and +1 classes
Recall_class = TP / (TP + FN) for that class
```

**Why this matters:** The -1 class ranges from 3.5% (Stimulation) to 13.8% (Self-direction) of labels. Models overwhelmingly hedge toward neutral (0), achieving >80% hedging rates. Minority recall directly measures whether the model can break through this tendency.

**Interpretation:**
- Recall > 30%: Reasonable detection of non-neutral entries
- Recall 10 – 30%: Poor; model catches some signal but misses most
- Recall < 10%: Model effectively ignores the minority class

**Current corrected-split default:** median `recall_-1` is 31.3% and median minority recall is 44.8% for `run_019`-`run_021` BalancedSoftmax. This is materially better than the old pre-split baselines on misalignment sensitivity, but still not strong enough to treat drift triggers as production-ready.

#### Circumplex Diagnostics

The VIF is not just a 10-head classifier. Its outputs should also respect the
structure of Schwartz's value circumplex:

- **Opposite-pair violation mean**: how often theoretically opposing values are
  activated in the same direction
- **Adjacent-pair support mean**: how often adjacent/compatible values are
  co-activated positively

Why this matters: a model can improve QWK or minority recall while still
producing value combinations that are structurally implausible.

**Current corrected-split default:** the incumbent `run_019`-`run_021`
BalancedSoftmax family sits at median `opposite_violation_mean = 0.070` and
median `adjacent_support_mean = 0.077`. The latest post-lift rerun
`run_025`-`run_027` did not replace it because opposite-pair violation worsened
to `0.082` without an adjacent-support gain.

### Persona-Level Metrics (Downstream Goal)

These metrics assess whether entry-level scores, when aggregated across a persona's journal entries, recover the persona's declared value profile. They depend on adequate entry-level performance: if QWK and minority recall are poor, persona-level aggregation inherits noisy inputs.

#### Spearman Correlation

Compare the model's predicted value rankings against ground truth orderings.

```
ρ = Spearman correlation between:
  - Predicted: Model's top-K value rankings for the persona
  - Ground truth: Declared value ordering from persona generation
```

**Interpretation:**
- ρ = 1.0: Perfect agreement (model predicts exact same ordering)
- ρ > 0.7: Strong correlation (acceptable for POC)
- ρ < 0.5: Weak correlation (model not capturing value priorities)

#### Top-K Accuracy

Does the model correctly identify the persona's top 1-3 values?

```
Top-1 Accuracy = % of personas where predicted #1 value matches declared #1 value
Top-3 Accuracy = % of personas where predicted top-3 contains all declared core values
```

---

## Evaluation Protocol

### Step 1: Generate Test Personas
- Create 3-5 synthetic personas with distinct value profiles
- Ensure diversity: vary age, culture, profession
- Each persona has 1-2 declared core values

### Step 2: Generate Journal Entries
- Generate 5-10 entries per persona
- Entries should naturally reflect their value priorities
- Use existing synthetic data pipeline

### Step 3: Run VIF Inference
- Process each persona's entries through the trained Critic
- Aggregate per-entry scores into a persona-level value profile
- Rank values by aggregated scores

### Step 4: Compute Metrics
```python
from scipy.stats import spearmanr

for persona in test_personas:
    predicted_ranking = model.predict_value_ranking(persona.entries)
    ground_truth = persona.declared_value_ordering
    rho, p_value = spearmanr(predicted_ranking, ground_truth)
    results.append(rho)

mean_rho = np.mean(results)
```

---

## Success Criteria

### Entry-Level (per-entry classification)

| Metric | Target | Rationale |
|--------|--------|-----------|
| QWK (mean) | > 0.4 (moderate) | Chance-corrected ordinal agreement; demonstrates model learns value structure beyond majority-class bias |
| Minority Recall (-1) | > 20% | Model can detect value misalignment — the signal Twinkl exists to surface |
| Minority Recall (+1) | > 40% | Model can detect value alignment with reasonable sensitivity |

**Why these thresholds:** At QWK > 0.4 with meaningful minority recall, the Critic's per-entry scores are reliable enough that persona-level aggregation can compensate for remaining entry-level noise. Below these thresholds, aggregation inherits too much bias toward neutral.

### Persona-Level (aggregated profile recovery)

From PRD (Evaluation Strategy, Row 2):

| Metric | Target | Rationale |
|--------|--------|-----------|
| Spearman ρ | > 0.7 on 3-5 personas | Strong correlation indicates model captures declared priorities |
| Top-1 Accuracy | > 60% | Model identifies the most important value more often than chance |

---

## Known Limitations

1. **Synthetic bias**: Model may learn artifacts of the generation process rather than true value signals
2. **Small sample size**: 3-5 personas limits statistical power
3. **Value leakage**: If personas explicitly mention values in entries, the evaluation is trivial

**Mitigations:**
- Use banned terms validation to prevent explicit value mentions
- Qualitatively review entries to ensure values are shown through behavior, not stated
- Report confidence intervals alongside point estimates

---

## References

- `docs/vif/03_model_training.md` — Training approach and loss function
- `docs/prd.md` — Evaluation Strategy table (Row 2)
- `config/schwartz_values.yaml` — Value dimension definitions
