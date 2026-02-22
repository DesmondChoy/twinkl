# Value & Identity Modeling Evaluation

## What We're Evaluating

The VIF (Value Identity Function) maps journal entries to a 10-dimensional Schwartz value vector. This evaluation validates that the learned mapping captures the user's true value priorities.

---

## Implementation Status

**Status:** 🟡 In Progress (as of 2026-02-22)

### What's Implemented
- Evaluation specification complete (this document)
- Judge training data: 1 460 labeled entries across 180 personas in [`logs/judge_labels/judge_labels.parquet`](../../logs/judge_labels/judge_labels.parquet)
- Ground truth value orderings embedded in persona bios
- Critic architecture: MLP ordinal ([`src/vif/critic_ordinal.py`](../../src/vif/critic_ordinal.py)) and BNN ([`src/vif/critic_bnn.py`](../../src/vif/critic_bnn.py))
- Training pipeline with 5 loss functions: CORAL, CORN, EMD, SoftOrdinal, weighted MSE ([`src/vif/train.py`](../../src/vif/train.py))
- Text and state encoders: nomic-embed-text-v1.5, MiniLM ([`src/vif/encoders.py`](../../src/vif/encoders.py))
- Evaluation metrics: QWK, Spearman ρ, MAE, calibration, per-dimension breakdowns ([`src/vif/eval.py`](../../src/vif/eval.py))
- 9 experiment runs (37 configurations) logged in [`logs/experiments/index.md`](../../logs/experiments/index.md)
- Best result: run_007 CORN — QWK **0.413**, Cal 0.838, MAE 0.205

### What's Missing
- **QWK below target**: Best QWK is 0.413 (run_007 CORN); further experimentation is in progress as of 2026-02-22 to improve per-value discrimination
- Persona-level aggregation protocol (aggregate per-entry scores into persona-level value profile for Top-K accuracy)
- Formal held-out evaluation against declared value orderings (Spearman ρ > 0.7 target, current best 0.402)

### Next Steps
1. Boost Critic QWK through ongoing experiments (data expansion, architecture tuning, loss function exploration)
2. Implement persona-level score aggregation across entries
3. Compute Spearman ρ between aggregated profiles and declared orderings
4. Report Top-K accuracy on synthetic personas

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

### Primary: Spearman Correlation

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

### Secondary: Top-K Accuracy

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
