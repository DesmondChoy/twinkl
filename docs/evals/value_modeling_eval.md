# Value & Identity Modeling Evaluation

## What We're Evaluating

The VIF (Value Identity Function) maps journal entries to a 10-dimensional Schwartz value vector. This evaluation validates that the learned mapping captures the user's true value priorities.

---

## Implementation Status

**Status:** 🔴 Blocked

### What's Implemented
- Evaluation specification complete (this document)
- Judge training data ready (904 labeled entries in [`logs/judge_labels/judge_labels.parquet`](../../logs/judge_labels/judge_labels.parquet))
- Ground truth value orderings embedded in persona bios

### What's Missing
- Critic/MLP training code
- Spearman correlation calculation
- Top-K accuracy metrics
- VIF inference pipeline

### Blocking Dependencies
Requires Critic model implementation — the neural network that maps text embeddings to value scores is not yet built.

### Next Steps
1. Implement Critic architecture in `src/vif/`
2. Train Critic on Judge-labeled data
3. Implement evaluation metrics (Spearman ρ, Top-K accuracy)
4. Run evaluation on held-out synthetic personas

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
