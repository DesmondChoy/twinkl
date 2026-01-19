# LLM-as-Judge Validation Evaluation

## What We're Evaluating

The Judge (LLM-as-Judge) produces training labels for the VIF. This evaluation validates that the Judge's alignment scores are consistent, accurate, and agree with human intuition.

---

## Implementation Status

**Status:** 🟡 Partial

### What's Implemented
- Judge labeling pipeline operational ([`src/judge/consolidate.py`](../../src/judge/consolidate.py))
- Rationale storage in parquet ([`logs/judge_labels/judge_labels.parquet`](../../logs/judge_labels/judge_labels.parquet))
- Data models with rationale support ([`src/models/judge.py`](../../src/models/judge.py))

### What's Missing
- Cohen's κ calculation (no inter-rater agreement code)
- Automated quality checks (all-zero rate, sparsity, distribution)
- Consistency checks (re-running same entry multiple times)

### Blocking Dependencies
Needs human annotations for κ comparison — currently only have Judge labels, no human ground truth to compare against.

### Next Steps
1. Complete human annotation of 20-30 entries using the annotation tool
2. Implement κ calculation function in `src/judge/`
3. Add automated quality checks as post-labeling validation

---

## The 3-Point Categorical Rubric

The Judge uses a strict categorical protocol to reduce subjective noise:

| Score | Meaning | Criteria |
|-------|---------|----------|
| **+1** | **Aligned** | Entry actively supports/demonstrates this value through behavior |
| **0** | **Neutral** | Entry is irrelevant to this value OR maintains status quo |
| **-1** | **Misaligned** | Entry actively conflicts with/neglects this value |

**Why categorical?**
- Clear decision boundaries improve consistency
- Avoids arbitrary distinctions (4.5 vs 4.2)
- Maps cleanly to regression targets for the Critic

---

## Validation Approaches

### 1. Inter-Rater Agreement (Cohen's κ)

Compare Judge labels against human annotations on a sample of entries.

**Target: κ > 0.60** (substantial agreement)

```
κ interpretation:
  < 0.20: Poor
  0.21-0.40: Fair
  0.41-0.60: Moderate
  0.61-0.80: Substantial  ← Target
  0.81-1.00: Almost perfect
```

### 2. Internal Consistency

Check that the Judge produces consistent scores for similar entries:

- **Same entry, multiple runs**: Run Judge on same entry 3x with temperature=0. All scores should match.
- **Semantically similar entries**: Paraphrased versions of same content should get same scores.

### 3. Calibration Checks

Verify the Judge doesn't systematically over- or under-predict:

- **All-zero check**: Flag entries scored 0 across all dimensions (may indicate overly conservative Judge)
- **Sparsity check**: Flag personas where >80% of entries are all-zero
- **Distribution check**: Scores should roughly match expected base rates per value

---

## Manual Review Protocol

### Sample Selection

From each synthetic data run, review 10-20 entries across different:
- Personas (variety of value profiles)
- Entry types (standalone vs. nudged)
- Tones (positive, negative, neutral)

### Review Questions

For each entry, the human reviewer asks:

1. **Do the non-zero scores match your intuition?**
   - Which values does this entry clearly touch?
   - Did Judge identify them correctly?

2. **Did Judge conflate similar values?**
   - Achievement vs. Power (both about success but different motivations)
   - Benevolence vs. Universalism (both about helping but different scope)
   - Conformity vs. Tradition (both about social norms but different anchors)

3. **Are similar entries getting similar scores?**
   - Compare entries with similar content across personas
   - Flag inconsistencies

### Documentation Template

```markdown
## Entry Review: Persona 3, Entry 5

**Entry**: "Stayed late again to finish the deck. Told myself I'd leave by 6..."

**Judge scores**: Achievement: +1, Hedonism: -1, all others: 0

**Reviewer assessment**:
- [x] Achievement +1: Agree — clearly prioritizing work performance
- [x] Hedonism -1: Agree — sacrificing present comfort for work
- [ ] Benevolence: Should this be -1? Implicitly neglecting relationships
- Notes: Consider adding Benevolence signal to rubric

**Agreement**: 2/2 non-zero scores correct, 1 potential miss
```

---

## Iteration Loop

If validation reveals quality issues:

1. **Identify pattern**: What type of entries does Judge struggle with?
   - Value conflation (Achievement ↔ Power)
   - Tone sensitivity (sarcasm, hedging)
   - Cultural context

2. **Refine rubrics**: Update `config/schwartz_values.yaml` with:
   - Clearer distinguishing criteria
   - More examples of edge cases
   - Anti-patterns to avoid

3. **Update prompt**: Add constraints or clarifications to Judge prompt

4. **Re-run on flagged entries**: Verify improvement before full re-labeling

---

## Automated Quality Checks

Run these checks automatically on every Judge labeling run:

```python
def validate_judge_output(labels_df):
    issues = []

    # Check 1: All-zero entries
    all_zero = labels_df[labels_df[value_columns].sum(axis=1) == 0]
    if len(all_zero) / len(labels_df) > 0.3:
        issues.append(f"High all-zero rate: {len(all_zero)/len(labels_df):.1%}")

    # Check 2: Per-persona sparsity
    for persona_id in labels_df['persona_id'].unique():
        persona_labels = labels_df[labels_df['persona_id'] == persona_id]
        zero_rate = (persona_labels[value_columns].sum(axis=1) == 0).mean()
        if zero_rate > 0.8:
            issues.append(f"Persona {persona_id} has {zero_rate:.0%} all-zero entries")

    # Check 3: Value distribution
    for col in value_columns:
        dist = labels_df[col].value_counts(normalize=True)
        if dist.get(0, 0) > 0.9:
            issues.append(f"{col} is >90% neutral — may be under-detected")

    return issues
```

---

## Success Criteria

| Metric | Target | Rationale |
|--------|--------|-----------|
| Cohen's κ (human vs Judge) | > 0.60 | Substantial agreement |
| Internal consistency | 100% | Same entry → same scores |
| All-zero rate | < 30% | Most entries have signal |
| Per-persona sparsity | < 20% with >80% zeros | Personas should show patterns |

---

## Known Limitations

1. **LLM consistency**: Even with temperature=0, some variance is possible
2. **Human subjectivity**: Inter-rater agreement among humans is also imperfect
3. **Value conflation**: Some values genuinely overlap; perfect separation is impossible

**Mitigations:**
- Use multiple human reviewers, report inter-annotator agreement
- Document expected conflation patterns as acceptable
- Focus on obvious errors, not edge cases

---

## Checklist

**Before labeling run:**
- [ ] Verify rubrics in `config/schwartz_values.yaml` are up to date
- [ ] Test Judge on 3-5 sample entries manually

**During labeling run:**
- [ ] Monitor for API errors or timeouts
- [ ] Spot-check first 10 entries per persona

**After labeling run:**
- [ ] Run automated quality checks
- [ ] Manual review of 10-20 entries
- [ ] Calculate κ on reviewed sample
- [ ] Document any rubric refinements needed

---

## References

- `docs/VIF/judge_implementation_spec.md` — Judge implementation details
- `docs/synthetic_data/annotation_guidelines.md` — Human annotation protocol
- `config/schwartz_values.yaml` — Value rubrics and elaborations
