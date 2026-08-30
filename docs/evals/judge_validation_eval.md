# LLM-Judge Validation Evaluation

## What We're Evaluating

The LLM-Judge produces LLM-Judge VIF Labels for the VIF Critic (Offline). This evaluation
measures repeated-call consistency and overlap with a shared project-team
annotation subset. The subset is not independent human ground truth, so the
evaluation does not establish label accuracy by itself.

---

## Implementation Status

**Status:** 🟢 Operational

### What's Implemented
- LLM-Judge labeling workflow operational ([`src/judge/consolidate.py`](../../src/judge/consolidate.py))
- 1 651 labeled Journal Entries across 204 personas in [`logs/judge_labels/judge_labels.parquet`](../../logs/judge_labels/judge_labels.parquet)
- 1 651 confidence-tiered consensus labels in [`logs/judge_labels/consensus_labels.parquet`](../../logs/judge_labels/consensus_labels.parquet)
- Data models with rationale support ([`src/models/judge.py`](../../src/models/judge.py))
- Human annotation tool ([`src/annotation_tool/app.py`](../../src/annotation_tool/app.py))
- 3 annotators × 115 shared Journal Entries across 19 personas (des, jl, km) in [`logs/annotations/`](../../logs/annotations/)
- Cohen's κ and Fleiss' κ calculation ([`src/annotation_tool/agreement_metrics.py`](../../src/annotation_tool/agreement_metrics.py))
- Agreement report: avg Cohen's κ 0.66, Fleiss' κ 0.56 ([report](../../logs/exports/agreement_report_20260318_130642.md))
- Historical reachability audit completed for hard dimensions with a diagnostic recommendation to change the target regime for `security` and use targeted relabeling for `hedonism` / `stimulation` ([report](../../logs/exports/twinkl_747/reachability_audit_report.md)); its legacy experiment setups did not exactly match the active VIF Critic (Offline) input and did not create a repaired target
- Repeated-call self-consistency and full-corpus stability analysis completed via the 5-pass `twinkl-754` rerun ([report](../../logs/exports/twinkl_754/consensus_rejudging_report.md))
- Receipt-bound exact-state Security review and the
  `security_active_critic_state_v1` target completed under `twinkl-a30f`
- Matched Hedonism hard-set review completed under `twinkl-748`; the result did
  not establish a stronger VIF Critic (Offline) operating point

### What's Missing
- Automated quality checks (all-zero rate, sparsity, distribution)
- Optional human re-annotation after future rubric or target changes

### Next Steps
1. Add automated quality checks as post-labeling validation
2. Re-annotate a subset only if a future rubric or target change requires a new
   agreement measure

No further VIF Critic (Offline) target work is planned for the time-boxed capstone.

---

## The 3-Point Categorical Rubric

The LLM-Judge uses a strict categorical protocol to reduce subjective noise:

| Score | Meaning | Criteria |
|-------|---------|----------|
| **+1** | **Aligned** | Entry actively supports/demonstrates this value through behavior |
| **0** | **Neutral** | Entry is irrelevant to this value OR maintains status quo |
| **-1** | **Misaligned** | Entry actively conflicts with/neglects this value |

**Why categorical?**
- Clear decision boundaries improve consistency
- Avoids arbitrary distinctions (4.5 vs 4.2)
- Maps cleanly to training targets for the VIF Critic (Offline)

---

## Validation Approaches

### 1. Agreement Metrics (Cohen's κ and Fleiss' κ)

Two complementary kappa metrics describe agreement on the shared subset:

#### Understanding the Two Metrics

```
                    Fleiss' κ
        (Do humans agree with EACH OTHER?)
           ┌──────────┼──────────┐
           │          │          │
      Annotator 1  Annotator 2  Annotator 3
           │          │          │
           └────┬─────┴─────┬────┘
                │           │
             Cohen's κ   Cohen's κ
      (Does THIS human agree with LLM-Judge?)
                │           │
                └─────┬─────┘
                      │
                   LLM-Judge
```

| Metric | What It Measures | Question It Answers |
|--------|------------------|---------------------|
| **Cohen's κ** | Agreement between one project-team annotator and the LLM-Judge | "How closely do this annotator's labels overlap with the LLM-Judge VIF Labels?" |
| **Fleiss' κ** | Agreement among all three project-team annotators | "How consistently do the annotators apply the rubric on this subset?" |

#### Why Both Metrics Matter

| Scenario | Fleiss' κ | Cohen's κ | Interpretation |
|----------|-----------|-----------|----------------|
| Project-team annotators agree with each other and with LLM-Judge | High | High | Both agreement views are strong on the sampled data |
| Project-team annotators agree with each other but not with LLM-Judge | High | Low | Inspect LLM-Judge errors, prompt behavior, and label provenance |
| Project-team annotators show weak agreement | Low | Varies | Inspect rubric ambiguity, class prevalence, sample coverage, and annotator differences |

The two coefficients use different rater structures and are not a direct model
comparison. A low value can reflect rubric ambiguity, class prevalence, sample
composition, annotator differences, or LLM-Judge behavior. Interpret the
aggregate values with per-value coefficients and case-level disagreements.

#### Interpretation Scale (Landis & Koch, 1977)

**Target: κ > 0.60** (substantial agreement)

```
κ interpretation:
  < 0.00: Poor (worse than chance)
  0.00-0.20: Slight
  0.21-0.40: Fair
  0.41-0.60: Moderate
  0.61-0.80: Substantial  ← Target
  0.81-1.00: Almost perfect
```

#### Running Agreement Analysis

```bash
# Generate agreement report
uv run shiny run src/annotation_tool/app.py
# → Navigate to Analysis section → Export Agreement Report
```

Reports are saved to `logs/exports/agreement_report_<timestamp>.md`.

### 2. Internal Consistency

Check that the LLM-Judge produces consistent scores when the same workflow is rerun:

- **Repeated-call reruns**: `twinkl-754` reran the profile-only LLM-Judge workflow 5 times over all 1,651 Journal Entries and measured per-dimension Fleiss' κ from **0.775** (`security`) to **0.890** (`universalism`). This is the main repeated-call consistency benchmark in the repository.
- **Semantically similar Journal Entries**: Paraphrased versions of same content should get same scores.

### 3. Distribution Checks

Verify the LLM-Judge doesn't systematically over- or under-predict:

- **All-zero check**: Flag Journal Entries scored 0 across all dimensions (may indicate an overly conservative LLM-Judge)
- **Sparsity check**: Flag personas where >80% of Journal Entries are all-zero
- **Distribution check**: Scores should roughly match expected base rates per value

---

## Manual Review Protocol

### Sample Selection

From each synthetic data run, review 10-20 Journal Entries across different:
- Personas (variety of value profiles)
- Entry types (standalone vs. nudged)
- Tones (positive, negative, neutral)

### Review Questions

For each Journal Entry, the human reviewer asks:

1. **Do the non-zero scores match your intuition?**
   - Which values does this Journal Entry clearly touch?
   - Did the LLM-Judge identify them correctly?

2. **Did the LLM-Judge conflate similar values?**
   - Achievement vs. Power (both about success but different motivations)
   - Benevolence vs. Universalism (both about helping but different scope)
   - Conformity vs. Tradition (both about social norms but different anchors)

3. **Are similar Journal Entries getting similar scores?**
   - Compare Journal Entries with similar content across personas
   - Flag inconsistencies

### Documentation Template

```markdown
## Entry Review: Persona 3, Entry 5

**Entry**: "Stayed late again to finish the deck. Told myself I'd leave by 6..."

**LLM-Judge VIF Labels**: Achievement: +1, Hedonism: -1, all others: 0

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

### Step 0: Diagnose the Root Cause

Inspect Fleiss' κ, Cohen's κ, class prevalence, sample coverage, and exact
disagreements together before assigning a cause:

| Fleiss' κ | Cohen's κ | Diagnosis | Action |
|-----------|-----------|-----------|--------|
| Low | Low | Rubric, prevalence, sample, annotator, or LLM-Judge issue | Review cases before choosing Step 1 or Step 2 |
| High | Low | Possible LLM-Judge divergence | Review cases, provenance, and prompt behavior before Step 2 |
| High | High | Strong sampled agreement | Retain provenance and per-value monitoring |

### Step 1: Refine Rubrics (if Fleiss' κ is low)

When case-level review supports rubric ambiguity, clarify the definitions:

1. **Identify problem values**: Look at per-value Fleiss' κ breakdown
2. **Update `config/schwartz_values.yaml`** with:
   - Clearer distinguishing criteria
   - Concrete examples of edge cases
   - Anti-patterns to avoid
3. **Re-annotate subset**: Have humans re-label 10-20 Journal Entries
4. **Check Fleiss' κ improvement**: Repeat until κ > 0.60

### Step 2: Fix the LLM-Judge (if Cohen's κ is low but Fleiss' κ is high)

When case-level review supports LLM-Judge divergence, adjust the LLM-Judge
prompt and verify the affected cases:

1. **Identify pattern**: What type of Journal Entries does the LLM-Judge struggle with?
   - Value conflation (Achievement ↔ Power)
   - Tone sensitivity (sarcasm, hedging)
   - Cultural context
2. **Update LLM-Judge prompt**: Add constraints or clarifications
3. **Re-run on flagged Journal Entries**: Verify improvement before full re-labeling

---

## Planned Automated Quality Checks

The post-labeling validation does not yet run these checks automatically. The
intended checks are:

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

## Interpretation Thresholds

| Metric | Target | Rationale |
|--------|--------|-----------|
| Fleiss' κ (project-team annotators) | > 0.60 | Descriptive substantial-agreement threshold for a future shared subset |
| Cohen's κ (project-team annotator vs LLM-Judge) | > 0.60 | Descriptive substantial-agreement threshold for each pair |
| Repeated-call self-consistency | Fleiss' κ > 0.75 | The same LLM-Judge workflow should remain strongly stable when rerun on the same data |
| All-zero rate | < 30% | Most Journal Entries have signal |
| Per-persona sparsity | < 20% with >80% zeros | Personas should show patterns |

Interpret Fleiss' κ, each Cohen's κ, prevalence, sample coverage, and exact
disagreements together. Neither coefficient establishes label accuracy or a
ceiling for the other.

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
- [ ] Test the LLM-Judge on 3-5 sample Journal Entries manually

**During labeling run:**
- [ ] Monitor for API errors or timeouts
- [ ] Spot-check first 10 Journal Entries per persona

**After labeling run:**
- [ ] Run automated quality checks
- [ ] Manual review of 10-20 Journal Entries
- [ ] Calculate κ on reviewed sample
- [ ] Document any rubric refinements needed

---

## References

- `docs/pipeline/judge_implementation_spec.md` — LLM-Judge implementation details
- `docs/pipeline/annotation_tool_plan.md` — Annotation tool implementation plan
- `src/annotation_tool/agreement_metrics.py` — κ calculation implementation
- `config/schwartz_values.yaml` — Value rubrics and elaborations
- `logs/exports/agreement_report_*.md` — Generated agreement reports
