# Judge Validation Summary

**Last Updated:** 2026-04-02

**Purpose:** This document summarizes inter-rater reliability findings for the shared human-annotation subset. The results support using the LLM Judge as a scalable supervision source for the POC, while later reachability and consensus audits add important caveats for the hardest dimensions and for student-label reachability.

**Analysis Source:** `src/annotation_tool/agreement_metrics.py`
**Full Report:** `logs/exports/agreement_report_20260318_130642.md`
**Evaluation Spec:** [`docs/evals/judge_validation_eval.md`](judge_validation_eval.md)

---

## Key Findings

### Agreement Metrics (All 10 Dimensions)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Human-Human Agreement** (Fleiss' κ) | 0.56 | Moderate |
| **Judge-Human Agreement** (Avg Cohen's κ) | 0.66 | Substantial |

**Conclusion:** On the shared 115-entry subset, the Judge exceeds human-human consistency across 9 of 10 Schwartz value dimensions. This supports using automated labeling as a scalable supervision source for the POC, while later audits show that aggregate agreement alone does not guarantee that every hard-dimension label is a clean distillation target for the current student.

#### Why This Matters

The key insight is that **Fleiss' κ (0.56) establishes the ceiling of human consistency**---it measures how much humans agree with *each other*. This represents the inherent subjectivity in the labeling task; even trained annotators interpret the same journal entry differently.

**Cohen's κ (0.66) measures Judge-Human alignment**, averaged across all three annotators. Since the Judge achieves *higher* agreement with individual humans than humans achieve with each other, this implies:

1. **The Judge is not an outlier**---it sits "within the distribution" of human judgments, not outside it
2. **The Judge may capture consensus**---its labels likely approximate what a majority of humans would agree on, even when individual humans disagree
3. **Automated labels are defensible**---if we trust human-labeled data for training, we can trust Judge-labeled data at least as much, since the Judge is more consistent with humans than they are with themselves

In practical terms: the Judge is strong enough to replace large-scale manual labeling for most dimensions in the current POC, and it may even *reduce* some inter-annotator noise. However, the later `twinkl-747` reachability audit and `twinkl-754` consensus re-judging work show that a few hard dimensions, especially `Security`, still require tighter target design and follow-up analysis.

#### Diagnostic Framework

The evaluation spec ([`judge_validation_eval.md`](judge_validation_eval.md)) defines the following diagnostic:

| Fleiss' κ | Cohen's κ | Diagnosis |
|-----------|-----------|-----------|
| High | High | Judge is well-calibrated |
| High | Low | Judge has systematic bias --- fix Judge prompt |
| Low | Varies | Rubric is ambiguous --- fix definitions first |

**Observed pattern:** Moderate Fleiss' κ with higher Cohen's κ. This indicates the Judge is well-calibrated relative to the level of human consensus achievable for this task. Where individual annotators show low Cohen's κ (e.g., Des on Conformity: 0.30), Fleiss' κ for those dimensions is also low (0.43), suggesting rubric ambiguity rather than Judge error.

### Per-Dimension Breakdown

| Value Dimension | Fleiss' κ (Human-Human) | Avg Cohen's κ (Judge-Human) | Judge > Human? |
|-----------------|-------------------------|------------------------------|----------------|
| Self-Direction | 0.44 | 0.64 | Yes |
| Stimulation | 0.58 | 0.67 | Yes |
| Hedonism | 0.64 | 0.65 | Yes |
| Achievement | 0.47 | 0.62 | Yes |
| Power | 0.61 | 0.60 | **No** (marginal) |
| Security | 0.48 | 0.52 | Yes |
| Conformity | 0.43 | 0.58 | Yes |
| Tradition | 0.50 | 0.69 | Yes |
| Benevolence | 0.61 | 0.68 | Yes |
| Universalism | 0.72 | 0.83 | Yes |

All metrics in this table are computed on the shared 115-entry subset to ensure like-for-like comparison with Fleiss' κ.

**Power** is the sole dimension where average Cohen's κ (0.60) falls slightly below Fleiss' κ (0.61). The gap is marginal (0.01) and both values fall within the Moderate-Substantial range.

### Per-Annotator Cohen's κ vs Judge

All values below are computed on the shared 115-entry subset for consistency with Fleiss' κ.

| Value | Des | JL | KM |
|-------|-----|-----|-----|
| Self-Direction | 0.43 | 0.86 | 0.64 |
| Stimulation | 0.58 | 0.77 | 0.66 |
| Hedonism | 0.64 | 0.76 | 0.55 |
| Achievement | 0.46 | 0.74 | 0.66 |
| Power | 0.49 | 0.65 | 0.66 |
| Security | 0.32 | 0.70 | 0.52 |
| Conformity | 0.30 | 0.69 | 0.76 |
| Tradition | 0.33 | 0.91 | 0.83 |
| Benevolence | 0.59 | 0.78 | 0.67 |
| Universalism | 0.74 | 0.96 | 0.78 |
| **Aggregate** | **0.50** | **0.80** | **0.69** |

JL shows the highest alignment with the Judge (0.80, Substantial), followed by KM (0.69, Substantial) and Des (0.50, Moderate). Variation across annotators is expected and reflects individual differences in rubric interpretation.

---

## Annotation Sample

### Sample Composition (19 Personas, 115 Shared Entries)

Three annotators (Des, JL, KM) independently labeled all entries for 19 personas (annotation orders 1--19). 9 of 10 Schwartz value dimensions meet the target of >= 3 core personas; Stimulation remains at 2.

### Core Persona Value Coverage

| Value Dimension | Personas in Sample | Status |
|-----------------|-------------------|--------|
| Self-Direction | 4 | Adequate |
| Stimulation | 2 | Below target |
| Hedonism | 4 | Adequate |
| Achievement | 3 | Adequate |
| Power | 3 | Adequate |
| Security | 4 | Adequate |
| Conformity | 3 | Adequate |
| Tradition | 3 | Adequate |
| Benevolence | 3 | Adequate |
| Universalism | 4 | Adequate |

**Target of >= 3 personas per dimension has been met for 9 of 10 dimensions.** Stimulation has 2 core personas; the original persona selection targeted 3 (via annotation order 21, which was not annotated). Despite the shortfall, Stimulation shows Moderate Fleiss' κ (0.58) and the highest avg Cohen's κ among the below-target dimensions (0.67), suggesting the available signal is still informative.

### Why Core Persona Values Drive Reliable Signal

Coverage is measured by **Core Persona Values**---the count of personas whose profile includes a dimension as a core value. This is the best predictor of reliable kappa calculation because personas consistently express their core values across multiple journal entries.

Entry-level signal (count of entries with non-zero labels) can be misleading---dimensions like Achievement showed entry-level signal through incidental "crossover" expressions in the initial 10-persona sample, despite having zero core personas. Such crossover signal is less consistent, harder to validate, and less reliable for kappa calculation.

---

## Completed Steps

### 1. Generate Additional Synthetic Data --- NOT NEEDED

The registry already contained sufficient personas for all dimensions (204 personas, 292 value assignments, mean 29.2 per value).

### 2. Run Judge Labeling --- COMPLETED

All 204 personas have been labeled by the Judge. Labels stored in `logs/judge_labels/judge_labels.parquet` (1,651 entries).

### 3. Conduct Additional Human Annotation Round --- COMPLETED

9 additional personas were annotated (annotation orders 11--19), expanding the sample from 10 to 19 personas (46 to 115 shared entries). All three annotators labeled all entries. Persona selection was optimized to maximize dimension coverage with minimum annotations.

### 4. Re-calculate Agreement Metrics --- COMPLETED

Agreement report generated: `logs/exports/agreement_report_20260318_130642.md`

**Success criteria evaluation:**
- All 10 dimensions have >= 3 personas with that core value in the annotated sample --- **MET for 9/10 dimensions** (Stimulation has 2 core personas; annotation order 21 was not completed)
- Judge-Human kappa >= Fleiss' kappa for all dimensions --- **MET for 9/10 dimensions** (Power is the sole exception with a marginal gap of 0.01)

---

## Dimensions That May Benefit from Rubric Clarification

The following dimensions show the lowest Fleiss' kappa (human-human agreement), suggesting annotators find them hardest to judge consistently:

| Dimension | Fleiss' κ | Possible Source of Ambiguity |
|-----------|-----------|------------------------------|
| Conformity | 0.43 | Overlap with Tradition (both involve social norms) |
| Self-Direction | 0.44 | Broad scope---autonomy, creativity, and curiosity all qualify |
| Achievement | 0.47 | Overlap with Power (both involve competence/success) |

Per the evaluation spec, low Fleiss' kappa indicates rubric ambiguity rather than Judge error. Improving rubric definitions for these dimensions would be expected to raise both human-human and Judge-human agreement.

---

## Methodology Notes

### Kappa Interpretation (Landis & Koch, 1977)

| κ Range | Interpretation |
|---------|----------------|
| < 0.00 | Poor |
| 0.00--0.20 | Slight |
| 0.21--0.40 | Fair |
| 0.41--0.60 | Moderate |
| 0.61--0.80 | Substantial |
| 0.81--1.00 | Almost Perfect |

### Metrics Used

- **Cohen's κ**: Pairwise agreement between one human annotator and the Judge (accounts for chance agreement)
- **Fleiss' κ**: Multi-rater agreement among all human annotators (measures human consensus)

### Sample Size

- **Human Annotators:** 3 (Des, JL, KM)
- **Personas Annotated:** 19 (annotation orders 1--19)
- **Shared Entries:** 115
- **Total Personas in Registry:** 204
