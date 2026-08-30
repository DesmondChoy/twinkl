# LLM-Judge Validation Summary

**Last Verified:** 2026-08-31

**Purpose:** This document summarizes inter-rater reliability findings for the
shared project-team annotation subset. The results support bounded development
use of the persisted LLM-Judge VIF Labels. They do not establish independent
human validation, objective labels, or a human-consistency ceiling. The
`twinkl-747` reachability audit and `twinkl-754` consensus rerun add further
caveats for the hardest dimensions and for LLM-Judge VIF Label reachability.

**Analysis Source:** `src/annotation_tool/agreement_metrics.py`
**Full Report:** `logs/exports/agreement_report_20260318_130642.md`
**Reachability Audit:** `logs/exports/twinkl_747/reachability_audit_report.md`
**Consensus Report:** `logs/exports/twinkl_754/consensus_rejudging_report.md`
**Evaluation Spec:** [`docs/evals/judge_validation_eval.md`](judge_validation_eval.md)

---

## Key Findings

### Agreement Metrics (All 10 Dimensions)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Human-Human Agreement** (Fleiss' κ) | 0.56 | Moderate |
| **LLM-Judge-Human Agreement** (Avg Cohen's κ) | 0.66 | Substantial |

**Conclusion:** On the shared 115-Journal-Entry subset, mean LLM-Judge-human
Cohen's κ is numerically larger than human-human Fleiss' κ for 9 of 10 Schwartz
values. The coefficients use different rater structures, so this numerical
pattern does not rank the LLM-Judge against the annotators. It supports bounded
POC supervision when paired with the per-value results, label provenance, and
later reachability evidence.

#### Why This Matters

Fleiss' κ measures agreement among the three project-team annotators. Mean
Cohen's κ summarizes three separate LLM-Judge-to-annotator comparisons. These
statistics answer complementary questions, but their numerical difference is
not a paired performance advantage. Kappa is also sensitive to class prevalence
and sample composition.

The 115 Journal Entries cover 19 synthetic personas, and the annotators are
project team members rather than an independent external panel. The result
therefore supports development-scale use and focused error analysis. It does
not show that the LLM-Judge captures a human majority, can replace manual
labeling without review, or is more reliable than people.

#### Diagnostic Framework

The evaluation spec ([`judge_validation_eval.md`](judge_validation_eval.md)) defines the following diagnostic:

| Fleiss' κ | Cohen's κ | Diagnosis |
|-----------|-----------|-----------|
| High | High | Both agreement views are strong on the sampled data |
| High | Low | Inspect exact disagreements, label provenance, and LLM-Judge prompt behavior |
| Low | Varies | Inspect rubric ambiguity, class prevalence, sample coverage, and annotator differences |

**Observed pattern:** Human-human Fleiss' κ is moderate and mean
LLM-Judge-human Cohen's κ is substantial under the conventional interpretation
bands. Per-value coefficients identify weaker agreement for focused review.
They do not determine whether a difference comes from the rubric, prevalence,
the sampled personas, one annotator, or the LLM-Judge without case-level
analysis.

### Per-Dimension Breakdown

| Value Dimension | Fleiss' κ (Human-Human) | Avg Cohen's κ (LLM-Judge-Human) | Cohen value numerically larger? |
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

All metrics in this table are computed on the shared 115-Journal-Entry subset to ensure like-for-like comparison with Fleiss' κ.

**Power** is the sole dimension where average Cohen's κ (0.60) falls slightly below Fleiss' κ (0.61). The gap is marginal (0.01) and both values fall within the Moderate-Substantial range.

### Per-Annotator Cohen's κ vs LLM-Judge

All values below are computed on the shared 115-Journal-Entry subset for consistency with Fleiss' κ.

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

JL has the highest observed overlap with the LLM-Judge (0.80, Substantial),
followed by KM (0.69, Substantial) and Des (0.50, Moderate). These differences
require case-level analysis; the aggregate coefficients do not identify their
cause.

### Additional Audit Caveats

The human-overlap benchmark and two additional audits describe distinct parts
of the current evidence:

1. **Reachability is the main hard-dimension caveat.** The `twinkl-747` audit sampled 50 cases and found that aggregate LLM-Judge-human agreement did not guarantee that every hard-dimension label was a clean VIF Critic (Offline) target. Its recommendation grid was:
   - `security` → `change_distillation_target`
   - `hedonism` → `targeted_relabeling`
   - `stimulation` → `targeted_relabeling`
   This was a historical diagnostic recommendation. The legacy reduced-context
   prompts did not exactly represent the active session-plus-profile VIF Critic (Offline)
   state, so the audit did not create or validate repaired labels.
2. **Consensus improved stability, not the active frontier target.** The `twinkl-754` five-pass profile-only rerun showed strong repeated-call self-consistency (per-dimension Fleiss' κ `0.775` to `0.890`) and passed the full-corpus stability gate for `security`, `hedonism`, and `stimulation`. But it did **not** become the default supervision source for frontier claims, because the consensus branch changed labels on the frozen holdout and did not improve the advisory human-overlap benchmark enough to replace persisted labels cleanly.

Practical takeaway: the shared-subset agreement supports bounded use of the
persisted LLM-Judge VIF Labels for POC development. It does not show that every
hard-dimension label is equally reachable or equally suitable as a target for
the VIF Critic (Offline).

---

## Annotation Sample

### Sample Composition (19 Personas, 115 Shared Entries)

Three annotators (Des, JL, KM) independently labeled all Journal Entries for 19 personas (annotation orders 1--19). 9 of 10 Schwartz value dimensions meet the target of >= 3 core personas; Stimulation remains at 2.

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

**Target of >= 3 personas per dimension has been met for 9 of 10 dimensions.**
Stimulation has 2 Core Value personas; the original persona selection targeted
3 through annotation order 21, which was not annotated. Its Fleiss' κ is 0.58
and its mean Cohen's κ is 0.67, but the below-target coverage remains a sample
limitation.

### Why Core Persona Values Drive Reliable Signal

Coverage is measured by **Core Persona Values**---the count of personas whose profile includes a dimension as a Core Value. This is the best predictor of reliable kappa calculation because personas consistently express their Core Values across multiple Journal Entries.

Entry-level signal (count of Journal Entries with non-zero labels) can be misleading---dimensions like Achievement showed entry-level signal through incidental "crossover" expressions in the initial 10-persona sample, despite having zero core personas. Such crossover signal is less consistent, harder to validate, and less reliable for kappa calculation.

---

## Completed Steps

### 1. Generate Additional Synthetic Data --- NOT NEEDED

The registry already contained sufficient personas for all dimensions (204 personas, 292 value assignments, mean 29.2 per value).

### 2. Run LLM-Judge Labeling --- COMPLETED

All 204 personas have been labeled by the LLM-Judge. Labels stored in `logs/judge_labels/judge_labels.parquet` (1,651 Journal Entries).

### 3. Conduct Additional Human Annotation Round --- COMPLETED

9 additional personas were annotated (annotation orders 11--19), expanding the sample from 10 to 19 personas (46 to 115 shared Journal Entries). All three annotators labeled all Journal Entries. Persona selection was optimized to maximize dimension coverage with minimum annotations.

### 4. Re-calculate Agreement Metrics --- COMPLETED

Agreement report generated: `logs/exports/agreement_report_20260318_130642.md`

**Recorded coverage and agreement results:**

- The target of at least 3 personas with each Core Value is met for 9 of 10
  values. Stimulation has 2 Core Value personas because annotation order 21 was
  not completed.
- Mean LLM-Judge-human Cohen's κ is numerically larger than human-human Fleiss'
  κ for 9 of 10 values. Power is the exception by 0.01. This pattern is
  descriptive because the coefficients use different rater structures.

---

## Dimensions That May Benefit from Rubric Clarification

The following values have the lowest human-human Fleiss' κ and are priorities
for case-level review:

| Dimension | Fleiss' κ | Possible Source of Ambiguity |
|-----------|-----------|------------------------------|
| Conformity | 0.43 | Overlap with Tradition (both involve social norms) |
| Self-Direction | 0.44 | Broad scope---autonomy, creativity, and curiosity all qualify |
| Achievement | 0.47 | Overlap with Power (both involve competence/success) |

Low Fleiss' κ can reflect rubric ambiguity, class prevalence, sample coverage,
or annotator differences. A rubric change is appropriate only when case-level
review identifies ambiguity; the changed rubric then requires fresh agreement
measurement.

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

- **Cohen's κ**: Pairwise agreement between one human annotator and the LLM-Judge (accounts for chance agreement)
- **Fleiss' κ**: Multi-rater agreement among all human annotators (measures human consensus)

### Sample Size

- **Human Annotators:** 3 (Des, JL, KM)
- **Personas Annotated:** 19 (annotation orders 1--19)
- **Shared Entries:** 115
- **Total Personas in Registry:** 204
