# Vision: Annotation Tool for Judge Validation

## 1. Executive Summary

The LLM-as-Judge produces alignment labels for synthetic journal entries across 10 Schwartz value dimensions. Before using these labels to train the VIF Critic, we need to validate that the Judge agrees with human intuition. This document proposes a **custom Shiny for Python annotation tool** that enables efficient human labeling and automatic calculation of inter-rater agreement metrics.

**Key insight from [Hamel Husain's Evals FAQ](https://hamel.dev/blog/posts/evals-faq/):**
> Custom annotation tools represent "the single most impactful investment you can make for your AI evaluation workflow." Teams with custom tools iterate ~10x faster than those relying on generic solutions.

**Why build custom instead of using off-the-shelf?**
- Consolidates persona context + entry text + scoring grid in one domain-specific view
- Renders entries with nudge/response threading (unique to our pipeline)
- Integrates with existing Polars/Parquet tooling
- Calculates agreement metrics automatically
- Buildable in 3-4 hours with modern AI-assisted development

---

## 2. Problem Statement

### Current State

The Judge validation process relies on manual markdown review:

```markdown
## Entry Review: Persona 3, Entry 5
**Entry**: "Stayed late again to finish the deck..."
**Judge scores**: Achievement: +1, Hedonism: -1, all others: 0
**Reviewer assessment**:
- [x] Achievement +1: Agree
- [ ] Benevolence: Should this be -1?
```

### Limitations

| Problem | Impact |
|---------|--------|
| No structured data capture | Can't compute Cohen's κ systematically |
| Manual copy-paste workflow | Slow (~5 min/entry), error-prone |
| Judge scores visible during review | Anchoring bias skews assessments |
| Single annotator | No inter-annotator agreement measurement |
| Scattered context | Must cross-reference persona files manually |

### Target Scale

- **100-200 entries** for statistically meaningful agreement metrics
- **2+ annotators** for inter-annotator reliability (Fleiss' κ)
- **~2-3 min/entry** target throughput with proper tooling

---

## 3. Proposed Solution

### Overview

A single-page Shiny for Python app that:

1. Loads entries with full persona context using `parse_synthetic_data_run()`
2. Supports multiple annotators with separate annotation files
3. Presents a 10-value scoring grid with clear +1/0/-1 options
4. **Hides Judge scores until after annotation** (critical for avoiding bias)
5. Calculates and displays agreement metrics in real-time
6. Exports results for downstream analysis

### Why Shiny for Python?

Per [Hamel Husain's recommendation](https://hamel.dev/notes/llm/finetuning/data_cleaning.html), we use Shiny for Python instead of Streamlit. See also his [detailed comparison](https://shiny.posit.co/py/docs/comp-streamlit.html).

| Option | Pros | Cons | Recommendation |
|--------|------|------|----------------|
| **Shiny for Python** | Efficient reactivity (only re-renders what changes), scales without rewriting, small API, native Quarto integration, WASM support, no data collection | Newer ecosystem | **Recommended** |
| Streamlit | Fast to prototype, large ecosystem | Full-script reruns on every interaction, state management complexity at scale, limited UI customization, privacy concerns (data sent to Snowflake) | Not recommended |
| Gradio | Very fast prototyping | Clunky for multi-step workflows | For ML demos |
| HTML/JS + JSON | Maximum control, offline | No backend, harder aggregation | For simple cases |

**Why not Streamlit?**
- **Full-script reruns**: Every user interaction (clicking a radio button, navigating) reruns the ENTIRE app script
- **State management pain**: As apps grow, you must manually manipulate hidden session state variables
- **Privacy concerns**: Collects behavioral data sent to Snowflake servers unless explicitly disabled

**Key insight from Hamel**: "Shiny apps always required much less code and were easier to understand" compared to alternatives.

---

## 4. Technical Design

### 4.1 Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ANNOTATION TOOL ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐     ┌──────────────┐     ┌─────────────────────────┐   │
│  │   Data      │────▶│  Shiny       │────▶│  Annotation Store       │   │
│  │   Loader    │     │  App         │     │  (Parquet per annotator)│   │
│  └─────────────┘     └──────────────┘     └─────────────────────────┘   │
│        │                    │                        │                   │
│        │                    ▼                        │                   │
│        │             ┌──────────────┐                │                   │
│        │             │  Session     │                │                   │
│        │             │  Manager     │                │                   │
│        │             └──────────────┘                │                   │
│        │                    │                        │                   │
│        ▼                    ▼                        ▼                   │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Analysis Module                               │    │
│  │  - Cohen's κ (annotator vs Judge)                               │    │
│  │  - Fleiss' κ (inter-annotator)                                  │    │
│  │  - Per-value breakdown + confusion matrices                     │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Data Model

**Input (existing):** Use `parse_synthetic_data_run()` from `src.wrangling` to load entries as DataFrame

```python
{
    "persona_id": int,
    "persona_name": str,
    "persona_age": str,
    "persona_profession": str,
    "persona_culture": str,
    "persona_core_values": list[str],
    "persona_bio": str,
    "t_index": int,
    "date": str,
    "initial_entry": str,
    "nudge_text": str | None,
    "response_text": str | None,
}
```

**Judge labels:** `logs/judge_labels/<timestamp>/judge_labels.parquet`

```python
{
    "persona_id": int,
    "t_index": int,
    "alignment_self_direction": int,  # -1, 0, +1
    "alignment_stimulation": int,
    # ... (all 10 values)
}
```

**Human annotations (new):** `logs/annotations/<annotator_id>.parquet`

```python
{
    "persona_id": int,
    "t_index": int,
    "annotator_id": str,
    "timestamp": datetime,
    "alignment_self_direction": int,
    # ... (all 10 values)
    "notes": str | None,
    "confidence": int | None,  # 1-5 self-reported
}
```

### 4.3 UI Layout

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Twinkl Value Alignment Annotation Tool           Annotator: [Desmond ▼] │
├──────────────────────────────────────────────────────────────────────────┤
│  Progress: ████████░░░░░░░░ 47/100 entries (47%)   [◀ Prev] [Next ▶]    │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ PERSONA CONTEXT                                                      │ │
│  │ Gabriela Mendoza (31, Parent, Latin American)                        │ │
│  │ Core Values: Power                                                   │ │
│  │ Bio: [collapsed by default, click to expand]                         │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ JOURNAL ENTRY (Entry 3 of 7)                           [2025-12-11] │ │
│  ├─────────────────────────────────────────────────────────────────────┤ │
│  │ **Initial Entry:**                                                   │ │
│  │ Spent the afternoon at my sister Lucia's place...                   │ │
│  │                                                                      │ │
│  │ **Nudge:** "What would turning it off even look like?"              │ │
│  │                                                                      │ │
│  │ **Response:** I don't know. Maybe not scanning every room...        │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ ALIGNMENT SCORING                                                    │ │
│  ├────────────────────┬──────────┬──────────┬──────────┬───────────────┤ │
│  │ Value              │ -1 (Mis) │  0 (Neu) │ +1 (Ali) │ Tip           │ │
│  ├────────────────────┼──────────┼──────────┼──────────┼───────────────┤ │
│  │ Self-Direction     │   ○      │    ●     │    ○     │ Autonomy      │ │
│  │ Stimulation        │   ○      │    ●     │    ○     │ Novelty       │ │
│  │ Hedonism           │   ○      │    ○     │    ●     │ Pleasure      │ │
│  │ Achievement        │   ○      │    ●     │    ○     │ Success       │ │
│  │ Power              │   ●      │    ○     │    ○     │ Control       │ │
│  │ Security           │   ○      │    ●     │    ○     │ Stability     │ │
│  │ Conformity         │   ○      │    ●     │    ○     │ Fit in        │ │
│  │ Tradition          │   ○      │    ●     │    ○     │ Heritage      │ │
│  │ Benevolence        │   ○      │    ○     │    ●     │ Close others  │ │
│  │ Universalism       │   ○      │    ●     │    ○     │ All/nature    │ │
│  └────────────────────┴──────────┴──────────┴──────────┴───────────────┘ │
│                                                                          │
│  Confidence: [1] [2] [3] [4] [5]    Notes: [________________]            │
│                                                                          │
│  [Save & Next →]                                                         │
└──────────────────────────────────────────────────────────────────────────┘
```

### 4.4 Critical Design Decision: Hiding Judge Scores

**Requirement:** Judge scores must be hidden during annotation to avoid anchoring bias.

**Implementation:**
- `judge_labels.parquet` loaded but stored separately from display state
- Judge scores only revealed in Analysis view after annotation complete
- Toggle: "Show Judge comparison" reveals side-by-side view

---

## 5. Features

### 5.1 MVP (3-4 hours)

| Feature | Priority | Notes |
|---------|----------|-------|
| Load entries via `parse_synthetic_data_run()` | P0 | Use existing Polars patterns |
| Annotator selection/creation | P0 | Simple dropdown + text input |
| Entry display with persona context | P0 | Collapsible bio section |
| 10-value scoring grid (radio buttons) | P0 | Default to 0 (Neutral) |
| Save annotations to Parquet | P0 | One file per annotator |
| Progress indicator | P0 | "47/100 entries (47%)" |
| Navigation (prev/next) | P0 | Buttons + keyboard arrows |
| Cohen's κ calculation | P0 | Per-value + aggregate |
| Export results | P0 | CSV/Parquet + markdown report |

### 5.2 Nice-to-Have (+2-4 hours)

| Feature | Priority | Notes |
|---------|----------|-------|
| Keyboard shortcuts | P1 | 1-0 for values, +/-/= for scores |
| Fleiss' κ (multi-annotator) | P1 | Requires 2+ annotator files |
| Confusion matrix visualization | P1 | Heatmap per value |
| Per-value breakdown charts | P1 | Bar chart of agreement |
| Randomized entry order | P2 | Prevent order bias |
| Time tracking per entry | P2 | Automatic via timestamps |
| Entry filtering | P2 | By persona, by nudge status |
| Notes field | P2 | Free-text for edge cases |

### 5.3 Keyboard Shortcuts (if implemented)

| Key | Action |
|-----|--------|
| `1-9, 0` | Select value row (1=Self-Direction ... 0=Universalism) |
| `-`, `=` | Set score -1, +1 for selected row |
| `0` (in score mode) | Set score 0 |
| `Enter` | Save & Next |
| `Backspace` | Previous entry |
| `Space` | Toggle expanded persona bio |

---

## 6. Agreement Metrics

### 6.1 Cohen's Kappa (Annotator vs Judge)

Calculate per-value and aggregate:

```python
from sklearn.metrics import cohen_kappa_score

def calculate_cohen_kappa(human_labels, judge_labels) -> dict:
    values = ["self_direction", "stimulation", "hedonism", "achievement",
              "power", "security", "conformity", "tradition",
              "benevolence", "universalism"]

    results = {}
    for value in values:
        human = human_labels[f"alignment_{value}"].to_list()
        judge = judge_labels[f"alignment_{value}"].to_list()
        results[value] = cohen_kappa_score(human, judge, labels=[-1, 0, 1])

    # Aggregate across all values (flatten all scores)
    all_human = [s for v in values for s in human_labels[f"alignment_{v}"].to_list()]
    all_judge = [s for v in values for s in judge_labels[f"alignment_{v}"].to_list()]
    results["aggregate"] = cohen_kappa_score(all_human, all_judge, labels=[-1, 0, 1])

    return results
```

### 6.2 Fleiss' Kappa (Inter-Annotator)

For 2+ annotators rating the same entries:

```python
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters

def calculate_fleiss_kappa(annotations: list) -> dict:
    # Build matrix: rows = entries, cols = annotators
    # Convert to category counts, compute Fleiss' κ per value
    ...
```

### 6.3 Interpretation Guide

| κ Range | Interpretation |
|---------|----------------|
| < 0.00 | Poor (less than chance) |
| 0.00-0.20 | Slight |
| 0.21-0.40 | Fair |
| 0.41-0.60 | Moderate |
| **0.61-0.80** | **Substantial (target)** |
| 0.81-1.00 | Almost perfect |

**Target:** κ ≥ 0.60 (substantial agreement) between human annotators and Judge.

---

## 7. Implementation Approach

### 7.1 File Structure

```
src/
└── annotation_tool/
    ├── __init__.py
    ├── app.py                 # Main Shiny app
    ├── data_loader.py         # Load Parquet files
    ├── agreement_metrics.py   # Kappa calculations
    └── components/
        ├── entry_display.py   # Entry rendering
        ├── scoring_grid.py    # 10-value input
        └── analysis_view.py   # Results view
```

### 7.2 Implementation Order

**Phase 1: Core Loop (2 hours)**
1. Create `app.py` with basic Shiny structure
2. Implement `data_loader.py` to read Parquet files
3. Build entry display component
4. Build scoring grid with radio buttons
5. Implement save logic (append to annotator Parquet)
6. Add navigation and progress bar

**Phase 2: Analysis (1-2 hours)**
7. Implement `agreement_metrics.py`
8. Build analysis view with kappa display
9. Add Judge score reveal (post-completion only)
10. Add export functionality

**Phase 3: Polish (1 hour)**
11. Add keyboard shortcuts (optional)
12. Randomize entry order
13. Add value tooltips from schwartz_values.yaml
14. Error handling

### 7.3 Key Dependencies

```python
# pyproject.toml additions
dependencies = [
    "shiny>=1.0.0",
    "polars>=1.0.0",
    "scikit-learn>=1.3.0",  # For cohen_kappa_score
    "statsmodels>=0.14.0",  # For fleiss_kappa
]
```

---

## 8. Example Output

### Agreement Report

After annotation completion, generate `logs/annotations/agreement_report.md`:

```markdown
# Agreement Report

**Generated:** 2026-01-13 14:30:00
**Dataset:** logs/synthetic_data/2026-01-09_09-37-09
**Entries:** 100

## Annotators
- Desmond (100 entries)
- Rater2 (100 entries)

## Cohen's Kappa: Annotator vs Judge

| Value | Desmond vs Judge | Rater2 vs Judge |
|-------|------------------|-----------------|
| Self-Direction | 0.72 | 0.68 |
| Stimulation | 0.65 | 0.71 |
| Hedonism | 0.58 | 0.62 |
| Achievement | 0.81 | 0.77 |
| Power | 0.69 | 0.73 |
| Security | 0.55 | 0.51 |
| Conformity | 0.63 | 0.66 |
| Tradition | 0.70 | 0.68 |
| Benevolence | 0.74 | 0.79 |
| Universalism | 0.67 | 0.64 |
| **Aggregate** | **0.67** | **0.68** |

## Interpretation

- Judge achieves **substantial agreement** (κ > 0.60) with human annotators
- Highest agreement: Achievement (0.79 avg)
- Lowest agreement: Security (0.53 avg) — consider refining rubric
```

---

## 9. Schwartz Value Quick Reference

Include as tooltips in the UI (from `config/schwartz_values.yaml`):

| Value | One-liner |
|-------|-----------|
| Self-Direction | Making own choices, resisting control, autonomy |
| Stimulation | Seeking novelty, avoiding routine, excitement |
| Hedonism | Prioritizing pleasure, enjoyment, comfort |
| Achievement | Goals, performance, recognition, hard work |
| Power | Control, status, influence, being in charge |
| Security | Stability, safety, avoiding risk |
| Conformity | Following rules, meeting expectations, fitting in |
| Tradition | Honoring customs, family obligations, heritage |
| Benevolence | Helping close others (family, friends, team) |
| Universalism | Broader social concern, fairness, environment |

---

## 10. Success Criteria

| Metric | Target | Rationale |
|--------|--------|-----------|
| Cohen's κ (human vs Judge) | > 0.60 | Substantial agreement validates Judge |
| Inter-annotator κ | > 0.60 | Ensures task is well-defined |
| Annotation throughput | ~2-3 min/entry | 100 entries = 3-5 hours per annotator |
| All-zero disagreement | < 10% | Judge and humans flag signal in most entries |

---

## 11. Open Questions

1. **Sampling strategy:** Should we annotate all entries or stratified sample (by persona, by nudge status, by entry position)?

2. **Annotator training:** How much context do annotators need about Schwartz values before labeling? Should we include a training set of 5-10 entries with "correct" answers?

3. **Disagreement resolution:** When annotators disagree, do we adjudicate or treat majority vote as ground truth?

4. **Iteration loop:** If κ < 0.60, what's the protocol? Refine Judge prompt? Refine rubrics? Clarify annotator guidelines?

---

## 12. References

- **Judge implementation:** `docs/VIF/judge_implementation_spec.md`
- **Value rubrics:** `config/schwartz_values.yaml`
- **Existing validation protocol:** `docs/evals/judge_validation_eval.md`
- **Hamel Husain's Evals FAQ:** https://hamel.dev/blog/posts/evals-faq/
