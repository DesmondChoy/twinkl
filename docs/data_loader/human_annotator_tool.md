# Annotation Tool Implementation Plan

## Overview

Build a Shiny for Python annotation tool for **validating** LLM-as-Judge labels via blind human annotation across 10 Schwartz value dimensions. Humans provide independent scores WITHOUT seeing Judge labels first; the system then computes agreement metrics (Cohen's κ, Fleiss' κ) automatically.

**Key Clarification**: This is NOT a tool for humans to manually assign scores from scratch. The Judge has already labeled entries. This tool validates whether humans agree with the Judge's assessments.

## Data Schema Alignment

The `annotation_tool.md` proposal aligns well with `data_schema.md`:

| Aspect | annotation_tool.md | data_schema.md | Status |
|--------|-------------------|----------------|--------|
| Entry loader | Wrangled files (`logs/wrangled/`) | N/A (synthetic data) | ✓ Custom parser in `src/annotation_tool/data_loader.py` |
| Judge labels | `logs/judge_labels/judge_labels.parquet` | Same path | ✓ Matches |
| Composite key | `(persona_id, t_index)` | Same | ✓ Matches |
| Value columns | `alignment_self_direction`, etc. | Same names | ✓ Matches |
| Value order | 10 Schwartz values | `SCHWARTZ_VALUE_ORDER` constant | ✓ Defined in `src/models/judge.py` |

## File Structure

```
src/annotation_tool/
├── __init__.py
├── app.py                 # Main Shiny app entry point
├── data_loader.py         # Load entries + judge labels
├── annotation_store.py    # Save/load annotator parquet files
├── agreement_metrics.py   # Cohen's κ, Fleiss' κ
└── components/
    ├── __init__.py
    ├── header.py          # Annotator selector + progress
    ├── persona_context.py # Collapsible persona display
    ├── entry_display.py   # Journal entry rendering
    ├── scoring_grid.py    # 10-value radio buttons
    └── analysis_view.py   # Metrics display + export

logs/annotations/          # NEW directory
└── <annotator_id>.parquet # One file per annotator
```

## Dependencies to Add

```toml
# pyproject.toml additions
"shiny>=1.2.0",        # Shiny for Python framework
"scikit-learn>=1.3.0", # cohen_kappa_score
"statsmodels>=0.14.0", # fleiss_kappa
```

## Human Annotation Schema

New parquet files at `logs/annotations/<annotator_id>.parquet`:

| Column | Type | Description |
|--------|------|-------------|
| `persona_id` | `Utf8` | Links to registry |
| `t_index` | `Int64` | Entry index |
| `annotator_id` | `Utf8` | Who annotated |
| `timestamp` | `Datetime` | When annotated |
| `alignment_self_direction` | `Int8` | -1, 0, or +1 |
| ... (8 more values) | `Int8` | |
| `alignment_universalism` | `Int8` | -1, 0, or +1 |
| `notes` | `Utf8` | Optional free-text |
| `confidence` | `Int8` | 1-5, optional |

## Key Reusable Components

| Existing Code | Location | Reuse For | Used? |
|--------------|----------|-----------|-------|
| `SCHWARTZ_VALUE_ORDER` | `src/models/judge.py:30-41` | Canonical value ordering | ✅ Yes |
| `AlignmentScores` | `src/models/judge.py:44-77` | Score validation model | ❌ Not yet |
| `_write_registry_locked()` | `src/registry/personas.py:67-86` | File-locking pattern | ✅ Yes (pattern adapted) |
| `parse_synthetic_data_dir()` | `src/wrangling/parse_synthetic_data.py` | Load entries with persona context | ❌ No (custom wrangled parser) |
| `schwartz_values.yaml` | `config/schwartz_values.yaml` | Tooltip definitions | ❌ Not yet (Phase 3) |

## Implementation Phases

### Phase 1: Core Loop ✅ COMPLETE
- [x] Add dependencies to `pyproject.toml`
- [x] Create `data_loader.py` - load entries from wrangled files
- [x] Create `annotation_store.py` - save annotations with file locking
- [x] Create basic `app.py` with Shiny structure
- [x] Implement header component - annotator input + progress bar
- [x] Implement entry display - persona context + entry rendering
- [x] Implement scoring grid - 10-value radio buttons
- [x] Wire up navigation (prev/next) and save logic

**Phase 1 Testing:** ✅ ALL PASSED
- [x] Run app: `shiny run src/annotation_tool/app.py`
- [x] Verify entries load with persona context displayed
- [x] Create new annotator and annotate 3 entries with various scores
- [x] Verify `logs/annotations/<annotator>.parquet` created with correct schema
- [x] Test prev/next navigation and progress bar updates
- [x] Close and reopen app — verify progress persists

**Phase 1 Implementation Notes:**
- Components were consolidated into `app.py` rather than separate module files (simpler Shiny pattern)
- `data_loader.py` uses custom wrangled-format parser (not `parse_synthetic_data_dir()` which expects synthetic format)
- Entry ordering is sequential by persona (grouped), not shuffled (shuffling deferred to Phase 2)
- Annotator input is free-form text, not dropdown selector
- All-neutral warning modal implemented
- Collapsible persona bio implemented (originally planned for Phase 3)

### Phase 2: Analysis
- [ ] Implement `agreement_metrics.py` - kappa calculations
- [ ] Implement `components/analysis_view.py` - metrics display
- [ ] Add blind mode toggle (judge scores hidden until annotation complete)
- [ ] Add export functionality (CSV, Parquet, Markdown report)

**Phase 2 Testing:**
- [ ] Verify Judge scores hidden during annotation (no anchoring bias)
- [ ] After saving, verify Judge scores revealed for comparison
- [ ] Annotate 20+ entries, verify Cohen's κ calculation against manual spot-check
- [ ] Export markdown report — verify format matches `vision.md` example
- [ ] Test with 2 annotators — verify Fleiss' κ computes on shared entries

### Phase 3: Polish
- [ ] Add tooltips from `schwartz_values.yaml`
- [x] Implement collapsible persona bio *(completed in Phase 1)*
- [ ] Add confirmation dialog for unsaved navigation
- [ ] Error handling and loading states

**Phase 3 Testing:**
- [ ] Hover over value names — verify tooltips appear with definitions
- [x] Click persona bio toggle — verify expand/collapse works *(completed in Phase 1)*
- [ ] Make changes, click prev/next without saving — verify confirmation dialog appears
- [ ] Test with missing/malformed data — verify graceful error handling

## Annotation Methodology

### Workflow (Hamel's "Minimal Interface" Principle)

```
1. Annotator sees: Persona context + Journal entry (NO Judge scores)
2. Annotator provides: Their own -1/0/+1 scores for all 10 values
3. On save: System stores human scores, reveals Judge scores for comparison
4. Disagreements: Computed automatically by comparing human vs Judge
```

No explicit "agree/disagree" buttons — just independent scoring followed by automatic comparison.

### Annotator Strategy

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **Annotators** | 3 people | Small team, manageable coordination |
| **Assignment** | Per-entry (not per-persona) | Matches Judge's atomic input; reduces persona "theory" bias |
| **Entry order** | Shuffled across personas | Each annotator sees random mix of entries from all personas |
| **Overlap** | Shared subset (~30-50 entries) | All 3 rate shared set for Fleiss' κ; remaining entries split for coverage |

### Metrics Enabled

| Metric | Requires | What It Measures |
|--------|----------|------------------|
| **Cohen's κ** (per annotator) | Any annotation | How well each human agrees with Judge |
| **Fleiss' κ** | Shared subset | Whether humans agree with each other (rubric clarity) |
| **Per-value breakdown** | Any annotation | Which Schwartz values have lowest agreement |

## UI Specification

Based on user preferences, here's the detailed UI design:

### Layout Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Twinkl Value Alignment Annotation Tool              Annotator: [Desmond ▼] │
├─────────────────────────────────────────────────────────────────────────────┤
│  Progress: ████████░░░░░░░░ 47/100 entries (47%)      [◀ Prev] [Next ▶]    │
├───────────────────────────────────┬─────────────────────────────────────────┤
│                                   │                                         │
│  PERSONA CONTEXT (Collapsible)    │  SCORING GRID (Vertical Table)         │
│  ┌─────────────────────────────┐  │  ┌─────────────────────────────────┐   │
│  │ Gabriela Mendoza            │  │  │ Value          │ ➖  │ ⭕  │ ➕  │   │
│  │ 31 • Parent • Latin American│  │  ├────────────────┼─────┼─────┼─────┤   │
│  │ Core Values: Power          │  │  │ Self-Direction │ ○   │ ●   │ ○   │   │
│  │ [▼ Show Bio]                │  │  │ Stimulation    │ ○   │ ●   │ ○   │   │
│  └─────────────────────────────┘  │  │ Hedonism       │ ○   │ ○   │ ●   │   │
│                                   │  │ Achievement    │ ○   │ ●   │ ○   │   │
│  JOURNAL ENTRY (Stacked)          │  │ Power          │ ●   │ ○   │ ○   │   │
│  ┌─────────────────────────────┐  │  │ Security       │ ○   │ ●   │ ○   │   │
│  │ Entry 3 of 7       2025-12-11│  │  │ Conformity     │ ○   │ ●   │ ○   │   │
│  │ ─────────────────────────────│  │  │ Tradition      │ ○   │ ●   │ ○   │   │
│  │ **Initial Entry:**           │  │  │ Benevolence    │ ○   │ ○   │ ●   │   │
│  │ Spent the afternoon at my    │  │  │ Universalism   │ ○   │ ●   │ ○   │   │
│  │ sister Lucia's place...      │  │  └─────────────────────────────────┘   │
│  │                              │  │                                         │
│  │   > **Nudge:** "What would   │  │  Confidence: [1] [2] [3] [4] [5]       │
│  │   > turning it off look      │  │  Notes: [________________________]     │
│  │   > like?"                   │  │                                         │
│  │                              │  │  [Save & Next →]                        │
│  │   > **Response:** I don't    │  │                                         │
│  │   > know. Maybe not scanning │  │                                         │
│  │   > every room...            │  │                                         │
│  └─────────────────────────────┘  │                                         │
│                                   │                                         │
├───────────────────────────────────┴─────────────────────────────────────────┤
│  ▶ Analysis & Metrics (collapsed by default)                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### UI Decisions Summary

| Aspect | Choice | Details |
|--------|--------|---------|
| **Scoring grid** | Vertical table | All 10 values stacked, -1/0/+1 per row |
| **Score labels** | Icons + text | ➖ Misaligned \| ⭕ Neutral \| ➕ Aligned |
| **Persona context** | Collapsible sidebar | Left side, bio collapsed by default |
| **Entry threading** | Stacked sections | Initial → Nudge (indented) → Response (indented) |
| **Navigation** | Prev/Next + keyboard | Arrow keys + Enter to save |
| **Unsaved changes** | Confirmation dialog | "Save, Discard, or Cancel?" |
| **Entry order** | Randomized | Shuffle per annotator to reduce order bias |
| **Analysis view** | Collapsible accordion | Hidden during annotation, expand to check metrics |
| **All-neutral warning** | Optional confirmation | Warn but allow saving if all values are 0 |
| **Annotator count** | 3 users | Support Cohen's κ and Fleiss' κ |
| **Post-save reveal** | Show Judge comparison | After saving, display human vs Judge scores side-by-side for that entry |

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `↑` / `↓` | Navigate between value rows |
| `←` / `→` | Change score (-1 / +1) for selected row |
| `Enter` | Save & Next |
| `Backspace` | Previous entry |

## Critical Design Decisions

### Blind Mode → Reveal Flow
1. **During annotation**: Judge scores hidden (prevent anchoring bias)
2. **After saving each entry**: Reveal Judge's scores for that entry alongside human's scores
3. **Analysis view**: Aggregate comparison across all annotated entries

This gives immediate feedback per-entry while preventing bias during scoring.

### File Locking
Follow existing pattern from `src/registry/personas.py`:
```python
lock_path = ANNOTATIONS_PATH.with_suffix(".lock")
with open(lock_path) as lock_file:
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    # write parquet
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
```

### One Parquet Per Annotator
- Enables parallel annotation without merge conflicts
- Simple progress tracking per annotator
- Easy to compare annotators in analysis

## Verification Plan

1. **Run the app**: `shiny run src/annotation_tool/app.py`
2. **Test annotation flow**:
   - Create new annotator
   - Annotate 5 entries with various scores
   - Verify `logs/annotations/<annotator>.parquet` created correctly
3. **Test progress tracking**:
   - Close and reopen app
   - Verify progress persists and can resume
4. **Test blind mode**:
   - Verify judge scores not visible during annotation
   - Complete annotations, verify toggle enables
5. **Test metrics**:
   - Run with 20+ annotations
   - Verify Cohen's κ calculation against manual check
6. **Test export**:
   - Export markdown report
   - Verify format matches `annotation_tool.md` example

## Questions Addressed

| annotation_tool.md Question | Answer |
|----------------------------|--------|
| Sampling strategy | Start with all entries; can add filtering later |
| Annotator training | Include tooltips from `schwartz_values.yaml`; consider adding 5 "training" entries with feedback |
| Disagreement resolution | Treat each annotator independently; analysis shows agreement metrics |
| Iteration loop | If κ < 0.60, review confusion matrices to identify problem values |

## Files to Modify

| File | Action |
|------|--------|
| `pyproject.toml` | Add shiny, scikit-learn, statsmodels |
| `src/annotation_tool/*` | NEW - all annotation tool code |
| `logs/annotations/` | NEW directory for annotation parquet files |
