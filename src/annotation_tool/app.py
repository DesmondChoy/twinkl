"""Main Shiny app for the Schwartz Value Annotation Tool.

This app allows human annotators to label journal entries across 10 Schwartz
value dimensions for validating LLM Judge labels.

Usage:
    shiny run src/annotation_tool/app.py

Features:
    - Sequential entry navigation (grouped by persona)
    - Free-form annotator name input
    - 10-value scoring grid with radio buttons (-1, 0, +1)
    - Collapsible persona bio
    - Nudge/Response threading display
    - Progress tracking per annotator
    - All-neutral warning modal
    - Persistent annotations in parquet format
"""

import sys
from pathlib import Path

# Add project root to path for imports when running via shiny run
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from shiny import App, reactive, render, ui

from src.annotation_tool.annotation_store import (
    get_annotation,
    get_annotation_count,
    save_annotation,
)
from src.annotation_tool.data_loader import get_ordered_entries, load_entries
from src.models.judge import SCHWARTZ_VALUE_ORDER

# Load entries at module level (cached)
_entries_df = load_entries()
_all_entries = get_ordered_entries(_entries_df)
_total_entries = len(_all_entries)


def _load_judge_labels() -> dict[tuple[str, int], dict]:
    """Load judge labels into a lookup dictionary keyed by (persona_id, t_index)."""
    import polars as pl

    labels_path = Path("logs/judge_labels/judge_labels.parquet")
    if not labels_path.exists():
        return {}
    df = pl.read_parquet(labels_path)
    return {(row["persona_id"], row["t_index"]): row for row in df.to_dicts()}


# Cache judge labels at module level for O(1) lookups
_judge_labels_map = _load_judge_labels()


def get_judge_scores(persona_id: str, t_index: int) -> dict | None:
    """Get judge scores for a specific entry, or None if not available."""
    return _judge_labels_map.get((persona_id, t_index))


def _format_score(score: int) -> tuple[str, str]:
    """Format a score value for display. Returns (display_text, css_class)."""
    if score == 1:
        return "+1", "positive"
    elif score == -1:
        return "−1", "negative"
    else:
        return "0", "neutral"


def _get_match_type(human_score: int, judge_score: int) -> tuple[str, str, str]:
    """
    Determine the match type between human and judge scores.
    Returns (row_class, symbol, symbol_class).
    """
    diff = abs(human_score - judge_score)
    if diff == 0:
        return "match-exact", "✓", "exact"
    elif diff == 1:
        return "match-adjacent", "~", "adjacent"
    else:
        return "match-disagree", "✗", "disagree"


def build_comparison_modal_content(
    human_scores: dict[str, int], judge_data: dict | None
) -> ui.TagChild:
    """
    Build the comparison modal content.

    Args:
        human_scores: Dict mapping value names to human annotator scores
        judge_data: Full judge label row dict, or None if not available

    Returns:
        UI element for the modal body
    """
    # Handle missing judge labels
    if judge_data is None:
        return ui.div(
            ui.div("✓", class_="comparison-no-labels-icon", style="color: #10b981;"),
            ui.div(
                ui.strong("Annotation saved!"),
                ui.br(),
                "No Judge labels available for this entry.",
                class_="comparison-no-labels-text",
            ),
            ui.input_action_button(
                "comparison_continue",
                "Continue →",
                class_="comparison-continue-btn",
            ),
            class_="comparison-no-labels",
        )

    # Build comparison table rows
    rows = []
    exact_matches = 0

    for value in SCHWARTZ_VALUE_ORDER:
        human_score = human_scores.get(value, 0)
        # Judge scores are stored as alignment_{value} in the parquet
        judge_score = judge_data.get(f"alignment_{value}", 0)

        # Format scores for display
        human_display, human_class = _format_score(human_score)
        judge_display, judge_class = _format_score(judge_score)

        # Determine match type
        row_class, match_symbol, match_class = _get_match_type(human_score, judge_score)

        if row_class == "match-exact":
            exact_matches += 1

        rows.append(
            ui.tags.tr(
                ui.tags.td(VALUE_LABELS[value], class_="comparison-value-name"),
                ui.tags.td(
                    ui.span(human_display, class_=f"comparison-score {human_class}")
                ),
                ui.tags.td(
                    ui.span(judge_display, class_=f"comparison-score {judge_class}")
                ),
                ui.tags.td(
                    ui.span(match_symbol, class_=f"comparison-match {match_class}")
                ),
                class_=f"comparison-row {row_class}",
            )
        )

    return ui.div(
        ui.tags.table(
            ui.tags.thead(
                ui.tags.tr(
                    ui.tags.th("Value"),
                    ui.tags.th("You"),
                    ui.tags.th("Judge"),
                    ui.tags.th("Match"),
                )
            ),
            ui.tags.tbody(*rows),
            class_="comparison-table",
        ),
        ui.div(
            ui.span(f"{exact_matches}/10", class_="comparison-summary-count"),
            " exact matches",
            class_="comparison-summary",
        ),
        ui.input_action_button(
            "comparison_continue",
            "Continue →",
            class_="comparison-continue-btn",
        ),
        class_="comparison-modal-content",
    )


def _group_entries_by_persona(entries: list[dict]) -> list[list[dict]]:
    """Group entries by persona_id, preserving order within each persona."""
    from itertools import groupby

    grouped = []
    for _, group in groupby(entries, key=lambda e: e["persona_id"]):
        grouped.append(list(group))
    return grouped


# Group entries by persona for per-persona navigation
_personas = _group_entries_by_persona(_all_entries)
_total_personas = len(_personas)

# Human-readable labels for Schwartz values
VALUE_LABELS = {
    "self_direction": "Self-Direction",
    "stimulation": "Stimulation",
    "hedonism": "Hedonism",
    "achievement": "Achievement",
    "power": "Power",
    "security": "Security",
    "conformity": "Conformity",
    "tradition": "Tradition",
    "benevolence": "Benevolence",
    "universalism": "Universalism",
}


# Schwartz value groupings for organized display
VALUE_GROUPS = {
    "OPENNESS TO CHANGE": ["self_direction", "stimulation", "hedonism"],
    "SELF-ENHANCEMENT": ["achievement", "power"],
    "CONSERVATION": ["security", "conformity", "tradition"],
    "SELF-TRANSCENDENCE": ["benevolence", "universalism"],
}

GROUP_COLORS = {
    "OPENNESS TO CHANGE": "#6366f1",  # Indigo
    "SELF-ENHANCEMENT": "#f59e0b",    # Amber
    "CONSERVATION": "#10b981",         # Emerald
    "SELF-TRANSCENDENCE": "#ec4899",  # Pink
}

# CSS styles - Clean & Professional (Notion/Linear inspired)
app_css = """
/* ============================================
   BASE STYLES
   ============================================ */
* {
    box-sizing: border-box;
}

body {
    background: #fafafa;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
    margin: 0;
    padding: 0;
    color: #1a1a1a;
    line-height: 1.5;
}

/* ============================================
   HEADER STYLES
   ============================================ */
.header-component {
    background: #ffffff;
    border-bottom: 1px solid #e5e5e5;
    padding: 16px 24px;
    position: sticky;
    top: 0;
    z-index: 100;
}

.header-top-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
    gap: 16px;
}

.header-left {
    display: flex;
    align-items: center;
    gap: 16px;
}

.app-title {
    margin: 0;
    font-size: 18px;
    font-weight: 600;
    color: #1a1a1a;
}

.help-btn {
    background: none;
    border: 1px solid #e5e5e5;
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 14px;
    color: #666;
    cursor: pointer;
    transition: all 0.15s ease;
}

.help-btn:hover {
    background: #f5f5f5;
    border-color: #ccc;
}

.annotator-input input {
    background: #ffffff;
    border: 1px solid #e5e5e5;
    color: #1a1a1a;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 14px;
    width: 180px;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}

.annotator-input input:focus {
    outline: none;
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.annotator-input input::placeholder {
    color: #9ca3af;
}

.header-nav-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.progress-section {
    flex: 1;
    margin-right: 24px;
}

.progress-wrapper {
    display: flex;
    align-items: center;
    gap: 12px;
}

.progress-bar-container {
    flex: 1;
    height: 6px;
    background: #e5e5e5;
    border-radius: 3px;
    overflow: hidden;
    max-width: 300px;
}

.progress-fill {
    height: 100%;
    background: #3b82f6;
    border-radius: 3px;
    transition: width 0.3s ease;
}

.progress-text {
    font-size: 13px;
    color: #6b7280;
    white-space: nowrap;
}

.entry-indicator {
    font-size: 13px;
    color: #6b7280;
    margin-bottom: 6px;
}

.nav-buttons {
    display: flex;
    gap: 8px;
}

.nav-buttons .btn-secondary {
    background: #ffffff;
    border: 1px solid #e5e5e5;
    color: #374151;
    padding: 8px 16px;
    border-radius: 6px;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s ease;
}

.nav-buttons .btn-secondary:hover {
    background: #f9fafb;
    border-color: #d1d5db;
}

/* ============================================
   THREE-COLUMN LAYOUT (Desktop)
   ============================================ */
.main-container {
    max-width: 1600px;
    margin: 0 auto;
    padding: 24px;
}

/* Desktop: Full 3-column layout */
@media (min-width: 1200px) {
    .three-column-layout {
        display: grid;
        grid-template-columns: 340px 1fr 340px;
        gap: 20px;
        align-items: start;
    }

    .left-sidebar {
        position: sticky;
        top: 100px;
        max-height: calc(100vh - 140px);
        overflow-y: auto;
    }

    .right-column {
        position: sticky;
        top: 100px;
    }
}

/* Tablet: 2-column (sidebar collapses) */
@media (min-width: 768px) and (max-width: 1199px) {
    .three-column-layout {
        display: grid;
        grid-template-columns: 1fr 340px;
        gap: 24px;
        align-items: start;
    }

    .left-sidebar {
        grid-column: 1 / -1;
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        margin-bottom: 8px;
    }

    .left-sidebar .sidebar-persona {
        flex: 1;
        min-width: 200px;
    }

    .left-sidebar .sidebar-entry-nav {
        flex: 2;
        min-width: 300px;
    }

    .center-column {
        grid-column: 1;
    }

    .right-column {
        position: sticky;
        top: 100px;
    }
}

/* Mobile: Single column stack */
@media (max-width: 767px) {
    .three-column-layout {
        display: block;
    }

    .left-sidebar {
        margin-bottom: 16px;
    }

    .center-column {
        margin-bottom: 24px;
    }
}

/* ============================================
   ENTRY DISPLAY STYLES
   ============================================ */
.entry-display {
    background: #ffffff;
    border: 1px solid #e5e5e5;
    border-radius: 8px;
    overflow: hidden;
}

.persona-header-section {
    background: #fafafa;
    padding: 20px;
    border-bottom: 1px solid #e5e5e5;
}

.persona-name {
    margin: 0 0 8px 0;
    color: #1a1a1a;
    font-size: 15px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.persona-info {
    color: #6b7280;
    font-size: 14px;
    margin-bottom: 4px;
}

.persona-values {
    color: #374151;
    font-size: 14px;
    font-weight: 500;
}

.persona-bio-text {
    margin-top: 12px;
    padding: 16px;
    background: #ffffff;
    border: 1px solid #e5e5e5;
    border-radius: 6px;
    font-size: 14px;
    color: #4b5563;
    line-height: 1.7;
}

.btn-link {
    background: none;
    border: none;
    color: #3b82f6;
    font-size: 13px;
    padding: 0;
    cursor: pointer;
    text-decoration: none;
}

.btn-link:hover {
    text-decoration: underline;
}

.entry-content-section {
    padding: 20px;
}

.entry-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
}

.entry-number {
    font-weight: 600;
    color: #1a1a1a;
    font-size: 14px;
}

.entry-date {
    color: #9ca3af;
    font-size: 13px;
}

.entry-text {
    font-size: 16px;
    line-height: 1.8;
    color: #1a1a1a;
    white-space: pre-wrap;
}

.entry-nudge {
    margin-top: 20px;
    padding: 16px;
    background: #fffbeb;
    border-left: 3px solid #f59e0b;
    border-radius: 0 6px 6px 0;
    font-size: 15px;
    line-height: 1.7;
    color: #92400e;
}

.entry-response {
    margin-top: 12px;
    padding: 16px;
    background: #ecfdf5;
    border-left: 3px solid #10b981;
    border-radius: 0 6px 6px 0;
    font-size: 15px;
    line-height: 1.7;
    color: #065f46;
}

/* ============================================
   SCORING GRID STYLES (Fixed Radio Buttons)
   ============================================ */
.scoring-grid {
    background: #ffffff;
    border: 1px solid #e5e5e5;
    border-radius: 8px;
    padding: 20px;
}

.scoring-header {
    display: grid;
    grid-template-columns: 1fr 140px;
    align-items: center;
    padding-bottom: 12px;
    border-bottom: 1px solid #e5e5e5;
    margin-bottom: 4px;
}

.scoring-header-label {
    font-weight: 600;
    color: #374151;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.scoring-header-scores {
    display: flex;
    justify-content: center;
    gap: 16px;
    font-weight: 600;
    color: #374151;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Value group headers */
.value-group {
    margin-top: 16px;
}

.value-group:first-of-type {
    margin-top: 8px;
}

.value-group-header {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #9ca3af;
    padding: 8px 0 4px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

.value-group-header::before {
    content: '';
    display: inline-block;
    width: 3px;
    height: 12px;
    border-radius: 2px;
}

.value-group-header.openness::before { background: #6366f1; }
.value-group-header.enhancement::before { background: #f59e0b; }
.value-group-header.conservation::before { background: #10b981; }
.value-group-header.transcendence::before { background: #ec4899; }

/* Scoring rows */
.scoring-row {
    display: grid;
    grid-template-columns: 1fr 140px;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid #f3f4f6;
    transition: background-color 0.1s ease;
}

.scoring-row:last-child {
    border-bottom: none;
}

.scoring-row:hover {
    background: #fafafa;
}

.scoring-row.focused {
    background: #eff6ff;
    margin: 0 -20px;
    padding: 10px 20px;
}

.value-label {
    font-size: 15px;
    color: #1a1a1a;
    font-weight: 500;
}

/* Counter button scoring UI */
.score-btn-group {
    display: flex;
    gap: 6px;
    justify-content: center;
    align-items: center;
}

.score-btn {
    width: 32px;
    height: 32px;
    border: 1px solid #e5e5e5;
    border-radius: 6px;
    background: #ffffff;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    font-weight: 600;
    color: #6b7280;
    transition: all 0.15s ease;
    padding: 0;
}

.score-btn:hover {
    border-color: #9ca3af;
    background: #f9fafb;
}

.score-btn:active {
    transform: scale(0.95);
}

.score-btn.dec:hover {
    border-color: #ef4444;
    color: #ef4444;
    background: #fef2f2;
}

.score-btn.inc:hover {
    border-color: #10b981;
    color: #10b981;
    background: #ecfdf5;
}

/* Score display in the middle */
.score-display {
    min-width: 36px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    font-weight: 600;
    border-radius: 6px;
    transition: all 0.15s ease;
}

.score-display.negative {
    background: #fef2f2;
    color: #ef4444;
    border: 1px solid #fecaca;
}

.score-display.neutral {
    background: #f9fafb;
    color: #6b7280;
    border: 1px solid #e5e5e5;
}

.score-display.positive {
    background: #ecfdf5;
    color: #10b981;
    border: 1px solid #d1fae5;
}

/* ============================================
   LEFT SIDEBAR STYLES
   ============================================ */
.left-sidebar {
    display: flex;
    flex-direction: column;
    gap: 16px;
}

.sidebar-persona {
    background: #ffffff;
    border: 1px solid #e5e5e5;
    border-radius: 8px;
    padding: 16px;
}

.sidebar-persona-name {
    margin: 0 0 8px 0;
    color: #1a1a1a;
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.sidebar-persona-info {
    color: #6b7280;
    font-size: 12px;
    margin-bottom: 4px;
    line-height: 1.4;
}

.sidebar-persona-values {
    color: #374151;
    font-size: 12px;
    font-weight: 500;
    margin-bottom: 8px;
}

/* Collapsible bio in sidebar */
.sidebar-bio-container {
    margin-top: 8px;
}

.sidebar-bio-toggle {
    background: none;
    border: none;
    color: #3b82f6;
    font-size: 12px;
    padding: 4px 0;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 4px;
    transition: color 0.15s ease;
}

.sidebar-bio-toggle:hover {
    color: #2563eb;
}

.sidebar-bio-toggle .toggle-icon {
    font-size: 10px;
    transition: transform 0.2s ease;
}

.sidebar-bio-toggle.expanded .toggle-icon {
    transform: rotate(180deg);
}

.sidebar-bio-text {
    margin-top: 8px;
    padding: 12px;
    background: #fafafa;
    border: 1px solid #e5e5e5;
    border-radius: 6px;
    font-size: 12px;
    color: #4b5563;
    line-height: 1.6;
    max-height: 200px;
    overflow-y: auto;
}

/* Entry navigation in sidebar */
.sidebar-entry-nav {
    background: #ffffff;
    border: 1px solid #e5e5e5;
    border-radius: 8px;
    padding: 12px;
}

.sidebar-entry-nav-header {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #9ca3af;
    margin-bottom: 8px;
    padding-bottom: 8px;
    border-bottom: 1px solid #e5e5e5;
}

.sidebar-entry-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.sidebar-entry-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 10px;
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.15s ease;
    background: transparent;
    border: 1px solid transparent;
}

.sidebar-entry-item:hover:not(.locked) {
    background: #f5f5f5;
}

.sidebar-entry-item.current {
    background: #eff6ff;
    border-color: #3b82f6;
}

.sidebar-entry-item.locked {
    opacity: 0.5;
    cursor: not-allowed;
}

.sidebar-entry-status {
    font-size: 12px;
    width: 16px;
    text-align: center;
    flex-shrink: 0;
}

.sidebar-entry-status.annotated { color: #10b981; }
.sidebar-entry-status.current { color: #3b82f6; }
.sidebar-entry-status.locked { color: #9ca3af; }

.sidebar-entry-info {
    flex: 1;
    min-width: 0;
}

.sidebar-entry-title {
    font-size: 13px;
    font-weight: 500;
    color: #1a1a1a;
}

.sidebar-entry-date {
    font-size: 11px;
    color: #9ca3af;
}

/* ============================================
   CENTER COLUMN STYLES
   ============================================ */
.center-column {
    display: flex;
    flex-direction: column;
    gap: 16px;
}

.center-entry-display {
    background: #ffffff;
    border: 1px solid #e5e5e5;
    border-radius: 8px;
    padding: 24px;
}

.center-entry-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
    padding-bottom: 16px;
    border-bottom: 1px solid #e5e5e5;
}

.center-entry-number {
    font-weight: 600;
    color: #1a1a1a;
    font-size: 16px;
}

.center-entry-date {
    color: #9ca3af;
    font-size: 14px;
}

.center-entry-text {
    font-size: 16px;
    line-height: 1.8;
    color: #1a1a1a;
    white-space: pre-wrap;
}

.center-entry-nudge {
    margin-top: 24px;
    padding: 16px 20px;
    background: #fffbeb;
    border-left: 4px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    font-size: 15px;
    line-height: 1.7;
    color: #92400e;
}

.center-entry-response {
    margin-top: 16px;
    padding: 16px 20px;
    background: #ecfdf5;
    border-left: 4px solid #10b981;
    border-radius: 0 8px 8px 0;
    font-size: 15px;
    line-height: 1.7;
    color: #065f46;
}

/* ============================================
   ENTRY LIST (Legacy - kept for compatibility)
   ============================================ */
.entry-list {
    margin-bottom: 16px;
}

.entry-card {
    background: #ffffff;
    border: 1px solid #e5e5e5;
    border-radius: 6px;
    margin-bottom: 8px;
    transition: all 0.15s ease;
    cursor: pointer;
}

.entry-card:hover:not(.locked) {
    border-color: #3b82f6;
    box-shadow: 0 2px 4px rgba(59, 130, 246, 0.1);
}

.entry-card.expanded {
    border-color: #3b82f6;
    box-shadow: 0 2px 8px rgba(59, 130, 246, 0.15);
}

.entry-card.locked {
    opacity: 0.6;
    cursor: not-allowed;
}

.entry-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
}

.entry-card-left {
    display: flex;
    align-items: center;
    gap: 10px;
}

.entry-card-status {
    font-size: 14px;
    width: 20px;
    text-align: center;
}

.entry-card-status.annotated {
    color: #10b981;
}

.entry-card-status.current {
    color: #3b82f6;
}

.entry-card-status.locked {
    color: #9ca3af;
}

.entry-card-title {
    font-weight: 500;
    font-size: 14px;
    color: #1a1a1a;
}

.entry-card-date {
    font-size: 13px;
    color: #9ca3af;
}

/* ============================================
   NOTES & CONFIDENCE SECTION
   ============================================ */
.notes-section {
    margin-top: 20px;
    padding-top: 20px;
    border-top: 1px solid #e5e5e5;
}

.notes-section label {
    font-size: 13px;
    font-weight: 500;
    color: #374151;
    margin-bottom: 6px;
    display: block;
}

.notes-section textarea {
    width: 100%;
    border: 1px solid #e5e5e5;
    border-radius: 6px;
    padding: 12px;
    font-size: 14px;
    line-height: 1.5;
    resize: vertical;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}

.notes-section textarea:focus {
    outline: none;
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

/* Confidence section */
.confidence-section {
    margin-top: 16px;
}

.confidence-section label {
    font-size: 13px;
    font-weight: 500;
    color: #374151;
    margin-bottom: 8px;
    display: block;
}

.confidence-scale {
    display: flex;
    gap: 8px;
    align-items: center;
}

.confidence-label {
    font-size: 11px;
    color: #9ca3af;
    min-width: 50px;
}

.confidence-label.left { text-align: right; }
.confidence-label.right { text-align: left; }

.confidence-section .shiny-input-radiogroup {
    display: flex !important;
    gap: 4px;
}

.confidence-section .shiny-input-radiogroup label {
    display: none !important;
}

.confidence-section input[type="radio"] {
    -webkit-appearance: none;
    -moz-appearance: none;
    appearance: none;
    width: 36px;
    height: 36px;
    border: 1px solid #e5e5e5;
    border-radius: 6px;
    cursor: pointer;
    background: #ffffff;
    transition: all 0.15s ease;
    display: flex;
    align-items: center;
    justify-content: center;
}

.confidence-section input[type="radio"]:hover {
    border-color: #9ca3af;
    background: #fafafa;
}

.confidence-section input[type="radio"]:checked {
    background: #3b82f6;
    border-color: #3b82f6;
    color: #ffffff;
}

.confidence-section input[type="radio"]::after {
    content: attr(value);
    font-size: 14px;
    font-weight: 500;
    color: #6b7280;
}

.confidence-section input[type="radio"]:checked::after {
    color: #ffffff;
}

/* ============================================
   SAVE BUTTON
   ============================================ */
.save-section {
    margin-top: 20px;
    display: flex;
    justify-content: flex-end;
}

.save-section .btn-primary {
    background: #3b82f6;
    border: none;
    color: #ffffff;
    padding: 12px 24px;
    border-radius: 6px;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s ease;
}

.save-section .btn-primary:hover {
    background: #2563eb;
}

.save-section .btn-primary:focus {
    outline: none;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.3);
}

.save-section .btn-primary.ready {
    animation: subtle-pulse 2s infinite;
}

@keyframes subtle-pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0); }
    50% { box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.2); }
}

/* ============================================
   STATUS INDICATORS
   ============================================ */
.annotation-status {
    padding: 10px 14px;
    border-radius: 6px;
    font-size: 13px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.annotation-status.already-annotated {
    background: #ecfdf5;
    color: #065f46;
    border: 1px solid #d1fae5;
}

.annotation-status.not-annotated {
    background: #fef2f2;
    color: #991b1b;
    border: 1px solid #fecaca;
}

.no-annotator-warning {
    background: #fffbeb;
    color: #92400e;
    padding: 16px;
    border-radius: 8px;
    text-align: center;
    margin-bottom: 16px;
    border: 1px solid #fde68a;
    font-size: 14px;
}

/* ============================================
   KEYBOARD SHORTCUTS HELP
   ============================================ */
.keyboard-help {
    background: #ffffff;
    border: 1px solid #e5e5e5;
    border-radius: 8px;
    padding: 20px;
    margin-top: 16px;
}

.keyboard-help h4 {
    margin: 0 0 12px 0;
    font-size: 14px;
    font-weight: 600;
    color: #374151;
}

.shortcut-grid {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 8px 16px;
    font-size: 13px;
}

.shortcut-key {
    font-family: ui-monospace, SFMono-Regular, 'SF Mono', Menlo, monospace;
    background: #f3f4f6;
    border: 1px solid #e5e5e5;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 12px;
    color: #374151;
}

.shortcut-desc {
    color: #6b7280;
}

/* ============================================
   MODAL STYLES
   ============================================ */
.modal-content {
    border-radius: 12px;
    border: none;
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
}

.modal-header {
    border-bottom: 1px solid #e5e5e5;
    padding: 20px 24px;
}

.modal-body {
    padding: 24px;
}

.modal-footer {
    border-top: 1px solid #e5e5e5;
    padding: 16px 24px;
}

.warning-modal-content {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 16px;
}

.warning-modal-content h4 {
    margin: 0 0 8px 0;
    color: #92400e;
    font-size: 15px;
}

.warning-modal-content p {
    margin: 0;
    color: #92400e;
    font-size: 14px;
    line-height: 1.6;
}

.warning-examples {
    margin-top: 12px;
    font-size: 13px;
    color: #78716c;
}

.warning-examples ul {
    margin: 4px 0 0 0;
    padding-left: 20px;
}

/* ============================================
   ANIMATIONS
   ============================================ */
.fade-in {
    animation: fadeIn 0.2s ease;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(4px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ============================================
   COMPARISON MODAL STYLES
   ============================================ */
.comparison-modal-content {
    padding: 0;
}

.comparison-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
}

.comparison-table th {
    text-align: left;
    padding: 8px 12px;
    background: #f9fafb;
    border-bottom: 2px solid #e5e5e5;
    font-weight: 600;
    color: #374151;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.comparison-table th:not(:first-child) {
    text-align: center;
}

.comparison-table td {
    padding: 10px 12px;
    border-bottom: 1px solid #f3f4f6;
}

.comparison-table td:not(:first-child) {
    text-align: center;
    font-weight: 500;
}

.comparison-row.match-exact {
    background: #ecfdf5;
}

.comparison-row.match-adjacent {
    background: #fefce8;
}

.comparison-row.match-disagree {
    background: #fef2f2;
}

.comparison-value-name {
    color: #1a1a1a;
    font-weight: 500;
}

.comparison-score {
    display: inline-block;
    min-width: 28px;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 600;
}

.comparison-score.positive {
    color: #10b981;
    background: #d1fae5;
}

.comparison-score.negative {
    color: #ef4444;
    background: #fecaca;
}

.comparison-score.neutral {
    color: #6b7280;
    background: #f3f4f6;
}

.comparison-match {
    font-size: 16px;
}

.comparison-match.exact {
    color: #10b981;
}

.comparison-match.adjacent {
    color: #f59e0b;
}

.comparison-match.disagree {
    color: #ef4444;
}

.comparison-summary {
    margin-top: 16px;
    padding: 12px 16px;
    background: #f9fafb;
    border-radius: 6px;
    font-size: 14px;
    color: #374151;
    text-align: center;
}

.comparison-summary-count {
    font-weight: 600;
    font-size: 16px;
}

.comparison-continue-btn {
    background: #3b82f6;
    border: none;
    color: #ffffff;
    padding: 12px 24px;
    border-radius: 6px;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s ease;
    width: 100%;
    margin-top: 16px;
}

.comparison-continue-btn:hover {
    background: #2563eb;
}

.comparison-no-labels {
    text-align: center;
    padding: 24px;
    color: #6b7280;
}

.comparison-no-labels-icon {
    font-size: 32px;
    margin-bottom: 12px;
}

.comparison-no-labels-text {
    font-size: 14px;
    line-height: 1.6;
}
"""


def create_scoring_row(value: str, row_index: int) -> ui.TagChild:
    """Create a scoring row for a single Schwartz value with -/+ counter buttons."""
    label = VALUE_LABELS[value]
    return ui.div(
        ui.div(label, class_="value-label"),
        ui.div(
            ui.input_action_button(
                f"dec_{value}", "−", class_="score-btn dec"
            ),
            ui.output_ui(f"score_display_{value}"),
            ui.input_action_button(
                f"inc_{value}", "+", class_="score-btn inc"
            ),
            class_="score-btn-group",
        ),
        class_="scoring-row",
        id=f"row_{value}",
        **{"data-row-index": str(row_index), "data-value": value},
    )


def create_value_group(group_name: str, values: list, start_index: int) -> ui.TagChild:
    """Create a value group with header and rows."""
    # Map group names to CSS classes
    css_class_map = {
        "OPENNESS TO CHANGE": "openness",
        "SELF-ENHANCEMENT": "enhancement",
        "CONSERVATION": "conservation",
        "SELF-TRANSCENDENCE": "transcendence",
    }
    css_class = css_class_map.get(group_name, "")

    rows = []
    for i, value in enumerate(values):
        rows.append(create_scoring_row(value, start_index + i))

    return ui.div(
        ui.div(group_name, class_=f"value-group-header {css_class}"),
        *rows,
        class_="value-group",
    )


def create_grouped_scoring_grid() -> list:
    """Create all value groups for the scoring grid."""
    groups = []
    current_index = 0

    for group_name, values in VALUE_GROUPS.items():
        groups.append(create_value_group(group_name, values, current_index))
        current_index += len(values)

    return groups


# JavaScript for keyboard shortcuts
keyboard_js = """
<script>
document.addEventListener('DOMContentLoaded', function() {
    // Value order for keyboard navigation
    const valueOrder = ['self_direction', 'stimulation', 'hedonism', 'achievement',
                        'power', 'security', 'conformity', 'tradition',
                        'benevolence', 'universalism'];
    let focusedRowIndex = -1;
    let helpVisible = false;

    function updateFocusedRow(newIndex) {
        // Remove previous focus
        document.querySelectorAll('.scoring-row.focused').forEach(el => {
            el.classList.remove('focused');
        });

        if (newIndex >= 0 && newIndex < valueOrder.length) {
            focusedRowIndex = newIndex;
            const value = valueOrder[focusedRowIndex];
            const row = document.getElementById('row_' + value);
            if (row) {
                row.classList.add('focused');
                row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        }
    }

    function cycleScore(direction) {
        if (focusedRowIndex < 0 || focusedRowIndex >= valueOrder.length) return;

        const value = valueOrder[focusedRowIndex];
        // direction: -1 = decrement, 1 = increment
        const btnId = direction < 0 ? 'dec_' + value : 'inc_' + value;
        const btn = document.getElementById(btnId);
        if (btn) btn.click();
    }

    function jumpToValue(num) {
        // 1-9 for first 9 values, 0 for 10th (Universalism)
        let index = num === 0 ? 9 : num - 1;
        if (index >= 0 && index < valueOrder.length) {
            updateFocusedRow(index);
        }
    }

    function showHelp() {
        const modal = document.getElementById('keyboard-help-modal');
        const backdrop = document.getElementById('keyboard-help-backdrop');
        if (modal && backdrop) {
            const newState = !helpVisible;
            modal.style.display = newState ? 'block' : 'none';
            backdrop.style.display = newState ? 'block' : 'none';
            helpVisible = newState;
        }
    }

    // Global keyboard handler
    document.addEventListener('keydown', function(e) {
        // Ignore if typing in an input field
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
            // Allow Enter in input fields to trigger save
            if (e.key === 'Enter' && e.target.id === 'annotator_name') {
                return; // Let default behavior happen
            }
            return;
        }

        switch(e.key) {
            case 'ArrowUp':
                e.preventDefault();
                updateFocusedRow(focusedRowIndex <= 0 ? valueOrder.length - 1 : focusedRowIndex - 1);
                break;
            case 'ArrowDown':
                e.preventDefault();
                updateFocusedRow(focusedRowIndex >= valueOrder.length - 1 ? 0 : focusedRowIndex + 1);
                break;
            case 'ArrowLeft':
                e.preventDefault();
                cycleScore(-1);
                break;
            case 'ArrowRight':
                e.preventDefault();
                cycleScore(1);
                break;
            case 'Enter':
                e.preventDefault();
                const saveBtn = document.getElementById('save_btn');
                if (saveBtn) saveBtn.click();
                break;
            case 'Backspace':
                e.preventDefault();
                const prevBtn = document.getElementById('prev_btn');
                if (prevBtn) prevBtn.click();
                break;
            case '?':
                e.preventDefault();
                showHelp();
                break;
            case 'Escape':
                if (helpVisible) {
                    showHelp();
                }
                focusedRowIndex = -1;
                document.querySelectorAll('.scoring-row.focused').forEach(el => {
                    el.classList.remove('focused');
                });
                break;
            default:
                // Number keys 0-9 for jumping to values
                if (e.key >= '0' && e.key <= '9') {
                    e.preventDefault();
                    jumpToValue(parseInt(e.key));
                }
        }
    });

    // Initialize focus on first row if none selected
    if (focusedRowIndex === -1) {
        focusedRowIndex = 0;
    }

    // Help button click handler
    const helpBtn = document.getElementById('help_btn');
    if (helpBtn) {
        helpBtn.addEventListener('click', function(e) {
            e.preventDefault();
            showHelp();
        });
    }

    // Entry click handler - using event delegation for both sidebar and legacy cards
    document.addEventListener('click', function(e) {
        // Check for sidebar entry items first
        const sidebarItem = e.target.closest('.sidebar-entry-item');
        if (sidebarItem && !sidebarItem.classList.contains('locked')) {
            const entryIndex = sidebarItem.getAttribute('data-entry-index');
            if (entryIndex !== null) {
                Shiny.setInputValue('selected_entry_index', parseInt(entryIndex), {priority: 'event'});
            }
            return;
        }

        // Fall back to legacy entry card handling
        const entryCard = e.target.closest('.entry-card');
        if (entryCard && !entryCard.classList.contains('locked')) {
            const entryIndex = entryCard.getAttribute('data-entry-index');
            if (entryIndex !== null) {
                Shiny.setInputValue('selected_entry_index', parseInt(entryIndex), {priority: 'event'});
            }
        }
    });
});
</script>
"""

# Keyboard help modal HTML
keyboard_help_modal = """
<div id="keyboard-help-modal" style="display:none; position:fixed; top:50%; left:50%; transform:translate(-50%,-50%);
    background:white; border-radius:12px; padding:24px; box-shadow:0 20px 25px -5px rgba(0,0,0,0.1); z-index:1000; max-width:400px;">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
        <h3 style="margin:0; font-size:16px; font-weight:600;">Keyboard Shortcuts</h3>
        <button onclick="document.getElementById('keyboard-help-modal').style.display='none'"
            style="background:none; border:none; font-size:20px; cursor:pointer; color:#9ca3af;">×</button>
    </div>
    <div class="shortcut-grid">
        <span class="shortcut-key">↑ / ↓</span><span class="shortcut-desc">Navigate between values</span>
        <span class="shortcut-key">← / →</span><span class="shortcut-desc">Cycle score (−/○/+)</span>
        <span class="shortcut-key">1-9, 0</span><span class="shortcut-desc">Jump to value (1=Self-Direction, 0=Universalism)</span>
        <span class="shortcut-key">Enter</span><span class="shortcut-desc">Save & Next</span>
        <span class="shortcut-key">Backspace</span><span class="shortcut-desc">Previous entry</span>
        <span class="shortcut-key">?</span><span class="shortcut-desc">Toggle this help</span>
        <span class="shortcut-key">Esc</span><span class="shortcut-desc">Clear focus</span>
    </div>
</div>
<div id="keyboard-help-backdrop" onclick="document.getElementById('keyboard-help-modal').style.display='none'"
    style="display:none; position:fixed; top:0; left:0; right:0; bottom:0; background:rgba(0,0,0,0.3); z-index:999;"></div>
"""

app_ui = ui.page_fluid(
    ui.tags.style(app_css),
    ui.HTML(keyboard_js),
    ui.HTML(keyboard_help_modal),
    # Header
    ui.div(
        ui.div(
            ui.div(
                ui.h2("Schwartz Value Annotation Tool", class_="app-title"),
                ui.input_action_button("help_btn", "? Help", class_="help-btn"),
                class_="header-left",
            ),
            ui.div(
                ui.input_text(
                    id="annotator_name",
                    label=None,
                    placeholder="Enter your name...",
                    width="180px",
                ),
                class_="annotator-input",
            ),
            class_="header-top-row",
        ),
        ui.div(
            ui.div(
                ui.output_ui("progress_display"),
                class_="progress-section",
            ),
            ui.div(
                ui.input_action_button("prev_btn", "◀ Prev Persona", class_="btn-secondary"),
                ui.input_action_button("next_btn", "Next Persona ▶", class_="btn-secondary"),
                class_="nav-buttons",
            ),
            class_="header-nav-row",
        ),
        class_="header-component",
    ),
    # Main content area - Three column layout
    ui.div(
        ui.output_ui("annotator_warning"),
        ui.div(
            # Left sidebar: Persona info + Entry navigation
            ui.div(
                # Persona section
                ui.output_ui("sidebar_persona"),
                # Entry navigation
                ui.output_ui("sidebar_entry_nav"),
                class_="left-sidebar",
            ),
            # Center column: Annotation status + Entry content
            ui.div(
                ui.output_ui("annotation_status"),
                ui.div(
                    ui.output_ui("entry_content"),
                    class_="center-entry-display",
                ),
                class_="center-column",
            ),
            # Right column: Scoring grid (sticky)
            ui.div(
                ui.div(
                    ui.div(
                        ui.div("Value(s)", class_="scoring-header-label"),
                        ui.div(
                            ui.span("Score", title="−1 = Misaligned, 0 = Neutral, +1 = Aligned"),
                            class_="scoring-header-scores",
                        ),
                        class_="scoring-header",
                    ),
                    # Grouped value rows
                    *create_grouped_scoring_grid(),
                    # Notes section
                    ui.div(
                        ui.input_text_area(
                            id="notes",
                            label="Notes (optional)",
                            placeholder="Any observations about this entry...",
                            rows=2,
                        ),
                        class_="notes-section",
                    ),
                    # Save button
                    ui.div(
                        ui.input_action_button(
                            id="save_btn",
                            label="Save & Next →",
                            class_="btn-primary",
                        ),
                        class_="save-section",
                    ),
                    class_="scoring-grid",
                ),
                class_="right-column",
            ),
            class_="three-column-layout",
        ),
        class_="main-container",
    ),
)


def server(input, output, session):
    # Core reactive state - per-persona navigation
    current_persona_index = reactive.value(0)  # Which persona
    expanded_entry_index = reactive.value(0)   # Which entry is expanded within persona
    annotated_count = reactive.value(0)
    bio_expanded = reactive.value(True)  # Bio collapsible state (default expanded)

    # Score reactive values - one per Schwartz value
    scores = {value: reactive.value(0) for value in SCHWARTZ_VALUE_ORDER}

    # Pending save data (for modal confirmation)
    pending_save = reactive.value(None)

    # Create score display renderers for each Schwartz value
    def make_score_display_renderer(value: str):
        @output(id=f"score_display_{value}")
        @render.ui
        def _():
            score = scores[value]()
            if score == -1:
                display = "−1"
                css_class = "score-display negative"
            elif score == 1:
                display = "+1"
                css_class = "score-display positive"
            else:
                display = "0"
                css_class = "score-display neutral"
            return ui.span(display, class_=css_class)
        return _

    # Create score button handlers for each Schwartz value
    def make_dec_handler(value: str):
        @reactive.effect
        @reactive.event(input[f"dec_{value}"])
        def _():
            current = scores[value]()
            if current > -1:
                scores[value].set(current - 1)
        return _

    def make_inc_handler(value: str):
        @reactive.effect
        @reactive.event(input[f"inc_{value}"])
        def _():
            current = scores[value]()
            if current < 1:
                scores[value].set(current + 1)
        return _

    # Register all score display renderers and button handlers
    for value in SCHWARTZ_VALUE_ORDER:
        make_score_display_renderer(value)
        make_dec_handler(value)
        make_inc_handler(value)

    # Current persona's entries
    @reactive.calc
    def current_persona_entries():
        idx = current_persona_index()
        if 0 <= idx < _total_personas:
            return _personas[idx]
        return []

    # Currently expanded/active entry
    @reactive.calc
    def current_entry():
        entries = current_persona_entries()
        entry_idx = expanded_entry_index()
        if entries and 0 <= entry_idx < len(entries):
            return entries[entry_idx]
        return None

    # Number of unlocked entries for current persona
    # Entry N+1 unlocks after Entry N is saved
    @reactive.calc
    def unlocked_entry_count():
        # Depend on annotated_count to trigger re-evaluation after saves
        _ = annotated_count()

        entries = current_persona_entries()
        if not entries:
            return 0

        annotator = input.annotator_name()
        if not annotator:
            return 1  # Only first entry unlocked if no annotator

        unlocked = 1  # Entry 0 always unlocked
        for i in range(len(entries) - 1):
            entry = entries[i]
            if get_annotation(annotator, entry["persona_id"], entry["t_index"]):
                unlocked = i + 2  # Next entry is unlocked
            else:
                break
        return min(unlocked, len(entries))

    # Progress display
    @output
    @render.ui
    def progress_display():
        count = annotated_count()
        percentage = (count / _total_entries * 100) if _total_entries > 0 else 0
        persona_idx = current_persona_index()
        entries = current_persona_entries()
        entry_idx = expanded_entry_index()

        return ui.div(
            ui.div(
                f"Persona {persona_idx + 1} of {_total_personas} • "
                f"Entry {entry_idx + 1} of {len(entries)}",
                style="margin-bottom: 8px; color: #6b7280;",
            ),
            ui.div(
                ui.div(
                    ui.div(style=f"width: {percentage:.0f}%", class_="progress-fill"),
                    class_="progress-bar-container",
                ),
                ui.span(
                    f"{count}/{_total_entries} annotated ({percentage:.0f}%)",
                    class_="progress-text",
                ),
                class_="progress-wrapper",
            ),
        )

    # Entry list showing all entries for current persona
    @output
    @render.ui
    def entry_list():
        entries = current_persona_entries()
        if not entries:
            return ui.div("No entries for this persona")

        annotator = input.annotator_name()
        unlocked = unlocked_entry_count()
        current_idx = expanded_entry_index()

        cards = []
        for i, entry in enumerate(entries):
            is_locked = i >= unlocked
            is_current = i == current_idx
            is_annotated = False

            if annotator:
                existing = get_annotation(
                    annotator, entry["persona_id"], entry["t_index"]
                )
                is_annotated = existing is not None

            # Determine status icon
            if is_locked:
                status_icon = "🔒"
                status_class = "locked"
            elif is_annotated:
                status_icon = "✓"
                status_class = "annotated"
            elif is_current:
                status_icon = "●"
                status_class = "current"
            else:
                status_icon = "○"
                status_class = ""

            # Determine card classes
            card_classes = "entry-card"
            if is_current:
                card_classes += " expanded"
            if is_locked:
                card_classes += " locked"

            t_index = entry.get("t_index", 0)
            date = entry.get("date", "")

            card = ui.div(
                ui.div(
                    ui.div(
                        ui.span(status_icon, class_=f"entry-card-status {status_class}"),
                        ui.span(f"Entry {t_index + 1}", class_="entry-card-title"),
                        class_="entry-card-left",
                    ),
                    ui.span(date, class_="entry-card-date"),
                    class_="entry-card-header",
                ),
                class_=card_classes,
                id=f"entry_card_{i}",
                **{"data-entry-index": str(i)},
            )
            cards.append(card)

        return ui.div(*cards, class_="entry-list")

    # Sidebar persona section (compact with collapsible bio)
    @output
    @render.ui
    def sidebar_persona():
        entry = current_entry()
        if entry is None:
            return ui.div("No persona selected", class_="sidebar-persona")

        # Format core values
        core_values = entry.get("persona_core_values", [])
        if isinstance(core_values, list):
            values_str = ", ".join(core_values)
        else:
            values_str = str(core_values)

        # Build persona info parts on separate lines for narrow sidebar
        age = entry.get("persona_age", "")
        profession = entry.get("persona_profession", "")
        culture = entry.get("persona_culture", "")

        # Bio section with collapsible toggle
        bio = entry.get("persona_bio", "")
        is_expanded = bio_expanded()

        bio_section = None
        if bio:
            toggle_class = "sidebar-bio-toggle expanded" if is_expanded else "sidebar-bio-toggle"
            toggle_text = "▲ Hide Bio" if is_expanded else "▼ Show Bio"
            bio_section = ui.div(
                ui.input_action_button(
                    "bio_toggle",
                    ui.span(toggle_text),
                    class_=toggle_class,
                ),
                ui.div(bio, class_="sidebar-bio-text") if is_expanded else None,
                class_="sidebar-bio-container",
            )

        return ui.div(
            ui.h4(entry.get('persona_name', 'Unknown'), class_="sidebar-persona-name"),
            ui.div(f"{age} • {profession}", class_="sidebar-persona-info") if age and profession else None,
            ui.div(culture, class_="sidebar-persona-info") if culture else None,
            ui.div(f"Core: {values_str}", class_="sidebar-persona-values"),
            bio_section,
            class_="sidebar-persona",
        )

    # Sidebar entry navigation (compact list)
    @output
    @render.ui
    def sidebar_entry_nav():
        entries = current_persona_entries()
        if not entries:
            return ui.div("No entries", class_="sidebar-entry-nav")

        annotator = input.annotator_name()
        unlocked = unlocked_entry_count()
        current_idx = expanded_entry_index()

        items = []
        for i, entry in enumerate(entries):
            is_locked = i >= unlocked
            is_current = i == current_idx
            is_annotated = False

            if annotator:
                existing = get_annotation(
                    annotator, entry["persona_id"], entry["t_index"]
                )
                is_annotated = existing is not None

            # Determine status icon
            if is_locked:
                status_icon = "🔒"
                status_class = "locked"
            elif is_annotated:
                status_icon = "✓"
                status_class = "annotated"
            elif is_current:
                status_icon = "●"
                status_class = "current"
            else:
                status_icon = "○"
                status_class = ""

            # Determine item classes
            item_classes = "sidebar-entry-item"
            if is_current:
                item_classes += " current"
            if is_locked:
                item_classes += " locked"

            t_index = entry.get("t_index", 0)
            date = entry.get("date", "")

            item = ui.div(
                ui.span(status_icon, class_=f"sidebar-entry-status {status_class}"),
                ui.div(
                    ui.div(f"Entry {t_index + 1}", class_="sidebar-entry-title"),
                    ui.div(date, class_="sidebar-entry-date"),
                    class_="sidebar-entry-info",
                ),
                class_=item_classes,
                id=f"sidebar_entry_{i}",
                **{"data-entry-index": str(i)},
            )
            items.append(item)

        return ui.div(
            ui.div("Entries", class_="sidebar-entry-nav-header"),
            ui.div(*items, class_="sidebar-entry-list"),
            class_="sidebar-entry-nav",
        )

    # Bio toggle handler
    @reactive.effect
    @reactive.event(input.bio_toggle)
    def _on_bio_toggle():
        bio_expanded.set(not bio_expanded())

    # Persona header (legacy - kept for compatibility)
    @output
    @render.ui
    def persona_header():
        entry = current_entry()
        if entry is None:
            return ui.div("No entry selected")

        # Format core values
        core_values = entry.get("persona_core_values", [])
        if isinstance(core_values, list):
            values_str = ", ".join(core_values)
        else:
            values_str = str(core_values)

        # Build persona info line
        info_parts = [
            entry.get("persona_age", ""),
            entry.get("persona_profession", ""),
            entry.get("persona_culture", ""),
        ]
        info_line = " • ".join(p for p in info_parts if p)

        # Bio section - always visible
        bio = entry.get("persona_bio", "")

        return ui.div(
            ui.h4(f"PERSONA: {entry.get('persona_name', 'Unknown')}", class_="persona-name"),
            ui.div(info_line, class_="persona-info"),
            ui.div(f"Core Values: {values_str}", class_="persona-values"),
            ui.div(bio, class_="persona-bio-text") if bio else None,
        )

    # Entry content (center column)
    @output
    @render.ui
    def entry_content():
        entry = current_entry()
        if entry is None:
            return ui.div("No entry selected")

        t_index = entry.get("t_index", 0)
        entries = current_persona_entries()
        total = len(entries)
        date = entry.get("date", "")

        content_parts = [
            ui.div(
                ui.span(f"Entry {t_index + 1} of {total}", class_="center-entry-number"),
                ui.span(date, class_="center-entry-date"),
                class_="center-entry-header",
            ),
            ui.div(entry.get("initial_entry", ""), class_="center-entry-text"),
        ]

        # Nudge (if present)
        if entry.get("has_nudge") and entry.get("nudge_text"):
            content_parts.append(
                ui.div(
                    ui.strong("Nudge: "),
                    f'"{entry["nudge_text"]}"',
                    class_="center-entry-nudge",
                )
            )

        # Response (if present)
        if entry.get("has_response") and entry.get("response_text"):
            content_parts.append(
                ui.div(
                    ui.strong("Response: "),
                    entry["response_text"],
                    class_="center-entry-response",
                )
            )

        return ui.div(*content_parts)

    # Navigation - move between personas
    @reactive.effect
    @reactive.event(input.prev_btn)
    def _on_prev():
        new_idx = current_persona_index() - 1
        if new_idx >= 0:
            current_persona_index.set(new_idx)
            expanded_entry_index.set(0)  # Reset to first entry
            _load_existing_annotation()

    @reactive.effect
    @reactive.event(input.next_btn)
    def _on_next():
        new_idx = current_persona_index() + 1
        if new_idx < _total_personas:
            current_persona_index.set(new_idx)
            expanded_entry_index.set(0)  # Reset to first entry
            _load_existing_annotation()

    # Handle entry selection from clicking on entry cards
    @reactive.effect
    @reactive.event(input.selected_entry_index)
    def _on_entry_select():
        selected_idx = input.selected_entry_index()
        if selected_idx is None:
            return

        # Check if entry is unlocked
        unlocked = unlocked_entry_count()
        if selected_idx < unlocked:
            expanded_entry_index.set(selected_idx)
            _load_existing_annotation()

    # Load existing annotation for current entry
    def _load_existing_annotation():
        name = input.annotator_name()
        entry = current_entry()

        if not name or not entry:
            # Reset all scores to neutral
            for value in SCHWARTZ_VALUE_ORDER:
                scores[value].set(0)
            ui.update_text_area("notes", value="")
            return

        existing = get_annotation(name, entry["persona_id"], entry["t_index"])
        if existing:
            # Load existing scores into reactive values
            for value in SCHWARTZ_VALUE_ORDER:
                score_key = f"alignment_{value}"
                if score_key in existing:
                    scores[value].set(int(existing[score_key]))
                else:
                    scores[value].set(0)
            # Load notes
            if existing.get("notes"):
                ui.update_text_area("notes", value=existing["notes"])
            else:
                ui.update_text_area("notes", value="")
        else:
            # Reset all scores to neutral
            for value in SCHWARTZ_VALUE_ORDER:
                scores[value].set(0)
            ui.update_text_area("notes", value="")

    # Update annotated count when annotator changes
    @reactive.effect
    @reactive.event(input.annotator_name)
    def _update_annotated_count():
        name = input.annotator_name()
        if name:
            annotated_count.set(get_annotation_count(name))
            _load_existing_annotation()
        else:
            annotated_count.set(0)

    # Save button handler
    @reactive.effect
    @reactive.event(input.save_btn)
    def _on_save():
        name = input.annotator_name()
        if not name:
            ui.notification_show(
                "Please enter your name before saving.",
                type="warning",
                duration=3,
            )
            return

        # Collect scores from reactive values
        score_data = {}
        for value in SCHWARTZ_VALUE_ORDER:
            score_data[value] = scores[value]()

        # Get notes (empty string becomes None)
        notes = input.notes()
        if notes == "":
            notes = None

        # Confidence removed from UI, pass None for backward compatibility
        confidence = None

        # Check if all neutral
        all_neutral = all(s == 0 for s in score_data.values())
        if all_neutral:
            # Store pending save and show modal
            pending_save.set({
                "scores": score_data,
                "notes": notes,
                "confidence": confidence,
            })
            m = ui.modal(
                ui.div(
                    ui.div(
                        ui.h4("⚠️ All Neutral Scores", style="margin: 0 0 8px 0; color: #92400e; font-size: 16px;"),
                        ui.p(
                            "You've set all 10 values to neutral (○). This is valid for purely factual entries, "
                            "but please double-check before confirming.",
                            style="margin: 0 0 12px 0; color: #92400e; font-size: 14px; line-height: 1.6;",
                        ),
                        ui.div(
                            ui.strong("When all-neutral is appropriate:", style="display: block; margin-bottom: 6px; color: #78716c; font-size: 13px;"),
                            ui.tags.ul(
                                ui.tags.li("Purely factual entries: \"Ate lunch at noon. Worked until 5pm.\""),
                                ui.tags.li("Logistical notes: \"Dentist appointment next Tuesday.\""),
                                ui.tags.li("Neutral descriptions without emotional content"),
                                style="margin: 0; padding-left: 20px; font-size: 13px; color: #78716c; line-height: 1.6;",
                            ),
                            style="margin-bottom: 12px;",
                        ),
                        ui.div(
                            ui.strong("Consider if there's any:", style="display: block; margin-bottom: 6px; color: #78716c; font-size: 13px;"),
                            ui.tags.ul(
                                ui.tags.li("Value tension or conflict"),
                                ui.tags.li("Goals or aspirations mentioned"),
                                ui.tags.li("Emotional undertones"),
                                ui.tags.li("Relationships or social dynamics"),
                                style="margin: 0; padding-left: 20px; font-size: 13px; color: #78716c; line-height: 1.6;",
                            ),
                        ),
                        class_="warning-modal-content",
                    ),
                    style="padding: 0;",
                ),
                ui.div(
                    ui.input_action_button("modal_cancel", "← Go Back", class_="btn-secondary",
                                          style="background: #fff; border: 1px solid #e5e5e5; color: #374151; padding: 10px 20px; border-radius: 6px;"),
                    ui.input_action_button("modal_confirm", "Save All Neutral", class_="btn-warning",
                                          style="background: #f59e0b; border: none; color: #fff; padding: 10px 20px; border-radius: 6px;"),
                    style="display: flex; gap: 12px; justify-content: flex-end; padding-top: 16px; border-top: 1px solid #e5e5e5;",
                ),
                title=None,
                easy_close=True,
            )
            ui.modal_show(m)
        else:
            # Save directly
            _do_save(score_data, notes, confidence)

    def _do_save(scores: dict, notes: str | None, confidence: int | None):
        """Save the annotation and show comparison modal."""
        name = input.annotator_name()
        entry = current_entry()

        if not name or not entry:
            return

        save_annotation(
            annotator_id=name,
            persona_id=entry["persona_id"],
            t_index=entry["t_index"],
            scores=scores,
            notes=notes,
            confidence=confidence,
        )

        # Update annotated count
        annotated_count.set(get_annotation_count(name))

        # Get judge scores for comparison
        judge_data = get_judge_scores(entry["persona_id"], entry["t_index"])

        # Build and show comparison modal
        modal_content = build_comparison_modal_content(scores, judge_data)

        m = ui.modal(
            modal_content,
            title="Score Comparison",
            easy_close=False,  # Require explicit click to dismiss
            footer=None,  # Button is inside the content
        )
        ui.modal_show(m)

    # Modal button handlers
    @reactive.effect
    @reactive.event(input.modal_cancel)
    def _modal_cancel():
        ui.modal_remove()
        pending_save.set(None)

    @reactive.effect
    @reactive.event(input.modal_confirm)
    def _modal_confirm():
        ui.modal_remove()
        data = pending_save()
        if data:
            _do_save(data["scores"], data["notes"], data["confidence"])
        pending_save.set(None)

    # Comparison modal continue button handler
    @reactive.effect
    @reactive.event(input.comparison_continue)
    def _on_comparison_continue():
        """Handle clicking Continue in the comparison modal - advance to next entry."""
        ui.modal_remove()

        # Advance to next entry within persona
        entries = current_persona_entries()
        current_entry_idx = expanded_entry_index()
        new_entry_idx = current_entry_idx + 1

        if new_entry_idx < len(entries):
            # Move to next entry within this persona
            expanded_entry_index.set(new_entry_idx)
            _load_existing_annotation()
        else:
            # All entries in persona annotated - show completion message
            ui.notification_show(
                "All entries for this persona annotated! Use 'Next Persona' to continue.",
                type="message",
                duration=3,
            )

    # Annotator warning output
    @output
    @render.ui
    def annotator_warning():
        name = input.annotator_name()
        if not name:
            return ui.div(
                "⚠️ Please enter your name above to start annotating.",
                class_="no-annotator-warning",
            )
        return None

    # Annotation status output
    @output
    @render.ui
    def annotation_status():
        name = input.annotator_name()
        entry = current_entry()

        if not name or not entry:
            return None

        existing = get_annotation(name, entry["persona_id"], entry["t_index"])
        if existing:
            return ui.div(
                "✓ You have already annotated this entry. Changes will update your previous annotation.",
                class_="annotation-status already-annotated",
            )
        else:
            return ui.div(
                "○ This entry has not been annotated yet.",
                class_="annotation-status not-annotated",
            )


app = App(app_ui, server)


def _free_port(port: int = 8000) -> None:
    """Kill any process using the specified port before starting the server."""
    import subprocess

    try:
        # Find PIDs using the port
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
        )
        pids = result.stdout.strip().split("\n")
        pids = [p for p in pids if p]  # Filter empty strings

        if pids:
            for pid in pids:
                subprocess.run(["kill", pid], capture_output=True)
            print(f"Killed process(es) on port {port}: {', '.join(pids)}")
    except Exception:
        pass  # Silently continue if lsof isn't available or fails


if __name__ == "__main__":
    from shiny import run_app

    _free_port(8000)
    run_app(app)
