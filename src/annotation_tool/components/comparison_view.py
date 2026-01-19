"""Inline comparison view component for the annotation tool.

This module provides the inline comparison view that replaces the right column
scoring grid after saving an annotation, showing human vs judge score comparison.

Usage:
    from components.comparison_view import build_comparison_view

    view = build_comparison_view(human_scores, judge_data, entry_data)
"""

import json

from shiny import ui

from src.models.judge import SCHWARTZ_VALUE_ORDER

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


def _format_score(score: int) -> tuple[str, str]:
    """Format a score value for display.

    Args:
        score: Integer score (-1, 0, or 1)

    Returns:
        Tuple of (display_text, css_class)
    """
    if score == 1:
        return "+1", "positive"
    elif score == -1:
        return "−1", "negative"
    else:
        return "0", "neutral"


def _get_match_type(human_score: int, judge_score: int) -> tuple[str, str, str]:
    """Determine if human and judge scores match exactly.

    Args:
        human_score: Human annotator's score
        judge_score: LLM Judge's score

    Returns:
        Tuple of (row_class, symbol, symbol_class)
    """
    if human_score == judge_score:
        return "match-exact", "✓", "exact"
    else:
        return "match-disagree", "✗", "disagree"


def build_comparison_view(
    human_scores: dict[str, int],
    judge_data: dict | None,
    entry_data: dict | None = None,
) -> ui.TagChild:
    """Build the inline comparison view showing human vs judge scores.

    This replaces the scoring grid after a save, displaying the comparison
    without a modal overlay. The entry is already visible in the center column,
    so no entry preview is included here.

    Args:
        human_scores: Dict mapping value names to human annotator scores
        judge_data: Full judge label row dict, or None if not available
        entry_data: Current entry dict (unused, kept for API consistency)

    Returns:
        UI element for the inline comparison view
    """
    # Handle missing judge labels - simplified success view
    if judge_data is None:
        return ui.div(
            ui.div(
                ui.div("✓", class_="comparison-no-labels-icon"),
                ui.div(
                    ui.strong("Annotation saved!"),
                    ui.br(),
                    "No Judge labels available for this entry.",
                    class_="comparison-no-labels-text",
                ),
                ui.input_action_button(
                    "continue_btn",
                    "Continue →",
                    class_="comparison-continue-btn",
                ),
                class_="comparison-no-labels",
            ),
            class_="comparison-view",
        )

    # Parse rationales from judge_data
    rationales = {}
    if judge_data.get("rationales_json"):
        try:
            rationales = json.loads(judge_data["rationales_json"])
        except (json.JSONDecodeError, TypeError):
            pass

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

        # Build the main row (4 columns: Value, You, Judge, Match)
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

        # Add always-visible rationale row for non-zero judge values
        show_rationale = judge_score != 0 and value in rationales
        if show_rationale:
            rows.append(
                ui.tags.tr(
                    ui.tags.td(
                        ui.div(
                            rationales[value],
                            class_="comparison-rationale-content",
                        ),
                        colspan="4",
                        class_="comparison-rationale-cell",
                    ),
                    class_="comparison-rationale-row",
                )
            )

    # Build the comparison view
    return ui.div(
        # Title
        ui.div(
            "Score Comparison",
            class_="comparison-view-title",
        ),
        # Comparison table
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
        # Summary
        ui.div(
            ui.span(f"{exact_matches}/10", class_="comparison-summary-count"),
            " exact matches",
            class_="comparison-summary",
        ),
        # Continue button
        ui.input_action_button(
            "continue_btn",
            "Continue →",
            class_="comparison-continue-btn",
        ),
        class_="comparison-view",
    )
