"""Modal components for the annotation tool.

This module provides modal dialogs for:
- All-neutral warning confirmation
- Human vs Judge score comparison
- Keyboard shortcuts help

Usage:
    from components.modals import (
        show_all_neutral_modal,
        show_comparison_modal,
        get_keyboard_help_html,
    )
"""

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
    """Determine the match type between human and judge scores.

    Args:
        human_score: Human annotator's score
        judge_score: LLM Judge's score

    Returns:
        Tuple of (row_class, symbol, symbol_class)
    """
    diff = abs(human_score - judge_score)
    if diff == 0:
        return "match-exact", "✓", "exact"
    elif diff == 1:
        return "match-adjacent", "~", "adjacent"
    else:
        return "match-disagree", "✗", "disagree"


def build_all_neutral_modal() -> ui.Tag:
    """Build the all-neutral warning modal content.

    Returns:
        UI modal element ready to be shown with ui.modal_show()
    """
    return ui.modal(
        ui.div(
            ui.div(
                ui.h4(
                    "⚠️ All Neutral Scores",
                    style="margin: 0 0 8px 0; color: #92400e; font-size: 16px;",
                ),
                ui.p(
                    "You've set all 10 values to neutral (○). This is valid for purely factual entries, "
                    "but please double-check before confirming.",
                    style="margin: 0 0 12px 0; color: #92400e; font-size: 14px; line-height: 1.6;",
                ),
                ui.div(
                    ui.strong(
                        "When all-neutral is appropriate:",
                        style="display: block; margin-bottom: 6px; color: #78716c; font-size: 13px;",
                    ),
                    ui.tags.ul(
                        ui.tags.li('Purely factual entries: "Ate lunch at noon. Worked until 5pm."'),
                        ui.tags.li('Logistical notes: "Dentist appointment next Tuesday."'),
                        ui.tags.li("Neutral descriptions without emotional content"),
                        style="margin: 0; padding-left: 20px; font-size: 13px; color: #78716c; line-height: 1.6;",
                    ),
                    style="margin-bottom: 12px;",
                ),
                ui.div(
                    ui.strong(
                        "Consider if there's any:",
                        style="display: block; margin-bottom: 6px; color: #78716c; font-size: 13px;",
                    ),
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
            ui.input_action_button(
                "modal_cancel",
                "← Go Back",
                class_="btn-secondary",
                style="background: #fff; border: 1px solid #e5e5e5; color: #374151; padding: 10px 20px; border-radius: 6px;",
            ),
            ui.input_action_button(
                "modal_confirm",
                "Save All Neutral",
                class_="btn-warning",
                style="background: #f59e0b; border: none; color: #fff; padding: 10px 20px; border-radius: 6px;",
            ),
            style="display: flex; gap: 12px; justify-content: flex-end; padding-top: 16px; border-top: 1px solid #e5e5e5;",
        ),
        title=None,
        easy_close=True,
    )


def build_comparison_modal_content(
    human_scores: dict[str, int], judge_data: dict | None
) -> ui.TagChild:
    """Build the comparison modal content showing human vs judge scores.

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


def build_comparison_modal(human_scores: dict[str, int], judge_data: dict | None) -> ui.Tag:
    """Build the full comparison modal.

    Args:
        human_scores: Dict mapping value names to human annotator scores
        judge_data: Full judge label row dict, or None if not available

    Returns:
        UI modal element ready to be shown with ui.modal_show()
    """
    modal_content = build_comparison_modal_content(human_scores, judge_data)

    return ui.modal(
        modal_content,
        title="Score Comparison",
        easy_close=False,  # Require explicit click to dismiss
        footer=None,  # Button is inside the content
    )


def get_keyboard_help_html() -> str:
    """Get the keyboard help modal HTML.

    Returns:
        HTML string for the keyboard help modal and backdrop
    """
    return """
<div id="keyboard-help-modal" style="display:none; position:fixed; top:50%; left:50%; transform:translate(-50%,-50%);
    background:white; border-radius:12px; padding:24px; box-shadow:0 20px 25px -5px rgba(0,0,0,0.1); z-index:1000; max-width:400px;">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
        <h3 style="margin:0; font-size:16px; font-weight:600;">Keyboard Shortcuts</h3>
        <button onclick="document.getElementById('keyboard-help-modal').style.display='none'; document.getElementById('keyboard-help-backdrop').style.display='none';"
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
<div id="keyboard-help-backdrop" onclick="document.getElementById('keyboard-help-modal').style.display='none'; document.getElementById('keyboard-help-backdrop').style.display='none';"
    style="display:none; position:fixed; top:0; left:0; right:0; bottom:0; background:rgba(0,0,0,0.3); z-index:999;"></div>
"""
