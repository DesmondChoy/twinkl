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


def _build_entry_preview(entry_data: dict | None) -> ui.TagChild:
    """Build the collapsible entry preview section.

    Args:
        entry_data: Current entry dict with text, nudge, response fields

    Returns:
        UI element for entry preview, or None if no entry data
    """
    if entry_data is None:
        return None

    # Build the full entry text (initial + nudge + response)
    # Field names match wrangled data format: initial_entry, nudge_text, response_text
    entry_text = entry_data.get("initial_entry", "")
    nudge = entry_data.get("nudge_text")
    response = entry_data.get("response_text")

    # Truncate for collapsed preview (show ~150 chars)
    preview_text = entry_text[:150] + "..." if len(entry_text) > 150 else entry_text

    # Build the full thread content
    full_content_parts = [
        ui.div(entry_text, class_="entry-preview-initial"),
    ]

    if nudge:
        full_content_parts.append(
            ui.div(
                ui.span("💬 Nudge: ", class_="entry-preview-label"),
                nudge,
                class_="entry-preview-nudge",
            )
        )

    if response:
        full_content_parts.append(
            ui.div(
                ui.span("↳ Response: ", class_="entry-preview-label"),
                response,
                class_="entry-preview-response",
            )
        )

    return ui.div(
        ui.div(
            ui.span("📄 Entry Preview", class_="entry-preview-title"),
            ui.span("▼", class_="entry-preview-toggle-icon"),
            class_="entry-preview-header",
            onclick="toggleEntryPreview()",
        ),
        ui.div(
            preview_text,
            class_="entry-preview-collapsed",
            id="entry-preview-collapsed",
        ),
        ui.div(
            *full_content_parts,
            class_="entry-preview-expanded",
            id="entry-preview-expanded",
            style="display: none;",
        ),
        class_="entry-preview-container",
    )


def _get_comparison_modal_script() -> str:
    """Return the inline JavaScript for modal interactivity."""
    return """
<script>
(function() {
    // Toggle entry preview
    window.toggleEntryPreview = function() {
        var collapsed = document.getElementById('entry-preview-collapsed');
        var expanded = document.getElementById('entry-preview-expanded');
        var icon = document.querySelector('.entry-preview-toggle-icon');

        if (collapsed.style.display === 'none') {
            collapsed.style.display = 'block';
            expanded.style.display = 'none';
            icon.textContent = '▼';
        } else {
            collapsed.style.display = 'none';
            expanded.style.display = 'block';
            icon.textContent = '▲';
        }
    };

    // Toggle rationale rows
    window.toggleRationale = function(valueKey) {
        var rationaleRow = document.getElementById('rationale-row-' + valueKey);
        if (rationaleRow) {
            if (rationaleRow.style.display === 'none' || rationaleRow.style.display === '') {
                rationaleRow.style.display = 'table-row';
            } else {
                rationaleRow.style.display = 'none';
            }
        }
    };
})();
</script>
"""


def build_comparison_modal_content(
    human_scores: dict[str, int],
    judge_data: dict | None,
    entry_data: dict | None = None,
) -> ui.TagChild:
    """Build the comparison modal content showing human vs judge scores.

    Args:
        human_scores: Dict mapping value names to human annotator scores
        judge_data: Full judge label row dict, or None if not available
        entry_data: Current entry dict for preview, or None

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

        # Check if this value has a rationale
        has_rationale = value in rationales
        rationale_indicator = ""
        row_extra_class = ""
        row_onclick = ""

        if has_rationale:
            rationale_indicator = "💬"
            row_extra_class = " has-rationale"
            row_onclick = f"toggleRationale('{value}')"

        # Build the main row
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
                ui.tags.td(
                    ui.span(rationale_indicator, class_="rationale-indicator"),
                    class_="comparison-rationale-col",
                ),
                class_=f"comparison-row {row_class}{row_extra_class}",
                onclick=row_onclick if has_rationale else None,
            )
        )

        # Add rationale row (hidden by default) if rationale exists
        if has_rationale:
            rows.append(
                ui.tags.tr(
                    ui.tags.td(
                        ui.div(
                            rationales[value],
                            class_="comparison-rationale-content",
                        ),
                        colspan="5",
                        class_="comparison-rationale-cell",
                    ),
                    class_="comparison-rationale-row",
                    id=f"rationale-row-{value}",
                    style="display: none;",
                )
            )

    # Build entry preview section
    entry_preview = _build_entry_preview(entry_data)

    content_parts = []

    # Add entry preview at the top if available
    if entry_preview:
        content_parts.append(entry_preview)

    # Add the comparison table
    content_parts.append(
        ui.tags.table(
            ui.tags.thead(
                ui.tags.tr(
                    ui.tags.th("Value"),
                    ui.tags.th("You"),
                    ui.tags.th("Judge"),
                    ui.tags.th("Match"),
                    ui.tags.th("", class_="comparison-rationale-header"),
                )
            ),
            ui.tags.tbody(*rows),
            class_="comparison-table",
        )
    )

    # Add summary
    content_parts.append(
        ui.div(
            ui.span(f"{exact_matches}/10", class_="comparison-summary-count"),
            " exact matches",
            class_="comparison-summary",
        )
    )

    # Add continue button
    content_parts.append(
        ui.input_action_button(
            "comparison_continue",
            "Continue →",
            class_="comparison-continue-btn",
        )
    )

    # Add inline script for interactivity
    content_parts.append(ui.HTML(_get_comparison_modal_script()))

    return ui.div(
        *content_parts,
        class_="comparison-modal-content",
    )


def build_comparison_modal(
    human_scores: dict[str, int],
    judge_data: dict | None,
    entry_data: dict | None = None,
) -> ui.Tag:
    """Build the full comparison modal.

    Args:
        human_scores: Dict mapping value names to human annotator scores
        judge_data: Full judge label row dict, or None if not available
        entry_data: Current entry dict for preview, or None

    Returns:
        UI modal element ready to be shown with ui.modal_show()
    """
    modal_content = build_comparison_modal_content(human_scores, judge_data, entry_data)

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
        <button onclick="closeHelp()"
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
<div id="keyboard-help-backdrop" onclick="closeHelp()"
    style="display:none; position:fixed; top:0; left:0; right:0; bottom:0; background:rgba(0,0,0,0.3); z-index:999;"></div>
"""
