"""Scoring grid component for the right column.

This module provides the scoring grid containing:
- Value group headers with colored indicators
- Counter button rows for each Schwartz value (-/0/+)
- Notes text area
- Save & Next button

Usage:
    from components import scoring_grid

    # In UI definition
    scoring_grid.scoring_grid_ui("scoring")

    # In server function
    scoring_grid.scoring_grid_server(
        "scoring",
        state=state,
        on_save=handle_save,
    )
"""

from shiny import module, reactive, render, ui

from src.annotation_tool.components.constants import VALUE_LABELS
from src.annotation_tool.state import AppState
from src.models.judge import SCHWARTZ_VALUE_ORDER

# One-liner tooltips for annotator reference (from Schwartz Value Quick Reference)
VALUE_TOOLTIPS = {
    "self_direction": "Making own choices, resisting control, autonomy",
    "stimulation": "Seeking novelty, avoiding routine, excitement",
    "hedonism": "Prioritizing pleasure, enjoyment, comfort",
    "achievement": "Goals, performance, recognition, hard work",
    "power": "Control, status, influence, being in charge",
    "security": "Stability, safety, avoiding risk",
    "conformity": "Following rules, meeting expectations, fitting in",
    "tradition": "Honoring customs, family obligations, heritage",
    "benevolence": "Helping close others (family, friends, team)",
    "universalism": "Broader social concern, fairness, environment",
}

# Schwartz value groupings for organized display
VALUE_GROUPS = {
    "OPENNESS TO CHANGE": ["self_direction", "stimulation", "hedonism"],
    "SELF-ENHANCEMENT": ["achievement", "power"],
    "CONSERVATION": ["security", "conformity", "tradition"],
    "SELF-TRANSCENDENCE": ["benevolence", "universalism"],
}

# Map group names to CSS classes
GROUP_CSS_CLASSES = {
    "OPENNESS TO CHANGE": "openness",
    "SELF-ENHANCEMENT": "enhancement",
    "CONSERVATION": "conservation",
    "SELF-TRANSCENDENCE": "transcendence",
}


def _create_scoring_row(value: str, row_index: int) -> ui.TagChild:
    """Create a scoring row for a single Schwartz value with counter buttons.

    Args:
        value: The Schwartz value key (e.g., 'self_direction')
        row_index: The row index for keyboard navigation (0-9)

    Returns:
        UI div for the scoring row
    """
    label = VALUE_LABELS[value]
    tooltip = VALUE_TOOLTIPS[value]
    return ui.div(
        ui.div(label, class_="value-label", **{"data-tooltip": tooltip}),
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


def _create_value_group(group_name: str, values: list, start_index: int) -> ui.TagChild:
    """Create a value group with header and rows.

    Args:
        group_name: Display name for the group (e.g., 'OPENNESS TO CHANGE')
        values: List of value keys in this group
        start_index: Starting row index for keyboard navigation

    Returns:
        UI div for the value group
    """
    css_class = GROUP_CSS_CLASSES.get(group_name, "")

    rows = []
    for i, value in enumerate(values):
        rows.append(_create_scoring_row(value, start_index + i))

    return ui.div(
        ui.div(group_name, class_=f"value-group-header {css_class}"),
        *rows,
        class_="value-group",
    )


def _create_grouped_scoring_grid() -> list:
    """Create all value groups for the scoring grid.

    Returns:
        List of UI divs for all value groups
    """
    groups = []
    current_index = 0

    for group_name, values in VALUE_GROUPS.items():
        groups.append(_create_value_group(group_name, values, current_index))
        current_index += len(values)

    return groups


@module.ui
def scoring_grid_ui():
    """Generate the scoring grid UI component.

    Returns:
        UI div containing the complete scoring grid
    """
    return ui.div(
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
            *_create_grouped_scoring_grid(),
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
    )


@module.server
def scoring_grid_server(
    input,
    output,
    session,
    state: AppState,
    on_save: callable,
    on_modal_cancel: callable = None,
    on_modal_confirm: callable = None,
    on_comparison_continue: callable = None,
):
    """Server logic for the scoring grid.

    Args:
        input, output, session: Shiny module parameters
        state: Centralized app state with scores
        on_save: Callback function for save button. Receives (scores: dict, notes: str|None)
        on_modal_cancel: Callback for all-neutral modal cancel
        on_modal_confirm: Callback for all-neutral modal confirm
        on_comparison_continue: Callback for comparison view continue
    """

    # Create score display renderers for each Schwartz value
    # We need to create these explicitly to match the output_ui IDs
    def _render_score(value: str):
        score = state.scores[value]()
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

    @render.ui
    def score_display_self_direction():
        return _render_score("self_direction")

    @render.ui
    def score_display_stimulation():
        return _render_score("stimulation")

    @render.ui
    def score_display_hedonism():
        return _render_score("hedonism")

    @render.ui
    def score_display_achievement():
        return _render_score("achievement")

    @render.ui
    def score_display_power():
        return _render_score("power")

    @render.ui
    def score_display_security():
        return _render_score("security")

    @render.ui
    def score_display_conformity():
        return _render_score("conformity")

    @render.ui
    def score_display_tradition():
        return _render_score("tradition")

    @render.ui
    def score_display_benevolence():
        return _render_score("benevolence")

    @render.ui
    def score_display_universalism():
        return _render_score("universalism")

    # Create decrement button handlers for each Schwartz value
    def make_dec_handler(value: str):
        @reactive.effect
        @reactive.event(input[f"dec_{value}"])
        def _():
            current = state.scores[value]()
            if current > -1:
                state.scores[value].set(current - 1)
        return _

    # Create increment button handlers for each Schwartz value
    def make_inc_handler(value: str):
        @reactive.effect
        @reactive.event(input[f"inc_{value}"])
        def _():
            current = state.scores[value]()
            if current < 1:
                state.scores[value].set(current + 1)
        return _

    # Register button handlers for all values
    for value in SCHWARTZ_VALUE_ORDER:
        make_dec_handler(value)
        make_inc_handler(value)

    # Save button handler
    @reactive.effect
    @reactive.event(input.save_btn)
    def _on_save():
        # Collect scores from state
        scores = state.get_scores_dict()

        # Get notes (empty string becomes None)
        notes = input.notes()
        if notes == "":
            notes = None

        # Call the callback
        on_save(scores, notes)

    def get_notes():
        """Get the current notes value."""
        return input.notes()

    def set_notes(value: str):
        """Set the notes text area value."""
        ui.update_text_area("notes", value=value)

    # Modal button handlers (buttons in modals inherit this module's namespace)
    @reactive.effect
    @reactive.event(input.modal_cancel)
    def _handle_modal_cancel():
        if on_modal_cancel:
            on_modal_cancel()

    @reactive.effect
    @reactive.event(input.modal_confirm)
    def _handle_modal_confirm():
        if on_modal_confirm:
            on_modal_confirm()

    @reactive.effect
    @reactive.event(input.comparison_continue)
    def _handle_comparison_continue():
        if on_comparison_continue:
            on_comparison_continue()

    # Return accessor functions for notes
    return {"get_notes": get_notes, "set_notes": set_notes}
