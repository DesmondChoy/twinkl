"""Scoring grid component for rating entries across 10 Schwartz values.

Displays a vertical table with radio buttons for each value dimension,
plus notes and confidence inputs, and a Save & Next button.

Layout:
    | Value           |  −  |  ○  |  +  |
    |-----------------|-----|-----|-----|
    | Self-Direction  |  ○  |  ●  |  ○  |
    | Stimulation     |  ○  |  ●  |  ○  |
    | ...             |     |     |     |
    | Universalism    |  ○  |  ●  |  ○  |

    Notes: [________________________]
    Confidence: [1] [2] [3] [4] [5]
    [Save & Next →]
"""

from shiny import module, reactive, ui

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


@module.ui
def scoring_grid_ui():
    """Generate the scoring grid UI component."""
    # Build radio buttons for each value
    value_rows = []
    for value in SCHWARTZ_VALUE_ORDER:
        label = VALUE_LABELS[value]
        value_rows.append(
            ui.div(
                ui.div(label, class_="value-label"),
                ui.input_radio_buttons(
                    id=f"score_{value}",
                    label=None,
                    choices={"-1": "−", "0": "○", "1": "+"},
                    selected="0",
                    inline=True,
                ),
                class_="scoring-row",
            )
        )

    return ui.div(
        # Scoring header
        ui.div(
            ui.div("Value", class_="header-label"),
            ui.div(
                ui.span("−", title="Misaligned"),
                ui.span("○", title="Neutral"),
                ui.span("+", title="Aligned"),
                class_="header-scores",
            ),
            class_="scoring-header",
        ),
        # Value rows
        *value_rows,
        # Notes field
        ui.div(
            ui.input_text_area(
                id="notes",
                label="Notes (optional)",
                placeholder="Any observations about this entry...",
                rows=2,
            ),
            class_="notes-section",
        ),
        # Confidence rating
        ui.div(
            ui.input_radio_buttons(
                id="confidence",
                label="Confidence (optional)",
                choices={"": "—", "1": "1", "2": "2", "3": "3", "4": "4", "5": "5"},
                selected="",
                inline=True,
            ),
            class_="confidence-section",
        ),
        # Save button
        ui.div(
            ui.input_action_button(
                id="save_btn",
                label="Save & Next →",
                class_="btn-primary save-button",
            ),
            class_="save-section",
        ),
        class_="scoring-grid",
    )


@module.server
def scoring_grid_server(
    input,
    output,
    session,
    on_save: callable,
    load_scores: reactive.Value,
):
    """Server logic for the scoring grid.

    Args:
        input, output, session: Shiny module parameters
        on_save: Callback function to call with scores when save is clicked.
                 Receives (scores: dict, notes: str|None, confidence: int|None)
        load_scores: Reactive value that triggers score loading.
                     When changed, should contain dict with scores to load,
                     or None to reset to neutral.
    """

    # Track when scores are loaded externally vs changed by user
    @reactive.effect
    @reactive.event(load_scores)
    def _load_scores():
        """Load scores from external source (e.g., existing annotation)."""
        data = load_scores()
        if data is None:
            # Reset to neutral
            for value in SCHWARTZ_VALUE_ORDER:
                ui.update_radio_buttons(f"score_{value}", selected="0")
            ui.update_text_area("notes", value="")
            ui.update_radio_buttons("confidence", selected="")
        else:
            # Load existing scores
            for value in SCHWARTZ_VALUE_ORDER:
                score_key = f"alignment_{value}"
                if score_key in data:
                    ui.update_radio_buttons(
                        f"score_{value}",
                        selected=str(data[score_key])
                    )
            # Load notes and confidence
            if "notes" in data and data["notes"]:
                ui.update_text_area("notes", value=data["notes"])
            else:
                ui.update_text_area("notes", value="")

            if "confidence" in data and data["confidence"]:
                ui.update_radio_buttons("confidence", selected=str(data["confidence"]))
            else:
                ui.update_radio_buttons("confidence", selected="")

    @reactive.effect
    @reactive.event(input.save_btn)
    def _on_save():
        """Handle save button click."""
        # Collect scores
        scores = {}
        for value in SCHWARTZ_VALUE_ORDER:
            score_str = input[f"score_{value}"]()
            scores[value] = int(score_str)

        # Get notes (empty string becomes None)
        notes = input.notes()
        if notes == "":
            notes = None

        # Get confidence (empty string becomes None)
        confidence_str = input.confidence()
        confidence = int(confidence_str) if confidence_str else None

        # Call the callback
        on_save(scores, notes, confidence)


def get_scoring_grid_css() -> str:
    """Return CSS styles for the scoring grid component."""
    return """
    .scoring-grid {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        margin-top: 16px;
    }

    .scoring-header {
        display: grid;
        grid-template-columns: 1fr auto;
        align-items: center;
        padding-bottom: 8px;
        border-bottom: 2px solid #dee2e6;
        margin-bottom: 8px;
    }

    .header-label {
        font-weight: bold;
        color: #495057;
    }

    .header-scores {
        display: flex;
        gap: 24px;
        padding-right: 8px;
    }

    .header-scores span {
        font-size: 14px;
        color: #6c757d;
        width: 32px;
        text-align: center;
    }

    .scoring-row {
        display: grid;
        grid-template-columns: 1fr auto;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid #e9ecef;
    }

    .scoring-row:last-of-type {
        border-bottom: none;
    }

    .value-label {
        font-size: 14px;
        color: #212529;
    }

    .scoring-row .shiny-input-radiogroup {
        display: flex;
        gap: 8px;
    }

    .scoring-row .shiny-input-radiogroup .radio-inline {
        margin: 0;
        padding: 0;
    }

    .scoring-row .shiny-input-radiogroup input[type="radio"] {
        margin: 0 4px;
    }

    .notes-section {
        margin-top: 16px;
        padding-top: 16px;
        border-top: 2px solid #dee2e6;
    }

    .notes-section textarea {
        width: 100%;
        resize: vertical;
    }

    .confidence-section {
        margin-top: 12px;
    }

    .confidence-section .shiny-input-radiogroup {
        display: flex;
        gap: 8px;
        align-items: center;
    }

    .save-section {
        margin-top: 16px;
        display: flex;
        justify-content: flex-end;
    }

    .save-button {
        padding: 10px 24px;
        font-size: 16px;
    }
    """
