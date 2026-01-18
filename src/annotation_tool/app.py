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


def _get_entry_count_for_persona(persona_id: str) -> int:
    """Get the number of entries for a specific persona."""
    return len(_entries_df.filter(_entries_df["persona_id"] == persona_id))


# CSS styles
app_css = """
body {
    background: #f0f2f5;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    margin: 0;
    padding: 0;
}

/* Header styles */
.header-component {
    background: #212529;
    color: white;
    padding: 16px 24px;
    margin-bottom: 24px;
}

.header-top-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
}

.app-title {
    margin: 0;
    font-size: 20px;
    font-weight: 600;
}

.annotator-input input {
    background: #343a40;
    border: 1px solid #495057;
    color: white;
    border-radius: 4px;
    padding: 6px 12px;
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
    height: 8px;
    background: #495057;
    border-radius: 4px;
    overflow: hidden;
    max-width: 300px;
}

.progress-fill {
    height: 100%;
    background: #0d6efd;
    border-radius: 4px;
    transition: width 0.3s ease;
}

.progress-text {
    font-size: 13px;
    color: #adb5bd;
    white-space: nowrap;
}

.nav-buttons {
    display: flex;
    gap: 8px;
}

/* Main container */
.main-container {
    max-width: 800px;
    margin: 0 auto;
    padding: 0 16px 32px 16px;
}

/* Entry display styles */
.entry-display {
    background: white;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 16px;
}

.persona-header-section {
    background: #e9ecef;
    padding: 16px;
    border-bottom: 1px solid #dee2e6;
}

.persona-name {
    margin: 0 0 8px 0;
    color: #212529;
    font-size: 16px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.persona-info {
    color: #6c757d;
    font-size: 14px;
    margin-bottom: 4px;
}

.persona-values {
    color: #495057;
    font-size: 14px;
    font-weight: 500;
}

.persona-bio-text {
    margin-top: 8px;
    padding: 12px;
    background: white;
    border-radius: 4px;
    font-size: 13px;
    color: #495057;
    line-height: 1.5;
}

.entry-content-section {
    padding: 16px;
}

.entry-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}

.entry-number {
    font-weight: 600;
    color: #212529;
}

.entry-date {
    color: #6c757d;
    font-size: 14px;
}

.entry-text {
    font-size: 15px;
    line-height: 1.6;
    color: #212529;
    white-space: pre-wrap;
}

.entry-nudge {
    margin-top: 16px;
    padding: 12px;
    background: #fff3cd;
    border-left: 3px solid #ffc107;
    border-radius: 0 4px 4px 0;
    font-size: 14px;
    color: #664d03;
}

.entry-response {
    margin-top: 12px;
    padding: 12px;
    background: #d1e7dd;
    border-left: 3px solid #198754;
    border-radius: 0 4px 4px 0;
    font-size: 14px;
    color: #0f5132;
}

/* Scoring grid styles */
.scoring-grid {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 16px;
}

.scoring-header {
    display: grid;
    grid-template-columns: 1fr 120px;
    align-items: center;
    padding-bottom: 8px;
    border-bottom: 2px solid #dee2e6;
    margin-bottom: 8px;
    font-weight: bold;
    color: #495057;
}

.scoring-row {
    display: grid;
    grid-template-columns: 1fr 120px;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid #e9ecef;
}

.value-label {
    font-size: 14px;
    color: #212529;
}

.score-buttons {
    display: flex;
    gap: 8px;
    justify-content: center;
}

.score-buttons .form-check-inline {
    margin: 0;
}

.notes-section {
    margin-top: 16px;
    padding-top: 16px;
    border-top: 2px solid #dee2e6;
}

.save-section {
    margin-top: 16px;
    display: flex;
    justify-content: flex-end;
}

/* Status indicators */
.annotation-status {
    padding: 8px 12px;
    border-radius: 4px;
    font-size: 13px;
    margin-bottom: 8px;
}

.annotation-status.already-annotated {
    background: #d1e7dd;
    color: #0f5132;
}

.annotation-status.not-annotated {
    background: #f8d7da;
    color: #842029;
}

.no-annotator-warning {
    background: #fff3cd;
    color: #664d03;
    padding: 12px;
    border-radius: 8px;
    text-align: center;
    margin-bottom: 16px;
}
"""


def create_scoring_row(value: str) -> ui.TagChild:
    """Create a scoring row for a single Schwartz value."""
    label = VALUE_LABELS[value]
    return ui.div(
        ui.div(label, class_="value-label"),
        ui.div(
            ui.input_radio_buttons(
                id=f"score_{value}",
                label=None,
                choices={"-1": "−", "0": "○", "1": "+"},
                selected="0",
                inline=True,
            ),
            class_="score-buttons",
        ),
        class_="scoring-row",
    )


app_ui = ui.page_fluid(
    ui.tags.style(app_css),
    # Header
    ui.div(
        ui.div(
            ui.h2("Schwartz Value Annotation Tool", class_="app-title"),
            ui.div(
                ui.input_text(
                    id="annotator_name",
                    label=None,
                    placeholder="Enter your name...",
                    width="200px",
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
                ui.input_action_button("prev_btn", "◀ Prev", class_="btn-secondary"),
                ui.input_action_button("next_btn", "Next ▶", class_="btn-secondary"),
                class_="nav-buttons",
            ),
            class_="header-nav-row",
        ),
        class_="header-component",
    ),
    # Main content area
    ui.div(
        ui.output_ui("annotator_warning"),
        ui.output_ui("annotation_status"),
        # Entry display
        ui.div(
            ui.div(
                ui.output_ui("persona_header"),
                class_="persona-header-section",
            ),
            ui.div(
                ui.output_ui("entry_content"),
                class_="entry-content-section",
            ),
            class_="entry-display",
        ),
        # Scoring grid
        ui.div(
            ui.div(
                ui.div("Value", style="font-weight: bold;"),
                ui.div("−  ○  +", style="text-align: center; font-weight: bold;"),
                class_="scoring-header",
            ),
            *[create_scoring_row(v) for v in SCHWARTZ_VALUE_ORDER],
            ui.div(
                ui.input_text_area(
                    id="notes",
                    label="Notes (optional)",
                    placeholder="Any observations about this entry...",
                    rows=2,
                ),
                class_="notes-section",
            ),
            ui.div(
                ui.input_radio_buttons(
                    id="confidence",
                    label="Confidence (optional)",
                    choices={"": "—", "1": "1", "2": "2", "3": "3", "4": "4", "5": "5"},
                    selected="",
                    inline=True,
                ),
            ),
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
        class_="main-container",
    ),
)


def server(input, output, session):
    # Core reactive state
    current_index = reactive.value(0)
    annotated_count = reactive.value(0)
    bio_visible = reactive.value(False)

    # Pending save data (for modal confirmation)
    pending_save = reactive.value(None)

    # Current entry
    @reactive.calc
    def current_entry():
        idx = current_index()
        if 0 <= idx < _total_entries:
            return _all_entries[idx]
        return None

    # Entry count for current persona
    @reactive.calc
    def entry_count_for_persona():
        entry = current_entry()
        if entry:
            return _get_entry_count_for_persona(entry["persona_id"])
        return 0

    # Progress display
    @output
    @render.ui
    def progress_display():
        count = annotated_count()
        percentage = (count / _total_entries * 100) if _total_entries > 0 else 0
        idx = current_index()

        return ui.div(
            ui.div(f"Entry {idx + 1} of {_total_entries}", style="margin-bottom: 8px; color: #adb5bd;"),
            ui.div(
                ui.div(
                    ui.div(style=f"width: {percentage:.0f}%", class_="progress-fill"),
                    class_="progress-bar-container",
                ),
                ui.span(f"{count}/{_total_entries} annotated ({percentage:.0f}%)", class_="progress-text"),
                class_="progress-wrapper",
            ),
        )

    # Persona header
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

        # Bio section
        bio = entry.get("persona_bio", "")
        if bio_visible():
            bio_section = ui.div(
                ui.input_action_button("toggle_bio", "▲ Hide Bio", class_="btn-link btn-sm"),
                ui.div(bio, class_="persona-bio-text"),
            )
        else:
            bio_section = ui.input_action_button("toggle_bio", "▼ Show Bio", class_="btn-link btn-sm")

        return ui.div(
            ui.h4(f"PERSONA: {entry.get('persona_name', 'Unknown')}", class_="persona-name"),
            ui.div(info_line, class_="persona-info"),
            ui.div(f"Core Values: {values_str}", class_="persona-values"),
            ui.div(bio_section, style="margin-top: 8px;"),
        )

    # Entry content
    @output
    @render.ui
    def entry_content():
        entry = current_entry()
        if entry is None:
            return ui.div("No entry selected")

        t_index = entry.get("t_index", 0)
        total = entry_count_for_persona()
        date = entry.get("date", "")

        content_parts = [
            ui.div(
                ui.span(f"Entry {t_index + 1} of {total}", class_="entry-number"),
                ui.span(date, class_="entry-date"),
                class_="entry-header",
            ),
            ui.div(entry.get("initial_entry", ""), class_="entry-text"),
        ]

        # Nudge (if present)
        if entry.get("has_nudge") and entry.get("nudge_text"):
            content_parts.append(
                ui.div(
                    ui.strong("Nudge: "),
                    f'"{entry["nudge_text"]}"',
                    class_="entry-nudge",
                )
            )

        # Response (if present)
        if entry.get("has_response") and entry.get("response_text"):
            content_parts.append(
                ui.div(
                    ui.strong("Response: "),
                    entry["response_text"],
                    class_="entry-response",
                )
            )

        return ui.div(*content_parts)

    # Toggle bio visibility
    @reactive.effect
    @reactive.event(input.toggle_bio)
    def _toggle_bio():
        bio_visible.set(not bio_visible())

    # Reset bio when changing entries
    @reactive.effect
    @reactive.event(current_index)
    def _reset_bio():
        bio_visible.set(False)

    # Navigation
    @reactive.effect
    @reactive.event(input.prev_btn)
    def _on_prev():
        new_idx = current_index() - 1
        if new_idx >= 0:
            current_index.set(new_idx)
            _load_existing_annotation()

    @reactive.effect
    @reactive.event(input.next_btn)
    def _on_next():
        new_idx = current_index() + 1
        if new_idx < _total_entries:
            current_index.set(new_idx)
            _load_existing_annotation()

    # Load existing annotation for current entry
    def _load_existing_annotation():
        name = input.annotator_name()
        entry = current_entry()

        if not name or not entry:
            # Reset to neutral
            for value in SCHWARTZ_VALUE_ORDER:
                ui.update_radio_buttons(f"score_{value}", selected="0")
            ui.update_text_area("notes", value="")
            ui.update_radio_buttons("confidence", selected="")
            return

        existing = get_annotation(name, entry["persona_id"], entry["t_index"])
        if existing:
            # Load existing scores
            for value in SCHWARTZ_VALUE_ORDER:
                score_key = f"alignment_{value}"
                if score_key in existing:
                    ui.update_radio_buttons(f"score_{value}", selected=str(existing[score_key]))
            # Load notes and confidence
            if existing.get("notes"):
                ui.update_text_area("notes", value=existing["notes"])
            else:
                ui.update_text_area("notes", value="")
            if existing.get("confidence"):
                ui.update_radio_buttons("confidence", selected=str(existing["confidence"]))
            else:
                ui.update_radio_buttons("confidence", selected="")
        else:
            # Reset to neutral
            for value in SCHWARTZ_VALUE_ORDER:
                ui.update_radio_buttons(f"score_{value}", selected="0")
            ui.update_text_area("notes", value="")
            ui.update_radio_buttons("confidence", selected="")

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

        # Check if all neutral
        all_neutral = all(s == 0 for s in scores.values())
        if all_neutral:
            # Store pending save and show modal
            pending_save.set({
                "scores": scores,
                "notes": notes,
                "confidence": confidence,
            })
            m = ui.modal(
                ui.div(
                    ui.h4("⚠️ All Neutral Scores"),
                    ui.p(
                        "You've set all 10 values to neutral (○). This is valid if the entry "
                        "truly doesn't relate to any values, but please double-check."
                    ),
                    style="background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px; padding: 16px;",
                ),
                ui.div(
                    ui.input_action_button("modal_cancel", "Go Back", class_="btn-secondary"),
                    ui.input_action_button("modal_confirm", "Save Anyway", class_="btn-primary"),
                    style="display: flex; gap: 8px; justify-content: flex-end; margin-top: 16px;",
                ),
                title="Confirm Save",
                easy_close=True,
            )
            ui.modal_show(m)
        else:
            # Save directly
            _do_save(scores, notes, confidence)

    def _do_save(scores: dict, notes: str | None, confidence: int | None):
        """Actually save the annotation and advance."""
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

        ui.notification_show("Annotation saved!", type="message", duration=2)

        # Update annotated count
        annotated_count.set(get_annotation_count(name))

        # Auto-advance to next entry
        new_idx = current_index() + 1
        if new_idx < _total_entries:
            current_index.set(new_idx)
            _load_existing_annotation()

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


if __name__ == "__main__":
    from shiny import run_app
    run_app(app)
