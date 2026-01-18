"""Entry display component for the center column.

This module provides the center column entry display containing:
- Annotation status indicator
- Entry content with header, text, nudge, and response

Usage:
    from components import entry_display

    # In UI definition
    entry_display.entry_display_ui("entry")

    # In server function
    entry_display.entry_display_server(
        "entry",
        current_entry=current_entry,
        current_persona_entries=current_persona_entries,
        get_annotation=get_annotation,
        annotator_name=annotator_name,
    )
"""

from shiny import module, reactive, render, ui


@module.ui
def entry_display_ui():
    """Generate the entry display UI component.

    Returns:
        UI div containing annotation status and entry content placeholders
    """
    return ui.div(
        ui.output_ui("annotation_status"),
        ui.div(
            ui.output_ui("entry_content"),
            class_="center-entry-display",
        ),
        class_="center-column",
    )


@module.server
def entry_display_server(
    input,
    output,
    session,
    current_entry: reactive.calc,
    current_persona_entries: reactive.calc,
    get_annotation: callable,
    annotator_name: reactive.calc,
):
    """Server logic for the entry display.

    Args:
        input, output, session: Shiny module parameters
        current_entry: Reactive calc returning the current entry dict
        current_persona_entries: Reactive calc returning list of entries for current persona
        get_annotation: Function to check if annotation exists (annotator, persona_id, t_index) -> dict|None
        annotator_name: Reactive calc returning the current annotator name
    """

    @render.ui
    def annotation_status():
        """Render the annotation status indicator."""
        name = annotator_name()
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

    @render.ui
    def entry_content():
        """Render the journal entry content."""
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
