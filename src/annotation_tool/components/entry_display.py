"""Entry display component for showing persona context and journal entry.

Displays persona header with collapsible bio, and entry content with
optional nudge/response threading.

Layout:
    ┌─────────────────────────────────────────┐
    │ PERSONA: Emma Lindqvist                 │
    │ 25-34 • Teacher • Western European      │
    │ Core Values: Security                   │
    │ [▼ Show Bio]                            │
    ├─────────────────────────────────────────┤
    │ Entry 2 of 3              2025-12-20    │
    │ ─────────────────────────────────────── │
    │ Mikkel brought up the idea...           │
    │                                         │
    │   > **Nudge:** "What would you need..." │
    │                                         │
    │   > **Response:** A guarantee...        │
    └─────────────────────────────────────────┘
"""

from shiny import module, reactive, ui


@module.ui
def entry_display_ui():
    """Generate the entry display UI component."""
    return ui.div(
        # Persona header section
        ui.div(
            ui.output_ui("persona_header"),
            class_="persona-header-section",
        ),
        # Entry content section
        ui.div(
            ui.output_ui("entry_content"),
            class_="entry-content-section",
        ),
        class_="entry-display",
    )


@module.server
def entry_display_server(
    input,
    output,
    session,
    current_entry: reactive.Value,
    entry_count_for_persona: reactive.Value,
):
    """Server logic for the entry display.

    Args:
        input, output, session: Shiny module parameters
        current_entry: Reactive value containing the current entry dict.
                      Expected keys: persona_name, persona_age, persona_profession,
                      persona_culture, persona_core_values, persona_bio, t_index,
                      date, initial_entry, nudge_text, response_text, has_nudge,
                      has_response
        entry_count_for_persona: Reactive value with total entries for current persona
    """
    # Track bio visibility
    bio_visible = reactive.value(False)

    @reactive.effect
    @reactive.event(input.toggle_bio)
    def _toggle_bio():
        bio_visible.set(not bio_visible())

    @reactive.effect
    @reactive.event(current_entry)
    def _reset_bio_on_persona_change():
        """Collapse bio when navigating to a different persona."""
        bio_visible.set(False)

    @output
    @ui.render_ui
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

        # Bio section (collapsible)
        bio = entry.get("persona_bio", "")
        if bio_visible():
            bio_section = ui.div(
                ui.input_action_button(
                    "toggle_bio",
                    "▲ Hide Bio",
                    class_="btn-link btn-sm",
                ),
                ui.div(bio, class_="persona-bio-text"),
                class_="persona-bio-section expanded",
            )
        else:
            bio_section = ui.div(
                ui.input_action_button(
                    "toggle_bio",
                    "▼ Show Bio",
                    class_="btn-link btn-sm",
                ),
                class_="persona-bio-section collapsed",
            )

        return ui.div(
            ui.h4(
                f"PERSONA: {entry.get('persona_name', 'Unknown')}",
                class_="persona-name",
            ),
            ui.div(info_line, class_="persona-info"),
            ui.div(f"Core Values: {values_str}", class_="persona-values"),
            bio_section,
        )

    @output
    @ui.render_ui
    def entry_content():
        entry = current_entry()
        if entry is None:
            return ui.div("No entry selected")

        t_index = entry.get("t_index", 0)
        total = entry_count_for_persona()
        date = entry.get("date", "")

        # Entry header
        header = ui.div(
            ui.span(f"Entry {t_index + 1} of {total}", class_="entry-number"),
            ui.span(date, class_="entry-date"),
            class_="entry-header",
        )

        # Main entry content
        initial_entry = entry.get("initial_entry", "")
        content_parts = [
            ui.div(initial_entry, class_="entry-text"),
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

        return ui.div(
            header,
            ui.hr(class_="entry-divider"),
            *content_parts,
        )


def get_entry_display_css() -> str:
    """Return CSS styles for the entry display component."""
    return """
    .entry-display {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        overflow: hidden;
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

    .persona-bio-section {
        margin-top: 8px;
    }

    .persona-bio-section .btn-link {
        padding: 0;
        font-size: 12px;
        color: #0d6efd;
        text-decoration: none;
        border: none;
        background: none;
        cursor: pointer;
    }

    .persona-bio-section .btn-link:hover {
        text-decoration: underline;
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
    }

    .entry-number {
        font-weight: 600;
        color: #212529;
    }

    .entry-date {
        color: #6c757d;
        font-size: 14px;
    }

    .entry-divider {
        margin: 12px 0;
        border: none;
        border-top: 1px solid #dee2e6;
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
    """
