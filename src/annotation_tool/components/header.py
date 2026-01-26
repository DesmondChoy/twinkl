"""Header component for the annotation tool.

This module provides the app header containing:
- App title and help button
- Annotator name input
- Progress bar
- Persona navigation buttons

Usage:
    from components import header

    # In UI definition
    header.header_ui("header")

    # In server function
    header.header_server(
        "header",
        state=state,
        total_entries=total_entries,
        total_personas=total_personas,
        current_persona_entries=current_persona_entries,
        on_prev=handle_prev,
        on_next=handle_next,
    )
"""

from typing import Callable, Union

from shiny import module, reactive, render, ui

from src.annotation_tool.state import AppState


@module.ui
def header_ui():
    """Generate the header UI component.

    Returns:
        UI div containing the complete header
    """
    return ui.div(
        # Top row: Title, help button, and annotator input
        ui.div(
            ui.div(
                ui.h2("Schwartz Value Annotation Tool", class_="app-title"),
                ui.input_action_button("help_btn", "? Help", class_="help-btn"),
                ui.input_action_button(
                    "theme_toggle",
                    ui.HTML('<span class="theme-icon">🌙</span>'),
                    class_="theme-toggle",
                    title="Toggle dark mode",
                ),
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
        # Progress and navigation row
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
    )


@module.server
def header_server(
    input,
    output,
    session,
    state: AppState,
    total_entries: Union[int, Callable[[], int]],
    total_personas: Union[int, Callable[[], int]],
    current_persona_entries: reactive.calc,
    on_prev: Callable,
    on_next: Callable,
    on_unsaved_cancel: Callable = None,
    on_unsaved_discard: Callable = None,
    on_unsaved_save: Callable = None,
):
    """Server logic for the header component.

    Args:
        input, output, session: Shiny module parameters
        state: Centralized app state
        total_entries: Total number of entries (int or callable returning int)
        total_personas: Total number of personas (int or callable returning int)
        current_persona_entries: Reactive calc returning entries for current persona
        on_prev: Callback for previous persona button click
        on_next: Callback for next persona button click
        on_unsaved_cancel: Callback for unsaved changes modal - keep editing
        on_unsaved_discard: Callback for unsaved changes modal - discard changes
        on_unsaved_save: Callback for unsaved changes modal - save & continue

    Returns:
        Reactive calc for the annotator name
    """

    def _get_total_entries() -> int:
        """Get total entries, handling both static int and callable."""
        return total_entries() if callable(total_entries) else total_entries

    def _get_total_personas() -> int:
        """Get total personas, handling both static int and callable."""
        return total_personas() if callable(total_personas) else total_personas

    @render.ui
    def progress_display():
        """Render the progress bar and entry/persona indicators."""
        count = state.annotated_count()
        total = _get_total_entries()
        percentage = (count / total * 100) if total > 0 else 0
        persona_idx = state.persona_index()
        entries = current_persona_entries()
        entry_idx = state.entry_index()

        return ui.div(
            ui.div(
                f"Persona {persona_idx + 1} of {_get_total_personas()} • "
                f"Entry {entry_idx + 1} of {len(entries)}",
                class_="entry-indicator",
            ),
            ui.div(
                ui.div(
                    ui.div(style=f"width: {percentage:.0f}%", class_="progress-fill"),
                    class_="progress-bar-container",
                ),
                ui.span(
                    f"{count}/{total} annotated ({percentage:.0f}%)",
                    class_="progress-text",
                ),
                class_="progress-wrapper",
            ),
        )

    @reactive.effect
    @reactive.event(input.prev_btn)
    def _on_prev():
        on_prev()

    @reactive.effect
    @reactive.event(input.next_btn)
    def _on_next():
        on_next()

    # Unsaved changes modal button handlers
    # (modal is shown from on_prev/on_next, so buttons are namespaced here)
    @reactive.effect
    @reactive.event(input.unsaved_cancel)
    def _handle_unsaved_cancel():
        if on_unsaved_cancel:
            on_unsaved_cancel()

    @reactive.effect
    @reactive.event(input.unsaved_discard)
    def _handle_unsaved_discard():
        if on_unsaved_discard:
            on_unsaved_discard()

    @reactive.effect
    @reactive.event(input.unsaved_save)
    def _handle_unsaved_save():
        if on_unsaved_save:
            on_unsaved_save()

    # Return the annotator name as a reactive for the parent to use
    @reactive.calc
    def annotator_name():
        return input.annotator_name()

    return annotator_name
