"""Header component for annotator input, progress bar, and navigation.

Displays the annotator name input, progress indicator, and prev/next buttons.

Elements:
    - input_text("annotator_name") - Free-form annotator name
    - Progress bar: 47/100 entries (47%)
    - [◀ Prev] and [Next ▶] buttons
"""

from shiny import module, reactive, ui


@module.ui
def header_ui():
    """Generate the header UI component."""
    return ui.div(
        # Top row: Title and annotator input
        ui.div(
            ui.h2("Schwartz Value Annotation Tool", class_="app-title"),
            ui.div(
                ui.input_text(
                    id="annotator_name",
                    label="Annotator Name",
                    placeholder="Enter your name...",
                    width="200px",
                ),
                class_="annotator-input",
            ),
            class_="header-top-row",
        ),
        # Progress and navigation row
        ui.div(
            # Progress section
            ui.div(
                ui.output_ui("progress_display"),
                class_="progress-section",
            ),
            # Navigation buttons
            ui.div(
                ui.input_action_button(
                    id="prev_btn",
                    label="◀ Prev",
                    class_="btn-secondary nav-button",
                ),
                ui.input_action_button(
                    id="next_btn",
                    label="Next ▶",
                    class_="btn-secondary nav-button",
                ),
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
    total_entries: int,
    current_index: reactive.Value,
    annotated_count: reactive.Value,
    on_prev: callable,
    on_next: callable,
):
    """Server logic for the header component.

    Args:
        input, output, session: Shiny module parameters
        total_entries: Total number of entries to annotate
        current_index: Reactive value with current entry index (0-based)
        annotated_count: Reactive value with count of annotated entries
        on_prev: Callback for previous button click
        on_next: Callback for next button click
    """

    @output
    @ui.render_ui
    def progress_display():
        count = annotated_count()
        total = total_entries
        percentage = (count / total * 100) if total > 0 else 0
        idx = current_index()

        return ui.div(
            ui.div(
                f"Entry {idx + 1} of {total}",
                class_="current-position",
            ),
            ui.div(
                ui.div(
                    ui.div(
                        style=f"width: {percentage:.0f}%",
                        class_="progress-fill",
                    ),
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

    # Return the annotator name as a reactive for the parent to use
    @reactive.calc
    def annotator_name():
        return input.annotator_name()

    return annotator_name


def get_header_css() -> str:
    """Return CSS styles for the header component."""
    return """
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

    .annotator-input label {
        color: #adb5bd;
        font-size: 12px;
        margin-bottom: 4px;
    }

    .annotator-input input {
        background: #343a40;
        border: 1px solid #495057;
        color: white;
        border-radius: 4px;
        padding: 6px 12px;
    }

    .annotator-input input::placeholder {
        color: #6c757d;
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

    .current-position {
        font-size: 14px;
        color: #adb5bd;
        margin-bottom: 8px;
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

    .nav-button {
        padding: 8px 16px;
        font-size: 14px;
    }
    """
