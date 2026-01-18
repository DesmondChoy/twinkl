"""Left sidebar component for persona info and entry navigation.

This module provides the left sidebar containing:
- Persona information card with collapsible bio
- Entry navigation list showing status of each entry

Usage:
    from components import sidebar

    # In UI definition
    sidebar.sidebar_ui("sidebar")

    # In server function
    sidebar.sidebar_server(
        "sidebar",
        state=state,
        current_entry=current_entry,
        current_persona_entries=current_persona_entries,
        unlocked_entry_count=unlocked_entry_count,
        get_annotation=get_annotation,
    )
"""

from shiny import module, reactive, render, ui

from src.annotation_tool.state import AppState


@module.ui
def sidebar_ui():
    """Generate the left sidebar UI component.

    Returns:
        UI div containing persona info and entry navigation placeholders
    """
    return ui.div(
        # Persona section
        ui.output_ui("sidebar_persona"),
        # Entry navigation
        ui.output_ui("sidebar_entry_nav"),
        class_="left-sidebar",
    )


@module.server
def sidebar_server(
    input,
    output,
    session,
    state: AppState,
    current_entry: reactive.calc,
    current_persona_entries: reactive.calc,
    unlocked_entry_count: reactive.calc,
    get_annotation: callable,
    annotator_name: reactive.calc,
):
    """Server logic for the left sidebar.

    Args:
        input, output, session: Shiny module parameters
        state: Centralized app state
        current_entry: Reactive calc returning the current entry dict
        current_persona_entries: Reactive calc returning list of entries for current persona
        unlocked_entry_count: Reactive calc returning number of unlocked entries
        get_annotation: Function to check if annotation exists (annotator, persona_id, t_index) -> dict|None
        annotator_name: Reactive calc returning the current annotator name
    """

    @render.ui
    def sidebar_persona():
        """Render the persona information card."""
        entry = current_entry()
        if entry is None:
            return ui.div("No persona selected", class_="sidebar-persona")

        # Format core values
        core_values = entry.get("persona_core_values", [])
        if isinstance(core_values, list):
            values_str = ", ".join(core_values)
        else:
            values_str = str(core_values)

        # Build persona info parts
        age = entry.get("persona_age", "")
        profession = entry.get("persona_profession", "")
        culture = entry.get("persona_culture", "")

        # Bio section with collapsible toggle
        bio = entry.get("persona_bio", "")
        is_expanded = state.bio_expanded()

        bio_section = None
        if bio:
            toggle_class = "sidebar-bio-toggle expanded" if is_expanded else "sidebar-bio-toggle"
            toggle_text = "▲ Hide Bio" if is_expanded else "▼ Show Bio"
            bio_section = ui.div(
                ui.input_action_button(
                    "bio_toggle",
                    ui.span(toggle_text),
                    class_=toggle_class,
                ),
                ui.div(bio, class_="sidebar-bio-text") if is_expanded else None,
                class_="sidebar-bio-container",
            )

        return ui.div(
            ui.h4(entry.get("persona_name", "Unknown"), class_="sidebar-persona-name"),
            ui.div(f"{age} • {profession}", class_="sidebar-persona-info") if age and profession else None,
            ui.div(culture, class_="sidebar-persona-info") if culture else None,
            ui.div(f"Core: {values_str}", class_="sidebar-persona-values"),
            bio_section,
            class_="sidebar-persona",
        )

    @render.ui
    def sidebar_entry_nav():
        """Render the entry navigation list."""
        entries = current_persona_entries()
        if not entries:
            return ui.div("No entries", class_="sidebar-entry-nav")

        annotator = annotator_name()
        unlocked = unlocked_entry_count()
        current_idx = state.entry_index()

        items = []
        for i, entry in enumerate(entries):
            is_locked = i >= unlocked
            is_current = i == current_idx
            is_annotated = False

            if annotator:
                existing = get_annotation(
                    annotator, entry["persona_id"], entry["t_index"]
                )
                is_annotated = existing is not None

            # Determine status icon
            if is_locked:
                status_icon = "🔒"
                status_class = "locked"
            elif is_annotated:
                status_icon = "✓"
                status_class = "annotated"
            elif is_current:
                status_icon = "●"
                status_class = "current"
            else:
                status_icon = "○"
                status_class = ""

            # Determine item classes
            item_classes = "sidebar-entry-item"
            if is_current:
                item_classes += " current"
            if is_locked:
                item_classes += " locked"

            t_index = entry.get("t_index", 0)
            date = entry.get("date", "")

            item = ui.div(
                ui.span(status_icon, class_=f"sidebar-entry-status {status_class}"),
                ui.div(
                    ui.div(f"Entry {t_index + 1}", class_="sidebar-entry-title"),
                    ui.div(date, class_="sidebar-entry-date"),
                    class_="sidebar-entry-info",
                ),
                class_=item_classes,
                id=f"sidebar_entry_{i}",
                **{"data-entry-index": str(i)},
            )
            items.append(item)

        return ui.div(
            ui.div("Entries", class_="sidebar-entry-nav-header"),
            ui.div(*items, class_="sidebar-entry-list"),
            class_="sidebar-entry-nav",
        )

    # Bio toggle handler
    @reactive.effect
    @reactive.event(input.bio_toggle)
    def _on_bio_toggle():
        state.bio_expanded.set(not state.bio_expanded())
