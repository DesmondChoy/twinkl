"""Main Shiny app for the Schwartz Value Annotation Tool.

This app allows human annotators to label journal entries across 10 Schwartz
value dimensions for validating LLM Judge labels.

Usage:
    shiny run src/annotation_tool/app.py

Features:
    - Sequential entry navigation (grouped by persona)
    - Free-form annotator name input
    - 10-value scoring grid with counter buttons (-1, 0, +1)
    - Collapsible persona bio
    - Nudge/Response threading display
    - Progress tracking per annotator
    - All-neutral warning modal
    - Post-save comparison modal
    - Persistent annotations in parquet format
"""

import sys
from itertools import groupby
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
from src.annotation_tool.components import (
    analysis_view,
    comparison_view,
    entry_display,
    header,
    modals,
    scoring_grid,
    sidebar,
)
from src.annotation_tool.data_loader import get_ordered_entries, load_entries
from src.annotation_tool.state import create_app_state
from src.models.judge import SCHWARTZ_VALUE_ORDER

# =============================================================================
# Data Loading (cached at module level)
# =============================================================================

_entries_df = load_entries()
_all_entries = get_ordered_entries(_entries_df)
_total_entries = len(_all_entries)


def _group_entries_by_persona(entries: list[dict]) -> list[list[dict]]:
    """Group entries by persona_id, preserving order within each persona."""
    grouped = []
    for _, group in groupby(entries, key=lambda e: e["persona_id"]):
        grouped.append(list(group))
    return grouped


_personas = _group_entries_by_persona(_all_entries)
_total_personas = len(_personas)


def _load_judge_labels() -> dict[tuple[str, int], dict]:
    """Load judge labels into a lookup dictionary keyed by (persona_id, t_index)."""
    import polars as pl

    labels_path = Path("logs/judge_labels/judge_labels.parquet")
    if not labels_path.exists():
        return {}
    df = pl.read_parquet(labels_path)
    return {(row["persona_id"], row["t_index"]): row for row in df.to_dicts()}


_judge_labels_map = _load_judge_labels()


def get_judge_scores(persona_id: str, t_index: int) -> dict | None:
    """Get judge scores for a specific entry, or None if not available."""
    return _judge_labels_map.get((persona_id, t_index))


# =============================================================================
# UI Definition
# =============================================================================

# Get the directory containing this file for static assets
_app_dir = Path(__file__).parent

app_ui = ui.page_fluid(
    # External CSS (served from static_assets directory at root URL)
    ui.tags.link(rel="stylesheet", href="styles.css"),
    # External JS
    ui.tags.script(src="keyboard.js"),
    # Keyboard help modal HTML
    ui.HTML(modals.get_keyboard_help_html()),
    # Header component
    header.header_ui("header"),
    # Main content area
    ui.div(
        ui.output_ui("annotator_warning"),
        ui.div(
            # Left sidebar
            sidebar.sidebar_ui("sidebar"),
            # Center column
            entry_display.entry_display_ui("entry"),
            # Right column (dynamic: scoring grid or comparison view)
            ui.output_ui("right_column_content"),
            class_="three-column-layout",
        ),
        # Analysis view (collapsible accordion at bottom)
        analysis_view.analysis_view_ui("analysis"),
        class_="main-container",
    ),
)


# =============================================================================
# Server Logic
# =============================================================================


def server(input, output, session):
    # Create centralized state
    state = create_app_state()

    # ==========================================================================
    # Derived Reactive Calculations
    # ==========================================================================

    @reactive.calc
    def current_persona_entries():
        """Get entries for the current persona."""
        idx = state.persona_index()
        if 0 <= idx < _total_personas:
            return _personas[idx]
        return []

    @reactive.calc
    def current_entry():
        """Get the currently selected entry."""
        entries = current_persona_entries()
        entry_idx = state.entry_index()
        if entries and 0 <= entry_idx < len(entries):
            return entries[entry_idx]
        return None

    @reactive.calc
    def unlocked_entry_count():
        """Calculate how many entries are unlocked for the current persona."""
        # Depend on annotated_count to trigger re-evaluation after saves
        _ = state.annotated_count()

        entries = current_persona_entries()
        if not entries:
            return 0

        annotator = annotator_name()
        if not annotator:
            return 1  # Only first entry unlocked if no annotator

        unlocked = 1  # Entry 0 always unlocked
        for i in range(len(entries) - 1):
            entry = entries[i]
            if get_annotation(annotator, entry["persona_id"], entry["t_index"]):
                unlocked = i + 2
            else:
                break
        return min(unlocked, len(entries))

    # ==========================================================================
    # Header Component
    # ==========================================================================

    def handle_prev():
        """Handle previous persona navigation."""
        new_idx = state.persona_index() - 1
        if new_idx < 0:
            return

        # Check for unsaved changes
        current_notes = scoring_accessors["get_notes"]()
        if state.has_unsaved_changes(current_notes):
            state.pending_navigation.set({"direction": "prev"})
            ui.modal_show(modals.build_unsaved_changes_modal())
        else:
            _do_navigate_prev()

    def handle_next():
        """Handle next persona navigation."""
        new_idx = state.persona_index() + 1
        if new_idx >= _total_personas:
            return

        # Check for unsaved changes
        current_notes = scoring_accessors["get_notes"]()
        if state.has_unsaved_changes(current_notes):
            state.pending_navigation.set({"direction": "next"})
            ui.modal_show(modals.build_unsaved_changes_modal())
        else:
            _do_navigate_next()

    def handle_unsaved_cancel():
        """Keep editing - close modal, stay on current entry."""
        ui.modal_remove()
        state.pending_navigation.set(None)

    def handle_unsaved_discard():
        """Discard changes and navigate."""
        ui.modal_remove()
        nav = state.pending_navigation()
        state.pending_navigation.set(None)
        _execute_navigation(nav)

    def handle_unsaved_save():
        """Save current work then navigate."""
        ui.modal_remove()
        scores = state.get_scores_dict()
        notes = scoring_accessors["get_notes"]()
        _do_save(scores, notes)
        nav = state.pending_navigation()
        state.pending_navigation.set(None)
        _execute_navigation(nav)

    annotator_name = header.header_server(
        "header",
        state=state,
        total_entries=_total_entries,
        total_personas=_total_personas,
        current_persona_entries=current_persona_entries,
        on_prev=handle_prev,
        on_next=handle_next,
        on_unsaved_cancel=handle_unsaved_cancel,
        on_unsaved_discard=handle_unsaved_discard,
        on_unsaved_save=handle_unsaved_save,
    )

    # ==========================================================================
    # Sidebar Component
    # ==========================================================================

    sidebar.sidebar_server(
        "sidebar",
        state=state,
        current_entry=current_entry,
        current_persona_entries=current_persona_entries,
        unlocked_entry_count=unlocked_entry_count,
        get_annotation=get_annotation,
        annotator_name=annotator_name,
    )

    # ==========================================================================
    # Entry Display Component
    # ==========================================================================

    entry_display.entry_display_server(
        "entry",
        current_entry=current_entry,
        current_persona_entries=current_persona_entries,
        get_annotation=get_annotation,
        annotator_name=annotator_name,
    )

    # ==========================================================================
    # Scoring Grid Component
    # ==========================================================================

    def handle_save(scores: dict, notes: str | None):
        """Handle save button click from scoring grid."""
        name = annotator_name()
        if not name:
            ui.notification_show(
                "Please enter your name before saving.",
                type="warning",
                duration=3,
            )
            return

        # Check if all neutral
        if state.all_neutral():
            # Store pending save and show modal
            state.pending_save.set({
                "scores": scores,
                "notes": notes,
            })
            ui.modal_show(modals.build_all_neutral_modal())
        else:
            _do_save(scores, notes)

    def handle_modal_cancel():
        """Handle cancel button in all-neutral modal."""
        ui.modal_remove()
        state.pending_save.set(None)

    def handle_modal_confirm():
        """Handle confirm button in all-neutral modal."""
        ui.modal_remove()
        data = state.pending_save()
        if data:
            _do_save(data["scores"], data["notes"])
        state.pending_save.set(None)

    def handle_comparison_continue():
        """Handle continue button in comparison view - advance to next entry."""
        entries = current_persona_entries()
        current_entry_idx = state.entry_index()
        new_entry_idx = current_entry_idx + 1

        # Reset to scoring mode first
        state.set_scoring_mode()

        if new_entry_idx < len(entries):
            state.entry_index.set(new_entry_idx)
            _load_existing_annotation()
        else:
            ui.notification_show(
                "All entries for this persona annotated! Use 'Next Persona' to continue.",
                type="message",
                duration=3,
            )

    scoring_accessors = scoring_grid.scoring_grid_server(
        "scoring",
        state=state,
        on_save=handle_save,
        on_modal_cancel=handle_modal_cancel,
        on_modal_confirm=handle_modal_confirm,
        on_comparison_continue=handle_comparison_continue,
    )

    # ==========================================================================
    # Analysis View Component
    # ==========================================================================

    analysis_view.analysis_view_server(
        "analysis",
        state=state,
        annotator_name=annotator_name,
    )

    # ==========================================================================
    # Core Functions
    # ==========================================================================

    def _load_existing_annotation():
        """Load existing annotation for current entry into state."""
        name = annotator_name()
        entry = current_entry()

        if not name or not entry:
            state.reset_scores()
            scoring_accessors["set_notes"]("")
            # Capture baseline for new/empty entry
            state.capture_baseline(state.get_scores_dict(), "")
            return

        existing = get_annotation(name, entry["persona_id"], entry["t_index"])
        if existing:
            state.load_scores(existing)
            notes_value = existing.get("notes") or ""
            scoring_accessors["set_notes"](notes_value)
            # Capture baseline after loading existing annotation
            state.capture_baseline(state.get_scores_dict(), notes_value)
        else:
            state.reset_scores()
            scoring_accessors["set_notes"]("")
            # Capture baseline for new entry (all zeros, empty notes)
            state.capture_baseline(state.get_scores_dict(), "")

    def _do_save(scores: dict, notes: str | None):
        """Execute the actual save operation."""
        name = annotator_name()
        entry = current_entry()

        if not name or not entry:
            return

        save_annotation(
            annotator_id=name,
            persona_id=entry["persona_id"],
            t_index=entry["t_index"],
            scores=scores,
            notes=notes,
            confidence=None,  # Removed from UI
        )

        # Update annotated count
        state.annotated_count.set(get_annotation_count(name))

        # Capture baseline after save (resets dirty state)
        state.capture_baseline(scores, notes or "")

        # Get judge scores for comparison
        judge_data = get_judge_scores(entry["persona_id"], entry["t_index"])

        # Switch to inline comparison view (replaces modal)
        state.set_comparison_mode(scores, judge_data, entry)

    def _execute_navigation(nav: dict):
        """Execute a pending navigation action.

        Args:
            nav: Dict with 'direction' key ("prev", "next", or "entry")
                 and optional 'target_index' for entry navigation
        """
        if not nav:
            return

        direction = nav.get("direction")

        if direction == "prev":
            _do_navigate_prev()
        elif direction == "next":
            _do_navigate_next()
        elif direction == "entry":
            target_idx = nav.get("target_index")
            if target_idx is not None:
                _do_navigate_entry(target_idx)

    def _do_navigate_prev():
        """Execute previous persona navigation."""
        new_idx = state.persona_index() - 1
        if new_idx >= 0:
            state.set_scoring_mode()
            state.persona_index.set(new_idx)
            state.entry_index.set(0)
            _load_existing_annotation()

    def _do_navigate_next():
        """Execute next persona navigation."""
        new_idx = state.persona_index() + 1
        if new_idx < _total_personas:
            state.set_scoring_mode()
            state.persona_index.set(new_idx)
            state.entry_index.set(0)
            _load_existing_annotation()

    def _do_navigate_entry(target_idx: int):
        """Execute entry selection navigation."""
        unlocked = unlocked_entry_count()
        if target_idx < unlocked:
            state.set_scoring_mode()
            state.entry_index.set(target_idx)
            _load_existing_annotation()

    # ==========================================================================
    # Event Handlers
    # ==========================================================================

    @reactive.effect
    @reactive.event(annotator_name)
    def _on_annotator_change():
        """Update state when annotator name changes."""
        name = annotator_name()
        if name:
            state.annotated_count.set(get_annotation_count(name))
            _load_existing_annotation()
        else:
            state.annotated_count.set(0)

    @reactive.effect
    @reactive.event(input.selected_entry_index)
    def _on_entry_select():
        """Handle entry selection from sidebar clicks."""
        input_value = input.selected_entry_index()
        if input_value is None:
            return

        # Extract index from object (JS sends {index: int, nonce: timestamp})
        if isinstance(input_value, dict):
            selected_idx = input_value.get("index")
        else:
            selected_idx = input_value

        if selected_idx is None:
            return

        # Don't navigate to the same entry
        if selected_idx == state.entry_index():
            return

        unlocked = unlocked_entry_count()
        if selected_idx < unlocked:
            # Check for unsaved changes
            current_notes = scoring_accessors["get_notes"]()
            if state.has_unsaved_changes(current_notes):
                state.pending_navigation.set({
                    "direction": "entry",
                    "target_index": selected_idx,
                })
                ui.modal_show(modals.build_unsaved_changes_modal())
            else:
                _do_navigate_entry(selected_idx)

    # ==========================================================================
    # Output Renderers
    # ==========================================================================

    @render.ui
    def annotator_warning():
        """Show warning if no annotator name entered."""
        name = annotator_name()
        if not name:
            return ui.div(
                "⚠️ Please enter your name above to start annotating.",
                class_="no-annotator-warning",
            )
        return None

    @render.ui
    def right_column_content():
        """Render the right column: scoring grid or comparison view."""
        if state.ui_mode() == "comparison":
            data = state.comparison_data()
            if data:
                return comparison_view.build_comparison_view(
                    data["human_scores"], data["judge_data"], data["entry_data"]
                )
        # Default: show scoring grid
        return scoring_grid.scoring_grid_ui("scoring")

    @reactive.effect
    @reactive.event(input.continue_btn)
    def _on_continue_btn():
        """Handle continue button click from inline comparison view."""
        handle_comparison_continue()


# =============================================================================
# App Creation
# =============================================================================

# Create the app with static file directory
app = App(
    app_ui,
    server,
    static_assets=_app_dir / "static",
)


def _free_port(port: int = 8000) -> None:
    """Kill any process using the specified port before starting the server."""
    import subprocess

    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
        )
        pids = result.stdout.strip().split("\n")
        pids = [p for p in pids if p]

        if pids:
            for pid in pids:
                subprocess.run(["kill", pid], capture_output=True)
            print(f"Killed process(es) on port {port}: {', '.join(pids)}")
    except Exception:
        pass


if __name__ == "__main__":
    from shiny import run_app

    _free_port(8000)
    run_app(app)
