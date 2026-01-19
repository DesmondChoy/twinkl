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
    - Graceful error handling for missing/corrupted data files
"""

import sys
import traceback
from dataclasses import dataclass, field
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
# Data Loading with Error Handling
# =============================================================================


@dataclass
class DataLoadResult:
    """Result of attempting to load app data, with error state tracking.

    Attributes:
        entries: List of all journal entries (or None if load failed)
        personas: Entries grouped by persona (or None if load failed)
        judge_labels_map: Dict mapping (persona_id, t_index) to judge labels
        error: Exception if a critical error occurred (app cannot start)
        error_traceback: Formatted traceback string for debugging
        warning: Non-fatal warning message (app can still work)
        skipped_entries: Count of entries skipped during parsing
    """

    entries: list[dict] | None = None
    personas: list[list[dict]] | None = None
    judge_labels_map: dict[tuple[str, int], dict] = field(default_factory=dict)
    error: Exception | None = None
    error_traceback: str | None = None
    warning: str | None = None
    skipped_entries: int = 0

    @property
    def is_success(self) -> bool:
        """Check if data loaded successfully (critical data available)."""
        return self.error is None and self.entries is not None

    @property
    def total_entries(self) -> int:
        """Total number of loaded entries."""
        return len(self.entries) if self.entries else 0

    @property
    def total_personas(self) -> int:
        """Total number of personas."""
        return len(self.personas) if self.personas else 0


def _group_entries_by_persona(entries: list[dict]) -> list[list[dict]]:
    """Group entries by persona_id, preserving order within each persona."""
    grouped = []
    for _, group in groupby(entries, key=lambda e: e["persona_id"]):
        grouped.append(list(group))
    return grouped


def _load_judge_labels_safe() -> tuple[dict[tuple[str, int], dict], str | None]:
    """Load judge labels with error handling.

    Returns:
        Tuple of (labels_dict, warning_message_or_none)
    """
    import polars as pl

    labels_path = Path("logs/judge_labels/judge_labels.parquet")
    if not labels_path.exists():
        return {}, "Judge labels file not found. Comparison view will be limited."

    try:
        df = pl.read_parquet(labels_path)
        return {(row["persona_id"], row["t_index"]): row for row in df.to_dicts()}, None
    except Exception as e:
        return {}, f"Could not load judge labels ({type(e).__name__}). Comparison view will be limited."


def load_app_data() -> DataLoadResult:
    """Load all required data for the annotation tool with error handling.

    Returns:
        DataLoadResult containing either loaded data or error information.
    """
    result = DataLoadResult()
    warnings = []

    # Step 1: Load wrangled entries (critical - app cannot function without this)
    try:
        entries_df = load_entries()
        result.entries = get_ordered_entries(entries_df)
        result.personas = _group_entries_by_persona(result.entries)
    except FileNotFoundError as e:
        result.error = e
        result.error_traceback = traceback.format_exc()
        return result
    except Exception as e:
        result.error = e
        result.error_traceback = traceback.format_exc()
        return result

    # Step 2: Load judge labels (non-critical - app can work without)
    result.judge_labels_map, judge_warning = _load_judge_labels_safe()
    if judge_warning:
        warnings.append(judge_warning)

    # Combine warnings
    if warnings:
        result.warning = " ".join(warnings)

    return result


# Global data load result - deferred loading via function
_data_result: DataLoadResult | None = None


def _get_data_result() -> DataLoadResult:
    """Get or initialize the data load result."""
    global _data_result
    if _data_result is None:
        _data_result = load_app_data()
    return _data_result


def _reload_data() -> DataLoadResult:
    """Force reload of all data (for retry after error)."""
    global _data_result
    _data_result = load_app_data()
    return _data_result


def get_judge_scores(persona_id: str, t_index: int) -> dict | None:
    """Get judge scores for a specific entry, or None if not available."""
    result = _get_data_result()
    return result.judge_labels_map.get((persona_id, t_index))


# =============================================================================
# Error View UI
# =============================================================================


def _build_error_view(result: DataLoadResult) -> ui.Tag:
    """Build the error view UI when data loading fails.

    Args:
        result: DataLoadResult containing error information

    Returns:
        UI component displaying the error with retry option
    """
    error_type = type(result.error).__name__ if result.error else "Unknown Error"
    error_message = str(result.error) if result.error else "An unknown error occurred"

    # Provide helpful suggestions based on error type
    if isinstance(result.error, FileNotFoundError):
        suggestion = "Make sure the 'logs/wrangled/' directory exists and contains persona_*.md files."
        expected_path = "logs/wrangled/persona_*.md"
    else:
        suggestion = "Check that your data files are not corrupted."
        expected_path = "logs/wrangled/"

    return ui.div(
        ui.div(
            ui.div(
                ui.span("⚠️", class_="error-icon"),
                ui.h2("Unable to Load Data", class_="error-title"),
                class_="error-header",
            ),
            ui.p(
                f"The annotation tool could not start because required data files are missing or corrupted.",
                class_="error-description",
            ),
            ui.div(
                ui.div("What went wrong", class_="error-section-label"),
                ui.p(f"{error_type}: {error_message}", class_="error-message-text"),
                class_="error-section",
            ),
            ui.div(
                ui.div("How to fix it", class_="error-section-label"),
                ui.p(suggestion, class_="error-suggestion"),
                ui.div(
                    ui.span("Expected path: ", class_="error-path-label"),
                    ui.code(expected_path, class_="error-path-code"),
                    class_="error-path",
                ),
                class_="error-section",
            ),
            # Collapsible technical details (using HTML details element)
            ui.tags.details(
                ui.tags.summary("Technical details", class_="error-details-summary"),
                ui.tags.pre(
                    result.error_traceback or "No traceback available",
                    class_="error-traceback",
                ),
                class_="error-details",
            ),
            ui.div(
                ui.input_action_button(
                    "retry_load",
                    "🔄 Retry Loading",
                    class_="error-retry-btn",
                ),
                class_="error-actions",
            ),
            class_="error-card",
        ),
        class_="error-view-container",
    )


def _build_warning_banner(warning: str) -> ui.Tag:
    """Build a dismissible warning banner for non-fatal issues.

    Args:
        warning: Warning message to display

    Returns:
        UI component for the warning banner
    """
    return ui.div(
        ui.span("⚠️", class_="warning-banner-icon"),
        ui.span(warning, class_="warning-banner-text"),
        class_="warning-banner",
    )


def _build_empty_state_view() -> ui.Tag:
    """Build a view for when data loads but contains no valid entries.

    Returns:
        UI component displaying the empty state message
    """
    return ui.div(
        ui.div(
            ui.div(
                ui.span("📭", class_="empty-icon"),
                ui.h2("No Entries Found", class_="empty-title"),
                class_="empty-header",
            ),
            ui.p(
                "Data files loaded successfully, but no valid journal entries were found.",
                class_="empty-description",
            ),
            ui.div(
                ui.div("What to check", class_="error-section-label"),
                ui.tags.ul(
                    ui.tags.li("Ensure wrangled files contain properly formatted entries"),
                    ui.tags.li("Check that entries follow the ## Entry N - YYYY-MM-DD format"),
                    ui.tags.li("Verify each entry has content in the initial_entry section"),
                    class_="empty-checklist",
                ),
                class_="error-section",
            ),
            ui.div(
                ui.input_action_button(
                    "retry_load",
                    "🔄 Retry Loading",
                    class_="error-retry-btn",
                ),
                class_="error-actions",
            ),
            class_="error-card",
        ),
        class_="error-view-container",
    )


# =============================================================================
# UI Definition
# =============================================================================

# Get the directory containing this file for static assets
_app_dir = Path(__file__).parent


def _build_main_app_ui() -> ui.Tag:
    """Build the main app UI (when data loads successfully)."""
    return ui.TagList(
        # Keyboard help modal HTML
        ui.HTML(modals.get_keyboard_help_html()),
        # Header component
        header.header_ui("header"),
        # Main content area
        ui.div(
            # Warning banner (shown when there are non-fatal warnings)
            ui.output_ui("warning_banner"),
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


app_ui = ui.page_fluid(
    # External CSS (served from static_assets directory at root URL)
    ui.tags.link(rel="stylesheet", href="styles.css"),
    # External JS
    ui.tags.script(src="keyboard.js"),
    # Dynamic content: error view or main app
    ui.output_ui("app_content"),
)


# =============================================================================
# Server Logic
# =============================================================================


def server(input, output, session):
    # Create centralized state
    state = create_app_state()

    # Reactive value to track data load state and trigger reloads
    data_load_trigger = reactive.value(0)

    # ==========================================================================
    # Data Loading and Error Handling
    # ==========================================================================

    @reactive.calc
    def data_result() -> DataLoadResult:
        """Get the current data load result, reloading if triggered."""
        # Depend on trigger to allow retry
        _ = data_load_trigger()
        return _get_data_result()

    @render.ui
    def app_content():
        """Render either the error view, empty state, or main app based on load state."""
        result = data_result()
        if not result.is_success:
            return _build_error_view(result)
        if result.total_entries == 0:
            return _build_empty_state_view()
        return _build_main_app_ui()

    @reactive.effect
    @reactive.event(input.retry_load)
    def _on_retry_load():
        """Handle retry button click - reload data."""
        _reload_data()
        data_load_trigger.set(data_load_trigger() + 1)

    @render.ui
    def warning_banner():
        """Show warning banner for non-fatal issues."""
        result = data_result()
        if result.is_success and result.warning:
            return _build_warning_banner(result.warning)
        return None

    # ==========================================================================
    # Helper functions to safely access loaded data
    # ==========================================================================

    def _get_personas() -> list[list[dict]]:
        """Safely get personas list."""
        result = data_result()
        return result.personas if result.personas else []

    def _get_total_personas() -> int:
        """Safely get total persona count."""
        result = data_result()
        return result.total_personas

    def _get_total_entries() -> int:
        """Safely get total entry count."""
        result = data_result()
        return result.total_entries

    # ==========================================================================
    # Derived Reactive Calculations
    # ==========================================================================

    @reactive.calc
    def current_persona_entries():
        """Get entries for the current persona."""
        personas = _get_personas()
        idx = state.persona_index()
        if 0 <= idx < len(personas):
            return personas[idx]
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

    def _try_navigate(direction: str, is_valid: bool, navigate_fn: callable):
        """Check for unsaved changes before navigation, showing modal if needed."""
        if not is_valid:
            return

        current_notes = scoring_accessors["get_notes"]()
        if state.has_unsaved_changes(current_notes):
            state.pending_navigation.set({"direction": direction})
            ui.modal_show(modals.build_unsaved_changes_modal())
        else:
            navigate_fn()

    def handle_prev():
        """Handle previous persona navigation."""
        new_idx = state.persona_index() - 1
        _try_navigate("prev", new_idx >= 0, lambda: _navigate_to_persona(new_idx))

    def handle_next():
        """Handle next persona navigation."""
        new_idx = state.persona_index() + 1
        _try_navigate("next", new_idx < _get_total_personas(), lambda: _navigate_to_persona(new_idx))

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
        total_entries=_get_total_entries,  # Pass function instead of static value
        total_personas=_get_total_personas,  # Pass function instead of static value
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
        """Execute the actual save operation with error handling.

        On error, shows a notification and keeps the user on the current entry
        to prevent data loss.
        """
        name = annotator_name()
        entry = current_entry()

        if not name or not entry:
            return

        try:
            save_annotation(
                annotator_id=name,
                persona_id=entry["persona_id"],
                t_index=entry["t_index"],
                scores=scores,
                notes=notes,
                confidence=None,  # Removed from UI
            )
        except ValueError as e:
            # Validation error (invalid scores, etc.)
            ui.notification_show(
                f"Validation error: {e}",
                type="error",
                duration=5,
            )
            return
        except PermissionError as e:
            # File access error
            ui.notification_show(
                f"Could not save: Permission denied. Check that logs/annotations/ is writable.",
                type="error",
                duration=5,
            )
            return
        except Exception as e:
            # Generic error - log for debugging
            ui.notification_show(
                f"Save failed: {type(e).__name__}: {e}",
                type="error",
                duration=5,
            )
            # Log traceback for debugging
            import traceback
            traceback.print_exc()
            return

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
            _navigate_to_persona(state.persona_index() - 1)
        elif direction == "next":
            _navigate_to_persona(state.persona_index() + 1)
        elif direction == "entry":
            target_idx = nav.get("target_index")
            if target_idx is not None:
                _navigate_to_entry(target_idx)

    def _navigate_to_persona(new_idx: int):
        """Navigate to a specific persona index with bounds validation.

        Validates that:
        - new_idx is within valid range
        - Target persona has at least one entry
        """
        total = _get_total_personas()
        if total == 0:
            ui.notification_show(
                "No personas available to navigate to.",
                type="warning",
                duration=3,
            )
            return

        if not (0 <= new_idx < total):
            # Silently ignore invalid indices (shouldn't happen normally)
            return

        # Check if target persona has entries
        personas = _get_personas()
        if not personas[new_idx]:
            ui.notification_show(
                f"Persona {new_idx + 1} has no entries.",
                type="warning",
                duration=3,
            )
            return

        state.set_scoring_mode()
        state.persona_index.set(new_idx)
        state.entry_index.set(0)
        _load_existing_annotation()

    def _navigate_to_entry(target_idx: int):
        """Navigate to a specific entry index with bounds validation.

        Validates that:
        - target_idx is non-negative
        - target_idx is within unlocked entries range
        """
        if target_idx < 0:
            return  # Invalid index

        unlocked = unlocked_entry_count()
        if unlocked == 0:
            return  # No entries to navigate to

        if target_idx >= unlocked:
            # Entry is locked - shouldn't happen from UI, but be defensive
            return

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

        if selected_idx < unlocked_entry_count():
            # Check for unsaved changes
            current_notes = scoring_accessors["get_notes"]()
            if state.has_unsaved_changes(current_notes):
                state.pending_navigation.set({
                    "direction": "entry",
                    "target_index": selected_idx,
                })
                ui.modal_show(modals.build_unsaved_changes_modal())
            else:
                _navigate_to_entry(selected_idx)

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
