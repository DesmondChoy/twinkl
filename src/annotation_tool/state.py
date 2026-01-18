"""Centralized reactive state management for the annotation tool.

This module provides a single AppState class that consolidates all reactive
values, making data flow explicit and avoiding scattered state declarations.

Usage:
    from state import create_app_state

    def server(input, output, session):
        state = create_app_state()
        header.header_server("header", state=state, ...)
"""

from dataclasses import dataclass, field

from shiny import reactive

from src.models.judge import SCHWARTZ_VALUE_ORDER


@dataclass
class AppState:
    """Centralized reactive state for the annotation tool.

    Attributes:
        persona_index: Current persona index (0-based)
        entry_index: Current entry index within persona (0-based)
        annotated_count: Number of entries annotated by current annotator
        bio_expanded: Whether persona bio is expanded in sidebar
        scores: Dict mapping value names to reactive score values (-1, 0, 1)
        pending_save: Data pending confirmation in all-neutral modal
    """

    # Navigation state
    persona_index: reactive.value = field(default_factory=lambda: reactive.value(0))
    entry_index: reactive.value = field(default_factory=lambda: reactive.value(0))

    # Progress tracking
    annotated_count: reactive.value = field(default_factory=lambda: reactive.value(0))

    # UI state
    bio_expanded: reactive.value = field(default_factory=lambda: reactive.value(True))

    # Score values - one per Schwartz value
    scores: dict = field(default_factory=dict)

    # Pending save data (for modal confirmation)
    pending_save: reactive.value = field(default_factory=lambda: reactive.value(None))

    def __post_init__(self):
        """Initialize score reactive values for each Schwartz value."""
        if not self.scores:
            self.scores = {value: reactive.value(0) for value in SCHWARTZ_VALUE_ORDER}

    def reset_scores(self):
        """Reset all scores to neutral (0)."""
        for value in SCHWARTZ_VALUE_ORDER:
            self.scores[value].set(0)

    def load_scores(self, score_data: dict):
        """Load scores from a dictionary (e.g., existing annotation).

        Args:
            score_data: Dict with keys like 'alignment_self_direction' -> int
        """
        for value in SCHWARTZ_VALUE_ORDER:
            score_key = f"alignment_{value}"
            if score_key in score_data:
                self.scores[value].set(int(score_data[score_key]))
            else:
                self.scores[value].set(0)

    def get_scores_dict(self) -> dict:
        """Get current scores as a plain dict mapping value names to scores.

        Returns:
            Dict like {'self_direction': 0, 'stimulation': 1, ...}
        """
        return {value: self.scores[value]() for value in SCHWARTZ_VALUE_ORDER}

    def all_neutral(self) -> bool:
        """Check if all scores are neutral (0).

        Returns:
            True if all scores are 0
        """
        return all(self.scores[value]() == 0 for value in SCHWARTZ_VALUE_ORDER)


def create_app_state() -> AppState:
    """Factory function to create a fresh AppState instance.

    Returns:
        New AppState with all reactive values initialized
    """
    return AppState()
