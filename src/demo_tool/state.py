"""Centralized reactive state for the demo review app."""

from __future__ import annotations

from dataclasses import dataclass, field

from shiny import reactive


@dataclass
class DemoAppState:
    """Shared UI state for persona browsing and live pipeline runs."""

    selected_persona_id: reactive.value = field(default_factory=lambda: reactive.value(None))
    selected_checkpoint_path: reactive.value = field(default_factory=lambda: reactive.value(None))
    bio_expanded: reactive.value = field(default_factory=lambda: reactive.value(True))
    run_status: reactive.value = field(default_factory=lambda: reactive.value("idle"))
    run_error: reactive.value = field(default_factory=lambda: reactive.value(None))
    run_bundle: reactive.value = field(default_factory=lambda: reactive.value(None))
    active_selection_key: reactive.value = field(default_factory=lambda: reactive.value(None))

    def set_selection(self, persona_id: str | None, checkpoint_path: str | None) -> None:
        self.selected_persona_id.set(persona_id)
        self.selected_checkpoint_path.set(checkpoint_path)
        self.active_selection_key.set(self.selection_key(persona_id, checkpoint_path))

    @staticmethod
    def selection_key(persona_id: str | None, checkpoint_path: str | None) -> str | None:
        if not persona_id or not checkpoint_path:
            return None
        return f"{persona_id}::{checkpoint_path}"

    def clear_result(self, status: str = "idle") -> None:
        self.run_status.set(status)
        self.run_error.set(None)
        self.run_bundle.set(None)

    def set_error(self, message: str) -> None:
        self.run_status.set("error")
        self.run_error.set(message)

    def set_result(self, bundle: dict, *, status: str = "success") -> None:
        self.run_status.set(status)
        self.run_error.set(None)
        self.run_bundle.set(bundle)


def create_demo_state() -> DemoAppState:
    """Create a fresh demo app state container."""
    return DemoAppState()
