"""Centralized reactive state for the demo app session."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from shiny import reactive

from src.demo_tool.onboarding_flow import BwsResponse

# Onboarding sub-stage screens, in flow order. See onboarding_flow.py; this
# mirrors frontend/onboarding/src/session.ts's OnboardingStage exactly
# (there is no mirror/adjust step in the published SVBWS flow).
ONBOARDING_SCREENS = ("welcome", "bws", "goal", "summary")
BWS_SET_COUNT = 11


@dataclass
class DemoAppState:
    """Shared UI state for onboarding, the journal, and weekly review runs."""

    stage: reactive.value = field(default_factory=lambda: reactive.value("onboarding"))
    user_name: reactive.value = field(default_factory=lambda: reactive.value(""))
    core_values: reactive.value = field(default_factory=lambda: reactive.value(()))
    entries: reactive.value = field(default_factory=lambda: reactive.value(()))
    demo_persona_id: reactive.value = field(
        default_factory=lambda: reactive.value(None)
    )
    run_status: reactive.value = field(default_factory=lambda: reactive.value("idle"))
    run_error: reactive.value = field(default_factory=lambda: reactive.value(None))
    review_outcome: reactive.value = field(default_factory=lambda: reactive.value(None))

    # ── Journal composer nudge flow ──────────────────────────────────────────
    # An entry being drafted, staged here while the nudge decision/generation
    # call runs, so the user sees it as "saved" immediately rather than
    # waiting on the LLM before their own words are safely held.
    pending_entry: reactive.value = field(default_factory=lambda: reactive.value(None))
    # "idle" | "checking" | "ready" | "none" | "error"
    nudge_status: reactive.value = field(default_factory=lambda: reactive.value("idle"))
    pending_nudge_text: reactive.value = field(
        default_factory=lambda: reactive.value(None)
    )

    # ── Weekly Digest ────────────────────────────────────────────────────────
    digest_status: reactive.value = field(
        default_factory=lambda: reactive.value("idle")
    )
    digest_result: reactive.value = field(default_factory=lambda: reactive.value(None))
    digest_error: reactive.value = field(default_factory=lambda: reactive.value(None))

    # ── SVBWS onboarding sub-flow ────────────────────────────────────────────
    ob_screen: reactive.value = field(default_factory=lambda: reactive.value("welcome"))
    # Position 0-10 into ob_set_order; the *canonical* set being shown is
    # ob_set_order()[ob_step()], matching session.ts's set_order indirection.
    ob_step: reactive.value = field(default_factory=lambda: reactive.value(0))
    ob_set_order: reactive.value = field(default_factory=lambda: reactive.value(()))
    ob_displayed_orders: reactive.value = field(
        default_factory=lambda: reactive.value(())
    )
    # Responses indexed by canonical set index (0-10), not by step position.
    ob_responses: reactive.value = field(
        default_factory=lambda: reactive.value((None,) * BWS_SET_COUNT)
    )
    ob_draft_best: reactive.value = field(default_factory=lambda: reactive.value(None))
    ob_draft_worst: reactive.value = field(default_factory=lambda: reactive.value(None))
    ob_goal: reactive.value = field(default_factory=lambda: reactive.value(None))
    ob_output: reactive.value = field(default_factory=lambda: reactive.value(None))
    # Core Values confirmed on the summary screen, held here through the
    # complete/first_entry handoff screens until the first real Journal
    # Entry is saved and start_session() commits the journal session.
    ob_pending_core_values: reactive.value = field(
        default_factory=lambda: reactive.value(())
    )

    def __post_init__(self) -> None:
        # Weekly receipt cache, keyed by prompt hash. Deliberately non-reactive:
        # it only saves paid calls and never drives rendering.
        self.receipt_cache: dict[str, Any] = {}

    def reset_onboarding_flow(self) -> None:
        self.ob_screen.set("welcome")
        self.ob_step.set(0)
        self.ob_set_order.set(())
        self.ob_displayed_orders.set(())
        self.ob_responses.set((None,) * BWS_SET_COUNT)
        self.ob_draft_best.set(None)
        self.ob_draft_worst.set(None)
        self.ob_goal.set(None)
        self.ob_output.set(None)
        self.ob_pending_core_values.set(())

    def start_bws(
        self,
        set_order: tuple[int, ...],
        displayed_orders: tuple[tuple[str, ...], ...],
    ) -> None:
        """Begin the BWS assessment with a freshly shuffled set/item order."""
        self.ob_set_order.set(set_order)
        self.ob_displayed_orders.set(displayed_orders)
        self.ob_step.set(0)
        self.ob_responses.set((None,) * BWS_SET_COUNT)
        self.ob_draft_best.set(None)
        self.ob_draft_worst.set(None)
        self.ob_screen.set("bws")

    def current_set_index(self) -> int:
        """Canonical index (0-10) of the set currently on screen."""
        return self.ob_set_order()[self.ob_step()]

    def save_current_bws_set(self) -> None:
        """Commit the current set's Most/Least taps into ob_responses."""
        set_index = self.current_set_index()
        response = BwsResponse(
            set_index=set_index,
            most=self.ob_draft_best(),
            least=self.ob_draft_worst(),
        )
        responses = list(self.ob_responses())
        responses[set_index] = response
        self.ob_responses.set(tuple(responses))

    def _load_draft_for_current_step(self) -> None:
        """Restore the current step's saved Most/Least, if it was answered
        before (e.g. the user is revisiting via Back)."""
        saved = self.ob_responses()[self.current_set_index()]
        self.ob_draft_best.set(saved.most if saved else None)
        self.ob_draft_worst.set(saved.least if saved else None)

    def advance_bws_step(self) -> None:
        """Move to the next BWS set, or on to goal selection after the last."""
        step = self.ob_step()
        if step >= len(self.ob_set_order()) - 1:
            self.ob_screen.set("goal")
            return
        self.ob_step.set(step + 1)
        self._load_draft_for_current_step()

    def retreat_bws_step(self) -> None:
        """Move to the previous BWS set, restoring its saved choice if any.
        A no-op on the first set — nothing before it to go back to."""
        step = self.ob_step()
        if step <= 0:
            return
        self.save_current_bws_set()
        self.ob_step.set(step - 1)
        self._load_draft_for_current_step()

    def back_to_last_bws_step(self) -> None:
        """From the goal screen, return to the final BWS set."""
        self.ob_step.set(len(self.ob_set_order()) - 1)
        self._load_draft_for_current_step()
        self.ob_screen.set("bws")

    def start_session(
        self,
        *,
        user_name: str,
        core_values: tuple[str, ...],
        entries: tuple[dict[str, Any], ...] = (),
        demo_persona_id: str | None = None,
    ) -> None:
        self.user_name.set(user_name)
        self.core_values.set(core_values)
        self.entries.set(entries)
        self.demo_persona_id.set(demo_persona_id)
        self.review_outcome.set(None)
        self.run_status.set("idle")
        self.run_error.set(None)
        self.receipt_cache.clear()
        self.clear_pending_entry()
        self.digest_status.set("idle")
        self.digest_result.set(None)
        self.digest_error.set(None)
        self.stage.set("journal")

    def reset(self) -> None:
        self.stage.set("onboarding")
        self.user_name.set("")
        self.core_values.set(())
        self.entries.set(())
        self.demo_persona_id.set(None)
        self.review_outcome.set(None)
        self.run_status.set("idle")
        self.run_error.set(None)
        self.receipt_cache.clear()
        self.reset_onboarding_flow()
        self.clear_pending_entry()
        self.digest_status.set("idle")
        self.digest_result.set(None)
        self.digest_error.set(None)

    def set_error(self, message: str) -> None:
        self.run_status.set("error")
        self.run_error.set(message)

    def set_result(self, outcome: dict[str, Any]) -> None:
        self.run_status.set("success")
        self.run_error.set(None)
        self.review_outcome.set(outcome)

    # ── Journal composer nudge flow ──────────────────────────────────────────

    def begin_entry_draft(self, entry_date: str, initial_entry: str) -> None:
        """Stage a newly-written entry while the nudge check runs."""
        self.pending_entry.set({"date": entry_date, "initial_entry": initial_entry})
        self.nudge_status.set("checking")
        self.pending_nudge_text.set(None)

    def set_nudge_ready(self, nudge_text: str) -> None:
        self.nudge_status.set("ready")
        self.pending_nudge_text.set(nudge_text)

    def set_no_nudge(self) -> None:
        self.nudge_status.set("none")

    def set_nudge_error(self, message: str) -> None:
        self.nudge_status.set("error")
        self.run_error.set(message)

    def clear_pending_entry(self) -> None:
        self.pending_entry.set(None)
        self.nudge_status.set("idle")
        self.pending_nudge_text.set(None)

    def finalize_pending_entry(self, response_text: str | None = None) -> None:
        """Commit the staged entry (plus any nudge/response) into the journal."""
        pending = self.pending_entry()
        if pending is None:
            return
        entries = self.entries()
        entries = entries + (
            {
                "t_index": len(entries),
                "date": pending["date"],
                "initial_entry": pending["initial_entry"],
                "nudge_text": self.pending_nudge_text(),
                "response_text": response_text,
            },
        )
        self.entries.set(entries)
        self.clear_pending_entry()


def create_demo_state() -> DemoAppState:
    """Create a fresh demo app state container."""
    return DemoAppState()
