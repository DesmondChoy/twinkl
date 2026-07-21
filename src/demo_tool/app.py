"""Shiny demo app for the approved user-facing Twinkl flow.

Onboarding: the user completes the published SVBWS values assessment (or
preloads a synthetic demo persona), yielding 1-2 Core Values. Journal: the
user writes Journal Entries, then runs the Weekly Drift Reviewer. Each Journal
Entry gets a Conflict / Not Conflict / Abstain decision per Core Value, and
the Drift Detector confirms Drift on two consecutive Conflicts for the same
Core Value. There is no VIF Critic input on this path.

The SVBWS instrument (item bank, balanced incomplete block design, and
scoring) is ported from frontend/onboarding/src/domain.ts — see
src/demo_tool/onboarding_flow.py.
"""

from __future__ import annotations

import asyncio
import os
import random
import sys
import traceback
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from shiny import App, reactive, render, ui

# Add project root to path for imports when running via shiny run
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Load .env so the OpenAI and coach provider keys are available without exporting.
from dotenv import load_dotenv  # noqa: E402

load_dotenv(_project_root / ".env")

from src.coach.llm_client import build_llm_complete  # noqa: E402
from src.coach.weekly_digest import (  # noqa: E402
    attach_coach_artifacts,
    build_weekly_drift_reviewer_digest,
    generate_weekly_digest_coach,
    validate_weekly_digest_narrative,
)
from src.demo_tool.data_loader import (  # noqa: E402
    build_persona_choices,
    get_persona,
    load_demo_personas,
    summarize_warnings,
)
from src.demo_tool.nudge_bridge import check_for_nudge  # noqa: E402
from src.demo_tool.onboarding_flow import (  # noqa: E402
    BWS_OBJECTS,
    BWS_SETS,
    GOALS,
    VALUE_PHRASES,
    score_responses,
    top_core_values,
)
from src.demo_tool.runtime_bridge import (  # noqa: E402
    MAX_CORE_VALUES,
    demo_journal_from_persona,
    review_journal,
)
from src.demo_tool.state import DemoAppState, create_demo_state  # noqa: E402

GOAL_TEXT_BY_KEY = dict(GOALS)

VERDICT_LABELS = {
    "conflict": "Conflict",
    "not_conflict": "Not Conflict",
    "abstain": "Uncertain",
}
VERDICT_BADGES = {
    "conflict": "badge-negative",
    "not_conflict": "badge-positive",
    "abstain": "badge-neutral",
}
STATE_LABELS = {
    "stable": "Stable",
    "active": "Active Drift",
    "recovered": "Recovered",
    "uncertain": "Uncertain",
    "mixed": "Mixed",
}


@dataclass
class CatalogLoadResult:
    """Loaded demo persona catalog, warnings, and error state."""

    personas: list[dict[str, Any]] = field(default_factory=list)
    warning: str | None = None
    error: str | None = None


def _load_catalog_data() -> CatalogLoadResult:
    """Load synthetic demo personas. Failure only disables the demo shortcut."""
    try:
        persona_result = load_demo_personas()
    except Exception as exc:
        return CatalogLoadResult(
            error=f"{type(exc).__name__}: {exc}",
            warning=traceback.format_exc(limit=1),
        )
    return CatalogLoadResult(
        personas=persona_result.personas,
        warning=summarize_warnings(persona_result.warnings),
    )


_catalog_result: CatalogLoadResult | None = None


def _get_catalog_result() -> CatalogLoadResult:
    global _catalog_result
    if _catalog_result is None:
        _catalog_result = _load_catalog_data()
    return _catalog_result


def _format_value(value: str) -> str:
    """Display one Schwartz value name, e.g. self_direction -> Self-Direction."""
    return "-".join(part.capitalize() for part in value.split("_"))


def _build_badges(values: list[str], badge_class: str, empty_label: str) -> ui.Tag:
    """Render a row of small badges."""
    labels = values or [empty_label]
    return ui.div(
        *[ui.span(label, class_=f"badge {badge_class}") for label in labels],
        class_="badge-row",
    )


def _build_table(
    records: list[dict[str, Any]], columns: list[tuple[str, str]]
) -> ui.Tag:
    """Render a compact HTML table from simple row records."""
    if not records:
        return ui.div("No rows to display.", class_="empty-state")

    header_cells = [ui.tags.th(label) for _key, label in columns]
    body_rows = []
    for record in records:
        cells = []
        for key, _label in columns:
            value = record.get(key)
            if isinstance(value, list):
                value = ", ".join(str(item) for item in value)
            cells.append(ui.tags.td("" if value is None else value))
        body_rows.append(ui.tags.tr(*cells))

    return ui.div(
        ui.tags.table(
            ui.tags.thead(ui.tags.tr(*header_cells)),
            ui.tags.tbody(*body_rows),
            class_="data-table",
        ),
        class_="table-shell",
    )


# ── Onboarding: SVBWS values assessment ─────────────────────────────────────
#
# Ports frontend/onboarding/src/{domain,session,App,styles}.ts(x)/.css as a
# Shiny flow: welcome -> 11 BWS groups -> goal selection -> summary -> a
# handoff screen -> the first real Journal Entry, matching that app's
# presentation (palette, type, two-panel layout with an animated compass,
# card treatment) as well as its instrument. There is no mid-flow correction
# step in either app. The one deliberate content deviation is this welcome
# screen itself: his app has none (it opens straight on group 1), but this
# app also offers a demo-persona shortcut his doesn't need, so a minimal gate
# lives here — styled in the same visual language.
#
# Card selection is a single tap per card, not his pointer drag: the first
# tap in a group sets Most (the card turns green), the second sets Least (it
# turns red); tapping a set card clears it, and tapping a third, different
# card replaces Most. No drag-and-drop and no intermediate "picked" step —
# simpler than his own drag implementation and its tap fallback alike.

_OB_STAGE_COUNT = len(BWS_SETS) + 2  # 11 groups + goal + summary
_OB_COMPASS_STAR_COUNT = 10  # one per Schwartz value, matching his Compass component


def _ob_tap_onclick(input_id: str, value: str) -> str:
    """JS for a tappable card: fires a Shiny custom input event on click.

    ``value`` is always a static key from a fixed item bank (a BWS object
    key, a zone name, or a goal key), never user-entered text, so this is
    safe to inline.
    """
    return f"Shiny.setInputValue('{input_id}', '{value}', {{priority: 'event'}})"


def _ob_milestone(state: DemoAppState) -> int:
    screen = state.ob_screen()
    if screen == "welcome":
        return 0
    if screen == "bws":
        return min(state.ob_step() + 1, len(BWS_SETS))
    if screen == "goal":
        return len(BWS_SETS) + 1
    return len(BWS_SETS) + 2  # summary


def _render_compass(milestone: int) -> ui.Tag:
    progress_deg = milestone / _OB_STAGE_COUNT * 360
    stars = [
        ui.span(class_="ob-compass__star", style=f"--star-index:{i}")
        for i in range(_OB_COMPASS_STAR_COUNT)
    ]
    return ui.div(
        ui.div(*stars, class_="ob-compass__orbit"),
        ui.div(class_="ob-compass__needle"),
        ui.div(
            ui.span("✦", **{"aria-hidden": "true"}),
            ui.tags.small("compass"),
            class_="ob-compass__center",
        ),
        class_="ob-compass",
        style=f"--compass-progress:{progress_deg:.2f}deg",
        **{"aria-hidden": "true"},
    )


def _render_progress_header(state: DemoAppState, milestone: int) -> ui.Tag | None:
    screen = state.ob_screen()
    if milestone == 0 or screen == "first_entry":
        return None
    if screen == "bws":
        label = f"Values · {state.ob_step() + 1} of {len(BWS_SETS)}"
    elif screen == "goal":
        label = "Your focus"
    else:
        label = "Your compass"
    pct = milestone / _OB_STAGE_COUNT * 100
    return ui.div(
        ui.div(ui.span(label), class_="ob-progress-label"),
        ui.div(ui.span(style=f"width:{pct:.1f}%"), class_="ob-progress-track"),
        class_="ob-progress-header",
    )


def _ob_disabled_button(label: str) -> ui.Tag:
    return ui.tags.button(
        label, type="button", class_="btn-primary", disabled=True
    )


def _render_welcome(catalog: CatalogLoadResult) -> ui.Tag:
    demo_section: list[Any] = []
    if catalog.error:
        demo_section.append(
            ui.div(
                f"Demo personas unavailable — {catalog.error}",
                class_="status-bar s-warning",
            )
        )
    else:
        demo_section.append(
            ui.div(
                ui.tags.label("Or load a demo persona"),
                ui.input_select(
                    "demo_persona_id",
                    None,
                    choices=build_persona_choices(catalog.personas),
                ),
                ui.input_action_button(
                    "load_demo_btn", "Load demo persona", class_="btn-secondary"
                ),
                class_="ob-demo-shortcut",
            )
        )

    return ui.div(
        ui.h1("Your Inner Compass"),
        ui.p(
            "Most apps track what you do. Twinkl tracks whether what you do "
            "matches what you value. A quick values assessment first — "
            f"{len(BWS_SETS)} groups, choosing what matters most and least in "
            "each — then straight into journaling.",
            class_="ob-lede",
        ),
        ui.div(
            ui.tags.label("Your name (optional)"),
            ui.input_text("user_name", None, placeholder="e.g. Jodie"),
            class_="ob-welcome-field",
        ),
        ui.div(
            ui.input_action_button("ob_begin_btn", "Begin →", class_="btn-primary"),
            class_="ob-actions",
        ),
        *demo_section,
        class_="ob-stage",
    )


def _render_bws_set(state: DemoAppState) -> ui.Tag:
    step = state.ob_step()
    set_index = state.current_set_index()
    displayed_order = list(state.ob_displayed_orders()[set_index])
    draft_best = state.ob_draft_best()
    draft_worst = state.ob_draft_worst()

    def _card_style(key: str) -> str:
        index = displayed_order.index(key)
        bg = f"/card-backgrounds/memory-atlas-{index + 1:02d}.jpg"
        angle = (index - 2.5) * 0.8
        delay = index * 70
        return (
            f"--ob-card-bg:url('{bg}'); --ob-card-angle:{angle:.2f}deg; "
            f"--ob-card-delay:{delay}ms;"
        )

    def _card(key: str) -> ui.Tag:
        cls = "ob-value-card"
        if key == draft_best:
            cls += " ob-value-card--most"
        elif key == draft_worst:
            cls += " ob-value-card--least"
        return ui.tags.article(
            ui.span(BWS_OBJECTS[key][1], class_="ob-value-card__phrase"),
            class_=cls,
            style=_card_style(key),
            onclick=_ob_tap_onclick("bws_card_tap", key),
        )

    if draft_best is None and draft_worst is None:
        prompt = (
            "Across 11 groups, tap what matters most, then what matters "
            "least. Some cards will return."
            if step == 0
            else "Tap what matters most in this group, then what matters "
            "least."
        )
    elif draft_worst is None:
        prompt = "Now tap the principle that matters least to you in this group."
    else:
        prompt = "Both choices are set. Tap either card to reconsider."

    # Only "deal" the cards in on the round's first paint. Re-renders
    # triggered by tapping Most/Least within the same round should update
    # colors instantly, not replay the whole deck's entrance animation.
    deck_cls = "ob-card-deck"
    if draft_best is None and draft_worst is None:
        deck_cls += " ob-card-deck--fresh"

    ok = draft_best is not None and draft_worst is not None
    next_control = (
        ui.input_action_button("bws_next_btn", "Continue", class_="btn-primary")
        if ok
        else _ob_disabled_button("Continue")
    )
    back_control = (
        ui.input_action_button("bws_back_btn", "← Back", class_="btn-secondary")
        if step > 0
        else None
    )

    return ui.div(
        ui.h1("What matters most as you find your way?"),
        ui.p(
            "There are no right answers here. More than one principle can "
            "matter. Tap your first choice for Most (it turns green), then "
            "your second for Least (it turns red).",
            class_="ob-card-reassurance",
        ),
        ui.div(
            ui.span("Next step", class_="ob-card-prompt__label"),
            ui.span(prompt),
            class_="ob-card-prompt",
        ),
        ui.div(*(_card(key) for key in displayed_order), class_=deck_cls),
        ui.div(back_control, next_control, class_="ob-actions ob-actions--end"),
        class_="ob-stage",
    )


def _render_goal(state: DemoAppState) -> ui.Tag:
    selected_goal = state.ob_goal()
    cards = []
    for key, text in GOALS:
        cls = "ob-goal-card" + (" selected" if key == selected_goal else "")
        cards.append(
            ui.div(text, class_=cls, onclick=_ob_tap_onclick("goal_tap", key))
        )

    next_control = (
        ui.input_action_button(
            "goal_next_btn", "See my compass →", class_="btn-primary"
        )
        if selected_goal
        else _ob_disabled_button("See my compass →")
    )
    back_control = ui.input_action_button(
        "goal_back_btn", "← Back", class_="btn-secondary"
    )

    return ui.div(
        ui.h1("What brought you here right now?"),
        ui.p("Choose the one closest to what brought you here.", class_="ob-lede"),
        ui.div(*cards, class_="ob-goal-list"),
        ui.div(back_control, next_control, class_="ob-actions ob-actions--end"),
        class_="ob-stage",
    )


def _render_summary(state: DemoAppState) -> ui.Tag:
    _raw, profile = score_responses(list(state.ob_responses()))
    core_values = top_core_values(profile, max_values=MAX_CORE_VALUES)
    goal_text = GOAL_TEXT_BY_KEY.get(state.ob_goal() or "", "")

    core_cards = [
        ui.tags.article(
            ui.span("✦", **{"aria-hidden": "true"}),
            ui.p(VALUE_PHRASES[value]),
        )
        for value in core_values
    ]

    return ui.div(
        ui.h1("What sits at the center."),
        ui.div(*core_cards, class_="ob-core-values"),
        ui.div(
            ui.tags.small("What brought you here"),
            ui.p(goal_text),
            class_="ob-focus-line",
        ),
        ui.div(
            ui.input_action_button(
                "summary_back_btn", "← Back", class_="btn-secondary"
            ),
            ui.input_action_button(
                "confirm_onboarding_btn", "Set my compass", class_="btn-primary"
            ),
            class_="ob-actions ob-actions--end",
        ),
        class_="ob-stage",
    )


def _render_complete(_state: DemoAppState) -> ui.Tag:
    return ui.div(
        ui.h1("Your compass is ready."),
        ui.p(
            "Start with one moment from the past week. Twinkl will build from "
            "what you notice.",
            class_="ob-lede",
        ),
        ui.div(
            ui.tags.small("First Journal Entry"),
            ui.p("When did you feel most like yourself?"),
            class_="ob-journal-handoff",
        ),
        ui.div(
            ui.input_action_button(
                "ob_start_first_entry_btn",
                "Start my first Journal Entry",
                class_="btn-primary",
            ),
            class_="ob-actions ob-actions--end",
        ),
        class_="ob-stage",
    )


def _render_first_entry(_state: DemoAppState) -> ui.Tag:
    return ui.div(
        ui.p("First Journal Entry", class_="ob-eyebrow"),
        ui.h1("When did you feel most like yourself?"),
        ui.p(
            "Think of one moment from the past week. What was happening, and "
            "what felt true about it?",
            class_="ob-lede",
        ),
        ui.input_text_area(
            "ob_first_entry_text",
            None,
            placeholder="Start with the moment…",
            rows=8,
            width="100%",
        ),
        ui.div(
            ui.input_action_button(
                "ob_save_first_entry_btn", "Save Journal Entry →", class_="btn-primary"
            ),
            class_="ob-actions ob-actions--end",
        ),
        class_="ob-stage",
    )


_OB_SCREEN_RENDERERS = {
    "bws": _render_bws_set,
    "goal": _render_goal,
    "summary": _render_summary,
    "complete": _render_complete,
    "first_entry": _render_first_entry,
}


def _build_onboarding_ui(state: DemoAppState, catalog: CatalogLoadResult) -> ui.Tag:
    screen = state.ob_screen()
    milestone = _ob_milestone(state)
    stage_content = (
        _render_welcome(catalog)
        if screen == "welcome"
        else _OB_SCREEN_RENDERERS[screen](state)
    )

    topbar_children: list[Any] = [
        ui.tags.a("twinkl", ui.span("·"), class_="ob-wordmark", href="#ob-main"),
    ]
    if screen != "welcome":
        topbar_children.append(
            ui.tags.button(
                "Start over",
                type="button",
                class_="ob-restart",
                onclick=(
                    "Shiny.setInputValue('ob_restart_btn', Date.now(), "
                    "{priority: 'event'})"
                ),
            )
        )

    return ui.div(
        ui.div(*topbar_children, class_="ob-topbar"),
        ui.tags.main(
            ui.tags.aside(
                _render_compass(milestone),
                ui.div(
                    ui.p("Your inner compass", class_="ob-eyebrow"),
                    class_="ob-instrument-copy",
                ),
                class_="ob-instrument-panel",
            ),
            ui.div(
                _render_progress_header(state, milestone),
                stage_content,
                class_="ob-flow-panel",
            ),
            class_="ob-layout",
            id="ob-main",
        ),
        class_="onboard-shell",
    )


# ── Journal stage ────────────────────────────────────────────────────────────


def _build_control_sidebar() -> ui.Tag:
    """The persistent left rail: brand, profile, actions, live status."""
    return ui.sidebar(
        # ── Collapsed T-rail (shown only when .app-shell has .t-collapsed) ─
        ui.div(
            ui.div(
                "T",
                class_="mini-mark",
                title="Expand sidebar",
                onclick="document.querySelector('.app-shell').classList.remove('t-collapsed')",
            ),
            ui.div(
                "»",
                class_="mini-expand",
                title="Expand sidebar",
                onclick="document.querySelector('.app-shell').classList.remove('t-collapsed')",
            ),
            class_="sidebar-mini",
        ),
        # ── Brand (with collapse control) ────────────────────────────────
        ui.div(
            ui.div(
                ui.div("TWINKL", class_="brand-mark"),
                ui.div("Your journal", class_="brand-sub"),
            ),
            ui.div(
                "«",
                class_="sidebar-collapse-btn",
                title="Collapse sidebar",
                onclick="document.querySelector('.app-shell').classList.add('t-collapsed')",
            ),
            class_="sidebar-brand",
        ),
        # ── Profile ──────────────────────────────────────────────────────
        ui.output_ui("profile_summary"),
        # ── Actions ──────────────────────────────────────────────────────
        # No manual "run review" control: the weekly review and digest both
        # run automatically as entries change (see _auto_run_review_on_
        # entries_change / _auto_build_digest_on_outcome_change). Retry/
        # Refresh live contextually in run_status_banner / weekly_digest_panel.
        ui.div(
            ui.input_action_button(
                "reset_session_btn", "Start over", class_="btn-secondary"
            ),
            class_="sidebar-actions",
        ),
        # ── Live status ──────────────────────────────────────────────────
        ui.output_ui("run_status_banner"),
        id="controls_sidebar",
        title=None,
        width=308,
        open="always",  # collapse is driven by our own T-rail (.app-shell.t-collapsed)
        class_="app-sidebar",
    )


def _build_journal_ui() -> ui.Tag:
    main_content = ui.div(
        ui.div(
            ui.h1("Your journal", class_="app-title"),
            ui.p(
                "Write about your day. Twinkl quietly checks it against what "
                "matters to you and lets you know if a pattern needs a second "
                "look.",
                class_="app-subtitle",
            ),
            class_="app-header",
        ),
        ui.output_ui("entry_composer"),
        ui.output_ui("journal_list"),
        ui.output_ui("drift_summary"),
        ui.output_ui("weekly_digest_panel"),
        # Internal/debug only — everything the real product surface doesn't
        # need. Deleting this one line (and the internal_debug_panel render
        # function below it) removes it cleanly with nothing else to touch.
        ui.output_ui("internal_debug_panel"),
        class_="app-main",
    )
    return ui.div(
        ui.layout_sidebar(_build_control_sidebar(), main_content),
        class_="app-shell",
    )


def _render_pending_entry(
    pending: dict[str, Any], nudge_status: str, nudge_text: str | None
) -> ui.Tag:
    """The composer while a just-saved entry is mid-flow: checking for a
    nudge, or showing one and waiting on the user's own reply."""
    entry_block = ui.div(
        ui.div(
            ui.span(pending["date"], class_="entry-date"),
            class_="entry-meta",
        ),
        ui.p(pending["initial_entry"], class_="entry-text"),
        class_="timeline-card",
    )

    if nudge_status == "checking":
        return ui.div(
            ui.h2("Just saved", class_="composer-heading"),
            entry_block,
            ui.div(
                ui.strong("Checking in"),
                " — Twinkl is reading your entry for a follow-up question.",
                class_="status-bar s-running",
            ),
            class_="composer-card composer-card--hero",
        )

    # nudge_status == "ready"
    return ui.div(
        ui.h3("New Journal Entry", class_="section-heading"),
        entry_block,
        ui.div(
            ui.strong("Twinkl asks"),
            ui.p(nudge_text, class_="thread-text"),
            class_="thread-block thread-nudge",
        ),
        ui.input_text_area(
            "nudge_reply_text",
            None,
            placeholder="Your reply (optional)...",
            rows=3,
            width="100%",
        ),
        ui.div(
            ui.input_action_button(
                "nudge_reply_btn", "Send reply", class_="btn-primary"
            ),
            ui.input_action_button(
                "nudge_skip_btn", "Skip", class_="btn-secondary"
            ),
            class_="composer-actions",
        ),
        class_="composer-card composer-card--hero",
    )


def server(input, output, session):
    state: DemoAppState = create_demo_state()

    @render.ui
    def app_content():
        if state.stage() == "onboarding":
            return _build_onboarding_ui(state, _get_catalog_result())
        return _build_journal_ui()

    # ── Onboarding handlers ──────────────────────────────────────────────

    @reactive.effect
    @reactive.event(input.ob_restart_btn)
    def _ob_restart():
        state.reset()

    @reactive.effect
    @reactive.event(input.ob_begin_btn)
    def _ob_begin():
        state.user_name.set((input.user_name() or "").strip())
        set_order = tuple(random.sample(range(len(BWS_SETS)), len(BWS_SETS)))
        displayed_orders = tuple(
            tuple(random.sample(items, len(items))) for items in BWS_SETS
        )
        state.start_bws(set_order, displayed_orders)

    @reactive.effect
    @reactive.event(input.bws_card_tap)
    def _on_bws_card_tap():
        tapped = input.bws_card_tap()
        if not tapped:
            return
        best, worst = state.ob_draft_best(), state.ob_draft_worst()
        if tapped == best:
            state.ob_draft_best.set(None)
        elif tapped == worst:
            state.ob_draft_worst.set(None)
        elif best is None:
            state.ob_draft_best.set(tapped)
        elif worst is None:
            state.ob_draft_worst.set(tapped)
        else:
            # Both already set and this is a third, different card:
            # replace Most, matching the tap semantics used elsewhere.
            state.ob_draft_best.set(tapped)

    @reactive.effect
    @reactive.event(input.bws_next_btn)
    def _on_bws_next():
        if state.ob_draft_best() is None or state.ob_draft_worst() is None:
            return
        state.save_current_bws_set()
        state.advance_bws_step()

    @reactive.effect
    @reactive.event(input.bws_back_btn)
    def _on_bws_back():
        state.retreat_bws_step()

    @reactive.effect
    @reactive.event(input.goal_tap)
    def _on_goal_tap():
        tapped = input.goal_tap()
        if tapped:
            state.ob_goal.set(tapped)

    @reactive.effect
    @reactive.event(input.goal_next_btn)
    def _on_goal_next():
        if state.ob_goal():
            state.ob_screen.set("summary")

    @reactive.effect
    @reactive.event(input.goal_back_btn)
    def _on_goal_back():
        state.back_to_last_bws_step()

    @reactive.effect
    @reactive.event(input.summary_back_btn)
    def _on_summary_back():
        state.ob_screen.set("goal")

    @reactive.effect
    @reactive.event(input.confirm_onboarding_btn)
    def _on_confirm_onboarding():
        if not state.ob_goal():
            ui.notification_show(
                "Pick what brought you here first.", type="warning", duration=4
            )
            state.ob_screen.set("goal")
            return
        _raw, profile = score_responses(list(state.ob_responses()))
        core_values = top_core_values(profile, max_values=MAX_CORE_VALUES)
        state.ob_output.set(
            {
                "weights": profile.weights,
                "top_values": profile.top_values,
                "goal_category": state.ob_goal(),
            }
        )
        state.ob_pending_core_values.set(tuple(core_values))
        state.ob_screen.set("complete")

    @reactive.effect
    @reactive.event(input.ob_start_first_entry_btn)
    def _on_ob_start_first_entry():
        state.ob_screen.set("first_entry")

    @reactive.effect
    @reactive.event(input.ob_save_first_entry_btn)
    def _on_ob_save_first_entry():
        text = (input.ob_first_entry_text() or "").strip()
        if not text:
            ui.notification_show("Write something first.", type="warning", duration=3)
            return
        first_entry = {
            "t_index": 0,
            "date": date.today().isoformat(),
            "initial_entry": text,
            "nudge_text": None,
            "response_text": None,
        }
        state.start_session(
            user_name=state.user_name(),
            core_values=state.ob_pending_core_values(),
            entries=(first_entry,),
        )

    @reactive.effect
    @reactive.event(input.load_demo_btn)
    def _load_demo_persona():
        catalog = _get_catalog_result()
        persona = get_persona(catalog.personas, input.demo_persona_id())
        if persona is None:
            ui.notification_show(
                "Select a demo persona first.", type="warning", duration=4
            )
            return
        core_values, entries = demo_journal_from_persona(persona)
        if not core_values:
            ui.notification_show(
                "This persona has no Core Values on record.", type="warning", duration=4
            )
            return
        state.start_session(
            user_name=persona.get("persona_name") or persona["persona_id"],
            core_values=tuple(core_values),
            entries=tuple(entries),
            demo_persona_id=persona["persona_id"],
        )
        ui.notification_show(
            f"Loaded {len(entries)} Journal Entries for {persona['persona_name']}.",
            type="message",
            duration=4,
        )

    @reactive.effect
    @reactive.event(input.reset_session_btn)
    def _reset_session():
        state.reset()

    # ── Journal handlers ─────────────────────────────────────────────────

    def _next_entry_date() -> date:
        entries = state.entries()
        if entries:
            last = date.fromisoformat(str(entries[-1]["date"]))
            return last + timedelta(days=1)
        return date.today()

    @ui.bind_task_button(button_id="save_entry_btn")
    @reactive.extended_task
    async def run_nudge_check_task(
        entry_text: str,
        entry_date_str: str,
        previous_entries: list[dict[str, Any]],
    ) -> str | None:
        llm_complete = build_llm_complete()
        if llm_complete is None:
            return None
        return await check_for_nudge(
            entry_text=entry_text,
            entry_date=entry_date_str,
            previous_entries=previous_entries,
            llm_complete=llm_complete,
        )

    @reactive.effect
    @reactive.event(input.save_entry_btn)
    def _on_save_entry():
        if state.pending_entry() is not None:
            return  # already mid-flow (nudge check running or awaiting reply)
        text = (input.entry_text() or "").strip()
        if not text:
            ui.notification_show("Write something first.", type="warning", duration=3)
            return
        entry_date = input.entry_date()
        if entry_date is None:
            ui.notification_show(
                "Pick a date for this entry.", type="warning", duration=3
            )
            return
        entries = state.entries()
        if entries and str(entry_date) < str(entries[-1]["date"]):
            ui.notification_show(
                "Journal Entries must stay in date order — pick a date on or after "
                f"{entries[-1]['date']}.",
                type="warning",
                duration=5,
            )
            return
        state.begin_entry_draft(str(entry_date), text)
        ui.update_text_area("entry_text", value="", session=session)
        ui.update_date("entry_date", value=_next_entry_date(), session=session)
        run_nudge_check_task.invoke(text, str(entry_date), [dict(e) for e in entries])

    @reactive.effect
    def _observe_nudge_check():
        status = run_nudge_check_task.status()
        if status == "success":
            nudge_text = run_nudge_check_task.result()
            if nudge_text:
                state.set_nudge_ready(nudge_text)
            else:
                state.set_no_nudge()
                state.finalize_pending_entry()
            return
        if status == "error":
            try:
                run_nudge_check_task.result()
            except Exception as exc:  # pragma: no cover - exercised via UI
                state.set_nudge_error(f"{type(exc).__name__}: {exc}")
            # Degrade gracefully: the entry is still saved without a nudge.
            state.finalize_pending_entry()

    @reactive.effect
    @reactive.event(input.nudge_reply_btn)
    def _on_nudge_reply():
        reply = (input.nudge_reply_text() or "").strip()
        if not reply:
            ui.notification_show("Write a reply first.", type="warning", duration=3)
            return
        state.finalize_pending_entry(response_text=reply)
        ui.update_text_area("nudge_reply_text", value="", session=session)

    @reactive.effect
    @reactive.event(input.nudge_skip_btn)
    def _on_nudge_skip():
        state.finalize_pending_entry(response_text=None)

    @reactive.effect
    @reactive.event(input.remove_entry_btn)
    def _remove_last_entry():
        entries = state.entries()
        if not entries:
            return
        state.entries.set(entries[:-1])

    # ── Weekly review run ────────────────────────────────────────────────

    @ui.bind_task_button(button_id="run_review_btn")
    @reactive.extended_task
    async def run_review_task(
        user_id: str,
        core_values: list[str],
        entries: list[dict[str, Any]],
        receipt_cache: dict[str, Any],
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            review_journal,
            user_id=user_id,
            core_values=core_values,
            entries=entries,
            receipt_cache=receipt_cache,
        )

    def _launch_review_run(entries: tuple[dict[str, Any], ...]) -> None:
        state.run_status.set("running")
        state.run_error.set(None)
        run_review_task.invoke(
            state.user_name() or "demo-user",
            list(state.core_values()),
            [dict(entry) for entry in entries],
            state.receipt_cache,
        )

    @reactive.effect
    @reactive.event(input.run_review_btn)
    def _start_review_run():
        """Only reachable via the Retry button, which run_status_banner
        renders solely in the "error" state — entries are guaranteed
        non-empty by the time that state is possible (see the auto-run
        effect below), so there's no separate empty-journal guard here."""
        if not os.environ.get("OPENAI_API_KEY"):
            state.set_error(
                "OPENAI_API_KEY is not configured — the weekly review can't "
                "run. Add it to .env and restart."
            )
            return
        _launch_review_run(state.entries())

    @reactive.effect
    def _auto_run_review_on_entries_change():
        """Keep Drift current automatically: every time the journal changes
        (a new entry saves, one is removed, a demo persona loads), re-run
        the weekly review — no button press needed. Recovery from a failure
        (e.g. a quota error, a missing key) is the Retry button that
        run_status_banner renders only in its "error" state.

        state.entries() is the only reactive dependency this effect should
        have. Everything else (run_status, user_name, core_values, read
        transitively via _launch_review_run) is wrapped in reactive.isolate()
        so this never re-fires on the run's own status transitions — that
        would otherwise re-trigger itself the moment a run completes.
        """
        entries = state.entries()
        if not entries:
            return
        with reactive.isolate():
            if state.run_status() == "running":
                return
            if not os.environ.get("OPENAI_API_KEY"):
                state.set_error(
                    "OPENAI_API_KEY is not configured — the weekly review "
                    "can't run. Add it to .env and restart, then use Retry."
                )
                return
            _launch_review_run(entries)

    @reactive.effect
    def _observe_review_task():
        status = run_review_task.status()
        if status == "running":
            state.run_status.set("running")
            return
        if status == "success":
            state.set_result(run_review_task.result())
            ui.notification_show(
                "Weekly review complete.", type="message", duration=4
            )
            return
        if status == "error":
            try:
                run_review_task.result()
            except Exception as exc:  # pragma: no cover - exercised via UI
                state.set_error(f"{type(exc).__name__}: {exc}")
                ui.notification_show(
                    f"Weekly review failed: {type(exc).__name__}",
                    type="error",
                    duration=6,
                )

    # ── Weekly Digest ─────────────────────────────────────────────────────

    @ui.bind_task_button(button_id="build_digest_btn")
    @reactive.extended_task
    async def build_digest_task(
        user_id: str,
        core_values: list[str],
        entries: list[dict[str, Any]],
        decisions: list[Any],
        drift_result: Any,
        week_start: str,
        week_end: str,
    ) -> dict[str, Any]:
        profile = {"name": user_id, "core_values": core_values}
        journal_entries = [
            {
                "date": entry["date"],
                "t_index": entry["t_index"],
                "initial_entry": entry.get("initial_entry", ""),
                "has_response": bool(entry.get("response_text")),
                "response_text": entry.get("response_text"),
            }
            for entry in entries
        ]
        digest = build_weekly_drift_reviewer_digest(
            persona_id=user_id,
            week_start=week_start,
            week_end=week_end,
            core_values=core_values,
            decisions=decisions,
            drift_result=drift_result,
            profile=profile,
            journal_entries=journal_entries,
        )
        llm_complete = build_llm_complete()
        if llm_complete is not None:
            narrative, _prompt = await generate_weekly_digest_coach(
                digest, llm_complete
            )
            validation = (
                validate_weekly_digest_narrative(digest, narrative)
                if narrative is not None
                else None
            )
            digest = attach_coach_artifacts(digest, narrative, validation)
        return {"digest": digest}

    def _launch_digest_build(outcome: dict[str, Any]) -> None:
        target_receipt = outcome["receipts"][-1]
        state.digest_status.set("running")
        state.digest_error.set(None)
        build_digest_task.invoke(
            state.user_name() or "demo-user",
            list(state.core_values()),
            [dict(entry) for entry in state.entries()],
            outcome["decisions"],
            outcome["drift"],
            target_receipt.week_start,
            outcome["drift"].cutoff_date,
        )

    @reactive.effect
    @reactive.event(input.build_digest_btn)
    def _on_build_digest():
        """The Refresh button in weekly_digest_panel — re-generate on demand
        (e.g. after the auto-run below already produced one, or to retry
        after a failure) without waiting for entries to change again."""
        outcome = state.review_outcome()
        if outcome is None or not outcome["receipts"]:
            ui.notification_show(
                "Nothing to reflect on yet — add a Journal Entry first.",
                type="warning",
                duration=4,
            )
            return
        _launch_digest_build(outcome)

    @reactive.effect
    def _auto_build_digest_on_outcome_change():
        """Keep the Weekly Digest current automatically: whenever the
        weekly review produces a new outcome, refresh the digest — no
        button press needed. The Refresh button above stays available for
        an on-demand re-run or to retry after a failure.

        state.review_outcome() is the only reactive dependency this effect
        should have; digest_status is read inside reactive.isolate() so this
        never re-fires on the digest task's own status transitions.
        """
        outcome = state.review_outcome()
        if outcome is None or not outcome["receipts"]:
            return
        with reactive.isolate():
            if state.digest_status() == "running":
                return
            _launch_digest_build(outcome)

    @reactive.effect
    def _observe_digest_task():
        status = build_digest_task.status()
        if status == "running":
            state.digest_status.set("running")
            return
        if status == "success":
            state.digest_status.set("success")
            state.digest_error.set(None)
            state.digest_result.set(build_digest_task.result()["digest"])
            return
        if status == "error":
            try:
                build_digest_task.result()
            except Exception as exc:  # pragma: no cover - exercised via UI
                state.digest_status.set("error")
                state.digest_error.set(f"{type(exc).__name__}: {exc}")
                ui.notification_show(
                    f"Weekly Digest failed: {type(exc).__name__}",
                    type="error",
                    duration=6,
                )

    # ── Derived review lookups ───────────────────────────────────────────

    @reactive.calc
    def decision_map() -> dict[tuple[int, str], Any]:
        outcome = state.review_outcome()
        if outcome is None:
            return {}
        return {
            (decision.t_index, decision.core_value): decision
            for decision in outcome["decisions"]
        }

    @reactive.calc
    def drift_span_map() -> dict[tuple[int, str], str]:
        """Map (t_index, core_value) inside a confirmed Drift to its state."""
        outcome = state.review_outcome()
        if outcome is None:
            return {}
        spans: dict[tuple[int, str], str] = {}
        for drift in outcome["drift"].drifts:
            for t_index in drift.supporting_t_indices:
                spans[(t_index, drift.core_value)] = drift.delivery_state
        return spans

    @reactive.calc
    def unreviewed_t_indices() -> list[int]:
        outcome = state.review_outcome()
        if outcome is None:
            return []
        reviewed = {decision.t_index for decision in outcome["decisions"]}
        return [
            int(entry["t_index"])
            for entry in state.entries()
            if int(entry["t_index"]) not in reviewed
        ]

    # ── Renderers ────────────────────────────────────────────────────────

    @render.ui
    def profile_summary():
        name = state.user_name()
        values = [_format_value(value) for value in state.core_values()]
        demo_note = (
            ui.div("Demo persona journal", class_="profile-demo-note")
            if state.demo_persona_id()
            else None
        )
        return ui.div(
            ui.div(name or "Anonymous", class_="profile-name"),
            ui.tags.label("Core Values", class_="field-label"),
            _build_badges(values, "badge-core", "None selected"),
            demo_note,
            class_="profile-block",
        )

    @render.ui
    def run_status_banner():
        status = state.run_status()
        if status == "running":
            return ui.div(
                ui.strong("Reading"),
                " — Twinkl is checking your journal against what matters to you.",
                class_="status-bar s-running",
            )
        if status == "success":
            if unreviewed_t_indices():
                return ui.div(
                    ui.strong("Catching up"),
                    " — Twinkl will check your newest entry in a moment.",
                    class_="status-bar s-warning",
                )
            return ui.div(
                ui.strong("Up to date"),
                " — Twinkl has read everything so far.",
                class_="status-bar s-success",
            )
        if status == "error":
            return ui.div(
                ui.div(
                    ui.strong("Couldn't check in"),
                    f" — {state.run_error() or 'Unknown error'}",
                ),
                ui.input_task_button(
                    "run_review_btn", "Retry", class_="btn-secondary"
                ),
                class_="status-bar s-error status-bar--with-action",
            )
        return ui.div(
            ui.strong("Ready"),
            " — write your first entry below.",
            class_="status-bar s-neutral",
        )

    @render.ui
    def entry_composer():
        pending = state.pending_entry()
        nudge_status = state.nudge_status()

        if pending is not None and nudge_status in ("checking", "ready"):
            return _render_pending_entry(
                pending, nudge_status, state.pending_nudge_text()
            )

        return ui.div(
            ui.h2("What happened today?", class_="composer-heading"),
            ui.div(
                ui.tags.label("Date", class_="field-label"),
                ui.input_date("entry_date", None, value=_next_entry_date()),
                class_="field",
            ),
            ui.input_text_area(
                "entry_text",
                None,
                placeholder="What did you actually do today?",
                rows=4,
                width="100%",
            ),
            ui.div(
                ui.input_task_button(
                    "save_entry_btn", "Save Journal Entry", class_="btn-primary"
                ),
                ui.input_action_button(
                    "remove_entry_btn", "Remove last", class_="btn-secondary"
                )
                if state.entries()
                else None,
                class_="composer-actions",
            ),
            class_="composer-card composer-card--hero",
        )

    @render.ui
    def drift_summary():
        """User-facing Drift state only — one ambient badge per Core Value,
        no rationale. The raw rationale (entry indices, timestamps) and
        evidence quotes live in internal_debug_panel() instead; a real user
        doesn't think in terms of entry numbers, just "is this still true
        for me"."""
        outcome = state.review_outcome()
        if outcome is None:
            return None
        drift = outcome["drift"]

        value_cards = []
        for core_value in state.core_values():
            value_state = drift.core_value_states.get(core_value, "stable")
            value_cards.append(
                ui.div(
                    ui.div(_format_value(core_value), class_="dimension-name"),
                    ui.span(
                        STATE_LABELS.get(value_state, value_state),
                        class_=f"state-tag state-{value_state}",
                    ),
                    class_="drift-state-card",
                )
            )

        return ui.div(
            ui.h3(
                "How things are trending",
                class_="section-heading section-heading--ambient",
            ),
            ui.div(*value_cards, class_="drift-state-grid"),
            class_="results-section results-section--ambient",
        )

    @render.ui
    def journal_list():
        entries = state.entries()
        if not entries:
            return ui.div(
                ui.h3("Journal", class_="section-heading"),
                ui.div("No Journal Entries yet.", class_="empty-state"),
                class_="journal-section",
            )

        decisions = decision_map()
        spans = drift_span_map()

        cards = []
        for entry in entries:
            t_index = int(entry["t_index"])
            verdict_chips = []
            evidence_blocks = []
            in_drift = False
            for core_value in state.core_values():
                if (t_index, core_value) in spans:
                    in_drift = True
                decision = decisions.get((t_index, core_value))
                if decision is None:
                    # Not reviewed yet (or the review is still catching up) —
                    # say nothing rather than showing a placeholder chip.
                    continue
                label = VERDICT_LABELS.get(decision.verdict, decision.verdict)
                chip_title = (
                    f"confidence: {decision.confidence or 'n/a'}"
                    f" · reason: {decision.reason_code or 'n/a'}"
                )
                badge_class = VERDICT_BADGES.get(decision.verdict, "badge-neutral")
                verdict_chips.append(
                    ui.span(
                        f"{_format_value(core_value)}: {label}",
                        class_=f"badge {badge_class}",
                        title=chip_title,
                    )
                )
                if decision.verdict == "conflict" and decision.evidence_quote.strip():
                    evidence_blocks.append(
                        ui.tags.blockquote(
                            decision.evidence_quote, class_="evidence-quote"
                        )
                    )

            nudge_text = entry.get("nudge_text")
            response_text = entry.get("response_text")
            thread_blocks = []
            if nudge_text:
                thread_blocks.append(
                    ui.div(
                        ui.strong("Twinkl asked"),
                        ui.p(nudge_text, class_="thread-text"),
                        class_="thread-block thread-nudge",
                    )
                )
            if response_text:
                thread_blocks.append(
                    ui.div(
                        ui.strong("Response"),
                        ui.p(response_text, class_="thread-text"),
                        class_="thread-block thread-response",
                    )
                )

            cards.append(
                ui.div(
                    ui.div(
                        ui.span(f"Entry {t_index + 1}", class_="entry-index"),
                        ui.span(str(entry["date"]), class_="entry-date"),
                        ui.span("Drift", class_="badge badge-alert")
                        if in_drift
                        else None,
                        class_="entry-meta",
                    ),
                    ui.p(str(entry.get("initial_entry", "")), class_="entry-text"),
                    ui.div(*thread_blocks, class_="thread-stack")
                    if thread_blocks
                    else None,
                    ui.div(*verdict_chips, class_="badge-row")
                    if verdict_chips
                    else None,
                    *evidence_blocks,
                    class_=f"timeline-card {'timeline-card-alert' if in_drift else ''}",
                )
            )

        return ui.div(
            ui.h3(f"Journal ({len(entries)} entries)", class_="section-heading"),
            ui.div(*cards, class_="timeline-stack"),
            class_="journal-section",
        )

    @render.ui
    def weekly_digest_panel():
        """Refreshes automatically whenever the weekly review updates (see
        _auto_build_digest_on_outcome_change); the button here is an
        on-demand Refresh, not a first trigger."""
        outcome = state.review_outcome()
        if outcome is None:
            return None

        status = state.digest_status()
        refresh_btn = ui.input_task_button(
            "build_digest_btn", "Refresh", class_="btn-secondary"
        )

        running_note = None
        if status == "running":
            running_note = ui.div(
                ui.strong("Reflecting"),
                " — putting together this week's reflection.",
                class_="status-bar s-running",
            )

        error_banner = None
        if status == "error":
            error_banner = ui.div(
                ui.strong("Couldn't build your reflection"),
                f" — {state.digest_error() or 'Unknown error'}",
                class_="status-bar s-error",
            )

        body = None
        digest = state.digest_result()
        if status == "success" and digest is not None:
            narrative = digest.coach_narrative
            if narrative is not None:
                narrative_block = ui.div(
                    ui.div(
                        ui.span("Weekly mirror", class_="subsection-label"),
                        ui.p(narrative.weekly_mirror, class_="entry-text"),
                    ),
                    ui.div(
                        ui.span("What's in tension", class_="subsection-label"),
                        ui.p(narrative.tension_explanation, class_="entry-text"),
                    ),
                    ui.div(
                        ui.span("A question to sit with", class_="subsection-label"),
                        ui.p(narrative.reflective_question, class_="drift-rationale"),
                    ),
                    class_="subsection-stack",
                )
            else:
                narrative_block = ui.div(
                    "Narrative unavailable — add OPENAI_API_KEY or "
                    "GEMINI_API_KEY for the Weekly Coach reflection. Showing "
                    "the Drift-based summary only.",
                    class_="empty-state",
                )

            evidence_blocks = [
                ui.tags.blockquote(
                    f"{snippet.date}: {snippet.excerpt}", class_="evidence-quote"
                )
                for snippet in digest.evidence
            ]

            body = ui.div(
                ui.p(digest.mode_rationale, class_="drift-rationale"),
                narrative_block,
                ui.div(
                    ui.span("Evidence", class_="subsection-label"),
                    *evidence_blocks,
                    class_="subsection-stack",
                )
                if evidence_blocks
                else None,
            )

        return ui.div(
            ui.div(
                ui.h3("Your Weekly Reflection", class_="section-heading"),
                refresh_btn,
                class_="section-heading-row",
            ),
            running_note,
            error_banner,
            body,
            class_="results-section",
        )

    @render.ui
    def internal_debug_panel():
        """QA-only: raw Drift rationale (entry indices, timestamps), evidence
        quotes, and the Weekly Drift Reviewer run audit trail. None of this
        is product surface — collapsed by default, clearly labeled, and
        isolated to this one function plus its one ui.output_ui() call in
        _build_journal_ui(), so it can be deleted cleanly later.
        """
        outcome = state.review_outcome()
        if outcome is None:
            return None
        drift = outcome["drift"]

        drift_cards = []
        for record in drift.drifts:
            quotes = [
                ui.tags.blockquote(quote, class_="evidence-quote")
                for quote in record.evidence_quotes
            ]
            recovery = None
            if record.termination_verdict == "not_conflict":
                recovery = f"Recovered on {record.termination_date}."
            elif record.termination_verdict == "abstain":
                recovery = (
                    f"Uncertain after {record.termination_date} — the Weekly Drift "
                    "Reviewer abstained."
                )
            drift_cards.append(
                ui.div(
                    ui.div(
                        ui.span(
                            _format_value(record.core_value), class_="dimension-name"
                        ),
                        ui.span(
                            STATE_LABELS.get(
                                record.delivery_state, record.delivery_state
                            ),
                            class_=f"state-tag state-{record.delivery_state}",
                        ),
                        class_="dimension-header",
                    ),
                    ui.p(
                        f"Conflicts from {record.onset_date} (entry "
                        f"{record.onset_t_index + 1}) confirmed Drift on "
                        f"{record.confirmation_date} (entry "
                        f"{record.confirmation_t_index + 1})."
                        + (f" {recovery}" if recovery else ""),
                        class_="drift-rationale",
                    ),
                    *quotes,
                    class_="dimension-card",
                )
            )

        records = []
        for receipt in outcome["receipts"]:
            conflicts = sum(
                1 for decision in receipt.decisions if decision.verdict == "conflict"
            )
            detail = ""
            if receipt.status == "error":
                detail = f"{receipt.error_type or 'Error'}: {receipt.error or ''}"
            elif receipt.status == "refusal":
                detail = receipt.refusal or "Model refused to respond."
            elif receipt.status == "invalid":
                detail = receipt.validation_error or "Response failed validation."
            records.append(
                {
                    "window": f"{receipt.week_start} to {receipt.week_end}",
                    "status": receipt.status,
                    "model": receipt.resolved_model or receipt.requested_model,
                    "decisions": len(receipt.decisions),
                    "conflicts": conflicts,
                    "detail": detail,
                }
            )

        return ui.div(
            ui.tags.details(
                ui.tags.summary("🔧 Internal / debug (QA only)"),
                ui.p("Drift detail", class_="subsection-label"),
                ui.div(*drift_cards, class_="dimension-grid")
                if drift_cards
                else ui.div("No Drift records yet.", class_="empty-state"),
                ui.p(
                    f"Weekly Drift Reviewer runs ({len(records)})",
                    class_="subsection-label",
                ),
                _build_table(
                    records,
                    [
                        ("window", "Week window"),
                        ("status", "Status"),
                        ("model", "Model"),
                        ("decisions", "Decisions"),
                        ("conflicts", "Conflicts"),
                        ("detail", "Detail"),
                    ],
                ),
                class_="receipts-details",
            ),
            class_="results-section",
        )


_app_dir = Path(__file__).parent

app_ui = ui.page_fluid(
    ui.head_content(ui.include_css(_app_dir / "static" / "styles.css")),
    ui.output_ui("app_content"),
)

app = App(
    app_ui,
    server,
    static_assets=_app_dir / "static",
)


if __name__ == "__main__":
    from shiny import run_app

    run_app(app, port=8001)
