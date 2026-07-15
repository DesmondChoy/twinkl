"""Read-only Shiny app for Weekly Drift Reviewer evidence inspection."""

from __future__ import annotations

import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

from shiny import App, Inputs, Outputs, Session, reactive, render, req, ui

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.drift_review_app.data import (  # noqa: E402
    CaseRecord,
    Decision,
    DriftSpan,
    EntryRecord,
    ReviewData,
    load_review_data,
    value_label,
)

_review_data: ReviewData | None = None
_RESET_VIEW_ONCLICK = (
    "window.__twinklPreviousStageHeading = "
    "document.getElementById('stage-heading'); "
    "window.__twinklFocusNextStageHeading = true; "
    "window.scrollTo({top: 0, behavior: 'instant'}); "
)
_STAGE_FOCUS_SCRIPT = """
(() => {
    const installObserver = () => {
        if (window.__twinklStageHeadingObserver) return;
        window.__twinklStageHeadingObserver = new MutationObserver(() => {
            if (!window.__twinklFocusNextStageHeading) return;
            const heading = document.getElementById("stage-heading");
            if (!heading || heading === window.__twinklPreviousStageHeading) return;
            heading.focus({preventScroll: true});
            window.__twinklFocusNextStageHeading = false;
            window.__twinklPreviousStageHeading = null;
        });
        window.__twinklStageHeadingObserver.observe(document.body, {
            childList: true,
            subtree: true,
        });
    };
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", installObserver, {once: true});
    } else {
        installObserver();
    }
})();
"""
CURRENT_SETUP_KEY = "luna_low"
ALL_CASES_FOCUS = "All persona/Core Value cases"
REVIEW_FOCUS_LABELS = {
    ALL_CASES_FOCUS: "All matching cases",
    "Missed Drift": "Missed known Drift",
    "False Drift alert": "False Drift alert",
    "Run disagreement": "Run variability",
    "Model disagreement": "Experiment setup disagreement",
    "Unresolved because of Abstain": "Unresolved because of Abstain",
    "Invalid response": "Invalid Weekly Drift Reviewer response",
    "Uncertain LLM-Judge Conflict Label": "Uncertain LLM-Judge Conflict Label",
}
ABSTAIN_EXPLANATIONS = {
    "ambiguous": (
        "The displayed text supports more than one reasonable interpretation."
    ),
    "direct_aligned_or_neutral_behavior": (
        "The Weekly Drift Reviewer identified aligned or neutral behavior but "
        "returned Abstain instead of a Not Conflict Decision."
    ),
    "direct_behavior_or_choice": (
        "The Journal Entry shows a behavior or choice, but the Weekly Drift "
        "Reviewer did not decide whether it conflicts with the Core Value."
    ),
    "external_constraint": (
        "The Journal Entry describes an external constraint, not a choice made "
        "by the writer."
    ),
    "feeling_or_intent_only": (
        "The Journal Entry describes a feeling or intention, not a completed "
        "behavior or choice."
    ),
    "missing_text": "The displayed Journal Entry is missing or incomplete.",
    "needs_hidden_context": (
        "A decision would require information not shown to the Weekly Drift Reviewer."
    ),
}
MODEL_LINKS = {
    "gpt-5.4-mini": "https://developers.openai.com/api/docs/models/gpt-5.4-mini",
    "gpt-5.6-luna": "https://developers.openai.com/api/docs/models/gpt-5.6-luna",
    "gpt-5.6-sol": "https://developers.openai.com/api/docs/models/gpt-5.6-sol",
    "claude-opus-4-8": (
        "https://platform.claude.com/docs/en/about-claude/models/overview"
    ),
}


def _get_review_data() -> ReviewData:
    global _review_data
    if _review_data is None:
        _review_data = load_review_data(ROOT)
    return _review_data


def _format_percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


def _format_signed(value: int | float | None, unit: str = "") -> str:
    if value is None:
        return "—"
    normalized_unit = unit.removesuffix("s") if unit and abs(value) == 1 else unit
    suffix = f" {normalized_unit}" if normalized_unit else ""
    return f"{value:+g}{suffix}"


def _verdict_label(decision: Decision) -> str:
    if decision.response_status != "ok":
        return "Invalid response"
    return {
        "conflict": "Conflict",
        "not_conflict": "Not Conflict",
        "abstain": "Abstain",
    }.get(decision.verdict or "", "Missing response")


def _verdict_class(decision: Decision) -> str:
    if decision.response_status != "ok":
        return "is-invalid"
    return {
        "conflict": "is-conflict",
        "not_conflict": "is-not-conflict",
        "abstain": "is-abstain",
    }.get(decision.verdict or "", "is-invalid")


def _pill(text: str, class_name: str = "") -> ui.Tag:
    return ui.span(text, class_=f"pill {class_name}".strip())


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    return singular if count == 1 else plural or f"{singular}s"


def _format_span(onset_t_index: int, end_t_index: int) -> str:
    return f"t{onset_t_index}–t{end_t_index}"


def _week_key(value: str) -> str:
    parsed = date.fromisoformat(value)
    return (parsed - timedelta(days=parsed.weekday())).isoformat()


def _matching_cases(
    data: ReviewData,
    drift_status: str,
    dimension: str,
    review_focus: str = ALL_CASES_FOCUS,
) -> tuple[CaseRecord, ...]:
    """Return cases for one Core Value, known Drift status, and review focus."""
    has_drift = drift_status == "has"
    focused_case_ids = data.queue_case_ids(review_focus, CURRENT_SETUP_KEY)
    return tuple(
        sorted(
            (
                case
                for case in data.cases.values()
                if case.dimension == dimension
                and bool(data.reference_drifts[case.case_id]) is has_drift
                and case.case_id in focused_case_ids
            ),
            key=lambda case: (
                str(data.profiles[case.persona_id]["name"]),
                case.persona_id,
            ),
        )
    )


def _dimension_choices(data: ReviewData, drift_status: str) -> dict[str, str]:
    has_drift = drift_status == "has"
    dimensions = sorted({case.dimension for case in data.cases.values()})
    choices = {}
    for dimension in dimensions:
        count = sum(
            bool(data.reference_drifts[case.case_id]) is has_drift
            for case in data.cases.values()
            if case.dimension == dimension
        )
        choices[dimension] = f"{value_label(dimension)} ({count})"
    return choices


def _review_focus_choices(data: ReviewData) -> dict[str, str]:
    return {
        queue: (
            f"{label} ({len(data.queue_case_ids(queue, CURRENT_SETUP_KEY))})"
            if queue != ALL_CASES_FOCUS
            else label
        )
        for queue, label in REVIEW_FOCUS_LABELS.items()
    }


def _period_choices(
    data: ReviewData, case: CaseRecord, setup_key: str
) -> dict[str, str]:
    choices = {"full": "Full timeline"}
    for row in data.reference_drifts[case.case_id]:
        crossing = " · cross-week" if row.crosses_week else ""
        choices[f"ref|{row.drift_id}|{row.onset_t_index}|{row.end_t_index}"] = (
            f"Known Drift · "
            f"{_format_span(row.onset_t_index, row.end_t_index)}{crossing}"
        )
    for run in (1, 2, 3):
        for row in data.predicted_drifts[(setup_key, run, case.case_id)]:
            choices[
                f"pred|{run}|{row.onset_t_index}|{row.end_t_index}|{row.detection_date}"
            ] = (
                f"Run {run} Drift alert · "
                f"{_format_span(row.onset_t_index, row.end_t_index)} · "
                f"detected {row.detection_date}"
            )
    for week in sorted({_week_key(entry.date) for entry in case.entries}):
        choices[f"week|{week}"] = f"Assessed in week of {week}"
    return choices


def _default_period(data: ReviewData, case: CaseRecord, setup_key: str) -> str:
    """Return the first known Drift or Drift alert available for review."""
    references = data.reference_drifts[case.case_id]
    if references:
        row = references[0]
        return f"ref|{row.drift_id}|{row.onset_t_index}|{row.end_t_index}"
    for run in (1, 2, 3):
        predictions = data.predicted_drifts[(setup_key, run, case.case_id)]
        if predictions:
            row = predictions[0]
            return (
                f"pred|{run}|{row.onset_t_index}|{row.end_t_index}|{row.detection_date}"
            )
    return "full"


def _visible_entries(case: CaseRecord, period: str) -> tuple[EntryRecord, ...]:
    if not period or period == "full":
        return case.entries
    parts = period.split("|")
    if parts[0] == "ref" and len(parts) >= 4:
        onset, end = int(parts[-2]), int(parts[-1])
        return tuple(entry for entry in case.entries if onset <= entry.t_index <= end)
    if parts[0] == "pred" and len(parts) >= 4:
        onset, end = int(parts[2]), int(parts[3])
        return tuple(entry for entry in case.entries if onset <= entry.t_index <= end)
    if parts[0] == "week" and len(parts) == 2:
        return tuple(
            entry for entry in case.entries if _week_key(entry.date) == parts[1]
        )
    return case.entries


def _reference_label(entry: EntryRecord) -> tuple[str, str]:
    if entry.final_conflict is True:
        return "Conflict", "is-conflict"
    if entry.final_conflict is False:
        return "Not Conflict", "is-not-conflict"
    return "Uncertain", "is-uncertain"


def _resolution_note(entry: EntryRecord) -> str | None:
    if entry.resolution_status == "resolved" and entry.resolution_method == "agreement":
        return None
    labels = {
        "adjudication": "Judge disagreement resolved by adjudication",
        "paired_agreement": "Paired review agreement",
        "opus_adjudication": "Earlier Uncertain label resolved by Claude Opus review",
        "paired_disagreement": "Judge disagreement remains Uncertain",
    }
    return labels.get(
        entry.resolution_method,
        "Non-routine label review",
    )


def _provenance_summary(case: CaseRecord) -> tuple[str, str]:
    training = (
        "Yes — included in historical VIF Critic training."
        if case.historical_split == "training"
        else "No — not included in historical VIF Critic training."
    )
    role = {
        "development_reference": "Development reference evidence",
        "development_only": "Development review only",
        "retired_audit_only": "Retired-case audit only",
    }.get(case.analysis_role, case.analysis_role.replace("_", " ").title())
    return training, role


def _reference_membership(
    data: ReviewData, case_id: str, t_index: int
) -> tuple[DriftSpan, ...]:
    return tuple(
        row
        for row in data.reference_drifts[case_id]
        if row.onset_t_index <= t_index <= row.end_t_index
    )


def _predicted_membership(
    data: ReviewData, setup_key: str, run: int, case_id: str, t_index: int
) -> tuple[DriftSpan, ...]:
    return tuple(
        row
        for row in data.predicted_drifts[(setup_key, run, case_id)]
        if row.onset_t_index <= t_index <= row.end_t_index
    )


def _entry_cell(entry: EntryRecord) -> ui.Tag:
    return ui.div(
        ui.div(
            ui.span(f"Journal Entry {entry.t_index}", class_="entry-index"),
            ui.tags.time(entry.date, datetime=entry.date),
            class_="entry-meta",
        ),
        ui.p(entry.initial_entry, class_="entry-copy"),
        ui.div(
            ui.div(
                ui.span("Displayed nudge", class_="thread-label"),
                ui.p(entry.nudge_text or "No displayed nudge"),
                class_="thread-part",
            ),
            ui.div(
                ui.span("Response", class_="thread-label"),
                ui.p(entry.response_text or "No response"),
                class_="thread-part",
            ),
            class_="entry-thread",
        ),
        class_="timeline-cell journal-cell",
        role="cell",
    )


def _reference_cell(data: ReviewData, case: CaseRecord, entry: EntryRecord) -> ui.Tag:
    label, label_class = _reference_label(entry)
    references = _reference_membership(data, case.case_id, entry.t_index)
    resolution_note = _resolution_note(entry)
    reviewer = (
        "claude-opus-4-8 · reasoning high"
        if entry.opus_resolved
        else "gpt-5.6-sol · reasoning xhigh"
    )
    return ui.div(
        ui.div(
            _pill(label, label_class),
            ui.span(f"t{entry.t_index}", class_="reference-index"),
            class_="reference-heading",
        ),
        ui.div(
            *[
                _pill(
                    f"Known Drift · "
                    f"{_format_span(row.onset_t_index, row.end_t_index)}"
                    + (" · cross-week" if row.crosses_week else ""),
                    "is-reference",
                )
                for row in references
            ],
            class_="pill-row",
        )
        if references
        else ui.p("Not inside known Drift", class_="reference-membership"),
        ui.p(reviewer, class_="reference-reviewer"),
        ui.p(resolution_note, class_="provenance-line") if resolution_note else None,
        class_="timeline-cell reference-cell",
        role="cell",
    )


def _decision_cell(
    data: ReviewData,
    case: CaseRecord,
    entry: EntryRecord,
    setup_key: str,
    run: int,
) -> ui.Tag:
    decision = data.decision(setup_key, run, case.case_id, entry.t_index)
    predictions = _predicted_membership(
        data, setup_key, run, case.case_id, entry.t_index
    )
    unresolved = any(
        entry.t_index in pair
        for pair in data.unresolved_pairs(setup_key, run, case.case_id)
    )
    abstain_explanation = ABSTAIN_EXPLANATIONS.get(
        decision.reason_code or "",
        "The displayed Journal Entry does not support a reliable Weekly Drift "
        "Reviewer Decision.",
    )
    return ui.div(
        ui.div(
            _pill(_verdict_label(decision), _verdict_class(decision)),
            class_="decision-heading",
        ),
        (
            ui.tags.details(
                ui.tags.summary(
                    "Why the Weekly Drift Reviewer abstained"
                    if decision.verdict == "abstain"
                    else "View evidence"
                ),
                ui.div(
                    ui.div(
                        ui.span(
                            "Why the Weekly Drift Reviewer abstained",
                            class_="field-name",
                        ),
                        ui.p(abstain_explanation),
                        class_="decision-field",
                    )
                    if decision.verdict == "abstain"
                    else None,
                    ui.div(
                        ui.span(
                            "Evidence from the Journal Entry",
                            class_="field-name",
                        ),
                        ui.tags.blockquote(decision.evidence_quote),
                        class_="decision-field",
                    )
                    if decision.evidence_quote
                    else None,
                    class_="decision-details-body",
                ),
                class_="decision-details",
            )
            if decision.verdict == "abstain" or decision.evidence_quote
            else None
        )
        if decision.response_status == "ok"
        else ui.div(
            ui.strong("Invalid or missing response"),
            ui.p(
                "The weekly receipt failed validation; parsed decisions were ignored."
            ),
            ui.tags.details(
                ui.tags.summary("Technical validation error"),
                ui.code(decision.validation_error or "No validation detail recorded."),
            ),
            class_="invalid-warning",
            role="alert",
        ),
        ui.div(
            *[
                _pill(
                    "Drift alert · "
                    + (
                        "matched known Drift"
                        if row.result == "hit"
                        else "false Drift alert"
                    ),
                    "is-hit" if row.result == "hit" else "is-false",
                )
                for row in predictions
            ],
            _pill("Adjacent pair unresolved because of Abstain", "is-abstain")
            if unresolved
            else None,
            class_="decision-flags",
        ),
        class_="timeline-cell run-cell",
        role="cell",
    )


def _format_indices(indices: tuple[int, ...]) -> str:
    if not indices:
        return "None"
    ranges: list[str] = []
    start = previous = indices[0]
    for value in indices[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(f"t{start}" if start == previous else f"t{start}–{previous}")
        start = previous = value
    ranges.append(f"t{start}" if start == previous else f"t{start}–{previous}")
    return ", ".join(ranges)


def _stage_trail(active: str) -> ui.Tag:
    labels = (
        ("filters", "1 · Filter"),
        ("personas", "2 · Personas"),
        ("detail", "3 · Journal Entries"),
    )
    return ui.tags.nav(
        *[
            ui.span(
                label,
                class_="is-active" if key == active else "",
                aria_current="step" if key == active else None,
            )
            for key, label in labels
        ],
        class_="stage-trail",
        aria_label="Review progress",
    )


def _filter_screen(
    data: ReviewData,
    *,
    selected_drift: str = "has",
    selected_dimension: str = "",
    selected_focus: str = ALL_CASES_FOCUS,
    error: str | None = None,
) -> ui.Tag:
    dimensions = _dimension_choices(data, selected_drift)
    return ui.tags.main(
        _stage_trail("filters"),
        ui.div(
            ui.div(
                ui.p("START A REVIEW", class_="eyebrow"),
                ui.h2(
                    "Choose the persona cases to inspect",
                    id="stage-heading",
                    tabindex="-1",
                ),
                ui.p(
                    "Filter by known Drift status and Core Value. Journal Entries "
                    "stay hidden until you choose a persona.",
                    class_="lede",
                ),
                class_="filter-intro",
            ),
            ui.div(
                ui.input_radio_buttons(
                    "reference_drift_filter",
                    "1. Known Drift status",
                    choices={
                        "has": "Has known Drift",
                        "none": "Has no known Drift",
                    },
                    selected=selected_drift,
                ),
                ui.input_select(
                    "core_value_filter",
                    "2. Core Value",
                    choices={"": "Choose a Core Value", **dimensions},
                    selected=selected_dimension,
                ),
                ui.tags.details(
                    ui.tags.summary("Advanced review focus"),
                    ui.input_select(
                        "review_focus_filter",
                        "Show cases with",
                        choices=_review_focus_choices(data),
                        selected=selected_focus,
                    ),
                    ui.p(
                        "Review focus uses the current development Weekly Drift "
                        "Reviewer: Luna at reasoning low. Counts cover all 292 cases; "
                        "the selected focus is then combined with the two filters "
                        "above.",
                        class_="controls-note",
                    ),
                    class_="filter-advanced",
                ),
                ui.p(error, class_="form-error", role="alert") if error else None,
                ui.input_action_button(
                    "show_personas",
                    "Show matching personas",
                    class_="primary-action",
                    onclick=_RESET_VIEW_ONCLICK,
                ),
                class_="filter-form",
            ),
            class_="filter-panel",
        ),
        _at_a_glance(data),
        _how_it_works_panel(),
        class_="stage-main filter-stage",
        id="main-content",
    )


def _inspect_id(case_id: str) -> str:
    return "inspect_" + case_id.replace(":", "__")


def _glance_heading(title: str, note: str) -> ui.Tag:
    return ui.div(
        ui.h3(title),
        ui.p(note),
        class_="glance-block-heading",
    )


def _core_value_breakdown(data: ReviewData) -> ui.Tag:
    dimensions = sorted(
        {case.dimension for case in data.cases.values()}, key=value_label
    )
    return ui.div(
        ui.tags.table(
            ui.tags.caption("Development data by Core Value"),
            ui.tags.thead(
                ui.tags.tr(
                    ui.tags.th("Core Value", scope="col"),
                    ui.tags.th("Cases", scope="col"),
                    ui.tags.th("With known Drift", scope="col"),
                    ui.tags.th("With no known Drift", scope="col"),
                    ui.tags.th("Known Drifts", scope="col"),
                )
            ),
            ui.tags.tbody(
                *[
                    ui.tags.tr(
                        ui.tags.th(value_label(dimension), scope="row"),
                        ui.tags.td(
                            str(
                                sum(
                                    case.dimension == dimension
                                    for case in data.cases.values()
                                )
                            )
                        ),
                        ui.tags.td(
                            str(
                                sum(
                                    case.dimension == dimension
                                    and bool(data.reference_drifts[case.case_id])
                                    for case in data.cases.values()
                                )
                            )
                        ),
                        ui.tags.td(
                            str(
                                sum(
                                    case.dimension == dimension
                                    and not data.reference_drifts[case.case_id]
                                    for case in data.cases.values()
                                )
                            )
                        ),
                        ui.tags.td(
                            str(
                                sum(
                                    len(data.reference_drifts[case.case_id])
                                    for case in data.cases.values()
                                    if case.dimension == dimension
                                )
                            )
                        ),
                    )
                    for dimension in dimensions
                ]
            ),
            class_="core-value-breakdown-table",
        ),
        class_="table-scroll",
    )


def _dataset_overview(data: ReviewData) -> ui.Tag:
    cases = tuple(data.cases.values())
    cases_with_drift = tuple(
        case for case in cases if data.reference_drifts[case.case_id]
    )
    personas_with_drift = {case.persona_id for case in cases_with_drift}
    reference_drifts = tuple(
        row for case in cases for row in data.reference_drifts[case.case_id]
    )
    label_counts = Counter(
        entry.final_conflict for case in cases for entry in case.entries
    )
    historical_training_drifts = sum(
        data.cases[row.case_id].historical_split == "training"
        for row in reference_drifts
    )
    cross_week_drifts = sum(row.crosses_week for row in reference_drifts)

    return ui.div(
        _glance_heading(
            "Dataset",
            "The same complete synthetic development data underlies every filter.",
        ),
        ui.div(
            ui.div(
                ui.span("PERSONAS", class_="field-name"),
                ui.strong(f"{len(data.profiles)} synthetic personas"),
                ui.div(
                    ui.span(
                        f"{len(personas_with_drift)} with at least one known Drift"
                    ),
                    ui.span(
                        f"{len(data.profiles) - len(personas_with_drift)} with none"
                    ),
                    class_="dataset-split",
                ),
                class_="dataset-stage",
            ),
            ui.div(
                "evaluated across their Core Values",
                class_="dataset-connector",
            ),
            ui.div(
                ui.span("REVIEW UNIT", class_="field-name"),
                ui.strong(f"{len(cases)} persona/Core Value cases"),
                ui.div(
                    ui.span(f"{len(cases_with_drift)} with known Drift"),
                    ui.span(
                        f"{len(cases) - len(cases_with_drift)} with no known Drift"
                    ),
                    class_="dataset-split",
                ),
                ui.p(
                    f"{len(reference_drifts)} known Drifts across the "
                    f"{len(cases_with_drift)} cases",
                    class_="dataset-drift-total",
                ),
                class_="dataset-stage",
            ),
            class_="dataset-map",
        ),
        ui.p(
            "Known Drift is derived from AI-reviewed LLM-Judge Conflict Labels. "
            "These counts are development evidence, not human validation or "
            "real-user prevalence.",
            class_="dataset-provenance",
        ),
        ui.tags.details(
            ui.tags.summary("More about the development data"),
            ui.div(
                ui.tags.dl(
                    ui.div(
                        ui.tags.dt("Journal Entries"),
                        ui.tags.dd(f"{int(data.integrity['entries']):,}"),
                    ),
                    ui.div(
                        ui.tags.dt("Persona-weeks"),
                        ui.tags.dd(f"{int(data.integrity['persona_weeks']):,}"),
                    ),
                    ui.div(
                        ui.tags.dt("Journal Entry/Core Value combinations"),
                        ui.tags.dd(f"{int(data.integrity['entry_value_cells']):,}"),
                    ),
                    ui.div(
                        ui.tags.dt("Known Drift timing"),
                        ui.tags.dd(
                            f"{cross_week_drifts} cross-week · "
                            f"{len(reference_drifts) - cross_week_drifts} same-week"
                        ),
                    ),
                    class_="dataset-facts",
                ),
                ui.p(
                    f"Across {int(data.integrity['entry_value_cells']):,} Journal "
                    f"Entry/Core Value combinations: {label_counts[True]:,} "
                    f"Conflict, {label_counts[False]:,} Not Conflict, and "
                    f"{label_counts[None]:,} Uncertain. All {len(cases)} case-level "
                    "Drift outcomes are resolved.",
                    class_="dataset-method-note",
                ),
                ui.p(
                    f"{historical_training_drifts} of {len(reference_drifts)} known "
                    "Drifts have historical VIF Critic training provenance. This "
                    "matters when interpreting VIF Critic results; the Weekly Drift "
                    "Reviewers shown here received no VIF Critic predictions.",
                    class_="dataset-method-note",
                ),
                _core_value_breakdown(data),
                class_="dataset-disclosure-body",
            ),
            class_="dataset-disclosure",
        ),
        class_="glance-block dataset-overview",
    )


def _model_link(name: str, model_key: str, description: str) -> ui.Tag:
    return ui.div(
        ui.a(
            name,
            ui.span(" ↗", aria_hidden="true"),
            href=MODEL_LINKS[model_key],
            target="_blank",
            rel="noopener noreferrer",
            aria_label=f"{name} official model page",
        ),
        ui.p(description),
        class_="model-entry",
    )


def _llm_overview() -> ui.Tag:
    return ui.div(
        _glance_heading(
            "LLMs used",
            "The Weekly Drift Reviewers and LLM-Judge review had separate roles.",
        ),
        ui.div(
            ui.div(
                ui.p("WEEKLY DRIFT REVIEWER", class_="model-role-title"),
                _model_link(
                    "GPT-5.4 mini",
                    "gpt-5.4-mini",
                    "Frozen baseline at reasoning none; a faster, efficient "
                    "OpenAI model for high-volume work.",
                ),
                _model_link(
                    "GPT-5.6 Luna",
                    "gpt-5.6-luna",
                    "Compared at reasoning none and low; an OpenAI model designed "
                    "for cost-sensitive, high-volume work.",
                ),
                class_="model-role",
            ),
            ui.div(
                ui.p("LLM-JUDGE CONFLICT LABEL REVIEW", class_="model-role-title"),
                _model_link(
                    "GPT-5.6 Sol",
                    "gpt-5.6-sol",
                    "Used in two isolated review lanes and disagreement-only "
                    "adjudication at reasoning xhigh; OpenAI's frontier GPT-5.6 "
                    "model.",
                ),
                _model_link(
                    "Claude Opus 4.8",
                    "claude-opus-4-8",
                    "Separately reviewed four earlier Uncertain LLM-Judge Conflict "
                    "Labels at reasoning high; Anthropic's model for complex "
                    "agentic and enterprise work.",
                ),
                class_="model-role",
            ),
            class_="model-roster",
        ),
        class_="glance-block llm-overview",
    )


def _aggregate_overview(data: ReviewData) -> ui.Tag:
    return ui.div(
        _glance_heading(
            "Results",
            "All three preserved Runs across the complete development data. No "
            "Core Value or known Drift filter has been applied.",
        ),
        ui.div(
            ui.tags.table(
                ui.tags.caption(
                    "Known Drift hits, false Drift alerts, and coverage for all "
                    "three preserved Runs"
                ),
                ui.tags.thead(
                    ui.tags.tr(
                        ui.tags.th("Weekly Drift Reviewer setup", scope="col"),
                        ui.tags.th("Run 1", scope="col"),
                        ui.tags.th("Run 2", scope="col"),
                        ui.tags.th("Run 3", scope="col"),
                    )
                ),
                ui.tags.tbody(
                    *[
                        ui.tags.tr(
                            ui.tags.th(
                                ui.div(
                                    ui.strong(spec.label),
                                    _pill(
                                        "Current development selection",
                                        "is-selected-value",
                                    )
                                    if setup_key == CURRENT_SETUP_KEY
                                    else None,
                                    class_="aggregate-setup-name",
                                ),
                                scope="row",
                            ),
                            *[
                                ui.tags.td(
                                    ui.strong(
                                        f"{int(row['drift_hits'])}/"
                                        f"{int(row['known_drifts'])} hits"
                                    ),
                                    ui.span(
                                        str(int(row["false_drift_alerts"]))
                                        + " false alerts",
                                    ),
                                    ui.span(
                                        f"{_format_percent(row['coverage'])} coverage"
                                    ),
                                )
                                for row in data.aggregate_results[setup_key]
                            ],
                        )
                        for setup_key, spec in data.setup_specs.items()
                    ]
                ),
                class_="aggregate-table",
            ),
            class_="table-scroll",
        ),
        ui.p(
            "Coverage is the share of cases where a Run either produced a Drift "
            "alert or actively ruled out every adjacent pair.",
            class_="overview-note",
        ),
        ui.tags.details(
            ui.tags.summary("Why Luna at reasoning low is selected for development"),
            ui.div(
                ui.p(
                    "It did not pass the earlier preregistered gate that allowed at "
                    "most 0.05 coverage loss. It was later selected under the "
                    "approved hierarchy: known Drift recall first, false Drift "
                    "alerts second, and coverage as a diagnostic."
                ),
                ui.p(
                    "Against Luna at reasoning none, the paired recall difference "
                    "was +0.071 with a 95% interval from −0.071 to +0.205. The false "
                    "Drift alert reduction and coverage loss were clearer. Because "
                    "the interval includes zero, the observed Drift recall gain is "
                    "not statistically clear. This is development evidence, not "
                    "deployment approval."
                ),
                class_="selection-disclosure-body",
            ),
            class_="selection-disclosure",
        ),
        class_="glance-block aggregate-overview",
        role="region",
        aria_label="Complete development results",
    )


def _at_a_glance(data: ReviewData) -> ui.Tag:
    return ui.tags.section(
        ui.div(
            ui.div(
                ui.p("COMPLETE SYNTHETIC DEVELOPMENT DATA", class_="eyebrow"),
                ui.h2("At a glance", id="at-a-glance-heading"),
            ),
            ui.p(
                "Dataset, LLM roles, and complete development results before any "
                "filter is applied.",
                class_="section-note",
            ),
            class_="section-heading at-a-glance-heading",
        ),
        _dataset_overview(data),
        _llm_overview(),
        _aggregate_overview(data),
        class_="at-a-glance",
        aria_labelledby="at-a-glance-heading",
    )


def _personas_screen(
    data: ReviewData,
    cases: tuple[CaseRecord, ...],
    drift_status: str,
    dimension: str,
    review_focus: str = ALL_CASES_FOCUS,
) -> ui.Tag:
    status_label = "Has known Drift" if drift_status == "has" else "Has no known Drift"
    return ui.tags.main(
        _stage_trail("personas"),
        ui.div(
            ui.input_action_button(
                "change_filters",
                "← Change filters",
                class_="text-action",
                onclick=_RESET_VIEW_ONCLICK,
            ),
            ui.h2(
                f"{value_label(dimension)} · {status_label}",
                id="stage-heading",
                tabindex="-1",
            ),
            ui.p(
                f"{len(cases)} matching {'persona' if len(cases) == 1 else 'personas'}",
                class_="result-count",
            ),
            ui.p(
                f"Review focus: {REVIEW_FOCUS_LABELS[review_focus]}",
                class_="result-focus",
            )
            if review_focus != ALL_CASES_FOCUS
            else None,
            class_="results-heading",
        ),
        ui.div(
            ui.p("MATCHING PERSONAS", class_="eyebrow"),
            ui.h2("Choose a persona to inspect"),
            class_="persona-list-heading",
        ),
        ui.div(
            *[
                ui.tags.article(
                    ui.div(
                        ui.h3(str(data.profiles[case.persona_id]["name"])),
                        ui.p(case.persona_id, class_="persona-id"),
                    ),
                    ui.div(
                        ui.span(f"{len(case.entries)} Journal Entries"),
                        ui.span(
                            f"{len(data.reference_drifts[case.case_id])} known "
                            + (
                                "Drift"
                                if len(data.reference_drifts[case.case_id]) == 1
                                else "Drifts"
                            )
                        ),
                        class_="result-facts",
                    ),
                    ui.input_action_button(
                        _inspect_id(case.case_id),
                        "Inspect Journal Entries",
                        class_="secondary-action",
                        onclick=_RESET_VIEW_ONCLICK,
                        aria_label=(
                            "Inspect Journal Entries for "
                            f"{data.profiles[case.persona_id]['name']}"
                        ),
                    ),
                    class_="persona-row",
                )
                for case in cases
            ],
            ui.div(
                ui.h3("No matching personas"),
                ui.p("Change the known Drift status, Core Value, or review focus."),
                class_="empty-state",
            )
            if not cases
            else None,
            class_="persona-list",
        ),
        class_="stage-main results-stage",
        id="main-content",
    )


def _how_it_works_panel() -> ui.Tag:
    return ui.tags.section(
        ui.div(
            ui.div(
                ui.p("FROZEN WEEKLY DRIFT REVIEWER INPUT CONTRACT", class_="eyebrow"),
                ui.h2("How it works"),
            ),
            ui.p(
                "This contract applies to every persona and review week. The exact "
                "Journal Entries change with each verified weekly cutoff.",
                class_="section-note",
            ),
            class_="section-heading how-it-works-heading",
        ),
        ui.div(
            ui.div(
                ui.h3("Shown to the Weekly Drift Reviewer"),
                ui.tags.ul(
                    ui.tags.li("All declared Core Values"),
                    ui.tags.li(
                        "Displayed Journal Entries through the review week, including "
                        "any nudge and response"
                    ),
                    ui.tags.li("Which Journal Entries belong to the current week"),
                ),
                class_="boundary-column is-shown",
            ),
            ui.div(
                ui.h3("Not shown"),
                ui.tags.ul(
                    ui.tags.li("Persona biography or demographics"),
                    ui.tags.li("Journal Entries after the review week"),
                    ui.tags.li("AI-reviewed LLM-Judge Conflict Labels or known Drift"),
                    ui.tags.li("VIF Critic predictions or other setup/Run decisions"),
                ),
                class_="boundary-column is-hidden",
            ),
            class_="boundary-grid",
        ),
        ui.div(
            ui.div(
                ui.span("WHAT COUNTS AS CONFLICT", class_="field-name"),
                ui.p(
                    "A Journal Entry is a Conflict when the displayed text clearly "
                    "shows the writer making a behavior or choice against a Core "
                    "Value."
                ),
                class_="contract-note",
            ),
            ui.div(
                ui.span("WHAT DOES NOT COUNT", class_="field-name"),
                ui.p(
                    "Feelings, guilt, wishes, intentions, external constraints, "
                    "biography, and ambiguous prose do not count as Conflict on "
                    "their own."
                ),
                class_="contract-note",
            ),
            class_="contract-notes",
        ),
        ui.div(
            ui.span("DRIFT RULE", class_="field-name"),
            ui.p(
                "Two consecutive Conflicts for the same Core Value form one Drift. "
                "A longer uninterrupted run is still one Drift."
            ),
            class_="drift-rule-note",
        ),
        ui.tags.details(
            ui.tags.summary("Known Drift labels and review timing"),
            ui.div(
                ui.p(
                    "LLM-Judge Conflict Labels were AI-reviewed with two isolated "
                    "gpt-5.6-sol lanes at reasoning xhigh and disagreement-only "
                    "adjudication. Four earlier Uncertain labels were separately "
                    "reviewed with claude-opus-4-8 at reasoning high. Known Drift "
                    "labels are derived from these labels; they are not ground truth "
                    "or human validation."
                ),
                ui.p(
                    "Abstain produces no Drift claim. t0 is the first Journal Entry. "
                    "A cross-week Drift spans two review weeks, so its second "
                    "Conflict is not assessed until the next weekly review. In "
                    "these development Runs, detection was about four days slower "
                    "than for same-week Drift."
                ),
                class_="contract-disclosure-body",
            ),
            class_="contract-disclosure",
        ),
        class_="how-it-works",
    )


def _cutoff_evidence(data: ReviewData, case: CaseRecord) -> ui.Tag:
    boundaries = data.boundaries_for_persona(case.persona_id)
    all_indices = tuple(entry.t_index for entry in case.entries)
    return ui.tags.details(
        ui.tags.summary("Verified Weekly Drift Reviewer input cutoffs"),
        ui.div(
            ui.tags.table(
                ui.tags.thead(
                    ui.tags.tr(
                        ui.tags.th("Review week"),
                        ui.tags.th("Visible history"),
                        ui.tags.th("Assessed now"),
                        ui.tags.th("Still hidden"),
                        ui.tags.th("Prompt SHA-256"),
                    )
                ),
                ui.tags.tbody(
                    *[
                        ui.tags.tr(
                            ui.tags.td(boundary.week_start),
                            ui.tags.td(_format_indices(boundary.visible_t_indices)),
                            ui.tags.td(_format_indices(boundary.current_t_indices)),
                            ui.tags.td(
                                _format_indices(
                                    tuple(
                                        index
                                        for index in all_indices
                                        if index > boundary.cutoff_t_index
                                    )
                                )
                            ),
                            ui.tags.td(ui.code(boundary.prompt_sha256[:12] + "…")),
                        )
                        for boundary in boundaries
                    ]
                ),
                class_="boundary-table",
            ),
            class_="table-scroll",
        ),
        ui.p(
            "For this persona, the loader verifies prompt text, weekly cutoffs, "
            "Journal Entry text, declared Core Values, and empty VIF Critic input. "
            "This demonstrates the intended inference-time boundary; it is not "
            "deployment approval.",
            class_="boundary-proof",
        ),
        class_="drawer cutoff-drawer",
    )


def _scoreboard_cell(
    data: ReviewData, case: CaseRecord, setup_key: str, run: int
) -> ui.Tag:
    metrics = data.case_metrics[(setup_key, run, case.case_id)]
    predictions = data.predicted_drifts[(setup_key, run, case.case_id)]
    reference_count = int(metrics["reference_drifts"])
    hits = int(metrics["drift_hits"])
    missed = int(metrics["missed_drifts"])
    false_alerts = int(metrics["false_drift_alerts"])
    invalid = int(metrics["invalid_responses"])
    covered = bool(metrics["covered"])

    if reference_count and missed:
        outcome = (
            f"Found {hits}/{reference_count} · missed {missed}"
            if hits
            else f"Missed {missed} {_plural(missed, 'Drift')}"
        )
        outcome_class = "is-missed"
    elif reference_count:
        outcome = f"Found {hits}/{reference_count} {_plural(reference_count, 'Drift')}"
        outcome_class = "is-hit"
    elif false_alerts:
        outcome = f"{false_alerts} false {_plural(false_alerts, 'Drift alert')}"
        outcome_class = "is-false"
    elif covered:
        outcome = "No Drift alert · all adjacent pairs ruled out"
        outcome_class = "is-hit"
    else:
        outcome = "No Drift alert · review incomplete"
        outcome_class = "is-abstain"

    return ui.tags.td(
        _pill(outcome, outcome_class),
        ui.div(
            *[
                ui.span(
                    f"{_format_span(row.onset_t_index, row.end_t_index)} · "
                    + (
                        "matched known Drift"
                        if row.result == "hit"
                        else "false Drift alert"
                    ),
                    class_=(
                        "score-span is-hit"
                        if row.result == "hit"
                        else "score-span is-false"
                    ),
                )
                for row in predictions
            ],
            ui.span("No Drift alert", class_="score-span") if not predictions else None,
            class_="score-spans",
        ),
        ui.p(
            f"{false_alerts} false {_plural(false_alerts, 'Drift alert')}"
            if false_alerts and reference_count
            else None,
            (
                f"{invalid} invalid Weekly Drift Reviewer "
                f"{_plural(invalid, 'response')}"
                if invalid
                else None
            ),
            class_="score-caveat",
        )
        if (false_alerts and reference_count) or invalid
        else None,
        ui.p(
            "At least one adjacent pair remains unresolved because the Weekly Drift "
            "Reviewer returned Abstain or no valid Weekly Drift Reviewer Decision.",
            class_="score-incomplete-note",
        )
        if not covered and not predictions
        else None,
        class_="scoreboard-cell",
    )


def _scoreboard(data: ReviewData, case: CaseRecord) -> ui.Tag:
    references = data.reference_drifts[case.case_id]
    reference_count = len(references)
    return ui.tags.section(
        ui.div(
            ui.div(
                ui.p("PERSONA SCOREBOARD", class_="eyebrow"),
                ui.h2("What each Weekly Drift Reviewer found", id="scoreboard-heading"),
            ),
            ui.p(
                "All experiment setups and Runs are compared against the same "
                "known outcome derived from AI-reviewed LLM-Judge Conflict Labels.",
                class_="section-note",
            ),
            class_="section-heading scoreboard-heading",
        ),
        ui.div(
            ui.div(
                ui.span("KNOWN OUTCOME", class_="field-name"),
                ui.strong(
                    f"{reference_count} known {_plural(reference_count, 'Drift')}"
                    if reference_count
                    else "No known Drift"
                ),
                class_="reference-outcome-label",
            ),
            ui.div(
                *[
                    _pill(
                        _format_span(row.onset_t_index, row.end_t_index)
                        + (" · cross-week" if row.crosses_week else ""),
                        "is-reference",
                    )
                    for row in references
                ],
                ui.span("No consecutive Conflict labels", class_="score-span")
                if not references
                else None,
                class_="reference-outcome-spans",
            ),
            class_="reference-outcome",
        ),
        ui.div(
            ui.tags.table(
                ui.tags.caption(
                    "Persona-level Drift outcome by Weekly Drift Reviewer setup and Run"
                ),
                ui.tags.thead(
                    ui.tags.tr(
                        ui.tags.th("Weekly Drift Reviewer setup", scope="col"),
                        ui.tags.th("Run 1", scope="col"),
                        ui.tags.th("Run 2", scope="col"),
                        ui.tags.th("Run 3", scope="col"),
                    )
                ),
                ui.tags.tbody(
                    *[
                        ui.tags.tr(
                            ui.tags.th(
                                ui.div(
                                    ui.strong(spec.label),
                                    _pill(
                                        "Current development selection",
                                        "is-selected-value",
                                    )
                                    if setup_key == CURRENT_SETUP_KEY
                                    else None,
                                    class_="score-setup-name",
                                ),
                                ui.span(spec.model, class_="score-model"),
                                scope="row",
                            ),
                            *[
                                _scoreboard_cell(data, case, setup_key, run)
                                for run in (1, 2, 3)
                            ],
                        )
                        for setup_key, spec in data.setup_specs.items()
                    ]
                ),
                class_="scoreboard-table",
            ),
            class_="table-scroll",
        ),
        ui.tags.details(
            ui.tags.summary("How hits are counted"),
            ui.p(
                "A Drift alert counts as a hit if confirmed between the known "
                "Drift's first Journal Entry and two Journal Entries after its end. "
                "This affects scoring, not the Drift definition."
            ),
            class_="matching-disclosure",
        )
        if reference_count
        else None,
        class_="scoreboard-panel",
        aria_labelledby="scoreboard-heading",
    )


def _detail_screen(data: ReviewData, case: CaseRecord) -> ui.Tag:
    profile = data.profiles[case.persona_id]
    training_provenance, evidence_role = _provenance_summary(case)
    period_choices = _period_choices(data, case, CURRENT_SETUP_KEY)
    default_period = _default_period(data, case, CURRENT_SETUP_KEY)
    return ui.tags.main(
        _stage_trail("detail"),
        ui.div(
            ui.input_action_button(
                "back_to_personas",
                "← Back to personas",
                class_="text-action",
                onclick=_RESET_VIEW_ONCLICK,
            ),
            ui.div(
                ui.div(
                    ui.p("PERSONA EVIDENCE", class_="eyebrow"),
                    ui.h2(
                        str(profile["name"]),
                        id="stage-heading",
                        tabindex="-1",
                    ),
                    ui.p(case.persona_id, class_="persona-id"),
                ),
                ui.div(
                    _pill(value_label(case.dimension), "is-selected-value"),
                    _pill(
                        "Has known Drift"
                        if data.reference_drifts[case.case_id]
                        else "Has no known Drift",
                        "is-reference" if data.reference_drifts[case.case_id] else "",
                    ),
                    class_="title-pills",
                ),
                class_="detail-title-row",
            ),
            class_="detail-heading",
        ),
        _scoreboard(data, case),
        ui.tags.section(
            ui.div(
                ui.div(
                    ui.input_radio_buttons(
                        "setup",
                        "Weekly Drift Reviewer setup",
                        choices={
                            key: (
                                f"{spec.label} · current development selection"
                                if key == CURRENT_SETUP_KEY
                                else spec.label
                            )
                            for key, spec in data.setup_specs.items()
                        },
                        selected=CURRENT_SETUP_KEY,
                        inline=True,
                    ),
                    class_="review-control setup-control",
                ),
                ui.div(
                    ui.input_select(
                        "period",
                        "Journal Entries shown",
                        choices=period_choices,
                        selected=default_period,
                    ),
                    class_="review-control period-control",
                ),
                class_="review-controls",
            ),
            ui.p(
                "Choose one setup for the Journal Entry comparison below; the "
                "scoreboard always compares all setups. The relevant Drift span "
                "is shown first. Choose Full timeline to inspect the complete "
                "history.",
                class_="controls-note",
            ),
            class_="controls-panel",
        ),
        ui.tags.section(
            ui.div(
                ui.div(
                    ui.p("PRIMARY REVIEW", class_="eyebrow"),
                    ui.h2("Journal Entries and three Runs"),
                ),
                ui.p(
                    "The LLM-Judge Conflict Label stays visible beside the same "
                    "Journal Entry. Runs 1–3 repeat the same frozen setup on the "
                    "same input and are never merged; disagreement shows Run "
                    "variability, not a data error.",
                    class_="section-note",
                ),
                class_="section-heading",
            ),
            ui.output_ui("timeline"),
            class_="timeline-section",
            id="timeline",
        ),
        ui.tags.details(
            ui.tags.summary("Supporting Drift and Run results"),
            ui.output_ui("supporting_results"),
            class_="drawer",
        ),
        _cutoff_evidence(data, case),
        ui.tags.details(
            ui.tags.summary(
                "Human-only persona context — not Weekly Drift Reviewer input"
            ),
            ui.div(
                ui.p(str(profile["bio"]), class_="persona-bio"),
                ui.tags.dl(
                    ui.div(
                        ui.tags.dt("Declared Core Values"),
                        ui.tags.dd(", ".join(profile["core_values"])),
                    ),
                    ui.div(
                        ui.tags.dt("VIF Critic training"),
                        ui.tags.dd(training_provenance),
                    ),
                    ui.div(
                        ui.tags.dt("Development evidence role"),
                        ui.tags.dd(evidence_role),
                    ),
                    ui.div(
                        ui.tags.dt("Source record"),
                        ui.tags.dd(
                            ui.tags.details(
                                ui.tags.summary("View source identifiers"),
                                ui.code(
                                    f"{case.historical_split} · "
                                    f"{case.cohort_source} · {case.cohort_role}"
                                ),
                                class_="source-record",
                            )
                        ),
                    ),
                    class_="persona-facts",
                ),
                class_="drawer-body",
            ),
            class_="drawer",
        ),
        class_="stage-main detail-stage",
        id="main-content",
    )


def _timeline(
    data: ReviewData,
    case: CaseRecord,
    setup_key: str,
    period: str,
) -> ui.Tag:
    entries = _visible_entries(case, period)
    if not entries:
        return ui.div(
            ui.h3("No Journal Entries in this period"),
            ui.p("Choose Full timeline or another Journal Entry period."),
            class_="empty-state",
        )
    return ui.div(
        ui.div(
            ui.div("Journal Entry", class_="timeline-header", role="columnheader"),
            ui.div(
                "LLM-Judge Conflict Label",
                class_="timeline-header",
                role="columnheader",
            ),
            ui.div("Run 1", class_="timeline-header", role="columnheader"),
            ui.div("Run 2", class_="timeline-header", role="columnheader"),
            ui.div("Run 3", class_="timeline-header", role="columnheader"),
            class_="timeline-head",
            role="row",
        ),
        *[
            ui.div(
                _entry_cell(entry),
                _reference_cell(data, case, entry),
                *[
                    _decision_cell(data, case, entry, setup_key, run)
                    for run in (1, 2, 3)
                ],
                class_="timeline-row",
                role="row",
            )
            for entry in entries
        ],
        class_="timeline-scroll",
        tabindex="0",
        role="table",
        aria_colcount="5",
        aria_label=(
            "Journal Entry, LLM-Judge Conflict Label, and Weekly Drift Reviewer "
            "Run comparison"
        ),
    )


def _supporting_results(data: ReviewData, case: CaseRecord, setup_key: str) -> ui.Tag:
    references = data.reference_drifts[case.case_id]
    tracks = [
        ui.div(
            ui.strong("Known Drift"),
            ui.div(
                *[
                    _pill(
                        _format_span(row.onset_t_index, row.end_t_index)
                        + (" · cross-week" if row.crosses_week else ""),
                        "is-reference",
                    )
                    for row in references
                ],
                ui.span("No known Drift") if not references else None,
                class_="track-spans",
            ),
            class_="drift-track",
        )
    ]
    for run in (1, 2, 3):
        rows = data.predicted_drifts[(setup_key, run, case.case_id)]
        tracks.append(
            ui.div(
                ui.strong(f"Run {run}"),
                ui.div(
                    *[
                        _pill(
                            f"{_format_span(row.onset_t_index, row.end_t_index)} · "
                            + (
                                "matched known Drift"
                                if row.result == "hit"
                                else "false Drift alert"
                            ),
                            "is-hit" if row.result == "hit" else "is-false",
                        )
                        for row in rows
                    ],
                    ui.span("No Drift alert") if not rows else None,
                    class_="track-spans",
                ),
                class_="drift-track",
            )
        )
    metric_rows = [
        data.case_metrics[(setup_key, run, case.case_id)] for run in (1, 2, 3)
    ]
    return ui.div(
        ui.div(*tracks, class_="drift-map"),
        ui.p(
            "Detection delay is the predicted detection date minus the known Drift "
            "confirmation date.",
            class_="supporting-note",
        ),
        ui.div(
            ui.tags.table(
                ui.tags.thead(
                    ui.tags.tr(
                        ui.tags.th("Run"),
                        ui.tags.th("Drift hits"),
                        ui.tags.th("Missed Drifts"),
                        ui.tags.th("False Drift alerts"),
                        ui.tags.th("Drift recall"),
                        ui.tags.th("Trajectory resolved"),
                        ui.tags.th("Detection delay"),
                    )
                ),
                ui.tags.tbody(
                    *[
                        ui.tags.tr(
                            ui.tags.td(f"Run {run}"),
                            ui.tags.td(str(metrics["drift_hits"])),
                            ui.tags.td(str(metrics["missed_drifts"])),
                            ui.tags.td(str(metrics["false_drift_alerts"])),
                            ui.tags.td(_format_percent(metrics["drift_recall"])),
                            ui.tags.td("Yes" if metrics["covered"] else "No"),
                            ui.tags.td(
                                _format_signed(metrics["median_delay_days"], "days")
                            ),
                        )
                        for run, metrics in enumerate(metric_rows, start=1)
                    ]
                ),
                class_="results-table",
            ),
            class_="table-scroll",
        ),
        class_="drawer-body supporting-results",
    )


def _app_shell() -> ui.Tag:
    return ui.div(
        ui.a("Skip to main content", href="#main-content", class_="skip-link"),
        ui.tags.header(
            ui.div(
                ui.p(
                    "READ-ONLY EVIDENCE INSPECTION · FROZEN DEVELOPMENT INPUTS",
                    class_="eyebrow",
                ),
                ui.h1("Drift inspection app"),
            ),
            ui.div(
                _pill("Read only"),
                _pill("No provider calls"),
                class_="header-facts",
            ),
            class_="app-header",
        ),
        ui.output_ui("screen"),
        ui.tags.footer(
            "AI-reviewed synthetic development evidence · not human validation · "
            "not a fresh final test · no deployment approval",
            class_="app-footer",
        ),
        ui.tags.script(_STAGE_FOCUS_SCRIPT),
        class_="app-shell",
    )


app_ui = ui.page_fluid(
    ui.head_content(ui.tags.link(rel="stylesheet", href="styles.css")),
    _app_shell(),
    title="Drift inspection app",
)


def _server(input: Inputs, output: Outputs, session: Session) -> None:
    data = _get_review_data()
    stage: reactive.Value[str] = reactive.value("filters")
    selected_case_id: reactive.Value[str | None] = reactive.value(None)
    selected_drift: reactive.Value[str] = reactive.value("has")
    selected_dimension: reactive.Value[str] = reactive.value("")
    selected_focus: reactive.Value[str] = reactive.value(ALL_CASES_FOCUS)
    filter_error: reactive.Value[str | None] = reactive.value(None)

    def matching_cases() -> tuple[CaseRecord, ...]:
        dimension = selected_dimension.get()
        if not dimension:
            return ()
        return _matching_cases(
            data,
            selected_drift.get(),
            dimension,
            selected_focus.get(),
        )

    @render.ui
    def screen() -> ui.Tag:
        current = stage.get()
        if current == "filters":
            return _filter_screen(
                data,
                selected_drift=selected_drift.get(),
                selected_dimension=selected_dimension.get(),
                selected_focus=selected_focus.get(),
                error=filter_error.get(),
            )
        if current == "personas":
            return _personas_screen(
                data,
                matching_cases(),
                selected_drift.get(),
                selected_dimension.get(),
                selected_focus.get(),
            )
        case_id = selected_case_id.get()
        req(case_id)
        assert case_id is not None
        return _detail_screen(data, data.cases[case_id])

    @reactive.effect
    @reactive.event(input.show_personas)
    def _show_personas() -> None:
        dimension = str(input.core_value_filter() or "")
        selected_drift.set(str(input.reference_drift_filter() or "has"))
        selected_focus.set(str(input.review_focus_filter() or ALL_CASES_FOCUS))
        if not dimension:
            filter_error.set("Choose a Core Value to continue.")
            return
        selected_dimension.set(dimension)
        filter_error.set(None)
        stage.set("personas")

    @reactive.effect
    @reactive.event(input.reference_drift_filter)
    def _sync_core_value_counts() -> None:
        req(stage.get() == "filters")
        drift_status = str(input.reference_drift_filter() or "has")
        current = str(input.core_value_filter() or "")
        choices = {"": "Choose a Core Value", **_dimension_choices(data, drift_status)}
        ui.update_select(
            "core_value_filter",
            choices=choices,
            selected=current if current in choices else "",
        )

    @reactive.effect
    @reactive.event(input.change_filters)
    def _change_filters() -> None:
        selected_case_id.set(None)
        stage.set("filters")

    @reactive.effect
    @reactive.event(input.back_to_personas)
    def _back_to_personas() -> None:
        selected_case_id.set(None)
        stage.set("personas")

    for case_id in sorted(data.cases):

        @reactive.effect
        @reactive.event(getattr(input, _inspect_id(case_id)))
        def _inspect(case_id: str = case_id) -> None:
            selected_case_id.set(case_id)
            stage.set("detail")

    @reactive.calc
    def selected_case() -> CaseRecord:
        case_id = selected_case_id.get()
        req(case_id)
        assert case_id is not None
        return data.cases[case_id]

    @reactive.effect
    def _sync_periods() -> None:
        req(stage.get() == "detail", input.setup())
        case = selected_case()
        choices = _period_choices(data, case, str(input.setup()))
        current = str(input.period() or "full")
        ui.update_select(
            "period",
            choices=choices,
            selected=(
                current
                if current in choices
                else _default_period(data, case, str(input.setup()))
            ),
        )

    @render.ui
    def timeline() -> ui.Tag:
        req(stage.get() == "detail", input.setup())
        return _timeline(
            data,
            selected_case(),
            str(input.setup()),
            str(input.period() or "full"),
        )

    @render.ui
    def supporting_results() -> ui.Tag:
        req(stage.get() == "detail", input.setup())
        return _supporting_results(data, selected_case(), str(input.setup()))


app = App(app_ui, _server, static_assets=Path(__file__).parent / "static")
