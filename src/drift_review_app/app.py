"""Read-only Shiny app for Weekly Drift Reviewer evidence inspection."""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any, cast, get_args

from htmltools import Tag
from shiny import App, Inputs, Outputs, Session, reactive, render, req, ui

ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).parent / "static"
STYLESHEET_VERSION = sha256((STATIC_DIR / "styles.css").read_bytes()).hexdigest()[:12]
EXPLAINER_SCRIPT_VERSION = sha256(
    (STATIC_DIR / "drift_explainer.js").read_bytes()
).hexdigest()[:12]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.drift_review_app.data import (  # noqa: E402
    CURRENT_SETUP_KEY,
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
from src.drift_review_app.exploration import (  # noqa: E402
    ExplorationQuery,
    ExplorationResult,
    QueryKind,
    ReferenceStatus,
    explore,
)
from src.drift_review_app.report_figures import report_figure  # noqa: E402

PROMPT_DEFAULTS = {
    "v0": "luna_low",
    "v1": "definitions_off_low",
    "v2": CURRENT_SETUP_KEY,
}
PROMPT_CHOICES = {
    "v0": "v0 · Initial evidence rules",
    "v1": "v1 · Confidence and evidence rules",
    "v2": "v2 · Core Value definitions and motivation",
}
PROMPT_NOTES = {
    "v0": (
        "The initial evidence rules",
        "Assess current-week Journal Entries against declared Core Values using "
        "cumulative history. Count clear behavior or choices as Conflict and cite "
        "exact text.",
        "Inspect the original preserved Decisions on the same synthetic "
        "Journal Entry histories.",
    ),
    "v1": (
        "Confidence and evidence rules",
        "Require a direct behavior or choice with at least medium confidence for "
        "Conflict. Low-confidence cases become Abstain; Not Conflict does not "
        "prove improvement. Add confidence, reason, and exact-quote rules.",
        "The effect of these changes against v0 was not measured separately. "
        "This version is the baseline for the Core Value definition study in "
        "Section 4.3.3.",
    ),
    "v2": (
        "Core Value definitions and motivation",
        "Keep v1's evidence rules and add each selected Core Value's existing "
        "definition and core motivation. This is the current prompt contract.",
        "Inspect Decisions made with explicit Core Value context. The separate "
        "definition study preserves detection changes rather than an accuracy "
        "evaluation.",
    ),
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


def _dimension_choices(data: ReviewData, drift_status: str) -> dict[str, str]:
    dimensions = sorted({case.dimension for case in data.cases.values()})
    choices = {}
    for dimension in dimensions:
        count = sum(
            (
                drift_status == "all"
                or bool(data.reference_drifts[case.case_id]) is (drift_status == "has")
            )
            for case in data.cases.values()
            if case.dimension == dimension
        )
        choices[dimension] = f"{value_label(dimension)} ({count})"
    return choices


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


def _default_period(_data: ReviewData, _case: CaseRecord, _setup_key: str) -> str:
    """Open on the complete chronology; narrower evidence periods remain available."""
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


def _predicted_drift_pill(data: ReviewData, setup_key: str, span: DriftSpan) -> ui.Tag:
    if not data.setup_specs[setup_key].accuracy_evaluated:
        return _pill(
            "Detected Drift · " + _format_span(span.onset_t_index, span.end_t_index)
        )
    return _pill(
        "Drift alert · "
        + ("matched known Drift" if span.result == "hit" else "false Drift alert"),
        "is-hit" if span.result == "hit" else "is-false",
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
            *[_predicted_drift_pill(data, setup_key, row) for row in predictions],
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
        ("contributors", "2 · Contributors"),
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


def _query_onclick(kind: str, dimension: str | None = None) -> str:
    payload = json.dumps({"kind": kind, "dimension": dimension})
    return (
        "Shiny.setInputValue('inspect_query', "
        + payload
        + ", {priority: 'event'}); "
        + _RESET_VIEW_ONCLICK
    )


def _count_button(
    result: ExplorationResult,
    kind: str,
    dimension: str | None = None,
    *,
    label: str | None = None,
    prominent: bool = False,
) -> ui.Tag:
    text = label or result.title
    return ui.tags.button(
        ui.strong(f"{result.count:,}"),
        ui.span(text) if prominent else None,
        type="button",
        class_="contributor-count is-prominent" if prominent else "contributor-count",
        onclick=_query_onclick(kind, dimension),
        aria_label=f"Inspect {result.count:,} {text} · Run {result.query.run}",
        **{
            "data-exploration-kind": kind,
            "data-exploration-dimension": dimension or "",
            "data-exploration-run": str(result.query.run),
        },
    )


def _scoped_result(
    data: ReviewData, query: ExplorationQuery, kind: str, dimension: str | None = None
) -> ExplorationResult:
    return explore(
        data,
        ExplorationQuery(
            kind=cast(QueryKind, kind),
            setup_key=query.setup_key,
            run=query.run,
            reference_status=query.reference_status,
            dimension=query.dimension if dimension is None else dimension,
        ),
    )


def _data_breakdowns(data: ReviewData, query: ExplorationQuery) -> ui.Tag:
    spec = data.setup_specs[query.setup_key]
    corpus = (
        ("corpus_personas", "Personas"),
        ("corpus_entries", "Journal Entries"),
        ("cases", "Persona/Core Value histories"),
        ("reference_drifts", "Known Drifts"),
    )
    outcome_kinds = [
        ("detected", "Detected Drifts"),
        *(
            (
                ("found", "Known Drifts found"),
                ("missed", "Known Drifts missed"),
                ("false", "False Drift alerts"),
            )
            if spec.accuracy_evaluated
            else ()
        ),
        ("covered", "Covered histories"),
        ("unresolved", "Unresolved histories"),
        ("invalid", "Decisions affected by invalid reviews"),
    ]
    return ui.div(
        ui.div(
            ui.h2("Explore the contributing data"),
            ui.p(
                "Select a number to see every contributing Persona, Journal "
                "Entry, history, or Drift. Counts follow the filters above.",
                class_="section-note",
            ),
            class_="exploration-heading",
        ),
        ui.div(
            *[
                _count_button(
                    _scoped_result(data, query, kind),
                    kind,
                    label=label,
                    prominent=True,
                )
                for kind, label in corpus
            ],
            class_="corpus-counts",
            aria_label="Corpus counts and contributors",
        ),
        ui.div(
            ui.div(
                ui.h3("Known Drift breakdown"),
                ui.tags.dl(
                    *[
                        ui.div(
                            ui.tags.dt(label),
                            ui.tags.dd(
                                _count_button(
                                    _scoped_result(data, query, kind), kind, label=label
                                )
                            ),
                        )
                        for kind, label in (
                            ("known_cases", "Histories with known Drift"),
                            ("no_known_cases", "Histories with no known Drift"),
                            ("reference_same_week", "Same-week known Drifts"),
                            ("reference_cross_week", "Cross-week known Drifts"),
                        )
                    ],
                    class_="count-breakdown",
                ),
            ),
            ui.div(
                ui.h3(f"{spec.label} · Run {query.run}"),
                ui.p(
                    "One preserved Run. Outcomes below link to its contributing "
                    "Drifts, histories, or Journal Entry/Core Value Decisions.",
                    class_="overview-note",
                ),
                ui.tags.dl(
                    *[
                        ui.div(
                            ui.tags.dt(label),
                            ui.tags.dd(
                                _count_button(
                                    _scoped_result(data, query, kind), kind, label=label
                                )
                            ),
                        )
                        for kind, label in outcome_kinds
                    ],
                    class_="count-breakdown",
                ),
                ui.p(
                    "The Core Value definition study reported detection changes. "
                    "These reference comparisons are descriptive replay; it "
                    "did not establish accuracy."
                )
                if not spec.accuracy_evaluated
                else None,
            ),
            class_="exploration-breakdowns",
        ),
        ui.tags.details(
            ui.tags.summary("Explore Core Values and entry-level labels"),
            ui.div(
                ui.tags.table(
                    ui.tags.caption("Contributors by Core Value"),
                    ui.tags.thead(
                        ui.tags.tr(
                            *[
                                ui.tags.th(label, scope="col")
                                for label in (
                                    "Core Value",
                                    "Histories",
                                    "With known Drift",
                                    "With no known Drift",
                                    "Known Drifts",
                                    "Reference Conflict labels",
                                )
                            ]
                        )
                    ),
                    ui.tags.tbody(
                        *[
                            ui.tags.tr(
                                ui.tags.th(value_label(dimension), scope="row"),
                                *[
                                    ui.tags.td(
                                        _count_button(
                                            _scoped_result(
                                                data, query, kind, dimension
                                            ),
                                            kind,
                                            dimension,
                                        )
                                    )
                                    for kind in (
                                        "cases",
                                        "known_cases",
                                        "no_known_cases",
                                        "reference_drifts",
                                        "reference_conflict",
                                    )
                                ],
                            )
                            for dimension in sorted(
                                {case.dimension for case in data.cases.values()}
                            )
                            if not query.dimension
                            or query.dimension == "all"
                            or query.dimension == dimension
                        ]
                    ),
                    class_="results-table contributor-breakdown-table",
                ),
                class_="table-scroll",
            ),
            ui.div(
                *[
                    ui.div(
                        ui.span(label),
                        _count_button(_scoped_result(data, query, kind), kind),
                    )
                    for kind, label in (
                        ("core_values", "Core Values"),
                        ("entry_value_cells", "Journal Entry/Core Value combinations"),
                        ("reference_conflict", "Reference Conflict labels"),
                        ("reference_not_conflict", "Reference Not Conflict labels"),
                        ("uncertain", "Uncertain reference labels"),
                        ("conflict", "Run Conflict Decisions"),
                        ("not_conflict", "Run Not Conflict Decisions"),
                        ("abstain", "Run Abstain Decisions"),
                    )
                ],
                class_="entry-counts",
            ),
            class_="exploration-detail-breakdown",
        ),
        ui.p(
            "Known Drift comes from consecutive AI-reviewed LLM-Judge Conflict "
            "Labels. It is synthetic development evidence. Coverage means a "
            "Run detected Drift or ruled out every adjacent pair; Abstain can "
            "leave a history unresolved.",
            class_="exploration-provenance",
        ),
        class_="data-exploration",
    )


def _prompt_explanation(prompt_version: str) -> ui.Tag:
    title, change, context = PROMPT_NOTES[prompt_version]
    return ui.tags.details(
        ui.tags.summary(f"Prompt {prompt_version}: {title} · what changed"),
        ui.p(change),
        ui.p(context),
        ui.p(
            "The synthetic corpus, weekly cutoffs, cumulative Journal Entry "
            "history, and deterministic Drift rules stay fixed. Report versions "
            "v0 / v1 / v2 correspond to archived prompt versions 2.0 / 3.0 / 4.0."
        ),
        class_="prompt-explanation",
    )


def _filter_screen(
    data: ReviewData,
    *,
    selected_drift: str = "all",
    selected_dimension: str = "",
    selected_prompt: str = "v2",
    selected_setup: str = CURRENT_SETUP_KEY,
    selected_run: int = 1,
    error: str | None = None,
) -> ui.Tag:
    return ui.tags.main(
        ui.div(
            ui.h2(
                "Inspect the synthetic data and Drift evidence",
                id="stage-heading",
                tabindex="-1",
            ),
            ui.p(
                "Choose a preserved experiment and Run, then open any count "
                "to inspect the Journal Entries and Decisions behind it.",
                class_="report-lede",
            ),
            class_="inspection-intro",
        ),
        ui.tags.section(
            ui.div(
                ui.input_select(
                    "report_prompt",
                    "Report prompt version",
                    choices=PROMPT_CHOICES,
                    selected=selected_prompt,
                ),
                ui.input_select(
                    "landing_setup",
                    "Preserved experiment",
                    choices={
                        key: spec.label
                        for key, spec in data.setups_for_prompt(selected_prompt).items()
                    },
                    selected=selected_setup,
                ),
                ui.input_select(
                    "landing_run",
                    "Run",
                    choices={str(run): f"Run {run}" for run in (1, 2, 3)},
                    selected=str(selected_run),
                ),
                class_="experiment-filter-row",
            ),
            ui.div(
                ui.input_select(
                    "reference_drift_filter",
                    "Known Drift status",
                    choices={
                        "all": "All histories",
                        "has": "Has known Drift",
                        "none": "Has no known Drift",
                    },
                    selected=selected_drift,
                ),
                ui.input_select(
                    "core_value_filter",
                    "Core Value",
                    choices={
                        "": "All Core Values",
                        **_dimension_choices(data, selected_drift),
                    },
                    selected=selected_dimension,
                ),
                ui.input_action_button(
                    "show_personas",
                    "Inspect matching histories",
                    class_="primary-action",
                    onclick=_RESET_VIEW_ONCLICK,
                ),
                class_="case-filter-row",
            ),
            ui.p(error, class_="form-error", role="alert") if error else None,
            ui.output_ui("prompt_explanation"),
            class_="inspection-filters",
        ),
        ui.output_ui("data_breakdowns"),
        ui.tags.details(
            ui.tags.summary("How the review uses Journal Entries and reference labels"),
            ui.div(
                _how_it_works_panel(),
                report_figure("matching"),
                class_="evidence-library-body",
            ),
            class_="evidence-library",
        ),
        class_="stage-main filter-stage",
        id="main-content",
    )


def _contributor_onclick(index: int, case_id: str) -> str:
    payload = json.dumps({"index": index, "case_id": case_id})
    return (
        "Shiny.setInputValue('inspect_contributor', "
        + payload
        + ", {priority: 'event'}); "
        + _RESET_VIEW_ONCLICK
    )


def _contributor_screen(
    data: ReviewData,
    query: ExplorationQuery,
    result: ExplorationResult,
    page: int = 0,
) -> ui.Tag:
    page_size = 50
    start = page * page_size
    stop = min(start + page_size, result.count)
    spec = data.setup_specs[query.setup_key]
    articles = []
    for index, row in enumerate(result.rows[start:stop], start=start):
        case_ids = row.related_case_ids or (row.case_id,)
        if query.kind == "core_values":
            case_ids = ()
        case = data.cases.get(row.case_id)
        entry = (
            next(
                (entry for entry in case.entries if entry.t_index == row.t_index), None
            )
            if case
            else None
        )
        articles.append(
            ui.tags.article(
                ui.div(
                    ui.h3(
                        value_label(row.dimension)
                        if query.kind == "core_values"
                        else str(data.profiles[row.persona_id]["name"])
                        if row.persona_id
                        else row.label
                    ),
                    ui.p(row.label, class_="contributor-unit-detail")
                    if query.kind != "core_values"
                    else None,
                    ui.p(
                        row.start_date
                        + (
                            f" to {row.end_date}"
                            if row.end_date != row.start_date
                            else ""
                        ),
                        class_="contributor-date",
                    )
                    if row.start_date and query.kind != "core_values"
                    else None,
                    _pill(row.status)
                    if row.status and query.kind != "core_values"
                    else None,
                    ui.p(
                        f"{row.persona_id} · "
                        + ", ".join(
                            value_label(data.cases[case_id].dimension)
                            for case_id in case_ids
                            if case_id in data.cases
                        ),
                        class_="persona-id",
                    )
                    if row.persona_id and query.kind != "core_values"
                    else None,
                    ui.p(
                        entry.runtime_text[:320]
                        + ("…" if len(entry.runtime_text) > 320 else ""),
                        class_="contributor-entry",
                    )
                    if entry and query.kind != "core_values"
                    else None,
                ),
                ui.div(
                    *[
                        ui.tags.button(
                            f"Inspect {value_label(data.cases[case_id].dimension)} "
                            "Journal Entries",
                            type="button",
                            class_="secondary-action",
                            onclick=_contributor_onclick(index, case_id),
                            **{
                                "data-inspect-case-id": case_id,
                                "data-inspect-index": str(index),
                            },
                        )
                        for case_id in case_ids
                        if case_id in data.cases
                    ],
                    ui.tags.button(
                        "Inspect contributing histories",
                        type="button",
                        class_="secondary-action",
                        onclick=_query_onclick("cases", row.dimension),
                    )
                    if query.kind == "core_values"
                    else None,
                    class_="contributor-actions",
                ),
                class_="contributor-row",
                **{
                    "data-contributor-index": str(index),
                    "data-contributor-period": row.period or "full",
                },
            )
        )
    return ui.tags.main(
        _stage_trail("contributors"),
        ui.input_action_button(
            "change_filters",
            "← Back to data exploration",
            class_="text-action",
            onclick=_RESET_VIEW_ONCLICK,
        ),
        ui.div(
            ui.h2(result.title, id="stage-heading", tabindex="-1"),
            ui.p(f"{result.count:,} {result.unit}", class_="contributor-total"),
            ui.p(result.description, class_="section-note"),
            ui.p(
                f"Prompt {spec.report_prompt_version} · {spec.label} · "
                f"Run {query.run} · "
                + (
                    {
                        "all": "All histories",
                        "has": "Has known Drift",
                        "none": "Has no known Drift",
                    }[query.reference_status]
                )
                + " · "
                + (
                    value_label(query.dimension)
                    if query.dimension and query.dimension != "all"
                    else "All Core Values"
                ),
                class_="selected-query-context",
            ),
            class_="contributor-heading",
        ),
        ui.div(*articles, class_="contributor-list")
        if articles
        else ui.div(
            ui.h3("No contributors for these filters"),
            ui.p("Choose another Core Value, known Drift status, or preserved Run."),
            class_="empty-state",
        ),
        ui.div(
            ui.input_action_button(
                "previous_contributors",
                "Previous 50",
                disabled=page == 0,
            ),
            ui.p(
                f"Showing {start + 1 if result.count else 0}–{stop} "
                f"of {result.count:,} {result.unit}"
            ),
            ui.input_action_button(
                "next_contributors",
                "Next 50",
                disabled=stop >= result.count,
            ),
            class_="contributor-pagination",
        ),
        class_="stage-main results-stage",
        id="main-content",
    )


def _how_it_works_panel() -> ui.Tag:
    def chip(text: str, class_name: str = "", sequence: int = 0) -> ui.Tag:
        return ui.span(
            text,
            class_=f"flow-chip {class_name}".strip(),
            style=f"--sequence: {sequence}",
        )

    def column(label: str, *items: ui.Tag, class_name: str = "") -> ui.Tag:
        return ui.div(
            ui.span(label, class_="flow-label"),
            ui.div(*items, class_="flow-items"),
            class_=f"flow-column {class_name}".strip(),
        )

    def step(
        part: str,
        number: int,
        kicker: str,
        title: str,
        body: str,
        visual: ui.Tag,
        *,
        active: bool = False,
    ) -> ui.Tag:
        return ui.tags.article(
            ui.div(
                ui.p(kicker, class_="explainer-kicker"),
                ui.h3(title, id=f"explainer-step-{number}-heading", tabindex="-1"),
                ui.p(body, class_="explainer-description"),
                class_="explainer-copy",
            ),
            ui.div(visual, class_="explainer-canvas", aria_hidden="true"),
            class_="explainer-step",
            data_part=part,
            data_step=str(number),
            aria_labelledby=f"explainer-step-{number}-heading",
            hidden=None if active else "hidden",
        )

    development_steps = (
        step(
            "development",
            1,
            "Development · Prepare",
            "Create review cases from synthetic data",
            "Each review case pairs one synthetic persona with one Core Value and "
            "its chronological Journal Entries.",
            ui.div(
                column(
                    "Synthetic development data",
                    chip("Synthetic persona", sequence=0),
                    chip("Declared Core Values", sequence=1),
                    chip("Journal Entries", sequence=2),
                    class_name="is-source",
                ),
                ui.span("becomes", class_="flow-link"),
                column(
                    "Review case",
                    chip("One persona × one Core Value", "is-output", sequence=3),
                    chip("Chronological Journal Entries", "is-output", sequence=4),
                    class_name="is-destination",
                ),
                class_="explainer-flow",
            ),
            active=True,
        ),
        step(
            "development",
            2,
            "Development · Reference lane",
            "Create LLM-Judge Conflict Labels separately",
            "The LLM-Judge creates LLM-Judge Conflict Labels and known Drift for "
            "evaluation. This is a separate lane from the Weekly Drift Reviewer.",
            ui.div(
                column(
                    "Reference inputs",
                    chip("Journal Entry", sequence=0),
                    chip("Core Value", sequence=1),
                    class_name="is-source",
                ),
                ui.span("reviewed by", class_="flow-link"),
                column(
                    "LLM-Judge",
                    chip("Isolated AI review", "is-reference", sequence=2),
                    class_name="is-component",
                ),
                ui.span("creates", class_="flow-link"),
                column(
                    "Evaluation only",
                    chip("LLM-Judge Conflict Labels", "is-reference", sequence=3),
                    chip("Known Drift", "is-reference", sequence=4),
                    class_name="is-destination",
                ),
                class_="explainer-flow is-wide",
            ),
        ),
        step(
            "development",
            3,
            "Development · Input boundary",
            "Keep evaluation evidence out of the review",
            "Only information available at the verified weekly cutoff reaches the "
            "Weekly Drift Reviewer.",
            ui.div(
                ui.div(
                    column(
                        "Shown",
                        chip("All declared Core Values", "is-allowed", sequence=0),
                        chip(
                            "Journal Entries through the cutoff",
                            "is-allowed",
                            sequence=1,
                        ),
                        chip("Current-week markers", "is-allowed", sequence=2),
                    ),
                    ui.div(
                        ui.span("Input gate", class_="gate-label"),
                        ui.span("Pass", class_="gate-status"),
                        class_="flow-gate is-open",
                    ),
                    column(
                        "Weekly Drift Reviewer",
                        chip("Visible inputs only", "is-component", sequence=3),
                    ),
                    class_="boundary-route is-allowed-route",
                ),
                ui.div(
                    column(
                        "Not shown",
                        chip("Persona biography", "is-blocked", sequence=0),
                        chip("Future Journal Entries", "is-blocked", sequence=1),
                        chip(
                            "LLM-Judge Conflict Labels or known Drift",
                            "is-blocked",
                            sequence=2,
                        ),
                        chip("VIF Critic Predictions", "is-blocked", sequence=3),
                    ),
                    ui.div(
                        ui.span("Input gate", class_="gate-label"),
                        ui.span("Stopped", class_="gate-status"),
                        class_="flow-gate is-closed",
                    ),
                    column(
                        "Outside the review",
                        chip("Never enters the prompt", "is-blocked", sequence=4),
                    ),
                    class_="boundary-route is-blocked-route",
                ),
                class_="boundary-demo",
            ),
        ),
        step(
            "development",
            4,
            "Development · Evaluation",
            "Compare only after decisions are complete",
            "Weekly Drift Reviewer Decisions meet the isolated LLM-Judge Conflict "
            "Labels only afterward, when development performance is calculated.",
            ui.div(
                column(
                    "Weekly Drift Reviewer Decisions",
                    chip("Conflict", "is-conflict", sequence=0),
                    chip("Not Conflict", "is-not-conflict", sequence=1),
                    chip("Abstain", "is-abstain", sequence=2),
                    class_name="is-source",
                ),
                ui.span("compared after review", class_="flow-link is-join"),
                column(
                    "Development comparison",
                    chip("Drift recall", "is-output", sequence=4),
                    chip("False Drift alerts", "is-output", sequence=5),
                    class_name="is-component",
                ),
                ui.span("against", class_="flow-link is-join"),
                column(
                    "Isolated LLM-Judge Conflict Labels",
                    chip("Known Drift", "is-reference", sequence=3),
                    class_name="is-destination",
                ),
                class_="explainer-flow is-wide is-comparison",
            ),
        ),
    )

    deployment_steps = (
        step(
            "deployment",
            5,
            "Intended deployed flow · New evidence",
            "Add new Journal Entries chronologically",
            "The user's declared Core Values stay available while each new Journal "
            "Entry extends the timeline.",
            ui.div(
                ui.div(
                    ui.div(
                        ui.span("t0", class_="timeline-index"),
                        ui.span("Earlier Journal Entry", class_="timeline-copy"),
                        class_="mini-entry",
                    ),
                    ui.div(
                        ui.span("t1", class_="timeline-index"),
                        ui.span("New Journal Entry", class_="timeline-copy"),
                        class_="mini-entry is-new",
                    ),
                    ui.div(
                        ui.span("t2", class_="timeline-index"),
                        ui.span("New Journal Entry", class_="timeline-copy"),
                        class_="mini-entry is-new",
                    ),
                    class_="mini-timeline",
                ),
                ui.div(
                    ui.span("Declared Core Values", class_="flow-label"),
                    chip("Benevolence", "is-allowed", sequence=3),
                    chip("Achievement", "is-allowed", sequence=4),
                    class_="value-rail",
                ),
                class_="timeline-stage",
            ),
        ),
        step(
            "deployment",
            6,
            "Intended deployed flow · Weekly cutoff",
            "Freeze exactly what can be reviewed",
            "Journal Entries through the weekly cutoff can enter the review. Later "
            "Journal Entries remain locked for a future review.",
            ui.div(
                ui.div(
                    ui.div(
                        chip("t0", "is-allowed", sequence=0),
                        chip("t1", "is-allowed", sequence=1),
                        chip("t2", "is-allowed", sequence=2),
                        class_="cutoff-entries is-visible",
                    ),
                    ui.div(
                        ui.span("Verified weekly cutoff", class_="cutoff-label"),
                        class_="cutoff-marker",
                    ),
                    ui.div(
                        chip("t3 · future", "is-blocked", sequence=3),
                        class_="cutoff-entries is-future",
                    ),
                    class_="cutoff-track",
                ),
                ui.div(
                    ui.span("Review input", class_="flow-label"),
                    ui.span(
                        "Core Values + Journal Entries through t2 + "
                        "current-week markers",
                        class_="cutoff-receipt",
                    ),
                    class_="cutoff-summary",
                ),
                ui.div(
                    ui.span("Not in the input", class_="flow-label"),
                    ui.span(
                        "LLM-Judge Conflict Labels · known Drift · "
                        "VIF Critic Predictions",
                        class_="cutoff-exclusion-copy",
                    ),
                    class_="cutoff-exclusions",
                ),
                class_="cutoff-stage",
            ),
        ),
        step(
            "deployment",
            7,
            "Intended deployed flow · Review",
            "Decide Conflict one Journal Entry at a time",
            "For each Journal Entry and Core Value, the Weekly Drift Reviewer decides "
            "Conflict, Not Conflict, or Abstain from the displayed text.",
            ui.div(
                column(
                    "Eligible Journal Entries",
                    chip("t0 + Benevolence", sequence=0),
                    chip("t1 + Benevolence", sequence=1),
                    chip("t2 + Benevolence", sequence=2),
                    class_name="is-source",
                ),
                ui.span("reviewed by", class_="flow-link"),
                column(
                    "Weekly Drift Reviewer",
                    chip("Displayed behavior or choice", "is-component", sequence=3),
                    class_name="is-component",
                ),
                ui.span("decides", class_="flow-link"),
                column(
                    "Weekly Drift Reviewer Decisions",
                    chip("t0 · Not Conflict", "is-not-conflict", sequence=4),
                    chip("t1 · Conflict", "is-conflict", sequence=5),
                    chip("t2 · Conflict", "is-conflict", sequence=6),
                    class_name="is-destination",
                ),
                class_="explainer-flow is-wide",
            ),
        ),
        step(
            "deployment",
            8,
            "Intended deployed flow · Deterministic rule",
            "Two consecutive Conflicts form one Drift",
            "The Drift Detector—not the Weekly Drift Reviewer—marks Drift when two "
            "consecutive Weekly Drift Reviewer Conflict decisions concern the same "
            "Core Value.",
            ui.div(
                ui.div(
                    ui.span("Same Core Value", class_="decision-sequence-label"),
                    ui.div(
                        chip("t0 · Not Conflict", "is-not-conflict", sequence=0),
                        chip("t1 · Conflict", "is-conflict", sequence=1),
                        chip("t2 · Conflict", "is-conflict", sequence=2),
                        class_="decision-sequence",
                    ),
                    ui.div(
                        ui.span("consecutive pair", class_="pair-label"),
                        class_="pair-bracket",
                    ),
                    class_="decision-track",
                ),
                ui.span("applies", class_="flow-link"),
                column(
                    "Drift Detector",
                    chip("Exact two-Conflict rule", "is-component", sequence=3),
                    class_name="is-component",
                ),
                ui.span("marks", class_="flow-link"),
                column(
                    "Result",
                    chip("One Drift", "is-drift", sequence=4),
                    class_name="is-destination",
                ),
                ui.p(
                    "Intended deployed flow · not yet deployment-approved",
                    class_="deployment-caveat",
                ),
                class_="detector-stage",
            ),
        ),
    )

    step_titles = (
        "Create synthetic review cases",
        "Create LLM-Judge Conflict Labels separately",
        "Keep evaluation evidence out",
        "Compare after review",
        "Add new Journal Entries",
        "Freeze the weekly cutoff",
        "Make Weekly Drift Reviewer Decisions",
        "Apply the Drift Detector rule",
    )

    return ui.tags.section(
        ui.div(
            ui.div(
                ui.p("FROZEN WEEKLY DRIFT REVIEWER INPUT CONTRACT", class_="eyebrow"),
                ui.h2("How it works"),
            ),
            ui.p(
                "Follow the same input boundary during synthetic development and "
                "the intended deployed flow.",
                class_="section-note",
            ),
            class_="section-heading how-it-works-heading",
        ),
        ui.div(
            ui.div(
                ui.tags.button(
                    ui.span("1", class_="part-number"),
                    ui.span(
                        ui.strong("Synthetic development"),
                        ui.tags.small(
                            "Why LLM-Judge Conflict Labels never enter the review"
                        ),
                    ),
                    id="explainer-development-tab",
                    type="button",
                    role="tab",
                    aria_selected="true",
                    aria_controls="explainer-development-panel",
                    tabindex="0",
                    class_="explainer-part-tab is-active",
                    data_part_target="development",
                ),
                ui.tags.button(
                    ui.span("2", class_="part-number"),
                    ui.span(
                        ui.strong("Intended deployed flow"),
                        ui.tags.small("What happens when new Journal Entries arrive"),
                    ),
                    id="explainer-deployment-tab",
                    type="button",
                    role="tab",
                    aria_selected="false",
                    aria_controls="explainer-deployment-panel",
                    tabindex="-1",
                    class_="explainer-part-tab",
                    data_part_target="deployment",
                ),
                class_="explainer-part-tabs",
                role="tablist",
                aria_label="How Drift detection works",
            ),
            ui.div(
                ui.div(
                    *development_steps,
                    id="explainer-development-panel",
                    role="tabpanel",
                    aria_labelledby="explainer-development-tab",
                    class_="explainer-part-panel",
                    data_part_panel="development",
                ),
                ui.div(
                    *deployment_steps,
                    id="explainer-deployment-panel",
                    role="tabpanel",
                    aria_labelledby="explainer-deployment-tab",
                    class_="explainer-part-panel",
                    data_part_panel="deployment",
                    hidden="hidden",
                ),
                class_="explainer-stage",
            ),
            ui.div(
                ui.div(
                    ui.tags.button(
                        "Previous",
                        type="button",
                        class_="explainer-control",
                        data_explainer_action="previous",
                        disabled="disabled",
                    ),
                    ui.tags.button(
                        "Play sequence",
                        type="button",
                        class_="explainer-control is-primary",
                        data_explainer_action="toggle",
                        aria_pressed="false",
                    ),
                    ui.tags.button(
                        "Next",
                        type="button",
                        class_="explainer-control",
                        data_explainer_action="next",
                    ),
                    class_="explainer-playback",
                    aria_label="Animation controls",
                ),
                ui.div(
                    *[
                        ui.tags.button(
                            str(index),
                            type="button",
                            class_=(
                                "explainer-step-button is-active"
                                if index == 1
                                else "explainer-step-button"
                            ),
                            data_step_target=str(index),
                            aria_label=f"Step {index}: {title}",
                            aria_current="step" if index == 1 else None,
                        )
                        for index, title in enumerate(step_titles, start=1)
                    ],
                    class_="explainer-step-nav",
                    aria_label="Choose an animation step",
                ),
                ui.p(
                    "Step 1 of 8",
                    class_="explainer-progress",
                    aria_live="polite",
                    aria_atomic="true",
                    data_explainer_progress="true",
                ),
                class_="explainer-footer",
            ),
            class_="drift-explainer",
            data_drift_explainer="true",
        ),
        ui.tags.details(
            ui.tags.summary("Read the complete input contract"),
            ui.div(
                ui.div(
                    ui.h3("Shown to the Weekly Drift Reviewer"),
                    ui.tags.ul(
                        ui.tags.li("All declared Core Values"),
                        ui.tags.li(
                            "Displayed Journal Entries through the review week, "
                            "including any nudge and response"
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
                        ui.tags.li(
                            "AI-reviewed LLM-Judge Conflict Labels or known Drift"
                        ),
                        ui.tags.li(
                            "VIF Critic Predictions or other experiment setup or Run "
                            "decisions"
                        ),
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
                    "Two consecutive Weekly Drift Reviewer Conflict decisions for "
                    "the same Core Value form one Drift. A longer uninterrupted run "
                    "is still one Drift."
                ),
                class_="drift-rule-note",
            ),
            class_="full-contract",
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
                    "the historical v0 development Runs, detection was about four "
                    "days slower "
                    "than for same-week Drift."
                ),
                class_="contract-disclosure-body",
            ),
            class_="contract-disclosure",
        ),
        class_="how-it-works",
    )


def _cutoff_evidence(
    data: ReviewData, case: CaseRecord, setup_key: str = CURRENT_SETUP_KEY
) -> ui.Tag:
    boundaries = data.boundaries_for_persona(case.persona_id, setup_key)
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


def _setup_context(data: ReviewData, setup_key: str) -> ui.Tag:
    spec = data.setup_specs[setup_key]
    return ui.tags.aside(
        _pill(f"Prompt {spec.report_prompt_version}", "is-selected-value"),
        ui.strong(spec.label),
        ui.span(spec.study_label),
        ui.p(
            "Detection study: reference comparisons below are descriptive replay; "
            "the submitted report makes no accuracy claim for this study."
        )
        if not spec.accuracy_evaluated
        else None,
        class_="selected-experiment",
        aria_label="Selected experiment",
    )


def _detection_scoreboard(
    data: ReviewData, case: CaseRecord, setups: dict[str, Any]
) -> ui.Tag:
    return ui.tags.section(
        ui.h2("Detected Drifts in the selected prompt study"),
        ui.p(
            "These preserved decisions support the Core Value definition study. "
            "Counts describe detected Drifts; the study did not measure accuracy "
            "against reference labels.",
            class_="section-note",
        ),
        ui.div(
            ui.tags.table(
                ui.tags.thead(
                    ui.tags.tr(
                        ui.tags.th("Setup", scope="col"),
                        *[ui.tags.th(f"Run {run}", scope="col") for run in (1, 2, 3)],
                    )
                ),
                ui.tags.tbody(
                    *[
                        ui.tags.tr(
                            ui.tags.th(spec.label, scope="row"),
                            *[
                                ui.tags.td(
                                    *[
                                        _pill(
                                            _format_span(
                                                row.onset_t_index, row.end_t_index
                                            )
                                        )
                                        for row in data.predicted_drifts[
                                            (key, run, case.case_id)
                                        ]
                                    ],
                                    "No Drift alert"
                                    if not data.predicted_drifts[
                                        (key, run, case.case_id)
                                    ]
                                    else None,
                                )
                                for run in (1, 2, 3)
                            ],
                        )
                        for key, spec in setups.items()
                    ]
                ),
                class_="scoreboard-table",
            ),
            class_="table-scroll",
        ),
        class_="scoreboard-panel",
    )


def _scoreboard(
    data: ReviewData, case: CaseRecord, setup_key: str = CURRENT_SETUP_KEY
) -> ui.Tag:
    selected = data.setup_specs[setup_key]
    setups = {setup_key: selected}
    if not selected.accuracy_evaluated:
        return _detection_scoreboard(data, case, setups)
    references = data.reference_drifts[case.case_id]
    reference_count = len(references)
    return ui.tags.section(
        ui.div(
            ui.div(
                ui.p("PERSONA SCOREBOARD", class_="eyebrow"),
                ui.h2(
                    "What the selected Weekly Drift Reviewer found",
                    id="scoreboard-heading",
                ),
            ),
            ui.p(
                f"Three preserved Runs of prompt {selected.report_prompt_version}, "
                f"{selected.label}, are compared against the same "
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
        ui.tags.details(
            ui.tags.summary(
                ui.span("Inspect the three preserved Runs"),
                ui.span(
                    f"{len(setups)} setups · {len(setups) * 3} preserved Runs",
                    class_="comparison-note",
                ),
            ),
            ui.div(
                ui.tags.table(
                    ui.tags.caption(
                        "Persona-level Drift outcome by Weekly Drift Reviewer "
                        "setup and Run"
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
                                            "Current contract"
                                            if setup_key == CURRENT_SETUP_KEY
                                            else "Selected experiment",
                                            "is-selected-value",
                                        )
                                        if setup_key == selected.key
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
                            for setup_key, spec in setups.items()
                        ]
                    ),
                    class_="scoreboard-table",
                ),
                class_="table-scroll",
            ),
            class_="scoreboard-comparison",
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


def _detail_screen(
    data: ReviewData,
    case: CaseRecord,
    setup_key: str = CURRENT_SETUP_KEY,
    selected_period: str = "full",
) -> ui.Tag:
    profile = data.profiles[case.persona_id]
    training_provenance, evidence_role = _provenance_summary(case)
    period_choices = _period_choices(data, case, setup_key)
    default_period = selected_period if selected_period in period_choices else "full"
    selected_spec = data.setup_specs[setup_key]
    return ui.tags.main(
        _stage_trail("detail"),
        ui.div(
            ui.input_action_button(
                "back_to_personas",
                "← Back to contributors",
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
        ui.output_ui("detail_experiment"),
        ui.output_ui("persona_scoreboard"),
        ui.tags.section(
            ui.div(
                ui.div(
                    ui.input_radio_buttons(
                        "setup",
                        "Weekly Drift Reviewer setup",
                        choices={
                            key: spec.label
                            for key, spec in data.setups_for_prompt(
                                selected_spec.report_prompt_version
                            ).items()
                        },
                        selected=setup_key,
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
                "scoreboard compares three Runs of the selected setup. "
                "The contributor opens at its Journal Entry or Drift span; "
                "use the period control to isolate a known Drift, Drift alert, or "
                "review week.",
                class_="controls-note",
            ),
            class_="controls-panel",
        ),
        ui.tags.section(
            ui.div(
                ui.div(
                    ui.p("PRIMARY REVIEW", class_="eyebrow"),
                    ui.h2("Journal Entry trajectory and evidence"),
                ),
                ui.p(
                    "Each line is one Run of the selected setup. Select a point to "
                    "inspect its Journal Entry, Weekly Drift Reviewer Decision, and "
                    "LLM-Judge Conflict Label; Runs are never merged.",
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
        ui.output_ui("cutoff_evidence"),
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


_TRAJECTORY_CATEGORY_LABELS = {
    "not_conflict": "Not Conflict",
    "abstain": "Abstain",
    "conflict": "Conflict",
    "invalid": "Invalid response",
}


def _trajectory_category(decision: Decision) -> str:
    if decision.response_status != "ok":
        return "invalid"
    if decision.verdict in {"not_conflict", "abstain", "conflict"}:
        return decision.verdict
    return "invalid"


def _trajectory_comparison(entry: EntryRecord, decision: Decision) -> tuple[str, str]:
    if decision.response_status != "ok":
        return (
            "neutral",
            "Invalid Weekly Drift Reviewer response; no label comparison.",
        )
    if entry.final_conflict is None:
        return (
            "neutral",
            "Uncertain LLM-Judge Conflict Label; no resolved comparison.",
        )
    if decision.verdict == "abstain":
        reference_label = "Conflict" if entry.final_conflict else "Not Conflict"
        return (
            "mismatch",
            "Weekly Drift Reviewer Decision Abstain differs from the resolved "
            f"{reference_label} LLM-Judge Conflict Label.",
        )
    if decision.verdict not in {"conflict", "not_conflict"}:
        return (
            "neutral",
            "No valid Weekly Drift Reviewer Decision to compare.",
        )
    decision_conflict = decision.verdict == "conflict"
    if decision_conflict == entry.final_conflict:
        return "match", "Matches the resolved LLM-Judge Conflict Label."
    return "mismatch", "Differs from the resolved LLM-Judge Conflict Label."


def _trajectory_annotation_tags(
    data: ReviewData,
    case: CaseRecord,
    setup_key: str,
    run: int,
    entries: tuple[EntryRecord, ...],
    x_positions: dict[int, float],
    category_index: dict[str, int],
) -> tuple[ui.Tag, ...]:
    visible = {entry.t_index: entry for entry in entries}
    predictions = data.predicted_drifts[(setup_key, run, case.case_id)]
    annotations: list[ui.Tag] = []
    label_specs: list[tuple[str, str, float, float, str, bool, str]] = []

    def add_pair_annotation(
        onset_t_index: int,
        confirmation_t_index: int,
        label: str,
        class_name: str,
    ) -> None:
        confirmation_entry = visible.get(confirmation_t_index)
        if confirmation_entry is None:
            return
        confirmation_decision = data.decision(
            setup_key, run, case.case_id, confirmation_t_index
        )
        category = _trajectory_category(confirmation_decision)
        y_percent = ((category_index[category] + 0.5) / len(category_index)) * 100
        confirmation_x = x_positions[confirmation_t_index]
        onset_x = x_positions.get(onset_t_index)
        if onset_x is not None:
            left = min(onset_x, confirmation_x)
            width = max(2.2, abs(confirmation_x - onset_x))
            annotations.append(
                ui.span(
                    class_=f"trajectory-alert-bracket {class_name}",
                    style=(
                        f"left: {left:.3f}%; width: {width:.3f}%; "
                        f"top: {y_percent:.3f}%;"
                    ),
                    aria_hidden="true",
                )
            )
        description = (
            f"Run {run} · {label} · {_format_span(onset_t_index, confirmation_t_index)}"
        )
        label_specs.append(
            (
                label,
                class_name,
                confirmation_x,
                y_percent,
                description,
                False,
                "below" if category == "not_conflict" else "above",
            )
        )

    if data.setup_specs[setup_key].accuracy_evaluated:
        for span in predictions:
            if span.result == "false Drift alert":
                add_pair_annotation(
                    span.onset_t_index,
                    span.confirmation_t_index,
                    "False Drift alert",
                    "is-false",
                )

        matched_reference_ids = {
            span.matched_reference_id
            for span in predictions
            if span.result == "hit" and span.matched_reference_id
        }
        for span in data.reference_drifts[case.case_id]:
            if span.drift_id not in matched_reference_ids:
                add_pair_annotation(
                    span.onset_t_index,
                    span.confirmation_t_index,
                    "Missed known Drift",
                    "is-missed",
                )

    for entry in entries:
        decision = data.decision(setup_key, run, case.case_id, entry.t_index)
        if decision.response_status == "ok":
            continue
        category = _trajectory_category(decision)
        y_percent = ((category_index[category] + 0.5) / len(category_index)) * 100
        x_percent = x_positions[entry.t_index]
        description = f"Run {run} · Invalid response · t{entry.t_index}"
        label_specs.append(
            (
                "Invalid response",
                "is-invalid",
                x_percent,
                y_percent,
                description,
                True,
                "above",
            )
        )

    occupied_rects: list[tuple[float, float, float, float]] = []
    for label, class_name, x_percent, y_percent, description, is_point, side in sorted(
        label_specs, key=lambda spec: spec[2]
    ):
        leading = x_percent < 24
        estimated_width = min(42.0, max(24.0, 10.0 + len(label) * 1.2))
        interval = (
            (x_percent, min(100.0, x_percent + estimated_width))
            if leading
            else (max(0.0, x_percent - estimated_width), x_percent)
        )
        base_y_px = y_percent * 1.04
        lane = 0
        while True:
            top_px = (
                base_y_px - 31.2 - lane * 19.2
                if side == "above"
                else base_y_px + 10.4 + lane * 19.2
            )
            candidate = (
                interval[0],
                interval[1],
                top_px,
                top_px + 16.0,
            )
            if all(
                candidate[1] + 2.0 < used[0]
                or candidate[0] - 2.0 > used[1]
                or candidate[3] + 3.0 < used[2]
                or candidate[2] - 3.0 > used[3]
                for used in occupied_rects
            ):
                break
            lane += 1
        occupied_rects.append(candidate)
        annotations.append(
            ui.span(
                label,
                class_=(
                    f"trajectory-alert-label {class_name}"
                    + (" is-point" if is_point else "")
                    + (" is-leading" if leading else "")
                    + (" is-below" if side == "below" else "")
                ),
                style=(
                    f"left: {x_percent:.3f}%; top: {y_percent:.3f}%; "
                    f"--annotation-lane: {lane};"
                ),
                title=description,
                role="note",
                aria_label=description,
            )
        )

    return tuple(annotations)


def _date_x_percent(value: date, start: date, end: date) -> float:
    if start == end:
        return 50.0
    elapsed = (value - start).days
    duration = (end - start).days
    return 7.0 + (elapsed / duration) * 86.0


def _reference_window_positions(
    data: ReviewData,
    case: CaseRecord,
    entries: tuple[EntryRecord, ...],
) -> tuple[tuple[DriftSpan, float, float], ...]:
    if not entries:
        return ()
    start = date.fromisoformat(entries[0].date)
    end = date.fromisoformat(entries[-1].date)
    positions: list[tuple[DriftSpan, float, float]] = []
    for span in data.reference_drifts[case.case_id]:
        span_start = max(start, date.fromisoformat(span.onset_date))
        span_end = min(end, date.fromisoformat(span.end_date))
        if span_start > span_end:
            continue
        left = max(2.0, _date_x_percent(span_start, start, end) - 1.25)
        right = min(98.0, _date_x_percent(span_end, start, end) + 1.25)
        positions.append((span, left, max(2.5, right - left)))
    return tuple(positions)


def _default_inspection_point(
    data: ReviewData,
    case: CaseRecord,
    setup_key: str,
    period: str,
) -> tuple[int | None, int]:
    entries = _visible_entries(case, period)
    if not entries:
        return None, 1
    visible = {entry.t_index for entry in entries}
    for span in data.reference_drifts[case.case_id]:
        if span.confirmation_t_index not in visible:
            continue
        for run in (1, 2, 3):
            decision = data.decision(
                setup_key, run, case.case_id, span.confirmation_t_index
            )
            if _trajectory_category(decision) == "conflict":
                return span.confirmation_t_index, run
        return span.confirmation_t_index, 1
    for run in (1, 2, 3):
        for span in data.predicted_drifts[(setup_key, run, case.case_id)]:
            if span.confirmation_t_index in visible:
                return span.confirmation_t_index, run
    for entry in entries:
        for run in (1, 2, 3):
            decision = data.decision(setup_key, run, case.case_id, entry.t_index)
            if _trajectory_category(decision) == "conflict":
                return entry.t_index, run
    return entries[0].t_index, 1


def _trajectory_step_path(points: list[tuple[float, float]]) -> str:
    if not points:
        return ""
    commands = [f"M {points[0][0]:.2f} {points[0][1]:.2f}"]
    previous_x = points[0][0]
    for x, y in points[1:]:
        midpoint = (previous_x + x) / 2
        commands.extend((f"H {midpoint:.2f}", f"V {y:.2f}", f"H {x:.2f}"))
        previous_x = x
    return " ".join(commands)


def _trajectory_onclick(t_index: int, run: int) -> str:
    return (
        "Shiny.setInputValue('trajectory_point', "
        f"'{t_index}|{run}', {{priority: 'event'}});"
    )


def _trajectory_plot(
    data: ReviewData,
    case: CaseRecord,
    setup_key: str,
    period: str,
    selected_t_index: int,
    selected_run: int,
) -> ui.Tag:
    entries = _visible_entries(case, period)
    all_decisions = [
        data.decision(setup_key, run, case.case_id, entry.t_index)
        for entry in entries
        for run in (1, 2, 3)
    ]
    category_keys = ["not_conflict", "abstain", "conflict"]
    if any(_trajectory_category(decision) == "invalid" for decision in all_decisions):
        category_keys.append("invalid")
    category_index = {key: index for index, key in enumerate(category_keys)}
    chart_height = len(category_keys) * 100.0
    start = date.fromisoformat(entries[0].date)
    end = date.fromisoformat(entries[-1].date)
    x_positions = {
        entry.t_index: _date_x_percent(date.fromisoformat(entry.date), start, end)
        for entry in entries
    }

    reference_positions = _reference_window_positions(data, case, entries)
    run_plots: list[ui.Tag] = []
    for run in (1, 2, 3):
        points: list[tuple[float, float]] = []
        point_buttons: list[ui.Tag] = []
        for entry in entries:
            decision = data.decision(setup_key, run, case.case_id, entry.t_index)
            category = _trajectory_category(decision)
            comparison, comparison_note = _trajectory_comparison(entry, decision)
            reference_label, _reference_class = _reference_label(entry)
            y_units = category_index[category] * 100 + 50
            points.append((x_positions[entry.t_index] * 10, y_units))
            selected = entry.t_index == selected_t_index and run == selected_run
            parsed_date = date.fromisoformat(entry.date)
            point_buttons.append(
                ui.tags.button(
                    type="button",
                    class_=(
                        f"trajectory-point comparison-{comparison} is-{category}"
                        + (" is-selected" if selected else "")
                    ),
                    style=(
                        f"left: {x_positions[entry.t_index]:.3f}%; "
                        f"top: {(y_units / chart_height) * 100:.3f}%;"
                    ),
                    onclick=_trajectory_onclick(entry.t_index, run),
                    aria_label=(
                        f"Run {run}, Journal Entry {entry.t_index}, "
                        f"{parsed_date.strftime('%d %B %Y')}, "
                        f"Weekly Drift Reviewer Decision "
                        f"{_verdict_label(decision)}, "
                        f"LLM-Judge Conflict Label {reference_label}. "
                        f"{comparison_note} Select evidence."
                    ),
                    aria_pressed="true" if selected else "false",
                    title=(
                        f"Run {run} · Journal Entry {entry.t_index} · "
                        f"Decision: {_verdict_label(decision)} · "
                        f"LLM-Judge: {reference_label} · {comparison_note}"
                    ),
                )
            )
        line_path = Tag(
            "path",
            d=_trajectory_step_path(points),
            class_="trajectory-run-line",
            fill="none",
            vector_effect="non-scaling-stroke",
            _add_ws=False,
        )
        alert_annotations = _trajectory_annotation_tags(
            data,
            case,
            setup_key,
            run,
            entries,
            x_positions,
            category_index,
        )
        run_plots.append(
            ui.div(
                ui.div(
                    ui.span(class_="trajectory-legend-mark"),
                    ui.strong(f"Run {run}"),
                    ui.span(
                        "Selected Run" if run == selected_run else None,
                        class_="trajectory-run-selected",
                    ),
                    class_="trajectory-run-heading",
                ),
                ui.div(
                    ui.div(
                        *[
                            ui.span(
                                _TRAJECTORY_CATEGORY_LABELS[key],
                                class_=(f"trajectory-category-label is-{key}"),
                            )
                            for key in category_keys
                        ],
                        class_="trajectory-y-axis",
                        style=(f"--trajectory-row-count: {len(category_keys)};"),
                        aria_hidden="true",
                    ),
                    ui.div(
                        *[
                            ui.span(
                                class_="known-drift-window",
                                style=(f"left: {left:.3f}%; width: {width:.3f}%;"),
                            )
                            for _span, left, width in reference_positions
                        ],
                        *[
                            ui.span(
                                class_=f"trajectory-band is-{key}",
                                style=(
                                    f"top: "
                                    f"{(index / len(category_keys)) * 100:.3f}%; "
                                    f"height: "
                                    f"{100 / len(category_keys):.3f}%;"
                                ),
                            )
                            for index, key in enumerate(category_keys)
                        ],
                        ui.tags.svg(
                            line_path,
                            class_="trajectory-lines",
                            viewBox=f"0 0 1000 {chart_height:.0f}",
                            preserveAspectRatio="none",
                            aria_hidden="true",
                            focusable="false",
                        ),
                        *point_buttons,
                        *alert_annotations,
                        class_="trajectory-canvas",
                        style=(f"--trajectory-row-count: {len(category_keys)};"),
                    ),
                    class_="trajectory-chart",
                ),
                class_=(
                    f"trajectory-run-plot run-{run}"
                    + (" is-selected" if run == selected_run else "")
                ),
                role="group",
                aria_label=f"Run {run} Weekly Drift Reviewer Decisions",
            )
        )

    return ui.div(
        ui.div(
            ui.div(
                ui.p("DECISION COMPARISON", class_="eyebrow"),
                ui.h3("Weekly Drift Reviewer Decisions"),
            ),
            ui.span("3 aligned Runs", class_="trajectory-plot-count"),
            class_="trajectory-plot-heading",
        ),
        ui.div(
            ui.span(
                ui.span(class_="trajectory-status-mark is-match"),
                "Matches resolved LLM-Judge Conflict Label",
                class_="trajectory-status-key",
            ),
            ui.span(
                ui.span(class_="trajectory-status-mark is-mismatch"),
                "Differs from resolved label, including Abstain",
                class_="trajectory-status-key",
            ),
            ui.span(
                ui.span(class_="trajectory-status-mark is-neutral"),
                "Invalid response or unresolved label",
                class_="trajectory-status-key",
            ),
            class_="trajectory-comparison-legend",
            aria_label=(
                "Circle colour compares each Weekly Drift Reviewer Decision "
                "with its LLM-Judge Conflict Label"
            ),
        ),
        ui.div(
            ui.span("Known Drift", class_="trajectory-reference-label"),
            ui.div(
                *[
                    ui.span(
                        _format_span(span.onset_t_index, span.end_t_index),
                        class_="trajectory-reference-span",
                        style=f"left: {left:.3f}%; width: {width:.3f}%;",
                        title=(
                            "Known Drift · "
                            f"{_format_span(span.onset_t_index, span.end_t_index)}"
                        ),
                    )
                    for span, left, width in reference_positions
                ],
                ui.span("None in this view", class_="trajectory-reference-empty")
                if not reference_positions
                else None,
                class_="trajectory-reference-track",
            ),
            class_="trajectory-reference-row",
        ),
        ui.div(
            *run_plots,
            class_="trajectory-run-stack",
            role="group",
            aria_label=("Three date-aligned Weekly Drift Reviewer Run plots"),
        ),
        ui.div(
            ui.span("Journal date", class_="trajectory-axis-label"),
            ui.div(
                *[
                    ui.tags.time(
                        ui.span(str(date.fromisoformat(entry.date).day)),
                        ui.span(date.fromisoformat(entry.date).strftime("%b")),
                        datetime=entry.date,
                        class_=(
                            "trajectory-date"
                            f" lane-{entry.position % 2}"
                            + (
                                " is-selected"
                                if entry.t_index == selected_t_index
                                else ""
                            )
                        ),
                        style=f"left: {x_positions[entry.t_index]:.3f}%;",
                        title=f"Journal Entry {entry.t_index} · {entry.date}",
                    )
                    for entry in entries
                ],
                class_="trajectory-x-axis",
                aria_hidden="true",
            ),
            class_="trajectory-shared-x-axis",
        ),
        ui.p(
            "Point position is the Weekly Drift Reviewer Decision; circle colour "
            "shows its comparison with the LLM-Judge Conflict Label. Rows are "
            "categories, not a numeric scale.",
            class_="trajectory-note",
        ),
        class_="trajectory-plot-panel",
    )


def _trajectory_inspector(
    data: ReviewData,
    case: CaseRecord,
    setup_key: str,
    entry: EntryRecord,
    selected_run: int,
) -> ui.Tag:
    reference_label, reference_class = _reference_label(entry)
    references = _reference_membership(data, case.case_id, entry.t_index)
    selected_decision = data.decision(
        setup_key, selected_run, case.case_id, entry.t_index
    )
    comparison, comparison_note = _trajectory_comparison(entry, selected_decision)
    comparison_label = {
        "match": "Matches label",
        "mismatch": "Mismatch",
        "neutral": "Not compared",
    }[comparison]
    comparison_class = {
        "match": "is-hit",
        "mismatch": "is-false",
        "neutral": "is-abstain",
    }[comparison]
    predictions = _predicted_membership(
        data, setup_key, selected_run, case.case_id, entry.t_index
    )
    unresolved = any(
        entry.t_index in pair
        for pair in data.unresolved_pairs(setup_key, selected_run, case.case_id)
    )
    abstain_explanation = ABSTAIN_EXPLANATIONS.get(
        selected_decision.reason_code or "",
        "The displayed Journal Entry does not support a reliable Weekly Drift "
        "Reviewer Decision.",
    )
    reviewer = (
        "claude-opus-4-8 · reasoning high"
        if entry.opus_resolved
        else "gpt-5.6-sol · reasoning xhigh"
    )
    return ui.tags.article(
        ui.div(
            ui.div(
                ui.p("SELECTED EVIDENCE", class_="eyebrow"),
                ui.h3(f"Journal Entry {entry.t_index}"),
                ui.tags.time(entry.date, datetime=entry.date),
            ),
            ui.div(
                _pill(f"LLM-Judge · {reference_label}", reference_class),
                _pill(
                    f"Run {selected_run} · {_verdict_label(selected_decision)}",
                    _verdict_class(selected_decision),
                ),
                ui.span(
                    comparison_label,
                    class_=f"pill {comparison_class}",
                    title=comparison_note,
                ),
                class_="inspector-heading-pills",
            ),
            class_="trajectory-inspector-heading",
        ),
        ui.div(
            ui.p(entry.initial_entry, class_="inspector-entry-copy"),
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
                class_="inspector-entry-thread",
            ),
            class_="inspector-entry",
        ),
        ui.div(
            ui.div(
                ui.span("LLM-Judge Conflict Label", class_="field-name"),
                ui.div(
                    _pill(reference_label, reference_class),
                    ui.span(f"t{entry.t_index}", class_="reference-index"),
                    class_="inspector-reference-heading",
                ),
            ),
            ui.div(
                *[
                    _pill(
                        "Known Drift · "
                        f"{_format_span(span.onset_t_index, span.end_t_index)}"
                        + (" · cross-week" if span.crosses_week else ""),
                        "is-reference",
                    )
                    for span in references
                ],
                ui.span("Not inside known Drift", class_="reference-membership")
                if not references
                else None,
                class_="inspector-reference-membership",
            ),
            ui.p(reviewer, class_="reference-reviewer"),
            class_="inspector-reference",
        ),
        ui.div(
            ui.div(
                ui.p("WEEKLY DRIFT REVIEWER DECISION", class_="eyebrow"),
                ui.h3(f"Run {selected_run} · {_verdict_label(selected_decision)}"),
                class_="inspector-decision-title",
            ),
            (
                ui.div(
                    ui.strong("Invalid or missing response"),
                    ui.p(
                        "The weekly receipt failed validation; parsed decisions "
                        "were ignored."
                    ),
                    ui.tags.details(
                        ui.tags.summary("Technical validation error"),
                        ui.code(
                            selected_decision.validation_error
                            or "No validation detail recorded."
                        ),
                    ),
                    class_="invalid-warning",
                    role="alert",
                )
                if selected_decision.response_status != "ok"
                else ui.div(
                    ui.div(
                        ui.span(
                            "Why the Weekly Drift Reviewer abstained",
                            class_="field-name",
                        ),
                        ui.p(abstain_explanation),
                        class_="inspector-decision-field",
                    )
                    if selected_decision.verdict == "abstain"
                    else None,
                    ui.div(
                        ui.span(
                            "Evidence from the Journal Entry",
                            class_="field-name",
                        ),
                        ui.tags.blockquote(selected_decision.evidence_quote),
                        class_="inspector-decision-field",
                    )
                    if selected_decision.evidence_quote
                    else ui.p(
                        "No evidence quote was recorded for this Decision.",
                        class_="inspector-empty-evidence",
                    ),
                )
            ),
            ui.div(
                *[_predicted_drift_pill(data, setup_key, span) for span in predictions],
                _pill("Adjacent pair unresolved because of Abstain", "is-abstain")
                if unresolved
                else None,
                class_="decision-flags",
            ),
            class_="inspector-decision",
        ),
        class_="trajectory-inspector",
        aria_label=(f"Journal Entry {entry.t_index} and Run {selected_run} evidence"),
    )


def _trajectory_review(
    data: ReviewData,
    case: CaseRecord,
    setup_key: str,
    period: str,
    selected_t_index: int | None,
    selected_run: int,
) -> ui.Tag:
    entries = _visible_entries(case, period)
    if not entries:
        return ui.div(
            ui.h3("No Journal Entries in this period"),
            ui.p("Choose Full timeline or another Journal Entry period."),
            class_="empty-state",
        )
    visible_indices = {entry.t_index for entry in entries}
    if selected_t_index not in visible_indices or selected_run not in {1, 2, 3}:
        selected_t_index, selected_run = _default_inspection_point(
            data, case, setup_key, period
        )
    assert selected_t_index is not None
    selected_entry = next(
        entry for entry in entries if entry.t_index == selected_t_index
    )
    return ui.div(
        ui.div(
            _trajectory_plot(
                data,
                case,
                setup_key,
                period,
                selected_t_index,
                selected_run,
            ),
            _trajectory_inspector(
                data,
                case,
                setup_key,
                selected_entry,
                selected_run,
            ),
            class_="trajectory-review-grid",
        ),
        ui.tags.details(
            ui.tags.summary("Full Journal Entry and Run comparison"),
            ui.p(
                "Use the complete table for exhaustive side-by-side comparison.",
                class_="full-comparison-note",
            ),
            _timeline(data, case, setup_key, period),
            class_="drawer full-comparison-drawer",
        ),
        class_="trajectory-review",
    )


def _supporting_results(data: ReviewData, case: CaseRecord, setup_key: str) -> ui.Tag:
    references = data.reference_drifts[case.case_id]
    accuracy_evaluated = data.setup_specs[setup_key].accuracy_evaluated
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
                            )
                            if accuracy_evaluated
                            else _format_span(row.onset_t_index, row.end_t_index),
                            ("is-hit" if row.result == "hit" else "is-false")
                            if accuracy_evaluated
                            else "",
                        )
                        for row in rows
                    ],
                    ui.span("No Drift alert") if not rows else None,
                    class_="track-spans",
                ),
                class_="drift-track",
            )
        )
    if not accuracy_evaluated:
        return ui.div(
            ui.div(*tracks, class_="drift-map"),
            ui.p(
                "The prompt definition study measured detected Drifts. Shared "
                "reference labels are shown for descriptive inspection; no "
                "accuracy result was reported for this study.",
                class_="supporting-note",
            ),
            class_="drawer-body supporting-results",
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
                    "Twinkl synthetic data inspection",
                    class_="eyebrow",
                ),
                ui.h1(
                    ui.input_action_link(
                        "go_home",
                        "Drift inspection app",
                        class_="app-home-link",
                        title="Return to homepage",
                        aria_label="Drift inspection app homepage",
                        onclick=_RESET_VIEW_ONCLICK,
                    )
                ),
            ),
            ui.div(
                _pill("Read only"),
                _pill("Preserved experiments"),
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
    ui.head_content(
        ui.tags.link(
            rel="stylesheet",
            href=f"styles.css?v={STYLESHEET_VERSION}",
        ),
        ui.tags.script(
            src=f"drift_explainer.js?v={EXPLAINER_SCRIPT_VERSION}",
            defer="defer",
        ),
    ),
    _app_shell(),
    title="Drift inspection app",
)


def _server(input: Inputs, output: Outputs, session: Session) -> None:
    data = _get_review_data()
    stage: reactive.Value[str] = reactive.value("filters")
    selected_case_id: reactive.Value[str | None] = reactive.value(None)
    selected_drift: reactive.Value[str] = reactive.value("all")
    selected_dimension: reactive.Value[str] = reactive.value("")
    selected_prompt: reactive.Value[str] = reactive.value("v2")
    selected_setup: reactive.Value[str] = reactive.value(CURRENT_SETUP_KEY)
    selected_landing_run: reactive.Value[int] = reactive.value(1)
    selected_period: reactive.Value[str] = reactive.value("full")
    selected_t_index: reactive.Value[int | None] = reactive.value(None)
    selected_run: reactive.Value[int] = reactive.value(1)
    active_query: reactive.Value[ExplorationQuery | None] = reactive.value(None)
    contributor_page: reactive.Value[int] = reactive.value(0)

    def landing_query(
        kind: str = "cases", dimension: str | None = None
    ) -> ExplorationQuery:
        prompt_version = str(input.report_prompt() or selected_prompt.get())
        setup_key = str(input.landing_setup() or selected_setup.get())
        if setup_key not in data.setups_for_prompt(prompt_version):
            setup_key = PROMPT_DEFAULTS[prompt_version]
        return ExplorationQuery(
            kind=cast(QueryKind, kind),
            setup_key=setup_key,
            run=int(input.landing_run() or selected_landing_run.get()),
            reference_status=cast(
                ReferenceStatus, str(input.reference_drift_filter() or "all")
            ),
            dimension=str(input.core_value_filter() or "")
            if dimension is None
            else dimension,
        )

    def open_query(query: ExplorationQuery) -> None:
        selected_setup.set(query.setup_key)
        selected_prompt.set(data.setup_specs[query.setup_key].report_prompt_version)
        selected_landing_run.set(query.run)
        selected_drift.set(query.reference_status)
        selected_dimension.set(query.dimension)
        active_query.set(query)
        contributor_page.set(0)
        stage.set("contributors")

    @render.ui
    def screen() -> ui.Tag:
        current = stage.get()
        with reactive.isolate():
            prompt_version = selected_prompt.get()
            setup_key = selected_setup.get()
            landing_run = selected_landing_run.get()
            period = selected_period.get()
        if current == "filters":
            return _filter_screen(
                data,
                selected_drift=selected_drift.get(),
                selected_dimension=selected_dimension.get(),
                selected_prompt=prompt_version,
                selected_setup=setup_key,
                selected_run=landing_run,
            )
        if current == "contributors":
            query = active_query.get()
            req(query)
            assert query is not None
            return _contributor_screen(
                data, query, explore(data, query), contributor_page.get()
            )
        case_id = selected_case_id.get()
        req(case_id)
        assert case_id is not None
        return _detail_screen(data, data.cases[case_id], setup_key, period)

    @render.ui
    def prompt_explanation() -> ui.Tag:
        req(stage.get() == "filters")
        version = str(input.report_prompt() or "v2")
        return _prompt_explanation(version if version in PROMPT_CHOICES else "v2")

    @render.ui
    def data_breakdowns() -> ui.Tag:
        req(stage.get() == "filters")
        return _data_breakdowns(data, landing_query())

    @reactive.effect
    @reactive.event(input.report_prompt)
    def _sync_landing_setups() -> None:
        req(stage.get() == "filters")
        version = str(input.report_prompt() or "v2")
        if version not in PROMPT_CHOICES:
            return
        choices = data.setups_for_prompt(version)
        current = selected_setup.get()
        ui.update_select(
            "landing_setup",
            choices={key: spec.label for key, spec in choices.items()},
            selected=current if current in choices else PROMPT_DEFAULTS[version],
        )
        selected_prompt.set(version)
        if current not in choices:
            selected_setup.set(PROMPT_DEFAULTS[version])

    @reactive.effect
    @reactive.event(input.landing_setup)
    def _remember_landing_setup() -> None:
        req(stage.get() == "filters")
        key = str(input.landing_setup() or "")
        if key in data.setups_for_prompt(selected_prompt.get()):
            selected_setup.set(key)

    @reactive.effect
    @reactive.event(input.landing_run)
    def _remember_landing_run() -> None:
        req(stage.get() == "filters")
        run = int(input.landing_run() or 1)
        if run in {1, 2, 3}:
            selected_landing_run.set(run)

    @reactive.effect
    @reactive.event(input.setup)
    def _remember_detail_setup() -> None:
        req(stage.get() == "detail")
        key = str(input.setup() or "")
        if key in data.setups_for_prompt(selected_prompt.get()):
            selected_setup.set(key)

    @reactive.effect
    @reactive.event(input.show_personas)
    def _show_personas() -> None:
        open_query(landing_query())

    @reactive.effect
    @reactive.event(input.inspect_query)
    def _open_contributors() -> None:
        payload = input.inspect_query()
        if not isinstance(payload, dict):
            return
        kind = str(payload.get("kind", "cases"))
        if kind not in get_args(QueryKind):
            return
        dimension = payload.get("dimension")
        if stage.get() == "filters":
            query = landing_query(kind, dimension)
        else:
            current = active_query.get()
            req(current)
            assert current is not None
            query = ExplorationQuery(
                kind=cast(QueryKind, kind),
                setup_key=current.setup_key,
                run=current.run,
                reference_status=current.reference_status,
                dimension=current.dimension if dimension is None else str(dimension),
            )
        open_query(query)

    @reactive.effect
    @reactive.event(input.inspect_contributor)
    def _inspect_contributor() -> None:
        query = active_query.get()
        payload = input.inspect_contributor()
        req(query)
        assert query is not None
        if not isinstance(payload, dict):
            return
        rows = explore(data, query).rows
        index = int(payload.get("index", -1))
        if not 0 <= index < len(rows):
            return
        row = rows[index]
        case_id = str(payload.get("case_id", row.case_id))
        related = tuple(getattr(row, "related_case_ids", ())) or (row.case_id,)
        if case_id not in related or case_id not in data.cases:
            return
        case = data.cases[case_id]
        selected_case_id.set(case_id)
        selected_setup.set(query.setup_key)
        selected_landing_run.set(query.run)
        selected_period.set(row.period or "full")
        t_index, _ = _default_inspection_point(
            data, case, query.setup_key, row.period or "full"
        )
        selected_t_index.set(row.t_index if row.t_index is not None else t_index)
        selected_run.set(row.run if row.run is not None else query.run)
        stage.set("detail")

    @reactive.effect
    @reactive.event(input.reference_drift_filter)
    def _sync_core_value_counts() -> None:
        req(stage.get() == "filters")
        status = str(input.reference_drift_filter() or "all")
        current = str(input.core_value_filter() or "")
        choices = {"": "All Core Values", **_dimension_choices(data, status)}
        ui.update_select(
            "core_value_filter",
            choices=choices,
            selected=current if current in choices else "",
        )

    @reactive.effect
    @reactive.event(input.change_filters, input.go_home)
    def _return_home() -> None:
        selected_case_id.set(None)
        selected_t_index.set(None)
        stage.set("filters")

    @reactive.effect
    @reactive.event(input.back_to_personas)
    def _back_to_contributors() -> None:
        query = active_query.get()
        req(query)
        assert query is not None
        selected_case_id.set(None)
        selected_t_index.set(None)
        selected_setup.set(query.setup_key)
        selected_landing_run.set(query.run)
        stage.set("contributors")

    @reactive.effect
    @reactive.event(input.previous_contributors)
    def _previous_page() -> None:
        contributor_page.set(max(0, contributor_page.get() - 1))

    @reactive.effect
    @reactive.event(input.next_contributors)
    def _next_page() -> None:
        query = active_query.get()
        req(query)
        assert query is not None
        last = max(0, (explore(data, query).count - 1) // 50)
        contributor_page.set(min(last, contributor_page.get() + 1))

    @reactive.calc
    def selected_case() -> CaseRecord:
        case_id = selected_case_id.get()
        req(case_id)
        assert case_id is not None
        return data.cases[case_id]

    @reactive.effect
    @reactive.event(input.period)
    def _remember_detail_period() -> None:
        req(stage.get() == "detail")
        period = str(input.period() or "full")
        if period in _period_choices(data, selected_case(), selected_setup.get()):
            selected_period.set(period)

    @reactive.effect
    def _sync_periods() -> None:
        req(stage.get() == "detail")
        choices = _period_choices(data, selected_case(), selected_setup.get())
        current = selected_period.get()
        if current not in choices:
            current = "full"
            selected_period.set(current)
        ui.update_select("period", choices=choices, selected=current)

    @reactive.effect
    @reactive.event(input.trajectory_point)
    def _select_trajectory_point() -> None:
        parts = str(input.trajectory_point() or "").split("|")
        if len(parts) != 2:
            return
        try:
            t_index, run = (int(value) for value in parts)
        except ValueError:
            return
        if run in {1, 2, 3} and t_index in {
            entry.t_index for entry in selected_case().entries
        }:
            selected_t_index.set(t_index)
            selected_run.set(run)

    @reactive.effect
    def _sync_trajectory_selection() -> None:
        req(stage.get() == "detail")
        case = selected_case()
        setup_key = selected_setup.get()
        period = selected_period.get()
        indices = {entry.t_index for entry in _visible_entries(case, period)}
        if selected_t_index.get() not in indices:
            t_index, _ = _default_inspection_point(data, case, setup_key, period)
            selected_t_index.set(t_index)

    @render.ui
    def timeline() -> ui.Tag:
        req(stage.get() == "detail")
        return _trajectory_review(
            data,
            selected_case(),
            selected_setup.get(),
            selected_period.get(),
            selected_t_index.get(),
            selected_run.get(),
        )

    @render.ui
    def detail_experiment() -> ui.Tag:
        req(stage.get() == "detail")
        return _setup_context(data, selected_setup.get())

    @render.ui
    def persona_scoreboard() -> ui.Tag:
        req(stage.get() == "detail")
        return _scoreboard(data, selected_case(), selected_setup.get())

    @render.ui
    def cutoff_evidence() -> ui.Tag:
        req(stage.get() == "detail")
        return _cutoff_evidence(data, selected_case(), selected_setup.get())

    @render.ui
    def supporting_results() -> ui.Tag:
        req(stage.get() == "detail")
        return _supporting_results(data, selected_case(), selected_setup.get())


app = App(app_ui, _server, static_assets=STATIC_DIR)
