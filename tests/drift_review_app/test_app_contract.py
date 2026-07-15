from __future__ import annotations

from hashlib import sha256
from html import unescape
from pathlib import Path

from src.drift_review_app.app import (
    ABSTAIN_EXPLANATIONS,
    ALL_CASES_FOCUS,
    STYLESHEET_VERSION,
    _at_a_glance,
    _decision_cell,
    _default_period,
    _detail_screen,
    _dimension_choices,
    _filter_screen,
    _matching_cases,
    _period_choices,
    _personas_screen,
    _provenance_summary,
    _reference_cell,
    _scoreboard,
    _scoreboard_cell,
    _supporting_results,
    _timeline,
    _visible_entries,
    app_ui,
)
from src.drift_review_app.data import load_review_data

ROOT = Path(__file__).resolve().parents[2]


def test_reference_and_predicted_periods_show_the_selected_span() -> None:
    data = load_review_data(ROOT)
    case = data.cases["02fb94f3:tradition"]
    choices = _period_choices(data, case, "luna_low")
    reference = next(key for key in choices if key.startswith("ref|"))
    predicted = next(key for key in choices if key.startswith("pred|"))
    assert [entry.t_index for entry in _visible_entries(case, reference)] == [4, 5, 6]
    assert [entry.t_index for entry in _visible_entries(case, predicted)] == [4, 5]


def test_relevant_drift_period_is_selected_before_full_timeline() -> None:
    data = load_review_data(ROOT)
    known_drift_case = data.cases["02fb94f3:tradition"]
    false_alert_case = data.cases["f6180c27:benevolence"]
    quiet_case = next(
        case
        for case in data.cases.values()
        if not data.reference_drifts[case.case_id]
        and all(
            not data.predicted_drifts[("luna_low", run, case.case_id)]
            for run in (1, 2, 3)
        )
    )

    assert _default_period(data, known_drift_case, "luna_low").startswith("ref|")
    assert _default_period(data, false_alert_case, "luna_low").startswith("pred|")
    assert _default_period(data, quiet_case, "luna_low") == "full"


def test_main_page_opens_on_two_filters_without_password_gate() -> None:
    data = load_review_data(ROOT)
    shell_html = str(app_ui)
    dependency_html = "".join(
        str(dependency.head or "") for dependency in app_ui.get_dependencies()
    )
    html = shell_html + str(_filter_screen(data))
    assert "Drift inspection app" in html
    assert f'href="styles.css?v={STYLESHEET_VERSION}"' in dependency_html
    assert (
        STYLESHEET_VERSION
        == sha256(
            (ROOT / "src/drift_review_app/static/styles.css").read_bytes()
        ).hexdigest()[:12]
    )
    assert "204 synthetic personas" not in shell_html
    for input_id in (
        "reference_drift_filter",
        "core_value_filter",
        "review_focus_filter",
        "show_personas",
    ):
        assert f'id="{input_id}"' in html
    for input_id in ("persona_id", "period", "setup"):
        assert f'id="{input_id}"' not in html
    assert "Journal Entries stay hidden until you choose a persona" in html
    assert "How it works" in html
    assert "FROZEN WEEKLY DRIFT REVIEWER INPUT CONTRACT" in html
    assert "gpt-5.6-sol lanes at reasoning xhigh" in html
    assert "Known Drift status" in html
    assert "Achievement (0)" in html
    assert "Advanced review focus" in html
    assert "At a glance" in html
    assert html.index("At a glance") < html.index("How it works")
    assert "WHAT COUNTS AS CONFLICT" in html
    assert "A Journal Entry is a Conflict when the displayed text clearly" in html
    assert "WHAT DOES NOT COUNT" in html
    assert "Two consecutive Conflicts for the same Core Value form one Drift" in html
    assert "about four days slower" in html
    assert "MutationObserver" in html
    assert "heading === window.__twinklPreviousStageHeading" in html
    assert "focus({preventScroll: true})" in html
    assert "setTimeout" not in html
    assert "review_password" not in html
    assert "sign_in" not in html


def test_reference_drift_and_core_value_filter_cases() -> None:
    data = load_review_data(ROOT)
    with_drift = _matching_cases(data, "has", "universalism")
    without_drift = _matching_cases(data, "none", "universalism")
    assert len(with_drift) == 11
    assert len(without_drift) == 21
    assert all(data.reference_drifts[case.case_id] for case in with_drift)
    assert all(not data.reference_drifts[case.case_id] for case in without_drift)
    assert all(case.dimension == "universalism" for case in with_drift + without_drift)


def test_core_value_counts_and_review_focus_are_applied() -> None:
    data = load_review_data(ROOT)
    with_drift = _dimension_choices(data, "has")
    without_drift = _dimension_choices(data, "none")
    assert with_drift["achievement"] == "Achievement (0)"
    assert with_drift["universalism"] == "Universalism (11)"
    assert without_drift["universalism"] == "Universalism (21)"

    missed = _matching_cases(data, "has", "universalism", "Missed Drift")
    all_cases = _matching_cases(data, "has", "universalism", ALL_CASES_FOCUS)
    assert missed
    assert set(missed) < set(all_cases)
    assert all(
        case.case_id in data.queue_case_ids("Missed Drift", "luna_low")
        for case in missed
    )


def test_at_a_glance_combines_dataset_llms_and_results() -> None:
    data = load_review_data(ROOT)
    html = unescape(str(_at_a_glance(data)))

    assert "Dataset" in html
    assert "204 synthetic personas" in html
    assert "35 with at least one known Drift" in html
    assert "169 with none" in html
    assert "292 persona/Core Value cases" in html
    assert "36 with known Drift" in html
    assert "256 with no known Drift" in html
    assert "42 known Drifts across the 36 cases" in html
    assert "1,651" in html
    assert "2,377" in html
    assert "269 Conflict" in html
    assert "2,106 Not Conflict" in html
    assert "28 cross-week · 14 same-week" in html

    assert "LLMs used" in html
    for model_url in (
        "https://developers.openai.com/api/docs/models/gpt-5.4-mini",
        "https://developers.openai.com/api/docs/models/gpt-5.6-luna",
        "https://developers.openai.com/api/docs/models/gpt-5.6-sol",
        "https://platform.claude.com/docs/en/about-claude/models/overview",
    ):
        assert model_url in html
    assert html.count('target="_blank"') == 4
    assert html.count('rel="noopener noreferrer"') == 4

    assert "Results" in html
    assert "No Core Value or known Drift filter has been applied" in html
    assert "24/42 hits" in html
    assert "23/42 hits" in html
    assert "Current development selection" in html
    assert "Why Luna at reasoning low is selected for development" in html
    assert "95% interval from −0.071 to +0.205" in html
    assert "observed Drift recall gain is not statistically clear" in html


def test_persona_results_use_one_list_heading() -> None:
    data = load_review_data(ROOT)
    cases = _matching_cases(data, "has", "conformity")
    html = str(_personas_screen(data, cases, "has", "conformity"))
    assert html.count("MATCHING PERSONAS") == 1
    assert "At a glance" not in html
    assert "24/42 hits" not in html


def test_detail_controls_have_overflow_safe_desktop_structure() -> None:
    data = load_review_data(ROOT)
    case = data.cases["02fb94f3:tradition"]
    html = str(_detail_screen(data, case))
    styles = (ROOT / "src/drift_review_app/static/styles.css").read_text()
    for class_name in ("setup-control", "period-control"):
        assert class_name in html
    assert "show_reference_labels" not in html
    assert "What each Weekly Drift Reviewer found" in html
    assert "How it works" not in html
    assert "FROZEN WEEKLY DRIFT REVIEWER INPUT CONTRACT" not in html
    assert "Verified Weekly Drift Reviewer input cutoffs" in html
    assert "Journal Entries and three Runs" in html
    assert 'aria-current="step"' in html
    assert '"setup"' in styles
    assert "grid-template-columns: repeat(3, minmax(0, 1fr))" in styles
    assert "white-space: nowrap" not in styles


def test_filter_spacing_overrides_bootstrap_generated_rules() -> None:
    styles = (ROOT / "src/drift_review_app/static/styles.css").read_text()

    assert (
        ".filter-form .control-label {\n"
        "    margin-bottom: var(--space-md);\n"
        "}" in styles
    )
    assert (
        ".filter-form .shiny-input-radiogroup .shiny-options-group {\n"
        "    margin-top: 0;\n"
        "}" in styles
    )


def test_contract_dividers_stay_aligned() -> None:
    styles = (ROOT / "src/drift_review_app/static/styles.css").read_text()

    assert ".boundary-grid,\n.contract-notes" in styles
    assert "grid-template-columns: repeat(2, minmax(0, 1fr))" in styles


def test_timeline_uses_semantic_roles_and_collapses_decision_evidence() -> None:
    data = load_review_data(ROOT)
    case = data.cases["02fb94f3:tradition"]
    timeline_html = str(_timeline(data, case, "luna_low", "full"))
    assert 'role="table"' in timeline_html
    assert timeline_html.count('role="columnheader"') == 5
    assert timeline_html.count('role="row"') == len(case.entries) + 1
    assert timeline_html.count('role="cell"') == len(case.entries) * 5
    assert "LLM-Judge Conflict Label" in timeline_html
    assert "gpt-5.6-sol · reasoning xhigh" in timeline_html

    entry = next(
        entry
        for entry in case.entries
        if data.decision("luna_low", 1, case.case_id, entry.t_index).response_status
        == "ok"
        and data.decision("luna_low", 1, case.case_id, entry.t_index).verdict
        != "abstain"
        and data.decision("luna_low", 1, case.case_id, entry.t_index).evidence_quote
    )
    decision_html = str(_decision_cell(data, case, entry, "luna_low", 1))
    assert "decision-details" in decision_html
    assert "View evidence" in decision_html
    assert "Evidence from the Journal Entry" in decision_html
    assert "Reason code" not in decision_html
    assert 'class="confidence"' not in decision_html

    abstain_entry = next(
        entry
        for entry in case.entries
        if data.decision("luna_low", 1, case.case_id, entry.t_index).verdict
        == "abstain"
    )
    abstain_decision = data.decision("luna_low", 1, case.case_id, abstain_entry.t_index)
    abstain_html = str(_decision_cell(data, case, abstain_entry, "luna_low", 1))
    assert "Why the Weekly Drift Reviewer abstained" in abstain_html
    assert abstain_decision.reason_code not in abstain_html

    routine_entry = next(
        entry
        for entry in case.entries
        if entry.resolution_status == "resolved"
        and entry.resolution_method == "agreement"
    )
    reference_html = str(_reference_cell(data, case, routine_entry))
    assert "resolved · agreement" not in reference_html


def test_every_abstain_reason_has_a_plain_english_explanation() -> None:
    data = load_review_data(ROOT)
    reason_codes = {
        decision.reason_code
        for decision in data.decisions.values()
        if decision.verdict == "abstain" and decision.reason_code
    }

    assert reason_codes <= ABSTAIN_EXPLANATIONS.keys()
    assert "direct_behavior_or_choice" in reason_codes
    assert (
        "direct_behavior_or_choice"
        not in ABSTAIN_EXPLANATIONS["direct_behavior_or_choice"]
    )


def test_invalid_response_always_explains_fail_closed_behavior() -> None:
    data = load_review_data(ROOT)
    decision = next(
        decision
        for decision in data.decisions.values()
        if decision.response_status != "ok" and decision.validation_error
    )
    case = data.cases[decision.case_id]
    entry = next(row for row in case.entries if row.t_index == decision.t_index)
    html = unescape(
        str(_decision_cell(data, case, entry, decision.setup_key, decision.run))
    )
    assert "weekly receipt failed validation" in html
    assert "parsed decisions were ignored" in html
    assert "Technical validation error" in html
    assert decision.validation_error in html


def test_erik_scoreboard_makes_the_missed_reference_drift_explicit() -> None:
    data = load_review_data(ROOT)
    case = data.cases["1f86f569:conformity"]
    html = unescape(str(_scoreboard(data, case)))

    assert "1 known Drift" in html
    assert "t1–t2" in html
    assert html.count("Missed 1 Drift") == 9
    assert html.count("No Drift alert") == 9
    for spec in data.setup_specs.values():
        assert spec.label in html


def test_scoreboard_preserves_partial_drift_hits() -> None:
    data = load_review_data(ROOT)
    case = data.cases["02fb94f3:tradition"]
    html = unescape(str(_scoreboard(data, case)))

    assert "2 known Drifts" in html
    assert "Found 1/2 · missed 1" in html
    assert "t4–t5 · matched known Drift" in html
    assert "How hits are counted" in html
    assert "This affects scoring, not the Drift definition" in html


def test_persona_provenance_and_detection_delay_are_explained() -> None:
    data = load_review_data(ROOT)
    training_case = next(
        case for case in data.cases.values() if case.historical_split == "training"
    )
    non_training_case = next(
        case for case in data.cases.values() if case.historical_split != "training"
    )
    assert _provenance_summary(training_case)[0].startswith("Yes")
    assert _provenance_summary(non_training_case)[0].startswith("No")

    detail_html = str(_detail_screen(data, training_case))
    assert "VIF Critic training" in detail_html
    assert "included in historical VIF Critic training" in detail_html
    assert "View source identifiers" in detail_html

    supporting_html = str(_supporting_results(data, training_case, "luna_low"))
    assert "predicted detection date minus the known Drift confirmation date" in (
        supporting_html
    )


def test_no_drift_scoreboard_distinguishes_resolved_from_unresolved() -> None:
    data = load_review_data(ROOT)
    unresolved_case = next(
        case
        for case in data.cases.values()
        if not data.reference_drifts[case.case_id]
        and not data.case_metrics[("luna_low", 1, case.case_id)]["covered"]
        and not data.case_metrics[("luna_low", 1, case.case_id)]["false_drift_alerts"]
    )
    resolved_case = next(
        case
        for case in data.cases.values()
        if not data.reference_drifts[case.case_id]
        and data.case_metrics[("luna_low", 1, case.case_id)]["covered"]
        and not data.case_metrics[("luna_low", 1, case.case_id)]["false_drift_alerts"]
    )
    unresolved = unescape(str(_scoreboard_cell(data, unresolved_case, "luna_low", 1)))
    resolved = unescape(str(_scoreboard_cell(data, resolved_case, "luna_low", 1)))
    assert "No Drift alert · review incomplete" in unresolved
    assert "At least one adjacent pair remains unresolved" in unresolved
    assert "is-abstain" in unresolved
    assert "No Drift alert · all adjacent pairs ruled out" in resolved
    assert "is-hit" in resolved


def test_railway_package_has_no_provider_dependency_or_password_gate() -> None:
    requirements = (ROOT / "requirements-review-app.txt").read_text()
    dockerfile = (ROOT / "Dockerfile.review_app").read_text()
    railway = (ROOT / "railway.json").read_text()
    app_source = (ROOT / "src/drift_review_app/app.py").read_text()
    launch_guide = (ROOT / "docs/demo/weekly_drift_review_app.md").read_text()
    assert "openai" not in requirements.lower()
    assert "google" not in requirements.lower()
    assert "OPENAI_API_KEY" not in app_source
    assert "GEMINI_API_KEY" not in app_source
    assert "TWINKL_REVIEW_PASSWORD" not in app_source + dockerfile + launch_guide
    assert "Shared review password" not in app_source
    assert "Drift inspection app" in app_source
    assert "Read the evidence, not the average." not in app_source
    assert "0.0.0.0" in dockerfile and "${PORT:-8000}" in dockerfile
    assert '"dockerfilePath": "Dockerfile.review_app"' in railway
