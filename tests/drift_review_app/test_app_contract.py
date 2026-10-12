from __future__ import annotations

import re
from functools import lru_cache
from hashlib import sha256
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

from src.drift_review_app.app import (
    ABSTAIN_EXPLANATIONS,
    CURRENT_SETUP_KEY,
    EXPLAINER_SCRIPT_VERSION,
    STYLESHEET_VERSION,
    _contributor_screen,
    _cutoff_evidence,
    _data_breakdowns,
    _decision_cell,
    _default_inspection_point,
    _default_period,
    _detail_screen,
    _dimension_choices,
    _filter_screen,
    _how_it_works_panel,
    _period_choices,
    _prompt_explanation,
    _provenance_summary,
    _reference_cell,
    _scoreboard,
    _scoreboard_cell,
    _supporting_results,
    _timeline,
    _trajectory_comparison,
    _trajectory_review,
    _visible_entries,
    app_ui,
)
from src.drift_review_app.data import ReviewData, load_review_data
from src.drift_review_app.exploration import ExplorationQuery, explore

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _review_data() -> ReviewData:
    return load_review_data(ROOT)


class _CountParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.buttons: list[tuple[dict[str, str], str]] = []
        self.current: dict[str, str] | None = None
        self.text = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key: value or "" for key, value in attrs}
        if tag == "button" and "data-exploration-kind" in values:
            self.current = values
            self.text = ""

    def handle_data(self, data: str) -> None:
        if self.current is not None:
            self.text += data

    def handle_endtag(self, tag: str) -> None:
        if tag == "button" and self.current is not None:
            number = re.search(r"[\d,]+", self.text)
            assert number is not None
            self.buttons.append((self.current, number.group()))
            self.current = None


def test_contributor_pagination_and_multiple_core_values_preserve_membership() -> None:
    data = _review_data()
    query = ExplorationQuery("corpus_personas")
    result = explore(data, query)
    first = str(_contributor_screen(data, query, result, page=0))
    last = str(_contributor_screen(data, query, result, page=4))
    assert result.count == 204
    assert first.count('class="contributor-row"') == 50
    assert last.count('class="contributor-row"') == 4
    assert 'id="next_contributors"' in first
    two_values = next(row for row in result.rows if len(row.related_case_ids) == 2)
    page = result.rows.index(two_values) // 50
    html = str(_contributor_screen(data, query, result, page=page))
    for case_id in two_values.related_case_ids:
        assert f'data-inspect-case-id="{case_id}"' in html


def test_core_value_contributors_open_histories_without_arbitrary_personas() -> None:
    data = _review_data()
    query = ExplorationQuery("core_values")
    result = explore(data, query)
    html = str(_contributor_screen(data, query, result))
    assert result.count == 10
    assert "Universalism" in html and "Achievement" in html
    assert "Inspect contributing histories" in html
    assert "data-inspect-case-id" not in html


def test_entry_and_span_contributors_retain_explicit_positions_and_run() -> None:
    data = _review_data()
    query = ExplorationQuery("missed", run=2)
    result = explore(data, query)
    html = str(_contributor_screen(data, query, result))
    assert result.count == 27
    assert "Run 2" in html
    assert html.count('class="contributor-row"') == 27
    for row in result.rows:
        assert row.run == 2
        assert row.t_index in {
            entry.t_index
            for entry in _visible_entries(data.cases[row.case_id], row.period)
        }


def test_reference_and_predicted_periods_show_the_selected_span() -> None:
    data = _review_data()
    case = data.cases["02fb94f3:tradition"]
    choices = _period_choices(data, case, "luna_low")
    reference = next(key for key in choices if key.startswith("ref|"))
    predicted = next(key for key in choices if key.startswith("pred|"))
    assert [entry.t_index for entry in _visible_entries(case, reference)] == [4, 5, 6]
    assert [entry.t_index for entry in _visible_entries(case, predicted)] == [4, 5]


def test_full_timeline_is_selected_before_narrower_evidence_periods() -> None:
    data = _review_data()
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

    assert _default_period(data, known_drift_case, "luna_low") == "full"
    assert _default_period(data, false_alert_case, "luna_low") == "full"
    assert _default_period(data, quiet_case, "luna_low") == "full"


def test_main_page_prioritizes_data_controls_and_prunes_model_comparisons() -> None:
    data = _review_data()
    html = str(app_ui) + str(_filter_screen(data))
    dependency_html = "".join(str(dep.head or "") for dep in app_ui.get_dependencies())
    assert "Inspect the synthetic data and Drift evidence" in html
    for name in (
        "report_prompt",
        "landing_setup",
        "landing_run",
        "reference_drift_filter",
        "core_value_filter",
        "show_personas",
    ):
        assert f'id="{name}"' in html
    assert 'value="all" selected' in html
    assert 'value="gpt6_low" selected' in html
    assert 'id="data_breakdowns"' in html
    assert 'id="prompt_explanation"' in html
    assert f'href="styles.css?v={STYLESHEET_VERSION}"' in dependency_html
    assert f'src="drift_explainer.js?v={EXPLAINER_SCRIPT_VERSION}"' in dependency_html
    assert (
        STYLESHEET_VERSION
        == sha256(
            (ROOT / "src/drift_review_app/static/styles.css").read_bytes()
        ).hexdigest()[:12]
    )
    assert (
        EXPLAINER_SCRIPT_VERSION
        == sha256(
            (ROOT / "src/drift_review_app/static/drift_explainer.js").read_bytes()
        ).hexdigest()[:12]
    )
    for removed in (
        "Table 12",
        "Table 11",
        "Calculated token cost",
        "Current contract:",
        "operating-points.png",
        "per-value.png",
        "figure-09-corpus.png",
    ):
        assert removed not in html
    assert "figure-10-matching.png" in html
    assert "Known Drift status" in html and "All Core Values" in html
    assert "MutationObserver" in html and "review_password" not in html


def test_how_it_works_animation_preserves_the_input_contract() -> None:
    html = unescape(str(_how_it_works_panel()))
    script = (ROOT / "src/drift_review_app/static/drift_explainer.js").read_text()
    styles = (ROOT / "src/drift_review_app/static/styles.css").read_text()

    assert html.count('class="explainer-step"') == 8
    assert html.count('role="tab"') == 2
    assert 'data-part-target="development"' in html
    assert 'data-part-target="deployment"' in html
    assert "Synthetic development" in html
    assert "Intended deployed flow" in html
    assert "not yet deployment-approved" in html
    assert "The Drift Detector—not the Weekly Drift Reviewer—marks Drift" in html
    assert "LLM-Judge Conflict Labels or known Drift" in html
    assert "Never enters the prompt" in html
    assert "LLM-Judge Conflict Labels · known Drift · VIF Critic Predictions" in html
    assert "Create LLM-Judge Conflict Labels separately" in html
    assert "Compare only after decisions are complete" in html
    assert "Read the complete input contract" in html
    assert 'aria-live="polite"' in html
    assert html.count('data-step-target="') == 8

    assert "IntersectionObserver" in script
    assert "prefers-reduced-motion: reduce" in script
    assert 'event.key !== "ArrowLeft"' in script
    assert 'button.setAttribute("aria-current", "step")' in script
    assert "STEP_DELAY_MS" in script
    assert "@media (prefers-reduced-motion: reduce)" in styles
    assert "animation-duration: 0.01ms" in styles


def test_core_value_counts_are_scoped_to_known_drift_status() -> None:
    data = _review_data()
    with_drift = _dimension_choices(data, "has")
    without_drift = _dimension_choices(data, "none")
    assert with_drift["achievement"] == "Achievement (0)"
    assert with_drift["universalism"] == "Universalism (11)"
    assert without_drift["universalism"] == "Universalism (21)"


def test_every_data_breakdown_count_routes_to_its_exact_population() -> None:
    data = _review_data()
    query = ExplorationQuery("cases")
    html = str(_data_breakdowns(data, query))
    parser = _CountParser()
    parser.feed(html)
    assert len(parser.buttons) > 50
    for attrs, text in parser.buttons:
        kind = attrs["data-exploration-kind"]
        dimension = attrs.get("data-exploration-dimension", "") or ""
        expected = explore(data, ExplorationQuery(kind, dimension=dimension))
        assert int(text.replace(",", "")) == expected.count
        assert "Shiny.setInputValue('inspect_query'" in attrs["onclick"]
        assert str(expected.count) in attrs["aria-label"].replace(",", "")
    assert "17" in html and "42" in html and "204" in html and "1,651" in html
    assert "median" not in html.lower() and "token cost" not in html.lower()


def test_prompt_explanation_keeps_versions_without_static_results() -> None:
    initial = unescape(str(_prompt_explanation("v0")))
    tightened = unescape(str(_prompt_explanation("v1")))
    definitions = unescape(str(_prompt_explanation("v2")))
    assert "2.0 / 3.0 / 4.0" in initial
    assert "Low-confidence cases become Abstain" in tightened
    assert "definition and core motivation" in definitions
    assert "<table" not in initial + tightened + definitions
    data = _review_data()
    html = str(
        _data_breakdowns(
            data, ExplorationQuery("cases", setup_key="definitions_off_low")
        )
    )
    assert 'data-exploration-kind="detected"' in html
    for kind in ("found", "missed", "false"):
        assert f'data-exploration-kind="{kind}"' not in html


def test_selected_setup_controls_verified_prompt_cutoffs() -> None:
    data = _review_data()
    case = data.cases["02fb94f3:tradition"]
    historical = str(_cutoff_evidence(data, case, "luna_low"))
    current = str(_cutoff_evidence(data, case, "gpt6_low"))
    historical_hash = data.boundaries_for_persona(case.persona_id, "luna_low")[
        0
    ].prompt_sha256[:12]
    current_hash = data.boundaries_for_persona(case.persona_id, "gpt6_low")[
        0
    ].prompt_sha256[:12]
    assert historical_hash != current_hash
    assert historical_hash in historical and current_hash in current


def test_definition_study_trajectory_has_no_unreported_accuracy_labels() -> None:
    data = _review_data()
    case = data.cases["0d42f985:benevolence"]
    index, run = _default_inspection_point(data, case, "definitions_off_low", "full")
    html = unescape(
        str(_trajectory_review(data, case, "definitions_off_low", "full", index, run))
    )
    for phrase in (
        "False Drift alert",
        "false Drift alert",
        "Missed known Drift",
        "matched known Drift",
    ):
        assert phrase not in html
    assert "Detected Drift" in html
    assert "LLM-Judge Conflict Label" in html


def test_detail_controls_have_overflow_safe_desktop_structure() -> None:
    data = _review_data()
    case = data.cases["02fb94f3:tradition"]
    html = str(_detail_screen(data, case))
    scoreboard = str(_scoreboard(data, case))
    styles = (ROOT / "src/drift_review_app/static/styles.css").read_text()
    for class_name in ("setup-control", "period-control"):
        assert class_name in html
    assert "show_reference_labels" not in html
    assert "What the selected Weekly Drift Reviewer found" in scoreboard
    assert "Inspect the three preserved Runs" in scoreboard
    assert "Current contract" in scoreboard
    assert "GPT-6 Luna — reasoning low" in html
    assert "fixed model contract" not in html.lower()
    assert 'class="scoreboard-comparison"' in scoreboard
    assert "How it works" not in html
    assert "FROZEN WEEKLY DRIFT REVIEWER INPUT CONTRACT" not in html
    assert 'id="cutoff_evidence"' in html
    assert 'id="detail_experiment"' in html
    assert 'id="persona_scoreboard"' in html
    assert "Journal Entry trajectory and evidence" in html
    assert 'aria-current="step"' in html
    assert '"setup"' in styles
    assert "grid-template-columns: repeat(3, minmax(0, 1fr))" in styles
    assert "grid-template-columns: minmax(23rem, 0.38fr) minmax(0, 0.62fr)" in styles
    radio_start = styles.index(".review-controls .radio-inline {")
    radio_end = styles.index("}", radio_start)
    assert "white-space: normal" in styles[radio_start:radio_end]


def test_visual_direction_keeps_bokeh_behind_crisp_evidence_panels() -> None:
    styles = (ROOT / "src/drift_review_app/static/styles.css").read_text()

    assert '--font-display: "Avenir Next"' in styles
    assert ".app-shell::before" in styles
    assert styles.count("radial-gradient(") >= 6
    assert "filter: blur(42px)" in styles
    assert ".filter-panel::before" in styles
    assert ".evidence-library" in styles
    assert ".scoreboard-comparison" in styles
    assert ".timeline-row" in styles


def test_timeline_uses_semantic_roles_and_collapses_decision_evidence() -> None:
    data = _review_data()
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


def test_trajectory_review_links_the_plot_to_selected_evidence() -> None:
    data = _review_data()
    styles = (ROOT / "src/drift_review_app/static/styles.css").read_text()
    case = data.cases["02fb94f3:tradition"]
    t_index, run = _default_inspection_point(data, case, "luna_low", "full")
    html = unescape(
        str(
            _trajectory_review(
                data,
                case,
                "luna_low",
                "full",
                t_index,
                run,
            )
        )
    )

    assert 'class="trajectory-review-grid"' in html
    assert html.count('class="trajectory-run-plot ') == 3
    assert html.count('class="trajectory-point comparison-') == len(case.entries) * 3
    assert html.count("trajectory-run-line") == 3
    assert "Three date-aligned Weekly Drift Reviewer Run plots" in html
    assert "Rows are categories, not a numeric scale" in html
    assert "Matches resolved LLM-Judge Conflict Label" in html
    assert "Differs from resolved label, including Abstain" in html
    assert "Invalid response or unresolved label" in html
    assert "Known Drift" in html
    assert "LLM-Judge Conflict Label" in html
    assert f"Journal Entry {t_index}" in html
    assert f"Run {run}" in html
    assert 'aria-pressed="true"' in html
    assert "Compare Runs" not in html
    assert "run-switcher" not in html
    assert ".trajectory-point.run-2" not in styles
    assert ".trajectory-point.run-3" not in styles
    assert ".trajectory-point.comparison-mismatch::after" in styles
    assert "animation: trajectory-mismatch-pulse" in styles
    assert ".trajectory-point.comparison-neutral" in styles
    assert ".run-switcher" not in styles
    assert "Full Journal Entry and Run comparison" in html
    assert 'role="table"' in html


def test_trajectory_scores_resolved_label_abstains_as_mismatches() -> None:
    data = _review_data()
    lukas = data.cases["d7a0683f:universalism"]
    conflict_entry = next(entry for entry in lukas.entries if entry.t_index == 1)
    abstain = data.decision("luna_low", 1, lukas.case_id, 1)

    comparison, note = _trajectory_comparison(conflict_entry, abstain)

    assert conflict_entry.final_conflict is True
    assert abstain.verdict == "abstain"
    assert comparison == "mismatch"
    assert "Abstain differs from the resolved Conflict" in note


def test_trajectory_keeps_unscorable_comparisons_neutral() -> None:
    data = _review_data()
    invalid = next(
        decision
        for decision in data.decisions.values()
        if decision.response_status != "ok"
    )
    invalid_case = data.cases[invalid.case_id]
    invalid_entry = next(
        entry for entry in invalid_case.entries if entry.t_index == invalid.t_index
    )
    unresolved_case = next(
        case
        for case in data.cases.values()
        if any(entry.final_conflict is None for entry in case.entries)
    )
    unresolved_entry = next(
        entry for entry in unresolved_case.entries if entry.final_conflict is None
    )
    resolved_decision = next(
        data.decision(setup_key, run, unresolved_case.case_id, unresolved_entry.t_index)
        for setup_key in data.setup_specs
        for run in (1, 2, 3)
        if data.decision(
            setup_key, run, unresolved_case.case_id, unresolved_entry.t_index
        ).response_status
        == "ok"
    )

    assert _trajectory_comparison(invalid_entry, invalid)[0] == "neutral"
    assert _trajectory_comparison(unresolved_entry, resolved_decision)[0] == "neutral"


def test_trajectory_marks_false_missed_and_invalid_outcomes() -> None:
    data = _review_data()
    lukas = data.cases["d7a0683f:universalism"]
    t_index, run = _default_inspection_point(data, lukas, "luna_low", "full")
    lukas_html = unescape(
        str(
            _trajectory_review(
                data,
                lukas,
                "luna_low",
                "full",
                t_index,
                run,
            )
        )
    )

    assert 'class="trajectory-alert-bracket is-false"' in lukas_html
    assert 'class="trajectory-alert-label is-false"' in lukas_html
    assert "Run 1 · False Drift alert · t9–t10" in lukas_html
    assert (
        "Run 1 · Journal Entry 10 · Decision: Conflict · LLM-Judge: Not Conflict"
    ) in lukas_html

    mini_html = unescape(
        str(
            _trajectory_review(
                data,
                lukas,
                "mini_none",
                "full",
                t_index,
                run,
            )
        )
    )
    assert mini_html.count('class="trajectory-alert-label is-missed') == 3
    assert "Missed known Drift" in mini_html

    invalid = next(
        decision
        for decision in data.decisions.values()
        if decision.response_status != "ok"
    )
    invalid_case = data.cases[invalid.case_id]
    invalid_html = unescape(
        str(
            _trajectory_review(
                data,
                invalid_case,
                invalid.setup_key,
                "full",
                invalid.t_index,
                invalid.run,
            )
        )
    )
    assert "trajectory-alert-label is-invalid is-point" in invalid_html
    assert f"Run {invalid.run} · Invalid response · t{invalid.t_index}" in invalid_html


def test_trajectory_separates_nearby_rare_annotation_labels() -> None:
    data = _review_data()
    lukas_vetter = data.cases["a24b8d8f:universalism"]
    t_index, run = _default_inspection_point(data, lukas_vetter, "luna_low", "full")
    html = unescape(
        str(
            _trajectory_review(
                data,
                lukas_vetter,
                "luna_low",
                "full",
                t_index,
                run,
            )
        )
    )

    assert html.count("--annotation-lane: 0;") == 3
    assert html.count("--annotation-lane: 1;") == 3
    for run_number in (1, 2, 3):
        assert f"Run {run_number} · False Drift alert · t1–t2" in html
        assert f"Run {run_number} · Missed known Drift · t3–t4" in html


def test_every_abstain_reason_has_a_plain_english_explanation() -> None:
    data = _review_data()
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
    data = _review_data()
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
    data = _review_data()
    case = data.cases["1f86f569:conformity"]
    html = unescape(str(_scoreboard(data, case)))

    assert "1 known Drift" in html
    assert "t1–t2" in html
    assert html.count("Missed 1 Drift") == 3
    assert html.count("No Drift alert") == 3
    assert data.setup_specs[CURRENT_SETUP_KEY].label in html
    assert data.setup_specs["gpt56_low"].label not in html


def test_scoreboard_preserves_partial_drift_hits() -> None:
    data = _review_data()
    case = data.cases["02fb94f3:tradition"]
    html = unescape(str(_scoreboard(data, case, "luna_low")))

    assert "2 known Drifts" in html
    assert "Found 1/2 · missed 1" in html
    assert "t4–t5 · matched known Drift" in html
    assert "How hits are counted" in html
    assert "This affects scoring, not the Drift definition" in html


def test_persona_provenance_and_detection_delay_are_explained() -> None:
    data = _review_data()
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
    data = _review_data()
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
