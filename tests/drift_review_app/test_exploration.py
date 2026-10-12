"""Counts route to exact archived contributors, with explicit units and Runs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from src.drift_review_app.data import ReviewData, load_review_data
from src.drift_review_app.exploration import (
    ExplorationQuery,
    QueryKind,
    core_value_breakdown,
    explore,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def data() -> ReviewData:
    return load_review_data(ROOT)


@pytest.mark.parametrize(
    ("kind", "count", "unit"),
    [
        ("corpus_personas", 204, "Personas"),
        ("corpus_entries", 1651, "Journal Entries"),
        ("cases", 292, "Persona/Core Value histories"),
        ("core_values", 10, "Core Values"),
        ("entry_value_cells", 2377, "entry/Core Value combinations"),
        ("reference_conflict", 269, "entry/Core Value combinations"),
        ("reference_not_conflict", 2106, "entry/Core Value combinations"),
        ("uncertain", 2, "entry/Core Value combinations"),
        ("reference_drifts", 42, "reference Drifts"),
        ("known_cases", 36, "Persona/Core Value histories"),
        ("no_known_cases", 256, "Persona/Core Value histories"),
        ("reference_same_week", 14, "reference Drifts"),
        ("reference_cross_week", 28, "reference Drifts"),
    ],
)
def test_corpus_counts_have_exact_contributors_and_units(
    data: ReviewData,
    kind: str,
    count: int,
    unit: str,
) -> None:
    result = explore(data, ExplorationQuery(cast(QueryKind, kind)))
    assert result.count == len(result.rows) == count
    assert result.unit == unit
    assert all(row.case_id in data.cases for row in result.rows)


def test_persona_and_entry_dedup_preserve_all_matching_histories(
    data: ReviewData,
) -> None:
    personas = explore(data, ExplorationQuery("corpus_personas"))
    entries = explore(data, ExplorationQuery("corpus_entries"))
    assert len({row.persona_id for row in personas.rows}) == personas.count
    assert len({(row.persona_id, row.t_index) for row in entries.rows}) == entries.count
    for row in (*personas.rows, *entries.rows):
        expected = {case.case_id for case in data.cases_for_persona(row.persona_id)}
        assert set(row.related_case_ids) == expected
        assert set(row.dimensions) == {data.cases[key].dimension for key in expected}
    assert any(len(row.related_case_ids) == 2 for row in personas.rows)
    bars = core_value_breakdown(data, ExplorationQuery("corpus_personas"))
    assert sum(bar.count for bar in bars) == 292
    assert sum(bar.count for bar in bars) != personas.count


@pytest.mark.parametrize(
    ("run", "found", "false", "covered"),
    [
        (1, 17, 2, 289),
        (2, 15, 2, 289),
        (3, 17, 1, 288),
    ],
)
def test_current_results_route_to_one_selected_run(
    data: ReviewData,
    run: int,
    found: int,
    false: int,
    covered: int,
) -> None:
    query = ExplorationQuery("found", run=run)
    assert explore(data, query).count == found
    assert explore(data, replace(query, kind="missed")).count == 42 - found
    assert explore(data, replace(query, kind="false")).count == false
    assert explore(data, replace(query, kind="detected")).count == found + false
    assert explore(data, replace(query, kind="covered")).count == covered
    assert explore(data, replace(query, kind="unresolved")).count == 292 - covered
    assert all(row.run == run for row in explore(data, query).rows)


def test_found_and_missed_are_reference_events_false_is_predicted(
    data: ReviewData,
) -> None:
    query = ExplorationQuery("reference_drifts", run=2)
    reference = explore(data, query)
    found = explore(data, replace(query, kind="found"))
    missed = explore(data, replace(query, kind="missed"))
    false = explore(data, replace(query, kind="false"))
    all_ids = {row.reference_id for row in reference.rows}
    found_ids = {row.reference_id for row in found.rows}
    missed_ids = {row.reference_id for row in missed.rows}
    assert found_ids.isdisjoint(missed_ids)
    assert found_ids | missed_ids == all_ids
    assert len(found_ids) == found.count
    for row in found.rows:
        prediction = next(
            span
            for span in data.predicted_drifts[("gpt6_low", 2, row.case_id)]
            if span.drift_id == row.drift_id
        )
        assert prediction.matched_reference_id == row.reference_id
        assert row.t_index == prediction.confirmation_t_index
    for row in false.rows:
        prediction = next(
            span
            for span in data.predicted_drifts[("gpt6_low", 2, row.case_id)]
            if span.drift_id == row.drift_id
        )
        assert prediction.result == "false Drift alert"
        assert row.reference_id is None
        assert row.span == (
            prediction.onset_t_index,
            prediction.confirmation_t_index,
            prediction.end_t_index,
        )
        assert row.period and row.period.startswith("pred|2|")
    assert all(row.period == "full" for row in found.rows)


def test_lagged_found_event_shows_matched_prediction_evidence(data: ReviewData) -> None:
    result = explore(data, ExplorationQuery("found", setup_key="luna_low", run=1))
    row = next(row for row in result.rows if row.case_id == "3a3b15e4:tradition")
    reference = next(
        span
        for span in data.reference_drifts[row.case_id]
        if span.drift_id == row.reference_id
    )
    prediction = next(
        span
        for span in data.predicted_drifts[("luna_low", 1, row.case_id)]
        if span.drift_id == row.drift_id
    )
    assert row.span == (3, 4, 4)
    assert row.span == (
        reference.onset_t_index,
        reference.confirmation_t_index,
        reference.end_t_index,
    )
    assert prediction.confirmation_t_index == row.t_index == 5
    assert prediction.confirmation_t_index > reference.end_t_index
    assert prediction.confirmation_t_index - reference.confirmation_t_index <= 2
    assert row.period == "full" and row.status == "found"
    assert "Matched alert confirmed at Entry 5" in row.label
    assert any(
        entry.t_index == row.t_index for entry in data.cases[row.case_id].entries
    )
    assert explore(data, ExplorationQuery("found")).count == 17


@pytest.mark.parametrize(
    "kind", ["found", "false", "reference_drifts", "conflict", "cases"]
)
def test_reference_and_core_value_filters_intersect_exactly(
    data: ReviewData, kind: str
) -> None:
    query = ExplorationQuery(
        cast(QueryKind, kind), dimension="tradition", reference_status="has"
    )
    result = explore(data, query)
    unfiltered = explore(data, replace(query, dimension="", reference_status="all"))
    expected = tuple(
        row
        for row in unfiltered.rows
        if row.dimension == "tradition" and data.reference_drifts[row.case_id]
    )
    assert result.rows == expected
    bars = core_value_breakdown(data, query)
    assert len(bars) == 1 and bars[0].count == result.count
    assert explore(data, bars[0].query).rows == result.rows


def test_zero_result_and_persona_filter(data: ReviewData) -> None:
    assert (
        explore(data, ExplorationQuery("known_cases", reference_status="none")).count
        == 0
    )
    assert explore(data, ExplorationQuery("found", reference_status="none")).rows == ()
    personas = explore(
        data, ExplorationQuery("corpus_personas", reference_status="has")
    )
    assert personas.count == 35
    row = next(row for row in personas.rows if len(row.related_case_ids) > 1)
    cases = explore(
        data,
        ExplorationQuery("cases", reference_status="has", persona_id=row.persona_id),
    )
    assert {case.case_id for case in cases.rows} == set(row.related_case_ids)


@pytest.mark.parametrize(
    "setup", ["gpt6_low", "gpt6_none", "luna_low", "definitions_off_low"]
)
def test_decision_contributors_partition_each_run(data: ReviewData, setup: str) -> None:
    for run in (1, 2, 3):
        populations = [
            explore(data, ExplorationQuery(kind, setup_key=setup, run=run))
            for kind in ("invalid", "abstain", "conflict", "not_conflict")
        ]
        coordinates = [
            (row.case_id, row.t_index) for result in populations for row in result.rows
        ]
        assert len(coordinates) == len(set(coordinates)) == 2377
        for result in populations:
            for row in result.rows:
                decision = data.decision(
                    setup, run, row.case_id, cast(int, row.t_index)
                )
                assert row.run == run
                if result.query.kind == "invalid":
                    assert decision.response_status != "ok"
                else:
                    assert decision.response_status == "ok"
                    assert decision.verdict == result.query.kind


def test_uncertain_labels_and_cross_week_events_keep_source_membership(
    data: ReviewData,
) -> None:
    uncertain = explore(data, ExplorationQuery("uncertain"))
    assert all(
        data.cases[row.case_id]
        .entries[
            next(
                index
                for index, entry in enumerate(data.cases[row.case_id].entries)
                if entry.t_index == row.t_index
            )
        ]
        .final_conflict
        is None
        for row in uncertain.rows
    )
    same = explore(data, ExplorationQuery("reference_same_week"))
    cross = explore(data, ExplorationQuery("reference_cross_week"))
    assert {row.reference_id for row in same.rows}.isdisjoint(
        row.reference_id for row in cross.rows
    )
    for result, expected in ((same, False), (cross, True)):
        for row in result.rows:
            reference = next(
                span
                for span in data.reference_drifts[row.case_id]
                if span.drift_id == row.reference_id
            )
            assert reference.crosses_week == expected
            assert row.period == (
                f"ref|{reference.drift_id}|"
                f"{reference.onset_t_index}|{reference.end_t_index}"
            )


@pytest.mark.parametrize("setup", ["definitions_off_low", "definitions_on_low"])
def test_definition_study_retains_detected_counts_without_accuracy_queries(
    data: ReviewData, setup: str
) -> None:
    counts = [22, 22, 21] if setup == "definitions_off_low" else [19, 21, 18]
    for run, expected in enumerate(counts, start=1):
        result = explore(data, ExplorationQuery("detected", setup_key=setup, run=run))
        assert result.count == expected
        assert all(
            row.status == "detected" and row.reference_id is None for row in result.rows
        )
    for kind in ("found", "missed", "false"):
        with pytest.raises(ValueError, match="accuracy queries are unavailable"):
            explore(data, ExplorationQuery(kind, setup_key=setup))
    assert explore(data, ExplorationQuery("invalid", setup_key=setup)).count > 0


@pytest.mark.parametrize(
    "change",
    [
        {"run": 0},
        {"run": 4},
        {"run": True},
        {"dimension": "invented"},
        {"setup_key": "invented"},
        {"reference_status": "invented"},
        {"persona_id": "invented"},
    ],
)
def test_query_rejects_unknown_or_unpreserved_selection(
    data: ReviewData, change: dict
) -> None:
    with pytest.raises(ValueError):
        explore(data, replace(ExplorationQuery("cases"), **change))
