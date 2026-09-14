"""Selected Coach evidence preserves the displayed context around decisions."""

import json

import pytest

from src.coach.schemas import CoachNarrative
from src.coach.weekly_digest import (
    _evidence_from_decisions,
    build_weekly_drift_reviewer_digest_from_entries,
    render_digest_messages,
)
from src.drift_detector import detect_drift
from src.evals.coach_narrative_judge import render_judge_prompt
from src.weekly_drift_reviewer import (
    WeeklyDriftReviewerDecision,
    WeeklyDriftReviewerEntry,
)


def _decision(index, date, *, conflict=False):
    return WeeklyDriftReviewerDecision(
        persona_id="casey",
        week_start="2025-01-06",
        week_end="2025-01-12",
        t_index=index,
        date=date,
        core_value="benevolence",
        verdict="conflict" if conflict else "not_conflict",
        confidence="high",
        reason_code=(
            "direct_behavior_or_choice"
            if conflict
            else "direct_aligned_or_neutral_behavior"
        ),
        evidence_quote="I declined it." if conflict else "",
        review_status="ok",
    )


def _digest(entries, decisions, *, result=None, evidence_policy="complete"):
    return build_weekly_drift_reviewer_digest_from_entries(
        persona_id="casey",
        persona_name="Casey",
        week_start="2025-01-06",
        week_end="2025-01-12",
        core_values=["benevolence"],
        entries=entries,
        decisions=decisions,
        drift_result=result or detect_drift(decisions, persona_id="casey"),
        evidence_policy=evidence_policy,
    )


@pytest.mark.parametrize("conflict", [False, True])
def test_selected_evidence_keeps_antecedents_late_decisions_and_shared_inputs(conflict):
    text = (
        "My sister asked me to collect her child from school. "
        + "I thought about the request while walking around the office. " * 5
        + "I declined it.\nShe arranged another pickup before I left."
    )
    entry = WeeklyDriftReviewerEntry.model_validate(
        {
            "t_index": 0,
            "date": "2025-01-06",
            "text": text,
            "generation_metadata": "FORBIDDEN_METADATA_SENTINEL",
        }
    )
    digest = _digest([entry], [_decision(0, entry.date, conflict=conflict)])
    assert len(text.split()) > 40
    assert digest.evidence[0].excerpt == text
    assert digest.evidence[0].date == entry.date
    assert digest.evidence[0].t_index == entry.t_index
    instructions, data = render_digest_messages(digest)
    evidence_lines = json.loads(data)["evidence_lines"]
    assert json.dumps(text) in evidence_lines[0]
    judge = render_judge_prompt(
        digest,
        CoachNarrative(
            weekly_mirror='You wrote "I declined it.".',
            tension_explanation="Your sister arranged another pickup.",
            reflective_question="What stayed with you?",
        ),
    )
    assert evidence_lines[0] in judge
    assert "FORBIDDEN_METADATA_SENTINEL" not in instructions + data + judge


def test_source_matching_uses_date_and_index_and_missing_source_keeps_exact_quote():
    decision = _decision(4, "2025-01-08", conflict=True)
    decision.evidence_quote = "Exact conflict wording " * 25
    snippets = _evidence_from_decisions(
        [decision],
        entry_texts={
            ("2025-01-07", 4): "Wrong date.",
            ("2025-01-08", 3): "Wrong index.",
        },
    )
    assert snippets[0].excerpt == decision.evidence_quote.strip()
    matching = _evidence_from_decisions(
        [decision],
        entry_texts={
            ("2025-01-08", 4): "Complete matching source.",
        },
    )
    assert matching[0].excerpt == "Complete matching source."
    assert _evidence_from_decisions([_decision(4, "2025-01-08")], entry_texts={}) == []


def test_current_and_previous_week_selections_remain_capped_at_three_each():
    dates = [
        "2025-01-01",
        "2025-01-02",
        "2025-01-03",
        "2025-01-04",
        "2025-01-06",
        "2025-01-07",
        "2025-01-08",
        "2025-01-09",
    ]
    entries = [
        WeeklyDriftReviewerEntry(t_index=i, date=date, text=f"Entry {i}. " * 50)
        for i, date in enumerate(dates)
    ]
    decisions = [_decision(i, date) for i, date in enumerate(dates)]
    digest = _digest(entries, decisions)
    comparison = digest.state_comparisons[0]
    assert [item.t_index for item in comparison.previous_evidence] == [1, 2, 3]
    assert [item.t_index for item in comparison.current_evidence] == [5, 6, 7]
    assert len(digest.evidence) == 6
    assert all(item.excerpt == entries[item.t_index].text for item in digest.evidence)


def test_no_selected_decision_fallback_keeps_last_two_complete_entries():
    entries = [
        WeeklyDriftReviewerEntry(
            t_index=i, date=f"2025-01-0{6 + i}", text=f"Displayed entry {i}. " * 50
        )
        for i in range(3)
    ]
    result = detect_drift([_decision(0, entries[0].date)], persona_id="casey")
    digest = _digest(entries, [], result=result)
    assert [(item.t_index, item.excerpt) for item in digest.evidence] == [
        (entry.t_index, entry.text) for entry in entries[-2:]
    ]
    historical = _digest(entries, [], result=result, evidence_policy="historical")
    assert [item.excerpt for item in historical.evidence] == [
        " ".join(entry.text.split()[:40]) + "..." for entry in entries[-2:]
    ]


def test_historical_evidence_reproduces_quote_and_prefix_without_changing_drift():
    text = (
        "My sister needed help. " + "I considered the request. " * 20 + "I declined it."
    )
    entries = [
        WeeklyDriftReviewerEntry(t_index=i, date=f"2025-01-0{6 + i}", text=text)
        for i in range(2)
    ]
    decisions = [
        _decision(i, entry.date, conflict=(i == 0)) for i, entry in enumerate(entries)
    ]
    complete = _digest(entries, decisions)
    historical = _digest(entries, decisions, evidence_policy="historical")
    assert [item.excerpt for item in complete.evidence] == [text, text]
    assert [item.excerpt for item in historical.evidence] == [
        "I declined it.",
        " ".join(text.split()[:40]) + "...",
    ]
    assert complete.model_dump(exclude={"evidence"}) == historical.model_dump(
        exclude={"evidence"}
    )
