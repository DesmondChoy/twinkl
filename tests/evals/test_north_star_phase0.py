"""Evidence boundaries and denominator checks for NSM's local feasibility gate."""

import json

import numpy as np
import pytest

from scripts.experiments.north_star_phase0 import (
    DOCUMENT_TEMPLATE,
    ROOT,
    baseline,
    earlier_entries,
    fraction,
    load_inputs,
    rank_entries,
    read_cohort,
    retrieval_metrics,
)


def test_reproduce_specification_baseline():
    episodes, persisted, consensus, _ = load_inputs(ROOT)
    assert baseline(episodes, persisted, consensus) == {
        "episodes": 42,
        "availability": {
            "any_earlier": 34,
            "persisted_positive": 26,
            "consensus_positive": 26,
        },
        "active_at_final_cutoff": {
            "episodes": 10,
            "persisted_positive": 9,
            "consensus_positive": 9,
        },
        "onset_zero": 8,
        "onset_one": 5,
        "label_coordinates": 16510,
        "all_label_disagreements": 1368,
        "unique_earlier_coordinates": 145,
        "earlier_disagreements": 17,
        "earlier_positives": {"persisted": 68, "consensus": 64},
        "consensus_positive_agreement": {"3": 12, "4": 3, "5": 49},
    }


def test_earlier_source_requires_order_and_date_and_nonempty_writing():
    entries = [
        {"t_index": 0, "date": "2026-09-01", "initial_entry": "Earlier day"},
        {"t_index": 1, "date": "2026-09-02", "initial_entry": "Earlier same day"},
        {"t_index": 2, "date": "2026-09-03", "initial_entry": "Future date"},
        {"t_index": 3, "date": "2026-09-02", "initial_entry": "First Conflict"},
        {"t_index": 4, "date": "2026-09-01", "initial_entry": "Backdated later"},
        {"t_index": -1, "date": "2026-09-01", "initial_entry": "  "},
    ]
    assert (
        earlier_entries(
            entries,
            {
                "onset_t_index": 3,
                "onset_date": "2026-09-02",
            },
        )
        == entries[:2]
    )


def test_rank_ties_use_recency_then_stable_identifier():
    entries = [
        {"t_index": 2, "entry_id": "z"},
        {"t_index": 1, "entry_id": "older"},
        {"t_index": 2, "entry_id": "a"},
        {"t_index": 0, "entry_id": "most_similar"},
    ]
    assert rank_entries(entries, np.array([0.5, 0.5, 0.5, 0.9])) == [3, 2, 0, 1]


def test_empty_denominator_is_undefined_not_a_pass():
    assert fraction(0, 0)["rate"] is None
    assert retrieval_metrics([])["persisted_positive"]["5"] == fraction(0, 0)


def test_proxy_recall_omits_histories_without_positive_reference():
    def row(positive):
        return {
            "persisted_positive": positive,
            "consensus_5": positive,
            "consensus_4": False,
            "consensus_3": False,
            "persisted_positive_disagreement": False,
        }

    cases = [
        {"ranking": [row(False), row(True)]},
        {"ranking": [row(True)]},
        {"ranking": [row(False)]},
        {"ranking": []},
    ]
    metrics = retrieval_metrics(cases)
    assert metrics["persisted_positive"]["1"] == fraction(1, 2)
    assert metrics["persisted_positive"]["3"] == fraction(2, 2)
    assert metrics["consensus_3"]["5"] == fraction(0, 0)


@pytest.mark.parametrize("scope", [None, "pending", "small_separate"])
def test_cohort_blocks_unresolved_or_unreserved_final_scope(tmp_path, scope):
    path = tmp_path / "cohort.json"
    path.write_text(
        json.dumps(
            {
                "evaluation_scope": scope,
                "decision_source": "test-only decision",
                "frozen_at": "2026-09-05T00:00:00+00:00",
                "development_persona_ids": ["a"],
                "reserved_persona_ids": [],
            }
        )
    )
    with pytest.raises(ValueError):
        read_cohort(path, {"a"})


def test_cohort_blocks_same_persona_across_development_and_final(tmp_path):
    path = tmp_path / "cohort.json"
    manifest = {
        "evaluation_scope": "small_separate",
        "decision_source": "test-only decision",
        "frozen_at": "2026-09-05T00:00:00+00:00",
        "development_persona_ids": ["a"],
        "reserved_persona_ids": ["a"],
    }
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="overlapping"):
        read_cohort(path, {"a", "b"})
    manifest["reserved_persona_ids"] = ["b"]
    path.write_text(json.dumps(manifest))
    assert read_cohort(path, {"a", "b"})[1] == {"a"}


@pytest.mark.parametrize(
    "reserved,frozen_at",
    [
        (["does-not-exist"], "2026-09-05T00:00:00+00:00"),
        (["b"], "not-a-date"),
        (["b"], "2026-09-05"),
        (["b"], "2999-09-05T00:00:00+00:00"),
    ],
)
def test_cohort_requires_real_reservation_and_valid_freeze(
    tmp_path, reserved, frozen_at
):
    path = tmp_path / "cohort.json"
    path.write_text(
        json.dumps(
            {
                "evaluation_scope": "small_separate",
                "decision_source": "test-only",
                "frozen_at": frozen_at,
                "development_persona_ids": ["a"],
                "reserved_persona_ids": reserved,
            }
        )
    )
    with pytest.raises(ValueError):
        read_cohort(path, {"a", "b"})


def test_frozen_template_reproduces_only_original_user_writing():
    entry = {
        "initial_entry": "User text",
        "response_text": "Unavailable response",
        "nudge_text": "AI text",
        "alignment_vector": [1],
    }
    frozen = json.loads(json.dumps({"document_template": DOCUMENT_TEMPLATE}))
    assert frozen["document_template"].format(**entry) == (
        "search_document: Journal Entry:\nUser text"
    )
