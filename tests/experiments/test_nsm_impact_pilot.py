import json

import pytest

from scripts.experiments import nsm_impact_pilot as pilot


def _key(pair_id: str, persona: str, first: str) -> dict[str, dict]:
    second = "without" if first == "with" else "with"
    return {
        f"{pair_id}-a": {
            "pair_id": pair_id,
            "persona_id": persona,
            "mode": "encouragement",
            "order": 0,
            "label_of": {"1": first, "2": second},
        },
        f"{pair_id}-b": {
            "pair_id": pair_id,
            "persona_id": persona,
            "mode": "encouragement",
            "order": 1,
            "label_of": {"1": second, "2": first},
        },
    }


def _verdict(preferred: str, flag_1: bool = False, flag_2: bool = False) -> dict:
    return {"preferred": preferred, "flags": {"1": flag_1, "2": flag_2}}


def test_score_pairs_requires_both_orders_to_agree():
    key = {
        **_key("win", "p1", "with"),
        **_key("flip", "p2", "with"),
        **_key("loss", "p3", "without"),
    }
    verdicts = {
        # "with" is Response 1 then Response 2: the same arm wins in both orders.
        "win-a": _verdict("1"),
        "win-b": _verdict("2"),
        # Always picking Response 1 is position bias, so it becomes a tie.
        "flip-a": _verdict("1"),
        "flip-b": _verdict("1"),
        "loss-a": _verdict("1", flag_2=True),
        "loss-b": _verdict("2", flag_1=True),
    }
    judgments, pairs = pilot.score_pairs(key, verdicts)
    outcomes = {p["pair_id"]: (p["outcome"], p["consistent"]) for p in pairs}
    assert outcomes == {
        "win": ("win", True),
        "flip": ("tie", False),
        "loss": ("loss", True),
    }
    assert pilot.win_difference(pairs) == 0.0
    # In the loss pair, both flags land on the "with" response.
    assert sum(j["flag_with"] for j in judgments) == 2
    assert sum(j["flag_without"] for j in judgments) == 0


def test_win_difference_and_bootstrap_bounds():
    pairs = [
        {"pair_id": str(i), "persona_id": f"p{i % 3}", "outcome": outcome}
        for i, outcome in enumerate(["win", "win", "win", "loss", "tie", "win"])
    ]
    assert pilot.win_difference(pairs) == pytest.approx(100 * 3 / 6)
    low, high = pilot.persona_bootstrap(pairs, resamples=500)
    assert -100 <= low <= pilot.win_difference(pairs) <= high <= 100
    assert pilot.persona_bootstrap([pairs[0]]) is None


def test_load_verdict_rejects_malformed_output(tmp_path):
    path = tmp_path / "task-01.json"
    path.write_text(
        json.dumps(
            {"task_id": "task-01", "preferred": "A", "flags": {"1": False, "2": False}}
        )
    )
    with pytest.raises(ValueError):
        pilot.load_verdict(path, "task-01")


def test_prepared_tasks_are_blind(tmp_path):
    manifest = pilot.prepare(tmp_path / "run")
    assert (manifest["pairs"], manifest["tasks"]) == (22, 44)
    assert manifest["judge"] == {
        "name": "nsm-impact-judge",
        "model": "claude-opus-5-5",
        "effort": "medium",
    }
    key = json.loads((tmp_path / "run/sealed/answer_key.json").read_text())
    names = {p["shared_input"]["persona_name"].lower() for p in pilot.load_pairs()}
    hidden = [
        "north star",
        "north_star",
        "internal schwartz",
        "run length",
        "not_conflict",
    ]
    for task_id in key:
        text = (tmp_path / f"run/tasks/{task_id}.md").read_text().lower()
        assert not [term for term in hidden if term in text], task_id
        assert not [name for name in names if name in text], task_id
    # Each pair appears once with each arm as Response 1.
    leads = {}
    for entry in key.values():
        leads.setdefault(entry["pair_id"], set()).add(entry["label_of"]["1"])
    assert all(value == {"with", "without"} for value in leads.values())
