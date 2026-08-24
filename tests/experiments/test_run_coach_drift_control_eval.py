"""Tests for the Drift against control Coach Digest sample builder.

The evaluation is only reproducible if target selection is a pure function of
the Parquet inputs, the wrangled directory, and the seed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import polars as pl
import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "experiments"
    / "run_coach_drift_control_eval.py"
)
_spec = importlib.util.spec_from_file_location(
    "run_coach_drift_control_eval", _MODULE_PATH
)
assert _spec is not None and _spec.loader is not None
driver = importlib.util.module_from_spec(_spec)
# The dataclass decorator resolves annotations through sys.modules, so the
# module must be registered before it executes.
sys.modules[_spec.name] = driver
_spec.loader.exec_module(driver)


def _write_wrangled(path: Path, dates: list[str]) -> None:
    persona_id = path.stem.replace("persona_", "")
    lines = [
        f"# Persona {persona_id}: Casey",
        "",
        "## Profile",
        f"- **Persona ID:** {persona_id}",
        "- **Name:** Casey",
        "- **Age:** 25-34",
        "- **Profession:** Engineer",
        "- **Culture:** Singaporean",
        "- **Core Values:** Benevolence",
        "- **Bio:** Sample builder test persona.",
        "",
        "---",
        "",
    ]
    for index, day in enumerate(dates):
        lines += [
            f"## Entry {index} - {day}",
            "",
            "A day of work and family.",
            "",
            "---",
            "",
        ]
    path.write_text("\n".join(lines))


@pytest.fixture
def sample_repo(tmp_path: Path) -> dict[str, Path]:
    wrangled = tmp_path / "wrangled"
    wrangled.mkdir()
    # Two drift personas and three clean control personas.
    _write_wrangled(
        wrangled / "persona_d1.md",
        ["2025-06-02", "2025-06-09", "2025-06-16"],
    )
    _write_wrangled(wrangled / "persona_d2.md", ["2025-07-07", "2025-07-14"])
    for name in ("c1", "c2", "c3"):
        _write_wrangled(
            wrangled / f"persona_{name}.md",
            ["2025-06-02", "2025-06-09", "2025-06-16"],
        )

    episodes = pl.DataFrame(
        [
            {
                "episode_id": "e1",
                "persona_id": "d1",
                "dimension": "benevolence",
                "confirmation_date": "2025-06-09",
                "delivery_state": "recovered",
            },
            {
                "episode_id": "e2",
                "persona_id": "d2",
                "dimension": "security",
                "confirmation_date": "2025-07-14",
                "delivery_state": "active",
            },
        ]
    )
    episodes_path = tmp_path / "episodes.parquet"
    episodes.write_parquet(episodes_path)

    cases = pl.DataFrame(
        [
            {
                "persona_id": "d1",
                "dimension": "benevolence",
                "historical_split": "training",
                "entry_count": 8,
                "has_drift": True,
            },
            {
                "persona_id": "d2",
                "dimension": "security",
                "historical_split": "training",
                "entry_count": 8,
                "has_drift": True,
            },
            {
                "persona_id": "c1",
                "dimension": "benevolence",
                "historical_split": "training",
                "entry_count": 8,
                "has_drift": False,
            },
            {
                "persona_id": "c2",
                "dimension": "security",
                "historical_split": "training",
                "entry_count": 8,
                "has_drift": False,
            },
            {
                "persona_id": "c3",
                "dimension": "tradition",
                "historical_split": "retired",
                "entry_count": 3,
                "has_drift": False,
            },
        ]
    )
    cases_path = tmp_path / "cases.parquet"
    cases.write_parquet(cases_path)

    return {
        "wrangled": wrangled,
        "episodes": episodes_path,
        "cases": cases_path,
    }


def test_entry_count_buckets_split_history_length():
    assert driver._entry_count_bucket(3) == "<=6"
    assert driver._entry_count_bucket(8) == "7-9"
    assert driver._entry_count_bucket(11) == "10-12"


def test_drift_targets_report_on_the_confirmation_week(sample_repo):
    targets = driver.load_drift_targets(
        sample_repo["episodes"], sample_repo["cases"], sample_repo["wrangled"]
    )

    assert len(targets) == 2
    by_persona = {t.persona_id: t for t in targets}
    assert by_persona["d1"].end_date == "2025-06-09"
    assert by_persona["d1"].arm == "drift"
    assert by_persona["d1"].delivery_state == "recovered"
    # d1 has weeks ending 2025-06-08, 06-15, 06-22; truncating at 06-09 leaves
    # the first two, so the confirmation week is the reported week.
    assert by_persona["d1"].n_truncated_weeks == 2
    assert by_persona["d2"].episode_id == "e2"


def test_controls_exclude_every_persona_that_drifts(sample_repo):
    drift_targets = driver.load_drift_targets(
        sample_repo["episodes"], sample_repo["cases"], sample_repo["wrangled"]
    )
    controls = driver.sample_control_targets(
        sample_repo["cases"],
        sample_repo["episodes"],
        drift_targets,
        sample_repo["wrangled"],
        seed=7,
    )

    assert controls
    assert {t.persona_id for t in controls}.isdisjoint({"d1", "d2"})
    assert all(t.arm == "control" for t in controls)
    # One control Persona is never reused across the arm.
    assert len({t.persona_id for t in controls}) == len(controls)


def test_control_sampling_is_deterministic_under_a_seed(sample_repo):
    def run(seed: int) -> list[tuple[str, str]]:
        drift_targets = driver.load_drift_targets(
            sample_repo["episodes"], sample_repo["cases"], sample_repo["wrangled"]
        )
        controls = driver.sample_control_targets(
            sample_repo["cases"],
            sample_repo["episodes"],
            drift_targets,
            sample_repo["wrangled"],
            seed=seed,
        )
        return [(t.persona_id, t.end_date) for t in controls]

    assert run(7) == run(7)


def test_controls_match_the_reviewed_week_count(sample_repo):
    drift_targets = driver.load_drift_targets(
        sample_repo["episodes"], sample_repo["cases"], sample_repo["wrangled"]
    )
    controls = driver.sample_control_targets(
        sample_repo["cases"],
        sample_repo["episodes"],
        drift_targets,
        sample_repo["wrangled"],
        seed=7,
    )
    by_episode = {t.matched_to: t for t in controls}
    for drift_target in drift_targets:
        control = by_episode.get(drift_target.episode_id)
        if control is None:
            continue
        # Evidence volume scales with reviewed weeks, so the arms must not
        # differ on it by more than the pool allows.
        assert abs(control.n_truncated_weeks - drift_target.n_truncated_weeks) <= 1


def test_build_targets_limit_applies_to_each_arm(sample_repo):
    targets = driver.build_targets(
        episodes_path=sample_repo["episodes"],
        case_outcomes_path=sample_repo["cases"],
        wrangled_dir=sample_repo["wrangled"],
        arm="both",
        seed=7,
        limit=1,
    )

    assert sum(1 for t in targets if t.arm == "drift") == 1
    assert sum(1 for t in targets if t.arm == "control") == 1


def test_build_targets_can_select_one_arm(sample_repo):
    drift_only = driver.build_targets(
        episodes_path=sample_repo["episodes"],
        case_outcomes_path=sample_repo["cases"],
        wrangled_dir=sample_repo["wrangled"],
        arm="drift",
        seed=7,
        limit=None,
    )
    assert {t.arm for t in drift_only} == {"drift"}

    control_only = driver.build_targets(
        episodes_path=sample_repo["episodes"],
        case_outcomes_path=sample_repo["cases"],
        wrangled_dir=sample_repo["wrangled"],
        arm="control",
        seed=7,
        limit=None,
    )
    assert {t.arm for t in control_only} == {"control"}
