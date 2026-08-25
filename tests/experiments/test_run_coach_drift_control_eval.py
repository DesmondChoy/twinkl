"""Tests for the Drift and control Coach Digest study runner."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
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
    "run_coach_drift_control_eval",
    _MODULE_PATH,
)
assert _spec is not None and _spec.loader is not None
driver = importlib.util.module_from_spec(_spec)
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
        "- **Bio:** Study runner test Persona.",
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
def sample_inputs(tmp_path: Path) -> dict[str, Path]:
    wrangled = tmp_path / "wrangled"
    wrangled.mkdir()
    _write_wrangled(
        wrangled / "persona_d1.md",
        ["2025-06-02", "2025-06-09", "2025-06-16"],
    )
    _write_wrangled(
        wrangled / "persona_d2.md",
        ["2025-07-07", "2025-07-14"],
    )
    for name in ("c1", "c2", "c3"):
        _write_wrangled(
            wrangled / f"persona_{name}.md",
            ["2025-06-02", "2025-06-09", "2025-06-16"],
        )

    episodes_path = tmp_path / "episodes.parquet"
    pl.DataFrame(
        [
            {
                "episode_id": "e1",
                "persona_id": "d1",
                "dimension": "benevolence",
                "confirmation_date": "2025-06-09",
                "delivery_state": "ended",
            },
            {
                "episode_id": "e2",
                "persona_id": "d2",
                "dimension": "security",
                "confirmation_date": "2025-07-14",
                "delivery_state": "active",
            },
        ]
    ).write_parquet(episodes_path)

    cases_path = tmp_path / "cases.parquet"
    pl.DataFrame(
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
    ).write_parquet(cases_path)
    return {
        "wrangled": wrangled,
        "episodes": episodes_path,
        "cases": cases_path,
    }


def test_drift_targets_use_the_confirmation_week(sample_inputs):
    targets = driver.load_drift_targets(
        sample_inputs["episodes"],
        sample_inputs["cases"],
        sample_inputs["wrangled"],
    )

    by_persona = {target.persona_id: target for target in targets}
    assert by_persona["d1"].target_id == "drift:e1"
    assert by_persona["d1"].end_date == "2025-06-09"
    assert by_persona["d1"].reviewed_week_count == 2
    assert by_persona["d1"].delivery_state == "ended"


def test_controls_are_deterministic_and_exclude_drift_personas(sample_inputs):
    def run() -> list[driver.EvalTarget]:
        drift_targets = driver.load_drift_targets(
            sample_inputs["episodes"],
            sample_inputs["cases"],
            sample_inputs["wrangled"],
        )
        return driver.sample_control_targets(
            sample_inputs["cases"],
            sample_inputs["episodes"],
            drift_targets,
            sample_inputs["wrangled"],
            seed=7,
        )

    first = run()
    second = run()
    assert [asdict(target) for target in first] == [
        asdict(target) for target in second
    ]
    assert {target.persona_id for target in first}.isdisjoint({"d1", "d2"})
    assert len({target.persona_id for target in first}) == len(first)
    assert all(target.group == "control" for target in first)
    assert all(target.target_id.startswith("control:") for target in first)
    drift_targets = run_drift_targets(sample_inputs)
    by_episode = {target.matched_to: target for target in first}
    for drift_target in drift_targets:
        control = by_episode[drift_target.episode_id]
        assert (
            abs(
                control.reviewed_week_count
                - drift_target.reviewed_week_count
            )
            <= 1
        )


def run_drift_targets(sample_inputs: dict[str, Path]) -> list[driver.EvalTarget]:
    return driver.load_drift_targets(
        sample_inputs["episodes"],
        sample_inputs["cases"],
        sample_inputs["wrangled"],
    )


def test_build_targets_limit_applies_to_each_group(sample_inputs):
    targets = driver.build_targets(
        episodes_path=sample_inputs["episodes"],
        case_outcomes_path=sample_inputs["cases"],
        wrangled_dir=sample_inputs["wrangled"],
        group="both",
        seed=7,
        limit=1,
    )

    assert sum(target.group == "drift" for target in targets) == 1
    assert sum(target.group == "control" for target in targets) == 1
    assert len({target.target_id for target in targets}) == len(targets)


def test_limit_must_be_positive():
    assert driver._positive_int("1") == 1
    with pytest.raises(argparse.ArgumentTypeError, match="at least 1"):
        driver._positive_int("0")


def test_target_output_directory_does_not_use_untrusted_target_text(tmp_path):
    output = driver._target_output_dir(tmp_path, "../../outside")

    assert output.parent == tmp_path
    assert output.name.startswith("target_")
    assert "outside" not in output.name


def test_resume_preserves_other_targets_and_rejects_changed_inputs(tmp_path):
    path = tmp_path / "targets.json"
    existing = {
        "schema_version": "coach-digest-drift-control-targets-v1",
        "seed": 7,
        "sources": {"source": "one"},
        "targets": [{"target_id": "drift:e1", "group": "drift"}],
    }
    path.write_text(json.dumps(existing))
    control = {
        **existing,
        "targets": [{"target_id": "control:e1:c1", "group": "control"}],
    }

    merged = driver.merge_target_catalog(path, control, resume=True)

    assert [item["target_id"] for item in merged["targets"]] == [
        "control:e1:c1",
        "drift:e1",
    ]
    changed = {**control, "seed": 8}
    with pytest.raises(ValueError, match="different schema, inputs, or seed"):
        driver.merge_target_catalog(path, changed, resume=True)


def test_resume_rejects_duplicate_manifest_target_ids(tmp_path):
    path = tmp_path / "manifest.json"
    item = {"target": {"target_id": "drift:e1"}}
    path.write_text(json.dumps([item, item]))

    with pytest.raises(ValueError, match="Duplicate manifest target ID"):
        driver._manifest_index(path, resume=True)


@pytest.mark.asyncio
async def test_resume_skips_targets_that_are_already_in_the_manifest(tmp_path):
    target = driver.EvalTarget(
        target_id="drift:e1",
        persona_id="d1",
        end_date="2025-06-09",
        group="drift",
        historical_split="training",
        entry_count=8,
        reviewed_week_count=2,
    )
    existing = {target.target_id: {"target": asdict(target)}}

    result = await driver.generate_missing_targets(
        [target],
        existing,
        parquet_path=tmp_path / "unused.parquet",
        output_dir=tmp_path / "unused-output",
        wrangled_dir=tmp_path / "unused-wrangled",
    )

    assert result is existing


@pytest.mark.asyncio
async def test_resume_rejects_a_different_generator_model(tmp_path, monkeypatch):
    target = driver.EvalTarget(
        target_id="drift:e2",
        persona_id="d2",
        end_date="2025-06-09",
        group="drift",
        historical_split="training",
        entry_count=8,
        reviewed_week_count=2,
    )
    existing = {
        "drift:e1": {
            "generator_model": "openai:model-one",
            "target": {"target_id": "drift:e1"},
        }
    }
    monkeypatch.setattr(
        driver,
        "resolve_coach_model",
        lambda: ("gemini", "model-two"),
    )

    with pytest.raises(ValueError, match="different Coach Digest generator"):
        await driver.generate_missing_targets(
            [target],
            existing,
            parquet_path=tmp_path / "unused.parquet",
            output_dir=tmp_path / "unused-output",
            wrangled_dir=tmp_path / "unused-wrangled",
        )


@pytest.mark.asyncio
async def test_completed_target_is_saved_before_a_later_failure(
    tmp_path,
    monkeypatch,
):
    targets = [
        driver.EvalTarget(
            target_id=f"drift:e{index}",
            persona_id=f"d{index}",
            end_date="2025-06-09",
            group="drift",
            historical_split="training",
            entry_count=8,
            reviewed_week_count=2,
        )
        for index in (1, 2)
    ]
    calls = 0

    async def runtime(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("later target failed")
        return object(), {}

    monkeypatch.setattr(driver, "build_llm_complete", lambda **_kwargs: object())
    monkeypatch.setattr(
        driver,
        "resolve_coach_model",
        lambda: ("openai", "model-one"),
    )
    monkeypatch.setattr(driver, "run_weekly_drift_coach_cycle", runtime)
    monkeypatch.setattr(
        driver,
        "_manifest_entry",
        lambda _digest, target, *_args: {"target": asdict(target)},
    )
    manifest_path = tmp_path / "manifest.json"

    with pytest.raises(RuntimeError, match="later target failed"):
        await driver.generate_missing_targets(
            targets,
            {},
            parquet_path=tmp_path / "unused.parquet",
            output_dir=tmp_path / "unused-output",
            wrangled_dir=tmp_path / "unused-wrangled",
            manifest_path=manifest_path,
        )

    saved = json.loads(manifest_path.read_text())
    assert [item["target"]["target_id"] for item in saved] == ["drift:e1"]
