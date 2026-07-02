"""Tests for the twinkl-w2mu small-LLM critic baseline script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "experiments" / "llm_critic_baseline.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "llm_critic_baseline_test",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_script_module()


def test_fixed_holdout_reconstructs_221_row_test_split():
    splits = mod.load_split_frames(
        labels_path=REPO_ROOT / mod.DEFAULT_LABELS_PATH,
        wrangled_dir=REPO_ROOT / mod.DEFAULT_WRANGLED_DIR,
        holdout_manifest=REPO_ROOT / mod.DEFAULT_HOLDOUT_MANIFEST,
    )

    mod.ensure_frozen_test_size(splits)

    assert splits["train"].height == 1213
    assert splits["val"].height == 217
    assert splits["test"].height == 221


def test_prompt_includes_profile_but_excludes_rich_context():
    splits = mod.load_split_frames(
        labels_path=REPO_ROOT / mod.DEFAULT_LABELS_PATH,
        wrangled_dir=REPO_ROOT / mod.DEFAULT_WRANGLED_DIR,
        holdout_manifest=REPO_ROOT / mod.DEFAULT_HOLDOUT_MANIFEST,
    )
    row = mod.rows_from_frame(splits["test"], "test")[0]

    prompt = mod.render_prompt(row, context_arm="student_visible")

    assert "Context arm: student_visible" in prompt
    assert "Normalized 10-dim value profile:" in prompt
    assert "Journal session:" in prompt
    assert "Previous journal entries" not in prompt
    assert "Bio:" not in prompt
    assert "Recent Entries" not in prompt
    for dimension in mod.SCHWARTZ_VALUE_ORDER:
        assert f"- {dimension}:" in prompt


def test_context_arms_control_history_and_bio():
    splits = mod.load_split_frames(
        labels_path=REPO_ROOT / mod.DEFAULT_LABELS_PATH,
        wrangled_dir=REPO_ROOT / mod.DEFAULT_WRANGLED_DIR,
        holdout_manifest=REPO_ROOT / mod.DEFAULT_HOLDOUT_MANIFEST,
    )
    profiles = mod.load_persona_profiles(REPO_ROOT / mod.DEFAULT_WRANGLED_DIR)
    rows = mod.rows_from_frame(
        splits["test"],
        "test",
        persona_profiles=profiles,
    )
    row = next(candidate for candidate in rows if candidate.t_index > 0)

    human_prompt = mod.render_prompt(row, context_arm="human_context")
    full_prompt = mod.render_prompt(row, context_arm="full_judge_context")

    assert "Context arm: human_context" in human_prompt
    assert "Previous journal entries before the current session:" in human_prompt
    assert f"Entry {row.t_index} (" not in human_prompt
    assert "Persona profile:" not in human_prompt
    assert "Bio:" not in human_prompt

    assert "Context arm: full_judge_context" in full_prompt
    assert "Previous journal entries before the current session:" in full_prompt
    assert "Persona profile:" in full_prompt
    assert "Bio:" in full_prompt


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-5.4-nano", 0.000325),
        ("gpt-5.4-mini", 0.0012),
    ],
)
def test_estimate_cost_uses_model_specific_pricing(model: str, expected: float):
    cost = mod.estimate_cost(model=model, input_tokens=1000, output_tokens=100)

    assert cost == pytest.approx(expected)


def test_score_records_computes_core_metrics():
    target = {dimension: 0 for dimension in mod.SCHWARTZ_VALUE_ORDER}
    target["security"] = -1
    target["hedonism"] = 1
    records = [
        {
            "status": "ok",
            "model": "gpt-5.4-nano-2026-03-17",
            "reasoning_effort": "low",
            "shots": 0,
            "target": target,
            "scores": dict(target),
            "estimated_input_tokens": 100,
            "usage": {"input_tokens": 100, "output_tokens": 40},
        },
        {
            "status": "ok",
            "model": "gpt-5.4-nano-2026-03-17",
            "reasoning_effort": "low",
            "shots": 0,
            "target": {dimension: 0 for dimension in mod.SCHWARTZ_VALUE_ORDER},
            "scores": {dimension: 0 for dimension in mod.SCHWARTZ_VALUE_ORDER},
            "estimated_input_tokens": 100,
            "usage": {"input_tokens": 100, "output_tokens": 40},
        },
    ]

    metrics = mod.score_records(records)

    assert metrics["n_ok"] == 2
    assert metrics["recall_minus1"] == pytest.approx(1.0)
    assert metrics["recall_plus1"] == pytest.approx(1.0)
    assert metrics["usage"]["input_tokens"] == 200
    assert metrics["usage"]["output_tokens"] == 80
