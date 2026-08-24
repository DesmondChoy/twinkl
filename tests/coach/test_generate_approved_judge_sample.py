"""Tests for the approved Coach Digest evaluation-sample generator."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import polars as pl
import pytest

from scripts.coach.generate_approved_judge_sample import (
    _build_parser,
    _extract_scenario_key_week_digests,
    _find_stored_digest_path,
    _generate_reusing_weekly_drift,
)
from src.coach.schemas import EvidenceSnippet, LLMCallMetrics, WeeklyDigest


def _digest(persona_id: str) -> WeeklyDigest:
    return WeeklyDigest(
        persona_id=persona_id,
        persona_name="Casey",
        week_start="2025-01-01",
        week_end="2025-01-07",
        response_mode="no_active_drift",
        mode_source="drift_detector",
        mode_rationale="No active Drift is confirmed.",
        signal_source="weekly_drift_reviewer",
        n_entries=1,
        overall_mean=None,
        core_values=["benevolence"],
        drift_states={"benevolence": "no_active_drift"},
        top_tensions=[],
        top_strengths=[],
        dimensions=[],
        evidence=[
            EvidenceSnippet(
                date="2025-01-03",
                t_index=1,
                direction="context",
                dimensions=["benevolence"],
                excerpt="called my mom and helped a colleague debug",
            )
        ],
    )


def _valid_response() -> str:
    return json.dumps(
        {
            "weekly_mirror": (
                'You wrote that you "called my mom and helped a colleague debug" '
                "which gives this week a concrete point of reflection."
            ),
            "tension_explanation": (
                "The available Journal Entry shows one specific choice. It does "
                "not establish a wider positive or negative pattern by itself."
            ),
            "reflective_question": (
                "What felt most important to you about that choice in the moment?"
            ),
        }
    )


def test_sample_generator_requires_explicit_personas():
    with pytest.raises(SystemExit):
        _build_parser().parse_args([])


def test_extracts_only_deployed_key_week_digests(tmp_path: Path):
    personas = ["11de77e8", "23d101f8", "8f83c818", "988d1a65", "02fb94f3"]

    sources = _extract_scenario_key_week_digests(personas, tmp_path)

    assert {
        persona_id: (source["scenario_id"], source["week_start"])
        for persona_id, source in sources.items()
    } == {
        "11de77e8": ("two-values-lukas", "2025-10-13"),
        "23d101f8": ("stable-meera", "2025-09-15"),
        "8f83c818": ("active-wei-jun", "2025-06-30"),
        "988d1a65": ("recovered-marc", "2025-03-17"),
        "02fb94f3": ("uncertain-noor", "2025-04-14"),
    }
    for persona_id in personas:
        paths = list(tmp_path.glob(f"{persona_id}_*.json"))
        assert len(paths) == 1
        digest = WeeklyDigest.model_validate_json(paths[0].read_text())
        assert digest.coach_narrative is None
        assert digest.validation is None


def test_find_stored_digest_requires_exactly_one_output(tmp_path: Path):
    path = tmp_path / "aaaa_2025-01-07.json"
    path.write_text("{}")
    (tmp_path / "aaaa_2025-01-07.drift.json").write_text("{}")

    assert _find_stored_digest_path("aaaa", tmp_path) == path


def test_coach_only_generation_reuses_outputs_and_builds_manifest(tmp_path: Path):
    output_dir = tmp_path / "weekly_drift"
    output_dir.mkdir()
    for persona_id in ("aaaa", "bbbb"):
        path = output_dir / f"{persona_id}_2025-01-07.json"
        path.write_text(_digest(persona_id).model_dump_json(indent=2))

    calls = 0
    call_metrics: list[LLMCallMetrics] = []

    async def llm_complete(
        _prompt: str,
        _response_format: dict | None,
        _instructions: str | None = None,
    ) -> str:
        nonlocal calls
        calls += 1
        call_metrics.append(
            LLMCallMetrics(
                provider="openai",
                model="gpt-5.6-luna",
                reasoning_effort="none",
                service_tier="default",
                latency_seconds=1.0,
                input_tokens=100,
                cached_input_tokens=0,
                cache_write_input_tokens=0,
                output_tokens=50,
                total_tokens=150,
                calculated_cost_usd=0.00008,
            )
        )
        return _valid_response()

    parquet_path = tmp_path / "weekly_digests.parquet"
    manifest = asyncio.run(
        _generate_reusing_weekly_drift(
            ["aaaa", "bbbb"],
            parquet_path,
            output_dir,
            llm_complete=llm_complete,
            call_metrics=call_metrics,
        )
    )

    assert calls == 2
    assert len(manifest) == 2
    assert all(
        item["provenance"]["coach_prompt_version"] == "4.1"
        for item in manifest
    )
    assert all(item["digest"]["state_comparisons"] == [] for item in manifest)
    assert all(
        len(item["provenance"]["coach_call_metrics"]) == 1
        for item in manifest
    )
    assert len(list(output_dir.glob("*.coach_diagnostic.json"))) == 2
    assert pl.read_parquet(parquet_path).height == 2
    for item, persona_id in zip(manifest, ("aaaa", "bbbb"), strict=True):
        stored_path = output_dir / f"{persona_id}_2025-01-07.json"
        stored = WeeklyDigest.model_validate_json(stored_path.read_text())
        assert stored.coach_narrative is not None
        assert stored.validation is not None
        diagnostic_path = Path(
            item["provenance"]["coach_diagnostic_paths"][0]
        )
        assert json.loads(diagnostic_path.read_text())["llm_call"] is not None
        assert item["provenance"]["weekly_drift_output_sha256"] == hashlib.sha256(
            stored_path.read_bytes()
        ).hexdigest()
        input_bytes = json.dumps(
            item["digest"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
        assert item["provenance"]["weekly_drift_input_sha256"] == hashlib.sha256(
            input_bytes
        ).hexdigest()


def test_coach_only_generation_fails_before_rewriting_outputs(tmp_path: Path):
    output_dir = tmp_path / "weekly_drift"
    output_dir.mkdir()
    digest_path = output_dir / "aaaa_2025-01-07.json"
    original = _digest("aaaa").model_dump_json(indent=2)
    digest_path.write_text(original)

    async def llm_complete(
        _prompt: str,
        _response_format: dict | None,
        _instructions: str | None = None,
    ) -> str:
        return "not-json"

    try:
        asyncio.run(
            _generate_reusing_weekly_drift(
                ["aaaa"],
                tmp_path / "weekly_digests.parquet",
                output_dir,
                llm_complete=llm_complete,
            )
        )
    except RuntimeError as exc:
        assert "json_parse" in str(exc)
    else:
        raise AssertionError("Expected Coach Digest-only generation to fail.")

    assert digest_path.read_text() == original
    assert not (tmp_path / "weekly_digests.parquet").exists()
    assert len(list(output_dir.glob("*.coach_diagnostic.json"))) == 1


def test_coach_only_generation_retries_one_validation_failure(tmp_path: Path):
    output_dir = tmp_path / "weekly_drift"
    output_dir.mkdir()
    digest_path = output_dir / "aaaa_2025-01-07.json"
    digest_path.write_text(_digest("aaaa").model_dump_json(indent=2))
    calls = 0

    async def llm_complete(
        _prompt: str,
        _response_format: dict | None,
        instructions: str | None = None,
    ) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            payload = json.loads(_valid_response())
            payload["tension_explanation"] += " This names Benevolence."
            return json.dumps(payload)
        assert instructions is not None
        assert "value_leakage" in instructions
        return _valid_response()

    manifest = asyncio.run(
        _generate_reusing_weekly_drift(
            ["aaaa"],
            tmp_path / "weekly_digests.parquet",
            output_dir,
            llm_complete=llm_complete,
        )
    )

    assert calls == 2
    assert len(manifest) == 1
    assert manifest[0]["provenance"]["coach_attempt_count"] == 2
    assert len(list(output_dir.glob("*.coach_diagnostic.json"))) == 2
