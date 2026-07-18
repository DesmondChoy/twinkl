"""Tests for the weekly VIF-to-Coach runtime bridge."""

import asyncio
import json
import logging
from pathlib import Path

import polars as pl
import pytest

from src.coach.runtime import run_weekly_coach_cycle
from src.coach.schemas import DriftDetectionResult
from src.coach.weekly_digest import build_weekly_digest
from src.models.judge import SCHWARTZ_VALUE_ORDER


def _write_runtime_wrangled(path: Path) -> None:
    path.write_text(
        """# Persona deadbeef: Casey

## Profile
- **Persona ID:** deadbeef
- **Name:** Casey
- **Age:** 25-34
- **Profession:** Engineer
- **Culture:** Singaporean
- **Core Values:** Achievement, Benevolence
- **Bio:** Runtime bridge test persona.

---

## Entry 0 - 2025-01-06

Stayed late again and cancelled family dinner.

---

## Entry 1 - 2025-01-08

Tried to repair things by calling home after work.

---

## Entry 2 - 2025-01-14

Protected the evening for family and left the laptop shut.

---

## Entry 3 - 2025-01-16

Took on another deadline and could feel myself slipping back into old habits.

---
"""
    )


def _signal_row(
    *,
    date: str,
    t_index: int,
    achievement: float,
    benevolence: float,
    overall_mean: float,
    overall_uncertainty: float = 0.1,
) -> dict:
    row = {
        "persona_id": "deadbeef",
        "persona_name": "Casey",
        "date": date,
        "t_index": t_index,
        "initial_entry": "fixture entry",
        "nudge_text": None,
        "response_text": None,
        "has_nudge": False,
        "has_response": False,
        "core_values": ["Achievement", "Benevolence"],
        "overall_mean": overall_mean,
        "overall_uncertainty": overall_uncertainty,
    }
    for dim in SCHWARTZ_VALUE_ORDER:
        row[f"alignment_{dim}"] = 0.0
        row[f"uncertainty_{dim}"] = 0.1
        row[f"profile_weight_{dim}"] = 0.0
    row["alignment_achievement"] = achievement
    row["alignment_benevolence"] = benevolence
    row["profile_weight_achievement"] = 0.4
    row["profile_weight_benevolence"] = 0.3
    row["alignment_vector"] = [row[f"alignment_{dim}"] for dim in SCHWARTZ_VALUE_ORDER]
    row["uncertainty_vector"] = [row[f"uncertainty_{dim}"] for dim in SCHWARTZ_VALUE_ORDER]
    return row


def test_build_weekly_digest_from_vif_signals_prefers_upstream_drift_result(tmp_path: Path):
    wrangled_dir = tmp_path / "wrangled"
    wrangled_dir.mkdir(parents=True, exist_ok=True)
    _write_runtime_wrangled(wrangled_dir / "persona_deadbeef.md")

    signals_df = pl.DataFrame(
        [
            _signal_row(date="2025-01-14", t_index=2, achievement=-0.7, benevolence=0.7, overall_mean=0.1),
            _signal_row(date="2025-01-16", t_index=3, achievement=-0.8, benevolence=0.2, overall_mean=-0.2),
        ]
    )
    drift_result = DriftDetectionResult(
        response_mode="evolution",
        rationale="Achievement is diverging with low volatility while benevolence is strengthening.",
        reasons=["low_volatility_directional_shift", "achievement"],
        source="drift_detector",
        trigger_type="evolution",
        week_start="2025-01-13",
        week_end="2025-01-19",
        overall_mean=-0.05,
        overall_uncertainty=0.1,
        triggered_dimensions=["achievement", "benevolence"],
        dimension_signals=[
            DriftDetectionResult.DimensionSignal(
                dimension="achievement",
                classification="evolution",
                mean_alignment=-0.75,
                mean_uncertainty=0.1,
                trigger="evolution",
                residual=-0.9,
                volatility=0.05,
            ),
            DriftDetectionResult.DimensionSignal(
                dimension="benevolence",
                classification="evolution",
                mean_alignment=0.45,
                mean_uncertainty=0.1,
                trigger="evolution",
                residual=0.45,
                volatility=0.05,
            ),
        ],
    )

    digest = build_weekly_digest(
        persona_id="deadbeef",
        labels_path=None,
        wrangled_dir=wrangled_dir,
        signals_df=signals_df,
        start_date="2025-01-13",
        end_date="2025-01-19",
        drift_result=drift_result,
    )

    assert digest.signal_source == "vif_runtime"
    assert digest.response_mode == "evolution"
    assert digest.mode_source == "drift_detector"
    assert digest.overall_uncertainty == 0.1
    assert "achievement" in digest.top_tensions
    assert "benevolence" in digest.top_strengths


def test_run_weekly_coach_cycle_persists_bridge_artifacts(tmp_path: Path, monkeypatch):
    wrangled_dir = tmp_path / "wrangled"
    wrangled_dir.mkdir(parents=True, exist_ok=True)
    _write_runtime_wrangled(wrangled_dir / "persona_deadbeef.md")

    timeline_df = pl.DataFrame(
        [
            _signal_row(date="2025-01-06", t_index=0, achievement=0.6, benevolence=-0.6, overall_mean=0.25),
            _signal_row(date="2025-01-08", t_index=1, achievement=0.3, benevolence=0.2, overall_mean=0.2),
            _signal_row(date="2025-01-14", t_index=2, achievement=-0.8, benevolence=0.6, overall_mean=-0.2),
            _signal_row(date="2025-01-16", t_index=3, achievement=-0.9, benevolence=0.1, overall_mean=-0.35),
        ]
    )

    monkeypatch.setattr(
        "src.coach.runtime.predict_persona_timeline",
        lambda **_kwargs: (
            timeline_df,
            {"persona_id": "deadbeef", "checkpoint_path": "unused"},
        ),
    )

    digest, artifact_paths = run_weekly_coach_cycle(
        persona_id="deadbeef",
        checkpoint_path="unused.pt",
        wrangled_dir=wrangled_dir,
        output_dir=tmp_path / "exports",
        parquet_path=tmp_path / "weekly_digests.parquet",
        end_date="2025-01-19",
    )

    assert digest.signal_source == "vif_runtime"
    assert digest.mode_source == "drift_detector"
    assert Path(artifact_paths["timeline_path"]).exists()
    assert Path(artifact_paths["weekly_path"]).exists()
    assert Path(artifact_paths["drift_json_path"]).exists()
    assert Path(artifact_paths["digest_json_path"]).exists()
    assert Path(artifact_paths["digest_md_path"]).exists()

    drift_payload = json.loads(Path(artifact_paths["drift_json_path"]).read_text())
    assert drift_payload["response_mode"] in {"crash", "rut", "evolution", "stable"}
    assert digest.coach_narrative is None


def _stub_timeline_df() -> pl.DataFrame:
    return pl.DataFrame(
        [
            _signal_row(date="2025-01-06", t_index=0, achievement=0.6, benevolence=-0.6, overall_mean=0.25),
            _signal_row(date="2025-01-08", t_index=1, achievement=0.3, benevolence=0.2, overall_mean=0.2),
            _signal_row(date="2025-01-14", t_index=2, achievement=-0.8, benevolence=0.6, overall_mean=-0.2),
            _signal_row(date="2025-01-16", t_index=3, achievement=-0.9, benevolence=0.1, overall_mean=-0.35),
        ]
    )


def test_run_weekly_coach_cycle_attaches_narrative_with_llm(tmp_path: Path, monkeypatch):
    wrangled_dir = tmp_path / "wrangled"
    wrangled_dir.mkdir(parents=True, exist_ok=True)
    _write_runtime_wrangled(wrangled_dir / "persona_deadbeef.md")

    monkeypatch.setattr(
        "src.coach.runtime.predict_persona_timeline",
        lambda **_kwargs: (
            _stub_timeline_df(),
            {"persona_id": "deadbeef", "checkpoint_path": "unused"},
        ),
    )

    async def stub_llm(prompt: str, response_format: dict | None) -> str:
        assert response_format is not None
        return json.dumps(
            {
                "weekly_mirror": "A short reflective mirror sentence for the week that stays grounded.",
                "tension_explanation": "The week pulled between staying late for work and protecting family time.",
                "reflective_question": "What made it easier to shut the laptop on the evenings you managed to?",
            }
        )

    digest, _paths = run_weekly_coach_cycle(
        persona_id="deadbeef",
        checkpoint_path="unused.pt",
        wrangled_dir=wrangled_dir,
        output_dir=tmp_path / "exports",
        parquet_path=tmp_path / "weekly_digests.parquet",
        end_date="2025-01-19",
        llm_complete=stub_llm,
    )

    assert digest.coach_narrative is not None
    assert digest.coach_narrative.weekly_mirror.startswith("A short reflective")
    assert digest.validation is not None


def test_build_llm_complete_returns_none_without_keys(monkeypatch):
    from src.coach.llm_client import build_llm_complete

    for var in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    assert build_llm_complete(provider="gemini") is None
    assert build_llm_complete(provider="openai") is None


def test_build_llm_complete_selects_provider_by_env(monkeypatch):
    from src.coach.llm_client import build_llm_complete

    for var in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    # Only a Gemini key is present: gemini builds, openai does not.
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("TWINKL_COACH_PROVIDER", "gemini")
    assert build_llm_complete() is not None
    assert build_llm_complete(provider="openai") is None

    # Provider defaults to openai: with only a Gemini key present, the default
    # build yields None. Unknown providers also yield None.
    monkeypatch.delenv("TWINKL_COACH_PROVIDER", raising=False)
    assert build_llm_complete() is None
    assert build_llm_complete(provider="not-a-provider") is None

    # With an OpenAI key present, the default provider builds.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert build_llm_complete() is not None


@pytest.mark.asyncio
async def test_openai_failure_logs_warning(monkeypatch, caplog):
    from src.coach.llm_client import build_llm_complete

    class FailingResponses:
        async def create(self, **_kwargs):
            raise RuntimeError("provider unavailable")

    class FailingClient:
        def __init__(self):
            self.responses = FailingResponses()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("openai.AsyncOpenAI", FailingClient)
    llm_complete = build_llm_complete(provider="openai", model="test-model")
    assert llm_complete is not None

    with caplog.at_level(logging.WARNING, logger="src.coach.llm_client"):
        result = await llm_complete("prompt", None)

    assert result is None
    assert "Weekly Coach OpenAI request failed for model test-model" in caplog.text


@pytest.mark.asyncio
async def test_gemini_failure_logs_warning(monkeypatch, caplog):
    from src.coach.llm_client import build_llm_complete

    async def fail_to_thread(*_args, **_kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(asyncio, "to_thread", fail_to_thread)
    llm_complete = build_llm_complete(provider="gemini", model="test-model")
    assert llm_complete is not None

    with caplog.at_level(logging.WARNING, logger="src.coach.llm_client"):
        result = await llm_complete("prompt", None)

    assert result is None
    assert "Weekly Coach Gemini request failed for model test-model" in caplog.text


def test_unwrap_json_schema_extracts_inner_schema():
    from src.coach.llm_client import _unwrap_json_schema
    from src.coach.schemas import WEEKLY_DIGEST_COACH_RESPONSE_FORMAT

    schema = _unwrap_json_schema(WEEKLY_DIGEST_COACH_RESPONSE_FORMAT)
    assert schema is not None
    assert schema["type"] == "object"
    assert "weekly_mirror" in schema["properties"]
    assert _unwrap_json_schema(None) is None
    assert _unwrap_json_schema({"type": "json_schema"}) is None
