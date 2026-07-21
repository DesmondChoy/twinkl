"""Tests for the approved Weekly Drift Reviewer runtime path."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.coach.weekly_digest import build_weekly_drift_reviewer_digest
from src.coach.weekly_drift_runtime import (
    load_onboarding_core_values,
    run_weekly_drift_coach_cycle,
)
from src.drift_detector import detect_drift
from src.weekly_drift_reviewer import (
    OpenAIWeeklyDriftReviewer,
    VerifierAssessment,
    WeeklyDriftReviewerDecision,
    WeeklyVerifierResponse,
)


def _write_wrangled(path: Path, *, core_values: str = "Benevolence") -> None:
    path.write_text(
        f"""# Persona deadbeef: Casey

## Profile
- **Persona ID:** deadbeef
- **Name:** Casey
- **Age:** 25-34
- **Profession:** Engineer
- **Culture:** Singaporean
- **Core Values:** {core_values}
- **Bio:** Runtime test persona.

---

## Entry 0 - 2025-01-06

Cancelled dinner with my family to stay at work.

---

## Entry 1 - 2025-01-13

Ignored my sister's call so I could finish another deadline.

---

## Entry 2 - 2025-01-20

Called my sister and protected the evening for family.

---
"""
    )


class _SequencedResponses:
    def __init__(self, dimension: str = "benevolence"):
        self.call_count = 0
        self.dimension = dimension

    async def parse(self, **_kwargs):
        t_index = self.call_count
        self.call_count += 1
        if t_index == 0:
            verdict = "conflict"
            quote = "Cancelled dinner with my family"
            reason = "direct_behavior_or_choice"
        elif t_index == 1:
            verdict = "conflict"
            quote = "Ignored my sister's call"
            reason = "direct_behavior_or_choice"
        else:
            verdict = "not_conflict"
            quote = ""
            reason = "direct_aligned_or_neutral_behavior"
        parsed = WeeklyVerifierResponse(
            assessments=[
                VerifierAssessment(
                    t_index=t_index,
                    dimension=self.dimension,
                    verdict=verdict,
                    confidence="high",
                    reason_code=reason,
                    evidence_quote=quote,
                )
            ]
        )
        return SimpleNamespace(
            output_parsed=parsed,
            model="gpt-5.6-luna",
            id=f"response-{t_index}",
            usage=SimpleNamespace(input_tokens=100, output_tokens=20),
            output=[],
        )


@pytest.mark.asyncio
async def test_approved_runtime_persists_reviews_drift_and_digest(tmp_path: Path):
    wrangled_dir = tmp_path / "wrangled"
    wrangled_dir.mkdir()
    _write_wrangled(wrangled_dir / "persona_deadbeef.md")
    responses = _SequencedResponses()
    reviewer = OpenAIWeeklyDriftReviewer(client=SimpleNamespace(responses=responses))

    digest, artifact_paths = await run_weekly_drift_coach_cycle(
        persona_id="deadbeef",
        wrangled_dir=wrangled_dir,
        output_dir=tmp_path / "exports",
        parquet_path=tmp_path / "weekly_digests.parquet",
        reviewer=reviewer,
    )

    assert responses.call_count == 3
    assert digest.response_mode == "recovered"
    assert digest.signal_source == "weekly_drift_reviewer"
    assert digest.overall_mean is None
    assert digest.drift_states == {"benevolence": "recovered"}
    assert len(digest.evidence) == 2
    assert Path(artifact_paths["review_receipt_1_path"]).exists()
    assert Path(artifact_paths["drift_json_path"]).exists()
    assert Path(artifact_paths["digest_json_path"]).exists()
    assert Path(artifact_paths["parquet_path"]).exists()

    payload = json.loads(Path(artifact_paths["drift_json_path"]).read_text())
    assert payload["delivery_state"] == "recovered"
    prompt = Path(artifact_paths["prompt_path"]).read_text()
    assert "Drift states: Benevolence: recovered" in prompt
    assert "Overall mean alignment: N/A" in prompt


def _write_onboarding_profile(path: Path, **overrides) -> None:
    payload = {
        "schema_version": 2,
        "onboarding_version": "2.1.0",
        "user_id": "deadbeef",
        "user_confirmed": True,
        "top_values": ["self_direction"],
        "value_profile": {"top_values": ["self_direction"]},
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload))


@pytest.mark.asyncio
async def test_onboarding_profile_supplies_runtime_core_values(tmp_path: Path):
    wrangled_dir = tmp_path / "wrangled"
    wrangled_dir.mkdir()
    _write_wrangled(
        wrangled_dir / "persona_deadbeef.md",
        core_values="Security",
    )
    profile_path = tmp_path / "profile.json"
    _write_onboarding_profile(profile_path)
    responses = _SequencedResponses(dimension="self_direction")
    reviewer = OpenAIWeeklyDriftReviewer(client=SimpleNamespace(responses=responses))

    digest, _artifact_paths = await run_weekly_drift_coach_cycle(
        persona_id="deadbeef",
        wrangled_dir=wrangled_dir,
        output_dir=tmp_path / "exports",
        parquet_path=tmp_path / "weekly_digests.parquet",
        profile_path=profile_path,
        reviewer=reviewer,
    )

    assert digest.core_values == ["self_direction"]
    assert digest.drift_states == {"self_direction": "recovered"}


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"schema_version": 1}, "schema_version"),
        ({"user_confirmed": False}, "confirmed"),
        ({"user_id": "someone-else"}, "user_id"),
        (
            {"top_values": [], "value_profile": {"top_values": []}},
            "non-empty",
        ),
        (
            {
                "top_values": ["not_a_value"],
                "value_profile": {"top_values": ["not_a_value"]},
            },
            "invalid Core Values",
        ),
        (
            {
                "top_values": ["benevolence", "self_direction"],
                "value_profile": {
                    "top_values": ["benevolence", "self_direction"]
                },
            },
            "canonical ordering",
        ),
        (
            {"value_profile": {"top_values": ["security"]}},
            "must match",
        ),
    ],
)
def test_onboarding_profile_rejects_invalid_contract(
    tmp_path: Path,
    overrides: dict,
    match: str,
):
    profile_path = tmp_path / "profile.json"
    _write_onboarding_profile(profile_path, **overrides)

    with pytest.raises(ValueError, match=match):
        load_onboarding_core_values(profile_path, persona_id="deadbeef")


def test_onboarding_profile_rejects_malformed_json(tmp_path: Path):
    profile_path = tmp_path / "profile.json"
    profile_path.write_text("not JSON")

    with pytest.raises(ValueError, match="Could not read onboarding Profile JSON"):
        load_onboarding_core_values(profile_path, persona_id="deadbeef")


def test_weekly_digest_keeps_simultaneous_core_value_evidence(tmp_path: Path):
    wrangled_dir = tmp_path / "wrangled"
    wrangled_dir.mkdir()
    _write_wrangled(wrangled_dir / "persona_deadbeef.md")
    decisions = [
        WeeklyDriftReviewerDecision(
            persona_id="deadbeef",
            week_start="2025-01-06" if t_index == 0 else "2025-01-13",
            week_end="2025-01-12" if t_index == 0 else "2025-01-19",
            t_index=t_index,
            date="2025-01-06" if t_index == 0 else "2025-01-13",
            core_value=core_value,
            verdict="conflict",
            confidence="high",
            reason_code="direct_behavior_or_choice",
            evidence_quote=(
                "Cancelled dinner with my family"
                if t_index == 0
                else "Ignored my sister's call"
            ),
            review_status="ok",
        )
        for t_index in (0, 1)
        for core_value in ("benevolence", "self_direction")
    ]
    drift_result = detect_drift(decisions, persona_id="deadbeef")

    digest = build_weekly_drift_reviewer_digest(
        persona_id="deadbeef",
        wrangled_dir=wrangled_dir,
        week_start="2025-01-06",
        week_end="2025-01-13",
        core_values=["benevolence", "self_direction"],
        decisions=decisions,
        drift_result=drift_result,
    )

    assert {
        (snippet.t_index, tuple(snippet.dimensions)) for snippet in digest.evidence
    } == {
        (0, ("benevolence",)),
        (0, ("self_direction",)),
        (1, ("benevolence",)),
        (1, ("self_direction",)),
    }
