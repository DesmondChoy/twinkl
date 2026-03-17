"""Tests for weekly digest Coach vertical slice."""

import asyncio
import json
from pathlib import Path

import polars as pl

from src.coach.weekly_digest import (
    attach_coach_artifacts,
    build_weekly_digest,
    generate_weekly_digest_coach,
    persist_weekly_digest_record,
    render_digest_markdown,
    render_digest_prompt,
    validate_weekly_digest_narrative,
)


def _write_test_wrangled(path: Path) -> None:
    path.write_text(
        """# Persona deadbeef: Casey

## Profile
- **Persona ID:** deadbeef
- **Name:** Casey
- **Age:** 25-34
- **Profession:** Engineer
- **Culture:** Singaporean
- **Core Values:** Self-Direction, Benevolence
- **Bio:** Test persona.

---

## Entry 0 - 2025-01-01

Skipped gym and doomscrolled after work.

---

## Entry 1 - 2025-01-03

Called my mom and helped a colleague debug.

---

## Entry 2 - 2025-01-05

Spent the morning planning my side project and writing.

---
"""
    )


def test_build_weekly_digest_and_render(tmp_path: Path):
    labels_path = tmp_path / "judge_labels.parquet"
    wrangled_dir = tmp_path / "wrangled"
    wrangled_dir.mkdir(parents=True, exist_ok=True)
    _write_test_wrangled(wrangled_dir / "persona_deadbeef.md")

    rows = [
        {
            "persona_id": "deadbeef",
            "t_index": 0,
            "date": "2025-01-01",
            "alignment_self_direction": -1,
            "alignment_stimulation": 0,
            "alignment_hedonism": -1,
            "alignment_achievement": 0,
            "alignment_power": 0,
            "alignment_security": 0,
            "alignment_conformity": 0,
            "alignment_tradition": 0,
            "alignment_benevolence": -1,
            "alignment_universalism": 0,
        },
        {
            "persona_id": "deadbeef",
            "t_index": 1,
            "date": "2025-01-03",
            "alignment_self_direction": 0,
            "alignment_stimulation": 0,
            "alignment_hedonism": 0,
            "alignment_achievement": 0,
            "alignment_power": 0,
            "alignment_security": 0,
            "alignment_conformity": 0,
            "alignment_tradition": 0,
            "alignment_benevolence": 1,
            "alignment_universalism": 0,
        },
        {
            "persona_id": "deadbeef",
            "t_index": 2,
            "date": "2025-01-05",
            "alignment_self_direction": 1,
            "alignment_stimulation": 0,
            "alignment_hedonism": 0,
            "alignment_achievement": 1,
            "alignment_power": 0,
            "alignment_security": 0,
            "alignment_conformity": 0,
            "alignment_tradition": 0,
            "alignment_benevolence": 0,
            "alignment_universalism": 0,
        },
    ]
    pl.DataFrame(rows).write_parquet(labels_path)

    digest = build_weekly_digest(
        persona_id="deadbeef",
        labels_path=labels_path,
        wrangled_dir=wrangled_dir,
        start_date="2025-01-01",
        end_date="2025-01-07",
    )

    assert digest.persona_name == "Casey"
    assert digest.response_mode in {"stable", "rut"}
    assert digest.mode_source == "fallback_heuristic"
    assert digest.n_entries == 3
    assert digest.core_values == ["self_direction", "benevolence"]
    assert len(digest.top_tensions) == 3
    assert len(digest.top_strengths) == 2
    assert len(digest.evidence) == 3
    assert len(digest.journal_history) == 3
    assert all(snippet.dimensions for snippet in digest.evidence)

    evidence_keys = [(snippet.date, snippet.t_index) for snippet in digest.evidence]
    assert len(evidence_keys) == len(set(evidence_keys))

    md = render_digest_markdown(digest)
    prompt = render_digest_prompt(digest)

    assert "Weekly Alignment Digest: Casey" in md
    assert "Response mode" in md
    assert "Full Journal History" in md
    assert "Evidence Snippets" in md
    assert "dims=" in md
    assert "Persona: Casey (deadbeef)" in prompt
    assert "Declared core values: Self Direction, Benevolence" in prompt
    assert "Full journal history:" in prompt
    assert "Primary tensions:" in prompt


def test_generate_validate_and_persist_weekly_digest(tmp_path: Path):
    labels_path = tmp_path / "judge_labels.parquet"
    wrangled_dir = tmp_path / "wrangled"
    wrangled_dir.mkdir(parents=True, exist_ok=True)
    _write_test_wrangled(wrangled_dir / "persona_deadbeef.md")

    rows = [
        {
            "persona_id": "deadbeef",
            "t_index": 0,
            "date": "2025-01-01",
            "alignment_self_direction": -1,
            "alignment_stimulation": 0,
            "alignment_hedonism": 0,
            "alignment_achievement": 0,
            "alignment_power": 0,
            "alignment_security": 0,
            "alignment_conformity": 0,
            "alignment_tradition": 0,
            "alignment_benevolence": 0,
            "alignment_universalism": 0,
        },
        {
            "persona_id": "deadbeef",
            "t_index": 1,
            "date": "2025-01-03",
            "alignment_self_direction": 0,
            "alignment_stimulation": 0,
            "alignment_hedonism": 0,
            "alignment_achievement": 0,
            "alignment_power": 0,
            "alignment_security": 0,
            "alignment_conformity": 0,
            "alignment_tradition": 0,
            "alignment_benevolence": 1,
            "alignment_universalism": 0,
        },
    ]
    pl.DataFrame(rows).write_parquet(labels_path)

    digest = build_weekly_digest(
        persona_id="deadbeef",
        labels_path=labels_path,
        wrangled_dir=wrangled_dir,
        start_date="2025-01-01",
        end_date="2025-01-07",
        response_mode="stable",
    )

    async def fake_llm_complete(prompt: str, response_format: dict | None) -> str:
        assert "Return JSON with exactly these keys" in prompt
        assert response_format is not None
        return json.dumps(
            {
                "weekly_mirror": 'This week felt split between "Skipped gym and doomscrolled after work" and moments of care for other people.',
                "tension_explanation": 'The pull seems to be between depleted self-direction and the parts of the week where you still showed up, especially when you wrote "Called my mom and helped a colleague debug."',
                "reflective_question": "What felt different between the moments when you drifted and the moments when you showed up with intention?",
            }
        )

    narrative, _prompt = asyncio.run(generate_weekly_digest_coach(digest, fake_llm_complete))
    assert narrative is not None

    validation = validate_weekly_digest_narrative(digest, narrative)
    assert validation.groundedness_passed
    assert validation.non_circularity_passed
    assert validation.length_passed

    enriched = attach_coach_artifacts(digest, narrative, validation)
    parquet_path = tmp_path / "weekly_digests.parquet"
    df = persist_weekly_digest_record(enriched, parquet_path)

    assert parquet_path.exists()
    assert df.height == 1
    assert json.loads(df["coach_narrative_json"][0])["weekly_mirror"].startswith("This week")
