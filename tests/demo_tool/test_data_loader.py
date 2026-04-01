from pathlib import Path

from src.demo_tool.data_loader import build_persona_choices, get_persona, load_demo_personas


def _write_persona(
    path: Path,
    *,
    persona_id: str,
    name: str,
    start_date: str,
    second_date: str,
) -> None:
    path.write_text(
        f"""# Persona {persona_id}: {name}

## Profile
- **Persona ID:** {persona_id}
- **Name:** {name}
- **Age:** 25-34
- **Profession:** Designer
- **Culture:** Singaporean
- **Core Values:** Benevolence, Self Direction
- **Bio:** Demo loader fixture.

---

## Entry 0 - {start_date}

First entry.

---

## Entry 1 - {second_date}

Second entry.

---
"""
    )


def test_load_demo_personas_groups_entries_and_builds_choices(tmp_path: Path):
    wrangled_dir = tmp_path / "wrangled"
    wrangled_dir.mkdir()

    _write_persona(
        wrangled_dir / "persona_deadbeef.md",
        persona_id="deadbeef",
        name="Casey",
        start_date="2025-01-06",
        second_date="2025-01-08",
    )
    _write_persona(
        wrangled_dir / "persona_cafebabe.md",
        persona_id="cafebabe",
        name="Jordan",
        start_date="2025-02-01",
        second_date="2025-02-03",
    )

    result = load_demo_personas(wrangled_dir=wrangled_dir, registry_path=tmp_path / "missing.parquet")

    assert result.total_personas == 2
    assert result.total_entries == 4
    assert result.warnings == []

    persona = get_persona(result.personas, "deadbeef")
    assert persona is not None
    assert persona["persona_name"] == "Casey"
    assert persona["n_entries"] == 2
    assert persona["entries"][0]["date"] == "2025-01-06"
    assert persona["entries"][1]["date"] == "2025-01-08"

    choices = build_persona_choices(result.personas)
    assert "deadbeef" in choices
    assert "2 entries" in choices["deadbeef"]
