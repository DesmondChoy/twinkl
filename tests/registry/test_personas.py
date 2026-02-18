"""Tests for persona registry CRUD operations and pipeline stage tracking."""

import polars as pl
import pytest

from src.registry.personas import (
    REGISTRY_SCHEMA,
    get_pending,
    get_registry,
    get_status,
    register_persona,
    update_stage,
)

# --- Isolation helper ---


def _patch_registry(monkeypatch, tmp_path):
    """Redirect all registry I/O to a temp directory."""
    fake = tmp_path / "personas.parquet"
    monkeypatch.setattr("src.registry.personas.REGISTRY_PATH", fake)
    return fake


def _register_sample(persona_id="aaaabbbb", **overrides):
    """Register a persona with sensible defaults, accepting overrides."""
    defaults = dict(
        persona_id=persona_id,
        name="Test User",
        age="25-34",
        profession="Engineer",
        culture="Western European",
        core_values=["Security", "Benevolence"],
        entry_count=5,
    )
    defaults.update(overrides)
    register_persona(**defaults)


class TestRegisterPersona:
    """Creation and duplicate protection."""

    def test_registers_new_persona(self, monkeypatch, tmp_path):
        _patch_registry(monkeypatch, tmp_path)
        _register_sample()

        df = get_registry()
        assert len(df) == 1

        row = df.row(0, named=True)
        assert row["persona_id"] == "aaaabbbb"
        assert row["name"] == "Test User"
        assert row["age"] == "25-34"
        assert row["profession"] == "Engineer"
        assert row["culture"] == "Western European"
        assert row["core_values"] == ["Security", "Benevolence"]
        assert row["entry_count"] == 5
        assert row["stage_synthetic"] is True
        assert row["stage_wrangled"] is False
        assert row["stage_labeled"] is False
        assert row["nudge_enabled"] is True
        assert row["annotation_order"] is None
        assert row["created_at"] is not None
        assert len(df.columns) == 13

    def test_duplicate_persona_id_raises(self, monkeypatch, tmp_path):
        _patch_registry(monkeypatch, tmp_path)
        _register_sample()

        with pytest.raises(ValueError, match="already exists"):
            _register_sample()

    def test_nudge_enabled_default_true(self, monkeypatch, tmp_path):
        _patch_registry(monkeypatch, tmp_path)
        _register_sample()

        row = get_registry().row(0, named=True)
        assert row["nudge_enabled"] is True

    def test_nudge_enabled_false(self, monkeypatch, tmp_path):
        _patch_registry(monkeypatch, tmp_path)
        _register_sample(nudge_enabled=False)

        row = get_registry().row(0, named=True)
        assert row["nudge_enabled"] is False

    def test_multiple_personas_accumulate(self, monkeypatch, tmp_path):
        _patch_registry(monkeypatch, tmp_path)
        _register_sample(persona_id="aaaa0001")
        _register_sample(persona_id="aaaa0002")
        _register_sample(persona_id="aaaa0003")

        assert len(get_registry()) == 3


class TestUpdateStage:
    """Stage progression and error handling."""

    def test_update_wrangled(self, monkeypatch, tmp_path):
        _patch_registry(monkeypatch, tmp_path)
        _register_sample()
        update_stage("aaaabbbb", "wrangled")

        row = get_registry().row(0, named=True)
        assert row["stage_wrangled"] is True
        assert row["stage_labeled"] is False

    def test_update_labeled(self, monkeypatch, tmp_path):
        _patch_registry(monkeypatch, tmp_path)
        _register_sample()
        update_stage("aaaabbbb", "labeled")

        row = get_registry().row(0, named=True)
        assert row["stage_labeled"] is True

    def test_invalid_stage_raises(self, monkeypatch, tmp_path):
        _patch_registry(monkeypatch, tmp_path)
        _register_sample()

        with pytest.raises(ValueError, match="Invalid stage"):
            update_stage("aaaabbbb", "bogus")

    def test_missing_persona_raises(self, monkeypatch, tmp_path):
        _patch_registry(monkeypatch, tmp_path)
        _register_sample()

        with pytest.raises(ValueError, match="not found"):
            update_stage("nonexistent", "wrangled")

    def test_update_does_not_affect_other_personas(self, monkeypatch, tmp_path):
        _patch_registry(monkeypatch, tmp_path)
        _register_sample(persona_id="aaaa0001")
        _register_sample(persona_id="aaaa0002")

        update_stage("aaaa0001", "wrangled")

        df = get_registry()
        other = df.filter(pl.col("persona_id") == "aaaa0002").row(0, named=True)
        assert other["stage_wrangled"] is False


class TestGetPending:
    """Stage-aware filtering."""

    def test_pending_wrangled_returns_unsynthetic_only(self, monkeypatch, tmp_path):
        _patch_registry(monkeypatch, tmp_path)
        _register_sample(persona_id="aaaa0001")
        _register_sample(persona_id="aaaa0002")
        update_stage("aaaa0001", "wrangled")

        pending = get_pending("wrangled")
        assert len(pending) == 1
        assert pending["persona_id"][0] == "aaaa0002"

    def test_pending_labeled_returns_wrangled_only(self, monkeypatch, tmp_path):
        _patch_registry(monkeypatch, tmp_path)
        _register_sample(persona_id="aaaa0001")
        _register_sample(persona_id="aaaa0002")
        update_stage("aaaa0001", "wrangled")
        update_stage("aaaa0002", "wrangled")
        update_stage("aaaa0001", "labeled")

        pending = get_pending("labeled")
        assert len(pending) == 1
        assert pending["persona_id"][0] == "aaaa0002"

    def test_pending_labeled_excludes_unwrangled(self, monkeypatch, tmp_path):
        _patch_registry(monkeypatch, tmp_path)
        _register_sample(persona_id="aaaa0001")  # synthetic only

        pending = get_pending("labeled")
        assert len(pending) == 0

    def test_empty_registry_returns_empty(self, monkeypatch, tmp_path):
        _patch_registry(monkeypatch, tmp_path)
        # Force registry creation without registering anyone
        get_registry()

        pending = get_pending("wrangled")
        assert len(pending) == 0


class TestGetStatus:
    """Summary counts."""

    def test_empty_registry_all_zeros(self, monkeypatch, tmp_path):
        _patch_registry(monkeypatch, tmp_path)
        get_registry()  # ensure file exists

        status = get_status()
        assert status == {
            "total": 0,
            "synthetic": 0,
            "wrangled": 0,
            "labeled": 0,
            "pending_wrangling": 0,
            "pending_labeling": 0,
        }

    def test_counts_after_full_pipeline(self, monkeypatch, tmp_path):
        _patch_registry(monkeypatch, tmp_path)
        _register_sample(persona_id="aaaa0001")
        _register_sample(persona_id="aaaa0002")
        _register_sample(persona_id="aaaa0003")

        update_stage("aaaa0001", "wrangled")
        update_stage("aaaa0002", "wrangled")
        update_stage("aaaa0001", "labeled")

        status = get_status()
        assert status["total"] == 3
        assert status["synthetic"] == 3
        assert status["wrangled"] == 2
        assert status["labeled"] == 1
        assert status["pending_wrangling"] == 1
        assert status["pending_labeling"] == 1


class TestRegistrySchema:
    """Schema contract."""

    def test_schema_matches_constant(self, monkeypatch, tmp_path):
        _patch_registry(monkeypatch, tmp_path)
        _register_sample()

        df = get_registry()
        assert dict(df.schema) == REGISTRY_SCHEMA

    def test_persona_id_utf8_for_join_compatibility(self, monkeypatch, tmp_path):
        _patch_registry(monkeypatch, tmp_path)
        _register_sample()

        df = get_registry()
        assert df.schema["persona_id"] == pl.Utf8
