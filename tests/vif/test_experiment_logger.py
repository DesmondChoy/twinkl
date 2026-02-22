"""Tests for experiment provenance helpers in src.vif.experiment_logger."""

import warnings
from pathlib import Path
from unittest.mock import patch

import yaml

import src.vif.experiment_logger as experiment_logger
from src.vif.experiment_logger import (
    _build_provenance,
    _canonicalize_run_config,
    _compute_config_delta,
    _encoder_family,
    _flatten_dict,
    _get_git_log_between,
    _rebuild_index,
    _strip_dirty,
    AUTO_TABLE_END,
    AUTO_TABLE_START,
    TABLE_HEADER,
)


# ─── _strip_dirty ────────────────────────────────────────────────────────────


class TestStripDirty:
    def test_removes_dirty_suffix(self):
        assert _strip_dirty("e1e08c4(dirty)") == "e1e08c4"

    def test_clean_commit_unchanged(self):
        assert _strip_dirty("a3f493f") == "a3f493f"

    def test_unknown_unchanged(self):
        assert _strip_dirty("unknown") == "unknown"

    def test_empty_string(self):
        assert _strip_dirty("") == ""


# ─── _encoder_family ─────────────────────────────────────────────────────────


class TestEncoderFamily:
    def test_minilm(self):
        assert _encoder_family("all-MiniLM-L6-v2") == "MiniLM"

    def test_nomic(self):
        assert _encoder_family("nomic-ai/nomic-embed-text-v1.5") == "nomic"

    def test_mpnet(self):
        assert _encoder_family("sentence-transformers/all-mpnet-base-v2") == "mpnet"

    def test_unknown_uses_last_segment(self):
        assert _encoder_family("org/some-custom-model") == "some-custom-model"

    def test_slashed_path(self):
        assert _encoder_family("my-org/my-minilm-variant") == "MiniLM"


# ─── _flatten_dict ───────────────────────────────────────────────────────────


class TestFlattenDict:
    def test_flat_dict(self):
        assert _flatten_dict({"a": 1, "b": 2}) == {"a": 1, "b": 2}

    def test_nested_dict(self):
        assert _flatten_dict({"a": {"b": 1}}) == {"a.b": 1}

    def test_deep_nesting(self):
        assert _flatten_dict({"a": {"b": {"c": 3}}}) == {"a.b.c": 3}

    def test_empty_dict(self):
        assert _flatten_dict({}) == {}

    def test_mixed_values(self):
        result = _flatten_dict({"x": 1, "y": {"z": 2}, "w": "hello"})
        assert result == {"x": 1, "y.z": 2, "w": "hello"}


# ─── _compute_config_delta ───────────────────────────────────────────────────


class TestComputeConfigDelta:
    def test_identical_configs(self):
        cfg = {"encoder": {"model_name": "foo"}, "training": {"lr": 0.001}}
        delta = _compute_config_delta(cfg, cfg)
        assert delta == {"added": {}, "removed": {}, "changed": {}}

    def test_added_key(self):
        prev = {"a": 1}
        curr = {"a": 1, "b": 2}
        delta = _compute_config_delta(curr, prev)
        assert delta["added"] == {"b": 2}
        assert delta["removed"] == {}
        assert delta["changed"] == {}

    def test_removed_key(self):
        prev = {"a": 1, "b": 2}
        curr = {"a": 1}
        delta = _compute_config_delta(curr, prev)
        assert delta["added"] == {}
        assert delta["removed"] == {"b": 2}
        assert delta["changed"] == {}

    def test_changed_value(self):
        prev = {"a": 1}
        curr = {"a": 2}
        delta = _compute_config_delta(curr, prev)
        assert delta["changed"] == {"a": {"from": 1, "to": 2}}

    def test_nested_removal(self):
        """Removing state_encoder.ema_alpha shows up in flattened delta."""
        prev = {
            "state_encoder": {"window_size": 3, "ema_alpha": 0.3},
        }
        curr = {
            "state_encoder": {"window_size": 3},
        }
        delta = _compute_config_delta(curr, prev)
        assert delta["removed"] == {"state_encoder.ema_alpha": 0.3}
        assert delta["added"] == {}
        assert delta["changed"] == {}

    def test_real_world_run001_to_run003(self):
        """Simulates the actual run_001 → run_003 config change (MiniLM family).

        Between these runs, ema_alpha was removed and data fields were added.
        """
        run001_config = {
            "encoder": {"model_name": "all-MiniLM-L6-v2"},
            "state_encoder": {"window_size": 3, "ema_alpha": 0.3},
            "model": {"hidden_dim": 256, "dropout": 0.2},
            "training": {
                "loss_fn": "coral",
                "learning_rate": 0.001,
                "seed": 2025,
            },
        }
        run003_config = {
            "encoder": {"model_name": "all-MiniLM-L6-v2"},
            "state_encoder": {"window_size": 3},
            "model": {"hidden_dim": 256, "dropout": 0.2},
            "training": {
                "loss_fn": "coral",
                "learning_rate": 0.001,
                "seed": 2025,
            },
        }
        delta = _compute_config_delta(run003_config, run001_config)
        assert delta["removed"] == {"state_encoder.ema_alpha": 0.3}
        assert delta["added"] == {}
        assert delta["changed"] == {}


# ─── _canonicalize_run_config ────────────────────────────────────────────────


class TestCanonicalizeRunConfig:
    def test_removes_head_specific_training_fields(self):
        config = {
            "encoder": {"model_name": "all-MiniLM-L6-v2"},
            "training": {
                "loss_fn": "corn",
                "weighted_mse_scale": 5.0,
                "learning_rate": 0.001,
            },
        }

        canonical = _canonicalize_run_config(config)

        assert canonical["training"] == {"learning_rate": 0.001}
        # Original input should remain unchanged.
        assert config["training"]["loss_fn"] == "corn"
        assert config["training"]["weighted_mse_scale"] == 5.0


# ─── _get_git_log_between ────────────────────────────────────────────────────


class TestGetGitLogBetween:
    def test_same_commit_returns_empty(self):
        assert _get_git_log_between("abc123", "abc123") == []

    def test_unknown_commits_returns_empty(self):
        """Non-existent commits should not crash, just return empty."""
        result = _get_git_log_between("0000000", "fffffff")
        assert isinstance(result, list)

    def test_dirty_suffix_stripped_for_range(self):
        """(dirty) suffix should be stripped before git log, but the call
        should still work (or fail gracefully)."""
        with patch("subprocess.check_output", return_value="abc fix\ndef add\n") as mock:
            result = _get_git_log_between("aaa(dirty)", "bbb")
            assert result == ["abc fix", "def add"]
            # Verify (dirty) was stripped in the git command
            args = mock.call_args[0][0]
            assert "aaa..bbb" in args[-1]

    def test_git_error_returns_empty(self):
        """Git failures (detached HEAD, shallow clone) return empty list."""
        with patch("subprocess.check_output", side_effect=Exception("git error")):
            assert _get_git_log_between("aaa", "bbb") == []


# ─── _build_provenance ───────────────────────────────────────────────────────


class TestBuildProvenance:
    @staticmethod
    def _write_run(path: Path, data: dict) -> None:
        path.write_text(yaml.dump(data, sort_keys=False), encoding="utf-8")

    def test_uses_run_level_canonical_delta(self, tmp_path, monkeypatch):
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(experiment_logger, "RUNS_DIR", runs_dir)

        previous = {
            "metadata": {
                "run_id": "run_001",
                "model_name": "CORAL",
                "git_commit": "e1e08c4(dirty)",
            },
            "config": {
                "encoder": {"model_name": "all-MiniLM-L6-v2"},
                "state_encoder": {"window_size": 3, "ema_alpha": 0.3},
                "training": {"loss_fn": "coral", "learning_rate": 0.001},
            },
        }
        self._write_run(runs_dir / "run_001_CORAL.yaml", previous)

        current_config = {
            "encoder": {"model_name": "all-MiniLM-L6-v2"},
            "state_encoder": {"window_size": 3},
            "training": {"loss_fn": "corn", "learning_rate": 0.001},
        }

        with patch(
            "src.vif.experiment_logger._get_git_log_between",
            return_value=["abc123 change"],
        ):
            provenance = _build_provenance("run_003", current_config, "a3f493f")

        assert provenance["prev_run_id"] == "run_001"
        assert provenance["prev_git_commit"] == "e1e08c4(dirty)"
        assert provenance["git_log"] == ["abc123 change"]
        assert provenance["config_delta"]["removed"] == {"state_encoder.ema_alpha": 0.3}
        # loss_fn is per-head and should not appear in canonical run-level delta
        assert provenance["config_delta"]["changed"] == {}


# ─── _rebuild_index ─────────────────────────────────────────────────────────


def _make_run_yaml(run_id: str, model_name: str) -> dict:
    """Build a minimal valid experiment dict for _rebuild_index tests."""
    return {
        "metadata": {
            "experiment_id": f"{run_id}_{model_name}",
            "run_id": run_id,
            "model_name": model_name,
        },
        "config": {
            "encoder": {"model_name": "nomic-ai/nomic-embed-text-v1.5", "truncate_dim": 256},
            "state_encoder": {"window_size": 1},
            "model": {"hidden_dim": 64, "dropout": 0.3},
            "training": {"loss_fn": "coral"},
        },
        "capacity": {"n_parameters": 22804, "param_sample_ratio": 22.4},
        "evaluation": {
            "mae_mean": 0.209,
            "accuracy_mean": 0.819,
            "qwk_mean": 0.364,
            "spearman_mean": 0.391,
            "calibration_global": 0.823,
            "minority_recall_mean": 0.244,
        },
    }


class TestRebuildIndex:
    """Tests for marker-preserving _rebuild_index behaviour."""

    def _setup_dirs(self, tmp_path, monkeypatch):
        """Wire RUNS_DIR and INDEX_PATH to tmp_path."""
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir(parents=True)
        index_path = tmp_path / "index.md"
        monkeypatch.setattr(experiment_logger, "RUNS_DIR", runs_dir)
        monkeypatch.setattr(experiment_logger, "INDEX_PATH", index_path)
        return runs_dir, index_path

    def _write_run(self, runs_dir: Path, run_id: str, model_name: str) -> None:
        data = _make_run_yaml(run_id, model_name)
        path = runs_dir / f"{run_id}_{model_name}.yaml"
        path.write_text(yaml.dump(data, sort_keys=False), encoding="utf-8")

    def test_fresh_file_created_with_markers(self, tmp_path, monkeypatch):
        """No existing index.md → creates file with markers and table."""
        runs_dir, index_path = self._setup_dirs(tmp_path, monkeypatch)
        self._write_run(runs_dir, "run_001", "CORAL")

        _rebuild_index()

        content = index_path.read_text(encoding="utf-8")
        assert AUTO_TABLE_START in content
        assert AUTO_TABLE_END in content
        assert "run_001" in content
        assert content.startswith("# VIF Experiment Index")

    def test_manual_sections_preserved(self, tmp_path, monkeypatch):
        """Content before START and after END survives rebuild."""
        runs_dir, index_path = self._setup_dirs(tmp_path, monkeypatch)
        self._write_run(runs_dir, "run_001", "CORAL")

        manual_before = "# My Index\n\n## Leaderboard\nBest: run_001\n\n"
        manual_after = "\n## Findings\nImportant analysis here.\n"
        old_table = AUTO_TABLE_START + TABLE_HEADER + "| old row |\n" + AUTO_TABLE_END
        index_path.write_text(manual_before + old_table + manual_after, encoding="utf-8")

        _rebuild_index()

        content = index_path.read_text(encoding="utf-8")
        assert "## Leaderboard" in content
        assert "Best: run_001" in content
        assert "## Findings" in content
        assert "Important analysis here." in content
        # Old row replaced with actual data
        assert "| old row |" not in content
        assert "run_001" in content

    def test_corrupt_yaml_emits_warning(self, tmp_path, monkeypatch):
        """Bad YAML emits warning, doesn't crash, skips the file."""
        runs_dir, index_path = self._setup_dirs(tmp_path, monkeypatch)
        self._write_run(runs_dir, "run_001", "CORAL")
        # Write invalid YAML
        bad_file = runs_dir / "run_002_BAD.yaml"
        bad_file.write_text("invalid: yaml: content: {broken", encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _rebuild_index()

        content = index_path.read_text(encoding="utf-8")
        assert "run_001" in content
        # Warning was emitted for the bad file
        yaml_warnings = [x for x in w if "run_002_BAD.yaml" in str(x.message)]
        assert len(yaml_warnings) == 1

    def test_legacy_file_without_markers_warns(self, tmp_path, monkeypatch):
        """No markers in existing file → appends table, emits warning."""
        runs_dir, index_path = self._setup_dirs(tmp_path, monkeypatch)
        self._write_run(runs_dir, "run_001", "CORAL")

        legacy_content = "# Old Index\n\nSome manual content.\n"
        index_path.write_text(legacy_content, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _rebuild_index()

        content = index_path.read_text(encoding="utf-8")
        # Original content preserved
        assert "# Old Index" in content
        assert "Some manual content." in content
        # Table appended with markers
        assert AUTO_TABLE_START in content
        assert AUTO_TABLE_END in content
        # Warning emitted
        marker_warnings = [x for x in w if "AUTO-TABLE" in str(x.message)]
        assert len(marker_warnings) == 1

    def test_markers_wrong_order_warns(self, tmp_path, monkeypatch):
        """END before START treated as missing markers (appends table)."""
        runs_dir, index_path = self._setup_dirs(tmp_path, monkeypatch)
        self._write_run(runs_dir, "run_001", "CORAL")

        wrong_order = "# Index\n" + AUTO_TABLE_END + "middle\n" + AUTO_TABLE_START
        index_path.write_text(wrong_order, encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _rebuild_index()

        content = index_path.read_text(encoding="utf-8")
        # Original content preserved (not destroyed)
        assert "# Index" in content
        assert "middle" in content
        # Warning emitted about missing markers
        marker_warnings = [x for x in w if "AUTO-TABLE" in str(x.message)]
        assert len(marker_warnings) == 1
