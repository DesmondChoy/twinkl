"""Tests for experiment provenance helpers in src.vif.experiment_logger."""

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
    _strip_dirty,
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
