"""Tests for judge label consolidation pipeline."""

import json
from unittest.mock import patch

import polars as pl
import pytest

from src.judge.consolidate import consolidate_judge_labels
from src.models.judge import SCHWARTZ_VALUE_ORDER


# --- Helpers ---


def _make_label_dict(persona_id="a1b2c3d4", n_entries=2):
    return {
        "persona_id": persona_id,
        "labels": [
            {
                "t_index": i,
                "date": f"2025-01-{i + 1:02d}",
                "scores": {
                    "self_direction": 1,
                    "stimulation": 0,
                    "hedonism": -1,
                    "achievement": 0,
                    "power": 0,
                    "security": 1,
                    "conformity": 0,
                    "tradition": 0,
                    "benevolence": 1,
                    "universalism": 0,
                },
                "rationales": {"self_direction": "Chose independent path"}
                if i == 0
                else None,
            }
            for i in range(n_entries)
        ],
    }


def _write_label_file(directory, persona_id, data_dict):
    path = directory / f"persona_{persona_id}_labels.json"
    path.write_text(json.dumps(data_dict))
    return path


# --- Tests ---


class TestConsolidateHappyPath:

    def test_single_valid_file(self, tmp_path):
        data = _make_label_dict()
        _write_label_file(tmp_path, "a1b2c3d4", data)

        df, errors = consolidate_judge_labels(tmp_path, update_registry=False)

        assert len(df) == 2
        assert errors == []

    def test_multiple_valid_files(self, tmp_path):
        for pid in ["aaa11111", "bbb22222", "ccc33333"]:
            _write_label_file(tmp_path, pid, _make_label_dict(persona_id=pid))

        df, errors = consolidate_judge_labels(tmp_path, update_registry=False)

        assert len(df) == 6
        assert df["persona_id"].n_unique() == 3
        assert errors == []

    def test_output_columns_complete(self, tmp_path):
        _write_label_file(tmp_path, "a1b2c3d4", _make_label_dict())

        df, _ = consolidate_judge_labels(tmp_path, update_registry=False)

        expected_columns = (
            ["persona_id", "t_index", "date", "alignment_vector"]
            + [f"alignment_{v}" for v in SCHWARTZ_VALUE_ORDER]
            + ["rationales_json"]
        )
        assert df.columns == expected_columns

    def test_alignment_vector_matches_individual_columns(self, tmp_path):
        _write_label_file(tmp_path, "a1b2c3d4", _make_label_dict())

        df, _ = consolidate_judge_labels(tmp_path, update_registry=False)

        for row in df.iter_rows(named=True):
            vector = row["alignment_vector"]
            individual = [row[f"alignment_{v}"] for v in SCHWARTZ_VALUE_ORDER]
            assert list(vector) == individual

    def test_rationales_json_populated(self, tmp_path):
        _write_label_file(tmp_path, "a1b2c3d4", _make_label_dict())

        df, _ = consolidate_judge_labels(tmp_path, update_registry=False)

        row_0 = df.filter(pl.col("t_index") == 0)
        rationale_str = row_0["rationales_json"][0]
        assert rationale_str is not None
        parsed = json.loads(rationale_str)
        assert parsed["self_direction"] == "Chose independent path"

    def test_rationales_none_when_absent(self, tmp_path):
        _write_label_file(tmp_path, "a1b2c3d4", _make_label_dict())

        df, _ = consolidate_judge_labels(tmp_path, update_registry=False)

        row_1 = df.filter(pl.col("t_index") == 1)
        assert row_1["rationales_json"][0] is None

    def test_writes_parquet_output(self, tmp_path):
        _write_label_file(tmp_path, "a1b2c3d4", _make_label_dict())
        output_path = tmp_path / "output.parquet"

        consolidate_judge_labels(
            tmp_path, output_path=output_path, update_registry=False
        )

        assert output_path.exists()
        reloaded = pl.read_parquet(output_path)
        assert len(reloaded) == 2


class TestConsolidateErrorHandling:

    def test_invalid_json_collected(self, tmp_path):
        bad_file = tmp_path / "persona_bad1bad1_labels.json"
        bad_file.write_text("{not valid json")
        _write_label_file(tmp_path, "a1b2c3d4", _make_label_dict())

        df, errors = consolidate_judge_labels(tmp_path, update_registry=False)

        assert len(df) == 2
        assert len(errors) == 1
        assert "Invalid JSON" in errors[0]

    def test_pydantic_validation_error(self, tmp_path):
        bad_data = _make_label_dict(persona_id="INVALID!")
        _write_label_file(tmp_path, "INVALID!", bad_data)
        _write_label_file(tmp_path, "a1b2c3d4", _make_label_dict())

        df, errors = consolidate_judge_labels(tmp_path, update_registry=False)

        assert len(df) == 2
        assert len(errors) == 1
        assert "Validation failed" in errors[0]

    def test_score_out_of_range(self, tmp_path):
        data = _make_label_dict()
        data["labels"][0]["scores"]["self_direction"] = 5
        _write_label_file(tmp_path, "a1b2c3d4", data)

        # Only file is invalid, so raises ValueError
        with pytest.raises(ValueError, match="No valid labels"):
            consolidate_judge_labels(tmp_path, update_registry=False)

    def test_all_invalid_raises_valueerror(self, tmp_path):
        bad_file = tmp_path / "persona_bad1bad1_labels.json"
        bad_file.write_text("{broken")

        with pytest.raises(ValueError, match="No valid labels"):
            consolidate_judge_labels(tmp_path, update_registry=False)

    def test_mixed_valid_and_invalid(self, tmp_path):
        _write_label_file(tmp_path, "a1b2c3d4", _make_label_dict())
        bad_file = tmp_path / "persona_bad1bad1_labels.json"
        bad_file.write_text("{broken")

        df, errors = consolidate_judge_labels(tmp_path, update_registry=False)

        assert len(df) == 2
        assert len(errors) == 1

    def test_empty_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            consolidate_judge_labels(tmp_path, update_registry=False)

    def test_empty_labels_list_only_file(self, tmp_path):
        data = {"persona_id": "a1b2c3d4", "labels": []}
        _write_label_file(tmp_path, "a1b2c3d4", data)

        with pytest.raises(ValueError, match="No valid labels"):
            consolidate_judge_labels(tmp_path, update_registry=False)


class TestConsolidateRegistryIntegration:

    @patch("src.registry.update_stage")
    def test_update_registry_true_calls_update_stage(self, mock_update, tmp_path):
        for pid in ["aaa11111", "bbb22222"]:
            _write_label_file(tmp_path, pid, _make_label_dict(persona_id=pid))

        consolidate_judge_labels(tmp_path, update_registry=True)

        assert mock_update.call_count == 2
        called_ids = {call.args[0] for call in mock_update.call_args_list}
        assert called_ids == {"aaa11111", "bbb22222"}

    @patch("src.registry.update_stage")
    def test_update_registry_false_skips(self, mock_update, tmp_path):
        _write_label_file(tmp_path, "a1b2c3d4", _make_label_dict())

        consolidate_judge_labels(tmp_path, update_registry=False)

        mock_update.assert_not_called()

    @patch("src.registry.update_stage", side_effect=ValueError("Not in registry"))
    def test_registry_error_non_fatal(self, mock_update, tmp_path):
        _write_label_file(tmp_path, "a1b2c3d4", _make_label_dict())

        df, errors = consolidate_judge_labels(tmp_path, update_registry=True)

        assert len(df) == 2
        assert any("Registry" in e for e in errors)


class TestJudgeSchemaContract:

    def test_output_dtypes(self, tmp_path):
        _write_label_file(tmp_path, "a1b2c3d4", _make_label_dict())

        df, _ = consolidate_judge_labels(tmp_path, update_registry=False)

        assert df.schema["persona_id"] == pl.Utf8
        assert df.schema["t_index"] == pl.Int64
        assert df.schema["alignment_vector"] == pl.List(pl.Int64)

    def test_alignment_columns_match_schwartz_order(self, tmp_path):
        _write_label_file(tmp_path, "a1b2c3d4", _make_label_dict())

        df, _ = consolidate_judge_labels(tmp_path, update_registry=False)

        alignment_cols = [
            c.replace("alignment_", "")
            for c in df.columns
            if c.startswith("alignment_") and c != "alignment_vector"
        ]
        assert alignment_cols == SCHWARTZ_VALUE_ORDER
