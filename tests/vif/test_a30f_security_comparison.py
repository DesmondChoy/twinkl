from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
import yaml

from scripts.experiments.evaluate_a30f_security_comparison import (
    evaluate_comparison,
    write_comparison_artifacts,
)
from src.models.judge import SCHWARTZ_VALUE_ORDER

ROOT = Path(__file__).resolve().parents[2]


def test_evaluate_comparison_builds_four_cells_and_changes_only_security():
    historical_labels = _labels([[-1, 0, 0, 0, 0, -1, 0, 0, 0, 1], [1] * 10])
    repaired_vectors = historical_labels["alignment_vector"].to_list()
    repaired_vectors[0][SCHWARTZ_VALUE_ORDER.index("security")] = 0
    repaired_labels = historical_labels.with_columns(
        pl.Series("alignment_vector", repaired_vectors)
    )

    result = evaluate_comparison(
        historical_outputs=_outputs(security_predictions=[-1.0, 1.0]),
        repaired_outputs=_outputs(security_predictions=[0.0, 1.0]),
        historical_labels=historical_labels,
        repaired_labels=repaired_labels,
        split="test",
    )

    assert result["sample_count"] == 2
    assert result["changed_security_target_count"] == 1
    assert {
        (cell["model_arm"], cell["target_lens"]) for cell in result["cells"]
    } == {
        ("historical_model", "historical_labels"),
        ("historical_model", "repaired_labels"),
        ("repaired_model", "historical_labels"),
        ("repaired_model", "repaired_labels"),
    }
    repaired_cell = next(
        cell
        for cell in result["cells"]
        if cell["model_arm"] == "repaired_model"
        and cell["target_lens"] == "repaired_labels"
    )
    assert repaired_cell["metrics"]["security_recall"]["zero"] == 1.0
    assert repaired_cell["metrics"]["security_recall"]["plus1"] == 1.0


def test_evaluate_comparison_rejects_nonsecurity_label_changes():
    historical_labels = _labels([[0] * 10, [1] * 10])
    repaired_vectors = historical_labels["alignment_vector"].to_list()
    repaired_vectors[0][SCHWARTZ_VALUE_ORDER.index("tradition")] = 1
    repaired_labels = historical_labels.with_columns(
        pl.Series("alignment_vector", repaired_vectors)
    )

    with pytest.raises(ValueError, match="non-Security"):
        evaluate_comparison(
            historical_outputs=_outputs(),
            repaired_outputs=_outputs(),
            historical_labels=historical_labels,
            repaired_labels=repaired_labels,
            split="test",
        )


def test_evaluate_comparison_rejects_mismatched_model_samples():
    repaired_outputs = _outputs().filter(pl.col("persona_id") == "persona-a")
    labels = _labels([[0] * 10, [1] * 10])

    with pytest.raises(ValueError, match="different samples"):
        evaluate_comparison(
            historical_outputs=_outputs(),
            repaired_outputs=repaired_outputs,
            historical_labels=labels,
            repaired_labels=labels,
            split="test",
        )


def test_write_comparison_artifacts_refuses_overwrite(tmp_path):
    labels = _labels([[0] * 10, [1] * 10])
    result = evaluate_comparison(
        historical_outputs=_outputs(),
        repaired_outputs=_outputs(),
        historical_labels=labels,
        repaired_labels=labels,
        split="test",
    )
    output_dir = tmp_path / "comparison"

    write_comparison_artifacts(result, output_dir)

    assert (output_dir / "comparison_summary.json").is_file()
    assert (output_dir / "comparison_cells.parquet").is_file()
    assert (output_dir / "comparison_report.md").is_file()
    with pytest.raises(FileExistsError):
        write_comparison_artifacts(result, output_dir)


@pytest.mark.parametrize("seed", [11, 22, 33])
def test_paired_configs_change_only_target_regime(seed):
    config_dir = ROOT / "config" / "experiments" / "vif"
    historical = yaml.safe_load(
        (config_dir / f"twinkl_a30f_historical_seed{seed}.yaml").read_text()
    )
    repaired = yaml.safe_load(
        (config_dir / f"twinkl_a30f_repaired_seed{seed}.yaml").read_text()
    )
    target_keys = {
        "labels_path",
        "target_regime",
        "security_target_policy",
        "security_target_artifact_path",
        "security_target_summary_path",
    }

    historical_core = {
        key: value for key, value in historical.items() if key not in target_keys
    }
    repaired_core = {
        key: value for key, value in repaired.items() if key not in target_keys
    }
    assert historical_core == repaired_core
    assert historical["model_seed"] == repaired["model_seed"] == seed
    assert historical["labels_path"] == "logs/judge_labels/judge_labels.parquet"
    assert repaired["labels_path"].endswith("security_repaired_labels.parquet")
    assert repaired["fixed_holdout_manifest_path"].endswith(
        "twinkl_681_5_holdout.yaml"
    )
    assert repaired["dimension_weighting_enabled"] is False


def _labels(vectors: list[list[int]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "persona_id": ["persona-a", "persona-b"],
            "t_index": [0, 0],
            "date": ["2026-01-01", "2026-01-02"],
            "alignment_vector": vectors,
        }
    )


def _outputs(
    *, security_predictions: list[float] | None = None
) -> pl.DataFrame:
    security_predictions = security_predictions or [0.0, 1.0]
    rows = []
    for sample_index, (persona_id, date) in enumerate(
        [("persona-a", "2026-01-01"), ("persona-b", "2026-01-02")]
    ):
        for dimension in SCHWARTZ_VALUE_ORDER:
            prediction = (
                security_predictions[sample_index]
                if dimension == "security"
                else float(sample_index)
            )
            rows.append(
                {
                    "persona_id": persona_id,
                    "t_index": 0,
                    "date": date,
                    "dimension": dimension,
                    "split": "test",
                    "mean_prediction": prediction,
                    "uncertainty": 0.1 + sample_index * 0.1,
                }
            )
    return pl.DataFrame(rows)
