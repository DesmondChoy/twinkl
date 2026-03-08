"""Tests for validation-only post-hoc tuning utilities."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from src.models.judge import SCHWARTZ_VALUE_ORDER
from src.vif.posthoc import (
    CLASS_VALUES,
    DEFAULT_CONFIG,
    NUM_CLASSES,
    NUM_DIMS,
    _best_softmax_family,
    _corn_margin_predictions,
    _is_candidate_eligible,
    _pick_best_candidate,
    _policy_family_for_model,
    _policy_rank_key,
    _render_report,
    _softmax_logit_adjustment,
    compute_class_priors,
    run_posthoc,
)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / exp_shifted.sum(axis=-1, keepdims=True)


def _expected_score(probabilities: np.ndarray) -> np.ndarray:
    return probabilities @ CLASS_VALUES


def test_softmax_logit_adjustment_tau_zero_is_identity():
    rng = np.random.default_rng(7)
    raw_logits = rng.normal(size=(4, NUM_DIMS, NUM_CLASSES))
    priors = np.tile(np.array([0.1, 0.8, 0.1], dtype=np.float64), (NUM_DIMS, 1))

    adjusted = _softmax_logit_adjustment(raw_logits, priors.reshape(1, NUM_DIMS, NUM_CLASSES), 0.0)

    assert np.allclose(adjusted, _softmax(raw_logits))


def test_compute_class_priors_counts_alignment_vectors():
    alignment_rows = [
        [-1, 0, 1, -1, 0, 1, -1, 0, 1, -1],
        [0, 0, 1, 1, 0, -1, -1, 1, 0, 1],
        [1, -1, 0, 0, -1, 0, 1, -1, 0, 0],
    ]
    train_df = pl.DataFrame({"alignment_vector": alignment_rows})

    priors = compute_class_priors(train_df)

    assert priors.shape == (NUM_DIMS, NUM_CLASSES)
    assert np.allclose(priors[0], np.array([1 / 3, 1 / 3, 1 / 3]))
    assert np.allclose(priors[1], np.array([1 / 3, 2 / 3, 0.0]), atol=1e-8)
    assert np.allclose(priors.sum(axis=1), 1.0)


def test_is_candidate_eligible_enforces_qwk_guard():
    config = {
        "selection_policy": {
            "max_qwk_drop": 0.03,
            "require_non_negative_calibration": True,
        }
    }
    baseline_metrics = {
        "qwk_mean": 0.42,
    }
    candidate_metrics = {
        "qwk_mean": 0.38,
        "qwk_nan_dims_count": 0,
        "calibration": {"error_uncertainty_correlation": 0.2},
    }

    eligible, reasons = _is_candidate_eligible(candidate_metrics, baseline_metrics, config)

    assert eligible is False
    assert "qwk_drop_exceeds_guard" in reasons


def test_is_candidate_eligible_rejects_non_finite_calibration():
    config = {
        "selection_policy": {
            "max_qwk_drop": 0.03,
            "require_non_negative_calibration": True,
        }
    }
    baseline_metrics = {
        "qwk_mean": 0.42,
    }
    candidate_metrics = {
        "qwk_mean": 0.41,
        "qwk_nan_dims_count": 0,
        "calibration": {"error_uncertainty_correlation": float("nan")},
    }

    eligible, reasons = _is_candidate_eligible(candidate_metrics, baseline_metrics, config)

    assert eligible is False
    assert "non_finite_calibration" in reasons


def test_policy_rank_key_respects_config_rank_order():
    metrics = {
        "recall_minus1": 0.20,
        "qwk_mean": 0.40,
        "decision_neutral_rate": 0.75,
        "calibration": {"error_uncertainty_correlation": 0.50},
    }
    config = {
        "selection_policy": {
            "rank_order": [
                "qwk_mean",
                "recall_minus1",
                "decision_neutral_rate",
                "calibration_global",
            ]
        }
    }

    rank_key = _policy_rank_key(metrics, config)

    assert rank_key == [0.40, 0.20, -0.75, 0.50]


def test_pick_best_candidate_falls_back_to_baseline_when_none_are_eligible():
    baseline = {
        "policy_name": "baseline",
        "eligible": False,
        "selection_score": {"rank_key": [0.1, 0.3, 0.5, -0.8]},
    }
    challenger = {
        "policy_name": "challenger",
        "eligible": False,
        "selection_score": {"rank_key": [0.2, 0.4, 0.4, -0.7]},
    }

    selected = _pick_best_candidate(
        candidate_records=[baseline, challenger],
        baseline_policy_name="baseline",
    )

    assert selected["policy_name"] == "baseline"


def test_policy_family_for_model_requires_enabled_target():
    config = deepcopy(DEFAULT_CONFIG)
    config["softmax_logit_adjustment"]["enabled"] = False

    assert _policy_family_for_model("CORN", config) == "corn"

    try:
        _policy_family_for_model("SoftOrdinal", config)
    except ValueError as exc:
        assert "not enabled" in str(exc)
    else:
        raise AssertionError("Expected disabled softmax target to raise a ValueError.")


def test_policy_family_for_model_uses_configured_softmax_targets():
    config = deepcopy(DEFAULT_CONFIG)
    config["softmax_logit_adjustment"]["target_models"] = ["BalancedSoftmax"]
    config["corn_threshold_policy"]["target_models"] = ["CORN"]

    assert _policy_family_for_model("BalancedSoftmax", config) == "softmax"

    try:
        _policy_family_for_model("SoftOrdinal", config)
    except ValueError as exc:
        assert "not enabled" in str(exc)
    else:
        raise AssertionError("Expected non-target softmax model to be rejected.")


def test_best_softmax_family_uses_config_target_models():
    summary_by_model = {
        "CDWCE_a3": {
            "recall_minus1_median": 0.30,
            "qwk_mean_median": 0.35,
            "calibration_global_median": 0.50,
            "decision_neutral_rate_median": 0.74,
        },
        "BalancedSoftmax": {
            "recall_minus1_median": 0.28,
            "qwk_mean_median": 0.36,
            "calibration_global_median": 0.60,
            "decision_neutral_rate_median": 0.70,
        },
    }
    config = deepcopy(DEFAULT_CONFIG)
    config["softmax_logit_adjustment"]["target_models"] = ["BalancedSoftmax"]

    assert _best_softmax_family(summary_by_model, config) == "BalancedSoftmax"


def test_best_softmax_family_returns_none_without_configured_targets():
    summary_by_model = {
        "BalancedSoftmax": {
            "recall_minus1_median": 0.31,
            "qwk_mean_median": 0.33,
            "calibration_global_median": 0.12,
            "decision_neutral_rate_median": 0.42,
        }
    }
    config = deepcopy(DEFAULT_CONFIG)
    config["softmax_logit_adjustment"]["target_models"] = []

    assert _best_softmax_family(summary_by_model, config) is None


def test_corn_margin_predictions_can_promote_minus1_without_research():
    probabilities = np.zeros((2, NUM_DIMS, NUM_CLASSES), dtype=np.float64)
    probabilities[0, :, :] = np.array([0.35, 0.40, 0.25])
    probabilities[1, :, :] = np.array([0.15, 0.30, 0.55])

    predicted = _corn_margin_predictions(
        probabilities,
        minus_margins=np.full(NUM_DIMS, 0.10),
        plus_margins=np.zeros(NUM_DIMS),
    )

    assert np.all(predicted[0] == -1)
    assert np.all(predicted[1] == 1)


def _write_output_artifact(
    path: Path,
    *,
    model_name: str,
    split: str,
    raw_logits: np.ndarray,
    probabilities: np.ndarray,
    targets: np.ndarray,
) -> None:
    rows = []
    expected_scores = _expected_score(probabilities)
    predicted_classes = probabilities.argmax(axis=-1) - 1

    for sample_idx in range(targets.shape[0]):
        persona_id = f"persona_{sample_idx:03d}"
        for dim_idx, dimension in enumerate(SCHWARTZ_VALUE_ORDER):
            rows.append(
                {
                    "persona_id": persona_id,
                    "t_index": sample_idx,
                    "date": f"2025-01-{sample_idx + 1:02d}",
                    "split": split,
                    "model_name": model_name,
                    "dimension": dimension,
                    "target": int(targets[sample_idx, dim_idx]),
                    "predicted_class": int(predicted_classes[sample_idx, dim_idx]),
                    "mean_prediction": float(expected_scores[sample_idx, dim_idx]),
                    "uncertainty": 0.15,
                    "raw_logits": raw_logits[sample_idx, dim_idx].tolist(),
                    "class_probabilities": probabilities[sample_idx, dim_idx].tolist(),
                }
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(path)


def _build_softmax_fixture_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_logits = np.array(
        [
            [-0.10, 0.40, -0.90],
            [-0.50, 1.00, -0.60],
            [-0.80, 0.10, 0.50],
            [-0.15, 0.35, -0.70],
        ],
        dtype=np.float64,
    )
    raw_logits = np.repeat(sample_logits[:, None, :], NUM_DIMS, axis=1)
    probabilities = _softmax(raw_logits)
    targets = np.repeat(np.array([[-1], [0], [1], [-1]], dtype=np.int64), NUM_DIMS, axis=1)
    return raw_logits, probabilities, targets


def _build_corn_fixture_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_probabilities = np.array(
        [
            [0.35, 0.40, 0.25],
            [0.15, 0.70, 0.15],
            [0.15, 0.35, 0.50],
            [0.34, 0.38, 0.28],
        ],
        dtype=np.float64,
    )
    probabilities = np.repeat(sample_probabilities[:, None, :], NUM_DIMS, axis=1)

    raw_logits = np.zeros((sample_probabilities.shape[0], NUM_DIMS, 2), dtype=np.float64)
    for sample_idx, probs in enumerate(sample_probabilities):
        first_threshold = np.clip(1.0 - probs[0], 1e-6, 1.0 - 1e-6)
        second_threshold = np.clip(probs[2] / first_threshold if first_threshold > 0 else 0.5, 1e-6, 1.0 - 1e-6)
        logits = np.array(
            [
                np.log(first_threshold / (1.0 - first_threshold)),
                np.log(second_threshold / (1.0 - second_threshold)),
            ],
            dtype=np.float64,
        )
        raw_logits[sample_idx, :, :] = logits

    targets = np.repeat(np.array([[-1], [0], [1], [-1]], dtype=np.int64), NUM_DIMS, axis=1)
    return raw_logits, probabilities, targets


def _summary_row(
    *,
    qwk: float,
    recall_minus1: float,
    minority_recall: float,
    neutral_rate: float,
    calibration: float,
    qwk_iqr: float = 0.01,
    recall_iqr: float = 0.02,
) -> dict[str, float]:
    return {
        "qwk_mean_median": qwk,
        "qwk_mean_iqr": qwk_iqr,
        "recall_minus1_median": recall_minus1,
        "recall_minus1_iqr": recall_iqr,
        "minority_recall_mean_median": minority_recall,
        "decision_neutral_rate_median": neutral_rate,
        "calibration_global_median": calibration,
    }


def test_render_report_uses_config_driven_softmax_labels_and_order():
    summary = {
        "tuned_runs": [
            {
                "run_id": "run_016",
                "model_name": "LDAM_DRW",
                "selected_policy": {"policy_name": "tau_0.3"},
                "test_metrics": {
                    "qwk_mean": 0.322,
                    "recall_minus1": 0.281,
                    "minority_recall_mean": 0.236,
                    "decision_neutral_rate": 0.418,
                    "calibration_global": 0.114,
                },
            },
            {
                "run_id": "run_016",
                "model_name": "BalancedSoftmax",
                "selected_policy": {"policy_name": "tau_0.5"},
                "test_metrics": {
                    "qwk_mean": 0.331,
                    "recall_minus1": 0.295,
                    "minority_recall_mean": 0.247,
                    "decision_neutral_rate": 0.401,
                    "calibration_global": 0.121,
                },
            },
        ],
        "summary_by_model": {
            "LDAM_DRW": _summary_row(
                qwk=0.322,
                recall_minus1=0.281,
                minority_recall=0.236,
                neutral_rate=0.418,
                calibration=0.114,
            ),
            "BalancedSoftmax": _summary_row(
                qwk=0.331,
                recall_minus1=0.295,
                minority_recall=0.247,
                neutral_rate=0.401,
                calibration=0.121,
            ),
            "CORN": _summary_row(
                qwk=0.300,
                recall_minus1=0.220,
                minority_recall=0.210,
                neutral_rate=0.450,
                calibration=0.090,
            ),
        },
        "family_delta_summary": {
            "softmax_logit_adjustment": {
                "median_recall_minus1_delta": 0.021,
                "median_qwk_mean_delta": 0.008,
            },
            "corn_margin_threshold": {
                "median_recall_minus1_delta": 0.010,
                "median_qwk_mean_delta": 0.004,
            },
        },
        "recommended_softmax_base": "BalancedSoftmax",
        "generated_at": "2026-03-07T10:00:00",
    }
    config = deepcopy(DEFAULT_CONFIG)
    config["models"] = ["LDAM_DRW", "BalancedSoftmax", "CORN"]
    config["summary_model_order"] = ["BalancedSoftmax", "LDAM_DRW", "CORN"]
    config["report_title"] = "Experiment Review — 2026-03-07 — twinkl-681.4 long-tail softmax losses"
    config["recommended_model_label"] = "Recommended long-tail softmax base for `twinkl-681.4`"

    report = _render_report(summary, config)

    assert "# Experiment Review — 2026-03-07 — twinkl-681.4 long-tail softmax losses" in report
    assert "- Recommended long-tail softmax base for `twinkl-681.4`: `BalancedSoftmax`." in report

    median_section = report.split("## Median / IQR by Family", maxsplit=1)[1].split(
        "## Policy-Level Takeaways",
        maxsplit=1,
    )[0]
    assert median_section.index("| BalancedSoftmax |") < median_section.index("| LDAM_DRW |")


def test_run_posthoc_smoke_writes_artifacts_and_preserves_source_runs(tmp_path, monkeypatch):
    runs_dir = tmp_path / "logs" / "experiments" / "runs"
    artifact_root = tmp_path / "logs" / "experiments" / "artifacts"
    report_path = tmp_path / "logs" / "experiments" / "reports" / "posthoc_report.md"
    config = deepcopy(DEFAULT_CONFIG)
    config["runs_dir"] = str(runs_dir.relative_to(tmp_path))
    config["artifact_root"] = str(artifact_root.relative_to(tmp_path))
    config["report_path"] = str(report_path.relative_to(tmp_path))
    config["run_ids"] = ["run_999"]
    config["models"] = ["SoftOrdinal", "CORN"]
    config["softmax_logit_adjustment"]["target_models"] = ["SoftOrdinal"]
    config["corn_threshold_policy"]["target_models"] = ["CORN"]

    monkeypatch.setattr(
        "src.vif.posthoc.reconstruct_train_priors",
        lambda **_: np.tile(np.array([0.1, 0.8, 0.1], dtype=np.float64), (NUM_DIMS, 1)),
    )

    softmax_val_logits, softmax_val_probs, softmax_val_targets = _build_softmax_fixture_arrays()
    softmax_test_logits, softmax_test_probs, softmax_test_targets = _build_softmax_fixture_arrays()
    corn_val_logits, corn_val_probs, corn_val_targets = _build_corn_fixture_arrays()
    corn_test_logits, corn_test_probs, corn_test_targets = _build_corn_fixture_arrays()

    softmax_val_path = artifact_root / "source" / "SoftOrdinal" / "val.parquet"
    softmax_test_path = artifact_root / "source" / "SoftOrdinal" / "test.parquet"
    corn_val_path = artifact_root / "source" / "CORN" / "val.parquet"
    corn_test_path = artifact_root / "source" / "CORN" / "test.parquet"

    _write_output_artifact(
        softmax_val_path,
        model_name="SoftOrdinal",
        split="val",
        raw_logits=softmax_val_logits,
        probabilities=softmax_val_probs,
        targets=softmax_val_targets,
    )
    _write_output_artifact(
        softmax_test_path,
        model_name="SoftOrdinal",
        split="test",
        raw_logits=softmax_test_logits,
        probabilities=softmax_test_probs,
        targets=softmax_test_targets,
    )
    _write_output_artifact(
        corn_val_path,
        model_name="CORN",
        split="val",
        raw_logits=corn_val_logits,
        probabilities=corn_val_probs,
        targets=corn_val_targets,
    )
    _write_output_artifact(
        corn_test_path,
        model_name="CORN",
        split="test",
        raw_logits=corn_test_logits,
        probabilities=corn_test_probs,
        targets=corn_test_targets,
    )

    for model_name, val_path, test_path in [
        ("SoftOrdinal", softmax_val_path, softmax_test_path),
        ("CORN", corn_val_path, corn_test_path),
    ]:
        run_path = runs_dir / f"run_999_{model_name}.yaml"
        run_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path = artifact_root / "source" / model_name / "selected_checkpoint.pt"
        checkpoint_path.write_bytes(b"stub")
        run_payload = {
            "config": {
                "data": {
                    "split_seed": 2025,
                    "train_ratio": 0.7,
                    "val_ratio": 0.15,
                }
            },
            "artifacts": {
                "checkpoint": str(checkpoint_path.relative_to(tmp_path)),
                "validation_outputs": str(val_path.relative_to(tmp_path)),
                "test_outputs": str(test_path.relative_to(tmp_path)),
            },
        }
        run_path.write_text(yaml.safe_dump(run_payload, sort_keys=False), encoding="utf-8")

    original_softordinal_yaml = (runs_dir / "run_999_SoftOrdinal.yaml").read_text(encoding="utf-8")
    original_corn_yaml = (runs_dir / "run_999_CORN.yaml").read_text(encoding="utf-8")

    summary = run_posthoc(config, repo_root=tmp_path)

    assert len(summary["tuned_runs"]) == 2
    assert Path(summary["summary_path"]).is_file()
    assert Path(summary["report_path"]).is_file()

    soft_record = next(record for record in summary["tuned_runs"] if record["model_name"] == "SoftOrdinal")
    corn_record = next(record for record in summary["tuned_runs"] if record["model_name"] == "CORN")

    assert soft_record["selected_policy"]["policy_family"] == "softmax_logit_adjustment"
    assert corn_record["selected_policy"]["policy_family"] == "corn_margin_threshold"
    assert Path(soft_record["selected_policy_path"]).is_file()
    assert Path(corn_record["selected_policy_path"]).is_file()

    soft_policy = yaml.safe_load(Path(soft_record["selected_policy_path"]).read_text(encoding="utf-8"))
    corn_policy = yaml.safe_load(Path(corn_record["selected_policy_path"]).read_text(encoding="utf-8"))

    assert "selection_score" in soft_policy
    assert "logit_policy" in soft_policy
    assert "threshold_policy" in corn_policy
    assert Path(soft_policy["artifacts"]["validation_sweep"]).is_file()
    assert Path(soft_policy["artifacts"]["tuned_test_outputs"]).is_file()
    assert Path(corn_policy["artifacts"]["tuned_test_outputs"]).is_file()

    assert (runs_dir / "run_999_SoftOrdinal.yaml").read_text(encoding="utf-8") == original_softordinal_yaml
    assert (runs_dir / "run_999_CORN.yaml").read_text(encoding="utf-8") == original_corn_yaml


def test_run_posthoc_respects_configured_softmax_model_labels_and_prefix(tmp_path, monkeypatch):
    runs_dir = tmp_path / "logs" / "experiments" / "runs"
    artifact_root = tmp_path / "logs" / "experiments" / "artifacts"
    report_path = tmp_path / "logs" / "experiments" / "reports" / "balanced_posthoc_report.md"
    config = deepcopy(DEFAULT_CONFIG)
    config["runs_dir"] = str(runs_dir.relative_to(tmp_path))
    config["artifact_root"] = str(artifact_root.relative_to(tmp_path))
    config["artifact_run_prefix"] = "posthoc_twinkl_681_4_stage1"
    config["report_path"] = str(report_path.relative_to(tmp_path))
    config["report_title"] = "Experiment Review — twinkl-681.4 tuned softmax long-tail comparison"
    config["recommended_model_label"] = "Recommended stage-1 softmax leader"
    config["summary_model_order"] = ["BalancedSoftmax"]
    config["run_ids"] = ["run_999"]
    config["models"] = ["BalancedSoftmax"]
    config["softmax_logit_adjustment"]["target_models"] = ["BalancedSoftmax"]
    config["corn_threshold_policy"]["enabled"] = False

    monkeypatch.setattr(
        "src.vif.posthoc.reconstruct_train_priors",
        lambda **_: np.tile(np.array([0.1, 0.8, 0.1], dtype=np.float64), (NUM_DIMS, 1)),
    )

    val_logits, val_probs, val_targets = _build_softmax_fixture_arrays()
    test_logits, test_probs, test_targets = _build_softmax_fixture_arrays()
    val_path = artifact_root / "source" / "BalancedSoftmax" / "val.parquet"
    test_path = artifact_root / "source" / "BalancedSoftmax" / "test.parquet"

    _write_output_artifact(
        val_path,
        model_name="BalancedSoftmax",
        split="val",
        raw_logits=val_logits,
        probabilities=val_probs,
        targets=val_targets,
    )
    _write_output_artifact(
        test_path,
        model_name="BalancedSoftmax",
        split="test",
        raw_logits=test_logits,
        probabilities=test_probs,
        targets=test_targets,
    )

    run_path = runs_dir / "run_999_BalancedSoftmax.yaml"
    run_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_root / "source" / "BalancedSoftmax" / "selected_checkpoint.pt"
    checkpoint_path.write_bytes(b"stub")
    run_payload = {
        "config": {
            "data": {
                "split_seed": 2025,
                "train_ratio": 0.7,
                "val_ratio": 0.15,
            }
        },
        "artifacts": {
            "checkpoint": str(checkpoint_path.relative_to(tmp_path)),
            "validation_outputs": str(val_path.relative_to(tmp_path)),
            "test_outputs": str(test_path.relative_to(tmp_path)),
        },
    }
    run_path.write_text(yaml.safe_dump(run_payload, sort_keys=False), encoding="utf-8")

    summary = run_posthoc(config, repo_root=tmp_path)

    assert summary["recommended_softmax_base"] == "BalancedSoftmax"
    assert Path(summary["output_root"]).name.startswith("posthoc_twinkl_681_4_stage1_")

    report_body = Path(summary["report_path"]).read_text(encoding="utf-8")
    assert "Recommended stage-1 softmax leader: `BalancedSoftmax`." in report_body
    assert "twinkl-681.4 tuned softmax long-tail comparison" in report_body
