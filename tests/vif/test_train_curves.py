"""Tests for train/val gap summaries and curve artifacts in src.vif.train."""

import json

import src.vif.train as train_module


def test_save_training_curve_artifacts_writes_png_and_json(tmp_path):
    history = {
        "train_loss": [0.9, 0.7],
        "val_loss": [1.1, 0.8],
        "train_val_gap": [0.2, 0.1],
        "learning_rate": [0.01, 0.005],
    }

    artifacts = train_module.save_training_curve_artifacts(
        history,
        best_epoch=1,
        output_dir=tmp_path,
    )

    assert artifacts == {
        "plot_path": str(tmp_path / "training_curves.png"),
        "history_path": str(tmp_path / "training_curves.json"),
    }

    with open(tmp_path / "training_curves.json") as f:
        curve_history = json.load(f)

    assert curve_history == {
        "epochs": [1, 2],
        "train_loss": [0.9, 0.7],
        "val_loss": [1.1, 0.8],
        "train_val_gap": [0.2, 0.1],
        "learning_rate": [0.01, 0.005],
        "best_epoch": 2,
    }
    assert (tmp_path / "training_curves.png").is_file()


def test_summarize_training_dynamics_uses_recorded_gap_history():
    history = {
        "train_loss": [1.0, 1.0, 1.0],
        "val_loss": [0.9, 0.9, 0.9],
        "train_val_gap": [0.25, -0.4, 0.1],
        "learning_rate": [0.01, 0.005, 0.0025],
        "grad_norm_mean": [1.0, 2.0, 3.0],
        "grad_norm_max": [1.5, 2.5, 3.5],
        "grad_batches_tracked": [2, 2, 2],
        "grad_clipped_fraction": [0.0, 0.5, 1.0],
    }

    training_dynamics = train_module._summarize_training_dynamics(
        history,
        gradient_config={
            "grad_clip": 1.0,
            "gradient_logging_enabled": True,
            "gradient_log_every": 1,
        },
        best_epoch=0,
        curve_artifacts={
            "plot_path": "/tmp/training_curves.png",
            "history_path": "/tmp/training_curves.json",
        },
    )

    assert training_dynamics["gap_summary"] == {
        "best_epoch": 1,
        "total_epochs": 3,
        "gap_at_best": 0.25,
        "gap_at_final": 0.1,
        "max_gap": 0.25,
        "min_gap": -0.4,
    }
    assert training_dynamics["gradient_config"] == {
        "grad_clip": 1.0,
        "gradient_logging_enabled": True,
        "sample_every_batches": 1,
    }
    assert training_dynamics["gradient_summary"] == {
        "epochs_with_gradient_samples": 3,
        "total_gradient_batches_tracked": 6,
        "grad_norm_mean_over_epochs": 2.0,
        "grad_norm_max_over_epochs": 3.5,
        "grad_clipped_fraction_mean": 0.5,
        "grad_clipped_fraction_max": 1.0,
    }
    assert training_dynamics["curve_artifacts"] == {
        "plot_path": "/tmp/training_curves.png",
        "history_path": "/tmp/training_curves.json",
    }
