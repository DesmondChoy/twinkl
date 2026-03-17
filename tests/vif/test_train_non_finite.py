"""Tests for non-finite loss handling in src.vif.train."""

import json

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import src.vif.train as train_module
from src.vif.critic import CriticMLP


class _DummyStateEncoder:
    def __init__(self, *_args, **_kwargs):
        self.state_dim = 4


class _DummyLoader:
    dataset = [0, 1, 2]


def _loader(num_samples: int = 1) -> DataLoader:
    torch.manual_seed(11)
    x = torch.randn((num_samples, 4), dtype=torch.float32)
    y = torch.randn((num_samples, 10), dtype=torch.float32)
    return DataLoader(TensorDataset(x, y), batch_size=1)


def _model_and_optimizer() -> tuple[CriticMLP, torch.optim.Optimizer]:
    model = CriticMLP(input_dim=4, hidden_dim=8, output_dim=10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optimizer


def _minimal_config(tmp_path):
    return {
        "encoder": {"model_name": "mock"},
        "state_encoder": {"window_size": 1},
        "model": {"hidden_dim": 8, "dropout": 0.1, "output_dim": 10},
        "training": {
            "epochs": 2,
            "batch_size": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "scheduler": {"factor": 0.5, "patience": 2, "min_lr": 1e-5},
            "early_stopping": {"patience": 3, "min_delta": 0.0},
            "lr_finder": {
                "start_lr": 1e-7,
                "end_lr": 1.0,
                "num_iter": 16,
                "output_path": None,
            },
        },
        "data": {
            "seed": 42,
            "labels_path": "unused",
            "wrangled_dir": "unused",
            "train_ratio": 0.7,
            "val_ratio": 0.15,
        },
        "mc_dropout": {"n_samples": 2},
        "output": {"checkpoint_dir": str(tmp_path), "log_dir": str(tmp_path)},
    }


def _lr_finder_result(tmp_path) -> dict:
    return {
        "enabled": True,
        "params": {"start_lr": 1e-7, "end_lr": 1.0, "num_iter": 16},
        "suggestions": {"lr_steep": 0.01, "lr_valley": 0.012},
        "lr_selected": 0.012,
        "configured_learning_rate": 0.001,
        "fallback_reason": None,
        "history_points": 16,
        "artifacts": {
            "plot_path": str(tmp_path / "lr_find_loss_vs_lr.png"),
            "history_path": str(tmp_path / "lr_find_history.json"),
        },
    }


def _gradient_metrics() -> dict:
    return {
        "grad_norm_mean": 1.2,
        "grad_norm_max": 2.4,
        "grad_batches_tracked": 2,
        "grad_clipped_fraction": 0.5,
    }


def test_train_epoch_raises_before_optimizer_step_on_non_finite_loss(monkeypatch):
    model, optimizer = _model_and_optimizer()
    loader = _loader()

    def _fail_if_step_called(*_args, **_kwargs):
        raise AssertionError("optimizer.step should not run after a non-finite loss")

    def _nan_loss(*_args, **_kwargs):
        return torch.tensor(float("nan"), dtype=torch.float32, requires_grad=True)

    monkeypatch.setattr(optimizer, "step", _fail_if_step_called)

    with pytest.raises(train_module.NonFiniteLossError) as exc_info:
        train_module.train_epoch(
            model,
            loader,
            optimizer,
            _nan_loss,
            "cpu",
            epoch=3,
        )

    error = exc_info.value
    assert error.phase == "train"
    assert error.epoch == 3
    assert error.batch_index == 1
    assert error.loss_name == "mse_loss"
    assert error.loss_value == "nan"
    assert "epoch 3, batch 1" in str(error)


def test_validate_raises_immediately_on_non_finite_loss():
    model, _ = _model_and_optimizer()
    loader = _loader()

    def _inf_loss(*_args, **_kwargs):
        return torch.tensor(float("inf"), dtype=torch.float32)

    with pytest.raises(train_module.NonFiniteLossError) as exc_info:
        train_module.validate(
            model,
            loader,
            _inf_loss,
            "cpu",
            epoch=2,
        )

    error = exc_info.value
    assert error.phase == "val"
    assert error.epoch == 2
    assert error.batch_index == 1
    assert error.loss_name == "mse_loss"
    assert error.loss_value == "inf"


def test_save_checkpoint_rejects_non_finite_val_loss_without_overwriting_best(tmp_path):
    model, optimizer = _model_and_optimizer()
    checkpoint_path = tmp_path / "best_model.pt"

    train_module.save_checkpoint(model, optimizer, 0, 0.5, {}, tmp_path)
    before_bytes = checkpoint_path.read_bytes()

    with pytest.raises(ValueError, match="non-finite val_loss=nan"):
        train_module.save_checkpoint(model, optimizer, 1, float("nan"), {}, tmp_path)

    assert checkpoint_path.read_bytes() == before_bytes


def test_save_checkpoint_rejects_non_finite_tensor_without_overwriting_best(tmp_path):
    model, optimizer = _model_and_optimizer()
    checkpoint_path = tmp_path / "best_model.pt"

    train_module.save_checkpoint(model, optimizer, 0, 0.5, {}, tmp_path)
    before_bytes = checkpoint_path.read_bytes()

    with torch.no_grad():
        next(model.parameters()).fill_(float("inf"))

    with pytest.raises(ValueError, match="contains non-finite values"):
        train_module.save_checkpoint(model, optimizer, 1, 0.4, {}, tmp_path)

    assert checkpoint_path.read_bytes() == before_bytes


def test_train_writes_partial_failure_log_and_preserves_best_checkpoint(
    tmp_path,
    monkeypatch,
):
    config = _minimal_config(tmp_path)
    observed_epochs = []
    observed_val_epochs = []
    outcomes = iter(
        [
            (0.5, _gradient_metrics()),
            train_module.NonFiniteLossError(
                phase="train",
                epoch=2,
                batch_index=3,
                loss_name="mse_loss",
                loss_value="nan",
            ),
        ]
    )

    monkeypatch.setattr(train_module, "create_encoder", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(train_module, "StateEncoder", _DummyStateEncoder)
    monkeypatch.setattr(
        train_module,
        "create_dataloaders",
        lambda *_args, **_kwargs: (_DummyLoader(), _DummyLoader(), _DummyLoader()),
    )
    monkeypatch.setattr(
        train_module,
        "run_lr_finder",
        lambda **_kwargs: _lr_finder_result(tmp_path),
    )

    def _fake_train_epoch(*_args, epoch=1, **_kwargs):
        observed_epochs.append(epoch)
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def _fake_validate(*_args, epoch=1, **_kwargs):
        observed_val_epochs.append(epoch)
        return 0.4

    def _unexpected(*_args, **_kwargs):
        raise AssertionError("final checkpoint loading and evaluation should be skipped")

    monkeypatch.setattr(train_module, "train_epoch", _fake_train_epoch)
    monkeypatch.setattr(train_module, "validate", _fake_validate)
    monkeypatch.setattr(train_module, "load_checkpoint", _unexpected)
    monkeypatch.setattr(train_module, "evaluate_with_uncertainty", _unexpected)

    with pytest.raises(train_module.NonFiniteLossError) as exc_info:
        train_module.train(config, verbose=False)

    error = exc_info.value
    assert error.phase == "train"
    assert observed_epochs == [1, 2]
    assert observed_val_epochs == [1]

    checkpoint_path = tmp_path / "best_model.pt"
    assert checkpoint_path.is_file()

    with open(tmp_path / "training_log.json") as f:
        log_data = json.load(f)

    assert log_data["epochs_completed"] == 1
    assert log_data["best_val_loss"] == 0.4
    assert log_data["history"]["train_loss"] == [0.5]
    assert log_data["history"]["val_loss"] == [0.4]
    assert log_data["history"]["train_val_gap"] == [-0.09999999999999998]
    assert log_data["test_metrics"] is None
    assert log_data["training_dynamics"]["gap_summary"]["total_epochs"] == 1
    assert log_data["training_dynamics"]["termination"] == {
        "status": "failed_non_finite_loss",
        "phase": "train",
        "epoch": 2,
        "batch_index": 3,
        "loss_name": "mse_loss",
        "loss_value": "nan",
        "best_checkpoint_path": str(checkpoint_path),
        "best_checkpoint_exists": True,
    }

    curve_artifacts = log_data["training_dynamics"]["curve_artifacts"]
    assert (tmp_path / "training_curves.png").is_file()
    assert curve_artifacts["plot_path"] == str(tmp_path / "training_curves.png")
    assert curve_artifacts["history_path"] == str(tmp_path / "training_curves.json")

    with open(tmp_path / "training_curves.json") as f:
        curve_history = json.load(f)

    assert curve_history == {
        "epochs": [1],
        "train_loss": [0.5],
        "val_loss": [0.4],
        "train_val_gap": [-0.09999999999999998],
        "learning_rate": [0.012],
        "best_epoch": 1,
    }


def test_train_logs_failure_before_best_checkpoint_exists(tmp_path, monkeypatch):
    config = _minimal_config(tmp_path)

    monkeypatch.setattr(train_module, "create_encoder", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(train_module, "StateEncoder", _DummyStateEncoder)
    monkeypatch.setattr(
        train_module,
        "create_dataloaders",
        lambda *_args, **_kwargs: (_DummyLoader(), _DummyLoader(), _DummyLoader()),
    )
    monkeypatch.setattr(
        train_module,
        "run_lr_finder",
        lambda **_kwargs: _lr_finder_result(tmp_path),
    )
    monkeypatch.setattr(
        train_module,
        "train_epoch",
        lambda *_args, **_kwargs: (0.5, _gradient_metrics()),
    )

    def _failing_validate(*_args, **_kwargs):
        raise train_module.NonFiniteLossError(
            phase="val",
            epoch=1,
            batch_index=1,
            loss_name="mse_loss",
            loss_value="-inf",
        )

    def _unexpected(*_args, **_kwargs):
        raise AssertionError("final checkpoint loading and evaluation should be skipped")

    monkeypatch.setattr(train_module, "validate", _failing_validate)
    monkeypatch.setattr(train_module, "load_checkpoint", _unexpected)
    monkeypatch.setattr(train_module, "evaluate_with_uncertainty", _unexpected)

    with pytest.raises(train_module.NonFiniteLossError) as exc_info:
        train_module.train(config, verbose=False)

    error = exc_info.value
    assert error.phase == "val"
    assert not (tmp_path / "best_model.pt").exists()

    with open(tmp_path / "training_log.json") as f:
        log_data = json.load(f)

    assert log_data["epochs_completed"] == 0
    assert log_data["best_val_loss"] is None
    assert log_data["history"]["train_loss"] == []
    assert log_data["history"]["val_loss"] == []
    assert log_data["history"]["train_val_gap"] == []
    assert log_data["test_metrics"] is None
    assert log_data["training_dynamics"]["termination"] == {
        "status": "failed_non_finite_loss",
        "phase": "val",
        "epoch": 1,
        "batch_index": 1,
        "loss_name": "mse_loss",
        "loss_value": "-inf",
        "best_checkpoint_path": str(tmp_path / "best_model.pt"),
        "best_checkpoint_exists": False,
    }

    with open(tmp_path / "training_curves.json") as f:
        curve_history = json.load(f)

    assert curve_history == {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "train_val_gap": [],
        "learning_rate": [],
        "best_epoch": None,
    }
