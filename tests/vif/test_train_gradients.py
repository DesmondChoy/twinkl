"""Tests for gradient clipping and telemetry in src.vif.train."""

import copy
import sys

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import src.vif.train as train_module
from src.vif.critic import CriticMLP


def _loader(num_samples: int, *, batch_size: int = 1) -> DataLoader:
    torch.manual_seed(7)
    x = torch.randn((num_samples, 4), dtype=torch.float32)
    y = torch.randn((num_samples, 10), dtype=torch.float32)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def _model_and_optimizer() -> tuple[CriticMLP, torch.optim.Optimizer]:
    model = CriticMLP(input_dim=4, hidden_dim=8, output_dim=10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optimizer


def _minimal_config():
    return {
        "encoder": {"model_name": "mock"},
        "state_encoder": {"window_size": 1},
        "model": {"hidden_dim": 8, "dropout": 0.1, "output_dim": 10},
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "scheduler": {"factor": 0.5, "patience": 2, "min_lr": 1e-5},
            "early_stopping": {"patience": 3, "min_delta": 0.0},
            "lr_finder": {"start_lr": 1e-7, "end_lr": 1.0, "num_iter": 16, "output_path": None},
        },
        "data": {
            "seed": 42,
            "labels_path": "unused",
            "wrangled_dir": "unused",
            "train_ratio": 0.7,
            "val_ratio": 0.15,
        },
        "mc_dropout": {"n_samples": 2},
        "output": {"checkpoint_dir": "unused", "log_dir": "unused"},
    }


def test_train_epoch_clips_gradients_before_optimizer_step(monkeypatch):
    model, optimizer = _model_and_optimizer()
    loader = _loader(1)
    criterion = torch.nn.MSELoss()
    call_order = []

    original_step = optimizer.step

    def _wrapped_step(*args, **kwargs):
        call_order.append("step")
        return original_step(*args, **kwargs)

    def _fake_clip(parameters, max_norm):
        params = list(parameters)
        assert max_norm == 1.0
        assert any(parameter.grad is not None for parameter in params)
        call_order.append("clip")
        return torch.tensor(3.5)

    monkeypatch.setattr(optimizer, "step", _wrapped_step)
    monkeypatch.setattr(train_module.torch.nn.utils, "clip_grad_norm_", _fake_clip)

    train_loss, gradient_metrics = train_module.train_epoch(
        model,
        loader,
        optimizer,
        criterion,
        "cpu",
        grad_clip=1.0,
        gradient_logging_enabled=True,
        gradient_log_every=1,
    )

    assert train_loss > 0
    assert call_order == ["clip", "step"]
    assert gradient_metrics["grad_norm_mean"] == pytest.approx(3.5)
    assert gradient_metrics["grad_norm_max"] == pytest.approx(3.5)
    assert gradient_metrics["grad_batches_tracked"] == 1
    assert gradient_metrics["grad_clipped_fraction"] == pytest.approx(1.0)


def test_train_epoch_skips_gradient_norm_work_when_logging_disabled(monkeypatch):
    model, optimizer = _model_and_optimizer()
    loader = _loader(2)
    criterion = torch.nn.MSELoss()

    def _unexpected_norm_call(_parameters):
        raise AssertionError("gradient norm telemetry should be disabled")

    monkeypatch.setattr(train_module, "_compute_total_grad_norm", _unexpected_norm_call)

    train_loss, gradient_metrics = train_module.train_epoch(
        model,
        loader,
        optimizer,
        criterion,
        "cpu",
        grad_clip=None,
        gradient_logging_enabled=False,
        gradient_log_every=1,
    )

    assert train_loss > 0
    assert gradient_metrics == {
        "grad_norm_mean": None,
        "grad_norm_max": None,
        "grad_batches_tracked": 0,
        "grad_clipped_fraction": None,
    }


def test_train_epoch_downsamples_gradient_logging(monkeypatch):
    model, optimizer = _model_and_optimizer()
    loader = _loader(3)
    criterion = torch.nn.MSELoss()
    observed_norms = []
    sampled_norms = iter([1.0, 2.0])

    def _fake_total_grad_norm(_parameters):
        value = next(sampled_norms)
        observed_norms.append(value)
        return value

    monkeypatch.setattr(train_module, "_compute_total_grad_norm", _fake_total_grad_norm)

    train_loss, gradient_metrics = train_module.train_epoch(
        model,
        loader,
        optimizer,
        criterion,
        "cpu",
        grad_clip=None,
        gradient_logging_enabled=True,
        gradient_log_every=2,
    )

    assert train_loss > 0
    assert observed_norms == [1.0, 2.0]
    assert gradient_metrics["grad_norm_mean"] == pytest.approx(1.5)
    assert gradient_metrics["grad_norm_max"] == pytest.approx(2.0)
    assert gradient_metrics["grad_batches_tracked"] == 2
    assert gradient_metrics["grad_clipped_fraction"] is None


def test_main_applies_gradient_cli_overrides(monkeypatch, capsys):
    config = _minimal_config()
    observed = {}

    monkeypatch.setattr(train_module, "load_config", lambda _path: copy.deepcopy(config))

    def _fake_train(received_config, verbose):
        observed["config"] = copy.deepcopy(received_config)
        observed["verbose"] = verbose
        return {
            "test_results": {
                "mse_mean": 0.11,
                "spearman_mean": 0.22,
                "accuracy_mean": 0.33,
            },
            "learning_rate_configured": 0.001,
            "learning_rate_applied": 0.001,
            "lr_finder": {"artifacts": {"plot_path": "unused"}},
        }

    monkeypatch.setattr(train_module, "train", _fake_train)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--grad-clip",
            "0",
            "--no-log-gradients",
            "--grad-log-every",
            "4",
            "--quiet",
        ],
    )

    train_module.main()
    captured = capsys.readouterr()

    training_config = observed["config"]["training"]
    assert observed["verbose"] is False
    assert training_config["grad_clip"] == 0.0
    assert training_config["gradient_logging"]["enabled"] is False
    assert training_config["gradient_logging"]["sample_every_batches"] == 4
    assert "Final test MSE" in captured.out
