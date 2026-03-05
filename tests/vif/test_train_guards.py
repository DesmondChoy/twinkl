"""Regression tests for empty-loader guardrails in VIF training paths."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import src.vif.train as train_module
import src.vif.train_bnn as train_bnn_module
from src.vif.critic import CriticMLP


def _empty_loader(input_dim: int = 4, output_dim: int = 10) -> DataLoader:
    x = torch.empty((0, input_dim), dtype=torch.float32)
    y = torch.empty((0, output_dim), dtype=torch.float32)
    return DataLoader(TensorDataset(x, y), batch_size=2)


class _DummyStateEncoder:
    def __init__(self, *_args, **_kwargs):
        self.state_dim = 4


class _DummyLoader:
    def __init__(self, size: int):
        self.dataset = list(range(size))


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


def test_train_epoch_empty_loader_raises():
    model = CriticMLP(input_dim=4, hidden_dim=8, output_dim=10)
    loader = _empty_loader()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    with pytest.raises(ValueError, match="produced zero batches"):
        train_module.train_epoch(model, loader, optimizer, criterion, "cpu")


def test_validate_empty_loader_raises():
    model = CriticMLP(input_dim=4, hidden_dim=8, output_dim=10)
    loader = _empty_loader()
    criterion = torch.nn.MSELoss()

    with pytest.raises(ValueError, match="produced zero batches"):
        train_module.validate(model, loader, criterion, "cpu")


def test_train_epoch_bnn_empty_loader_raises():
    model = torch.nn.Linear(4, 10)
    loader = _empty_loader()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    with pytest.raises(ValueError, match="produced zero batches"):
        train_bnn_module.train_epoch_bnn(model, loader, optimizer, criterion, "cpu")


def test_validate_bnn_empty_loader_raises():
    model = torch.nn.Linear(4, 10)
    loader = _empty_loader()
    criterion = torch.nn.MSELoss()

    with pytest.raises(ValueError, match="produced zero batches"):
        train_bnn_module.validate(model, loader, criterion, "cpu")


def test_train_main_raises_on_empty_split(monkeypatch):
    config = _minimal_config()

    monkeypatch.setattr(train_module, "create_encoder", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(train_module, "StateEncoder", _DummyStateEncoder)
    monkeypatch.setattr(
        train_module,
        "create_dataloaders",
        lambda *_args, **_kwargs: (_DummyLoader(3), _DummyLoader(0), _DummyLoader(2)),
    )

    with pytest.raises(ValueError, match="Empty dataset split\\(s\\): val"):
        train_module.train(config, verbose=False)


def test_train_bnn_main_raises_on_empty_split(monkeypatch):
    config = _minimal_config()

    monkeypatch.setattr(train_bnn_module, "create_encoder", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(train_bnn_module, "StateEncoder", _DummyStateEncoder)
    monkeypatch.setattr(
        train_bnn_module,
        "create_dataloaders",
        lambda *_args, **_kwargs: (_DummyLoader(3), _DummyLoader(0), _DummyLoader(2)),
    )

    with pytest.raises(ValueError, match="Empty dataset split\\(s\\): val"):
        train_bnn_module.train(config, verbose=False)
