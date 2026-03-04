"""Integration-style tests for default LR finder usage in train()."""

import json

import src.vif.train as train_module


class _DummyStateEncoder:
    def __init__(self, *_args, **_kwargs):
        self.state_dim = 4


class _DummyLoader:
    dataset = [0, 1, 2]


def _minimal_config(tmp_path):
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
        "output": {"checkpoint_dir": str(tmp_path), "log_dir": str(tmp_path)},
    }


def test_train_uses_lr_finder_selected_lr_and_logs_metadata(tmp_path, monkeypatch):
    config = _minimal_config(tmp_path)
    observed_lrs = []

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
        lambda **_kwargs: {
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
        },
    )

    def _fake_train_epoch(model, dataloader, optimizer, criterion, device):
        observed_lrs.append(optimizer.param_groups[0]["lr"])
        return 0.5

    monkeypatch.setattr(train_module, "train_epoch", _fake_train_epoch)
    monkeypatch.setattr(train_module, "validate", lambda *_args, **_kwargs: 0.4)
    monkeypatch.setattr(train_module, "save_checkpoint", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_module, "load_checkpoint", lambda *_args, **_kwargs: ("model", {}))
    monkeypatch.setattr(
        train_module,
        "evaluate_with_uncertainty",
        lambda *_args, **_kwargs: {
            "mse_mean": 0.11,
            "spearman_mean": 0.22,
            "accuracy_mean": 0.33,
            "mse_per_dim": {"achievement": 0.1},
            "spearman_per_dim": {"achievement": 0.2},
            "accuracy_per_dim": {"achievement": 0.3},
        },
    )

    results = train_module.train(config, verbose=False)

    assert observed_lrs == [0.012]
    assert results["learning_rate_configured"] == 0.001
    assert results["learning_rate_applied"] == 0.012

    log_path = tmp_path / "training_log.json"
    assert log_path.exists()

    with open(log_path) as f:
        log_data = json.load(f)

    assert log_data["lr_finder"]["lr_selected"] == 0.012
    assert log_data["lr_finder"]["suggestions"]["lr_valley"] == 0.012
    assert log_data["config"]["training"]["learning_rate_configured"] == 0.001
    assert log_data["config"]["training"]["learning_rate"] == 0.012
