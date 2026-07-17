"""Security regression tests for VIF Critic checkpoint loading."""

from pathlib import Path

import pytest
import torch

from scripts.experiments.evaluate_twinkl_748_hedonism_hard_set import (
    load_checkpoint_model,
)
from src.vif.extract_embeddings import _load_model
from src.vif.runtime import load_runtime_bundle
from src.vif.train import load_checkpoint as load_mlp_checkpoint
from src.vif.train_bnn import load_checkpoint as load_bnn_checkpoint


class _LoadIntercepted(Exception):
    pass


def _load_runtime(path: Path) -> None:
    load_runtime_bundle(path, config_path=None, device="cpu")


def _load_embeddings(path: Path) -> None:
    _load_model(path)


def _load_mlp(path: Path) -> None:
    load_mlp_checkpoint(path, device="cpu")


def _load_bnn(path: Path) -> None:
    load_bnn_checkpoint(path, device="cpu")


def _load_hedonism_evaluator(path: Path) -> None:
    load_checkpoint_model({"checkpoint_path": path}, device="cpu")


@pytest.mark.parametrize(
    "loader",
    [
        _load_runtime,
        _load_embeddings,
        _load_mlp,
        _load_bnn,
        _load_hedonism_evaluator,
    ],
)
def test_checkpoint_loaders_restrict_deserialization(monkeypatch, loader):
    calls = []

    def intercept_load(*_args, **kwargs):
        calls.append(kwargs)
        assert kwargs["weights_only"] is True
        raise _LoadIntercepted

    monkeypatch.setattr(torch, "load", intercept_load)

    with pytest.raises(_LoadIntercepted):
        loader(Path("untrusted.pt"))

    assert len(calls) == 1
