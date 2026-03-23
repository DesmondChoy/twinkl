"""Smoke test for the two-stage ordinal artifact path."""

from functools import partial

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.vif.class_balance import (
    class_counts_to_priors,
    compute_activation_class_counts,
    compute_polarity_class_counts,
)
from src.vif.critic_ordinal import (
    CriticMLPTwoStageBalancedSoftmax,
    OrdinalCriticBase,
    two_stage_balanced_softmax_loss_multi,
)
from src.vif.eval import (
    build_ordinal_selection_candidate,
    evaluate_with_uncertainty,
    export_ordinal_output_artifact,
)
from src.vif.training_traces import build_selection_trace_frame


class _MetadataDataset(TensorDataset):
    """Tensor dataset that exposes stable metadata for artifact export."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        super().__init__(x, y)
        self._metadata = [
            {
                "persona_id": f"persona_{idx // 2}",
                "t_index": idx,
                "date": f"2025-01-{idx + 1:02d}",
            }
            for idx in range(len(x))
        ]

    def get_all_metadata(self) -> list[dict]:
        return list(self._metadata)


def _make_targets() -> torch.Tensor:
    return torch.tensor(
        [
            [-1, 0, 1, 0, 1, -1, 0, 1, 0, -1],
            [0, 0, 1, 0, -1, 0, 0, 1, 0, 1],
            [1, -1, 0, 0, 0, 1, 0, 0, -1, 0],
            [0, 1, -1, 0, 1, 0, -1, 0, 1, 0],
            [-1, 0, 0, 1, 0, -1, 0, 1, 0, 1],
            [1, 0, 1, -1, 0, 1, 0, -1, 0, 0],
        ],
        dtype=torch.float32,
    )


def test_two_stage_smoke_writes_checkpoint_trace_and_output_artifacts(tmp_path):
    torch.manual_seed(7)
    x = torch.randn(6, 20, dtype=torch.float32)
    y = _make_targets()

    train_loader = DataLoader(TensorDataset(x, y), batch_size=3, shuffle=False)
    val_loader = DataLoader(_MetadataDataset(x[:4], y[:4]), batch_size=2, shuffle=False)
    test_loader = DataLoader(_MetadataDataset(x[2:], y[2:]), batch_size=2, shuffle=False)

    target_array = y.numpy().astype(np.int64)
    activation_priors = class_counts_to_priors(compute_activation_class_counts(target_array))
    polarity_priors = class_counts_to_priors(compute_polarity_class_counts(target_array))

    model = CriticMLPTwoStageBalancedSoftmax(input_dim=20, hidden_dim=16, dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    criterion = partial(
        two_stage_balanced_softmax_loss_multi,
        activation_priors=activation_priors,
        polarity_priors=polarity_priors,
    )

    model.train()
    train_loss = 0.0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()
        train_loss += float(loss.item())
    train_loss /= len(train_loader)

    val_results = evaluate_with_uncertainty(
        model,
        val_loader,
        n_mc_samples=2,
        include_ordinal_metrics=True,
        include_raw_outputs=True,
    )
    test_results = evaluate_with_uncertainty(
        model,
        test_loader,
        n_mc_samples=2,
        include_ordinal_metrics=True,
        include_raw_outputs=True,
    )

    candidate = build_ordinal_selection_candidate(
        epoch=0,
        val_loss=train_loss,
        eval_result=val_results,
    )
    selection_trace_path = tmp_path / "selection_trace.parquet"
    build_selection_trace_frame(
        [candidate],
        {"train_loss": [train_loss], "lr": [0.01]},
    ).write_parquet(selection_trace_path)

    checkpoint_path = tmp_path / "selected_checkpoint.pt"
    torch.save(
        {
            "epoch": 0,
            "selection_candidate": candidate,
            "model_state_dict": model.state_dict(),
            "model_config": model.get_config(),
        },
        checkpoint_path,
    )

    val_output_path = tmp_path / "val_outputs.parquet"
    test_output_path = tmp_path / "test_outputs.parquet"
    export_ordinal_output_artifact(
        val_results,
        val_loader,
        val_output_path,
        split="val",
        model_name="TwoStageBalancedSoftmax",
    )
    export_ordinal_output_artifact(
        test_results,
        test_loader,
        test_output_path,
        split="test",
        model_name="TwoStageBalancedSoftmax",
    )

    restored_model = OrdinalCriticBase.from_config(model.get_config())
    restored_model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])

    assert checkpoint_path.is_file()
    assert selection_trace_path.is_file()
    assert val_output_path.is_file()
    assert test_output_path.is_file()
    assert isinstance(restored_model, CriticMLPTwoStageBalancedSoftmax)

    val_df = pl.read_parquet(val_output_path)
    test_df = pl.read_parquet(test_output_path)
    assert val_df.height == 40
    assert test_df.height == 40
    assert set(val_df["split"].unique().to_list()) == {"val"}
    assert set(test_df["split"].unique().to_list()) == {"test"}
