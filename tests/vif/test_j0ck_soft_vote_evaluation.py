import numpy as np
import pytest

from scripts.experiments.evaluate_j0ck_soft_vote_comparison import (
    distribution_scores,
    summarize_families,
)


def test_distribution_scores_are_zero_for_exact_targets():
    targets = np.array([[1.0, 0.0, 0.0], [0.2, 0.6, 0.2]])
    entropy = np.array([0.0, 1.3709505944546687])

    scores = distribution_scores(targets, targets, entropy)

    expected_nll = -0.5 * (0.2 * np.log(0.2) + 0.6 * np.log(0.6) + 0.2 * np.log(0.2))
    assert scores["soft_nll"] == pytest.approx(expected_nll)
    assert scores["multiclass_brier"] == pytest.approx(0.0)
    assert scores["entropy_mae_bits"] == pytest.approx(0.0, abs=1e-9)
    assert scores["entropy_correlation"] == pytest.approx(1.0)


def test_distribution_scores_reject_mismatched_shapes():
    with pytest.raises(ValueError, match="share shape"):
        distribution_scores(
            np.array([[1.0, 0.0, 0.0]]),
            np.array([[1.0, 0.0]]),
            np.array([0.0]),
        )


def test_distribution_scores_omit_undefined_entropy_correlation():
    targets = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    scores = distribution_scores(targets, targets, np.array([0.0, 0.0]))

    assert scores["entropy_correlation"] is None


def test_family_summary_rejects_incomplete_pairs():
    run = {
        "run_id": "run_test",
        "training_target_mode": "hard",
        "model_seed": 11,
        "hard_metrics": {"qwk_mean": 0.3},
        "distribution_metrics": {"n": 1, "soft_nll": 0.5},
        "loss_space_distribution_metrics": {"n": 1, "soft_nll": 0.4},
    }

    with pytest.raises(ValueError, match="one hard and one vote_distribution"):
        summarize_families([run])
