import numpy as np

from src.vif.frontier_uncertainty import (
    bca_confidence_interval,
    build_stratified_target_permutation_indices,
    sample_persona_cluster_indices,
)


def test_bca_confidence_interval_falls_back_to_percentile_for_constant_jackknife():
    observed = 0.2
    bootstrap_values = np.array([0.1, 0.15, 0.19, 0.21, 0.22, 0.24, 0.3, 0.32, 0.35, 0.4])
    jackknife_values = np.full(8, observed)

    interval = bca_confidence_interval(
        observed,
        bootstrap_values,
        jackknife_values,
        confidence_level=0.95,
    )

    assert interval["method"] == "percentile"
    assert interval["ci_lower"] < interval["estimate"] < interval["ci_upper"]


def test_bca_confidence_interval_returns_bca_when_jackknife_varies():
    observed = 0.31
    bootstrap_values = np.array(
        [0.17, 0.19, 0.23, 0.25, 0.27, 0.29, 0.33, 0.35, 0.38, 0.41, 0.44, 0.47]
    )
    jackknife_values = np.array([0.26, 0.28, 0.3, 0.32, 0.33, 0.34, 0.36, 0.37])

    interval = bca_confidence_interval(
        observed,
        bootstrap_values,
        jackknife_values,
        confidence_level=0.95,
    )

    assert interval["method"] == "bca"
    assert interval["ci_lower"] < interval["estimate"] < interval["ci_upper"]


def test_sample_persona_cluster_indices_preserves_cluster_boundaries():
    persona_order = ["a", "b", "c"]
    persona_to_indices = {
        "a": np.array([0, 1]),
        "b": np.array([2]),
        "c": np.array([3, 4, 5]),
    }
    rng = np.random.default_rng(7)

    sampled = sample_persona_cluster_indices(persona_order, persona_to_indices, rng)

    valid_clusters = {tuple(indices) for indices in persona_to_indices.values()}
    cursor = 0
    sampled_clusters = []
    while cursor < len(sampled):
        matched = False
        for indices in valid_clusters:
            span = len(indices)
            if tuple(sampled[cursor : cursor + span]) == indices:
                sampled_clusters.append(indices)
                cursor += span
                matched = True
                break
        assert matched

    assert len(sampled_clusters) == len(persona_order)


def test_build_stratified_target_permutation_indices_only_swaps_equal_length_clusters():
    persona_order = ["a", "b", "c", "d", "e"]
    persona_to_indices = {
        "a": np.array([0, 1]),
        "b": np.array([2, 3]),
        "c": np.array([4]),
        "d": np.array([5]),
        "e": np.array([6, 7, 8]),
    }
    rng = np.random.default_rng(3)

    permuted = build_stratified_target_permutation_indices(persona_order, persona_to_indices, rng)

    expected_lengths = [len(persona_to_indices[persona_id]) for persona_id in persona_order]
    cursor = 0
    reconstructed_lengths = []
    for length in expected_lengths:
        cluster = tuple(permuted[cursor : cursor + length])
        reconstructed_lengths.append(len(cluster))
        cursor += length
        assert cluster in {
            tuple(indices)
            for indices in persona_to_indices.values()
            if len(indices) == length
        }

    assert reconstructed_lengths == expected_lengths
