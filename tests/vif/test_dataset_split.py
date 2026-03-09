"""Tests for dataset splitting with configurable train/val ratios."""

from unittest.mock import call, patch

import numpy as np
import polars as pl
import pytest

from src.vif.dataset import split_by_persona, create_dataloaders


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_fake_data(n_personas: int = 20, entries_per_persona: int = 5):
    """Create minimal labels and entries DataFrames for testing splits."""
    label_rows = []
    entry_rows = []
    for p in range(n_personas):
        pid = f"persona_{p:03d}"
        for t in range(entries_per_persona):
            label_rows.append({
                "persona_id": pid,
                "t_index": t,
                "date": f"2025-01-{t + 1:02d}",
                "alignment_vector": [0.0] * 10,
            })
            entry_rows.append({
                "persona_id": pid,
                "t_index": t,
                "date": f"2025-01-{t + 1:02d}",
                "initial_entry": f"Entry {t} for {pid}",
                "nudge_text": None,
                "response_text": None,
                "core_values": "achievement,benevolence",
            })

    labels_df = pl.DataFrame(label_rows)
    entries_df = pl.DataFrame(entry_rows)
    return labels_df, entries_df


def _make_skewed_data(n_personas: int = 60, entries_per_persona: int = 2):
    """Create data with sparse, sign-sensitive value patterns for stratification tests."""
    label_rows = []
    entry_rows = []

    for p in range(n_personas):
        pid = f"persona_{p:03d}"
        group = p // 12

        alignment = [0] * 10
        if group == 0:
            alignment[4] = 1      # power +
        elif group == 1:
            alignment[4] = -1     # power -
        elif group == 2:
            alignment[5] = 1      # security +
        elif group == 3:
            alignment[5] = -1     # security -
        else:
            alignment[9] = 1      # universalism +

        for t in range(entries_per_persona):
            label_rows.append({
                "persona_id": pid,
                "t_index": t,
                "date": f"2025-02-{t + 1:02d}",
                "alignment_vector": alignment,
            })
            entry_rows.append({
                "persona_id": pid,
                "t_index": t,
                "date": f"2025-02-{t + 1:02d}",
                "initial_entry": f"Skewed entry {t} for {pid}",
                "nudge_text": None,
                "response_text": None,
                "core_values": "achievement,benevolence",
            })

    return pl.DataFrame(label_rows), pl.DataFrame(entry_rows)


def _persona_feature_matrix(labels_df: pl.DataFrame, features: list[tuple[int, int]]) -> tuple[list[str], np.ndarray]:
    """Build persona-level binary matrix for selected (dimension, score) features."""
    persona_ids = sorted(labels_df.select("persona_id").unique().to_series().to_list())
    persona_to_idx = {pid: idx for idx, pid in enumerate(persona_ids)}
    matrix = np.zeros((len(persona_ids), len(features)), dtype=np.int8)

    for row in labels_df.select(["persona_id", "alignment_vector"]).iter_rows(named=True):
        p_idx = persona_to_idx[row["persona_id"]]
        vec = row["alignment_vector"]
        for f_idx, (dim_idx, score) in enumerate(features):
            if vec[dim_idx] == score:
                matrix[p_idx, f_idx] = 1

    return persona_ids, matrix


def _random_split_personas(
    labels_df: pl.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[set[str], set[str], set[str]]:
    """Replicate legacy one-shot random persona split behavior."""
    persona_ids = sorted(labels_df.select("persona_id").unique().to_series().to_list())

    rng = np.random.default_rng(seed)
    rng.shuffle(persona_ids)

    train_end = int(len(persona_ids) * train_ratio)
    val_end = int(len(persona_ids) * (train_ratio + val_ratio))

    return (
        set(persona_ids[:train_end]),
        set(persona_ids[train_end:val_end]),
        set(persona_ids[val_end:]),
    )


def _max_prevalence_gap(
    persona_ids: list[str],
    feature_matrix: np.ndarray,
    split_ids: set[str],
) -> float:
    """Compute max absolute prevalence gap for one split vs full dataset."""
    all_prevalence = feature_matrix.mean(axis=0)
    if not split_ids:
        return 0.0

    mask = np.array([pid in split_ids for pid in persona_ids], dtype=bool)
    split_prevalence = feature_matrix[mask].mean(axis=0)
    return float(np.max(np.abs(split_prevalence - all_prevalence)))


# ── split_by_persona tests ───────────────────────────────────────────────────


class TestSplitByPersona:
    """Tests for split_by_persona with various ratios."""

    def test_default_ratios(self):
        """Default 70/15/15 split produces expected persona counts."""
        labels_df, entries_df = _make_fake_data(n_personas=20)
        train_df, val_df, test_df = split_by_persona(labels_df, entries_df)

        # 20 personas: 14 train, 3 val, 3 test
        train_personas = train_df.select("persona_id").unique().height
        val_personas = val_df.select("persona_id").unique().height
        test_personas = test_df.select("persona_id").unique().height

        assert train_personas == 14
        assert val_personas == 3
        assert test_personas == 3

    def test_custom_ratios(self):
        """Non-default ratios (60/20/20) change split sizes."""
        labels_df, entries_df = _make_fake_data(n_personas=20)
        train_df, val_df, test_df = split_by_persona(
            labels_df, entries_df, train_ratio=0.60, val_ratio=0.20
        )

        # 20 personas: 12 train, 4 val, 4 test
        train_personas = train_df.select("persona_id").unique().height
        val_personas = val_df.select("persona_id").unique().height
        test_personas = test_df.select("persona_id").unique().height

        assert train_personas == 12
        assert val_personas == 4
        assert test_personas == 4

    def test_no_persona_leakage(self):
        """No persona appears in more than one split."""
        labels_df, entries_df = _make_fake_data(n_personas=20)
        train_df, val_df, test_df = split_by_persona(
            labels_df, entries_df, train_ratio=0.60, val_ratio=0.20
        )

        train_ids = set(train_df.select("persona_id").unique().to_series().to_list())
        val_ids = set(val_df.select("persona_id").unique().to_series().to_list())
        test_ids = set(test_df.select("persona_id").unique().to_series().to_list())

        assert train_ids & val_ids == set()
        assert train_ids & test_ids == set()
        assert val_ids & test_ids == set()

    def test_all_entries_preserved(self):
        """Total entry count across splits equals the merged total."""
        labels_df, entries_df = _make_fake_data(n_personas=20)
        train_df, val_df, test_df = split_by_persona(
            labels_df, entries_df, train_ratio=0.60, val_ratio=0.20
        )

        total = len(train_df) + len(val_df) + len(test_df)
        # All 20 personas x 5 entries = 100 entries
        assert total == 100

    def test_seed_reproducibility(self):
        """Same seed produces identical splits."""
        labels_df, entries_df = _make_fake_data(n_personas=20)

        split_a = split_by_persona(labels_df, entries_df, seed=99)
        split_b = split_by_persona(labels_df, entries_df, seed=99)

        for df_a, df_b in zip(split_a, split_b):
            assert df_a.equals(df_b)

    def test_different_seeds_differ(self):
        """Different seeds produce different splits."""
        labels_df, entries_df = _make_fake_data(n_personas=20)

        train_a, _, _ = split_by_persona(labels_df, entries_df, seed=1)
        train_b, _, _ = split_by_persona(labels_df, entries_df, seed=2)

        ids_a = set(train_a.select("persona_id").unique().to_series().to_list())
        ids_b = set(train_b.select("persona_id").unique().to_series().to_list())
        assert ids_a != ids_b

    def test_fixed_holdouts_are_respected(self):
        """Explicit val/test persona IDs should bypass reshuffling."""
        labels_df, entries_df = _make_fake_data(n_personas=12, entries_per_persona=2)

        train_df, val_df, test_df = split_by_persona(
            labels_df,
            entries_df,
            fixed_val_persona_ids={"persona_001", "persona_003"},
            fixed_test_persona_ids={"persona_005", "persona_007"},
        )

        train_ids = set(train_df.select("persona_id").unique().to_series().to_list())
        val_ids = set(val_df.select("persona_id").unique().to_series().to_list())
        test_ids = set(test_df.select("persona_id").unique().to_series().to_list())

        assert val_ids == {"persona_001", "persona_003"}
        assert test_ids == {"persona_005", "persona_007"}
        assert train_ids == {
            "persona_000",
            "persona_002",
            "persona_004",
            "persona_006",
            "persona_008",
            "persona_009",
            "persona_010",
            "persona_011",
        }

    def test_fixed_holdouts_require_both_sets(self):
        """Providing only one fixed holdout set should fail fast."""
        labels_df, entries_df = _make_fake_data(n_personas=8, entries_per_persona=2)

        with pytest.raises(ValueError, match="must either both be provided"):
            split_by_persona(
                labels_df,
                entries_df,
                fixed_val_persona_ids={"persona_001"},
            )

    def test_fixed_holdouts_reject_overlap(self):
        """Validation/test holdouts must be disjoint."""
        labels_df, entries_df = _make_fake_data(n_personas=8, entries_per_persona=2)

        with pytest.raises(ValueError, match="overlap"):
            split_by_persona(
                labels_df,
                entries_df,
                fixed_val_persona_ids={"persona_001"},
                fixed_test_persona_ids={"persona_001"},
            )

    def test_fixed_holdouts_reject_missing_personas(self):
        """Unknown persona IDs in the manifest should raise."""
        labels_df, entries_df = _make_fake_data(n_personas=8, entries_per_persona=2)

        with pytest.raises(ValueError, match="not present"):
            split_by_persona(
                labels_df,
                entries_df,
                fixed_val_persona_ids={"persona_001"},
                fixed_test_persona_ids={"persona_999"},
            )

    def test_sparse_signals_present_in_val_and_test_when_feasible(self):
        """Sparse sign features should appear in val and test when expected counts allow it."""
        labels_df, entries_df = _make_skewed_data(n_personas=60, entries_per_persona=2)
        _, val_df, test_df = split_by_persona(labels_df, entries_df, seed=42)

        sparse_features = [
            (4, 1),   # power +
            (4, -1),  # power -
            (5, 1),   # security +
            (5, -1),  # security -
            (9, 1),   # universalism +
        ]

        for split_df in (val_df, test_df):
            for dim_idx, score in sparse_features:
                persona_hits = set()
                for row in split_df.select(["persona_id", "alignment_vector"]).iter_rows(named=True):
                    if row["alignment_vector"][dim_idx] == score:
                        persona_hits.add(row["persona_id"])

                assert len(persona_hits) >= 1

    def test_stratified_split_improves_gap_vs_random_baseline(self):
        """Stratified val/test should not be worse than legacy random split on sparse signals."""
        labels_df, entries_df = _make_skewed_data(n_personas=60, entries_per_persona=2)
        train_df, val_df, test_df = split_by_persona(labels_df, entries_df, seed=42)

        train_random, val_random, test_random = _random_split_personas(labels_df, seed=42)

        features = [
            (4, 1), (4, -1),
            (5, 1), (5, -1),
            (9, 1),
        ]
        persona_ids, feature_matrix = _persona_feature_matrix(labels_df, features)

        val_ids = set(val_df.select("persona_id").unique().to_series().to_list())
        test_ids = set(test_df.select("persona_id").unique().to_series().to_list())
        train_ids = set(train_df.select("persona_id").unique().to_series().to_list())

        # Guard: split_by_persona still respects ratio-derived sizes.
        assert len(train_ids) == len(train_random)
        assert len(val_ids) == len(val_random)
        assert len(test_ids) == len(test_random)

        stratified_gap = max(
            _max_prevalence_gap(persona_ids, feature_matrix, val_ids),
            _max_prevalence_gap(persona_ids, feature_matrix, test_ids),
        )
        random_gap = max(
            _max_prevalence_gap(persona_ids, feature_matrix, val_random),
            _max_prevalence_gap(persona_ids, feature_matrix, test_random),
        )

        assert stratified_gap < random_gap

    def test_small_dataset_allows_empty_val_without_crash(self):
        """Tiny persona counts can yield empty val split but still return valid partitions."""
        labels_df, entries_df = _make_fake_data(n_personas=3, entries_per_persona=2)
        train_df, val_df, test_df = split_by_persona(
            labels_df,
            entries_df,
            train_ratio=0.80,
            val_ratio=0.10,
            seed=7,
        )

        train_ids = set(train_df.select("persona_id").unique().to_series().to_list())
        val_ids = set(val_df.select("persona_id").unique().to_series().to_list())
        test_ids = set(test_df.select("persona_id").unique().to_series().to_list())

        assert len(train_ids) == 2
        assert len(val_ids) == 0
        assert len(test_ids) == 1
        assert train_ids & val_ids == set()
        assert train_ids & test_ids == set()
        assert val_ids & test_ids == set()


# ── create_dataloaders passthrough test ──────────────────────────────────────


class TestCreateDataloadersPassthrough:
    """Verify create_dataloaders passes ratios through to split_by_persona."""

    @patch("src.vif.dataset.torch.utils.data.DataLoader")
    @patch("src.vif.dataset.VIFDataset")
    @patch("src.vif.dataset.split_by_persona")
    @patch("src.vif.dataset.load_all_data")
    def test_ratios_forwarded(self, mock_load, mock_split, mock_dataset, mock_loader):
        """train_ratio and val_ratio are forwarded to split_by_persona."""
        # Setup mocks so create_dataloaders doesn't need real data/encoder
        fake_labels = pl.DataFrame({"persona_id": ["p1"], "t_index": [0]})
        fake_entries = pl.DataFrame({"persona_id": ["p1"], "t_index": [0]})
        mock_load.return_value = (fake_labels, fake_entries)

        # split_by_persona needs to return 3 DataFrames
        empty_df = pl.DataFrame({
            "persona_id": pl.Series([], dtype=pl.Utf8),
            "t_index": pl.Series([], dtype=pl.Int64),
        })
        mock_split.return_value = (empty_df, empty_df, empty_df)
        encoder = object()
        mock_dataset.side_effect = ["train_dataset", "val_dataset", "test_dataset"]
        mock_loader.side_effect = ["train_loader", "val_loader", "test_loader"]

        train_loader, val_loader, test_loader = create_dataloaders(
            state_encoder=encoder,
            train_ratio=0.60,
            val_ratio=0.20,
            seed=99,
        )

        # Verify split_by_persona was called with the right ratios
        mock_split.assert_called_once_with(
            fake_labels, fake_entries,
            train_ratio=0.60,
            val_ratio=0.20,
            seed=99,
            fixed_val_persona_ids=None,
            fixed_test_persona_ids=None,
        )
        assert mock_dataset.call_args_list == [
            call(empty_df, encoder, cache_embeddings=True),
            call(empty_df, encoder, cache_embeddings=True),
            call(empty_df, encoder, cache_embeddings=True),
        ]
        assert mock_loader.call_args_list == [
            call("train_dataset", batch_size=16, shuffle=True),
            call("val_dataset", batch_size=16, shuffle=False),
            call("test_dataset", batch_size=16, shuffle=False),
        ]
        assert (train_loader, val_loader, test_loader) == (
            "train_loader",
            "val_loader",
            "test_loader",
        )

    @patch("src.vif.dataset.torch.utils.data.DataLoader")
    @patch("src.vif.dataset.VIFDataset")
    @patch("src.vif.dataset.split_by_persona")
    @patch("src.vif.dataset.load_fixed_holdout_ids")
    @patch("src.vif.dataset.load_all_data")
    def test_manifest_holdouts_loaded_and_forwarded(
        self,
        mock_load,
        mock_load_holdout_ids,
        mock_split,
        mock_dataset,
        mock_loader,
    ):
        """Manifest-backed holdouts should be resolved before splitting."""
        fake_labels = pl.DataFrame({"persona_id": ["p1"], "t_index": [0]})
        fake_entries = pl.DataFrame({"persona_id": ["p1"], "t_index": [0]})
        mock_load.return_value = (fake_labels, fake_entries)
        mock_load_holdout_ids.return_value = ({"persona_val"}, {"persona_test"})

        empty_df = pl.DataFrame({
            "persona_id": pl.Series([], dtype=pl.Utf8),
            "t_index": pl.Series([], dtype=pl.Int64),
        })
        mock_split.return_value = (empty_df, empty_df, empty_df)
        encoder = object()
        mock_dataset.side_effect = ["train_dataset", "val_dataset", "test_dataset"]
        mock_loader.side_effect = ["train_loader", "val_loader", "test_loader"]

        train_loader, val_loader, test_loader = create_dataloaders(
            state_encoder=encoder,
            fixed_holdout_manifest_path="config/experiments/vif/twinkl_681_5_holdout.yaml",
        )

        mock_load_holdout_ids.assert_called_once_with(
            "config/experiments/vif/twinkl_681_5_holdout.yaml"
        )
        mock_split.assert_called_once_with(
            fake_labels,
            fake_entries,
            train_ratio=0.70,
            val_ratio=0.15,
            seed=42,
            fixed_val_persona_ids={"persona_val"},
            fixed_test_persona_ids={"persona_test"},
        )
        assert mock_dataset.call_args_list == [
            call(empty_df, encoder, cache_embeddings=True),
            call(empty_df, encoder, cache_embeddings=True),
            call(empty_df, encoder, cache_embeddings=True),
        ]
        assert mock_loader.call_args_list == [
            call("train_dataset", batch_size=16, shuffle=True),
            call("val_dataset", batch_size=16, shuffle=False),
            call("test_dataset", batch_size=16, shuffle=False),
        ]
        assert (train_loader, val_loader, test_loader) == (
            "train_loader",
            "val_loader",
            "test_loader",
        )

    @patch("src.vif.dataset.load_all_data")
    def test_manifest_and_explicit_holdouts_conflict(self, mock_load):
        """Explicit holdout IDs and manifest path should be mutually exclusive."""
        fake_labels = pl.DataFrame({"persona_id": ["p1"], "t_index": [0]})
        fake_entries = pl.DataFrame({"persona_id": ["p1"], "t_index": [0]})
        mock_load.return_value = (fake_labels, fake_entries)

        class FakeEncoder:
            pass

        with pytest.raises(ValueError, match="either fixed_holdout_manifest_path or explicit"):
            create_dataloaders(
                state_encoder=FakeEncoder(),
                fixed_val_persona_ids={"persona_val"},
                fixed_test_persona_ids={"persona_test"},
                fixed_holdout_manifest_path="config/experiments/vif/twinkl_681_5_holdout.yaml",
            )
