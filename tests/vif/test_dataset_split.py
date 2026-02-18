"""Tests for dataset splitting with configurable train/val ratios."""

from unittest.mock import patch

import polars as pl

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


# ── create_dataloaders passthrough test ──────────────────────────────────────


class TestCreateDataloadersPassthrough:
    """Verify create_dataloaders passes ratios through to split_by_persona."""

    @patch("src.vif.dataset.split_by_persona")
    @patch("src.vif.dataset.load_all_data")
    def test_ratios_forwarded(self, mock_load, mock_split):
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

        # Mock state_encoder — VIFDataset needs it but we short-circuit earlier
        class FakeEncoder:
            pass

        try:
            create_dataloaders(
                state_encoder=FakeEncoder(),
                train_ratio=0.60,
                val_ratio=0.20,
                seed=99,
            )
        except Exception:
            pass  # VIFDataset may fail on the empty mock — that's fine

        # Verify split_by_persona was called with the right ratios
        mock_split.assert_called_once_with(
            fake_labels, fake_entries,
            train_ratio=0.60, val_ratio=0.20, seed=99,
        )
