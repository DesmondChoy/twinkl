"""Tests for merge_labels_and_entries integrity checks."""

import logging

import polars as pl
import pytest

from src.vif.dataset import merge_labels_and_entries


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_fake_data(n_personas: int = 3, entries_per_persona: int = 4):
    """Create minimal labels and entries DataFrames with matching keys."""
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


# ── Tests ────────────────────────────────────────────────────────────────────


class TestMergeIntegrity:
    """Tests for merge_labels_and_entries join validation."""

    def test_perfect_match_passes_silently(self):
        """No exception when all keys align between labels and entries."""
        labels_df, entries_df = _make_fake_data()
        merged = merge_labels_and_entries(labels_df, entries_df)
        assert len(merged) == len(labels_df)

    def test_labels_without_entries_raises(self):
        """ValueError when labels have an orphan persona not in entries."""
        labels_df, entries_df = _make_fake_data(n_personas=3)
        # Add extra label rows for a persona that has no entries
        extra = pl.DataFrame([{
            "persona_id": "persona_ghost",
            "t_index": 0,
            "date": "2025-01-01",
            "alignment_vector": [0.0] * 10,
        }])
        labels_df = pl.concat([labels_df, extra])

        with pytest.raises(ValueError, match="Labels without matching entries"):
            merge_labels_and_entries(labels_df, entries_df)

    def test_entries_without_labels_raises(self):
        """ValueError when entries have an orphan persona not in labels."""
        labels_df, entries_df = _make_fake_data(n_personas=3)
        # Add extra entry rows for a persona that has no labels
        extra = pl.DataFrame([{
            "persona_id": "persona_ghost",
            "t_index": 0,
            "date": "2025-01-01",
            "initial_entry": "Ghost entry",
            "nudge_text": None,
            "response_text": None,
            "core_values": "power",
        }])
        entries_df = pl.concat([entries_df, extra])

        with pytest.raises(ValueError, match="Entries without matching labels"):
            merge_labels_and_entries(labels_df, entries_df)

    def test_both_sides_orphans_raises(self):
        """ValueError reports both anti-join counts when both sides have orphans."""
        labels_df, entries_df = _make_fake_data(n_personas=3)
        # Orphan label
        extra_label = pl.DataFrame([{
            "persona_id": "persona_label_only",
            "t_index": 0,
            "date": "2025-01-01",
            "alignment_vector": [0.0] * 10,
        }])
        # Orphan entry
        extra_entry = pl.DataFrame([{
            "persona_id": "persona_entry_only",
            "t_index": 0,
            "date": "2025-01-01",
            "initial_entry": "Ghost",
            "nudge_text": None,
            "response_text": None,
            "core_values": "power",
        }])
        labels_df = pl.concat([labels_df, extra_label])
        entries_df = pl.concat([entries_df, extra_entry])

        with pytest.raises(ValueError, match="Labels without matching entries") as exc_info:
            merge_labels_and_entries(labels_df, entries_df)
        assert "Entries without matching labels" in str(exc_info.value)

    def test_t_index_mismatch_raises(self):
        """ValueError for t_index divergence within the same persona."""
        labels_df, entries_df = _make_fake_data(n_personas=1, entries_per_persona=3)
        # Add a label for t_index=99 that has no matching entry
        extra_label = pl.DataFrame([{
            "persona_id": "persona_000",
            "t_index": 99,
            "date": "2025-06-01",
            "alignment_vector": [0.0] * 10,
        }])
        labels_df = pl.concat([labels_df, extra_label])

        with pytest.raises(ValueError, match="Merge dropped rows"):
            merge_labels_and_entries(labels_df, entries_df)

    def test_strict_false_returns_partial(self):
        """strict=False returns the inner-join result despite drops."""
        labels_df, entries_df = _make_fake_data(n_personas=3)
        extra = pl.DataFrame([{
            "persona_id": "persona_ghost",
            "t_index": 0,
            "date": "2025-01-01",
            "alignment_vector": [0.0] * 10,
        }])
        labels_df = pl.concat([labels_df, extra])

        merged = merge_labels_and_entries(labels_df, entries_df, strict=False)
        # The orphan label row is dropped; original 3 personas remain
        assert len(merged) == 3 * 4  # 3 personas x 4 entries

    def test_strict_false_logs_warning(self, caplog):
        """Warning logged with diagnostics when strict=False and drops exist."""
        labels_df, entries_df = _make_fake_data(n_personas=3)
        extra = pl.DataFrame([{
            "persona_id": "persona_ghost",
            "t_index": 0,
            "date": "2025-01-01",
            "alignment_vector": [0.0] * 10,
        }])
        labels_df = pl.concat([labels_df, extra])

        with caplog.at_level(logging.WARNING, logger="src.vif.dataset"):
            merge_labels_and_entries(labels_df, entries_df, strict=False)

        assert "Merge dropped rows" in caplog.text
        assert "persona_ghost" in caplog.text

    def test_empty_dataframes_pass(self):
        """Two empty DataFrames with matching schemas merge cleanly."""
        labels_df = pl.DataFrame({
            "persona_id": pl.Series([], dtype=pl.Utf8),
            "t_index": pl.Series([], dtype=pl.Int64),
            "date": pl.Series([], dtype=pl.Utf8),
            "alignment_vector": pl.Series([], dtype=pl.List(pl.Float64)),
        })
        entries_df = pl.DataFrame({
            "persona_id": pl.Series([], dtype=pl.Utf8),
            "t_index": pl.Series([], dtype=pl.Int64),
            "date": pl.Series([], dtype=pl.Utf8),
            "initial_entry": pl.Series([], dtype=pl.Utf8),
            "nudge_text": pl.Series([], dtype=pl.Utf8),
            "response_text": pl.Series([], dtype=pl.Utf8),
            "core_values": pl.Series([], dtype=pl.Utf8),
        })

        merged = merge_labels_and_entries(labels_df, entries_df)
        assert len(merged) == 0
