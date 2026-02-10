"""Dataset and data loading utilities for VIF Critic training.

This module provides functions to load labeled training data and a PyTorch
Dataset class that builds state vectors with embedding caching.

Data sources:
- logs/judge_labels/judge_labels.parquet: Alignment labels
- logs/wrangled/persona_*.md: Entry text content

Usage:
    from src.vif.dataset import load_all_data, split_by_persona, VIFDataset

    # Load raw data
    labels_df, entries_df = load_all_data()

    # Split by persona (avoids leakage from correlated entries)
    train_df, val_df, test_df = split_by_persona(labels_df, entries_df)

    # Create PyTorch dataset
    dataset = VIFDataset(train_df, state_encoder)
"""

from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from src.vif.state_encoder import StateEncoder
from src.wrangling.parse_wrangled_data import parse_wrangled_file


def load_labels(
    labels_path: str | Path = "logs/judge_labels/judge_labels.parquet",
) -> pl.DataFrame:
    """Load alignment labels from parquet file.

    Args:
        labels_path: Path to judge_labels.parquet

    Returns:
        DataFrame with columns: persona_id, t_index, date, alignment_vector,
        and individual alignment_* columns for each Schwartz dimension.
    """
    return pl.read_parquet(labels_path)


def load_entries(
    wrangled_dir: str | Path = "logs/wrangled",
) -> pl.DataFrame:
    """Load all wrangled entries from markdown files.

    Uses the wrangled-format parser which expects entry text directly after
    date headers and **Nudge:**/**Response:** inline markers.

    Args:
        wrangled_dir: Path to directory containing persona_*.md files

    Returns:
        DataFrame with columns: persona_id, t_index, date, initial_entry,
        nudge_text, response_text, core_values, and other persona fields.

    Raises:
        FileNotFoundError: If no persona files found.
        ValueError: If all initial_entry values are null (parser mismatch).
    """
    wrangled_path = Path(wrangled_dir)
    persona_files = sorted(wrangled_path.glob("persona_*.md"))

    if not persona_files:
        raise FileNotFoundError(f"No persona_*.md files found in {wrangled_dir}")

    rows = []
    for filepath in persona_files:
        profile, entries, _warnings = parse_wrangled_file(filepath)

        for entry in entries:
            row = {
                "persona_id": profile["persona_id"],
                "persona_name": profile["name"],
                "core_values": profile["core_values"],
                "t_index": entry["t_index"],
                "date": entry["date"],
                "initial_entry": entry["initial_entry"],
                "nudge_text": entry["nudge_text"],
                "response_text": entry["response_text"],
            }
            rows.append(row)

    df = pl.DataFrame(rows)

    # Validate: if all initial_entry values are null, the parser didn't match the format
    if len(df) > 0 and df["initial_entry"].is_null().all():
        raise ValueError(
            f"All initial_entry values are null across {len(df)} entries from "
            f"{len(persona_files)} files in {wrangled_dir}. This likely indicates a "
            f"parser mismatch — ensure files are in wrangled format, not raw synthetic format."
        )

    return df


def load_all_data(
    labels_path: str | Path = "logs/judge_labels/judge_labels.parquet",
    wrangled_dir: str | Path = "logs/wrangled",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load both labels and entries, returning joined data.

    Args:
        labels_path: Path to judge_labels.parquet
        wrangled_dir: Path to directory containing wrangled persona_*.md files

    Returns:
        Tuple of (labels_df, entries_df) where:
        - labels_df: Alignment labels with persona_id, t_index, alignment_vector
        - entries_df: Entry text with persona_id, t_index, text content, core_values
    """
    labels_df = load_labels(labels_path)
    entries_df = load_entries(wrangled_dir)

    return labels_df, entries_df


def merge_labels_and_entries(
    labels_df: pl.DataFrame,
    entries_df: pl.DataFrame,
) -> pl.DataFrame:
    """Join labels and entries on (persona_id, t_index).

    Args:
        labels_df: DataFrame with alignment labels
        entries_df: DataFrame with entry text

    Returns:
        Merged DataFrame with both labels and text content
    """
    return labels_df.join(
        entries_df,
        on=["persona_id", "t_index"],
        how="inner",
    )


def split_by_persona(
    labels_df: pl.DataFrame,
    entries_df: pl.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split data by persona to avoid leakage from correlated entries.

    Entries from the same persona are kept together in the same split,
    since consecutive entries are temporally correlated and share context.

    Args:
        labels_df: DataFrame with alignment labels
        entries_df: DataFrame with entry text
        train_ratio: Fraction for training (default 0.70)
        val_ratio: Fraction for validation (default 0.15)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_df, val_df, test_df) where each is a merged DataFrame
        containing both labels and entries.
    """
    # Get unique persona IDs
    persona_ids = sorted(labels_df.select("persona_id").unique().to_series().to_list())
    n_personas = len(persona_ids)

    # Shuffle persona IDs
    rng = np.random.default_rng(seed)
    rng.shuffle(persona_ids)

    # Calculate split indices
    train_end = int(n_personas * train_ratio)
    val_end = int(n_personas * (train_ratio + val_ratio))

    train_personas = set(persona_ids[:train_end])
    val_personas = set(persona_ids[train_end:val_end])
    test_personas = set(persona_ids[val_end:])

    # Merge labels and entries
    merged_df = merge_labels_and_entries(labels_df, entries_df)

    # Filter by persona sets
    train_df = merged_df.filter(pl.col("persona_id").is_in(train_personas))
    val_df = merged_df.filter(pl.col("persona_id").is_in(val_personas))
    test_df = merged_df.filter(pl.col("persona_id").is_in(test_personas))

    return train_df, val_df, test_df


class VIFDataset(Dataset):
    """PyTorch Dataset for VIF Critic training.

    Builds state vectors from entries with optional embedding caching
    to avoid recomputing text embeddings during training.

    Each sample contains:
    - state: State vector (text embeddings + time gaps + history + profile)
    - target: Alignment score vector (10 dimensions)

    Example:
        state_encoder = StateEncoder(SBERTEncoder())
        dataset = VIFDataset(train_df, state_encoder)

        for state, target in DataLoader(dataset, batch_size=8):
            # state: (batch, state_dim), target: (batch, 10)
            ...
    """

    def __init__(
        self,
        data_df: pl.DataFrame,
        state_encoder: StateEncoder,
        cache_embeddings: bool = True,
    ):
        """Initialize the dataset.

        Args:
            data_df: Merged DataFrame with labels and entries
            state_encoder: StateEncoder instance for building state vectors
            cache_embeddings: If True, pre-compute and cache all text embeddings
        """
        self.data_df = data_df
        self.state_encoder = state_encoder
        self.cache_embeddings = cache_embeddings

        # Group data by persona for efficient history lookup
        self._build_persona_index()

        # Optionally pre-compute embeddings
        if cache_embeddings:
            self._cache_text_embeddings()

    def _build_persona_index(self):
        """Build index mapping persona_id to their entries."""
        self.persona_entries = {}
        # Direct lookup: (persona_id, t_index) -> row dict for O(1) access
        self.entry_lookup = {}

        for persona_id in self.data_df.select("persona_id").unique().to_series():
            persona_df = (
                self.data_df
                .filter(pl.col("persona_id") == persona_id)
                .sort("t_index")
            )
            self.persona_entries[persona_id] = persona_df

            # Build direct lookup for each entry
            for row in persona_df.iter_rows(named=True):
                self.entry_lookup[(persona_id, row["t_index"])] = row

        # Create flat list of (persona_id, t_index) for indexing
        self.index_map = []
        for row in self.data_df.iter_rows(named=True):
            self.index_map.append((row["persona_id"], row["t_index"]))

    def _cache_text_embeddings(self):
        """Pre-compute text embeddings for all entries."""
        self.embedding_cache = {}

        # Collect all unique texts
        texts = []
        text_keys = []

        for row in self.data_df.iter_rows(named=True):
            key = (row["persona_id"], row["t_index"])
            full_text = self.state_encoder.concatenate_entry_text(
                row["initial_entry"],
                row["nudge_text"],
                row["response_text"],
            )
            texts.append(full_text)
            text_keys.append(key)

        # Batch encode all texts
        if texts:
            embeddings = self.state_encoder.text_encoder.encode_batch(texts)
            for key, emb in zip(text_keys, embeddings):
                self.embedding_cache[key] = emb

    def _get_entry_text(self, row: dict) -> str:
        """Get concatenated text for an entry."""
        return self.state_encoder.concatenate_entry_text(
            row["initial_entry"],
            row["nudge_text"],
            row["response_text"],
        )

    def _get_embedding(self, persona_id: str, t_index: int) -> np.ndarray:
        """Get embedding for an entry, using cache if available."""
        key = (persona_id, t_index)
        if self.cache_embeddings and key in self.embedding_cache:
            return self.embedding_cache[key]

        # Fallback: encode on-the-fly if not cached
        row = self.entry_lookup.get(key)
        if row is None:
            return np.zeros(self.state_encoder.text_encoder.embedding_dim, dtype=np.float32)

        text = self._get_entry_text(row)
        return self.state_encoder.text_encoder.encode_batch([text])[0]

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single training sample.

        Args:
            idx: Index into the dataset

        Returns:
            Tuple of (state_tensor, target_tensor) where:
            - state_tensor: (state_dim,) float tensor
            - target_tensor: (10,) float tensor of alignment scores
        """
        persona_id, t_index = self.index_map[idx]
        persona_df = self.persona_entries[persona_id]

        # Get current entry via O(1) dict lookup
        current_row = self.entry_lookup[(persona_id, t_index)]

        # Build text window embeddings and dates (current + previous entries)
        window_size = self.state_encoder.window_size
        embeddings = []
        dates = []

        for offset in range(window_size):
            target_t = t_index - offset
            key = (persona_id, target_t)

            if target_t >= 0 and key in self.entry_lookup:
                # Use cached embedding if available, else encode on-the-fly
                emb = self._get_embedding(persona_id, target_t)
                embeddings.append(emb)
                dates.append(self.entry_lookup[key]["date"])
            else:
                # Zero embedding for missing/padding entries
                embeddings.append(
                    np.zeros(self.state_encoder.text_encoder.embedding_dim, dtype=np.float32)
                )
                dates.append(None)

        # Get alignment history (entries before current)
        alignment_history = []
        for hist_t in range(t_index):
            hist_key = (persona_id, hist_t)
            if hist_key in self.entry_lookup:
                alignment_history.append(
                    np.array(self.entry_lookup[hist_key]["alignment_vector"])
                )

        # Get core values
        core_values = current_row["core_values"]
        if isinstance(core_values, str):
            core_values = [v.strip() for v in core_values.split(",")]

        # Build state vector using pre-computed embeddings
        state = self.state_encoder.build_state_vector_from_embeddings(
            embeddings=embeddings,
            dates=dates,
            alignment_history=alignment_history,
            core_values=core_values,
        )

        # Get target (alignment vector)
        target = np.array(current_row["alignment_vector"], dtype=np.float32)

        return torch.from_numpy(state), torch.from_numpy(target)


def create_dataloaders(
    state_encoder: StateEncoder,
    batch_size: int = 16,
    seed: int = 42,
    labels_path: str | Path = "logs/judge_labels/judge_labels.parquet",
    wrangled_dir: str | Path = "logs/wrangled",
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train/val/test DataLoaders in one call.

    Convenience function that loads data, splits by persona, creates datasets,
    and wraps them in DataLoaders.

    Args:
        state_encoder: StateEncoder instance
        batch_size: Batch size for DataLoaders
        seed: Random seed for splitting
        labels_path: Path to labels parquet
        wrangled_dir: Path to wrangled markdown files

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load data
    labels_df, entries_df = load_all_data(labels_path, wrangled_dir)

    # Split by persona
    train_df, val_df, test_df = split_by_persona(
        labels_df, entries_df, seed=seed
    )

    # Create datasets
    train_dataset = VIFDataset(train_df, state_encoder, cache_embeddings=True)
    val_dataset = VIFDataset(val_df, state_encoder, cache_embeddings=True)
    test_dataset = VIFDataset(test_df, state_encoder, cache_embeddings=True)

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader
