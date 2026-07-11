"""State encoder for VIF Critic training.

This module constructs the full state vector from journal entries,
combining text embeddings with temporal features and persona context.

State Vector Components:
- current/raw text window: N × d_e
- optional compact prior-history summary: d_h + 1 history-count feature
- time_gaps: N-1 (raw-window mode only)
- user_profile: 10 (normalized weights from Core Values)

The default remains the current entry only. The compact-history experiment keeps
that full embedding and appends a small summary of strictly prior embeddings.

Note: Encoder choice, window size, and other hyperparameters in config/vif.yaml
are preliminary and subject to revision through ongoing model ablation studies.

Usage:
    from src.vif import StateEncoder, create_encoder

    encoder = create_encoder(config["encoder"])
    state_encoder = StateEncoder(encoder, **config["state_encoder"])

    # Build state for a single entry
    state = state_encoder.build_state_vector(
        texts=["Today I helped a colleague...", "Yesterday I worked late..."],
        dates=["2024-01-15", "2024-01-14"],
        core_values=["Benevolence", "Security"],
    )
"""

from collections.abc import Sequence
from datetime import datetime

import numpy as np

from src.models.judge import SCHWARTZ_VALUE_ORDER
from src.vif.encoders import TextEncoder

# Mapping from display names (in markdown) to canonical keys (in SCHWARTZ_VALUE_ORDER)
# Core Values in persona files use title case, need to map to snake_case
VALUE_NAME_TO_KEY = {
    "Self-Direction": "self_direction",
    "Self-direction": "self_direction",
    "Stimulation": "stimulation",
    "Hedonism": "hedonism",
    "Achievement": "achievement",
    "Power": "power",
    "Security": "security",
    "Conformity": "conformity",
    "Tradition": "tradition",
    "Benevolence": "benevolence",
    "Universalism": "universalism",
}


def core_values_to_profile_weights(core_values: Sequence[str]) -> np.ndarray:
    """Map declared core values to the runtime's normalized profile vector.

    The active VIF state encodes every declared core value with equal mass and
    uses a uniform fallback when no valid declaration is available. Keeping this
    mapping outside ``StateEncoder`` lets audit tooling render the exact profile
    information the student receives without constructing a text encoder.
    """
    weights = np.zeros(len(SCHWARTZ_VALUE_ORDER), dtype=np.float32)
    matched_indices = []

    for value in core_values:
        canonical_key = VALUE_NAME_TO_KEY.get(value)
        if canonical_key is None:
            canonical_key = VALUE_NAME_TO_KEY.get(value.strip())
        if canonical_key is None:
            lower_value = value.lower().replace("-", "_").strip()
            if lower_value in SCHWARTZ_VALUE_ORDER:
                canonical_key = lower_value

        if canonical_key and canonical_key in SCHWARTZ_VALUE_ORDER:
            matched_indices.append(SCHWARTZ_VALUE_ORDER.index(canonical_key))

    matched_indices = list(dict.fromkeys(matched_indices))
    if matched_indices:
        weight_per_value = 1.0 / len(matched_indices)
        for index in matched_indices:
            weights[index] = weight_per_value
    else:
        weights = np.full(
            len(SCHWARTZ_VALUE_ORDER),
            1.0 / len(SCHWARTZ_VALUE_ORDER),
            dtype=np.float32,
        )

    return weights


def concatenate_entry_text(
    initial_entry: str | None,
    nudge_text: str | None,
    response_text: str | None,
) -> str:
    """Build the exact per-entry text used by the trained and runtime VIF path."""
    parts = []
    if initial_entry:
        parts.append(initial_entry)
    if nudge_text:
        parts.append(f'Nudge: "{nudge_text}"')
    if response_text:
        parts.append(f"Response: {response_text}")
    return "\n\n".join(parts) if parts else ""


class StateEncoder:
    """Encodes journal entries into state vectors for VIF training.

    The state vector combines:
    1. Text embeddings from a sliding window of entries
    2. Temporal features (time gaps between entries)
    3. User profile (normalized value weights from Core Values)

    This encoder supports pluggable text encoders for ablation studies.

    Example:
        encoder = create_encoder(config["encoder"])
        state_encoder = StateEncoder(encoder, **config["state_encoder"])

        # Get state dimension (for MLP input size)
        print(f"State dim: {state_encoder.state_dim}")
    """

    def __init__(
        self,
        text_encoder: TextEncoder,
        window_size: int = 3,
        history_pooling: str = "none",
        history_window_size: int = 3,
        history_summary_dim: int = 64,
        **kwargs: object,
    ):
        """Initialize the state encoder.

        Args:
            text_encoder: TextEncoder instance for converting text to embeddings
            window_size: Number of entries in the text window (default: 3)
            history_pooling: ``none`` or ``mean``. Compact history is separate
                from the raw text window and is strictly prior-only.
            history_window_size: Maximum number of prior entries to summarize.
            history_summary_dim: Leading dimensions retained from the pooled
                prior embedding. The official experiment uses Nomic's
                Matryoshka ordering; other encoders need separate validation.
            **kwargs: Deprecated compatibility args. ``ema_alpha`` is accepted
                and ignored for v1 notebook compatibility; all other kwargs
                raise ``TypeError``.
        """
        kwargs.pop("ema_alpha", None)
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(
                "Unexpected StateEncoder kwargs: "
                f"{unknown}. Only deprecated 'ema_alpha' is accepted."
            )

        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if history_pooling not in {"none", "mean"}:
            raise ValueError("history_pooling must be 'none' or 'mean'")
        if history_window_size < 1:
            raise ValueError("history_window_size must be >= 1")
        if history_summary_dim < 1:
            raise ValueError("history_summary_dim must be >= 1")
        if (
            history_pooling != "none"
            and history_summary_dim > text_encoder.embedding_dim
        ):
            raise ValueError(
                "history_summary_dim cannot exceed the text embedding dimension"
            )
        if history_pooling != "none" and window_size != 1:
            raise ValueError(
                "compact history requires window_size=1 so raw concatenation "
                "and pooled history cannot be mixed"
            )

        self.text_encoder = text_encoder
        self.window_size = window_size
        self.history_pooling = history_pooling
        self.history_window_size = history_window_size
        self.history_summary_dim = history_summary_dim
        self.num_values = len(SCHWARTZ_VALUE_ORDER)  # 10

    @property
    def input_entry_count(self) -> int:
        """Number of current/prior entries needed to build one state."""
        if self.history_pooling == "none":
            return self.window_size
        return 1 + self.history_window_size

    @property
    def state_dim(self) -> int:
        """Total dimension of the state vector.

        Components:
        - text_window: window_size × embedding_dim
        - time_gaps: window_size - 1
        - user_profile: num_values (10)
        """
        raw_state_dim = (
            self.window_size * self.text_encoder.embedding_dim
            + (self.window_size - 1)  # time gaps
            + self.num_values  # profile weights
        )
        if self.history_pooling == "none":
            return raw_state_dim
        return (
            self.text_encoder.embedding_dim
            + self.history_summary_dim
            + 1  # normalized count of real prior entries
            + self.num_values
        )

    def parse_core_values_to_weights(self, core_values: list[str]) -> np.ndarray:
        """Convert Core Values list to a normalized 10-dim weight vector.

        This maps the persona's declared Core Values (e.g., ["Security", "Benevolence"])
        to a normalized weight vector aligned with SCHWARTZ_VALUE_ORDER.

        Args:
            core_values: List of Schwartz value names from persona profile
                        (e.g., ["Security", "Benevolence"])

        Returns:
            np.ndarray of shape (10,) with normalized weights summing to 1.0.
            Values mentioned in core_values get equal positive weight,
            unmentioned values get zero weight. If empty/invalid, returns
            uniform weights (0.1 each).

        Example:
            >>> values = ["Security", "Benevolence"]
            >>> weights = encoder.parse_core_values_to_weights(values)
            >>> print(weights)  # [0, 0, 0, 0, 0, 0.5, 0, 0, 0.5, 0]
            >>> print(weights.sum())  # 1.0
        """
        return core_values_to_profile_weights(core_values)

    def compute_time_gaps(
        self,
        dates: list[str],
    ) -> np.ndarray:
        """Compute time gaps in days between consecutive entries.

        Args:
            dates: List of date strings in YYYY-MM-DD format,
                  ordered from current (first) to oldest (last).
                  Length should be window_size.

        Returns:
            np.ndarray of shape (window_size - 1,) with day gaps.
            Gaps are clamped to [0, 30] and normalized to [0, 1].

        Note:
            For missing dates (padding), uses default gap of 7 days.
        """
        gaps = []
        default_gap = 7.0  # Default gap for padding

        for i in range(len(dates) - 1):
            current = dates[i]
            previous = dates[i + 1]

            if current and previous:
                try:
                    current_dt = datetime.strptime(current, "%Y-%m-%d")
                    previous_dt = datetime.strptime(previous, "%Y-%m-%d")
                    delta = (current_dt - previous_dt).days
                    # Clamp to reasonable range
                    delta = max(0, min(delta, 30))
                except ValueError:
                    delta = default_gap
            else:
                delta = default_gap

            gaps.append(delta)

        # Pad if needed
        while len(gaps) < self.window_size - 1:
            gaps.append(default_gap)

        # Normalize to [0, 1]
        gaps_array = np.array(gaps[: self.window_size - 1], dtype=np.float32) / 30.0

        return gaps_array

    def build_state_vector(
        self,
        texts: list[str],
        dates: list[str],
        core_values: list[str],
    ) -> np.ndarray:
        """Build the complete state vector for a single entry.

        Combines text embeddings, temporal features, and user profile
        into a single vector for the VIF model.

        Args:
            texts: List of entry texts, current entry first, then previous entries.
                  Length <= window_size. Zero-padded if fewer entries available.
            dates: List of dates corresponding to texts (YYYY-MM-DD format).
            core_values: List of Schwartz value names from persona profile.

        Returns:
            np.ndarray of shape (state_dim,) containing the full state vector.
        """
        # 1. Encode only real entries; assembly handles cold-start padding.
        text_embeddings = []
        for i in range(min(len(texts), self.input_entry_count)):
            if texts[i]:
                emb = self.text_encoder.encode_batch([texts[i]])[0]
            else:
                emb = np.zeros(self.text_encoder.embedding_dim, dtype=np.float32)
            text_embeddings.append(emb)

        return self._build_state_from_components(text_embeddings, dates, core_values)

    def build_state_vector_from_embeddings(
        self,
        embeddings: list[np.ndarray],
        dates: list[str],
        core_values: list[str],
    ) -> np.ndarray:
        """Build state vector from pre-computed embeddings.

        Use this when embeddings are cached to avoid redundant encoding.

        Args:
            embeddings: Real pre-computed embeddings, current entry first.
                Compact-history mode accepts current plus prior entries without
                padding; raw-window mode pads internally.
            dates: List of dates corresponding to embeddings (YYYY-MM-DD format).
            core_values: List of Schwartz value names from persona profile.

        Returns:
            np.ndarray of shape (state_dim,) containing the full state vector.
        """
        return self._build_state_from_components(embeddings, dates, core_values)

    def _build_state_from_components(
        self,
        text_embeddings: list[np.ndarray],
        dates: list[str],
        core_values: list[str],
    ) -> np.ndarray:
        """Internal method to assemble state vector from components."""
        embedding_dim = self.text_encoder.embedding_dim
        limited_embeddings = [
            np.asarray(embedding, dtype=np.float32)
            for embedding in text_embeddings[: self.input_entry_count]
        ]

        if self.history_pooling == "none":
            while len(limited_embeddings) < self.window_size:
                limited_embeddings.append(np.zeros(embedding_dim, dtype=np.float32))
            text_vector = np.concatenate(limited_embeddings[: self.window_size])
        else:
            current_embedding = (
                limited_embeddings[0]
                if limited_embeddings
                else np.zeros(embedding_dim, dtype=np.float32)
            )
            prior_embeddings = limited_embeddings[1 : 1 + self.history_window_size]
            history_summary = np.zeros(self.history_summary_dim, dtype=np.float32)
            if prior_embeddings:
                pooled = np.mean(np.stack(prior_embeddings), axis=0)
                history_summary = pooled[: self.history_summary_dim].astype(np.float32)
                summary_norm = float(np.linalg.norm(history_summary))
                if summary_norm > 0:
                    history_summary /= summary_norm
            history_count = np.array(
                [len(prior_embeddings) / self.history_window_size],
                dtype=np.float32,
            )
            text_vector = np.concatenate(
                [current_embedding, history_summary, history_count]
            )

        # 2. Time gaps
        padded_dates = dates + [None] * (self.window_size - len(dates))
        time_gaps = self.compute_time_gaps(padded_dates[: self.window_size])

        # 3. Profile weights
        profile_weights = self.parse_core_values_to_weights(core_values)

        # Concatenate all components
        state_vector = np.concatenate(
            [
                text_vector,
                time_gaps,
                profile_weights,
            ]
        )

        return state_vector.astype(np.float32)

    def concatenate_entry_text(
        self,
        initial_entry: str | None,
        nudge_text: str | None,
        response_text: str | None,
    ) -> str:
        """Concatenate entry components into a single text for embedding.

        The full context (entry + nudge + response) provides richer signals
        than the initial entry alone, as responses often contain deeper
        reflection about values.

        Args:
            initial_entry: The main journal entry text
            nudge_text: The follow-up question (if any)
            response_text: The user's response to the nudge (if any)

        Returns:
            Concatenated text with appropriate separators
        """
        return concatenate_entry_text(initial_entry, nudge_text, response_text)
