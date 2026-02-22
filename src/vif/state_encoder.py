"""State encoder for VIF Critic training.

This module constructs the full state vector from journal entries,
combining text embeddings with temporal features and persona context.

State Vector Components (per VIF_05 spec):
- text_window: N × d_e (current + N-1 previous entry embeddings)
- time_gaps: N-1 (days since previous entries)
- user_profile: 10 (normalized weights from Core Values)

Total dimension: N × d_e + (N-1) + 10  (see config/vif.yaml)

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
        print(f"State dim: {state_encoder.state_dim}")  # depends on encoder and window_size
    """

    def __init__(
        self,
        text_encoder: TextEncoder,
        window_size: int = 3,
        **kwargs: object,
    ):
        """Initialize the state encoder.

        Args:
            text_encoder: TextEncoder instance for converting text to embeddings
            window_size: Number of entries in the text window (default: 3)
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

        self.text_encoder = text_encoder
        self.window_size = window_size
        self.num_values = len(SCHWARTZ_VALUE_ORDER)  # 10

    @property
    def state_dim(self) -> int:
        """Total dimension of the state vector.

        Components:
        - text_window: window_size × embedding_dim
        - time_gaps: window_size - 1
        - user_profile: num_values (10)
        """
        return (
            self.window_size * self.text_encoder.embedding_dim
            + (self.window_size - 1)  # time gaps
            + self.num_values  # profile weights
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
            >>> weights = encoder.parse_core_values_to_weights(["Security", "Benevolence"])
            >>> print(weights)  # [0, 0, 0, 0, 0, 0.5, 0, 0, 0.5, 0]
            >>> print(weights.sum())  # 1.0
        """
        weights = np.zeros(self.num_values, dtype=np.float32)

        matched_indices = []
        for value in core_values:
            # Try direct lookup first (handles title case like "Security")
            canonical_key = VALUE_NAME_TO_KEY.get(value)

            # If not found, try lowercase version
            if canonical_key is None:
                canonical_key = VALUE_NAME_TO_KEY.get(value.strip())

            # If still not found, try matching directly against SCHWARTZ_VALUE_ORDER
            if canonical_key is None:
                lower_value = value.lower().replace("-", "_").strip()
                if lower_value in SCHWARTZ_VALUE_ORDER:
                    canonical_key = lower_value

            if canonical_key and canonical_key in SCHWARTZ_VALUE_ORDER:
                idx = SCHWARTZ_VALUE_ORDER.index(canonical_key)
                matched_indices.append(idx)

        # Assign equal weight to matched values
        if matched_indices:
            weight_per_value = 1.0 / len(matched_indices)
            for idx in matched_indices:
                weights[idx] = weight_per_value
        else:
            # Fallback: uniform weights if no valid core values found
            weights = np.full(self.num_values, 1.0 / self.num_values, dtype=np.float32)

        return weights

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
        gaps_array = np.array(gaps[:self.window_size - 1], dtype=np.float32) / 30.0

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
        # 1. Text embeddings with zero-padding for early entries
        text_embeddings = []
        for i in range(self.window_size):
            if i < len(texts) and texts[i]:
                emb = self.text_encoder.encode_batch([texts[i]])[0]
            else:
                # Zero embedding for missing entries
                emb = np.zeros(self.text_encoder.embedding_dim, dtype=np.float32)
            text_embeddings.append(emb)

        return self._build_state_from_components(
            text_embeddings, dates, core_values
        )

    def build_state_vector_from_embeddings(
        self,
        embeddings: list[np.ndarray],
        dates: list[str],
        core_values: list[str],
    ) -> np.ndarray:
        """Build state vector from pre-computed embeddings.

        Use this when embeddings are cached to avoid redundant encoding.

        Args:
            embeddings: List of pre-computed embeddings, current entry first.
                       Length should be window_size.
            dates: List of dates corresponding to embeddings (YYYY-MM-DD format).
            core_values: List of Schwartz value names from persona profile.

        Returns:
            np.ndarray of shape (state_dim,) containing the full state vector.
        """
        return self._build_state_from_components(
            embeddings, dates, core_values
        )

    def _build_state_from_components(
        self,
        text_embeddings: list[np.ndarray],
        dates: list[str],
        core_values: list[str],
    ) -> np.ndarray:
        """Internal method to assemble state vector from components."""
        text_vector = np.concatenate(text_embeddings)

        # 2. Time gaps
        padded_dates = dates + [None] * (self.window_size - len(dates))
        time_gaps = self.compute_time_gaps(padded_dates[:self.window_size])

        # 3. Profile weights
        profile_weights = self.parse_core_values_to_weights(core_values)

        # Concatenate all components
        state_vector = np.concatenate([
            text_vector,
            time_gaps,
            profile_weights,
        ])

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
        parts = []

        if initial_entry:
            parts.append(initial_entry)

        if nudge_text:
            parts.append(f'Nudge: "{nudge_text}"')

        if response_text:
            parts.append(f"Response: {response_text}")

        return "\n\n".join(parts) if parts else ""
