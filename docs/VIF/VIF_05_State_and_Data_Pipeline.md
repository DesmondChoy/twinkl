# VIF – State Representation & Data Pipeline (POC Spec)

This document makes the **concrete choices** for the capstone POC about:

- How the VIF **state vector** $s_{u,t}$ is constructed.
- How the **training dataset** is derived from the synthetic journals and Judge labels.
- What minimal target option and feature set we will use for V1.

It complements:

- `VIF_01_Concepts_and_Roadmap.md` (high‑level design & tiers)
- `VIF_02_System_Architecture.md` (generic state options)
- `VIF_03_Model_Training.md` (reward modeling options)

---

## 1. POC Scope Decisions

For the capstone POC, we commit to the following **concrete scope**:

- **Modalities**: **Text only**.
  - Audio/prosodic signals (A3D) are out of scope for capstone; multimodal extensions deferred to future work.

- **Target option**: **Option A – Immediate alignment** (from `VIF_03_Model_Training.md`).
  - The Critic learns to emulate the Judge's per‑dimension alignment vector $\hat{\vec{a}}_{u,t}$.
  - Longer‑term trends (weekly averages, crash/rut) are derived via **deterministic smoothing and rules**, not via discounted returns.

- **Window size**: $N = 3$ recent entries per user.
  - At each time step $t$, the state includes the current entry and up to the last **two** previous entries.

- **Values**: $K = 10$ Schwartz dimensions.
  - Each Judge alignment component $\hat{a}^{(j)}_{u,t} \in \{-1, 0, +1\}$, for $j = 1, \dots, 10$.

- **Text encoder**: frozen sentence encoder (e.g. SBERT).
  - $\phi_{\text{text}}(T_{u,t}) \in \mathbb{R}^{d_e}$, where $d_e$ is the embedding dimension (e.g. 384 or 768 depending on the chosen model).

These choices keep the POC **implementable within a semester** while still honouring the trajectory‑aware, vector‑valued and uncertainty‑aware design.

---

## 2. Formal State Definition $s_{u,t}$

For a user (or synthetic persona) $u$ and time step / entry index $t$, the POC state is defined as:

$$
 s_{u,t} = \text{Concat}\Big[
   \underbrace{\phi_{\text{text}}(T_{u,t}), \ldots, \phi_{\text{text}}(T_{u,t-N+1})}_{\text{text window}},
   \underbrace{\Delta t_{u,t}, \ldots, \Delta t_{u,t-N+2}}_{\text{time gaps}},
   \underbrace{\text{history\_stats}_{u,t}}_{\text{per-dimension history}},
   \underbrace{z_u}_{\text{user profile}}
 \Big]
$$

Where:

- **Text window** (`text_window`)
  - $\phi_{\text{text}}(T_{u,t-k}) \in \mathbb{R}^{d_e}$ for $k = 0, 1, \dots, N-1$.
  - If $t - k < 0$ (no such previous entry), we use a **zero vector** and track this via a mask flag (see below).

- **Time gaps** (`time_gaps`)
  - $\Delta t_{u,t} = t_{u,t} - t_{u,t-1}$ in days (or hours), for the last $N-1$ transitions.
  - If $t - 1 < 0$, we use 0 and mask.

- **History stats** (`history_stats`)
  - Derived from past Judge labels (or optionally from past Critic predictions), **up to but not including** $t$.
  - For the POC minimal version:
    - **EMA per value dimension** over a fixed horizon $H$ (e.g. last 7 entries):

      $$
      \text{ema}^{(j)}_{u,t} = \alpha \hat{a}^{(j)}_{u,t-1} + (1 - \alpha) \, \text{ema}^{(j)}_{u,t-1}
      $$

      with $\text{ema}^{(j)}_{u,0} = 0$ and $\alpha$ chosen (e.g. 0.3).

    - Optionally (stretch): **rolling std dev** and **entry counts** in recent windows.

- **User profile** $z_u$
  - For the POC, $z_u$ is a simple concatenation of:
    - Value weight vector $w_u \in \mathbb{R}^K$, normalised so $w_{u,k} \ge 0$ and $\sum_k w_{u,k} = 1$.
    - (Optional stretch) A small embedding of persona metadata (age range, culture, profession) if desired.

### 2.1 Minimal Feature Set for V1

To keep the first implementation focused, we can start with this **minimal feature set**:

- **Text window**: 3 embeddings (current + last 2).
- **Time gaps**: 2 scalars (time since last entry, time between previous two entries).
- **History stats**: 1 EMA per value dimension (10 scalars total).
- **User profile**: 10‑dim value weight vector \(w_u\).

This yields a state dimension of approximately:

$$
 \text{dim}(s_{u,t}) \approx 3 d_e + 2 + 10 + 10 = 3 d_e + 22
$$

For example, with $d_e = 384$, $\text{dim}(s_{u,t}) \approx 3 \times 384 + 22 = 1{,}174$.

### 2.2 Handling Early Timesteps and Missing History

- For early entries where there is **no full window** (e.g. $t = 0$ or $t = 1$):
  - Use **zero embeddings** for missing past entries.
  - Use **0** for missing time gaps.
  - Compute EMA with the usual recurrence (starting at 0), so early EMAs remain near 0.
- Optionally, we can append binary flags such as:
  - `has_full_window` (1 if at least 3 entries exist, else 0).
  - `num_prev_entries` (scalar count of previous entries).

The Critic can learn to treat early states as lower‑confidence or less informative, and the downstream uncertainty mechanism (MC Dropout + OOD logic) further protects against overconfident early predictions.

---

## 3. Data Objects and Logical Tables

We describe the synthetic → training pipeline in terms of **logical tables**. These do not have to be actual SQL tables, but structuring them this way makes the pipeline auditable and explainable.

### 3.1 Persona

One row per synthetic persona.

- `persona_id`: integer or UUID.
- `profile_json`: JSON blob containing:
  - `core_values` (e.g. ["Benevolence", "Self-Direction"]).
  - `value_weights` (vector of length 10, mapping onto Schwartz values).
  - Demographic/context fields (age range, culture, profession, etc.).
  - Narrative description (bio) used in prompts.

This table is produced by the **persona generation** step in the synthetic pipeline.

### 3.2 Entry

One row per journal entry.

- `persona_id`
- `t_index`: integer index of the entry **within that persona** (0, 1, 2, ...).
- `timestamp`: synthetic date/time (e.g. ISO string) or days‑since‑start.
- `text`: raw journal content.

Entries are generated sequentially for each persona, with random gaps between dates.

### 3.3 JudgeLabel

One row per entry with Judge scores.

- `persona_id`
- `t_index`
- `alignment_vector`: length‑10 vector, each component in $\{-1, 0, +1\}$, e.g.:  
  `[Health: -1, Achievement: +1, Benevolence: 0, ...]`.

This table is produced by running the **Judge (LLM‑as‑Judge)** over each `Entry.text` (with access to persona profile where needed) as specified in `VIF_03_Model_Training.md`.

### 3.4 StateTargetSample (Training Rows)

This is the dataset actually used to train the Critic.

One row per `(persona_id, t_index)` where a Judge label exists.

- `persona_id`
- `t_index`
- `state_vector`: flattened float vector representing $s_{u,t}$ as defined above.
- `target_vector`: length‑10 float vector equal to `alignment_vector` (possibly normalised to \([-1, 1]\), but still preserving −1/0/+1 semantics).
- Optional metadata:
  - `num_prev_entries`
  - `has_full_window`
  - Which synthetic config bucket this persona came from (for stratified splits).

---

## 4. State Construction Procedure (Offline)

This section gives an implementation‑ready recipe to go from `Persona`, `Entry`, `JudgeLabel` to `StateTargetSample`.

### 4.1 Pre‑compute Text Embeddings

For each `Entry` row (each `(persona_id, t_index)`):

1. Feed `text` into the frozen text encoder:
   - $\mathbf{e}_{u,t} = \phi_{\text{text}}(T_{u,t}) \in \mathbb{R}^{d_e}$.
2. Store $\mathbf{e}_{u,t}$ (e.g. in memory, on disk, or as a separate table `EntryEmbedding`).

### 4.2 Compute Time Gaps

Within each `persona_id` group:

1. Sort entries by `timestamp` to ensure $t$ is in chronological order.
2. For each index $t$:
   - If $t = 0$:
     - $\Delta t_{u,0} = 0$.
   - If $t > 0$:
     - $\Delta t_{u,t} = \text{days\_between}(t_{u,t}, t_{u,t-1})$ (or hours, but be consistent).

Store these time deltas as part of an intermediate structure.

### 4.3 Compute History Statistics (EMA)

Within each `persona_id` group, for each value dimension \(j\):

1. Initialize $\text{ema}^{(j)}_{u,0} = 0$.
2. For $t = 1, 2, \dots$:

   $$
   \text{ema}^{(j)}_{u,t} = \alpha \hat{a}^{(j)}_{u,t-1} + (1 - \alpha) \, \text{ema}^{(j)}_{u,t-1}
   $$

3. Optionally, keep separate EMAs for different horizons (e.g. short vs medium term) as multiple features.

At time $t$, the history stats vector `history_stats_{u,t}` is:

$$
\text{history\_stats}_{u,t} = \big(\text{ema}^{(1)}_{u,t}, \dots, \text{ema}^{(K)}_{u,t}\big)
$$

### 4.4 Assemble the State Vector

For each `(persona_id, t_index)` where a Judge label exists:

1. **Gather text embeddings** for the window:
   - For $k = 0, 1, 2$:
     - If $t - k \ge 0$: use $\mathbf{e}_{u,t-k}$.
     - Else: use zero vector.
2. **Gather time gaps**:
   - Use $\Delta t_{u,t}$ and $\Delta t_{u,t-1}$ (if $t-1 < 0$, use 0).
3. **Fetch history stats** at $t$:
   - Use $\text{history\_stats}_{u,t}$ computed in 4.3.
4. **Build user profile vector** $z_u$:
   - Map persona’s `core_values` and `value_weights` (from `profile_json`) to a fixed 10‑dim vector aligned with the Schwartz ordering.
5. **Concatenate in a fixed order**:
   - `[e_t, e_{t-1}, e_{t-2}, delta_t, delta_t_minus_1, history_stats, w_u]`.

The result is a fixed‑length `state_vector` for this row.

6. **Set the target**:
   - `target_vector = alignment_vector` from `JudgeLabel` (cast to floats if using MSE loss).

7. Append `(persona_id, t_index, state_vector, target_vector, meta)` to `StateTargetSample`.

### 4.5 Train / Validation / Test Splits

To avoid leakage and make evaluation realistic:

- Split at the **persona level**, not at the entry level.
- Example:
  - 70% of personas → training set.
  - 15% of personas → validation set.
  - 15% of personas → test set.

All `(persona_id, t_index)` rows for a given persona belong to the same split.

---

## 5. Link Back to Model Training and Inference

- **Training** (`VIF_03_Model_Training.md`):
  - The Critic MLP is trained on `StateTargetSample` with a multi‑output regression loss (e.g. MSE) to predict \(\hat{\vec{a}}_{u,t}\).
  - MC Dropout is applied during both training and inference to enable epistemic uncertainty estimation.

- **Inference** (`VIF_02_System_Architecture.md`):
  - At runtime, the system builds \(s_{u,t}\) **incrementally** from recent entries and the user’s profile, using the same feature definition as above.
  - The Critic’s mean and variance outputs feed into weekly aggregation, crash/rut detection, and the Coach triggers.

By pinning down \(s_{u,t}\) and the data pipeline in this document, we make the VIF implementation:

- **Auditable**: reviewers can trace exactly which features feed the Critic.
- **Reproducible**: synthetic generation → Judge labels → Critic training is a fixed, documented pipeline.
- **Extensible**: future work (e.g. audio, more complex temporal models, per‑user adapters) can layer on top of this baseline without changing the conceptual spine.

---

## 6. Implementation Reference

The state construction and data pipeline are implemented in `src/vif/`. Key modules:

| Module | Implements |
|--------|------------|
| `src/vif/state_encoder.py` | `StateEncoder` class — Sections 2 & 4 (state vector construction) |
| `src/vif/dataset.py` | `VIFDataset`, `load_all_data()`, `split_by_persona()` — Sections 3 & 4.5 |
| `src/vif/encoders.py` | `SBERTEncoder` — Section 4.1 (text embeddings) |

### 6.1 Verification of Spec Compliance

The implementation matches this spec exactly:

| Spec Item | Spec Value | Implementation |
|-----------|------------|----------------|
| Window size | N = 3 | `StateEncoder(window_size=3)` ✓ |
| State dimension | 3×d_e + 22 = 1,174 | `state_encoder.state_dim` returns 1,174 ✓ |
| EMA α | 0.3 | `StateEncoder(ema_alpha=0.3)` ✓ |
| Text encoder | SBERT, d_e = 384 | `SBERTEncoder("all-MiniLM-L6-v2")` ✓ |
| Train/Val/Test split | 70/15/15 by persona | `split_by_persona()` ✓ |
| Zero-padding for early entries | Yes | Handled in `build_state_vector()` ✓ |

### 6.2 Data Flow

```
logs/wrangled/persona_*.md     →  load_entries()
logs/judge_labels/judge_labels.parquet  →  load_labels()
                                    ↓
                            merge_labels_and_entries()
                                    ↓
                            split_by_persona() (70/15/15)
                                    ↓
                            VIFDataset (caches embeddings)
                                    ↓
                            StateEncoder.build_state_vector()
                                    ↓
                            (state_vector, target_vector) pairs
```

### 6.3 Key Implementation Details

- **Entry text**: Concatenates `initial_entry + nudge_text + response_text` for richer signal
- **Core Values → Profile weights**: Maps persona's declared values to 10-dim normalized vector via `parse_core_values_to_weights()`
- **Embedding caching**: `VIFDataset` pre-computes all text embeddings at initialization to avoid redundant encoding during training