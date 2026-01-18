# Data Schema Reference

This document describes the parquet files produced by the synthetic data and judge labeling pipelines, including schemas, relationships, and example queries.

## File Locations

| File | Purpose |
|------|---------|
| `logs/registry/personas.parquet` | Central registry tracking personas through pipeline stages |
| `logs/judge_labels/judge_labels.parquet` | Training labels for the VIF (all entries, all personas) |

---

## Persona Registry

**Path:** `logs/registry/personas.parquet`

Tracks each synthetic persona and their progress through the data pipeline.

### Schema

| Column | Type | Description |
|--------|------|-------------|
| `persona_id` | `str` | 8-character hex UUID (e.g., `"a3f8b2c1"`) |
| `name` | `str` | LLM-generated full name |
| `age` | `str` | Age bracket: `"18-24"`, `"25-34"`, `"35-44"`, `"45-54"`, `"55+"` |
| `profession` | `str` | Occupation (e.g., `"Teacher"`, `"Gig Worker"`, `"Entrepreneur"`) |
| `culture` | `str` | Cultural background (e.g., `"North American"`, `"East Asian"`) |
| `core_values` | `list[str]` | Declared Schwartz values (1-2 values per persona) |
| `entry_count` | `i64` | Number of journal entries generated |
| `created_at` | `datetime[μs, UTC]` | Generation timestamp |
| `stage_synthetic` | `bool` | `true` after synthetic journal generation |
| `stage_wrangled` | `bool` | `true` after data cleaning/parsing |
| `stage_labeled` | `bool` | `true` after judge labeling |
| `nudge_enabled` | `bool` | Whether nudges were enabled during generation |

### Randomized Attributes

These columns capture the randomized selections from `config/synthetic_data.yaml`:

- **age**: Sampled from 5 brackets
- **profession**: Sampled from ~9 occupations
- **culture**: Sampled from 6 cultural backgrounds
- **core_values**: Random subset (1-2) of 10 Schwartz values

### What's NOT Recorded

Generation-time metadata stripped during wrangling (not available in production):
- `tone` (introspective, venting, grateful, etc.)
- `verbosity` (terse, moderate, detailed)
- `reflection_mode` (spontaneous, prompted, etc.)

---

## Judge Labels

**Path:** `logs/judge_labels/judge_labels.parquet`

Training labels for the VIF. Each row represents one journal entry with its alignment scores across all 10 Schwartz value dimensions.

### Schema

| Column | Type | Description |
|--------|------|-------------|
| `persona_id` | `str` | Links to registry (foreign key) |
| `t_index` | `i64` | 0-based entry index within persona's journal |
| `date` | `str` | Entry date in `YYYY-MM-DD` format |
| `alignment_vector` | `list[i64]` | Full 10-element score vector (see order below) |
| `alignment_self_direction` | `i64` | Score: `-1`, `0`, or `+1` |
| `alignment_stimulation` | `i64` | Score: `-1`, `0`, or `+1` |
| `alignment_hedonism` | `i64` | Score: `-1`, `0`, or `+1` |
| `alignment_achievement` | `i64` | Score: `-1`, `0`, or `+1` |
| `alignment_power` | `i64` | Score: `-1`, `0`, or `+1` |
| `alignment_security` | `i64` | Score: `-1`, `0`, or `+1` |
| `alignment_conformity` | `i64` | Score: `-1`, `0`, or `+1` |
| `alignment_tradition` | `i64` | Score: `-1`, `0`, or `+1` |
| `alignment_benevolence` | `i64` | Score: `-1`, `0`, or `+1` |
| `alignment_universalism` | `i64` | Score: `-1`, `0`, or `+1` |

### Score Meanings

| Score | Meaning | Example |
|-------|---------|---------|
| `+1` | **Aligned** — Entry actively supports this value | Prioritizing family time → Benevolence +1 |
| `0` | **Neutral** — Entry is irrelevant to this value | Discussing weather → most values 0 |
| `-1` | **Misaligned** — Entry conflicts with this value | Skipping exercise for work → Security -1 |

### Vector Order

The `alignment_vector` column stores scores in this fixed order:
```
[self_direction, stimulation, hedonism, achievement, power,
 security, conformity, tradition, benevolence, universalism]
```

This order matches `src/models/judge.py:SCHWARTZ_VALUE_ORDER`.

---

## Example Queries

### Basic: Load and inspect

```python
import polars as pl

# Load both files
registry = pl.read_parquet("logs/registry/personas.parquet")
labels = pl.read_parquet("logs/judge_labels/judge_labels.parquet")

print(f"Personas: {len(registry)}")
print(f"Labeled entries: {len(labels)}")
```

### Join labels with persona metadata

```python
# Add persona demographics to each label row
enriched = labels.join(
    registry.select(["persona_id", "name", "age", "culture", "profession", "core_values"]),
    on="persona_id",
    how="left"
)
```

### Label distribution per Schwartz value

```python
value_cols = [c for c in labels.columns if c.startswith("alignment_") and "vector" not in c]

for col in value_cols:
    counts = labels.group_by(col).len().sort(col)
    print(f"{col}: {counts.to_dicts()}")
```

### Find entries with strong signals (multiple non-zero scores)

```python
# Count non-zero values per entry
labels_with_signal = labels.with_columns(
    pl.sum_horizontal([
        (pl.col(c) != 0).cast(pl.Int64)
        for c in value_cols
    ]).alias("signal_count")
)

# Entries with 3+ values expressed
high_signal = labels_with_signal.filter(pl.col("signal_count") >= 3)
print(f"High-signal entries: {len(high_signal)} / {len(labels)}")
```

### Check if personas express their declared values

```python
# Explode core_values to one row per value
persona_values = registry.select(["persona_id", "core_values"]).explode("core_values")

# Map value names to column names
value_to_col = {
    "Self-Direction": "alignment_self_direction",
    "Stimulation": "alignment_stimulation",
    "Hedonism": "alignment_hedonism",
    "Achievement": "alignment_achievement",
    "Power": "alignment_power",
    "Security": "alignment_security",
    "Conformity": "alignment_conformity",
    "Tradition": "alignment_tradition",
    "Benevolence": "alignment_benevolence",
    "Universalism": "alignment_universalism",
}

# For each persona, check alignment with their declared values
# (This requires pivoting/aggregation logic specific to your analysis)
```

### Aggregate scores per persona (trajectory summary)

```python
# Mean alignment per persona across all their entries
persona_means = labels.group_by("persona_id").agg([
    pl.col(c).mean().alias(f"{c}_mean") for c in value_cols
])
```

---

## Known Data Characteristics

Based on analysis of the current dataset:

### Class Imbalance

Most entries are neutral (0) for most values. This is realistic but creates training challenges:

| Value | Typical distribution |
|-------|---------------------|
| Benevolence | Most expressed (+1 common) |
| Self-Direction | Well-represented |
| Power, Stimulation | Rare (few +1 examples) |

### Sparsity

- Most entries have 1-3 non-zero scores
- Entries rarely touch all 10 values
- The VIF must handle partial observability

### Demographic Notes

- Check for cultural/profession imbalances in your dataset
- Some LLM-generated names may repeat across different personas

---

## Related Files

| File | Purpose |
|------|---------|
| `src/registry/personas.py` | Registry CRUD operations with file locking |
| `src/judge/consolidate.py` | Merges JSON labels → parquet |
| `src/models/judge.py` | Pydantic validation models, `SCHWARTZ_VALUE_ORDER` |
| `config/synthetic_data.yaml` | Randomization pools for persona attributes |
