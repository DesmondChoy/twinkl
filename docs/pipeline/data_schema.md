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
| `age` | `str` | Age label stored as a string using the standard brackets: `"18-24"`, `"25-34"`, `"35-44"`, `"45-54"`, `"55+"` |
| `profession` | `str` | Occupation (e.g., `"Teacher"`, `"Gig Worker"`, `"Entrepreneur"`) |
| `culture` | `str` | Cultural background (e.g., `"North American"`, `"East Asian"`) |
| `core_values` | `list[str]` | Declared Schwartz values (1-2 values per persona) |
| `entry_count` | `i64` | Number of journal entries generated |
| `created_at` | `datetime[μs, UTC]` | Generation timestamp |
| `stage_synthetic` | `bool` | `true` after synthetic journal generation |
| `stage_wrangled` | `bool` | `true` after data cleaning/parsing |
| `stage_labeled` | `bool` | `true` after judge labeling |
| `nudge_enabled` | `bool` | Whether nudges were enabled during generation |
| `annotation_order` | `i64 \| null` | Display order in annotation tool (1-indexed); null until set by annotation workflow |

### Randomized Attributes

These columns capture the randomized selections from `config/synthetic_data.yaml`:

- **age**: Sampled from 5 standard brackets
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
| `rationales_json` | `str \| null` | JSON object mapping Schwartz value names to free-text rationale strings; null if no rationales provided |

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

Based on the current committed corpus:

### Corpus Snapshot

- 192 personas in `logs/registry/personas.parquet`
- 1,555 labeled journal entries in `logs/judge_labels/judge_labels.parquet`
- Average of 8.1 entries per persona (range: 2-12)
- 280 total core-value assignments across personas
- 957 entries with generated nudges in `logs/synthetic_data/` (61.5% of all entries)

### Overall Label Balance

Across all 15,550 per-dimension labels:

| Label | Count | % |
|-------|-------|---|
| `-1` | 1,122 | 7.2% |
| `0` | 11,720 | 75.4% |
| `+1` | 2,708 | 17.4% |

### Per-Dimension Label Balance

| Value | `-1` | `0` | `+1` |
|-------|------|-----|------|
| Self-Direction | 214 (13.8%) | 941 (60.5%) | 400 (25.7%) |
| Stimulation | 55 (3.5%) | 1,373 (88.3%) | 127 (8.2%) |
| Hedonism | 118 (7.6%) | 1,279 (82.3%) | 158 (10.2%) |
| Achievement | 84 (5.4%) | 1,059 (68.1%) | 412 (26.5%) |
| Power | 146 (9.4%) | 1,242 (79.9%) | 167 (10.7%) |
| Security | 143 (9.2%) | 1,148 (73.8%) | 264 (17.0%) |
| Conformity | 127 (8.2%) | 1,098 (70.6%) | 330 (21.2%) |
| Tradition | 58 (3.7%) | 1,242 (79.9%) | 255 (16.4%) |
| Benevolence | 121 (7.8%) | 998 (64.2%) | 436 (28.0%) |
| Universalism | 56 (3.6%) | 1,340 (86.2%) | 159 (10.2%) |

### Sparsity

- 1,509 entries (97.0%) have at least one non-zero label
- 46 entries (3.0%) are all-zero across all 10 values
- Mean non-zero dimensions per entry: 2.46
- Median non-zero dimensions per entry: 2

### Demographic Notes

- Registry coverage currently spans 6 cultures and 9 professions
- Registry ages use the standard 5 brackets
- Some LLM-generated names may repeat across different personas

---

## Related Files

| File | Purpose |
|------|---------|
| `src/registry/personas.py` | Registry CRUD operations with file locking |
| `src/judge/consolidate.py` | Merges JSON labels → parquet |
| `src/models/judge.py` | Pydantic validation models, `SCHWARTZ_VALUE_ORDER` |
| `config/synthetic_data.yaml` | Randomization pools for persona attributes |
