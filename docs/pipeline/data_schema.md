# Data Schema Reference

This document describes the parquet files produced by the synthetic-data,
judge-labeling, and weekly Coach runtime paths, including schemas,
relationships, and example queries.

## File Locations

| File | Purpose |
|------|---------|
| `logs/registry/personas.parquet` | Central registry tracking personas through pipeline stages |
| `logs/judge_labels/judge_labels.parquet` | Training labels for the VIF (all entries, all personas) |
| `logs/judge_labels/consensus_labels.parquet` | Confidence-tiered 5-pass consensus label surface for stability analysis and diagnostic retrains |
| `logs/exports/weekly_digests/weekly_digests.parquet` | Consolidated weekly Coach digest records (created on first runtime/digest run) |

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

## Consensus Labels

**Path:** `logs/judge_labels/consensus_labels.parquet`

This parquet keeps the same row key and alignment columns as
`judge_labels.parquet`, then adds 5-pass consensus metadata per dimension.

### Additional Columns

For each Schwartz dimension, the file includes three extra column families:

- `confidence_<dimension>`
  - confidence tier for the consensus label
  - values come from the 5-pass voting result and include tiers such as
    `bare_majority`, `strong`, `unanimous`, and `no_majority`
- `consensus_agreement_<dimension>`
  - integer agreement count from the five consensus passes
- `label_changed_<dimension>`
  - boolean flag indicating whether the consensus label differs from the
    persisted label in `judge_labels.parquet`

### What It Is Used For

- full-corpus stability analysis
- confidence-tiered label inspection
- diagnostic retrains that explicitly point `data.labels_path` at this parquet

The active corrected-split frontier still treats
`logs/judge_labels/judge_labels.parquet` as the default label surface. The
consensus parquet is a diagnostics and audit artifact.

---

## Weekly Coach Digest Records

**Path:** `logs/exports/weekly_digests/weekly_digests.parquet`

This file is created by `src.coach.weekly_digest` and `src.coach.runtime` when
you run the weekly Coach flow. Each row is one `(persona_id, week_start,
week_end)` digest record.

### Schema

| Column | Type | Description |
|--------|------|-------------|
| `persona_id` | `str` | Persona ID |
| `week_start` | `str` | Inclusive week start in `YYYY-MM-DD` |
| `week_end` | `str` | Inclusive week end in `YYYY-MM-DD` |
| `persona_name` | `str` | Persona display name |
| `response_mode` | `str` | Coach response mode used for the digest |
| `mode_source` | `str` | Whether the mode came from upstream drift or local fallback logic |
| `mode_rationale` | `str` | Short rationale for the selected mode |
| `signal_source` | `str` | Numeric source for the digest (`judge_labels` or `vif_runtime`) |
| `n_entries` | `i64` | Number of entries included in the week |
| `overall_mean` | `f64` | Profile-weighted overall weekly alignment |
| `overall_uncertainty` | `f64` | Profile-weighted weekly uncertainty |
| `core_values_json` | `str` | JSON array of declared core values |
| `drift_reasons_json` | `str` | JSON array of drift-routing reasons |
| `top_tensions_json` | `str` | JSON array of ranked tension dimensions |
| `top_strengths_json` | `str` | JSON array of ranked strength dimensions |
| `dimensions_json` | `str` | JSON array of per-dimension weekly summaries |
| `evidence_json` | `str` | JSON array of representative evidence snippets |
| `journal_history_json` | `str` | JSON array of history entries capped at `week_end` |
| `coach_narrative_json` | `str \| null` | JSON Coach narrative payload if generation ran |
| `validation_json` | `str \| null` | JSON narrative-validation payload if validation ran |

### Related Runtime Artifacts

The weekly Coach flow and demo review UI also write per-run artifacts beside the
consolidated parquet:

- `vif_timeline.parquet` or `<persona_id>_vif_timeline.parquet`
- `vif_weekly.parquet` or `<persona_id>_vif_weekly.parquet`
- `*.drift.json`
- `*.json`
- `*.md`
- `*.prompt.txt`

The demo review UI stores those bundles under:

```text
logs/exports/demo_tool_runs/<persona_id>/<checkpoint-stem>-<hash>/
```

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

### Find low-confidence consensus rows for one value

```python
import polars as pl

consensus = pl.read_parquet("logs/judge_labels/consensus_labels.parquet")

security_rows = consensus.filter(
    pl.col("confidence_security").is_in(["bare_majority", "no_majority"])
)
```

### Find rows where consensus changed the persisted label

```python
import polars as pl

consensus = pl.read_parquet("logs/judge_labels/consensus_labels.parquet")

changed_hedonism = consensus.filter(pl.col("label_changed_hedonism"))
```

---

## Known Data Characteristics

Based on the current committed corpus:

### Corpus Snapshot

- 204 personas in `logs/registry/personas.parquet`
- 1,651 labeled journal entries in `logs/judge_labels/judge_labels.parquet`
- Average of 8.1 entries per persona (range: 2-12)
- 292 total core-value assignments across personas
- 1,028 entries with generated nudges in `logs/wrangled/` (62.3% of all entries)

### Overall Label Balance

Across all 16,510 per-dimension labels:

| Label | Count | % |
|-------|-------|---|
| `-1` | 1,165 | 7.1% |
| `0` | 12,535 | 75.9% |
| `+1` | 2,810 | 17.0% |

### Per-Dimension Label Balance

| Value | `-1` | `0` | `+1` |
|-------|------|-----|------|
| Self-Direction | 216 (13.1%) | 1,019 (61.7%) | 416 (25.2%) |
| Stimulation | 60 (3.6%) | 1,464 (88.7%) | 127 (7.7%) |
| Hedonism | 141 (8.5%) | 1,337 (81.0%) | 173 (10.5%) |
| Achievement | 85 (5.1%) | 1,130 (68.4%) | 436 (26.4%) |
| Power | 146 (8.8%) | 1,338 (81.0%) | 167 (10.1%) |
| Security | 151 (9.1%) | 1,212 (73.4%) | 288 (17.4%) |
| Conformity | 129 (7.8%) | 1,189 (72.0%) | 333 (20.2%) |
| Tradition | 58 (3.5%) | 1,337 (81.0%) | 256 (15.5%) |
| Benevolence | 123 (7.5%) | 1,073 (65.0%) | 455 (27.6%) |
| Universalism | 56 (3.4%) | 1,436 (87.0%) | 159 (9.6%) |

### Sparsity

- 1,594 entries (96.5%) have at least one non-zero label
- 57 entries (3.5%) are all-zero across all 10 values
- Mean non-zero dimensions per entry: 2.41
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
