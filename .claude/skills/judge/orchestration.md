---
name: judge
description: Run the judge labeling pipeline. Checks registry for pending personas, auto-wrangles if needed, launches parallel scoring subagents, validates output, and consolidates to parquet.
allowed-tools: Bash(python*), Read, Task, TaskOutput, Glob
---

# Judge Labeling Pipeline

This skill orchestrates the complete judge labeling workflow, from wrangling to final consolidation.

## Workflow Overview

```
Registry Check → Auto-Wrangle (if needed) → Parallel Labeling → Validation → Consolidation
```

## Step 1: Check Registry Status

First, check the pipeline status to understand what work needs to be done:

```bash
python3 -c "from src.registry import get_status; import json; print(json.dumps(get_status(), indent=2))"
```

This returns counts for:
- `pending_wrangling`: Personas generated but not yet wrangled
- `pending_labeling`: Personas wrangled but not yet labeled

## Step 2: Auto-Wrangle (if pending_wrangling > 0)

If there are personas pending wrangling, run the wrangling script first:

```bash
python -m src.wrangling.parse_synthetic_data
```

This reads from `logs/synthetic_data/` and writes cleaned markdown to `logs/wrangled/`.

## Step 3: Label Pending Personas

For each persona pending labeling, launch a subagent with full annotation context.

### Get Pending Personas

```bash
python3 -c "from src.registry import get_pending; print(get_pending('labeled').select(['persona_id', 'name', 'core_values']).to_dicts())"
```

### Subagent Context

Each subagent receives:
1. **Persona file**: Read from `logs/wrangled/persona_{id}.md`
2. **Annotation guide**: Read `.claude/skills/judge/annotation_guide.md`
3. **Value rubric**: Read `.claude/skills/judge/rubric.md`

### Subagent Instructions

For each pending persona, launch a Task subagent with this prompt pattern:

```
You are a Judge labeling agent for the VIF (Value Identity Function) training pipeline.

## Your Task
Score each journal entry for alignment with the persona's declared core values.

## Persona Data
[Read and include the full contents of logs/wrangled/persona_{id}.md]

## Annotation Guidelines
[Read and include .claude/skills/judge/annotation_guide.md]

## Value Rubric
[Read and include .claude/skills/judge/rubric.md]

## Scoring Rules
For each entry, score all 10 Schwartz value dimensions:
- **+1 (Aligned)**: Entry actively supports this value
- **0 (Neutral)**: Entry is irrelevant to this value
- **-1 (Misaligned)**: Entry conflicts with this value

Most entries will have mostly 0s with 1-3 non-zero scores.

## Output Format
Write a JSON file to `logs/judge_labels/persona_{id}_labels.json`:

```json
{
  "persona_id": "{id}",
  "labels": [
    {
      "t_index": 0,
      "date": "YYYY-MM-DD",
      "scores": {
        "self_direction": 0,
        "stimulation": 0,
        "hedonism": 0,
        "achievement": 0,
        "power": 0,
        "security": 0,
        "conformity": 0,
        "tradition": 0,
        "benevolence": 0,
        "universalism": 0
      }
    }
  ]
}
```

## Important
- Score ONLY based on entry content, not persona metadata
- Consider nudge responses as part of the entry when present
- Write the JSON file directly, no intermediate output
```

### Parallel Execution

Launch all pending persona subagents in parallel using the Task tool. Use `run_in_background: true` for large batches.

## Step 4: Validate and Consolidate

After all subagents complete, run consolidation:

```bash
python -m src.judge.consolidate logs/judge_labels
```

This:
1. Validates JSON against Pydantic models (`src/models/judge.py`)
2. Merges all labels into `logs/judge_labels/judge_labels.parquet`
3. Updates registry with `stage_labeled=True` for each processed persona

## Step 5: Report Final Status

Show the final registry status and score distribution:

```bash
python3 -c "
from src.registry import get_status
import polars as pl
from pathlib import Path

# Registry status
status = get_status()
print('Registry Status:')
print(f'  Total personas: {status[\"total\"]}')
print(f'  Labeled: {status[\"labeled\"]}')
print(f'  Pending labeling: {status[\"pending_labeling\"]}')

# Score distribution if parquet exists
parquet_path = Path('logs/judge_labels/judge_labels.parquet')
if parquet_path.exists():
    df = pl.read_parquet(parquet_path)
    print(f'\nLabel Statistics:')
    print(f'  Total entries labeled: {len(df)}')
    print(f'  Personas: {df[\"persona_id\"].n_unique()}')
"
```

## Idempotency

This pipeline is idempotent:
- Running `/judge` when nothing is pending will simply report "nothing to do"
- Re-running after partial completion will only process remaining personas
- The registry tracks state, so no work is duplicated

## Error Handling

If a subagent fails:
1. The consolidation script will report validation errors
2. Re-run `/judge` to retry failed personas (they won't be marked as labeled)
3. Check `logs/judge_labels/` for malformed JSON files

## Files

| File | Purpose |
|------|---------|
| `logs/synthetic_data/persona_*.md` | Raw generated personas |
| `logs/wrangled/persona_*.md` | Cleaned personas for labeling |
| `logs/judge_labels/persona_*_labels.json` | Individual label files |
| `logs/judge_labels/judge_labels.parquet` | Consolidated labels |
| `logs/registry/personas.parquet` | Central pipeline registry |
