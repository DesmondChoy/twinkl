---
description: Run the judge labeling pipeline (wrangle → label → consolidate)
allowed-tools: Bash(python*), Read, Task, TaskOutput, Glob, Write
---

# Judge Labeling Pipeline

Run the complete judge labeling workflow. This skill:
1. Checks the registry for pending work
2. Auto-wrangles personas that need it
3. Launches parallel subagents to label each persona
4. Validates and consolidates results to parquet

## Quick Reference

Full orchestration details are in `.claude/skills/judge/orchestration.md`.

Supplementary files for subagents:
- `.claude/skills/judge/annotation_guide.md` — Scorability heuristics and calibration examples
- `.claude/skills/judge/rubric.md` — Schwartz value reference for scoring

## Execution Steps

### 1. Check Registry Status

```bash
python3 -c "from src.registry import get_status; import json; print(json.dumps(get_status(), indent=2))"
```

### 2. Auto-Wrangle (if pending_wrangling > 0)

```bash
python -m src.wrangling.parse_synthetic_data
```

### 3. Get Pending Personas for Labeling

```bash
python3 -c "from src.registry import get_pending; print(get_pending('labeled').select(['persona_id', 'name', 'core_values']).to_dicts())"
```

### 4. Launch Subagents

For each pending persona, launch a Task subagent with:
- Read the wrangled persona file from `logs/wrangled/persona_{id}.md`
- Read the annotation guide from `.claude/skills/judge/annotation_guide.md`
- Read the value rubric from `.claude/skills/judge/rubric.md`
- Score all entries using the guidelines
- Write output to `logs/judge_labels/persona_{id}_labels.json`

Use parallel subagents for efficiency. Ensure `logs/judge_labels/` directory exists first.

### 5. Consolidate Results

```bash
python -m src.judge.consolidate logs/judge_labels
```

### 6. Report Final Status

```bash
python3 -c "from src.registry import get_status; s=get_status(); print(f'Labeled: {s[\"labeled\"]}/{s[\"total\"]} | Pending: {s[\"pending_labeling\"]}')"
```

## Idempotency

Running `/judge` multiple times is safe — it only processes pending work and skips already-labeled personas.
