# Claude Judge Labeling Instructions

Instructions for Claude Code to label synthetic journal data with Schwartz value alignment scores using parallel subagents.

**No API keys required** - this uses Claude Code's native Task tool, not external APIs.

**Prerequisites:** Run the Python wrangling script first to produce clean data from synthetic persona markdown files.

---

## Input

**Required:** Path to wrangled persona markdown files from Phase 1 (e.g., `logs/wrangled/2026-01-09_09-37-09/`)

---

## Prerequisites: Phase 1 Wrangling

Before running Judge labeling, you must wrangle the synthetic data:

```bash
source .venv/bin/activate
python -m src.wrangling.parse_synthetic_data logs/synthetic_data/<timestamp>
```

This creates clean `persona_*.md` files in `logs/wrangled/<timestamp>/` (one per persona) with generation metadata stripped, ready for judging.

---

## Source Files

| File | Contains |
|------|----------|
| `config/schwartz_values.yaml` | Value elaborations for rubric context |
| `prompts/judge_alignment.yaml` | Judge prompt template |
| `logs/wrangled/<timestamp>/persona_*.md` | Input: clean entry data (one file per persona) |

---

## Execution Architecture

### Per-Persona Subagents (Parallel)

Each persona is handled by a single subagent that scores all entries for that persona, then writes its results. All persona subagents run in parallel.

```
┌─────────────────────────────────────────────────────────────────────┐
│  MAIN ORCHESTRATOR (you)                                            │
│                                                                     │
│  0. Read wrangled persona_*.md files                                │
│  1. Create output directory: logs/judge_labels/<input-timestamp>/    │
│  2. Build value rubric context from schwartz_values.yaml            │
│                                                                     │
│  3. Launch all persona subagents with run_in_background=true        │
│     ┌────────────────┐ ┌────────────────┐ ┌────────────────┐       │
│     │ Judge Persona1 │ │ Judge Persona2 │ │ Judge Persona3 │       │
│     │ Score Entry 1  │ │ Score Entry 1  │ │ Score Entry 1  │       │
│     │ Score Entry 2  │ │ Score Entry 2  │ │ Score Entry 2  │       │
│     │ ...→Write JSON │ │ ...→Write JSON │ │ ...→Write JSON │       │
│     └───────┬────────┘ └───────┬────────┘ └───────┬────────┘       │
│             │                  │                  │                 │
│  4. Wait for all to complete with TaskOutput                        │
│             │                  │                  │                 │
│             ▼                  ▼                  ▼                 │
│  5. Consolidate JSON → Parquet, write config.md, report summary     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Execution Steps

### 1. Read Source Files

Read and internalize:
- `config/schwartz_values.yaml` for building value rubrics
- `prompts/judge_alignment.yaml` for the prompt template
- The wrangled markdown files from `logs/wrangled/<timestamp>/`

### 2. Create Output Directory

Extract the timestamp from the input wrangled folder path and use it for the output:

```bash
# If input is logs/wrangled/2026-01-09_09-37-09/, extract "2026-01-09_09-37-09"
mkdir -p logs/judge_labels/<input-timestamp>
```

This ensures traceability: `logs/wrangled/2026-01-09_09-37-09/` → `logs/judge_labels/2026-01-09_09-37-09/`

Store the directory path to pass to each subagent.

### 3. Build Value Rubric Context

Extract from `schwartz_values.yaml` for each of the 10 values:
- `core_motivation`: The fundamental drive
- `behavioral_manifestations`: First 3 behaviors (aligned examples)

Format as markdown for inclusion in the prompt:

```markdown
### Self-Direction
**Core Motivation:** The fundamental drive to think for oneself...

**Key Behaviors (Aligned):**
- Resists being told what to do...
- Seeks out problems that require novel solutions...
- Makes career or life choices that prioritize autonomy...

### Stimulation
...
```

### 4. Group Entries by Persona

Each wrangled `persona_*.md` file contains all entries for one persona. Each persona will be handled by one subagent.

### 5. Build Session Content

The wrangled markdown files use a minimal format — absence of sections is self-evident:

**Entry with nudge + response:**
```
[initial entry content]

**Nudge:** "[nudge_text]"

**Response:** [response_text]
```

**Entry with nudge, no response:**
```
[initial entry content]

**Nudge:** "[nudge_text]"
```

**Entry without nudge:**
```
[initial entry content]
```

### 6. Launch All Persona Subagents

Send **one message** with N Task tool calls, all with `run_in_background=true`:

```
Tool: Task
- description: "Judge Persona 1: Score entries"
- subagent_type: "general-purpose"
- run_in_background: true
- prompt: [Full persona judging prompt - see below]

Tool: Task
- description: "Judge Persona 2: Score entries"
- subagent_type: "general-purpose"
- run_in_background: true
- prompt: [Full persona judging prompt - see below]

... (all N personas in one message)
```

### 7. Persona Judge Subagent Prompt

Each subagent receives everything needed to score all entries for one persona.

**Input to subagent:**

1. **Persona profile**: name, age, profession, culture, core_values, bio
2. **Value rubric**: Built from schwartz_values.yaml
3. **All entries for this persona**: Each with session_content (pre-formatted)
4. **Output path**: Where to write JSON results
5. **Scoring instructions**: From prompts/judge_alignment.yaml template

**Subagent prompt structure:**

```
You are judging journal entries for alignment with Schwartz values.

## Persona Profile
- Name: [name]
- Age: [age]
- Profession: [profession]
- Culture: [culture]
- Core Values: [values]
- Bio: [bio]

## Schwartz Value Rubrics

[Built rubric context - all 10 values with core_motivation and behaviors]

## Scoring Instructions

For each entry, evaluate alignment with each of the 10 Schwartz value dimensions:
- **-1 (Misaligned)**: Entry actively conflicts with this value
- **0 (Neutral)**: Entry is irrelevant to this value
- **+1 (Aligned)**: Entry actively supports this value

**Trajectory context:** All entries are shown in chronological order. Use earlier entries to inform your understanding of later ones — a vague entry like "feeling better" gains meaning from preceding context. Score each entry based on its content, but let trajectory context resolve ambiguity.

Consider the entire session (initial entry + nudge + response) as a single unit.
Use the max-signal approach: if the response reveals alignment, score based on that.

## Entries to Score

### Entry 0 - [date]
[session_content]

### Entry 1 - [date]
[session_content]

...

## Output

For each entry, return scores as JSON. Then write results to:
[output_path]/persona_[id]_labels.json

Format:
{
  "persona_id": [id],
  "labels": [
    {
      "t_index": 0,
      "date": "[date]",
      "scores": {
        "self_direction": [int],
        "stimulation": [int],
        "hedonism": [int],
        "achievement": [int],
        "power": [int],
        "security": [int],
        "conformity": [int],
        "tradition": [int],
        "benevolence": [int],
        "universalism": [int]
      }
    },
    ...
  ]
}

After writing the file, return a summary:
"Scored [N] entries for [persona_name]. File: persona_[id]_labels.json"
```

### 8. Wait for All Subagents

Use TaskOutput to wait for each persona:

```
Tool: TaskOutput
- task_id: [persona 1 task ID]
- block: true
- timeout: 300000

Tool: TaskOutput
- task_id: [persona 2 task ID]
- block: true
- timeout: 300000

... (can poll multiple in parallel)
```

### 9. Consolidate Results

After all subagents complete, consolidate JSON files to Parquet **inline via Bash** (do NOT write a separate Python script to the output directory):

```bash
python3 -c "
import json
from pathlib import Path
import polars as pl

output_dir = Path('logs/judge_labels/<timestamp>')
json_files = sorted(output_dir.glob('persona_*_labels.json'))

all_rows = []
for jf in json_files:
    data = json.loads(jf.read_text())
    for label in data['labels']:
        s = label['scores']
        all_rows.append({
            'persona_id': data['persona_id'],
            't_index': label['t_index'],
            'date': label['date'],
            'alignment_vector': [s['self_direction'], s['stimulation'], s['hedonism'], s['achievement'], s['power'], s['security'], s['conformity'], s['tradition'], s['benevolence'], s['universalism']],
            'alignment_self_direction': s['self_direction'],
            'alignment_stimulation': s['stimulation'],
            'alignment_hedonism': s['hedonism'],
            'alignment_achievement': s['achievement'],
            'alignment_power': s['power'],
            'alignment_security': s['security'],
            'alignment_conformity': s['conformity'],
            'alignment_tradition': s['tradition'],
            'alignment_benevolence': s['benevolence'],
            'alignment_universalism': s['universalism'],
        })

pl.DataFrame(all_rows).write_parquet(output_dir / 'judge_labels.parquet')
print(f'Wrote {len(all_rows)} entries to judge_labels.parquet')
"
```

**Do NOT** write helper scripts (`.py` files) to the output directory. Only these files should exist:
- `persona_*_labels.json` (from subagents)
- `judge_labels.parquet` (consolidated)
- `config.md` and `validation_report.md` (metadata)

### 10. Write Config and Report

**config.md:**
```markdown
# Judge Labeling Run Configuration

**Timestamp**: [timestamp]
**Source**: [wrangled_data_path]
**Method**: Claude Code subagents (parallel)

## Parameters
- Personas: [N]
- Total entries: [N]
```

**validation_report.md:**
```markdown
# Judge Labeling Validation Report

## Run Summary
- Personas processed: X/X
- Entries labeled: X/X

## Score Distribution
| Value Dimension | -1 | 0 | +1 |
|-----------------|-----|-----|-----|
| Self-Direction  | X   | X   | X   |
| ...             | ... | ... | ... |

## Quality Flags
- All-zero entries: X (X%)
- Sparse personas (>80% neutral): X

## Sample Labels
[First 3 entries with scores for spot-checking]
```

### 11. Report Summary

Print:
- Personas processed: X/X
- Entries labeled: X
- Score distribution summary
- Output file location

---

## Output Schema

**judge_labels.parquet:**

```python
{
    "persona_id": int,
    "t_index": int,
    "date": str,
    "alignment_vector": list[int],  # [10 integers: {-1, 0, +1}]

    # Individual columns
    "alignment_self_direction": int,
    "alignment_stimulation": int,
    "alignment_hedonism": int,
    "alignment_achievement": int,
    "alignment_power": int,
    "alignment_security": int,
    "alignment_conformity": int,
    "alignment_tradition": int,
    "alignment_benevolence": int,
    "alignment_universalism": int,

    # Session metadata
    "has_nudge": bool,
    "has_response": bool,
}
```

---

## Checklist

- [ ] Run Phase 1 wrangling: `python -m src.wrangling.parse_synthetic_data <path>`
- [ ] Read wrangled persona_*.md files from logs/wrangled/<timestamp>/
- [ ] Read schwartz_values.yaml
- [ ] Create output directory using input folder's timestamp
- [ ] Build value rubric context
- [ ] Group entries by persona
- [ ] Build session content for each entry
- [ ] Launch all persona subagents with `run_in_background=true` in ONE message
  - Each subagent receives output directory path + persona data
- [ ] Wait for all subagents with TaskOutput
- [ ] Consolidate JSON → Parquet
- [ ] Write config.md and validation_report.md
- [ ] Report summary

**Subagent count:** N total (one per persona in wrangled directory)

**Example:** 5 persona files = 5 subagent calls (all launched in parallel in ONE message, each writes its own JSON)

---

## Example Invocation

User:
```
Run judge labeling on logs/synthetic_data/2026-01-09_09-37-09
```

Claude Code:
1. Checks for wrangled `persona_*.md` files in `logs/wrangled/<timestamp>/`
2. If not found, runs wrangling first
3. Reads markdown files, builds rubric, launches subagents
4. Consolidates results to `logs/judge_labels/<timestamp>/judge_labels.parquet`

---

## Limitations

- **Max 10 parallel personas**: Claude Code caps parallelism at 10; additional tasks queue automatically
- **Context per subagent**: Each subagent has separate context; pass all needed info
- **Timeout**: Set adequate timeout (5+ minutes) for personas with many entries
