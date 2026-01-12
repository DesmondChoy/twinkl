# Claude Judge Labeling Instructions

Instructions for Claude Code to label synthetic journal data with Schwartz value alignment scores using parallel subagents.

**No API keys required** - this uses Claude Code's native Task tool, not external APIs.

**Prerequisites:** Run the Python wrangling script first to produce clean data from synthetic persona markdown files.

---

## Configuration Variables

**Change these values to customize the labeling run:**

| Variable | Default | Description |
|----------|---------|-------------|
| `WRANGLED_DATA_PATH` | (required) | Path to wrangled_entries.parquet from Phase 1 |
| `INCLUDE_PREVIOUS_ENTRIES` | true | Pass last N entries as context to Judge |
| `NUM_PREVIOUS_ENTRIES` | 2 | How many previous entries to include |

---

## Prerequisites: Phase 1 Wrangling

Before running Judge labeling, you must wrangle the synthetic data:

```bash
source .venv/bin/activate
python -m src.wrangling.parse_synthetic_data logs/synthetic_data/<timestamp>
```

This creates `wrangled_entries.parquet` in the synthetic data directory with clean, structured data ready for judging.

---

## Source Files

| File | Contains |
|------|----------|
| `config/schwartz_values.yaml` | Value elaborations for rubric context |
| `prompts/judge_alignment.yaml` | Judge prompt template |
| `logs/synthetic_data/<timestamp>/wrangled_entries.parquet` | Input: clean entry data |

---

## Execution Architecture

### Per-Persona Subagents (Parallel)

Each persona is handled by a single subagent that scores all entries for that persona, then writes its results. All persona subagents run in parallel.

```
┌─────────────────────────────────────────────────────────────────────┐
│  MAIN ORCHESTRATOR (you)                                            │
│                                                                     │
│  0. Read wrangled_entries.parquet                                   │
│  1. Create output directory: logs/judge_labels/YYYY-MM-DD_HH-MM-SS/ │
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
- The wrangled Parquet file specified by user

### 2. Create Output Directory

```bash
mkdir -p logs/judge_labels/$(date +%Y-%m-%d_%H-%M-%S)
```

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

From the Parquet file, group entries by `persona_id`. Each persona will be handled by one subagent.

### 5. Build Session Content

For each entry, combine the components into a single scorable unit:

**Entry with nudge + response:**
```
**Initial Entry:**
[initial_entry content]

**Nudge:**
"[nudge_text]"

**Response:**
[response_text content]
```

**Entry with nudge, no response:**
```
**Initial Entry:**
[initial_entry content]

**Nudge:**
"[nudge_text]"

*(Persona did not respond)*
```

**Entry without nudge:**
```
**Initial Entry:**
[initial_entry content]
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

After all subagents complete:

1. Read all `persona_*_labels.json` files from output directory
2. Merge into single Polars DataFrame
3. Add derived columns:
   - `alignment_vector`: List of 10 scores
   - Individual `alignment_*` columns for each value
   - `has_nudge`, `has_response` from original data
4. Write `judge_labels.parquet`

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
- Previous entries context: [true/false]
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
- [ ] Read wrangled_entries.parquet
- [ ] Read schwartz_values.yaml
- [ ] Create output directory
- [ ] Build value rubric context
- [ ] Group entries by persona
- [ ] Build session content for each entry
- [ ] Launch all persona subagents with `run_in_background=true` in ONE message
- [ ] Wait for all subagents with TaskOutput
- [ ] Consolidate JSON → Parquet
- [ ] Write config.md and validation_report.md
- [ ] Report summary

---

## Example Invocation

User:
```
Run judge labeling on logs/synthetic_data/2026-01-09_09-37-09
```

Claude Code:
1. Checks for `wrangled_entries.parquet` in that directory
2. If not found, runs wrangling first
3. Reads Parquet, builds rubric, launches subagents
4. Consolidates results to `logs/judge_labels/<timestamp>/judge_labels.parquet`

---

## Limitations

- **Max 10 parallel personas**: Claude Code caps parallelism; additional tasks queue
- **Context per subagent**: Each subagent has separate context; pass all needed info
- **Timeout**: Set adequate timeout (5+ minutes) for personas with many entries
