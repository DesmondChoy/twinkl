# Claude Judge Labeling Instructions

Instructions for Claude Code to label synthetic journal data with Schwartz value alignment scores using parallel subagents.

> **Status:** Reference/spec document (historical workflow context).
> **Canonical execution path:** Run `/judge` using `.claude/commands/judge.md` and `.claude/skills/judge/orchestration.md`.
> If this document conflicts with the `.claude` judge command/skill files, treat the `.claude` files as authoritative.

**No API keys required** - this uses Claude Code's native Task tool, not external APIs.

**Prerequisites:** Run the Python wrangling script first to produce clean data from synthetic persona markdown files.

---

## Input

**Required:** Wrangled persona markdown files in `logs/wrangled/` (flat directory with UUID-based filenames)

---

## Prerequisites: Phase 1 Wrangling

Before running Judge labeling, you must wrangle the synthetic data:

```bash
source .venv/bin/activate
python -m src.wrangling.parse_synthetic_data
```

This creates clean `persona_*.md` files in `logs/wrangled/` (flat directory) with generation metadata stripped, ready for judging.

---

## Output Structure

**Flat directory with UUID-based filenames** (matches input structure):

```
logs/
├── registry/
│   └── personas.parquet              # Central tracking (stage_labeled updated here)
│
├── wrangled/
│   ├── persona_a3f8b2c1.md           # Input: clean persona data
│   ├── persona_e7d4f9a2.md
│   └── ...
│
└── judge_labels/
    ├── persona_a3f8b2c1_labels.json  # Output: one JSON per persona
    ├── persona_e7d4f9a2_labels.json
    └── judge_labels.parquet          # Consolidated labels
```

---

## Source Files

| File | Contains |
|------|----------|
| `config/schwartz_values.yaml` | Value elaborations for rubric context |
| `prompts/judge_alignment.yaml` | Judge prompt template |
| `src/models/judge.py` | Pydantic models for output validation |
| `src/judge/consolidate.py` | Consolidation script (JSON → Parquet + registry update) |
| `logs/wrangled/persona_*.md` | Input: clean entry data (one file per persona) |

---

## Execution Architecture

### Per-Persona Subagents (Parallel)

Each persona is handled by a single subagent that:
1. Extracts the UUID from the wrangled filename
2. Scores all entries for that persona
3. Writes results with UUID in filename

All persona subagents run in parallel.

```
┌─────────────────────────────────────────────────────────────────────┐
│  MAIN ORCHESTRATOR (you)                                            │
│                                                                     │
│  0. Read wrangled persona_*.md files from logs/wrangled/            │
│  1. Ensure output directory exists: logs/judge_labels/              │
│  2. Build value rubric context from schwartz_values.yaml            │
│                                                                     │
│  3. Launch all persona subagents with run_in_background=true        │
│     ┌────────────────┐ ┌────────────────┐ ┌────────────────┐       │
│     │ Judge a3f8b2c1 │ │ Judge e7d4f9a2 │ │ Judge c1b5e8d3 │       │
│     │ Score Entry 0  │ │ Score Entry 0  │ │ Score Entry 0  │       │
│     │ Score Entry 1  │ │ Score Entry 1  │ │ Score Entry 1  │       │
│     │ ...→Write JSON │ │ ...→Write JSON │ │ ...→Write JSON │       │
│     └───────┬────────┘ └───────┬────────┘ └───────┬────────┘       │
│             │                  │                  │                 │
│  4. Wait for all to complete with TaskOutput                        │
│             │                  │                  │                 │
│             ▼                  ▼                  ▼                 │
│  5. Consolidate JSON → Parquet, update registry, report summary     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Execution Steps

### 1. Read Source Files

Read and internalize:
- `config/schwartz_values.yaml` for building value rubrics
- `prompts/judge_alignment.yaml` for the prompt template
- The wrangled markdown files from `logs/wrangled/`

### 2. Ensure Output Directory Exists

Create the output directory if it doesn't exist:

```bash
mkdir -p logs/judge_labels
```

**Note:** No timestamp subfolders — all label files go in `logs/judge_labels/` with UUID-based filenames.

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
- description: "Judge a3f8b2c1: Score entries"
- subagent_type: "general-purpose"
- run_in_background: true
- prompt: [Full persona judging prompt - see below]

Tool: Task
- description: "Judge e7d4f9a2: Score entries"
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

## Output & Validation

After scoring all entries, follow this phased workflow:

### Step 1: Build JSON Output
Build the output dictionary with this exact schema (note: persona_id is a STRING, not an integer):
{
  "persona_id": "a3f8b2c1",
  "labels": [
    {
      "t_index": 0,
      "date": "[date]",
      "scores": {
        "self_direction": [int: -1, 0, or 1],
        "stimulation": [int: -1, 0, or 1],
        "hedonism": [int: -1, 0, or 1],
        "achievement": [int: -1, 0, or 1],
        "power": [int: -1, 0, or 1],
        "security": [int: -1, 0, or 1],
        "conformity": [int: -1, 0, or 1],
        "tradition": [int: -1, 0, or 1],
        "benevolence": [int: -1, 0, or 1],
        "universalism": [int: -1, 0, or 1]
      },
      "rationales": {
        "<value_name>": "<explanation>",
        ...only for non-zero scores...
      }
    },
    ...
  ]
}

**Rationale requirements:**
- Include rationales ONLY for non-zero scores (-1 or +1)
- Quote or reference specific content from the entry (use quotation marks)
- Explain the connection between behavior and value dimension
- Be concise (1-2 sentences)
- Do NOT include the value name in the rationale (avoid circular reasoning like "shows achievement behavior")

**Example entry label:**
```json
{
  "t_index": 0,
  "date": "2025-12-01",
  "scores": {
    "self_direction": 1, "stimulation": 0, "hedonism": 1,
    "achievement": -1, "power": 0, "security": 0,
    "conformity": 0, "tradition": 0, "benevolence": 0, "universalism": 0
  },
  "rationales": {
    "self_direction": "Deliberately turned down department position to preserve autonomy.",
    "hedonism": "Explicitly values sensory pleasures (mint tea, sunset colors) as 'the whole point'.",
    "achievement": "Actively rejects competitive orientation, stating 'not looking for excitement from my job'."
  }
}
```

### Step 2: Validate Before Writing
Run Python validation using the Pydantic models:

```bash
python3 -c "
import json
from src.models.judge import PersonaLabels

data = '''<paste your JSON here>'''
PersonaLabels.model_validate(json.loads(data))
print('Validation passed')
"
```

### Step 3: Handle Validation Errors
If validation fails:
- Read the error message to identify the issue (common: out-of-range scores like 2 or -2, missing fields, persona_id not a string)
- Fix the JSON structure
- Re-run validation
- Maximum 2 retry attempts

### Step 4: Write File
Only after validation passes, write to:
logs/judge_labels/persona_[uuid]_labels.json

### Step 5: Verify File
Confirm the file was written correctly:

```bash
python3 -c "
from src.models.judge import PersonaLabels
import json
with open('logs/judge_labels/persona_a3f8b2c1_labels.json') as f:
    PersonaLabels.model_validate(json.load(f))
print('File verified')
"
```

Return: "Scored [N] entries for [persona_name]. Validated and saved to persona_[uuid]_labels.json"
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

After all subagents complete, run the consolidation module to validate and merge JSON files:

```bash
python -m src.judge.consolidate
```

This module:
1. Reads all `persona_*_labels.json` files from `logs/judge_labels/`
2. Validates each file against Pydantic models (safety net in case subagent validation missed something)
3. Reports any validation errors with clear file + error messages
4. Merges valid data into `judge_labels.parquet`
5. **Updates registry**: marks each validated persona as `stage_labeled=true`

**Output files** in `logs/judge_labels/`:
- `persona_*_labels.json` (from subagents)
- `judge_labels.parquet` (consolidated)

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
    "persona_id": str,  # 8-char UUID hex (e.g., "a3f8b2c1")
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

    # Rationales (JSON string or null)
    "rationales_json": str | None,  # Sparse dict: {"value_name": "explanation", ...}
}
```

---

## Checklist

- [ ] Run Phase 1 wrangling: `python -m src.wrangling.parse_synthetic_data`
- [ ] Read wrangled persona_*.md files from `logs/wrangled/`
- [ ] Read schwartz_values.yaml
- [ ] Ensure output directory exists: `logs/judge_labels/`
- [ ] Build value rubric context
- [ ] Group entries by persona (extract UUID from filename)
- [ ] Build session content for each entry
- [ ] Launch all persona subagents with `run_in_background=true` in ONE message
  - Each subagent receives persona data with UUID
  - Each subagent writes `logs/judge_labels/persona_{uuid}_labels.json`
- [ ] Wait for all subagents with TaskOutput
- [ ] Consolidate JSON → Parquet (also updates registry)
- [ ] Report summary

**Subagent count:** N total (one per persona in wrangled directory)

**Example:** 5 persona files = 5 subagent calls (all launched in parallel, each with unique UUID)

**Check pipeline status:**
```bash
python3 -c "from src.registry import get_status; print(get_status())"
```

---

## Example Invocation

User:
```
Run judge labeling
```

Claude Code:
1. Checks for wrangled `persona_*.md` files in `logs/wrangled/`
2. If not found, runs wrangling first: `python -m src.wrangling.parse_synthetic_data`
3. Reads markdown files, builds rubric, launches subagents
4. Consolidates results to `logs/judge_labels/judge_labels.parquet`
5. Updates registry with `stage_labeled=true` for each validated persona

---

## Limitations

- **Max 10 parallel personas**: Claude Code caps parallelism at 10; additional tasks queue automatically
- **Context per subagent**: Each subagent has separate context; pass all needed info
- **Timeout**: Set adequate timeout (5+ minutes) for personas with many entries
