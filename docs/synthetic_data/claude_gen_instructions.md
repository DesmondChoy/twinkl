# Claude Synthetic Journal Generation Instructions

Instructions for Claude Code to generate synthetic conversational journal data using parallel subagents.

**No API keys required** - this uses Claude Code's native Task tool, not the external Claude Agent SDK.

---

## Configuration Variables

**Change these values to customize the generation run:**

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_PERSONAS` | 5 | Number of personas to generate |
| `NUM_ENTRIES_PER_PERSONA` | 5 | Journal entries per persona |
| `START_DATE` | 2025-12-01 | First entry date (YYYY-MM-DD) |
| `MIN_DAYS_BETWEEN_ENTRIES` | 2 | Minimum days between entries |
| `MAX_DAYS_BETWEEN_ENTRIES` | 10 | Maximum days between entries |

---

## Source Files

All prompts, decision logic, and configuration come from these files:

| File | Contains |
|------|----------|
| `config/synthetic_data.yaml` | Persona attributes, journal entry options, nudge settings |
| `config/schwartz_values.yaml` | Value elaborations for persona generation |
| `prompts/` | YAML prompt templates with Jinja2 |
| `notebooks/journal_nudge.ipynb` | Nudge decision logic, banned terms |

### Prompt Templates

Prompts are stored in `prompts/` as YAML files with embedded Jinja2 templates:

| File | Purpose |
|------|---------|
| `prompts/persona_generation.yaml` | Template for creating personas |
| `prompts/journal_entry.yaml` | Template for journal entries |
| `prompts/nudge_decision.yaml` | Template for LLM-based nudge classification |
| `prompts/nudge_generation.yaml` | Template for generating nudges |
| `prompts/nudge_response.yaml` | Template for persona responses |

### Notebook References

| Variable | Purpose |
|----------|---------|
| `SCHWARTZ_BANNED_TERMS` | Terms that must not appear in output |
| `decide_nudge_llm()` | LLM-based nudge decision logic |

---

## Execution Architecture

### Per-Persona Subagents (Parallel)

Each persona is handled by a single subagent that runs its entire pipeline internally, **including writing its own log file**. All persona subagents run in parallel.

```
┌─────────────────────────────────────────────────────────────────────┐
│  MAIN ORCHESTRATOR (you)                                            │
│                                                                     │
│  0. Create log directory: logs/synthetic_data/YYYY-MM-DD_HH-MM-SS/  │
│                                                                     │
│  1. Launch all persona subagents with run_in_background=true        │
│     ┌────────────────┐ ┌────────────────┐ ┌────────────────┐       │
│     │ Persona 1      │ │ Persona 2      │ │ Persona 3      │       │
│     │ Entry1→Nudge1  │ │ Entry1→Nudge1  │ │ Entry1→Nudge1  │       │
│     │ Entry2→Nudge2  │ │ Entry2→Nudge2  │ │ Entry2→Nudge2  │       │
│     │ Entry3→Nudge3  │ │ Entry3→Nudge3  │ │ Entry3→Nudge3  │       │
│     │ ...→Write Log  │ │ ...→Write Log  │ │ ...→Write Log  │       │
│     └───────┬────────┘ └───────┬────────┘ └───────┬────────┘       │
│             │                  │                  │                 │
│  2. Wait for all to complete with TaskOutput                        │
│             │                  │                  │                 │
│             ▼                  ▼                  ▼                 │
│  3. Write config.md, collect summaries, report results              │
└─────────────────────────────────────────────────────────────────────┘
```

**Key insight**: Each subagent handles one persona's full pipeline (all entries + nudges + responses) **and writes its own `persona_XXX.md` log file**. The orchestrator creates the log directory upfront, launches N subagents, then only writes `config.md` at the end.

---

## Claude Code Tools Used

### Orchestrator Tools

| Tool | Purpose |
|------|---------|
| `Task` | Spawn one subagent per persona (handles full pipeline + logging) |
| `TaskOutput` | Retrieve results from background tasks |
| `Write` | Create config.md log file |
| `Read` | Read source config files |
| `Bash` | Create log directory, generate random persona configs |

### Subagent Tools

| Tool | Purpose |
|------|---------|
| `Bash` | Random decisions (tone, verbosity, nudge gates, response probability) |
| `Write` | Write persona_XXX.md log file (parallel with other subagents) |

### Task Tool Parameters

```
Task tool call:
- description: "Persona N: Full pipeline"
- prompt: [persona config + all parameters + rules]
- subagent_type: "general-purpose"
- run_in_background: true  ← launches without blocking
```

### TaskOutput Tool Parameters

```
TaskOutput tool call:
- task_id: [ID returned from Task]
- block: true   ← waits for completion
- timeout: 300000  ← 5 minutes (full pipeline takes longer)
```

---

## Execution Steps

### 1. Read Source Files

Read and internalize:
- Both config YAML files
- The prompt templates from `prompts/` folder (including `nudge_decision.yaml` for LLM-based nudge classification)
- The `decide_nudge_llm()` function logic from `notebooks/journal_nudge.ipynb`

### 2. Create Log Directory

Create the timestamped log directory **before** launching subagents:
```bash
mkdir -p logs/synthetic_data/$(date +%Y-%m-%d_%H-%M-%S)
```

Store the directory path (e.g., `logs/synthetic_data/2026-01-06_23-02-23/`) to pass to each subagent.

### 3. Prepare Persona Configurations

For each of `{{NUM_PERSONAS}}` personas, randomly select:
- Age, profession, culture from `config/synthetic_data.yaml`
- 1-2 Schwartz values
- Look up value elaborations from `config/schwartz_values.yaml`
- Generate `{{NUM_ENTRIES_PER_PERSONA}}` dates starting from `{{START_DATE}}`

### 4. Launch All Persona Subagents

Send **one message** with `{{NUM_PERSONAS}}` Task tool calls, all with `run_in_background=true`:

```
Tool: Task
- description: "Persona 1: Full pipeline"
- subagent_type: "general-purpose"
- run_in_background: true
- prompt: [Full persona prompt - see below]

Tool: Task
- description: "Persona 2: Full pipeline"
- subagent_type: "general-purpose"
- run_in_background: true
- prompt: [Full persona prompt - see below]

... (all {{NUM_PERSONAS}} in one message)
```

Each Task returns immediately with a `task_id`. The subagent handles the full pipeline internally.

### 5. Wait for All Subagents

Use TaskOutput to wait for each persona to complete:

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

### 6. Persona Pipeline Subagent Prompt

Each subagent receives everything needed to generate a complete persona with all entries **and write its own log file**.

**Input to subagent:**

1. **Persona constraints** (randomly selected by orchestrator):
   - Age, profession, culture from `config/synthetic_data.yaml`
   - 1-2 Schwartz values
   - Value elaborations from `config/schwartz_values.yaml`

2. **Entry dates**: All `{{NUM_ENTRIES_PER_PERSONA}}` dates (pre-generated)

3. **Generation parameters**:
   - Tone, verbosity, reflection_mode options from config
   - Banned terms from `SCHWARTZ_BANNED_TERMS` (in notebook)
   - Prompt templates from `prompts/` folder

4. **Nudge rules**:
   - `decide_nudge_llm()` logic from notebook (LLM-based classification)
   - `prompts/nudge_decision.yaml` for semantic entry classification
   - Nudge generation instructions from `prompts/nudge_generation.yaml`

5. **Response parameters**:
   - `response_probability` from config
   - `response_modes` with weights from config
   - Response generation instructions from `prompts/nudge_response.yaml`

6. **Logging parameters** (NEW):
   - Log directory path (e.g., `logs/synthetic_data/2026-01-06_23-02-23/`)
   - Persona ID (for filename: `persona_001.md`, `persona_002.md`, etc.)
   - Log file format template

**Subagent internal loop:**

```
1. Generate persona (name, bio based on constraints)

For each entry date:
  2. Randomly select tone/verbosity/reflection_mode
  3. Generate entry content (with accumulated context from prior entries)
  4. Check session cap (2+ nudges in last 3 entries → no nudge)
  5. If not capped: call LLM with nudge_decision_prompt to classify entry
  6. If nudge category returned: generate nudge text
  7. If nudge generated: use Bash for random response decision:
     python3 -c "import random; print(random.random() < [response_probability])"
  8. If Bash returns True: generate response using weighted response_mode
  9. Store entry result, continue to next date

10. Write persona_XXX.md log file using Write tool
11. Return summary JSON with persona name + stats (for orchestrator report)
```

**Bash-Based Randomness:**
LLMs cannot generate true randomness. The subagent MUST use the Bash tool for probabilistic decisions:
```bash
python3 -c "import random; print(random.random() < 0.7)"
```
If output is `True`: Generate response. If `False`: Set `response: null`.

**Subagent writes log file using Write tool:**

The subagent uses the Write tool to create `persona_XXX.md` in the log directory:
```
Write tool call:
- file_path: [log_directory]/persona_001.md
- content: [formatted markdown - see format below]
```

**persona_XXX.md format** (written by subagent):
```markdown
# Persona XXX: [Name]

## Profile
- Age: ...
- Profession: ...
- Culture: ...
- Core Values: ...
- Bio: ...

---

## Entry N - [Date]

### Initial Entry
**Tone**: ... | **Verbosity**: ... | **Reflection Mode**: ...

[content]

### Nudge ([category])
**Trigger**: [reason]
"[nudge_text]"

### Response
**Mode**: [response_mode]
[response content]

*(Or if persona didn't respond:)*
*(No response - persona did not reply to nudge)*
```

**Output format (summary JSON from subagent for orchestrator):**

After writing the log file, subagent returns a lightweight summary:
```json
{
  "persona_id": 1,
  "persona_name": "...",
  "log_file": "persona_001.md",
  "stats": {
    "entries": 3,
    "nudges": 2,
    "responses": 1
  }
}
```

### 7. Collect Results & Write Config

The orchestrator collects summary JSONs from each subagent (the full data is already written to log files).

**Write config.md** (orchestrator only):
```markdown
# Run Configuration
**Timestamp**: ...
**Method**: Claude Code subagents (parallel with run_in_background)

## Parameters
- Personas: {{NUM_PERSONAS}}
- Entries per persona: {{NUM_ENTRIES_PER_PERSONA}}
- Nudge probability: [from config]
- Response probability: [from config]
```

### 8. Report Summary

Print:
- Personas generated: X/{{NUM_PERSONAS}}
- Total entries: X
- Total nudges: X
- Total responses: X (expect response_probability × nudges)
- Response rate: X% (should approximate response_probability from config)

---

## Quick Reference: Nudge Decision Tree

From `decide_nudge_llm()` in notebook — uses LLM classification instead of regex patterns:

```
1. Session cap hit? (2+ nudges in last 3 entries) → No nudge (code-based policy)
2. LLM classifies entry into one of:
   - "no_nudge" — Entry is complete and grounded
   - "clarification" — Entry too vague to understand
   - "elaboration" — Solid entry with unexplored depth
   - "tension_surfacing" — Hints at unresolved conflict
```

The LLM prompt (`prompts/nudge_decision.yaml`) provides semantic criteria for each category, enabling detection of nuanced vagueness, hedging language, and tension that regex patterns would miss.

> **Note**: Grounding nudges were removed—they relied on `reflection_mode` metadata unavailable in production. See `pipeline_specs.md` for details.

**Response Decision (after nudge is generated):**
```
Nudge generated?
   │
   └─YES─→ Run Bash: python3 -c "import random; print(random.random() < [response_probability])"
              │
              ├─Output "True"─→ Generate response using weighted response_mode from config
              │
              └─Output "False"─→ No response (response: null)
```
Note: Subagent reads `response_probability` from `config/synthetic_data.yaml` before running Bash command.

---

## Optional: Define Custom Subagent

For cleaner organization, you can define a specialized persona pipeline subagent in `.claude/agents/`:

**`.claude/agents/persona-pipeline.md`**:
```markdown
---
name: persona-pipeline
description: Generates complete persona with all journal entries, nudges, responses, and writes log file
tools:
  - Bash
  - Write
---

You are a persona pipeline generator. Given persona constraints and parameters,
generate a complete persona with all journal entries and write the log file.

For each entry:
1. Generate entry content following style rules
2. Apply nudge decision logic
3. If nudging: generate nudge, then use Bash for random response decision
4. Accumulate context for next entry

After all entries:
5. Write persona_XXX.md log file using Write tool
6. Return summary JSON with stats

[Include full prompt templates and decide_nudge_llm() logic here]
```

Then invoke with `subagent_type: "persona-pipeline"` instead of `"general-purpose"`.

**Benefits of custom subagent:**
- Reusable across runs
- Cleaner prompts (instructions live in agent definition)
- Tool restrictions (only Bash for randomness + Write for logging)
- Parallel logging built into each subagent

---

## Checklist

- [ ] Read source files (configs + prompts/ + notebook)
- [ ] Set configuration variables above
- [ ] **Create log directory** (before launching subagents)
- [ ] Prepare `{{NUM_PERSONAS}}` persona configurations (random selections)
- [ ] Launch all persona subagents with `run_in_background=true` in ONE message
  - Each subagent receives log directory path + persona ID
- [ ] Wait for all subagents to complete with `TaskOutput`
  - Subagents write their own `persona_XXX.md` files (parallel logging)
- [ ] Collect summary stats from each subagent
- [ ] Write `config.md` (orchestrator only)
- [ ] Report summary

**Subagent count:** `{{NUM_PERSONAS}}` total (one per persona)

**Example:** 5 personas = 5 subagent calls (all run in parallel, each writes its own log file)

---

## Limitations

- **Max 10 parallel personas**: Claude Code caps parallelism at 10; additional tasks queue automatically
- **Context isolation**: Each subagent has separate 200k context; pass all needed info in the prompt
- **Subagent timeout**: Set adequate timeout (5+ minutes) for full pipeline generation
