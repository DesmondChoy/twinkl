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
| `START_DATE` | 2025-10-25 | First entry date (YYYY-MM-DD) |
| `MIN_DAYS_BETWEEN_ENTRIES` | 2 | Minimum days between entries |
| `MAX_DAYS_BETWEEN_ENTRIES` | 10 | Maximum days between entries |

---

## Source Files

All prompts, decision logic, and configuration come from these files:

| File | Contains |
|------|----------|
| `config/synthetic_data.yaml` | Persona attributes, journal entry options, nudge settings |
| `config/schwartz_values.yaml` | Value elaborations for persona generation |
| `notebooks/journal_nudge.ipynb` | Prompt templates, nudge decision logic, banned terms |

### Key Notebook References

| Cell Variable | Purpose |
|---------------|---------|
| `persona_generation_prompt` | Template for creating personas |
| `journal_entry_prompt` | Template for journal entries |
| `nudge_generation_prompt` | Template for generating nudges |
| `nudge_response_prompt` | Template for persona responses |
| `SCHWARTZ_BANNED_TERMS` | Terms that must not appear in output |
| `HEDGING_PATTERNS` | Regex for detecting tension |
| `decide_nudge()` | Rule-based nudge decision logic |

---

## Execution Architecture

### Parallel Personas, Sequential Entries

```
┌─────────────────────────────────────────────────────────────────────┐
│  MAIN ORCHESTRATOR (you)                                            │
│                                                                     │
│  1. Launch all Entry 1 tasks with run_in_background=true            │
│     ┌──────────┐ ┌──────────┐ ┌──────────┐                         │
│     │Persona 1 │ │Persona 2 │ │Persona 3 │  ... (all start at once)│
│     │ Entry 1  │ │ Entry 1  │ │ Entry 1  │                         │
│     └────┬─────┘ └────┬─────┘ └────┬─────┘                         │
│          │            │            │                                │
│  2. Poll with TaskOutput until each completes                       │
│          │            │            │                                │
│          ▼            ▼            ▼                                │
│     ┌──────────┐ ┌──────────┐ ┌──────────┐                         │
│     │ Nudge 1  │ │ Nudge 1  │ │ Nudge 1  │  (sequential per track) │
│     └────┬─────┘ └────┬─────┘ └────┬─────┘                         │
│          │            │            │                                │
│          ▼            ▼            ▼                                │
│     ┌──────────┐ ┌──────────┐ ┌──────────┐                         │
│     │ Entry 2  │ │ Entry 2  │ │ Entry 2  │                         │
│     └────┬─────┘ └────┬─────┘ └────┬─────┘                         │
│          │            │            │                                │
│         ...          ...          ...                               │
│          │            │            │                                │
│          ▼            ▼            ▼                                │
│     [Collate]    [Collate]    [Collate]                            │
│                                                                     │
│  3. Write log files                                                 │
└─────────────────────────────────────────────────────────────────────┘
```

**Key insight**: Use `run_in_background=true` to launch tasks without blocking, then use `TaskOutput` to poll for results.

---

## Claude Code Tools Used

| Tool | Purpose |
|------|---------|
| `Task` | Spawn subagents for entry generation and nudge decisions |
| `TaskOutput` | Retrieve results from background tasks |
| `Write` | Create log files |
| `Read` | Read source config files |

### Task Tool Parameters

```
Task tool call:
- description: "Persona N: Generate entry M"
- prompt: [the full prompt text]
- subagent_type: "general-purpose"
- run_in_background: true  ← launches without blocking
```

### TaskOutput Tool Parameters

```
TaskOutput tool call:
- task_id: [ID returned from Task]
- block: true   ← waits for completion
- timeout: 60000  ← milliseconds
```

---

## Execution Steps

### 1. Read Source Files

Read and internalize:
- Both config YAML files
- The prompt templates from the notebook
- The `decide_nudge()` function logic

### 2. Prepare Persona Configurations

For each of `{{NUM_PERSONAS}}` personas, randomly select:
- Age, profession, culture from `config/synthetic_data.yaml`
- 1-2 Schwartz values
- Look up value elaborations from `config/schwartz_values.yaml`
- Generate `{{NUM_ENTRIES_PER_PERSONA}}` dates starting from `{{START_DATE}}`

### 3. Launch All Entry 1 Tasks in Background

Send **one message** with `{{NUM_PERSONAS}}` Task tool calls, all with `run_in_background=true`:

```
Tool: Task
- description: "Persona 1: Generate entry 1"
- subagent_type: "general-purpose"
- run_in_background: true
- prompt: [Persona creation + Entry 1 prompt]

Tool: Task
- description: "Persona 2: Generate entry 1"
- subagent_type: "general-purpose"
- run_in_background: true
- prompt: [Persona creation + Entry 1 prompt]

... (all {{NUM_PERSONAS}} in one message)
```

Each Task returns immediately with a `task_id`.

### 4. Poll and Process Sequential Steps

For each persona track:

```
Loop for entry_num = 1 to {{NUM_ENTRIES_PER_PERSONA}}:

    A. Get entry result (if not already received):
       Tool: TaskOutput
       - task_id: [entry task ID]
       - block: true

    B. Launch nudge decision task:
       Tool: Task
       - description: "Persona N: Nudge for entry M"
       - prompt: [Entry + nudge decision rules]

    C. If more entries needed, launch next entry task:
       Tool: Task
       - description: "Persona N: Generate entry M+1"
       - prompt: [Accumulated context + entry prompt]
```

**Optimization**: While waiting for one persona's nudge decision, you can process other personas that are ready.

### 5. Subagent Prompt Specifications

#### Entry Subagent Prompt (first entry)

Include:
- Persona constraints (age, profession, culture, values)
- Value elaborations from `config/schwartz_values.yaml`
- Entry instructions from `journal_entry_prompt` template
- Banned terms from `SCHWARTZ_BANNED_TERMS`
- Date for this entry
- Randomly assigned tone/verbosity/reflection_mode

**Output format:**
```json
{
  "persona": {
    "name": "...",
    "age": "...",
    "profession": "...",
    "culture": "...",
    "core_values": ["..."],
    "bio": "..."
  },
  "entry": {
    "date": "YYYY-MM-DD",
    "tone": "...",
    "verbosity": "...",
    "reflection_mode": "...",
    "content": "..."
  }
}
```

#### Entry Subagent Prompt (subsequent entries)

Include:
- Persona (from first entry output)
- All prior entries with their nudges/responses as context
- Entry instructions from `journal_entry_prompt` template
- Banned terms
- Date for this entry
- Randomly assigned tone/verbosity/reflection_mode

**Output format:**
```json
{
  "entry": {
    "date": "YYYY-MM-DD",
    "tone": "...",
    "verbosity": "...",
    "reflection_mode": "...",
    "content": "..."
  }
}
```

#### Nudge Decision Subagent Prompt

Include:
- The entry just generated
- All prior entries (for session cap check)
- Nudge decision rules from `decide_nudge()` function
- Nudge generation instructions from `nudge_generation_prompt`
- Response generation instructions from `nudge_response_prompt`
- Banned nudge phrases from config

**Output format:**
```json
{
  "should_nudge": true,
  "nudge": {
    "category": "tension_surfacing",
    "trigger_reason": "Hedging language detected in unsettled entry",
    "nudge_text": "Does that sit okay?"
  },
  "response": "Not really. I keep thinking about it."
}
```

Or if no nudge:
```json
{
  "should_nudge": false,
  "nudge": null,
  "response": null
}
```

### 6. Collate Results

After all entries complete for a persona, combine:
- Persona info (from first entry subagent)
- All entries with their nudges/responses

### 7. Create Log Files

Create directory: `logs/synthetic_data/YYYY-MM-DD_HH-MM-SS/`

**config.md**:
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

**persona_XXX.md** (one per persona):
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
[response or *(No response)*]
```

**prompts.md**: Document all prompts sent to subagents

### 8. Report Summary

Print:
- Personas generated: X/{{NUM_PERSONAS}}
- Total entries: X
- Total nudges: X
- Total responses: X
- Response rate: X%

---

## Quick Reference: Nudge Decision Tree

From `decide_nudge()` in notebook:

```
1. Session cap hit? (2+ nudges in last 3 entries) → No nudge
2. Entry too vague? (<15 words, no concrete details) → "clarification"
3. Neutral/routine entry? (Neutral mode + <80 words + no hedging) → No nudge
4. Hedging language + Unsettled mode? → "tension_surfacing"
5. Random gate (40% chance) → "elaboration"
6. Otherwise → No nudge
```

Note: "grounding" nudge was removed because it relied purely on `reflection_mode == "Grounded"`,
which is synthetic generation metadata not available in production (metadata leakage).

---

## Optional: Define Custom Subagents

For cleaner organization, you can define specialized subagents in `.claude/agents/`:

**`.claude/agents/entry-generator.md`**:
```markdown
---
name: entry-generator
description: Generates authentic journal entries for synthetic personas
tools:
  - Read
---

You are a journal entry generator. Given a persona and context, write an authentic
first-person journal entry following the style rules provided.

[Include full journal_entry_prompt template here]
```

**`.claude/agents/nudge-evaluator.md`**:
```markdown
---
name: nudge-evaluator
description: Evaluates journal entries and decides whether to generate nudges
tools:
  - Read
---

You are a nudge decision system. Analyze journal entries and apply the nudge
decision rules to determine if a follow-up question is warranted.

[Include decide_nudge() logic and nudge_generation_prompt here]
```

Then invoke with `subagent_type: "entry-generator"` instead of `"general-purpose"`.

**Benefits of custom subagents:**
- Reusable across runs
- Cleaner prompts (instructions live in agent definition)
- Tool restrictions (read-only for safety)

---

## Checklist

- [ ] Read source files (configs + notebook)
- [ ] Set configuration variables above
- [ ] Prepare `{{NUM_PERSONAS}}` persona configurations
- [ ] Launch all Entry 1 subagents with `run_in_background=true` in ONE message
- [ ] For each persona, loop sequentially:
  - [ ] Use `TaskOutput` to get entry result
  - [ ] Launch nudge decision subagent
  - [ ] Get nudge result
  - [ ] Launch next entry subagent (with accumulated context)
  - [ ] Repeat until `{{NUM_ENTRIES_PER_PERSONA}}` entries complete
- [ ] Collate all results per persona
- [ ] Write log files
- [ ] Report summary

**Subagent count per persona:** `{{NUM_ENTRIES_PER_PERSONA}}` entry + `{{NUM_ENTRIES_PER_PERSONA}}` nudge = `2 × {{NUM_ENTRIES_PER_PERSONA}}`

**Example:** 5 personas × 5 entries = 50 subagent calls (parallelized across personas)

---

## Limitations

- **Max 10 parallel tasks**: Claude Code caps parallelism; additional tasks queue automatically
- **No nested subagents**: Subagents cannot spawn their own subagents
- **Context isolation**: Each subagent has separate 200k context; pass needed info explicitly
