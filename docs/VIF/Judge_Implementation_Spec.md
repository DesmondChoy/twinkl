# LLM-as-Judge Implementation Specification

This document specifies the **Reward Model (LLM-as-Judge)** component that produces training labels for the Value Identity Function (VIF). This is **Phase 2** of the Generator-Judge-Critic pipeline.

---

## Table of Contents

1. [Why the Judge is Required](#why-the-judge-is-required)
2. [What the Judge Does](#what-the-judge-does)
3. [How to Implement Using Claude Code](#how-to-implement-using-claude-code)
4. [Judge Prompt Design](#judge-prompt-design)
5. [Execution Architecture](#execution-architecture)
6. [Validation & Quality Control](#validation--quality-control)

---

## Why the Judge is Required

### The Training Data Gap

The synthetic data generation pipeline (`journal_gen.ipynb`, `journal_nudge.ipynb`) produces:
- ✅ Synthetic personas with Schwartz value profiles
- ✅ Longitudinal journal entries showing value drift/conflicts
- ✅ Conversational nudges and responses

But it does **NOT** produce:
- ❌ Alignment scores per value dimension
- ❌ Training labels for the VIF

### The Problem: No Ground Truth

For real users, we never observe true alignment scores like `[Health: -0.8, Career: +0.6, Relationships: 0.0]`. We only have text.

**The VIF is a supervised learning problem** — it needs:
- **Input**: Text embeddings + user profile + history
- **Target**: Per-dimension alignment scores

Without labels, we can't train the VIF.

### Why Not Hand-Label?

Hand-labeling 1,000+ entries across 10 Schwartz dimensions would require:
- ~10-20 minutes per entry × 1,000 entries = **167-333 hours** (4-8 weeks full-time)
- Multiple annotators for inter-rater reliability
- Expert knowledge of Schwartz value theory

**LLM-as-Judge** reduces this to:
- ~10-30 seconds per entry (API calls)
- Consistent application of rubrics
- Scales to 10,000+ entries

### The Generator-Judge-Critic Architecture

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  GENERATOR   │─────▶│    JUDGE     │─────▶│    CRITIC    │
│  (Phase 1)   │      │  (Phase 2)   │      │  (Phase 3)   │
└──────────────┘      └──────────────┘      └──────────────┘
  Synthetic            Alignment             Distilled
  journal entries      score labels          MLP model

  Input: None          Input: Entries        Input: Labels
  Output: Text         Output: Labels        Output: Fast VIF
```

**Why this design?**
1. **Generator**: Creates diverse, realistic data (only LLMs can do this at scale)
2. **Judge**: Provides expensive but accurate labels (LLM has rich reasoning)
3. **Critic**: Fast inference for production (MLP runs in milliseconds, enables MC Dropout)

This is **knowledge distillation** from a slow, expensive Teacher (Judge) to a fast, cheap Student (Critic).

---

## What the Judge Does

### Core Functionality

The Judge takes a journal entry and outputs a **vector of alignment scores** — one score per Schwartz value dimension.

**Input:**
```json
{
  "entry": {
    "date": "2023-11-05",
    "content": "Stayed late again to finish the deck. Told myself I'd leave by 6, but then Sarah pinged about the investor meeting and I just... kept working. Skipped the gym. Again."
  },
  "persona": {
    "name": "Alex Chen",
    "age": "28",
    "profession": "Product Manager",
    "core_values": ["Achievement", "Health"],
    "bio": "Recently promoted to senior PM at a startup. Tracks quarterly OKRs religiously..."
  },
  "recent_entries": [
    // Optional: last 2-3 entries for context
  ]
}
```

**Output:**
```json
{
  "alignment_vector": {
    "Self-Direction": 0,
    "Stimulation": 0,
    "Hedonism": -1,
    "Achievement": +1,
    "Power": 0,
    "Security": 0,
    "Conformity": 0,
    "Tradition": 0,
    "Benevolence": 0,
    "Universalism": 0
  },
  "rationale": {
    "Achievement": "Entry shows prioritizing work performance (finishing deck for investor meeting) over personal plans. Clear demonstration of achievement-oriented behavior.",
    "Hedonism": "Skipping gym and extending work hours sacrifices physical wellbeing and present pleasure. Misaligned with enjoying/caring for the body.",
    "Health": "Explicitly mentions skipping gym 'Again' — pattern of neglecting physical health for work."
  },
  "confidence": {
    "Achievement": 0.95,
    "Hedonism": 0.85,
    "Health": 0.90
  },
  "primary_signal_source": "initial_entry",
  "flags": []
}
```

### The 3-Point Rubric

To avoid subjective noise (arbitrary "4.5 vs 4.2" scores), the Judge uses a **strict categorical protocol**:

| Score | Meaning | Criteria |
|-------|---------|----------|
| **+1** | **Aligned** | Entry actively supports/demonstrates this value through behavior |
| **0** | **Neutral** | Entry is irrelevant to this value OR maintains status quo |
| **-1** | **Misaligned** | Entry actively conflicts with/neglects this value |

**Why categorical?**
- Higher inter-rater reliability (clear decision boundaries)
- Maps cleanly to regression targets
- Reduces "middle-of-the-scale" hedging

### What Gets Scored

**Each of the 10 Schwartz values:**
- Self-Direction
- Stimulation
- Hedonism
- Achievement
- Power
- Security
- Conformity
- Tradition
- Benevolence
- Universalism

**Multi-dimensional scoring captures trade-offs:**
- Entry: "Worked 80 hours this week, missed kid's recital"
- Labels: `Achievement: +1, Benevolence: -1, Hedonism: -1`

This is the **core insight** — don't collapse to a single "good/bad" score. Keep tensions visible.

### Conversational Entry Scoring

For entries with nudge-responses, the Judge scores the **entire session as a unit**:

```json
{
  "initial_entry": "Feeling off today.",
  "nudge": "What happened right before that?",
  "response": "Had a call with my manager. She said I wasn't 'stepping up' enough. I've been working nights for two months straight but apparently it's not visible."
}
```

**Scoring approach: Max-signal**
- If initial entry was vague but response revealed misalignment, score reflects the **revealed** content
- Track which part provided the signal: `"primary_signal_source": "response"`

---

## How to Implement Using Claude Code

### Execution Model: Parallel Subagents

**Architecture**: Same as synthetic data generation — one subagent per persona, all running in parallel.

```
┌─────────────────────────────────────────────────────────────────────┐
│  MAIN ORCHESTRATOR                                                  │
│                                                                     │
│  1. Read all persona log files from synthetic data run             │
│                                                                     │
│  2. Launch parallel judge subagents (one per persona)               │
│     ┌────────────────┐ ┌────────────────┐ ┌────────────────┐       │
│     │ Judge Persona1 │ │ Judge Persona2 │ │ Judge Persona3 │       │
│     │ Score Entry 1  │ │ Score Entry 1  │ │ Score Entry 1  │       │
│     │ Score Entry 2  │ │ Score Entry 2  │ │ Score Entry 2  │       │
│     │ Score Entry 3  │ │ Score Entry 3  │ │ Score Entry 3  │       │
│     │ ...→Write CSV  │ │ ...→Write CSV  │ │ ...→Write CSV  │       │
│     └───────┬────────┘ └───────┬────────┘ └───────┬────────┘       │
│             │                  │                  │                 │
│  3. Wait for all to complete with TaskOutput                        │
│             │                  │                  │                 │
│             ▼                  ▼                  ▼                 │
│  4. Consolidate CSVs, validate, report stats                        │
└─────────────────────────────────────────────────────────────────────┘
```

**Why parallel?**
- Judge calls can be slow (5-15 seconds per entry)
- 100 personas × 10 entries = 1,000 entries
- Serial: ~3-5 hours. Parallel (10 concurrent): ~20-40 minutes.

### Configuration Variables

```yaml
# Input/Output paths
SYNTHETIC_DATA_RUN: "logs/synthetic_data/2026-01-06_23-02-23"
JUDGE_OUTPUT_DIR: "logs/judge_labels/2026-01-07_10-15-00"

# Judge parameters
INCLUDE_RECENT_CONTEXT: true  # Pass last 2 entries to Judge
MAX_CONCURRENT_JUDGES: 10     # Claude Code parallelism limit
```

### Orchestrator Steps

#### 1. Read Synthetic Data Run

```bash
# List all persona log files
ls logs/synthetic_data/2026-01-06_23-02-23/persona_*.md
```

Parse each `persona_XXX.md` file to extract:
- Persona profile (name, age, profession, culture, core_values, bio)
- All entries with dates, content, metadata
- Nudges and responses (if present)

**Data structure per persona:**
```json
{
  "persona_id": 1,
  "persona": { ... },
  "entries": [
    {
      "date": "2023-11-05",
      "initial_entry": "...",
      "nudge": { "text": "...", "category": "..." },
      "response": "...",
      "tone": "Self-reflective",
      "verbosity": "Medium",
      "reflection_mode": "Unsettled"
    }
  ]
}
```

#### 2. Create Judge Output Directory

```bash
mkdir -p logs/judge_labels/$(date +%Y-%m-%d_%H-%M-%S)
```

#### 3. Launch Judge Subagents

Send **one message** with multiple Task calls (all with `run_in_background=true`):

```
Tool: Task
- description: "Judge Persona 1 entries"
- subagent_type: "general-purpose"
- run_in_background: true
- prompt: [Judge prompt - see below]

Tool: Task
- description: "Judge Persona 2 entries"
- subagent_type: "general-purpose"
- run_in_background: true
- prompt: [Judge prompt - see below]

... (all personas)
```

#### 4. Wait for Completion

```
Tool: TaskOutput
- task_id: [persona 1 task ID]
- block: true
- timeout: 600000  # 10 minutes for scoring all entries

Tool: TaskOutput
- task_id: [persona 2 task ID]
- block: true
- timeout: 600000

... (all personas)
```

#### 5. Consolidate Results

Collect all `persona_XXX_labels.csv` files and merge into:
- `judge_labels_all.csv` — Combined dataset
- `judge_summary.json` — Statistics (entries scored, flags raised, etc.)

---

## Judge Prompt Design

### Subagent Judge Prompt

Each subagent receives:

1. **Schwartz Value Rubrics** (from `config/schwartz_values.yaml`):
   - Core motivation for each value
   - Behavioral manifestations
   - What +1, 0, -1 mean for this value

2. **Persona context**:
   - Full persona profile (to understand their value priorities)
   - Note: Persona has declared values, but Judge scores ALL 10 dimensions

3. **Entry to score**:
   - Date, initial_entry, nudge (if present), response (if present)
   - Optional: Previous 2 entries for trajectory context

4. **Scoring instructions**:
   - Use 3-point scale: {-1, 0, +1}
   - Provide rationale for non-zero scores
   - Optional: Confidence scores
   - For conversational entries: Score the full session (max-signal approach)

5. **Output format** (structured JSON):
   ```json
   {
     "alignment_vector": { ... },
     "rationale": { ... },
     "confidence": { ... },
     "primary_signal_source": "initial_entry" | "response" | "both"
   }
   ```

6. **Logging instructions**:
   - Write `persona_XXX_labels.csv` in judge output directory
   - Format: `date,entry_id,Self-Direction,Stimulation,...,Universalism,primary_signal_source`

### Judge Prompt Template

````markdown
You are evaluating journal entries to produce alignment scores across 10 Schwartz value dimensions.

## Persona Context

**Name**: {{persona.name}}
**Background**: {{persona.bio}}
**Declared Values**: {{persona.core_values | join(', ')}}

Note: The persona prioritizes these values, but you will score ALL 10 dimensions. An entry can be aligned with one value and misaligned with another.

## Schwartz Value Rubrics

For each value, use this 3-point scale:

| Score | Meaning |
|-------|---------|
| **+1** | Entry shows behavior that **actively supports** this value |
| **0** | Entry is **neutral/irrelevant** to this value OR maintains status quo |
| **-1** | Entry shows behavior that **actively conflicts with** this value |

### Self-Direction (Independent thought and action)

**Core Motivation**: {{schwartz_values['Self-Direction'].core_motivation}}

**+1 Examples** (Aligned):
- Makes autonomous decision despite external pressure
- Pursues self-chosen goal or learning
- Resists control or conformity to assert independence

**-1 Examples** (Misaligned):
- Follows others' direction without question when autonomy was possible
- Sacrifices personal choice for convenience/approval
- Accepts constraints that limit future options

**0 Examples** (Neutral):
- Routine task with no autonomy/constraint tension
- No decision-making involved
- Maintaining existing autonomous setup

### Stimulation (Excitement, novelty, challenge)

**Core Motivation**: {{schwartz_values['Stimulation'].core_motivation}}

**+1 Examples**:
- Seeks new experience or takes on challenging task
- Makes choice prioritizing novelty over comfort
- Engages with unfamiliar situation

**-1 Examples**:
- Avoids opportunity for new experience
- Chooses predictable routine over available excitement
- Expresses boredom but takes no action

**0 Examples**:
- Maintenance of existing routine
- No novelty/boredom tension present

### Hedonism (Pleasure, sensuous gratification)

**Core Motivation**: {{schwartz_values['Hedonism'].core_motivation}}

**+1 Examples**:
- Prioritizes enjoyment or physical wellbeing
- Takes time for pleasurable activity
- Makes choice favoring quality of life

**-1 Examples**:
- Sacrifices sleep, meals, or physical comfort for other goals
- Forgoes available pleasure/rest
- Pushes through exhaustion

**0 Examples**:
- Neutral daily activities
- No pleasure/sacrifice tension

### Achievement (Success through demonstrating competence)

**Core Motivation**: {{schwartz_values['Achievement'].core_motivation}}

**+1 Examples**:
- Works toward measurable goal or recognition
- Demonstrates competence or reaches milestone
- Compares favorably to standard/peers

**-1 Examples**:
- Misses deadline, underperforms, or avoids challenge
- Lets quality slip below standards
- Fails to meet self-set goal

**0 Examples**:
- Routine task, no performance evaluation involved
- Maintaining existing level

### Power (Social status, control over people/resources)

**Core Motivation**: {{schwartz_values['Power'].core_motivation}}

**+1 Examples**:
- Gains authority, control, or status
- Influences others or accumulates resources
- Asserts dominance in situation

**-1 Examples**:
- Loses control, status, or authority
- Defers when assertion was possible
- Feels powerless

**0 Examples**:
- No status or control dynamics present

### Security (Safety, stability, harmony)

**Core Motivation**: {{schwartz_values['Security'].core_motivation}}

**+1 Examples**:
- Takes action to increase safety/stability
- Maintains protective routine
- Avoids unnecessary risk

**-1 Examples**:
- Takes risk that threatens stability
- Neglects protective measures
- Creates/accepts instability

**0 Examples**:
- Stable situation, no security tension

### Conformity (Restraint of actions that might upset others)

**Core Motivation**: {{schwartz_values['Conformity'].core_motivation}}

**+1 Examples**:
- Restrains impulse to avoid upsetting others
- Follows social norms/expectations
- Prioritizes harmony over self-expression

**-1 Examples**:
- Violates norms or expectations
- Prioritizes self-expression over others' comfort
- Creates social friction

**0 Examples**:
- No norm/expectation tension

### Tradition (Respect for customs/ideas of culture/religion)

**Core Motivation**: {{schwartz_values['Tradition'].core_motivation}}

**+1 Examples**:
- Honors family/cultural tradition
- Makes choice reflecting heritage values
- Maintains cultural practice

**-1 Examples**:
- Breaks with tradition or family expectations
- Rejects cultural/religious practice
- Prioritizes modern/individual choice over tradition

**0 Examples**:
- No tradition/modernity tension

### Benevolence (Welfare of close others)

**Core Motivation**: {{schwartz_values['Benevolence'].core_motivation}}

**+1 Examples**:
- Helps family member or close friend
- Sacrifices for loved one's benefit
- Invests time/energy in close relationship

**-1 Examples**:
- Neglects family/friend need
- Prioritizes self over close others when help was needed
- Damages close relationship

**0 Examples**:
- No close relationship dynamics

### Universalism (Welfare of all people and nature)

**Core Motivation**: {{schwartz_values['Universalism'].core_motivation}}

**+1 Examples**:
- Takes action for social justice or environment
- Makes ethical choice considering broader impact
- Engages with causes beyond personal circle

**-1 Examples**:
- Ignores broader harm from actions
- Prioritizes narrow self-interest over ethics
- Disengages from social/environmental issues

**0 Examples**:
- No broader ethical dimension present

---

## Entry to Score

**Date**: {{entry.date}}

{% if entry.response %}
**Type**: Conversational (initial entry + nudge + response)

**Initial Entry**:
{{entry.initial_entry}}

**Nudge**: "{{entry.nudge.text}}"

**Response**:
{{entry.response}}

**Scoring instruction**: Score the FULL session. If the response revealed alignment/misalignment not apparent in the initial entry, use the response signal. Set `primary_signal_source` to indicate where the main signal came from.
{% else %}
**Type**: One-way entry

**Entry**:
{{entry.initial_entry}}
{% endif %}

{% if previous_entries %}
**Recent Context** (for trajectory awareness):
{% for prev in previous_entries %}
- {{prev.date}}: {{prev.initial_entry[:150]}}{% if prev.initial_entry|length > 150 %}...{% endif %}
{% endfor %}
{% endif %}

---

## Your Task

1. **Score each dimension** using {-1, 0, +1}
2. **Write rationale** for any non-zero score (2-3 sentences explaining what behavior you observed)
3. **Assess confidence** (0.0-1.0) for non-zero scores — How certain are you this score is correct?
4. **Identify primary signal source** (for conversational entries): `"initial_entry"`, `"response"`, or `"both"`

## Output Format

Return valid JSON:

```json
{
  "alignment_vector": {
    "Self-Direction": 0,
    "Stimulation": 0,
    "Hedonism": -1,
    "Achievement": 1,
    "Power": 0,
    "Security": 0,
    "Conformity": 0,
    "Tradition": 0,
    "Benevolence": 0,
    "Universalism": 0
  },
  "rationale": {
    "Achievement": "Entry shows prioritizing work (finishing deck for meeting) over personal plans. Clear achievement-oriented behavior.",
    "Hedonism": "Skipped gym and extended work hours, sacrificing physical wellbeing and rest."
  },
  "confidence": {
    "Achievement": 0.95,
    "Hedonism": 0.85
  },
  "primary_signal_source": "initial_entry"
}
```

**Rules**:
- MUST score all 10 dimensions (even if 0)
- MUST provide rationale for every non-zero score
- confidence optional but recommended for non-zero scores
- Confidence scale: 0.50 = unsure, 0.75 = moderate, 0.90+ = high
- Be strict: Reserve +1/-1 for clear demonstrations, not weak signals
- Multiple dimensions can be non-zero (trade-offs are expected)

---

## Subagent Loop

After scoring each entry:

1. Store result in memory
2. Move to next entry (with updated context)
3. After all entries scored:
   - Write CSV file: `persona_{{persona_id}}_labels.csv`
   - Return summary JSON

## CSV Format

```csv
date,entry_id,Self-Direction,Stimulation,Hedonism,Achievement,Power,Security,Conformity,Tradition,Benevolence,Universalism,primary_signal_source,avg_confidence
2023-11-05,1,0,0,-1,1,0,0,0,0,0,0,initial_entry,0.90
2023-11-08,2,1,0,0,0,0,-1,0,0,0,0,initial_entry,0.85
...
```

## Write CSV Using Write Tool

```
Write tool call:
- file_path: {{judge_output_dir}}/persona_{{persona_id}}_labels.csv
- content: [CSV with header + all rows]
```

## Return Summary JSON

```json
{
  "persona_id": {{persona_id}},
  "persona_name": "{{persona.name}}",
  "entries_scored": 10,
  "non_zero_scores": {
    "Achievement": 7,
    "Hedonism": 5,
    "Benevolence": 3
  },
  "avg_confidence": 0.87,
  "flags": []
}
```

````

---

## Execution Architecture

### File Organization

```
logs/
├── synthetic_data/
│   └── 2026-01-06_23-02-23/
│       ├── config.md
│       ├── persona_001.md
│       ├── persona_002.md
│       └── ...
└── judge_labels/
    └── 2026-01-07_10-15-00/
        ├── config.md
        ├── persona_001_labels.csv
        ├── persona_002_labels.csv
        ├── ...
        ├── judge_labels_all.csv       # Consolidated
        └── judge_summary.json          # Statistics
```

### Orchestrator Output

**`judge_summary.json`**:
```json
{
  "timestamp": "2026-01-07 10:15:00",
  "synthetic_data_source": "logs/synthetic_data/2026-01-06_23-02-23",
  "personas_processed": 100,
  "total_entries_scored": 1000,
  "avg_confidence": 0.86,
  "score_distribution": {
    "Achievement": {"+1": 234, "0": 612, "-1": 154},
    "Hedonism": {"+1": 145, "0": 701, "-1": 154},
    ...
  },
  "flags": [
    "Persona 27, Entry 5: All dimensions scored 0 (possible vague entry)"
  ]
}
```

**`judge_labels_all.csv`**:
```csv
persona_id,date,entry_id,Self-Direction,Stimulation,Hedonism,Achievement,Power,Security,Conformity,Tradition,Benevolence,Universalism,primary_signal_source,avg_confidence
1,2023-11-05,1,0,0,-1,1,0,0,0,0,0,0,initial_entry,0.90
1,2023-11-08,2,1,0,0,0,0,-1,0,0,0,0,initial_entry,0.85
...
```

This CSV is the **direct input to VIF training** (Phase 3).

---

## Validation & Quality Control

### Automated Checks

The orchestrator should flag:

1. **All-zero entries**: Entry scored 0 across all dimensions (likely too vague)
   ```
   Persona 12, Entry 3: All zeros — check if entry is scoreable
   ```

2. **Low confidence**: Avg confidence < 0.6
   ```
   Persona 45, Entry 7: Avg confidence 0.52 — review entry
   ```

3. **Extreme sparsity**: >80% of scores are zero for a persona
   ```
   Persona 78: 92% zero scores across all entries — persona may be too neutral
   ```

4. **Missing rationales**: Non-zero score without explanation
   ```
   Persona 23, Entry 4: Achievement=+1 but no rationale provided
   ```

### Manual Quality Checks

After first run, manually review:

1. **Sample 10-20 entries** across different personas
2. **Check if rationales match scores** (is +1 justified?)
3. **Look for value confusion** (did Judge conflate Achievement with Power?)
4. **Check consistency** (similar entries get similar scores?)

### Iteration Loop

If quality is poor:

1. **Refine rubrics** in Judge prompt (add more examples)
2. **Add constraints** ("Achievement requires measurable goal or milestone")
3. **Increase context** (pass 3 previous entries instead of 2)
4. **Re-run Judge** on flagged personas

**Target quality**:
- Avg confidence: >0.80
- Inter-rater agreement (Judge vs human on 50 entries): Cohen's κ > 0.60

---

## Checklist

- [ ] Read all persona log files from synthetic data run
- [ ] Create judge output directory
- [ ] Load Schwartz value elaborations from config
- [ ] Prepare Judge prompt template with rubrics
- [ ] Launch all judge subagents with `run_in_background=true` in ONE message
  - Each subagent receives: persona context + all entries + rubrics + output path
- [ ] Wait for all subagents to complete with `TaskOutput`
  - Each subagent writes `persona_XXX_labels.csv`
- [ ] Consolidate all CSVs into `judge_labels_all.csv`
- [ ] Generate `judge_summary.json` with statistics
- [ ] Run automated validation checks
- [ ] Manually review sample of 10-20 entries
- [ ] Iterate on Judge prompt if quality < target

---

## Next Steps (Phase 3: VIF Training)

Once you have `judge_labels_all.csv`:

1. **Load data**: Read synthetic entries + judge labels
2. **Embed entries**: Use SBERT to create text embeddings
3. **Construct state vectors**:
   ```python
   s_{u,t} = Concat[
       phi_text(entry_t),
       phi_text(entry_{t-1}),
       phi_text(entry_{t-2}),
       time_deltas,
       user_profile_embedding
   ]
   ```
4. **Train MLP**: Supervised regression `s → alignment_vector`
5. **Evaluate**: MSE, correlation on held-out personas
6. **Add MC Dropout**: Enable uncertainty estimation

See `VIF_03_Model_Training.md` for full training pipeline specification.

---

## Limitations

- **Context window**: Each subagent has 200k tokens; extremely long entries may need truncation
- **Judge consistency**: LLMs can be inconsistent across runs; consider temperature=0 for determinism
- **Schwartz expertise**: Judge may conflate similar values (Achievement vs Power); refine rubrics iteratively
- **Conversational scoring**: Max-signal approach may overweight responses; monitor `primary_signal_source` distribution

---

## References

- Generator-Judge-Critic workflow: `VIF_03_Model_Training.md` sections 1.2-1.2.3
- Schwartz value theory: `config/schwartz_values.yaml`
- Synthetic data generation: `claude_gen_instructions.md`
- VIF architecture: `VIF_01_Concepts_and_Roadmap.md`, `VIF_02_System_Architecture.md`
