# LLM-as-Judge Implementation Specification

This document specifies the **Reward Model (LLM-as-Judge)** component that produces training labels for the Value Identity Function (VIF). This is **Phase 2** of the Generator-Judge-Critic pipeline.

---

## Table of Contents

1. [Why the Judge is Required](#why-the-judge-is-required)
2. [What the Judge Does](#what-the-judge-does)
3. [Implementation](#implementation)
   - [Data Wrangling (TODO)](#data-wrangling-todo)
   - [Option A: Python Notebook](#option-a-python-notebook)
   - [Option B: Claude Code Subagents](#option-b-claude-code-subagents)
4. [Validation & Quality Control](#validation--quality-control)

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
  "previous_entries": [
    // All previous entries for this persona (trajectory context)
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

## Implementation

Two implementation options are available. Choose based on your needs:

| Option | Model | Pros | Cons |
|--------|-------|------|------|
| **A: Python Notebook** | OpenAI (gpt-4o-mini) | Fast iteration, lower cost, code in repo | Smaller model |
| **B: Claude Code** | Claude (Opus/Sonnet) | Larger model, may be more accurate | Manual orchestration |

---

### Data Wrangling (TODO)

> ⚠️ **Not yet implemented**: Both options require a data loader to parse synthetic data.

The `load_personas_and_entries()` function in the notebook is currently a stub. It needs to:

1. **Parse `persona_*.md` files** from `logs/synthetic_data/<run>/`
2. **Extract persona metadata**: name, age, profession, culture, core_values, bio
3. **Extract journal entries**: date, content, tone, verbosity, reflection_mode
4. **Handle conversational entries**: initial_entry + nudge + response (when present)

**Expected data structure per persona:**
```json
{
  "persona_id": 1,
  "persona": {
    "name": "Alex Chen",
    "age": "28",
    "profession": "Product Manager",
    "culture": "East Asian",
    "core_values": ["Achievement", "Security"],
    "bio": "Recently promoted to senior PM..."
  },
  "entries": [
    {
      "date": "2023-11-05",
      "initial_entry": "Stayed late again to finish the deck...",
      "nudge": { "text": "What made you decide to stay?", "category": "clarifying" },
      "response": "I guess I felt like I had to prove myself..."
    }
  ]
}
```

---

### Option A: Python Notebook

The Judge is implemented in **`notebooks/judge_labeling.ipynb`**. This notebook provides:

- Pydantic models for structured output (`AlignmentLabel`)
- Async batch processing with `asyncio.gather()`
- Integration with OpenAI API (gpt-4o-mini, with GPT-5 support)
- Parquet output for downstream training

**Prompt design**: The Judge prompt is intentionally lean (~60 lines). Rubrics are pulled dynamically from `config/schwartz_values.yaml` rather than duplicated in the prompt.

**Output format** (Parquet schema):
- `persona_id`, `t_index`, `alignment_vector` (list of 10 ints)
- Individual columns: `alignment_self_direction`, `alignment_stimulation`, etc.

**Data flow:**
```
logs/synthetic_data/<run>/persona_*.md  →  notebooks/judge_labeling.ipynb  →  data/judge_labels.parquet
```

---

### Option B: Claude Code Subagents

Use Claude Code's parallel Task tool to run one Judge subagent per persona.

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

**Configuration:**
```yaml
SYNTHETIC_DATA_RUN: "logs/synthetic_data/2026-01-06_23-02-23"
JUDGE_OUTPUT_DIR: "logs/judge_labels/2026-01-07_10-15-00"
MAX_CONCURRENT_JUDGES: 10
```

**Orchestrator steps:**

1. **Read synthetic data**: Parse `persona_*.md` files (see Data Wrangling above)

2. **Create output directory**: `mkdir -p logs/judge_labels/$(date +%Y-%m-%d_%H-%M-%S)`

3. **Launch subagents**: Send ONE message with multiple Task calls (all with `run_in_background=true`)
   ```
   Tool: Task
   - description: "Judge Persona 1 entries"
   - subagent_type: "general-purpose"
   - run_in_background: true
   - prompt: [Persona context + entries + rubrics + output path]
   ```

4. **Wait for completion**: Use `TaskOutput` with `block=true` for each task

5. **Consolidate results**: Merge `persona_XXX_labels.csv` into `judge_labels_all.csv`

**Output format** (CSV):
```csv
persona_id,date,entry_id,Self-Direction,Stimulation,Hedonism,Achievement,Power,Security,Conformity,Tradition,Benevolence,Universalism
1,2023-11-05,1,0,0,-1,1,0,0,0,0,0,0
```

---

## Validation & Quality Control

### Automated Checks

Flag entries where:
- **All-zero scores**: Entry scored 0 across all dimensions (likely too vague)
- **Extreme sparsity**: >80% of a persona's entries are all-zero (persona may be too neutral)

### Manual Quality Checks

After first run, manually review 10-20 entries across different personas:
- Do scores match your intuition?
- Did Judge conflate similar values (Achievement vs Power)?
- Are similar entries getting similar scores?

### Iteration Loop

If quality is poor:
1. Refine rubrics in `config/schwartz_values.yaml`
2. Add constraints to the prompt
3. Re-run on flagged personas

**Target**: Human agreement with Judge on sampled entries (Cohen's κ > 0.60)

---

## Checklist

**Data Wrangling (required for both options):**
- [ ] Implement `load_personas_and_entries()` to parse `persona_*.md` files
- [ ] Handle both one-way entries and conversational (nudge+response) entries

**Option A (Notebook) or Option B (Claude Code):**
- [ ] Run Judge on 3-5 personas as pilot
- [ ] Manually review pilot results
- [ ] If quality acceptable, run on full dataset
- [ ] Save labels (Parquet for Option A, CSV for Option B)

---

## Next Steps (Phase 3: VIF Training)

Once you have `data/judge_labels.parquet`, see `VIF_03_Model_Training.md` for the training pipeline.

---

## Limitations

- **Judge consistency**: LLMs can be inconsistent; consider temperature=0 for determinism
- **Value conflation**: Judge may conflate similar values (Achievement vs Power); refine rubrics iteratively

---

## References

- Option A implementation: `notebooks/judge_labeling.ipynb`
- Option B pattern: `docs/synthetic_data/claude_gen_instructions.md`
- Schwartz value rubrics: `config/schwartz_values.yaml`
- VIF training pipeline: `VIF_03_Model_Training.md`
