# LLM-Judge Implementation Specification

This document specifies the **LLM-Judge** component that produces training labels for the Value Identity Function (VIF). This is **Phase 2** of the Generator–LLM-Judge–VIF Critic workflow.

---

## Executive Summary: How the LLM-Judge Labels

> **Key points for anyone validating or comparing against LLM-Judge labels.**

| Aspect | LLM-Judge Behavior |
|--------|----------------|
| **Scoring scale** | Categorical: -1 (misaligned), 0 (neutral), +1 (aligned) |
| **Dimensions** | All 10 Schwartz values scored independently per Journal Entry |
| **Context window** | Entry + **all previous Journal Entries** for trajectory context |
| **Nudge sessions** | Entire session (Journal Entry + nudge + response) scored as one unit |
| **Signal source** | Max-signal approach — if response reveals more than Journal Entry, score reflects response |

### ⚠️ Critical for Human Validation

**The LLM-Judge considers previous Journal Entries when labeling.** A Journal Entry like "Took the smaller investment. Felt right." may receive Self-Direction +1 because the LLM-Judge saw a previous Journal Entry discussing the autonomy trade-off between investors.

When validating LLM-Judge labels, human annotators **must read Journal Entries in chronological order** and consider the cumulative context. Labeling Journal Entries in isolation will produce systematic disagreements.

See also: [`annotation_guidelines.md`](annotation_guidelines.md) for human annotation methodology.

---

## Table of Contents

1. [Why the LLM-Judge is Required](#why-the-llm-judge-is-required)
2. [What the LLM-Judge Does](#what-the-llm-judge-does)
3. [Implementation](#implementation)
   - [Data Wrangling](#data-wrangling)
   - [Option A: Python Scripts](#option-a-python-scripts)
   - [Option B: Claude Code Subagents](#option-b-claude-code-subagents)
4. [Validation & Quality Control](#validation--quality-control)

---

## Why the LLM-Judge is Required

### The Training Data Gap

The synthetic data generation workflow (`src/synthetic/generation.py`, `src/nudge/`) produces:
- ✅ Synthetic personas with Schwartz value profiles
- ✅ Longitudinal Journal Entries showing value-related tensions and conflicts
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

Hand-labeling 1,000+ Journal Entries across 10 Schwartz dimensions would require:
- ~10-20 minutes per Journal Entry × 1,000 Journal Entries = **167-333 hours** (4-8 weeks full-time)
- Multiple annotators for inter-rater reliability
- Expert knowledge of Schwartz value theory

**LLM-Judge** reduces this to:
- ~10-30 seconds per Journal Entry (API calls)
- Consistent application of rubrics
- Scales to 10,000+ Journal Entries

### The Generator-LLM-Judge-VIF Critic Architecture

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  GENERATOR   │─────▶│  LLM-JUDGE   │─────▶│ VIF CRITIC   │
│  (Phase 1)   │      │  (Phase 2)   │      │  (Phase 3)   │
└──────────────┘      └──────────────┘      └──────────────┘
  Synthetic            Alignment             Distilled
  Journal Entries      score labels          MLP model

  Input: None          Input: Journal Entries Input: Labels
  Output: Text         Output: Labels         Output: Predictions
```

**Why this design?**
1. **Generator**: Creates diverse, realistic data (only LLMs can do this at scale)
2. **LLM-Judge**: Provides expensive but accurate labels (LLM has rich reasoning)
3. **VIF Critic**: Fast inference for production (MLP runs in milliseconds, enables MC Dropout)

This trains the fast, local VIF Critic from labels created offline by the slower LLM-Judge.

---

## What the LLM-Judge Does

### Core Functionality

The LLM-Judge takes a Journal Entry and outputs a **vector of alignment scores** — one score per Schwartz value dimension.

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

To avoid subjective noise (arbitrary "4.5 vs 4.2" scores), the LLM-Judge uses a **strict categorical protocol**:

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

For Journal Entries with nudge-responses, the LLM-Judge scores the **entire session as a unit**:

```json
{
  "initial_entry": "Feeling off today.",
  "nudge": "What happened right before that?",
  "response": "Had a call with my manager. She said I wasn't 'stepping up' enough. I've been working nights for two months straight but apparently it's not visible."
}
```

**Scoring approach: Max-signal**
- If initial Journal Entry was vague but response revealed misalignment, score reflects the **revealed** content
- Track which part provided the signal: `"primary_signal_source": "response"`

---

## Implementation

Two implementation options are available. Choose based on your needs:

| Option | Model | Pros | Cons |
|--------|-------|------|------|
| **A: Python Scripts** | OpenAI (gpt-4o-mini) | Fast iteration, lower cost, code in repo | Smaller model |
| **B: Claude Code** | Claude (Opus/Sonnet) | Larger model, may be more accurate | Manual orchestration |

---

### Data Wrangling

Data wrangling is implemented in `src/wrangling/parse_synthetic_data.py`.

It currently:
1. **Parses `persona_*.md` files** from flat `logs/synthetic_data/`
2. **Extracts persona metadata**: name, age, profession, culture, core_values, bio
3. **Extracts conversational Journal Entries**: initial_entry + optional nudge + optional response
4. **Writes clean LLM-Judge-ready markdown** to `logs/wrangled/persona_*.md`
5. **Updates registry stage** to mark personas as wrangled

**Expected data structure per persona:**
```json
{
  "persona_id": "a3f8b2c1",
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

### Option A: Python Scripts

The LLM-Judge implementation is script-based via:
- `src/judge/labeling.py`
- `scripts/journalling/judge_sanity_check.py`
- `src/judge/consolidate.py`

These modules provide:

- Pydantic models for structured output (`AlignmentLabel`)
- Async batch processing with `asyncio.gather()`
- Integration with OpenAI API (gpt-4o-mini, with GPT-5 support)
- Parquet output for downstream training

**Prompt design**: The LLM-Judge prompt is intentionally lean (~60 lines). Rubrics are pulled dynamically from `config/schwartz_values.yaml` rather than duplicated in the prompt.

**Output format** (Parquet schema):
- `persona_id`, `t_index`, `alignment_vector` (list of 10 ints)
- Individual columns: `alignment_self_direction`, `alignment_stimulation`, etc.

**Data flow:**
```
logs/synthetic_data/persona_*.md
  → src/wrangling/parse_synthetic_data.py
  → logs/wrangled/persona_*.md
  → src/judge/labeling.py (per-persona JSON)
  → src/judge/consolidate.py
  → logs/judge_labels/judge_labels.parquet
```

---

### Option B: Claude Code Subagents

Use Claude Code's parallel Task tool to run one LLM-Judge subagent per persona.

**Architecture**: Same as synthetic data generation — one subagent per persona, all running in parallel.

```
┌─────────────────────────────────────────────────────────────────────┐
│  MAIN ORCHESTRATOR                                                  │
│                                                                     │
│  1. Read all persona log files from logs/synthetic_data/           │
│                                                                     │
│  2. Launch parallel judge subagents (one per persona)               │
│     ┌────────────────┐ ┌────────────────┐ ┌────────────────┐       │
│     │ LLM-Judge Persona1 │ │ LLM-Judge Persona2 │ │ LLM-Judge Persona3 │       │
│     │ Score Entry 1  │ │ Score Entry 1  │ │ Score Entry 1  │       │
│     │ Score Entry 2  │ │ Score Entry 2  │ │ Score Entry 2  │       │
│     │ Score Entry 3  │ │ Score Entry 3  │ │ Score Entry 3  │       │
│     │ ...→Write JSON │ │ ...→Write JSON │ │ ...→Write JSON │       │
│     └───────┬────────┘ └───────┬────────┘ └───────┬────────┘       │
│             │                  │                  │                 │
│  3. Wait for all to complete with TaskOutput                        │
│             │                  │                  │                 │
│             ▼                  ▼                  ▼                 │
│  4. Consolidate JSON labels, validate, report stats                 │
└─────────────────────────────────────────────────────────────────────┘
```

**Why parallel?**
- LLM-Judge calls can be slow (5-15 seconds per Journal Entry)
- 100 personas × 10 Journal Entries = 1,000 Journal Entries
- Serial: ~3-5 hours. Parallel (10 concurrent): ~20-40 minutes.

**Configuration:**
```yaml
SYNTHETIC_DATA_DIR: "logs/synthetic_data"
WRANGLED_DIR: "logs/wrangled"
JUDGE_OUTPUT_DIR: "logs/judge_labels"
MAX_CONCURRENT_JUDGES: 10
```

**Orchestrator steps:**

1. **Read synthetic data**: Parse `persona_*.md` files (see Data Wrangling above)

2. **Create output directory**: `mkdir -p logs/judge_labels`

3. **Launch subagents**: Send ONE message with multiple Task calls (all with `run_in_background=true`)
   ```
   Tool: Task
   - description: "LLM-Judge Persona 1 Journal Entries"
   - subagent_type: "general-purpose"
   - run_in_background: true
   - prompt: [Persona context + Journal Entries + rubrics + output path]
   ```

4. **Wait for completion**: Use `TaskOutput` with `block=true` for each task

5. **Consolidate results**: Merge `persona_*_labels.json` via `python -m src.judge.consolidate`

**Output format** (JSON per persona):
```json
{
  "persona_id": "a3f8b2c1",
  "labels": [
    {
      "t_index": 0,
      "date": "2023-11-05",
      "scores": {
        "self_direction": 0,
        "stimulation": 0,
        "hedonism": -1,
        "achievement": 1,
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

---

## Validation & Quality Control

### Automated Checks

Flag Journal Entries where:
- **All-zero scores**: Entry scored 0 across all dimensions (likely too vague)
- **Extreme sparsity**: >80% of a persona's Journal Entries are all-zero (persona may be too neutral)

### Manual Quality Checks

After first run, manually review 10-20 Journal Entries across different personas:
- Do scores match your intuition?
- Did LLM-Judge conflate similar values (Achievement vs Power)?
- Are similar Journal Entries getting similar scores?

### Iteration Loop

If quality is poor:
1. Refine rubrics in `config/schwartz_values.yaml`
2. Add constraints to the prompt
3. Re-run on flagged personas

**Target**: Human agreement with LLM-Judge on sampled Journal Entries (Cohen's κ > 0.60)

---

## Checklist

**Data Wrangling (required for both options):**
- [ ] Run `python -m src.wrangling.parse_synthetic_data`
- [ ] Confirm `logs/wrangled/persona_*.md` files were produced

**Option A (Scripts) or Option B (Claude Code):**
- [ ] Run LLM-Judge on 3-5 personas as pilot
- [ ] Manually review pilot results
- [ ] If quality acceptable, run on full dataset
- [ ] Save labels (`persona_*_labels.json`) and consolidate to parquet

---

## Next Steps (Phase 3: VIF Training)

Once you have `logs/judge_labels/judge_labels.parquet`, see `docs/vif/03_model_training.md` for the training workflow.

---

## Limitations

- **LLM-Judge consistency**: LLMs can be inconsistent; consider temperature=0 for determinism
- **Value conflation**: LLM-Judge may conflate similar values (Achievement vs Power); refine rubrics iteratively

---

## References

- Option A implementation: `src/judge/labeling.py`
- Option B pattern: `docs/pipeline/claude_gen_instructions.md`
- Schwartz value rubrics: `config/schwartz_values.yaml`
- VIF training workflow: `docs/vif/03_model_training.md`
