# Purpose

Create high-quality data to bootstrap (start from nothing) what Twinkl requires - data that enables value tagging (labeling text with Schwartz's value dimensions) and training a reward model (Critic).

Subsequently, an evaluation framework is required to detect bias and toxic content.

# Objectives

Journals should be:
- Realistic (Genuine personal reflections)
- Diverse in personas and scenarios
- Contain ground-truth alignment signals
- Longitudinal in nature to exhibit value drift, conflicts and ambiguity

# Implementation

**Primary generation**: Use `docs/pipeline/claude_gen_instructions.md` with Claude Code for parallel subagent generation.

**Experimentation**: Use `notebooks/journal_gen.ipynb` or `notebooks/journal_nudge.ipynb` for prompt iteration and testing.

## Configuration Files

### `config/synthetic_data.yaml`
Defines the parameter space for persona and journal generation:
- **Personas**: age_ranges, cultures, professions, schwartz_values (1-2 randomly assigned)
- **Journal entries**: tones, verbosity levels, reflection_mode (Unsettled/Grounded/Neutral)

### `config/schwartz_values.yaml`
Rich psychological elaborations for each of the 10 Schwartz values, based on academic literature. Each value includes:
- Core motivation and definition
- Behavioral manifestations (5+ concrete behaviors)
- Life domain expressions (work, relationships, leisure, conflict style)
- Typical stressors and goals
- Internal conflicts
- Adjacent/opposing values
- Cultural notes
- Persona narrative guidance

These elaborations are injected into prompts to help the LLM generate psychologically grounded personas and entries.

**Design decision**: We removed explicit "interaction patterns" for value pairs. The individual elaborations (especially `adjacent_compatible_values`, `opposing_tension_values`, and `internal_conflicts`) provide sufficient context for the LLM to infer how multiple values interact in a persona.

## Data Models

**Persona** (generated once, used for multiple entries):
- name, age, profession, culture
- core_values (1-2 Schwartz values)
- bio (2-4 sentences showing values through concrete life details, not labels)

**JournalEntry** (generated per date):
- date (YYYY-MM-DD)
- content (the journal text)

**JournalEntryResult** (metadata tracked separately):
- entry, tone, verbosity, reflection_mode (Unsettled/Grounded/Neutral)

**PersonaPipelineResult** (complete output from one persona's generation):
- persona_id, persona, entries (list of JournalEntryResult)
- persona_prompt, entry_prompts (captured for debugging/display)
- error (if generation failed)

## Generation Pipeline

The pipeline uses **async/await** for efficient I/O and supports **parallel persona generation**.

**Per-persona pipeline** (sequential within each persona):
1. **Persona creation**: Random sampling from config + LLM generation with value context injection
2. **Date sequence**: Random intervals (0-7 days) between entries, with 15% same-day probability and random start date in 2025
3. **Longitudinal entries**: Each entry receives previous entries for continuity
4. **Validation**: Banned terms check to prevent Schwartz label leakage

**Parallel execution** (across personas):
- Multiple personas run concurrently via `asyncio.gather()`
- Results return in input order regardless of completion time (Persona 1, 2, 3...)
- Failed pipelines return exceptions without crashing others (`return_exceptions=True`)
- All prompts/outputs are buffered and displayed in order after completion

## Prompt Design

Note: Synthetic journals read like typed text input, matching the text-only input modality of the production system.

### Persona Generation Prompt
- Receives: age, profession, culture, values, value_context (from schwartz_values.yaml)
- Constraints: Bio must show values through concrete details (job choices, relationships, conflicts), NOT through labels or personality adjectives
- Banned terms: All Schwartz value names and derivative adjectives

### Journal Entry Prompt
Determined before each entry:
- Tone (Self-reflective, Brief and factual, Emotional/Venting, Stream of consciousness, Exhausted, Defensive)
- Verbosity (Short: 25-80 words, Medium: 90-180 words, Long: 160-260 words)
- Reflection mode (Unsettled, Grounded, Neutral) — presented as natural "What to write about" guidance, not as a labeled parameter

**Design note**: The journal prompt does NOT receive explicit Schwartz value labels. Instead, the persona bio carries implicit value signals through concrete life details. The reflection mode guidance tells the LLM what *kind* of moment to write about (e.g., "something happened where you gave ground" for Unsettled) without naming what the person is drifting from. This keeps the generated content natural and prevents value label leakage.

Style rules enforced:
- No "Dear Diary" or audience-facing writing
- No time-of-day/weather openings
- No therapy speak or literary metaphors
- No headings, bullets, or numbered lists
- Cultural background as subtle flavor, not stereotypes

## Design Philosophy: Emergent vs Prescriptive Content

We deliberately avoid prescribing **events** or **purposes** for journal entries. Real journaling rarely starts with a declared intent like "I will now write a gratitude entry." People open their journal and write what's on their mind—the purpose emerges from the content, it's not a precondition.

Prescriptive categories create problems:
1. **Performative structure** - "Goal-setting" entries sound like exercises, not organic reflection
2. **Category homogeneity** - All "gratitude" entries sound alike, all "venting" entries sound alike
3. **Forcing retrospective categories prospectively** - A real entry might start as venting and become decision-making midway through

What actually drives journal entries are **states**, not declared purposes:
- Something happened they want to process
- A feeling is weighing on them
- A thought they don't want to forget
- It's their routine/habit
- They can't sleep

These states are already captured implicitly through:
- **Persona bio** - stressors, life situation, goals create natural tensions
- **Tone** - emotional/venting vs analytical implies different mental states
- **Reflection mode** - unsettled vs grounded vs neutral drives the nature of reflection

This leaner approach lets journal content emerge organically from persona context rather than forcing artificial categories.

## Entry Count Distribution: Training Data Diversity

### Why Variable Entry Counts Matter

If all synthetic personas have the same number of entries (e.g., all 5), the VIF may:
- **Overfit to medium-depth histories** — assumes it always has 3+ prior entries
- **Fail at cold start** — doesn't learn to handle uncertainty with sparse data
- **Underutilize long histories** — never sees patterns that emerge over 10+ entries

### Production Reality

| User Type | Typical Entry Count | Prevalence |
|-----------|---------------------|------------|
| New users (cold start) | 1-3 entries | High |
| Active users | 5-10 entries | Medium |
| Power users | 15-30+ entries | Low |
| Churned users | 2-5 entries | High |

### Minimum Entry Requirements

| VIF Capability | Minimum Entries | Rationale |
|----------------|-----------------|-----------|
| Crash detection | 2+ | Needs V_{t-1} vs V_t comparison |
| State vector window | 3+ | Uses sliding window of N=3 |
| Session cap (nudges) | 4+ | "2 nudges in last 3 entries" needs depth |
| Rut detection | ~8-10 | Requires 3+ consecutive weeks |

### Recommended Approach

Use `MIN_ENTRIES=2` and `MAX_ENTRIES=12` with the following configuration:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `MIN_ENTRIES` | 2 | Include cold-start scenarios; VIF must learn uncertainty with sparse data |
| `MAX_ENTRIES` | 12 | Better rut detection coverage (3+ consecutive weeks even with longer gaps) |
| `MIN_DAYS_BETWEEN_ENTRIES` | 0 | Allow same-day entries (realistic for venting, follow-up thoughts) |
| `MAX_DAYS_BETWEEN_ENTRIES` | 7 | Ensures at least ~1 entry per week for meaningful aggregation |
| `SAME_DAY_PROBABILITY` | 0.15 | 15% chance of same-day follow-up entry |
| `START_DATE` | Random in 2025 | Avoids seasonal bias; any date from 2025-01-01 to 2025-12-31 |

This ensures:
- Minimum of 2 entries for cold-start training
- Maximum of 12 covers extended rut detection windows
- Same-day entries expose VIF to rapid Δt=0 sequences (realistic ~15% of sessions)
- Random start dates prevent December bias in training data
- Variable counts train the VIF to handle uncertainty gracefully with sparse data

---

# Two-Way Conversational Journaling (Nudging System)

## Motivation

The current one-way pipeline (persona generates entries independently) has limitations:

1. **Unrealistic**: Real journaling apps have back-and-forth interaction
2. **Signal-poor short entries**: Brief or vague entries lack sufficient signal for VIF alignment scoring
3. **User friction**: People often need gentle prompts to articulate thoughts clearly

The nudging system transforms one-way journaling into a two-way conversational exchange. When an entry is vague or potentially rich with unexplored tension, the system responds with a brief nudge that invites elaboration.

**Design goal**: Nudges should feel like natural curiosity from a thoughtful companion, not interrogation or therapy.

## Nudge Categories

Three categories of nudges, each with a distinct purpose:

| Category | Purpose | Trigger Example | Nudge Example |
|----------|---------|-----------------|---------------|
| **Clarification** | Surface specifics from vague entries | "Feeling off today" | "What happened right before that?" |
| **Elaboration** | Invite depth on surface-level entries | "Stayed late again" | "And how did that land?" |
| **Tension-surfacing** | Probe potential value conflicts | "It was fine. Well, sort of." | "What's the 'sort of' part?" |

> **Note**: The original design included additional categories that were removed:
> - **Continuity**: Removed because word-overlap heuristics were fragile and added complexity without clear POC benefit.
> - **Grounding**: Removed because it relied purely on `reflection_mode == "Grounded"`, which is synthetic generation metadata not available in production (metadata leakage). See [Lesson Learned: Metadata Leakage](#lesson-learned-metadata-leakage-in-synthetic-data-generation) for details.

### Category Details

**Clarification Nudges**
- Triggered when: Entry is too vague to score (abstract emotion, vague pronouns, missing temporal anchors)
- Signal extracted: Behavioral grounding, disambiguation for value-dimension mapping
- Examples: "The meeting?", "Since when, roughly?", "What happened right before that?"

**Elaboration Nudges**
- Triggered when: Entry is substantive but surface-level (action without reflection, outcome without process)
- Signal extracted: Affective signal, agency/values expression, response patterns
- Examples: "And how did that land?", "What got you over the line?", "What did you end up doing?"

**Tension-Surfacing Nudges**
- Triggered when: Entry contains hedging, contradictions, or justification language
- Signal extracted: Unresolved tension, alignment/misalignment confirmation
- Examples: "What's the 'sort of' part?", "Does that sit okay with you?", "What stopped you?"

## Nudge Design Principles

### The "Thoughtful Companion" Test
Would a close friend who genuinely cares (but isn't a therapist) say this?

**Do:**
- Use simple curiosity: "What happened next?"
- Keep it brief: 2-12 words
- Reference something specific from the entry
- Match the user's register (casual entry = casual nudge)

**Don't:**
- Therapy speak: "I'm sensing...", "It sounds like..."
- Coaching framing: "Have you considered..."
- Value-labeling: "That conflicts with your health goals"
- Lead the witness: "That must have been frustrating"

### Voice Guidance (Prompt-Based)

Rather than using a post-hoc blacklist to reject bad nudges, we bake voice guidance directly into the prompt. This is more effective because the LLM understands *why* certain phrases feel wrong, not just *which* phrases to avoid.

**The "Close Friend" Test**: Would this feel weird if a friend texted it to you? If yes, don't write it.

**Anti-patterns explained in the prompt**:

| Anti-pattern | Examples | Why it fails |
|--------------|----------|--------------|
| Starting with "I" | "I notice...", "I'm sensing...", "I'm curious..." | Performative listening — makes the app feel like it's playing therapist |
| Reflective statements as questions | "It sounds like you're feeling..." | Creates uneven power dynamic (app as expert, user as patient) |
| Coaching language | "Have you considered...", "What would it look like if..." | Implies the app *knows better* — shifts from curious companion to prescriptive coach |
| Generic filler | "Tell me more about..." | Default LLM response that doesn't demonstrate the system actually "read" the entry |

**Why prompt guidance beats blacklists**: A blacklist catches specific strings but misses variants ("I'm noticing..." vs "I notice..."). Prompt guidance teaches the model the *principle*, so it avoids the entire category of performative language.

### Nudge Length Guidelines

| Entry Verbosity | Nudge Length |
|-----------------|--------------|
| Short (1-3 sentences) | 2-6 words |
| Medium (1-2 paragraphs) | 5-12 words |
| Long (detailed reflection) | Rarely nudge; if so, 5-15 words |

## Trigger Logic

### Decision Tree (LLM-Based Classification)

```
Entry Received
     │
     ▼
[Session cap: 2+ nudges in last 3 entries?] ──YES──► Skip (anti-annoyance, code-based)
     │
    NO
     ▼
[LLM classifies entry using nudge_decision.yaml prompt]
     │
     ├──► "no_nudge" — Entry is complete and grounded → Skip
     │
     ├──► "clarification" — Entry too vague to understand → Clarification Nudge
     │
     ├──► "elaboration" — Solid entry with unexplored depth → Elaboration Nudge
     │
     └──► "tension_surfacing" — Hints at unresolved conflict → Tension-Surfacing Nudge
```

> **Note**: The decision tree uses **content-only signals**. The session cap is enforced in code, then LLM classification handles semantic analysis. All synthetic metadata dependencies (tone, verbosity, reflection_mode) have been removed. See [Lesson Learned: Metadata Leakage](#lesson-learned-metadata-leakage-in-synthetic-data-generation).

> **Design change**: The original rule-based approach (word count thresholds, hedging regex, random gates) was replaced with LLM-based classification for better semantic understanding. The `prompts/nudge_decision.yaml` template provides criteria for each category.

### Classification Criteria (LLM-Based)

The LLM evaluates entries against semantic criteria defined in `prompts/nudge_decision.yaml`:

| Category | Semantic Criteria |
|----------|-------------------|
| `no_nudge` | Entry is complete, grounded, and provides sufficient signal |
| `clarification` | Entry too vague—abstract emotions, vague pronouns, missing temporal anchors |
| `elaboration` | Substantive but surface-level—action without reflection, outcome without process |
| `tension_surfacing` | Contains hedging, contradictions, justification language, or unresolved conflict |

All classification uses **content-only signals** available at inference time. No synthetic metadata (tone, verbosity, reflection_mode) is used.

### Anti-Annoyance Rules

1. **Frequency cap**: Maximum 1 nudge per entry (never chain nudges)
2. **Session cap**: If 2 nudges in last 3 entries, skip (code-based policy applied before LLM classification)

> **Note**: The original design included "Mood sensitivity" (skip nudging for exhausted/emotional entries based on `tone`) and a 40% random gate. These were removed—`tone` is synthetic metadata unavailable in production, and LLM classification provides more nuanced selection than random sampling. See [Lesson Learned: Metadata Leakage](#lesson-learned-metadata-leakage-in-synthetic-data-generation).

## Implementation: LLM-Based Approach

**Code** enforces session cap (2+ nudges in last 3 entries → skip).
**LLM** classifies entry into `no_nudge`, `clarification`, `elaboration`, or `tension_surfacing` using `prompts/nudge_decision.yaml`.
**LLM** generates the *natural language* nudge (with voice guidance baked into `prompts/nudge_generation.yaml`).
**Code** validates output length (voice quality is handled by prompt guidance).

### LLM Nudge Generation Prompt (Template)

```jinja2
You are generating a brief follow-up for a journaling app.

## Context
User's entry: {{ entry_content }}
Entry date: {{ entry_date }}
Nudge category: {{ nudge_category }}
{% if previous_entries %}
Recent entries (for context):
{% for prev in previous_entries %}
- {{ prev.date }}: {{ prev.content[:150] }}{% if prev.content|length > 150 %}...{% endif %}
{% endfor %}
{% endif %}

## Your Task
Generate a SHORT follow-up question ({{ min_words }}-{{ max_words }} words).

**Voice**: You're a close friend who just read their text and is firing back a quick question. Not a therapist, not a coach, not an app trying to sound empathetic.

**The test**: Would this feel weird if a friend texted it to you? If yes, don't write it.

**Anti-patterns to avoid**:
- Starting with "I" (e.g., "I notice...", "I'm sensing...", "I'm curious...") — this is performative listening
- Reflective statements disguised as questions ("It sounds like you're feeling...")
- Coaching language ("Have you considered...", "What would it look like if...")
- Invitations that feel like work ("Tell me more about...")

**Good questions are**:
- Direct, even blunt
- Reference a specific detail from the entry (a person, event, phrase they used)
- Short enough to type in 2 seconds
- The kind of thing you'd say without thinking

## Examples by Category
{% if nudge_category == 'clarification' %}
- "What happened right before that?"
- "The meeting?"
- "Since when?"
{% elif nudge_category == 'elaboration' %}
- "And how did that land?"
- "What did you end up doing?"
- "What got you over the line?"
{% elif nudge_category == 'tension_surfacing' %}
- "What's the 'sort of' part?"
- "Does that sit okay?"
- "What stopped you?"
{% endif %}

## Output
Return ONLY valid JSON:
{"nudge_text": "your question here"}
```

## Data Model Changes

### New Models for Conversational Pipeline

```python
class NudgeResult(BaseModel):
    """Generated nudge with metadata."""
    nudge_text: str
    # Note: "grounding" was removed - relied on reflection_mode metadata (not available in production)
    nudge_category: Literal[
        "clarification",
        "elaboration",
        "tension_surfacing"
    ]
    trigger_reason: str  # Why this nudge was generated
    was_responded_to: bool = False


class JournalTurn(BaseModel):
    """A single turn in the conversation (entry or response)."""
    date: str
    content: str
    turn_type: Literal["initial_entry", "nudge_response"]
    responding_to_nudge: str | None = None  # The nudge text if this is a response


class ConversationalEntry(BaseModel):
    """Complete conversational exchange for one journaling session."""
    initial_entry: JournalEntry
    nudge: NudgeResult | None
    response: JournalTurn | None  # User's response to the nudge
    # Metadata
    tone: str
    verbosity: str
    reflection_mode: str
```

### Updated PersonaPipelineResult

```python
class PersonaPipelineResult(BaseModel):
    """Updated to support conversational entries."""
    persona_id: int
    persona: Persona | None
    entries: list[ConversationalEntry]  # Changed from list[JournalEntryResult]
    persona_prompt: str | None
    entry_prompts: list[str]
    error: str | None
```

## Pipeline Flow Changes

### Before (One-Way)

```
Persona → Entry_1 → Entry_2 → Entry_3 → [Judge scores each]
```

### After (Conversational)

```
Persona → Entry_1 → [Nudge Decision] → Nudge_1? → Response_1? → Entry_2 → ...
                          │                 │            │
                          ▼                 ▼            ▼
                    [stored]          [stored]    [Judge scores session]
```

### Updated Generation Function

```python
async def generate_conversational_entry(
    persona: Persona,
    config: dict,
    date_str: str,
    previous_entries: list[ConversationalEntry] | None = None,
) -> ConversationalEntry:
    """Generate entry, decide on nudge, optionally generate response."""

    # Step 1: Generate initial entry (existing logic)
    entry_result, prompt = await generate_journal_entry(
        persona, config, date_str,
        previous_entries=[e.initial_entry for e in (previous_entries or [])]
    )

    # Step 2: Decide whether to nudge (content-only signals)
    should_nudge, nudge_category, trigger_reason = decide_nudge_llm(
        entry=entry_result.entry,
        previous_entries=previous_entries,
        config=config
    )

    nudge_result = None
    response = None

    if should_nudge:
        # Step 3: Generate nudge
        nudge_text = await generate_nudge(
            entry=entry_result.entry,
            category=nudge_category,
            previous_entries=previous_entries,
            persona=persona
        )

        nudge_result = NudgeResult(
            nudge_text=nudge_text,
            nudge_category=nudge_category,
            trigger_reason=trigger_reason
        )

        # Step 4: Decide if persona responds (probabilistic)
        if random.random() < config["nudge"]["response_probability"]:
            response = await generate_nudge_response(
                persona=persona,
                original_entry=entry_result.entry,
                nudge=nudge_result,
                config=config
            )
            nudge_result.was_responded_to = True

    return ConversationalEntry(
        initial_entry=entry_result.entry,
        nudge=nudge_result,
        response=response,
        tone=entry_result.tone,
        verbosity=entry_result.verbosity,
        reflection_mode=entry_result.reflection_mode
    )
```

## Configuration Additions

Add to `config/synthetic_data.yaml`:

```yaml
nudge:
  # Probability that the persona responds to a nudge
  response_probability: 0.7

  # Note: category_weights were removed - LLM classification now determines
  # nudge category semantically using prompts/nudge_decision.yaml
  # Note: base_probability (40% random gate) was removed - LLM classification
  # provides more nuanced selection than random sampling

  # Response modes (simplified from original 5 to 3)
  response_modes:
    - mode: "Answering directly"
      weight: 0.50
      description: "Clear, helpful response to the question"
    - mode: "Deflecting/redirecting"
      weight: 0.30
      description: "Brief acknowledgment or topic change ('Yeah, just the usual')"
    - mode: "Revealing deeper thought"
      weight: 0.20
      description: "Unexpected honesty or vulnerability"
```

> **Note**: The original design had 5 response modes including "Elaborating with context" (25%) and "Brief acknowledgment" (5%). These were removed to reduce complexity - the 5% mode was almost never triggered, and "Elaborating with context" overlapped with "Answering directly".

> **Design change**: The original `category_weights` and `base_probability` were removed. LLM-based classification now determines nudge category semantically, providing better signal detection than probabilistic selection.

## Judge Scoring Updates

### Combined Signal Approach

The Judge scores the entire conversational session as a single unit, not the initial entry and response separately. This uses a **max-signal** approach: if the initial entry was vague but the response revealed misalignment, the combined score reflects the revealed misalignment.

```python
class JudgeLabelConversational(BaseModel):
    """Judge scores the entire session as a unit."""
    combined_alignment_vector: list[int]  # 10-dim, {-1, 0, +1} per value
    # Max-signal logic: if initial was neutral but response revealed misalignment,
    # the combined score reflects the revealed misalignment

    nudge_effectiveness: Literal["none", "low", "medium", "high"]
    # "none" = no nudge given
    # "low" = nudge was deflected or minimally answered
    # "medium" = nudge extracted some additional signal
    # "high" = nudge revealed major alignment/misalignment

    primary_signal_source: Literal["initial", "response", "both"]
    # Where did the alignment signal primarily come from?
```

### Judge Prompt Addition

```
Score this journaling session as a whole. The user's response to the nudge
may reveal alignment signals not present in the initial entry. Use the
strongest signal available (if initial was vague but response clarified,
score based on the clarified content).

Rate nudge_effectiveness as:
- "none": No nudge was given
- "low": User deflected or gave minimal response
- "medium": Response added useful context
- "high": Response revealed significant alignment/misalignment not in original
```

## VIF State Vector Updates

For entries with nudge-responses, extend the state vector:

```python
s_u_t = Concat[
    phi_text(initial_entry),
    phi_text(response),           # NEW: response embedding (zero vector if no response)
    has_nudge_response,           # NEW: binary flag
    nudge_category_onehot,        # NEW: 5-dim (or zero if no nudge)
    # ... existing features (time gaps, history stats, user profile)
]
```

This allows the Critic to learn:
- Whether nudge-augmented entries have different alignment patterns
- Which nudge categories are most effective for extracting signal
- How to weight initial vs response content

## Expected Outcomes

| Metric | Before (One-Way) | After (Conversational) | Improvement |
|--------|------------------|------------------------|-------------|
| Average words per session | 80-120 | 120-200 | +50-67% |
| Entries with clear value signal | ~60% | ~80% | +33% |
| Entries with multiple value signals | ~30% | ~50% | +67% |
| Entries scoreable with low uncertainty | ~70% | ~85% | +21% |

### Key VIF Training Benefits

1. **Better calibration for short entries**: Even brief initial entries can be augmented with nudge-responses
2. **Explicit tension detection training**: Tension-surfacing nudges create labeled examples of tensions being confirmed or denied
3. **Trajectory continuity**: Continuity nudges create explicit links between entries for sequence learning
4. **"When to ask" calibration**: High-uncertainty entries that become clear after nudge provide ground truth for active learning

## Implementation Status

**Current**: Notebook-first development in `notebooks/journal_nudge.ipynb`

**Phase 1 (Foundation)**: ✅ Complete
- [x] Set up output logging system (directory structure, utility functions)
- [x] Add `logs/synthetic_data/` to `.gitignore`
- [x] Add nudge config to `config/synthetic_data.yaml`
- [x] Implement nudge decision logic (LLM-based classification)
- [x] Add `NudgeResult`, `ConversationalEntry` data models

**Phase 2 (Generation)**: ✅ Complete
- [x] Implement LLM nudge generation with validation
- [x] Implement nudge-response generation for personas
- [x] Update `generate_persona_pipeline()` to use conversational flow

**Phase 3 (Validation)**: 🔄 In Progress
- [ ] Generate 10-20 personas with conversational entries
- [ ] Review nudge quality, response variety, signal improvement
- [ ] Iterate on prompts and parameters

**Phase 4 (Integration)**:
- [ ] Update Judge prompt for multi-turn scoring
- [ ] Test end-to-end pipeline
- [ ] Tune probabilities and voice guidance prompts

**Phase 5 (Future - Productionize)**:
- [ ] Once notebook output is satisfactory, extract to Python scripts
- [ ] Add unit tests
- [ ] Formalize data models in `src/` directory

---

## Judge Labeling Pipeline (`/judge` Skill)

Once synthetic data is generated, the `/judge` skill orchestrates scoring journal entries against Schwartz value dimensions.

### Pipeline Overview

```
Synthetic Data → Wrangling → Judge Labeling → Consolidation → Training Data
     ↓               ↓              ↓               ↓
logs/synthetic_data/ → logs/wrangled/ → logs/judge_labels/*.json → judge_labels.parquet
```

### Running the Pipeline

Execute `/judge` in Claude Code. The skill:

1. **Registry Check**: Queries `src.registry.get_status()` to find pending work
2. **Auto-Wrangle**: If `pending_wrangling > 0`, runs `python -m src.wrangling.parse_synthetic_data`
3. **Parallel Labeling**: Spawns one Task subagent per pending persona
4. **Validation**: Pydantic models (`src/models/judge.py`) validate JSON output
5. **Consolidation**: Merges all labels into `logs/judge_labels/judge_labels.parquet`

### Subagent Context

Each labeling subagent receives:
- **Persona file**: `logs/wrangled/persona_{id}.md` — the entries to score
- **Annotation guide**: `.claude/skills/judge/annotation_guide.md` — scorability heuristics
- **Value rubric**: `.claude/skills/judge/rubric.md` — Schwartz value definitions

### Output Format

Each subagent writes JSON to `logs/judge_labels/persona_{id}_labels.json`:

```json
{
  "persona_id": "8-char-hex",
  "labels": [
    {
      "t_index": 0,
      "date": "YYYY-MM-DD",
      "scores": {
        "self_direction": 0, "stimulation": 0, "hedonism": 0,
        "achievement": 1, "power": 0, "security": 0,
        "conformity": 0, "tradition": 0, "benevolence": -1,
        "universalism": 0
      }
    }
  ]
}
```

### Idempotency

The pipeline is idempotent — re-running `/judge` only processes pending work. The registry (`logs/registry/personas.parquet`) tracks which personas have been wrangled and labeled.

### Key Files

| File | Purpose |
|------|---------|
| `.claude/commands/judge.md` | Skill entry point |
| `.claude/skills/judge/orchestration.md` | Detailed workflow with subagent prompt template |
| `.claude/skills/judge/annotation_guide.md` | Scorability checklist and calibration examples |
| `.claude/skills/judge/rubric.md` | 10 Schwartz values with aligned/misaligned behaviors |
| `src/models/judge.py` | Pydantic validation models |
| `src/judge/consolidate.py` | Merges JSON to parquet, updates registry |

---

## Lesson Learned: Metadata Leakage in Synthetic Data Generation

### The Problem

During initial implementation, the nudge decision logic used **synthetic generation metadata** that would not be available in a production system. This is a form of "overfitting" to the synthetic data generation process.

### What Was Initially Implemented (Incorrect)

```python
def decide_nudge_llm(
    entry: JournalEntry,
    reflection_mode: str,
    tone: str,  # ← PROBLEM: Synthetic metadata
    previous_entries: list[ConversationalEntry] | None,
    config: dict,
) -> tuple[bool, NudgeCategory | None, str | None]:

    # Guard 1: Mood sensitivity based on TONE
    if tone in ["Exhausted", "Emotional/Venting"]:
        if random.random() > 0.2:  # 80% skip
            return False, None, None  # ← Using unavailable data

    # ... rest of logic
```

The `tone` parameter was randomly assigned *before* entry generation as an instruction to the LLM ("write in an exhausted tone"). The nudge decision then checked this label to decide whether to skip nudging.

**Why this is wrong**: In production:
1. A user writes a journal entry
2. We have the entry **content** only
3. We do NOT have a pre-labeled "tone" - we'd need to **detect** it from content
4. Using the synthetic label is tantamount to cheating

### The Distinction: Generation Metadata vs Observable Data

| Data Type | Example | Available in Production? | Use in Nudge Decision? |
|-----------|---------|-------------------------|------------------------|
| Entry content | "Had a rough day..." | ✅ Yes | ✅ OK |
| Previous entries | History of entries | ✅ Yes | ✅ OK |
| Nudge history | Which entries got nudges | ✅ Yes | ✅ OK |
| **Tone** (generation instruction) | "Exhausted" | ❌ No | ❌ Wrong |
| **Verbosity** (generation instruction) | "Short" | ❌ No | ❌ Wrong |

### What Was Corrected

1. **Removed tone-based guard**: The 80% skip for exhausted/emotional entries was removed entirely. This guard used data we wouldn't have in production.

2. **Removed continuity nudge category**: The continuity detection used fragile word-overlap heuristics that added complexity without clear benefit for POC scope.

3. **Simplified response modes**: Reduced from 5 to 3 modes, removing near-zero probability options.

4. **Removed all synthetic metadata**: The `decide_nudge_llm()` function signature was simplified to only accept content-based inputs:

```python
# Step 2: Decide whether to nudge (content-only signals)
should_nudge, nudge_category, trigger_reason = decide_nudge_llm(
    entry=entry,
    previous_entries=previous_entries,
    config=config,
)
```

### Metadata Leakage: Fully Resolved

All synthetic metadata dependencies have been removed from the nudge decision logic:

| Removed Check | Why Removed |
|---------------|-------------|
| `tone` guard (Exhausted/Emotional skip) | Synthetic generation instruction, not observable |
| `reflection_mode == "Neutral"` check | Synthetic generation instruction, not observable |
| `reflection_mode == "Unsettled"` requirement | Synthetic generation instruction, not observable |
| `reflection_mode == "Grounded"` check | Synthetic generation instruction, not observable |

**Current implementation**: The `decide_nudge_llm()` function now uses **only** content-based signals:
- Entry word count
- Presence of concrete details (nouns/verbs)
- Hedging language patterns
- Previous nudge history

This ensures the decision logic works identically during synthetic data generation and production inference.

### General Principle

When building synthetic data pipelines, be vigilant about the distinction between:
- **Generation instructions** (what we tell the LLM to produce)
- **Observable outputs** (what we'd actually have in production)

Never use generation instructions as inputs to downstream decision logic - this creates a form of data leakage that makes the synthetic pipeline unrealistic.

---

## Output Logging System

### Purpose

Enable iterative development by logging all synthetic data output for inspection and quality review. This supports both human developers tinkering with the notebook and Claude-assisted iteration.

### Iterative Development Loop

```
1. Implement changes in notebook
         │
         ▼
2. Run notebook → output stored in timestamped log folder
         │
         ▼
3. Inspect logs to verify quality (Claude or human)
         │
         ▼
4. Quality acceptable? ──YES──► Done
         │
        NO
         │
         ▼
5. Make changes → go to step 2
```

### Log Configuration

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **Format** | Markdown | Human-readable, easy to review in IDE/GitHub |
| **Contents** | Output + prompts + config | Full reproducibility for debugging |
| **Organization** | Timestamped folders | Each run is isolated, history preserved |
| **Summary** | No separate file | Review persona files directly |

### Log Directory Structure

```
logs/synthetic_data/
├── 2024-01-15_14-30-00/
│   ├── config.md           # Config/parameters used for this run
│   ├── persona_001.md      # Persona + all entries + nudges + responses
│   ├── persona_002.md
│   ├── persona_003.md
│   └── prompts.md          # All prompts sent to LLM (for debugging)
├── 2024-01-15_16-45-22/
│   └── ...
└── 2024-01-16_09-12-05/
    └── ...
```

### Per-Persona Log Format

Each persona gets its own markdown file with complete output:

```markdown
# Persona 001: [Name]

## Profile
- Age: 32
- Profession: Software Engineer
- Culture: East Asian
- Core Values: Self-Direction, Achievement
- Bio: [generated bio]

---

## Entry 1 - 2024-01-15

### Initial Entry
**Tone**: Self-reflective | **Verbosity**: Medium | **Reflection Mode**: Unsettled

[entry content here]

### Nudge (Elaboration)
**Trigger**: Action without reflection detected

"And how did that land?"

### Response
**Mode**: Answering directly

[response content here]

---

## Entry 2 - 2024-01-18

### Initial Entry
**Tone**: Brief and factual | **Verbosity**: Short | **Reflection Mode**: Neutral

[entry content here]

*(No nudge for this entry)*

---

## Entry 3 - 2024-01-22
...
```

### Config Log Format

The `config.md` file captures all parameters used:

```markdown
# Run Configuration

**Timestamp**: 2024-01-15 14:30:00
**Notebook**: journal_gen.ipynb

## Persona Generation
- Num personas: 5
- Entries per persona: 3

## Nudge Settings
- Base probability: 0.4
- Response probability: 0.7
- Category weights: clarification=0.25, elaboration=0.30, ...

## Model Settings
- Model: gpt-4o-mini
- Reasoning effort: medium
```

### Prompts Log Format

The `prompts.md` file captures all LLM prompts for debugging:

```markdown
# Prompts Log

## Persona 001

### Persona Generation Prompt
[full prompt text]

### Entry 1 - Initial Entry Prompt
[full prompt text]

### Entry 1 - Nudge Generation Prompt
[full prompt text]

### Entry 1 - Response Generation Prompt
[full prompt text]

...
```

### Implementation Notes

- **Additive, not replacement**: Notebook retains interactive display for human users. Logging is in addition to notebook output.
- **Write-only**: New runs create new timestamped folders. Never overwrite previous logs.
- **Timestamp format**: `datetime.now().strftime("%Y-%m-%d_%H-%M-%S")`
- **Gitignore**: Add `logs/synthetic_data/` to `.gitignore` to avoid bloating the repo.

*Implementation tasks are tracked in the [Implementation Status](#implementation-status) section above.*

---

## Technical Notes

- **Model**: gpt-5-mini via OpenAI Responses API
- **Async client**: `AsyncOpenAI` for non-blocking I/O operations
- **Reasoning effort**: Configurable (minimal/low/medium/high) - temperature not supported by gpt-5 models
- **Structured output**: JSON schemas with `strict: True` for reliable parsing
- **Retry logic**: Up to 2 attempts per generation with validation

### Usage

```python
# Single persona (await directly in Jupyter)
result = await generate_persona_pipeline(
    persona_id=1,
    config=config,
    schwartz_config=schwartz_config,
    num_entries=3,
    start_date="2023-10-27"
)
display_persona_results(result)

# Multiple personas in parallel
results = await run_parallel_personas(
    num_personas=3,
    config=config,
    schwartz_config=schwartz_config,
    num_entries=3
)
for result in results:
    display_persona_results(result)
```

By systematically varying persona profiles and prompts, we obtain a synthetic dataset that covers a broad spectrum of human experiences, crucial for training alignment models that generalize well.

**Neutral entries** are equally important. In real life, not every journal entry shows clear movement toward or away from one's sense of self. Sometimes a user writes a neutral update or routine reflection. We include entries where the reflection mode is "Neutral"—these provide baseline contrast against unsettled/grounded entries.

# Stretch Goals

- **Style verifier pass:** After generation, run a second pass that checks if the entry reads like a real journal (not an essay, not performative), and rewrites it into a more natural journaling voice while preserving the underlying event + emotions.
