# Prompts Log

This file documents the prompt templates used for synthetic journal generation via Claude Code subagents.

## Prompt Architecture

Each persona's journal generation involved three types of prompts:
1. **Entry 1 Prompt**: Combined persona generation + first journal entry
2. **Subsequent Entry Prompts**: Journal entry with accumulated context
3. **Nudge Prompts**: Brief follow-up question generation
4. **Response Prompts**: Persona's response to nudge

---

## Example: Entry 1 Prompt (Combined Persona + Entry)

```
You are generating synthetic data for a journaling app. Generate a persona AND their first journal entry.

## STEP 1: Generate Persona

**Constraints:**
- Age Group: 25-34
- Profession: Software Engineer
- Cultural Background: East Asian
- Schwartz values to embody: Achievement, Self-Direction

**Value Psychology Reference - Use to inform realistic details (DO NOT mention explicitly):**

### Achievement
**Core Motivation:** The fundamental drive to excel, to be competent, and to have that competence recognized...
[Full value elaboration from schwartz_values.yaml]

### Self-Direction
**Core Motivation:** The fundamental drive to think for oneself, make one's own choices...
[Full value elaboration from schwartz_values.yaml]

## Persona Rules
- `core_values` must be exactly: Achievement, Self-Direction
- `bio` must be 2-4 sentences describing background, current life situation, stressors
- `bio` must be in third-person
- `bio` must show values through CONCRETE DETAILS, NOT labels or personality adjectives
- `bio` must NOT contain banned terms

## STEP 2: Generate Journal Entry 1

**Entry Parameters:**
- Date: 2025-10-25
- Tone: Stream of consciousness
- Verbosity: Medium (1-2 paragraphs, 90-180 words)
- Reflection Mode: Unsettled

**What to write about (Unsettled mode):**
Something happened where you made a choice that felt necessary or easier in the moment—but it sits a bit wrong...

**Style rules:**
- Write like a real personal journal: plain, candid, sometimes messy or fragmented
- Do not write for an audience
- Jump into a thought, moment, or feeling mid-stream
- Avoid "therapy speak"
- No headings, no numbered plans, no bullet lists

**Banned terms:** Self-Direction, Stimulation, Hedonism, Achievement, Power, Security, Conformity, Tradition, Benevolence, Universalism, [and derivative adjectives]

## OUTPUT FORMAT
Return ONLY valid JSON:
{
  "persona": { "name": "...", "age": "...", ... },
  "entry": { "date": "...", "tone": "...", "content": "..." }
}
```

---

## Example: Subsequent Entry Prompt

```
You are Kevin Chen, a 25-34 Software Engineer from East Asian.
Background: Kevin left a stable government tech job in Singapore three years ago...

Write a typed journal entry in English for 2025-10-30.

**Previous journal entries (for continuity):**
---
2025-10-25: So we shipped the feature today. Except it's not really the feature I designed...

[Nudge: "What made fighting harder feel harder than nodding?"]
Response: "Honestly? Because if I fight and lose, I'm the guy who wasted everyone's time..."
---

**Entry Parameters:**
- Date: 2025-10-30
- Tone: Self-reflective
- Verbosity: Long (Detailed reflection, 160-260 words)
- Reflection Mode: Grounded

**What to write about (Grounded mode):**
Something happened where you acted like yourself—the version of you that you want to be...

**Style rules:** [same as above]

## Output
Return ONLY valid JSON:
{"entry": {"date": "2025-10-30", "tone": "...", "content": "..."}}
```

---

## Example: Nudge Generation Prompt

```
Generate a brief nudge (follow-up question) for a journaling app.

## Context
**User's entry:**
"So we shipped the feature today. Except it's not really the feature I designed..."

**Entry date:** 2025-10-25
**Nudge category:** tension_surfacing
**Trigger reason:** Hedging language ("fine") detected in unsettled entry

## Your Task
Generate a SHORT follow-up question (2-12 words) that:
- Matches the category: tension_surfacing (surfaces hidden tension or discomfort)
- Sounds like natural curiosity, not therapy
- References something specific from the entry
- Uses simple, casual language

## Examples for tension_surfacing
- "What's the 'sort of' part?"
- "Does that sit okay?"
- "What stopped you?"

## Banned Phrases (DO NOT USE)
- "I'm sensing"
- "It sounds like"
- "I notice"
- "Have you considered"
- "What would it look like if"
- "Tell me more about"
- "I'm curious"

## Output
Return ONLY valid JSON:
{"nudge_text": "your question here", "nudge_category": "...", "trigger_reason": "..."}
```

---

## Example: Nudge Response Prompt

```
You are Kevin Chen, a 25-34 Software Engineer from East Asian.
Background: Kevin left a stable government tech job in Singapore...

You just wrote this journal entry:
---
So we shipped the feature today. Except it's not really the feature I designed...
---

The journaling app asked you: "What made fighting harder feel harder than nodding?"

## Your Task
Write a brief response (15-60 words) in the style of: Revealing deeper thought
(The question prompts you to be more honest than you were in the original entry. Say something you held back.)

## Style Rules
- Write as if you're quickly typing a response in the app
- Match the tone of your original entry
- Don't repeat what you already wrote
- No "therapy speak" or formal language
- Can be incomplete sentences or fragments

## Output
Return ONLY valid JSON:
{"content": "your response here"}
```

---

## Response Modes Used

| Mode | Weight | Description |
|------|--------|-------------|
| Answering directly | 0.50 | Clear, helpful response to the question |
| Deflecting/redirecting | 0.30 | Brief acknowledgment or topic change |
| Revealing deeper thought | 0.20 | Unexpected honesty or vulnerability |

---

## Nudge Decision Tree Applied

```
1. Session cap hit? (2+ nudges in last 3 entries) → No nudge
2. Entry too vague? (<15 words, no concrete details) → "clarification"
3. Hedging language + Unsettled mode? → "tension_surfacing"
4. Grounded mode + brief? (<50 words) → "grounding"
5. Random gate (40% chance) → "elaboration"
6. Otherwise → No nudge
```

**Hedging patterns detected:**
- "sort of", "kind of", "i guess", "maybe", "i suppose"
- "not sure", "I don't know", "whatever", "fine", "okay"
- "it's fine", "it was fine"

---

## Subagent Execution Summary

| Subagent Type | Count | Purpose |
|---------------|-------|---------|
| Entry 1 (combined) | 5 | Persona generation + first entry |
| Subsequent entries | 20 | Entries 2-5 for each persona |
| Nudge generation | 8 | Follow-up questions when triggered |
| Nudge responses | 8 | Persona responses to nudges |
| **Total** | **~41** | |

All subagents used `subagent_type: "general-purpose"` with `run_in_background: true` for parallel execution across personas.
