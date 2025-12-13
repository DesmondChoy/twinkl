# Purpose

Create high-quality data to bootstrap (start from nothing) what Twinkl requires - data that enables value tagging (labeling text with Schwartz’s value dimensions) and training a reward model (Critic).

Subsequently, an evaluation framework is required to detect bias and toxic content.

# Objectives

Journals should be:
- Realistic (Genuine personal reflections)
- Diverse in personas and scenarios
- Contain ground-truth alignment signals
- Longitudinal in nature to exhibit value drive, conflicts and ambiguity


# Prompt Design

Config file with parameters? Composable.

Note: For now, synthetic journals should read like typed text input (not voice-note transcripts). A voice-note transcript variant can be added later.

Determined once during persona creation
- Age
- Culture/Nationality
- Profession
- Background/Scenario (in-depth)
- Schwartz’s values

Determined before each journal entry
- Time of day
- Tone (Conversational, Self-reflective, Brief and factual, Emotional/Venting, Analytical, Stream of consciousness)
- Verbosity level
- Value drift (Drift, Convergence, Unchanged)

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
- **Value drift** - drift vs convergence vs unchanged drives the nature of reflection
- **Time of day** - late night vs lunch break implies different energy and candor

This leaner approach lets journal content emerge organically from persona context rather than forcing artificial categories.

Note: Temperature parameter is not supported by gpt-5 models in the Responses API.

Alternate persona attributes (different ages, cultures, professions, value priorities) or even ask the LLM to “reflect from a different perspective” to avoid homogeneous outputs.

By systematically varying persona profiles and prompts, we obtain a synthetic dataset that covers a broad spectrum of human experiences, crucial for training alignment models that generalize well.

Journals are time-sequenced, so adding a simple timestamp or day indicator can contextualize entries. More importantly, check that entries for the same persona do not contradict each other (unless intentionally simulating changing minds).

**Unchanged entries** are equally important. In real life, not every journal entry shows clear movement toward or away from values. Sometimes a user writes a neutral update or routine reflection. We include entries where value drift is minimal—these provide baseline contrast against drift/convergence entries.

# Stretch Goals

- **Style verifier pass:** After generation, run a second pass that checks if the entry reads like a real journal (not an essay, not performative), and rewrites it into a more natural journaling voice while preserving the underlying event + emotions.
