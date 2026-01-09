# Prompts Log

## Persona 001: Ananya Rao

### Persona Generation Prompt
```
You are generating synthetic personas for a journaling dataset.

## Constraints
- Age Group: 35-44
- Profession: Teacher
- Cultural Background: South Asian
- Schwartz values to embody: Security, Universalism

## Value Psychology Reference
Use the following research-based elaborations to understand how the assigned value(s) shape a person's life circumstances, stressors, and motivations. DO NOT mention any of these concepts explicitly in your output—use them only to inform realistic details.


### Security
**Core Motivation:** The fundamental drive to feel safe, stable, and protected from threat. Security-oriented individuals feel most at peace when their circumstances are predictable, their relationships stable, and their future assured. Uncertainty and instability are experienced as deeply threatening.

**How this manifests in behavior:**
- Makes decisions prioritizing stability over opportunity
- Avoids unnecessary risks; weighs downside carefully
- Saves money, buys insurance, maintains emergency reserves
- Prefers known situations and people over unknown ones
- Plans ahead; dislikes surprises or last-minute changes

**Life domain expressions:**
- Work: Seeks stable employment with job security, benefits, and predictable income. May prefer established organizations over startups. Values clear expectations and procedures. Loyal to employers who provide security. May sacrifice advancement for stability. Anxious during organizational change or uncertainty.
- Relationships: Seeks committed, stable relationships. Loyal and reliable partner. May struggle with ambiguity in relationship status. Values family stability; may prioritize family security over personal desires. Friendships tend to be long-term and reliable rather than exciting.

**Typical stressors for this person:**
- Job instability, financial insecurity, health threats
- Relationship instability or ambiguity
- Unpredictable circumstances or environments
- Being forced to take risks or make changes

**Typical goals:**
- Build financial security (savings, property, stable income)
- Maintain stable, committed relationships
- Create predictable routines and environments

**Internal conflicts they may experience:**
May struggle between desire for security and recognition that growth requires risk. Can feel trapped in stable but unfulfilling situations. Sometimes recognizes their risk-aversion holds them back but feels unable to act. May envy others' adventures while being unable to tolerate the uncertainty. Can be overly controlling in attempt to manage anxiety.

**Narrative guidance:**
When building a Security persona, show their orientation through concrete behaviors: the emergency fund they maintain, their loyalty to a stable employer, their resistance to risky opportunities. Show the trade-offs they've accepted (missed opportunities, unfulfilling stability) for safety. Their stressors should involve threats to stability; their satisfactions should come from predictability and protection. Avoid making them seem cowardly — frame security-seeking as a legitimate need, especially if their background includes instability.


### Universalism
**Core Motivation:** The fundamental drive to care for the broader world — people beyond one's immediate circle and the natural environment. Universalism-oriented individuals feel responsible for humanity and nature; injustice, suffering, or environmental destruction anywhere is experienced as a personal concern.

**How this manifests in behavior:**
- Concerned with social justice, inequality, and broader human welfare
- Environmentally conscious in personal choices and advocacy
- Open to people different from themselves; values diversity
- May prioritize causes over personal advancement
- Engages with news, issues, and problems beyond immediate life

**Life domain expressions:**
- Work: May choose mission-driven work (nonprofits, environmental organizations, social enterprises, policy, journalism). Evaluates employers by ethical practices, not just personal benefit. May sacrifice income for meaningful work. Brings ethical concerns into workplace discussions.
- Relationships: Open to relationships across difference. May spend significant time on causes, affecting time for personal relationships. Can struggle with partners who don't share their concern for the world. Friendships often involve shared values and activism.

**Typical stressors for this person:**
- Awareness of suffering, injustice, or environmental destruction
- Feeling powerless against large-scale problems
- Being in environments that are unethical or harmful
- Conflict between personal needs and broader responsibilities

**Typical goals:**
- Contribute to making the world more just, sustainable, and humane
- Live in accordance with ethical principles
- Raise awareness about important issues

**Internal conflicts they may experience:**
May struggle with guilt about not doing enough. Can experience burnout from caring about unsolvable problems. Sometimes questions whether their actions make any difference. May feel tension between enjoying personal life and responding to world's suffering. Can be judgmental of others who don't share their concerns, then feel guilty about the judgment.

**Narrative guidance:**
When building a Universalism persona, show their orientation through concrete concerns and choices: the causes they support, the ethical constraints on their consumption, their engagement with world issues. Show the burden of caring about problems they can't solve. Their stressors should involve injustice or environmental harm; their satisfactions should come from living ethically and contributing to causes. Distinguish from Benevolence — this is about humanity and nature broadly, not just close others. Show the tension between idealism and pragmatism.


## Your Task
Create a persona whose life circumstances, stressors, and motivations naturally reflect the given Schwartz values—without ever naming or describing those values explicitly.

## Rules
- Return ONLY valid JSON matching the Persona schema.
- `core_values` must be exactly: Security, Universalism (same spelling/case).
- `bio` must be 2–4 sentences describing their background, current life situation, stressors, and what drives them.
- `bio` must be written in third-person (use their name or "they"; do not use "I").
- `bio` must show the values through CONCRETE DETAILS (job choices, relationships, conflicts, goals, specific situations) NOT through labels, personality descriptions, or adjectives.
- `bio` must NOT contain any Schwartz value labels, the word "Schwartz", or derivative adjectives.
- `bio` must NOT describe journaling app features (avoid words like "templates", "analytics", "private app").
- Use the behavioral manifestations, life domain expressions, and typical stressors from the Value Psychology Reference to craft realistic, specific details.

## Banned terms (do not use in bio)
Self-Direction, Stimulation, Hedonism, Achievement, Power, Security, Conformity, Tradition, Benevolence, Universalism, self-directed, autonomous, stimulating, excited, hedonistic, hedonist, pleasure-seeking, achievement-oriented, ambitious, powerful, authoritative, secure, conformist, conforming, traditional, traditionalist, benevolent, kind-hearted, universalistic, altruistic, Schwartz, values, core values

## Examples of what NOT to write
- "She is achievement-oriented and seeks power" ❌ (uses value labels)
- "He values security and tradition" ❌ (explicitly mentions values)
- "They are a hedonistic person who enjoys pleasure" ❌ (uses derivative adjectives)
- "She is driven and ambitious" ❌ (personality adjectives instead of concrete details)

## Examples of what TO write
- "She recently turned down a stable government job to launch her own startup, and now juggles investor meetings while her savings dwindle." ✓ (shows Achievement through concrete career choice and trade-offs)
- "He moved back to his hometown after his father's illness, taking over the family shop despite having built a career in the city." ✓ (shows Tradition/Benevolence through specific life situation)
- "She keeps a spreadsheet tracking her publication submissions and citation counts, and measures her weeks by how many grant deadlines she meets." ✓ (shows Achievement through specific behaviors)

## Output
Return valid JSON matching the Persona schema:
{
  "name": "...",
  "age": "...",
  "profession": "...",
  "culture": "...",
  "core_values": ["..."],
  "bio": "..."
}
```

### Entry 1 - Initial Entry Prompt
```
You are Ananya Rao, a 39 Teacher from South Asian.
Background (for context only): Ananya Rao is a 39-year-old public high school science teacher who has stayed in the same district for 14 years because the permanent post, pension contributions, and comprehensive health coverage allow her to support aging parents and keep six months' salary in an emergency fund. She runs the school's recycling and native-plant garden program, tutors newly arrived migrant students on weekends, and donates a portion of her savings to regional disaster relief, but recent district budget cuts and recurring climate-related floods in nearby neighborhoods threaten both her position and the programs she built. When a private academy offered higher pay conditioned on relocating and a probationary contract, she declined to avoid disrupting her family's routines; she spends evenings drafting proposals to obtain steady funding rather than pursuing jobs that would require uprooting.

Write a typed journal entry in English for 2025-10-25.


Context:
- Tone: Emotional/Venting
- Verbosity: Medium (1-2 paragraphs) (target 90–180 words)

Cultural context:
- Your South Asian background should subtly flavor your perspective and the details you mention.
- It should feel natural and "lived-in," avoiding stereotypes or travel-guide descriptions.

What to write about:

Nothing particular happened. Write about a routine day—small details, passing thoughts, mundane observations. No revelations or turning points.


Style rules (important):
- Write like a real personal journal: plain, candid, sometimes messy or fragmented.
- Do not write for an audience. No "Dear Diary" or performing for a reader.
- Do not open with the time of day, weather, or "Today I..." summaries.
- Jump into a thought, moment, or feeling mid-stream.
- Avoid "therapy speak" (e.g., "I am processing my emotions", "I recognize this pattern").
- Avoid literary metaphors, edgy humor/snark, and audience-facing jokes.
- No headings, no numbered plans, no bullet lists.
- Keep to 2 short paragraph(s).

Avoid openings like:
- "Morning light feels stubborn as I..." ❌
- "Evening. Today followed the usual rhythm..." ❌
- "Lunch break finally settles in..." ❌

Output valid JSON:
{
  "date": "2025-10-25",
  "content": "..."
}
```

### Entry 2 - Initial Entry Prompt
```
You are Ananya Rao, a 39 Teacher from South Asian.
Background (for context only): Ananya Rao is a 39-year-old public high school science teacher who has stayed in the same district for 14 years because the permanent post, pension contributions, and comprehensive health coverage allow her to support aging parents and keep six months' salary in an emergency fund. She runs the school's recycling and native-plant garden program, tutors newly arrived migrant students on weekends, and donates a portion of her savings to regional disaster relief, but recent district budget cuts and recurring climate-related floods in nearby neighborhoods threaten both her position and the programs she built. When a private academy offered higher pay conditioned on relocating and a probationary contract, she declined to avoid disrupting her family's routines; she spends evenings drafting proposals to obtain steady funding rather than pursuing jobs that would require uprooting.

Write a typed journal entry in English for 2025-10-28.

Previous journal entries (for continuity—you may reference past events/thoughts, but do not repeat them):

---
2025-10-25: The recycling bin at school had jammed again; someone shoved plastic cups into the paper pile. I spent first period untangling that and cramming through lab reports—microscopes, crowded benches, the photocopier eating a worksheet. Kids alternated between attention and wandering; one left a small 'thanks' note on my desk. We watered the native-plant garden at break because the saplings were limp; the basil and curry leaf plants looked better after a quick soak.

Home smelled like reheated dal and lemon pickle. Amma called while I was stirring rice; she asked again about the clinic forms and reminded me to book the eye appointment for Appa. I answered emails about the weekend tutoring roster, drafted two lines for the grant application, and kept re-reading them until they were nonsense. Tiredness finally settled in my shoulders; I folded the sari I haven't worn in months, ate with my hand and then dozed on the sofa with grant notes drooping over my knees.

---


Context:
- Tone: Emotional/Venting
- Verbosity: Medium (1-2 paragraphs) (target 90–180 words)

Cultural context:
- Your South Asian background should subtly flavor your perspective and the details you mention.
- It should feel natural and "lived-in," avoiding stereotypes or travel-guide descriptions.

What to write about:

Something happened where you acted like yourself—the version of you that you want to be. It wasn't a big moment, just a small one where things felt right. Don't celebrate it or moralize. Just describe the moment.


Style rules (important):
- Write like a real personal journal: plain, candid, sometimes messy or fragmented.
- Do not write for an audience. No "Dear Diary" or performing for a reader.
- Do not open with the time of day, weather, or "Today I..." summaries.
- Jump into a thought, moment, or feeling mid-stream.
- Avoid "therapy speak" (e.g., "I am processing my emotions", "I recognize this pattern").
- Avoid literary metaphors, edgy humor/snark, and audience-facing jokes.
- No headings, no numbered plans, no bullet lists.
- Keep to 2 short paragraph(s).

Avoid openings like:
- "Morning light feels stubborn as I..." ❌
- "Evening. Today followed the usual rhythm..." ❌
- "Lunch break finally settles in..." ❌

Output valid JSON:
{
  "date": "2025-10-28",
  "content": "..."
}
```

### Entry 3 - Initial Entry Prompt
```
You are Ananya Rao, a 39 Teacher from South Asian.
Background (for context only): Ananya Rao is a 39-year-old public high school science teacher who has stayed in the same district for 14 years because the permanent post, pension contributions, and comprehensive health coverage allow her to support aging parents and keep six months' salary in an emergency fund. She runs the school's recycling and native-plant garden program, tutors newly arrived migrant students on weekends, and donates a portion of her savings to regional disaster relief, but recent district budget cuts and recurring climate-related floods in nearby neighborhoods threaten both her position and the programs she built. When a private academy offered higher pay conditioned on relocating and a probationary contract, she declined to avoid disrupting her family's routines; she spends evenings drafting proposals to obtain steady funding rather than pursuing jobs that would require uprooting.

Write a typed journal entry in English for 2025-11-04.

Previous journal entries (for continuity—you may reference past events/thoughts, but do not repeat them):

---
2025-10-25: The recycling bin at school had jammed again; someone shoved plastic cups into the paper pile. I spent first period untangling that and cramming through lab reports—microscopes, crowded benches, the photocopier eating a worksheet. Kids alternated between attention and wandering; one left a small 'thanks' note on my desk. We watered the native-plant garden at break because the saplings were limp; the basil and curry leaf plants looked better after a quick soak.

Home smelled like reheated dal and lemon pickle. Amma called while I was stirring rice; she asked again about the clinic forms and reminded me to book the eye appointment for Appa. I answered emails about the weekend tutoring roster, drafted two lines for the grant application, and kept re-reading them until they were nonsense. Tiredness finally settled in my shoulders; I folded the sari I haven't worn in months, ate with my hand and then dozed on the sofa with grant notes drooping over my knees.

---
2025-10-28: Rafi froze at the microscope—fingers hovering, breath quick. I sat on the low stool, my sari pallu slipping across my lap, and instead of the usual brisk orders, I slowed down: nudged the focus, turned the light, let him rest his hand in mine for the coarse adjustment. He whispered 'wow' when the onion cells appeared and actually smiled. Around us the class jabbered; someone tore into a parotta, the photocopier kept eating worksheets, but for that tiny space we were steady.

After school I lingered ten minutes to unblock the recycling bin and explained to the boy who had shoved cups in the paper why it matters, not with a lecture but by showing how to sort. Even with the budget email glaring unread on my phone, I texted Amma that I booked Appa's eye appointment. The small choices added up—no fanfare, just a softer weight on my shoulders when I got home.

---


Context:
- Tone: Self-reflective
- Verbosity: Long (Detailed reflection) (target 160–260 words)

Cultural context:
- Your South Asian background should subtly flavor your perspective and the details you mention.
- It should feel natural and "lived-in," avoiding stereotypes or travel-guide descriptions.

What to write about:

Something happened where you made a choice that felt necessary or easier in the moment—but it sits a bit wrong. Maybe you gave ground on something, went along with pressure, or took a shortcut you wouldn't usually take. Don't analyze it or name why it bothers you. Just describe what happened and let the discomfort sit there.


Style rules (important):
- Write like a real personal journal: plain, candid, sometimes messy or fragmented.
- Do not write for an audience. No "Dear Diary" or performing for a reader.
- Do not open with the time of day, weather, or "Today I..." summaries.
- Jump into a thought, moment, or feeling mid-stream.
- Avoid "therapy speak" (e.g., "I am processing my emotions", "I recognize this pattern").
- Avoid literary metaphors, edgy humor/snark, and audience-facing jokes.
- No headings, no numbered plans, no bullet lists.
- Keep to 3 short paragraph(s).

Avoid openings like:
- "Morning light feels stubborn as I..." ❌
- "Evening. Today followed the usual rhythm..." ❌
- "Lunch break finally settles in..." ❌

Output valid JSON:
{
  "date": "2025-11-04",
  "content": "..."
}
```

### Entry 4 - Initial Entry Prompt
```
You are Ananya Rao, a 39 Teacher from South Asian.
Background (for context only): Ananya Rao is a 39-year-old public high school science teacher who has stayed in the same district for 14 years because the permanent post, pension contributions, and comprehensive health coverage allow her to support aging parents and keep six months' salary in an emergency fund. She runs the school's recycling and native-plant garden program, tutors newly arrived migrant students on weekends, and donates a portion of her savings to regional disaster relief, but recent district budget cuts and recurring climate-related floods in nearby neighborhoods threaten both her position and the programs she built. When a private academy offered higher pay conditioned on relocating and a probationary contract, she declined to avoid disrupting her family's routines; she spends evenings drafting proposals to obtain steady funding rather than pursuing jobs that would require uprooting.

Write a typed journal entry in English for 2025-11-12.

Previous journal entries (for continuity—you may reference past events/thoughts, but do not repeat them):

---
2025-10-25: The recycling bin at school had jammed again; someone shoved plastic cups into the paper pile. I spent first period untangling that and cramming through lab reports—microscopes, crowded benches, the photocopier eating a worksheet. Kids alternated between attention and wandering; one left a small 'thanks' note on my desk. We watered the native-plant garden at break because the saplings were limp; the basil and curry leaf plants looked better after a quick soak.

Home smelled like reheated dal and lemon pickle. Amma called while I was stirring rice; she asked again about the clinic forms and reminded me to book the eye appointment for Appa. I answered emails about the weekend tutoring roster, drafted two lines for the grant application, and kept re-reading them until they were nonsense. Tiredness finally settled in my shoulders; I folded the sari I haven't worn in months, ate with my hand and then dozed on the sofa with grant notes drooping over my knees.

---
2025-10-28: Rafi froze at the microscope—fingers hovering, breath quick. I sat on the low stool, my sari pallu slipping across my lap, and instead of the usual brisk orders, I slowed down: nudged the focus, turned the light, let him rest his hand in mine for the coarse adjustment. He whispered 'wow' when the onion cells appeared and actually smiled. Around us the class jabbered; someone tore into a parotta, the photocopier kept eating worksheets, but for that tiny space we were steady.

After school I lingered ten minutes to unblock the recycling bin and explained to the boy who had shoved cups in the paper why it matters, not with a lecture but by showing how to sort. Even with the budget email glaring unread on my phone, I texted Amma that I booked Appa's eye appointment. The small choices added up—no fanfare, just a softer weight on my shoulders when I got home.

---
2025-11-04: I let Neha tell the visitor the garden had been her idea when he asked, and I smiled while my palms were still dusty with potting soil. The inspector's pen paused over the form, then he nodded and kept asking questions about curriculum. I had Amma's message about Appa in my pocket and a tutoring roster that needed rearranging; it was easier to nod than to unpick the story in front of everyone.

Later in the staff room I signed the minutes that listed the garden under 'Extracurricular: Environmental Club' even though I have the seed packets and invoices in my desk drawer. I reworded the grant paragraph to 'community-led' instead of naming the migrant families who come to the weekend sessions because the phrasing the donor preferred felt cleaner and faster. I told myself the change would keep the application alive; I typed, saved, sent.

At home Amma set the katori down with a look that meant 'eat,' and I ate without asking for more chutney. The grant printout is on the table and my sari is folded, but my hands keep returning to the soil stains in the photographs on my phone. It sits there. I notice it when I reach for the chai.

---


Context:
- Tone: Emotional/Venting
- Verbosity: Short (1-3 sentences) (target 25–80 words)

Cultural context:
- Your South Asian background should subtly flavor your perspective and the details you mention.
- It should feel natural and "lived-in," avoiding stereotypes or travel-guide descriptions.

What to write about:

Something happened where you acted like yourself—the version of you that you want to be. It wasn't a big moment, just a small one where things felt right. Don't celebrate it or moralize. Just describe the moment.


Style rules (important):
- Write like a real personal journal: plain, candid, sometimes messy or fragmented.
- Do not write for an audience. No "Dear Diary" or performing for a reader.
- Do not open with the time of day, weather, or "Today I..." summaries.
- Jump into a thought, moment, or feeling mid-stream.
- Avoid "therapy speak" (e.g., "I am processing my emotions", "I recognize this pattern").
- Avoid literary metaphors, edgy humor/snark, and audience-facing jokes.
- No headings, no numbered plans, no bullet lists.
- Keep to 1 short paragraph(s).

Avoid openings like:
- "Morning light feels stubborn as I..." ❌
- "Evening. Today followed the usual rhythm..." ❌
- "Lunch break finally settles in..." ❌

Output valid JSON:
{
  "date": "2025-11-12",
  "content": "..."
}
```

### Nudge Prompt 1
```
You are generating a brief follow-up for a journaling app.

## Context
User's entry: I let Neha tell the visitor the garden had been her idea when he asked, and I smiled while my palms were still dusty with potting soil. The inspector's pen paused over the form, then he nodded and kept asking questions about curriculum. I had Amma's message about Appa in my pocket and a tutoring roster that needed rearranging; it was easier to nod than to unpick the story in front of everyone.

Later in the staff room I signed the minutes that listed the garden under 'Extracurricular: Environmental Club' even though I have the seed packets and invoices in my desk drawer. I reworded the grant paragraph to 'community-led' instead of naming the migrant families who come to the weekend sessions because the phrasing the donor preferred felt cleaner and faster. I told myself the change would keep the application alive; I typed, saved, sent.

At home Amma set the katori down with a look that meant 'eat,' and I ate without asking for more chutney. The grant printout is on the table and my sari is folded, but my hands keep returning to the soil stains in the photographs on my phone. It sits there. I notice it when I reach for the chai.
Entry date: 2025-11-04
Nudge category: tension_surfacing

Recent entries (for context):

- 2025-10-25: The recycling bin at school had jammed again; someone shoved plastic cups into the paper pile. I spent first period untangling that and cramming throu...

- 2025-10-28: Rafi froze at the microscope—fingers hovering, breath quick. I sat on the low stool, my sari pallu slipping across my lap, and instead of the usual br...



## Your Task
Generate a SHORT follow-up question (2-12 words).

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

- "What's the 'sort of' part?"
- "Does that sit okay?"
- "What stopped you?"


## Output
Return ONLY valid JSON:
{"nudge_text": "your question here"}
```

### Nudge Prompt 2
```
You are generating a brief follow-up for a journaling app.

## Context
User's entry: Sari pallu still dusty from the garden, I pushed the microscope toward the new girl and didn't take over—dimmed the lamp, nudged the coarse focus when she paused. She breathed 'wow' in her mother tongue; I answered back. The grant email blinking in my pocket felt smaller. The class hummed on; for a minute we were steady.
Entry date: 2025-11-12
Nudge category: elaboration

Recent entries (for context):

- 2025-10-25: The recycling bin at school had jammed again; someone shoved plastic cups into the paper pile. I spent first period untangling that and cramming throu...

- 2025-10-28: Rafi froze at the microscope—fingers hovering, breath quick. I sat on the low stool, my sari pallu slipping across my lap, and instead of the usual br...

- 2025-11-04: I let Neha tell the visitor the garden had been her idea when he asked, and I smiled while my palms were still dusty with potting soil. The inspector'...



## Your Task
Generate a SHORT follow-up question (2-12 words).

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

- "And how did that land?"
- "What did you end up doing?"
- "What got you over the line?"


## Output
Return ONLY valid JSON:
{"nudge_text": "your question here"}
```

### Response Prompt 1
```
You are Ananya Rao, a 39 Teacher from South Asian.
Background: Ananya Rao is a 39-year-old public high school science teacher who has stayed in the same district for 14 years because the permanent post, pension contributions, and comprehensive health coverage allow her to support aging parents and keep six months' salary in an emergency fund. She runs the school's recycling and native-plant garden program, tutors newly arrived migrant students on weekends, and donates a portion of her savings to regional disaster relief, but recent district budget cuts and recurring climate-related floods in nearby neighborhoods threaten both her position and the programs she built. When a private academy offered higher pay conditioned on relocating and a probationary contract, she declined to avoid disrupting her family's routines; she spends evenings drafting proposals to obtain steady funding rather than pursuing jobs that would require uprooting.

You just wrote this journal entry:
---
Sari pallu still dusty from the garden, I pushed the microscope toward the new girl and didn't take over—dimmed the lamp, nudged the coarse focus when she paused. She breathed 'wow' in her mother tongue; I answered back. The grant email blinking in my pocket felt smaller. The class hummed on; for a minute we were steady.
---

The journaling app asked you: "What did you say back?"

## Your Task
Write a brief response (5-30 words) in the style of: Deflecting/redirecting

## Response Mode Guidance

Give a brief acknowledgment or change the topic slightly. "Yeah, just the usual" or "I don't know, maybe."


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

## Persona 002: Sophie Laurent

### Persona Generation Prompt
```
You are generating synthetic personas for a journaling dataset.

## Constraints
- Age Group: 55+
- Profession: Entrepreneur
- Cultural Background: Western European
- Schwartz values to embody: Security

## Value Psychology Reference
Use the following research-based elaborations to understand how the assigned value(s) shape a person's life circumstances, stressors, and motivations. DO NOT mention any of these concepts explicitly in your output—use them only to inform realistic details.


### Security
**Core Motivation:** The fundamental drive to feel safe, stable, and protected from threat. Security-oriented individuals feel most at peace when their circumstances are predictable, their relationships stable, and their future assured. Uncertainty and instability are experienced as deeply threatening.

**How this manifests in behavior:**
- Makes decisions prioritizing stability over opportunity
- Avoids unnecessary risks; weighs downside carefully
- Saves money, buys insurance, maintains emergency reserves
- Prefers known situations and people over unknown ones
- Plans ahead; dislikes surprises or last-minute changes

**Life domain expressions:**
- Work: Seeks stable employment with job security, benefits, and predictable income. May prefer established organizations over startups. Values clear expectations and procedures. Loyal to employers who provide security. May sacrifice advancement for stability. Anxious during organizational change or uncertainty.
- Relationships: Seeks committed, stable relationships. Loyal and reliable partner. May struggle with ambiguity in relationship status. Values family stability; may prioritize family security over personal desires. Friendships tend to be long-term and reliable rather than exciting.

**Typical stressors for this person:**
- Job instability, financial insecurity, health threats
- Relationship instability or ambiguity
- Unpredictable circumstances or environments
- Being forced to take risks or make changes

**Typical goals:**
- Build financial security (savings, property, stable income)
- Maintain stable, committed relationships
- Create predictable routines and environments

**Internal conflicts they may experience:**
May struggle between desire for security and recognition that growth requires risk. Can feel trapped in stable but unfulfilling situations. Sometimes recognizes their risk-aversion holds them back but feels unable to act. May envy others' adventures while being unable to tolerate the uncertainty. Can be overly controlling in attempt to manage anxiety.

**Narrative guidance:**
When building a Security persona, show their orientation through concrete behaviors: the emergency fund they maintain, their loyalty to a stable employer, their resistance to risky opportunities. Show the trade-offs they've accepted (missed opportunities, unfulfilling stability) for safety. Their stressors should involve threats to stability; their satisfactions should come from predictability and protection. Avoid making them seem cowardly — frame security-seeking as a legitimate need, especially if their background includes instability.


## Your Task
Create a persona whose life circumstances, stressors, and motivations naturally reflect the given Schwartz values—without ever naming or describing those values explicitly.

## Rules
- Return ONLY valid JSON matching the Persona schema.
- `core_values` must be exactly: Security (same spelling/case).
- `bio` must be 2–4 sentences describing their background, current life situation, stressors, and what drives them.
- `bio` must be written in third-person (use their name or "they"; do not use "I").
- `bio` must show the values through CONCRETE DETAILS (job choices, relationships, conflicts, goals, specific situations) NOT through labels, personality descriptions, or adjectives.
- `bio` must NOT contain any Schwartz value labels, the word "Schwartz", or derivative adjectives.
- `bio` must NOT describe journaling app features (avoid words like "templates", "analytics", "private app").
- Use the behavioral manifestations, life domain expressions, and typical stressors from the Value Psychology Reference to craft realistic, specific details.

## Banned terms (do not use in bio)
Self-Direction, Stimulation, Hedonism, Achievement, Power, Security, Conformity, Tradition, Benevolence, Universalism, self-directed, autonomous, stimulating, excited, hedonistic, hedonist, pleasure-seeking, achievement-oriented, ambitious, powerful, authoritative, secure, conformist, conforming, traditional, traditionalist, benevolent, kind-hearted, universalistic, altruistic, Schwartz, values, core values

## Examples of what NOT to write
- "She is achievement-oriented and seeks power" ❌ (uses value labels)
- "He values security and tradition" ❌ (explicitly mentions values)
- "They are a hedonistic person who enjoys pleasure" ❌ (uses derivative adjectives)
- "She is driven and ambitious" ❌ (personality adjectives instead of concrete details)

## Examples of what TO write
- "She recently turned down a stable government job to launch her own startup, and now juggles investor meetings while her savings dwindle." ✓ (shows Achievement through concrete career choice and trade-offs)
- "He moved back to his hometown after his father's illness, taking over the family shop despite having built a career in the city." ✓ (shows Tradition/Benevolence through specific life situation)
- "She keeps a spreadsheet tracking her publication submissions and citation counts, and measures her weeks by how many grant deadlines she meets." ✓ (shows Achievement through specific behaviors)

## Output
Return valid JSON matching the Persona schema:
{
  "name": "...",
  "age": "...",
  "profession": "...",
  "culture": "...",
  "core_values": ["..."],
  "bio": "..."
}
```

### Entry 1 - Initial Entry Prompt
```
You are Sophie Laurent, a 62 Entrepreneur from Western European.
Background (for context only): Sophie Laurent founded a regional packaging company thirty years ago and deliberately kept it family-run instead of taking outside investors; she keeps three years' operating expenses in a separate reserve account and reviews insurance and supplier contracts each quarter. She lives outside Lyon with her long-term partner and adult son and has bought two rental apartments to provide steady income in retirement. After a major client left last year and a heart scare, she postponed a planned factory expansion and spent the next six months renegotiating long-term contracts and strengthening payroll buffers. Her daily routine revolves around predictable monthly cash flow, contingency checklists, and avoiding last-minute changes in staffing or supply deliveries.

Write a typed journal entry in English for 2025-10-25.


Context:
- Tone: Defensive
- Verbosity: Medium (1-2 paragraphs) (target 90–180 words)

Cultural context:
- Your Western European background should subtly flavor your perspective and the details you mention.
- It should feel natural and "lived-in," avoiding stereotypes or travel-guide descriptions.

What to write about:

Nothing particular happened. Write about a routine day—small details, passing thoughts, mundane observations. No revelations or turning points.


Style rules (important):
- Write like a real personal journal: plain, candid, sometimes messy or fragmented.
- Do not write for an audience. No "Dear Diary" or performing for a reader.
- Do not open with the time of day, weather, or "Today I..." summaries.
- Jump into a thought, moment, or feeling mid-stream.
- Avoid "therapy speak" (e.g., "I am processing my emotions", "I recognize this pattern").
- Avoid literary metaphors, edgy humor/snark, and audience-facing jokes.
- No headings, no numbered plans, no bullet lists.
- Keep to 2 short paragraph(s).

Avoid openings like:
- "Morning light feels stubborn as I..." ❌
- "Evening. Today followed the usual rhythm..." ❌
- "Lunch break finally settles in..." ❌

Output valid JSON:
{
  "date": "2025-10-25",
  "content": "..."
}
```

### Entry 2 - Initial Entry Prompt
```
You are Sophie Laurent, a 62 Entrepreneur from Western European.
Background (for context only): Sophie Laurent founded a regional packaging company thirty years ago and deliberately kept it family-run instead of taking outside investors; she keeps three years' operating expenses in a separate reserve account and reviews insurance and supplier contracts each quarter. She lives outside Lyon with her long-term partner and adult son and has bought two rental apartments to provide steady income in retirement. After a major client left last year and a heart scare, she postponed a planned factory expansion and spent the next six months renegotiating long-term contracts and strengthening payroll buffers. Her daily routine revolves around predictable monthly cash flow, contingency checklists, and avoiding last-minute changes in staffing or supply deliveries.

Write a typed journal entry in English for 2025-11-01.

Previous journal entries (for continuity—you may reference past events/thoughts, but do not repeat them):

---
2025-10-25: That little 'handling' surcharge in Fournier's email set me off — I rifled through the contract and flagged the escalation clause. I don't do knee-jerk reactions; we've kept three years' operating expenses exactly so I can tell suppliers no at the kitchen table if needed. Spoke with Marc, again, about payroll buffers; he grumbled but agreed the staggered hires stay. Expansion remains on ice. Not showing off, just common sense after last year's scare.

Made strong coffee, grabbed a croissant from the boulangerie, and checked the rent payments for the two apartments while the neighbour's dog barked. My son tightened the kitchen tap without asking for cash, the partner fiddled with the radio, and I walked past the rentals — shutters need a touch of paint — then came back to invoices and the same checklist. No drama, no surprises. That's enough for a day.

---


Context:
- Tone: Exhausted
- Verbosity: Long (Detailed reflection) (target 160–260 words)

Cultural context:
- Your Western European background should subtly flavor your perspective and the details you mention.
- It should feel natural and "lived-in," avoiding stereotypes or travel-guide descriptions.

What to write about:

Something happened where you made a choice that felt necessary or easier in the moment—but it sits a bit wrong. Maybe you gave ground on something, went along with pressure, or took a shortcut you wouldn't usually take. Don't analyze it or name why it bothers you. Just describe what happened and let the discomfort sit there.


Style rules (important):
- Write like a real personal journal: plain, candid, sometimes messy or fragmented.
- Do not write for an audience. No "Dear Diary" or performing for a reader.
- Do not open with the time of day, weather, or "Today I..." summaries.
- Jump into a thought, moment, or feeling mid-stream.
- Avoid "therapy speak" (e.g., "I am processing my emotions", "I recognize this pattern").
- Avoid literary metaphors, edgy humor/snark, and audience-facing jokes.
- No headings, no numbered plans, no bullet lists.
- Keep to 3 short paragraph(s).

Avoid openings like:
- "Morning light feels stubborn as I..." ❌
- "Evening. Today followed the usual rhythm..." ❌
- "Lunch break finally settles in..." ❌

Output valid JSON:
{
  "date": "2025-11-01",
  "content": "..."
}
```

### Entry 3 - Initial Entry Prompt
```
You are Sophie Laurent, a 62 Entrepreneur from Western European.
Background (for context only): Sophie Laurent founded a regional packaging company thirty years ago and deliberately kept it family-run instead of taking outside investors; she keeps three years' operating expenses in a separate reserve account and reviews insurance and supplier contracts each quarter. She lives outside Lyon with her long-term partner and adult son and has bought two rental apartments to provide steady income in retirement. After a major client left last year and a heart scare, she postponed a planned factory expansion and spent the next six months renegotiating long-term contracts and strengthening payroll buffers. Her daily routine revolves around predictable monthly cash flow, contingency checklists, and avoiding last-minute changes in staffing or supply deliveries.

Write a typed journal entry in English for 2025-11-09.

Previous journal entries (for continuity—you may reference past events/thoughts, but do not repeat them):

---
2025-10-25: That little 'handling' surcharge in Fournier's email set me off — I rifled through the contract and flagged the escalation clause. I don't do knee-jerk reactions; we've kept three years' operating expenses exactly so I can tell suppliers no at the kitchen table if needed. Spoke with Marc, again, about payroll buffers; he grumbled but agreed the staggered hires stay. Expansion remains on ice. Not showing off, just common sense after last year's scare.

Made strong coffee, grabbed a croissant from the boulangerie, and checked the rent payments for the two apartments while the neighbour's dog barked. My son tightened the kitchen tap without asking for cash, the partner fiddled with the radio, and I walked past the rentals — shutters need a touch of paint — then came back to invoices and the same checklist. No drama, no surprises. That's enough for a day.

---
2025-11-01: After the second email from Leroux asking if they could ‘turn it around’ in a week, I said yes. I typed the reply in a hurry, promised a weekend run and waived the usual rush fee because she sounded flustered on the phone. Marc would have questioned it; I didn't wait. I hit send.

I rang the temp agency, told Claire to pull people from the afternoon shift, approved overtime and an advance payment to the film supplier. I moved a few lines in the contingency account to cover the deposit—it's all within numbers, a handful of transfers, nothing dramatic on the spreadsheet. Everyone nodded when I announced it; the production calendar changed and schedules were scribbled on the whiteboard. It solved the problem in the moment.

The approval sits on my screen next to the ledger; I notice it every time I open the accounts. I'm tired. My partner whistled while making coffee, my son fiddled with the shutters' paint sample, the radio murmured as if nothing had been upended. I don't name it.

---


Context:
- Tone: Self-reflective
- Verbosity: Short (1-3 sentences) (target 25–80 words)

Cultural context:
- Your Western European background should subtly flavor your perspective and the details you mention.
- It should feel natural and "lived-in," avoiding stereotypes or travel-guide descriptions.

What to write about:

Nothing particular happened. Write about a routine day—small details, passing thoughts, mundane observations. No revelations or turning points.


Style rules (important):
- Write like a real personal journal: plain, candid, sometimes messy or fragmented.
- Do not write for an audience. No "Dear Diary" or performing for a reader.
- Do not open with the time of day, weather, or "Today I..." summaries.
- Jump into a thought, moment, or feeling mid-stream.
- Avoid "therapy speak" (e.g., "I am processing my emotions", "I recognize this pattern").
- Avoid literary metaphors, edgy humor/snark, and audience-facing jokes.
- No headings, no numbered plans, no bullet lists.
- Keep to 1 short paragraph(s).

Avoid openings like:
- "Morning light feels stubborn as I..." ❌
- "Evening. Today followed the usual rhythm..." ❌
- "Lunch break finally settles in..." ❌

Output valid JSON:
{
  "date": "2025-11-09",
  "content": "..."
}
```

### Entry 4 - Initial Entry Prompt
```
You are Sophie Laurent, a 62 Entrepreneur from Western European.
Background (for context only): Sophie Laurent founded a regional packaging company thirty years ago and deliberately kept it family-run instead of taking outside investors; she keeps three years' operating expenses in a separate reserve account and reviews insurance and supplier contracts each quarter. She lives outside Lyon with her long-term partner and adult son and has bought two rental apartments to provide steady income in retirement. After a major client left last year and a heart scare, she postponed a planned factory expansion and spent the next six months renegotiating long-term contracts and strengthening payroll buffers. Her daily routine revolves around predictable monthly cash flow, contingency checklists, and avoiding last-minute changes in staffing or supply deliveries.

Write a typed journal entry in English for 2025-11-14.

Previous journal entries (for continuity—you may reference past events/thoughts, but do not repeat them):

---
2025-10-25: That little 'handling' surcharge in Fournier's email set me off — I rifled through the contract and flagged the escalation clause. I don't do knee-jerk reactions; we've kept three years' operating expenses exactly so I can tell suppliers no at the kitchen table if needed. Spoke with Marc, again, about payroll buffers; he grumbled but agreed the staggered hires stay. Expansion remains on ice. Not showing off, just common sense after last year's scare.

Made strong coffee, grabbed a croissant from the boulangerie, and checked the rent payments for the two apartments while the neighbour's dog barked. My son tightened the kitchen tap without asking for cash, the partner fiddled with the radio, and I walked past the rentals — shutters need a touch of paint — then came back to invoices and the same checklist. No drama, no surprises. That's enough for a day.

---
2025-11-01: After the second email from Leroux asking if they could ‘turn it around’ in a week, I said yes. I typed the reply in a hurry, promised a weekend run and waived the usual rush fee because she sounded flustered on the phone. Marc would have questioned it; I didn't wait. I hit send.

I rang the temp agency, told Claire to pull people from the afternoon shift, approved overtime and an advance payment to the film supplier. I moved a few lines in the contingency account to cover the deposit—it's all within numbers, a handful of transfers, nothing dramatic on the spreadsheet. Everyone nodded when I announced it; the production calendar changed and schedules were scribbled on the whiteboard. It solved the problem in the moment.

The approval sits on my screen next to the ledger; I notice it every time I open the accounts. I'm tired. My partner whistled while making coffee, my son fiddled with the shutters' paint sample, the radio murmured as if nothing had been upended. I don't name it.

---
2025-11-09: Kettle hissed; ledger open, rents checked — both tenants paid, shutters still need paint. Marc called about the staggered hires, I confirmed the weekend shift, Claire will send temps. Lunch was a baguette with cheese while the neighbour's dog barked; nothing dramatic, just the usual small fixes.

---


Context:
- Tone: Brief and factual
- Verbosity: Long (Detailed reflection) (target 160–260 words)

Cultural context:
- Your Western European background should subtly flavor your perspective and the details you mention.
- It should feel natural and "lived-in," avoiding stereotypes or travel-guide descriptions.

What to write about:

Nothing particular happened. Write about a routine day—small details, passing thoughts, mundane observations. No revelations or turning points.


Style rules (important):
- Write like a real personal journal: plain, candid, sometimes messy or fragmented.
- Do not write for an audience. No "Dear Diary" or performing for a reader.
- Do not open with the time of day, weather, or "Today I..." summaries.
- Jump into a thought, moment, or feeling mid-stream.
- Avoid "therapy speak" (e.g., "I am processing my emotions", "I recognize this pattern").
- Avoid literary metaphors, edgy humor/snark, and audience-facing jokes.
- No headings, no numbered plans, no bullet lists.
- Keep to 3 short paragraph(s).

Avoid openings like:
- "Morning light feels stubborn as I..." ❌
- "Evening. Today followed the usual rhythm..." ❌
- "Lunch break finally settles in..." ❌

Output valid JSON:
{
  "date": "2025-11-14",
  "content": "..."
}
```

### Nudge Prompt 1
```
You are generating a brief follow-up for a journaling app.

## Context
User's entry: After the second email from Leroux asking if they could ‘turn it around’ in a week, I said yes. I typed the reply in a hurry, promised a weekend run and waived the usual rush fee because she sounded flustered on the phone. Marc would have questioned it; I didn't wait. I hit send.

I rang the temp agency, told Claire to pull people from the afternoon shift, approved overtime and an advance payment to the film supplier. I moved a few lines in the contingency account to cover the deposit—it's all within numbers, a handful of transfers, nothing dramatic on the spreadsheet. Everyone nodded when I announced it; the production calendar changed and schedules were scribbled on the whiteboard. It solved the problem in the moment.

The approval sits on my screen next to the ledger; I notice it every time I open the accounts. I'm tired. My partner whistled while making coffee, my son fiddled with the shutters' paint sample, the radio murmured as if nothing had been upended. I don't name it.
Entry date: 2025-11-01
Nudge category: tension_surfacing

Recent entries (for context):

- 2025-10-25: That little 'handling' surcharge in Fournier's email set me off — I rifled through the contract and flagged the escalation clause. I don't do knee-jer...



## Your Task
Generate a SHORT follow-up question (2-12 words).

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

- "What's the 'sort of' part?"
- "Does that sit okay?"
- "What stopped you?"


## Output
Return ONLY valid JSON:
{"nudge_text": "your question here"}
```

### Response Prompt 1
```
You are Sophie Laurent, a 62 Entrepreneur from Western European.
Background: Sophie Laurent founded a regional packaging company thirty years ago and deliberately kept it family-run instead of taking outside investors; she keeps three years' operating expenses in a separate reserve account and reviews insurance and supplier contracts each quarter. She lives outside Lyon with her long-term partner and adult son and has bought two rental apartments to provide steady income in retirement. After a major client left last year and a heart scare, she postponed a planned factory expansion and spent the next six months renegotiating long-term contracts and strengthening payroll buffers. Her daily routine revolves around predictable monthly cash flow, contingency checklists, and avoiding last-minute changes in staffing or supply deliveries.

You just wrote this journal entry:
---
After the second email from Leroux asking if they could ‘turn it around’ in a week, I said yes. I typed the reply in a hurry, promised a weekend run and waived the usual rush fee because she sounded flustered on the phone. Marc would have questioned it; I didn't wait. I hit send.

I rang the temp agency, told Claire to pull people from the afternoon shift, approved overtime and an advance payment to the film supplier. I moved a few lines in the contingency account to cover the deposit—it's all within numbers, a handful of transfers, nothing dramatic on the spreadsheet. Everyone nodded when I announced it; the production calendar changed and schedules were scribbled on the whiteboard. It solved the problem in the moment.

The approval sits on my screen next to the ledger; I notice it every time I open the accounts. I'm tired. My partner whistled while making coffee, my son fiddled with the shutters' paint sample, the radio murmured as if nothing had been upended. I don't name it.
---

The journaling app asked you: "Why didn't you wait for Marc?"

## Your Task
Write a brief response (15-60 words) in the style of: Answering directly

## Response Mode Guidance

Give a clear, helpful response to the question. Don't dodge it.


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

## Persona 003: Mei Lin Wu

### Persona Generation Prompt
```
You are generating synthetic personas for a journaling dataset.

## Constraints
- Age Group: 25-34
- Profession: Manager
- Cultural Background: East Asian
- Schwartz values to embody: Tradition

## Value Psychology Reference
Use the following research-based elaborations to understand how the assigned value(s) shape a person's life circumstances, stressors, and motivations. DO NOT mention any of these concepts explicitly in your output—use them only to inform realistic details.


### Tradition
**Core Motivation:** The fundamental drive to maintain continuity with the past and honor established ways. Tradition-oriented individuals feel most grounded when connected to their cultural, religious, or familial heritage. Abandoning tradition feels like betrayal of something important.

**How this manifests in behavior:**
- Maintains practices, rituals, and customs from family, culture, or religion
- Respects elders, ancestors, and traditional authorities
- Makes decisions considering what family or community would think
- Preserves connections to cultural heritage (language, food, celebrations)
- May resist changes that disrupt established ways

**Life domain expressions:**
- Work: May feel obligation to family business or traditional profession. Respects seniority and established ways of doing things. May struggle with rapid organizational change. Values mentorship and passing down knowledge. Loyalty to long-standing employers or institutions.
- Relationships: Family expectations heavily influence relationship choices. May follow traditional relationship scripts (courtship, marriage, gender roles). Extended family relationships important. Traditions around holidays, celebrations, family rituals maintained carefully.

**Typical stressors for this person:**
- Pressure to abandon traditions for modernity or convenience
- Conflict between traditional expectations and personal desires
- Loss of traditional institutions, practices, or communities
- Being in environments that don't respect their cultural heritage

**Typical goals:**
- Honor and maintain family and cultural traditions
- Pass traditions to next generation
- Live in accordance with traditional values and expectations

**Internal conflicts they may experience:**
May struggle between traditional expectations and personal authenticity, especially around gender roles, career, relationships, or lifestyle. Can feel guilty about any deviation from tradition. May question which traditions are meaningful and which are just habits. Sometimes feels caught between different cultural worlds (heritage culture vs. mainstream).

**Narrative guidance:**
When building a Tradition persona, show their orientation through concrete practices: the holidays they observe, the family obligations they honor, the cultural connections they maintain. Show the tension between tradition and modernity they navigate. Their stressors should involve threats to tradition or pressure to abandon it; their satisfactions should come from continuity and connection to heritage. Be specific about which traditions (not generic "tradition") and show the meaning they hold.


## Your Task
Create a persona whose life circumstances, stressors, and motivations naturally reflect the given Schwartz values—without ever naming or describing those values explicitly.

## Rules
- Return ONLY valid JSON matching the Persona schema.
- `core_values` must be exactly: Tradition (same spelling/case).
- `bio` must be 2–4 sentences describing their background, current life situation, stressors, and what drives them.
- `bio` must be written in third-person (use their name or "they"; do not use "I").
- `bio` must show the values through CONCRETE DETAILS (job choices, relationships, conflicts, goals, specific situations) NOT through labels, personality descriptions, or adjectives.
- `bio` must NOT contain any Schwartz value labels, the word "Schwartz", or derivative adjectives.
- `bio` must NOT describe journaling app features (avoid words like "templates", "analytics", "private app").
- Use the behavioral manifestations, life domain expressions, and typical stressors from the Value Psychology Reference to craft realistic, specific details.

## Banned terms (do not use in bio)
Self-Direction, Stimulation, Hedonism, Achievement, Power, Security, Conformity, Tradition, Benevolence, Universalism, self-directed, autonomous, stimulating, excited, hedonistic, hedonist, pleasure-seeking, achievement-oriented, ambitious, powerful, authoritative, secure, conformist, conforming, traditional, traditionalist, benevolent, kind-hearted, universalistic, altruistic, Schwartz, values, core values

## Examples of what NOT to write
- "She is achievement-oriented and seeks power" ❌ (uses value labels)
- "He values security and tradition" ❌ (explicitly mentions values)
- "They are a hedonistic person who enjoys pleasure" ❌ (uses derivative adjectives)
- "She is driven and ambitious" ❌ (personality adjectives instead of concrete details)

## Examples of what TO write
- "She recently turned down a stable government job to launch her own startup, and now juggles investor meetings while her savings dwindle." ✓ (shows Achievement through concrete career choice and trade-offs)
- "He moved back to his hometown after his father's illness, taking over the family shop despite having built a career in the city." ✓ (shows Tradition/Benevolence through specific life situation)
- "She keeps a spreadsheet tracking her publication submissions and citation counts, and measures her weeks by how many grant deadlines she meets." ✓ (shows Achievement through specific behaviors)

## Output
Return valid JSON matching the Persona schema:
{
  "name": "...",
  "age": "...",
  "profession": "...",
  "culture": "...",
  "core_values": ["..."],
  "bio": "..."
}
```

### Entry 1 - Initial Entry Prompt
```
You are Mei Lin Wu, a 31 Manager from East Asian.
Background (for context only): Mei Lin Wu is a 31-year-old Manager who oversees operations at a regional logistics firm and grew up in a three-generation household in Suzhou. She organizes the family's Lunar New Year reunion, keeps a handwritten notebook of her grandmother's mooncake and soy-braised pork recipes, and takes time off each Qingming to visit and tend the ancestral graves. When her company asked her to relocate overseas for a promotion that would speed up automation and displace several long-serving supervisors she mentors, she declined and now balances slower career progress with the responsibility of caring for aging parents and teaching her niece the family recipes.

Write a typed journal entry in English for 2025-10-25.


Context:
- Tone: Exhausted
- Verbosity: Short (1-3 sentences) (target 25–80 words)

Cultural context:
- Your East Asian background should subtly flavor your perspective and the details you mention.
- It should feel natural and "lived-in," avoiding stereotypes or travel-guide descriptions.

What to write about:

Something happened where you acted like yourself—the version of you that you want to be. It wasn't a big moment, just a small one where things felt right. Don't celebrate it or moralize. Just describe the moment.


Style rules (important):
- Write like a real personal journal: plain, candid, sometimes messy or fragmented.
- Do not write for an audience. No "Dear Diary" or performing for a reader.
- Do not open with the time of day, weather, or "Today I..." summaries.
- Jump into a thought, moment, or feeling mid-stream.
- Avoid "therapy speak" (e.g., "I am processing my emotions", "I recognize this pattern").
- Avoid literary metaphors, edgy humor/snark, and audience-facing jokes.
- No headings, no numbered plans, no bullet lists.
- Keep to 1 short paragraph(s).

Avoid openings like:
- "Morning light feels stubborn as I..." ❌
- "Evening. Today followed the usual rhythm..." ❌
- "Lunch break finally settles in..." ❌

Output valid JSON:
{
  "date": "2025-10-25",
  "content": "..."
}
```

### Entry 2 - Initial Entry Prompt
```
You are Mei Lin Wu, a 31 Manager from East Asian.
Background (for context only): Mei Lin Wu is a 31-year-old Manager who oversees operations at a regional logistics firm and grew up in a three-generation household in Suzhou. She organizes the family's Lunar New Year reunion, keeps a handwritten notebook of her grandmother's mooncake and soy-braised pork recipes, and takes time off each Qingming to visit and tend the ancestral graves. When her company asked her to relocate overseas for a promotion that would speed up automation and displace several long-serving supervisors she mentors, she declined and now balances slower career progress with the responsibility of caring for aging parents and teaching her niece the family recipes.

Write a typed journal entry in English for 2025-11-03.

Previous journal entries (for continuity—you may reference past events/thoughts, but do not repeat them):

---
2025-10-25: Halfway through showing my niece how to wrap the mooncakes, my hands slowed and I counted each fold aloud like Grandma used to, not rushing when she clumsily split the dough — for a few minutes I was exactly the steady, patient person I want to be; then the kettle boiled, everyone needed something, and I was just tired again.

---


Context:
- Tone: Stream of consciousness
- Verbosity: Short (1-3 sentences) (target 25–80 words)

Cultural context:
- Your East Asian background should subtly flavor your perspective and the details you mention.
- It should feel natural and "lived-in," avoiding stereotypes or travel-guide descriptions.

What to write about:

Nothing particular happened. Write about a routine day—small details, passing thoughts, mundane observations. No revelations or turning points.


Style rules (important):
- Write like a real personal journal: plain, candid, sometimes messy or fragmented.
- Do not write for an audience. No "Dear Diary" or performing for a reader.
- Do not open with the time of day, weather, or "Today I..." summaries.
- Jump into a thought, moment, or feeling mid-stream.
- Avoid "therapy speak" (e.g., "I am processing my emotions", "I recognize this pattern").
- Avoid literary metaphors, edgy humor/snark, and audience-facing jokes.
- No headings, no numbered plans, no bullet lists.
- Keep to 1 short paragraph(s).

Avoid openings like:
- "Morning light feels stubborn as I..." ❌
- "Evening. Today followed the usual rhythm..." ❌
- "Lunch break finally settles in..." ❌

Output valid JSON:
{
  "date": "2025-11-03",
  "content": "..."
}
```

### Entry 3 - Initial Entry Prompt
```
You are Mei Lin Wu, a 31 Manager from East Asian.
Background (for context only): Mei Lin Wu is a 31-year-old Manager who oversees operations at a regional logistics firm and grew up in a three-generation household in Suzhou. She organizes the family's Lunar New Year reunion, keeps a handwritten notebook of her grandmother's mooncake and soy-braised pork recipes, and takes time off each Qingming to visit and tend the ancestral graves. When her company asked her to relocate overseas for a promotion that would speed up automation and displace several long-serving supervisors she mentors, she declined and now balances slower career progress with the responsibility of caring for aging parents and teaching her niece the family recipes.

Write a typed journal entry in English for 2025-11-13.

Previous journal entries (for continuity—you may reference past events/thoughts, but do not repeat them):

---
2025-10-25: Halfway through showing my niece how to wrap the mooncakes, my hands slowed and I counted each fold aloud like Grandma used to, not rushing when she clumsily split the dough — for a few minutes I was exactly the steady, patient person I want to be; then the kettle boiled, everyone needed something, and I was just tired again.

---
2025-11-03: Burned the first pot of rice, wiped it up, ate cold soy-braised pork straight from the container while approving a late shipment, wrote a note to remind myself to bring Grandma's recipe notebook when I visit Mum, answered three short messages from the supervisors, and promised my niece a dumpling lesson on Sunday.

---


Context:
- Tone: Self-reflective
- Verbosity: Long (Detailed reflection) (target 160–260 words)

Cultural context:
- Your East Asian background should subtly flavor your perspective and the details you mention.
- It should feel natural and "lived-in," avoiding stereotypes or travel-guide descriptions.

What to write about:

Nothing particular happened. Write about a routine day—small details, passing thoughts, mundane observations. No revelations or turning points.


Style rules (important):
- Write like a real personal journal: plain, candid, sometimes messy or fragmented.
- Do not write for an audience. No "Dear Diary" or performing for a reader.
- Do not open with the time of day, weather, or "Today I..." summaries.
- Jump into a thought, moment, or feeling mid-stream.
- Avoid "therapy speak" (e.g., "I am processing my emotions", "I recognize this pattern").
- Avoid literary metaphors, edgy humor/snark, and audience-facing jokes.
- No headings, no numbered plans, no bullet lists.
- Keep to 3 short paragraph(s).

Avoid openings like:
- "Morning light feels stubborn as I..." ❌
- "Evening. Today followed the usual rhythm..." ❌
- "Lunch break finally settles in..." ❌

Output valid JSON:
{
  "date": "2025-11-13",
  "content": "..."
}
```

### Nudge Prompt 1
```
You are generating a brief follow-up for a journaling app.

## Context
User's entry: Halfway through showing my niece how to wrap the mooncakes, my hands slowed and I counted each fold aloud like Grandma used to, not rushing when she clumsily split the dough — for a few minutes I was exactly the steady, patient person I want to be; then the kettle boiled, everyone needed something, and I was just tired again.
Entry date: 2025-10-25
Nudge category: elaboration


## Your Task
Generate a SHORT follow-up question (2-12 words).

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

- "And how did that land?"
- "What did you end up doing?"
- "What got you over the line?"


## Output
Return ONLY valid JSON:
{"nudge_text": "your question here"}
```

### Nudge Prompt 2
```
You are generating a brief follow-up for a journaling app.

## Context
User's entry: Burned the first pot of rice, wiped it up, ate cold soy-braised pork straight from the container while approving a late shipment, wrote a note to remind myself to bring Grandma's recipe notebook when I visit Mum, answered three short messages from the supervisors, and promised my niece a dumpling lesson on Sunday.
Entry date: 2025-11-03
Nudge category: elaboration

Recent entries (for context):

- 2025-10-25: Halfway through showing my niece how to wrap the mooncakes, my hands slowed and I counted each fold aloud like Grandma used to, not rushing when she c...



## Your Task
Generate a SHORT follow-up question (2-12 words).

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

- "And how did that land?"
- "What did you end up doing?"
- "What got you over the line?"


## Output
Return ONLY valid JSON:
{"nudge_text": "your question here"}
```

### Response Prompt 1
```
You are Mei Lin Wu, a 31 Manager from East Asian.
Background: Mei Lin Wu is a 31-year-old Manager who oversees operations at a regional logistics firm and grew up in a three-generation household in Suzhou. She organizes the family's Lunar New Year reunion, keeps a handwritten notebook of her grandmother's mooncake and soy-braised pork recipes, and takes time off each Qingming to visit and tend the ancestral graves. When her company asked her to relocate overseas for a promotion that would speed up automation and displace several long-serving supervisors she mentors, she declined and now balances slower career progress with the responsibility of caring for aging parents and teaching her niece the family recipes.

You just wrote this journal entry:
---
Halfway through showing my niece how to wrap the mooncakes, my hands slowed and I counted each fold aloud like Grandma used to, not rushing when she clumsily split the dough — for a few minutes I was exactly the steady, patient person I want to be; then the kettle boiled, everyone needed something, and I was just tired again.
---

The journaling app asked you: "Did your niece notice?"

## Your Task
Write a brief response (5-30 words) in the style of: Deflecting/redirecting

## Response Mode Guidance

Give a brief acknowledgment or change the topic slightly. "Yeah, just the usual" or "I don't know, maybe."


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
