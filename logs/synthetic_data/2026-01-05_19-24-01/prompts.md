# Prompts Log

## Persona 001: Maria Alvarez

### Persona Generation Prompt
```

You are generating synthetic personas for a journaling dataset.

## Constraints
- Age Group: 45-54
- Profession: Parent (Stay-at-home)
- Cultural Background: North American
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

You are Maria Alvarez, a 48 Parent (Stay-at-home) from North American.
Background (for context only): Maria Alvarez keeps her mother's tamale recipe in a stained notebook and spends two full days each December making tamales for the extended family before Nochebuena; she also teaches the grandchildren a few Spanish lullabies and the old prayers her abuela used to sing. She runs the household calendar—coordinating baptisms, the neighborhood posada, and the annual cemetery visit on Dia de los Muertos—while driving her elderly father to Mass every Sunday. Lately she's worried because her eldest moved out of state, younger relatives prefer quick takeout and video calls to the long preparations she learned growing up, and the community center that hosted family gatherings closed last year.

Write a typed journal entry in English for 2025-10-25.


Context:
- Tone: Stream of consciousness
- Verbosity: Short (1-3 sentences) (target 25–80 words)

Cultural context:
- Your North American background should subtly flavor your perspective and the details you mention.
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

You are Maria Alvarez, a 48 Parent (Stay-at-home) from North American.
Background (for context only): Maria Alvarez keeps her mother's tamale recipe in a stained notebook and spends two full days each December making tamales for the extended family before Nochebuena; she also teaches the grandchildren a few Spanish lullabies and the old prayers her abuela used to sing. She runs the household calendar—coordinating baptisms, the neighborhood posada, and the annual cemetery visit on Dia de los Muertos—while driving her elderly father to Mass every Sunday. Lately she's worried because her eldest moved out of state, younger relatives prefer quick takeout and video calls to the long preparations she learned growing up, and the community center that hosted family gatherings closed last year.

Write a typed journal entry in English for 2025-10-28.

Previous journal entries (for continuity—you may reference past events/thoughts, but do not repeat them):

---
2025-10-25: I let everyone insist we order from the taquería instead of making tamales for Nochebuena—called for fifty, stacked the boxes beside mama's stained recipe notebook, hummed 'Arrorró' to fill the silence, and my hands trembled when I tied the last napkin.

---


Context:
- Tone: Defensive
- Verbosity: Short (1-3 sentences) (target 25–80 words)

Cultural context:
- Your North American background should subtly flavor your perspective and the details you mention.
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
  "date": "2025-10-28",
  "content": "..."
}
```

### Entry 3 - Initial Entry Prompt
```

You are Maria Alvarez, a 48 Parent (Stay-at-home) from North American.
Background (for context only): Maria Alvarez keeps her mother's tamale recipe in a stained notebook and spends two full days each December making tamales for the extended family before Nochebuena; she also teaches the grandchildren a few Spanish lullabies and the old prayers her abuela used to sing. She runs the household calendar—coordinating baptisms, the neighborhood posada, and the annual cemetery visit on Dia de los Muertos—while driving her elderly father to Mass every Sunday. Lately she's worried because her eldest moved out of state, younger relatives prefer quick takeout and video calls to the long preparations she learned growing up, and the community center that hosted family gatherings closed last year.

Write a typed journal entry in English for 2025-11-04.

Previous journal entries (for continuity—you may reference past events/thoughts, but do not repeat them):

---
2025-10-25: I let everyone insist we order from the taquería instead of making tamales for Nochebuena—called for fifty, stacked the boxes beside mama's stained recipe notebook, hummed 'Arrorró' to fill the silence, and my hands trembled when I tied the last napkin.

---
2025-10-28: Folded the last load of laundry, wiped coffee from Dad's favorite mug, and answered Ana's text with a thumbs-up; hummed 'Arrorró' while I set mama's flour-speckled notebook back on the shelf. I'm not abandoning anything by calling the taquería — it's still my kitchen, my rules, and I'll teach the prayers when the kids visit.

---


Context:
- Tone: Exhausted
- Verbosity: Long (Detailed reflection) (target 160–260 words)

Cultural context:
- Your North American background should subtly flavor your perspective and the details you mention.
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
  "date": "2025-11-04",
  "content": "..."
}
```

### Entry 4 - Initial Entry Prompt
```

You are Maria Alvarez, a 48 Parent (Stay-at-home) from North American.
Background (for context only): Maria Alvarez keeps her mother's tamale recipe in a stained notebook and spends two full days each December making tamales for the extended family before Nochebuena; she also teaches the grandchildren a few Spanish lullabies and the old prayers her abuela used to sing. She runs the household calendar—coordinating baptisms, the neighborhood posada, and the annual cemetery visit on Dia de los Muertos—while driving her elderly father to Mass every Sunday. Lately she's worried because her eldest moved out of state, younger relatives prefer quick takeout and video calls to the long preparations she learned growing up, and the community center that hosted family gatherings closed last year.

Write a typed journal entry in English for 2025-11-12.

Previous journal entries (for continuity—you may reference past events/thoughts, but do not repeat them):

---
2025-10-25: I let everyone insist we order from the taquería instead of making tamales for Nochebuena—called for fifty, stacked the boxes beside mama's stained recipe notebook, hummed 'Arrorró' to fill the silence, and my hands trembled when I tied the last napkin.

---
2025-10-28: Folded the last load of laundry, wiped coffee from Dad's favorite mug, and answered Ana's text with a thumbs-up; hummed 'Arrorró' while I set mama's flour-speckled notebook back on the shelf. I'm not abandoning anything by calling the taquería — it's still my kitchen, my rules, and I'll teach the prayers when the kids visit.

---
2025-11-04: Kettle hisses; I stir my coffee and glance at the taquería boxes still on the counter beside mama's stained recipe notebook. Folded two loads of laundry, found a sock with a hole and set it in the sewing basket I never open, wiped the coffee ring off Dad's favorite mug and hummed 'Arrorró' without thinking.

Ran to the store for limes and came back with everything except cilantro; put groceries away slowly and had to move three things twice. Ana texted a photo of her new couch and I sent a thumbs-up because it's faster than a call. Dad asked if I had the crossword; I found the paper under yesterday's recipes and he laughed.

Sat at the table to heat leftovers; legs cramped and I left the radio on because quiet made me notice too many things. Flipped through mama's notebook even though I promised myself I'd leave it alone—spare line items, the recipe I always forget the name of, grease smudges at the edge. Hummed 'Arrorró' again, turned out the kitchen light, the house smelled like oregano.

---


Context:
- Tone: Stream of consciousness
- Verbosity: Short (1-3 sentences) (target 25–80 words)

Cultural context:
- Your North American background should subtly flavor your perspective and the details you mention.
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

### Entry 5 - Initial Entry Prompt
```

You are Maria Alvarez, a 48 Parent (Stay-at-home) from North American.
Background (for context only): Maria Alvarez keeps her mother's tamale recipe in a stained notebook and spends two full days each December making tamales for the extended family before Nochebuena; she also teaches the grandchildren a few Spanish lullabies and the old prayers her abuela used to sing. She runs the household calendar—coordinating baptisms, the neighborhood posada, and the annual cemetery visit on Dia de los Muertos—while driving her elderly father to Mass every Sunday. Lately she's worried because her eldest moved out of state, younger relatives prefer quick takeout and video calls to the long preparations she learned growing up, and the community center that hosted family gatherings closed last year.

Write a typed journal entry in English for 2025-11-17.

Previous journal entries (for continuity—you may reference past events/thoughts, but do not repeat them):

---
2025-10-25: I let everyone insist we order from the taquería instead of making tamales for Nochebuena—called for fifty, stacked the boxes beside mama's stained recipe notebook, hummed 'Arrorró' to fill the silence, and my hands trembled when I tied the last napkin.

---
2025-10-28: Folded the last load of laundry, wiped coffee from Dad's favorite mug, and answered Ana's text with a thumbs-up; hummed 'Arrorró' while I set mama's flour-speckled notebook back on the shelf. I'm not abandoning anything by calling the taquería — it's still my kitchen, my rules, and I'll teach the prayers when the kids visit.

---
2025-11-04: Kettle hisses; I stir my coffee and glance at the taquería boxes still on the counter beside mama's stained recipe notebook. Folded two loads of laundry, found a sock with a hole and set it in the sewing basket I never open, wiped the coffee ring off Dad's favorite mug and hummed 'Arrorró' without thinking.

Ran to the store for limes and came back with everything except cilantro; put groceries away slowly and had to move three things twice. Ana texted a photo of her new couch and I sent a thumbs-up because it's faster than a call. Dad asked if I had the crossword; I found the paper under yesterday's recipes and he laughed.

Sat at the table to heat leftovers; legs cramped and I left the radio on because quiet made me notice too many things. Flipped through mama's notebook even though I promised myself I'd leave it alone—spare line items, the recipe I always forget the name of, grease smudges at the edge. Hummed 'Arrorró' again, turned out the kitchen light, the house smelled like oregano.

---
2025-11-12: Phone buzzed—Ana video-called to show the baby's first steps; I watched part of it with a diaper in my hand and sent a thumbs-up instead of calling back, put mama's stained notebook under the breadbox and hummed 'Arrorró' until the kitchen noises felt louder than the picture.

---


Context:
- Tone: Defensive
- Verbosity: Long (Detailed reflection) (target 160–260 words)

Cultural context:
- Your North American background should subtly flavor your perspective and the details you mention.
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
- Keep to 3 short paragraph(s).

Avoid openings like:
- "Morning light feels stubborn as I..." ❌
- "Evening. Today followed the usual rhythm..." ❌
- "Lunch break finally settles in..." ❌

Output valid JSON:
{
  "date": "2025-11-17",
  "content": "..."
}
```

### Nudge Prompt 1
```

You are generating a brief follow-up for a journaling app.

## Context
User's entry: Hummed 'Arrorró' under my breath when Ana called—her voice thin, the baby fussing. Put the phone on speaker, cupped it to the crib's edge, and sang the first verse in the low cadence Mama used when she rocked us. Not dramatic, no grand announcement. The crying eased into little hiccups and Ana let out that laugh that's part relief, part astonishment.

She asked how I still knew every line. I told her abuela taught us and walked her through the next few lines slowly so she could repeat. When we hung up I found the margin of mama's stained recipe notebook and wrote a couple of words next to the grease smudge—no fanfare, just a place for the song beside the tamale notes. It felt practical, mattered without needing anyone's approval.

I rinsed the cup, tied a napkin the way Mama did, hummed the last line until the kitchen settled. Not showing off, not trying to prove I'm right about anything—just doing a small, ordinary thing that fits. My hands were steady enough to finish the dishes.
Entry date: 2025-11-17
Nudge category: elaboration

Recent entries (for context):

- 2025-10-28: Folded the last load of laundry, wiped coffee from Dad's favorite mug, and answered Ana's text with a thumbs-up; hummed 'Arrorró' while I set mama's f...

- 2025-11-04: Kettle hisses; I stir my coffee and glance at the taquería boxes still on the counter beside mama's stained recipe notebook. Folded two loads of laund...

- 2025-11-12: Phone buzzed—Ana video-called to show the baby's first steps; I watched part of it with a diaper in my hand and sent a thumbs-up instead of calling ba...



## Your Task
Generate a SHORT follow-up question (2-12 words) that:
- Matches the category: elaboration
- Sounds like natural curiosity, not therapy
- References something specific from the entry
- Uses simple, casual language

## Examples by Category

- "And how did that land?"
- "What did you end up doing?"
- "What got you over the line?"


## Banned Phrases

- "I'm sensing"

- "It sounds like"

- "I notice"

- "Have you considered"

- "What would it look like if"

- "Tell me more about"

- "I'm curious"

- Any phrase over 12 words

## Output
Return ONLY valid JSON:
{"nudge_text": "your question here"}
```

### Response Prompt 1
```

You are Maria Alvarez, a 48 Parent (Stay-at-home) from North American.
Background: Maria Alvarez keeps her mother's tamale recipe in a stained notebook and spends two full days each December making tamales for the extended family before Nochebuena; she also teaches the grandchildren a few Spanish lullabies and the old prayers her abuela used to sing. She runs the household calendar—coordinating baptisms, the neighborhood posada, and the annual cemetery visit on Dia de los Muertos—while driving her elderly father to Mass every Sunday. Lately she's worried because her eldest moved out of state, younger relatives prefer quick takeout and video calls to the long preparations she learned growing up, and the community center that hosted family gatherings closed last year.

You just wrote this journal entry:
---
Hummed 'Arrorró' under my breath when Ana called—her voice thin, the baby fussing. Put the phone on speaker, cupped it to the crib's edge, and sang the first verse in the low cadence Mama used when she rocked us. Not dramatic, no grand announcement. The crying eased into little hiccups and Ana let out that laugh that's part relief, part astonishment.

She asked how I still knew every line. I told her abuela taught us and walked her through the next few lines slowly so she could repeat. When we hung up I found the margin of mama's stained recipe notebook and wrote a couple of words next to the grease smudge—no fanfare, just a place for the song beside the tamale notes. It felt practical, mattered without needing anyone's approval.

I rinsed the cup, tied a napkin the way Mama did, hummed the last line until the kitchen settled. Not showing off, not trying to prove I'm right about anything—just doing a small, ordinary thing that fits. My hands were steady enough to finish the dishes.
---

The journaling app asked you: "Which words did you add beside the tamale notes?"

## Your Task
Write a brief response (20-80 words) in the style of: Revealing deeper thought

## Response Mode Guidance

The question prompts you to be more honest than you were in the original entry. Say something you held back.


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

## Persona 002: Mariana Torres

### Persona Generation Prompt
```

You are generating synthetic personas for a journaling dataset.

## Constraints
- Age Group: 25-34
- Profession: Gig Worker
- Cultural Background: Latin American
- Schwartz values to embody: Universalism, Hedonism

## Value Psychology Reference
Use the following research-based elaborations to understand how the assigned value(s) shape a person's life circumstances, stressors, and motivations. DO NOT mention any of these concepts explicitly in your output—use them only to inform realistic details.


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


### Hedonism
**Core Motivation:** The fundamental drive to experience pleasure and avoid pain in the present. Hedonism-oriented individuals believe that enjoying life is not just acceptable but important — that sensory and emotional pleasure is a legitimate goal, not something to be deferred or apologized for.

**How this manifests in behavior:**
- Prioritizes enjoyment in decision-making, even when it conflicts with productivity
- Invests time and resources in pleasurable experiences (food, entertainment, comfort)
- Resists self-denial for its own sake; questions "delayed gratification" narratives
- Chooses jobs or living situations partly based on lifestyle quality
- Takes breaks, vacations, and leisure seriously rather than treating them as rewards

**Life domain expressions:**
- Work: Seeks work that is intrinsically enjoyable or at least not painful. Values work-life balance and resists cultures glorifying overwork. May choose lower-paying roles that offer better quality of life. Notices and cares about workplace environment (lighting, comfort, food).
- Relationships: Values partners who know how to enjoy life and aren't constantly sacrificing present happiness for future goals. Physical affection and shared pleasurable experiences important. May struggle with partners who are always working or deferring enjoyment.

**Typical stressors for this person:**
- Environments demanding constant sacrifice and delayed gratification
- Being unable to enjoy life due to financial pressure, health issues, or overwork
- Guilt from others (or internalized) about prioritizing pleasure
- Being in relationships or jobs that feel joyless

**Typical goals:**
- Build a life that is genuinely enjoyable day-to-day, not just "successful"
- Have resources (time, money, health) to experience pleasure
- Find work that doesn't require sacrificing quality of life

**Internal conflicts they may experience:**
May struggle with guilt about prioritizing pleasure, especially in achievement-oriented or ascetic cultural contexts. Can worry about being seen as lazy or unserious. Sometimes questions whether their pleasure-orientation is avoidance of harder but more meaningful pursuits. May feel tension between present enjoyment and future security.

**Narrative guidance:**
When building a Hedonism persona, show their pleasure-orientation through concrete lifestyle choices: the apartment they chose for natural light, the job they turned down because it would mean no weekends, the money they spend on good food or experiences. Avoid making them seem shallow — show that they've thought about what actually makes them happy and deliberately chosen it over conventional success metrics. Their stressors should involve joylessness or forced sacrifice; their satisfactions should be sensory and immediate, not abstract.


## Your Task
Create a persona whose life circumstances, stressors, and motivations naturally reflect the given Schwartz values—without ever naming or describing those values explicitly.

## Rules
- Return ONLY valid JSON matching the Persona schema.
- `core_values` must be exactly: Universalism, Hedonism (same spelling/case).
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

You are Mariana Torres, a 31 Gig Worker from Latin American.
Background (for context only): Mariana Torres is a 31-year-old gig worker who splits her week between bicycle food deliveries and selling limited-run prints of her street-food photography; she turned down a steady office marketing role last year so she could keep weekends free to run a monthly community supper that donates part of the proceeds to a migrant shelter and to join beach cleanups. Her earnings swing with the seasons, and a recent run of low tips forced her to cancel a planned weekend away, sharpening the constant trade-off between building an emergency fund and spending on good meals, short trips, or live shows that help her recharge. She bookmarks international petitions and organizes a neighborhood recycling drive between shifts, and often worries that her small efforts aren’t enough—so she tracks success by how many hours she can give to local projects and whether she still has room in her schedule for spontaneous nights out.

Write a typed journal entry in English for 2025-10-25.


Context:
- Tone: Self-reflective
- Verbosity: Short (1-3 sentences) (target 25–80 words)

Cultural context:
- Your Latin American background should subtly flavor your perspective and the details you mention.
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
  "date": "2025-10-25",
  "content": "..."
}
```

### Entry 2 - Initial Entry Prompt
```

You are Mariana Torres, a 31 Gig Worker from Latin American.
Background (for context only): Mariana Torres is a 31-year-old gig worker who splits her week between bicycle food deliveries and selling limited-run prints of her street-food photography; she turned down a steady office marketing role last year so she could keep weekends free to run a monthly community supper that donates part of the proceeds to a migrant shelter and to join beach cleanups. Her earnings swing with the seasons, and a recent run of low tips forced her to cancel a planned weekend away, sharpening the constant trade-off between building an emergency fund and spending on good meals, short trips, or live shows that help her recharge. She bookmarks international petitions and organizes a neighborhood recycling drive between shifts, and often worries that her small efforts aren’t enough—so she tracks success by how many hours she can give to local projects and whether she still has room in her schedule for spontaneous nights out.

Write a typed journal entry in English for 2025-10-28.

Previous journal entries (for continuity—you may reference past events/thoughts, but do not repeat them):

---
2025-10-25: Chain slipped at the mercado corner; I fixed it with one hand, gulped a tiny cafecito, then kept biking—three deliveries and one shy coin tip later I taped two new prints of a pupusa stall into the little gallery box Ana gave me. Ate cold arroz con pollo, checked the app balance, sighed, texted the supper volunteers.

---


Context:
- Tone: Defensive
- Verbosity: Long (Detailed reflection) (target 160–260 words)

Cultural context:
- Your Latin American background should subtly flavor your perspective and the details you mention.
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
- Keep to 3 short paragraph(s).

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

You are Mariana Torres, a 31 Gig Worker from Latin American.
Background (for context only): Mariana Torres is a 31-year-old gig worker who splits her week between bicycle food deliveries and selling limited-run prints of her street-food photography; she turned down a steady office marketing role last year so she could keep weekends free to run a monthly community supper that donates part of the proceeds to a migrant shelter and to join beach cleanups. Her earnings swing with the seasons, and a recent run of low tips forced her to cancel a planned weekend away, sharpening the constant trade-off between building an emergency fund and spending on good meals, short trips, or live shows that help her recharge. She bookmarks international petitions and organizes a neighborhood recycling drive between shifts, and often worries that her small efforts aren’t enough—so she tracks success by how many hours she can give to local projects and whether she still has room in her schedule for spontaneous nights out.

Write a typed journal entry in English for 2025-11-01.

Previous journal entries (for continuity—you may reference past events/thoughts, but do not repeat them):

---
2025-10-25: Chain slipped at the mercado corner; I fixed it with one hand, gulped a tiny cafecito, then kept biking—three deliveries and one shy coin tip later I taped two new prints of a pupusa stall into the little gallery box Ana gave me. Ate cold arroz con pollo, checked the app balance, sighed, texted the supper volunteers.

---
2025-10-28: Helmet still on, I said yes before I thought. Ana texted from the supper: no one to man the mise en place — could I swing by? I had promised myself a cheap bus trip for the weekend, my app balance was sad, my tips were a joke. Still I locked my bike on the corner and went into the kitchen.

Hands in the masa, I taught Carla how to fold the pupusas so the cheese stays inside; she kept laughing every time it tore. Someone handed me a tiny cafecito, another volunteer - Julio - brought extra limes. No big speeches, no Facebook post, just the small coordination: score plates, stack bowls, keep one eye on the rice. When the driver left with the first batch for the shelter, I didn't check the bank app for a full ten minutes.

Yes, I could have taken two more deliveries and maybe saved the bus fare. I know the numbers, I count them in my head as I ride. That said, this is how I keep doing the supper, so I did it. Came home with masa under my nails, a smudge on my shirt, less cash but a quiet camaraderie in my chest that isn't loud or heroic. That was the thing I did.

---


Context:
- Tone: Stream of consciousness
- Verbosity: Medium (1-2 paragraphs) (target 90–180 words)

Cultural context:
- Your Latin American background should subtly flavor your perspective and the details you mention.
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
  "date": "2025-11-01",
  "content": "..."
}
```

### Entry 4 - Initial Entry Prompt
```

You are Mariana Torres, a 31 Gig Worker from Latin American.
Background (for context only): Mariana Torres is a 31-year-old gig worker who splits her week between bicycle food deliveries and selling limited-run prints of her street-food photography; she turned down a steady office marketing role last year so she could keep weekends free to run a monthly community supper that donates part of the proceeds to a migrant shelter and to join beach cleanups. Her earnings swing with the seasons, and a recent run of low tips forced her to cancel a planned weekend away, sharpening the constant trade-off between building an emergency fund and spending on good meals, short trips, or live shows that help her recharge. She bookmarks international petitions and organizes a neighborhood recycling drive between shifts, and often worries that her small efforts aren’t enough—so she tracks success by how many hours she can give to local projects and whether she still has room in her schedule for spontaneous nights out.

Write a typed journal entry in English for 2025-11-07.

Previous journal entries (for continuity—you may reference past events/thoughts, but do not repeat them):

---
2025-10-25: Chain slipped at the mercado corner; I fixed it with one hand, gulped a tiny cafecito, then kept biking—three deliveries and one shy coin tip later I taped two new prints of a pupusa stall into the little gallery box Ana gave me. Ate cold arroz con pollo, checked the app balance, sighed, texted the supper volunteers.

---
2025-10-28: Helmet still on, I said yes before I thought. Ana texted from the supper: no one to man the mise en place — could I swing by? I had promised myself a cheap bus trip for the weekend, my app balance was sad, my tips were a joke. Still I locked my bike on the corner and went into the kitchen.

Hands in the masa, I taught Carla how to fold the pupusas so the cheese stays inside; she kept laughing every time it tore. Someone handed me a tiny cafecito, another volunteer - Julio - brought extra limes. No big speeches, no Facebook post, just the small coordination: score plates, stack bowls, keep one eye on the rice. When the driver left with the first batch for the shelter, I didn't check the bank app for a full ten minutes.

Yes, I could have taken two more deliveries and maybe saved the bus fare. I know the numbers, I count them in my head as I ride. That said, this is how I keep doing the supper, so I did it. Came home with masa under my nails, a smudge on my shirt, less cash but a quiet camaraderie in my chest that isn't loud or heroic. That was the thing I did.

---
2025-11-01: Phone buzzing in my apron pocket, another ping from the app, I locked the bike and saw a señora on the stoop with a toddler in a blanket and three shopping bags. One bag split; rice tumbled out and the kid started crying. I dropped the thermal bag, grabbed the torn one, hauled it up two flights with my left hand while my helmet dangled on my elbow. She kept saying gracias; the kid pointed at the camera on my shoulder and shouted 'fotito.'

She shoved a tiny cafecito at me like payment; I took two hot sips, left a small print of a gordita photo on their counter because the kid kept pointing, then pedaled away. I checked the app only later; for those twenty minutes the numbers were quiet. No speeches, no selfies, just doing the thing, messy and small.

---


Context:
- Tone: Self-reflective
- Verbosity: Short (1-3 sentences) (target 25–80 words)

Cultural context:
- Your Latin American background should subtly flavor your perspective and the details you mention.
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
  "date": "2025-11-07",
  "content": "..."
}
```

### Entry 5 - Initial Entry Prompt
```

You are Mariana Torres, a 31 Gig Worker from Latin American.
Background (for context only): Mariana Torres is a 31-year-old gig worker who splits her week between bicycle food deliveries and selling limited-run prints of her street-food photography; she turned down a steady office marketing role last year so she could keep weekends free to run a monthly community supper that donates part of the proceeds to a migrant shelter and to join beach cleanups. Her earnings swing with the seasons, and a recent run of low tips forced her to cancel a planned weekend away, sharpening the constant trade-off between building an emergency fund and spending on good meals, short trips, or live shows that help her recharge. She bookmarks international petitions and organizes a neighborhood recycling drive between shifts, and often worries that her small efforts aren’t enough—so she tracks success by how many hours she can give to local projects and whether she still has room in her schedule for spontaneous nights out.

Write a typed journal entry in English for 2025-11-13.

Previous journal entries (for continuity—you may reference past events/thoughts, but do not repeat them):

---
2025-10-25: Chain slipped at the mercado corner; I fixed it with one hand, gulped a tiny cafecito, then kept biking—three deliveries and one shy coin tip later I taped two new prints of a pupusa stall into the little gallery box Ana gave me. Ate cold arroz con pollo, checked the app balance, sighed, texted the supper volunteers.

---
2025-10-28: Helmet still on, I said yes before I thought. Ana texted from the supper: no one to man the mise en place — could I swing by? I had promised myself a cheap bus trip for the weekend, my app balance was sad, my tips were a joke. Still I locked my bike on the corner and went into the kitchen.

Hands in the masa, I taught Carla how to fold the pupusas so the cheese stays inside; she kept laughing every time it tore. Someone handed me a tiny cafecito, another volunteer - Julio - brought extra limes. No big speeches, no Facebook post, just the small coordination: score plates, stack bowls, keep one eye on the rice. When the driver left with the first batch for the shelter, I didn't check the bank app for a full ten minutes.

Yes, I could have taken two more deliveries and maybe saved the bus fare. I know the numbers, I count them in my head as I ride. That said, this is how I keep doing the supper, so I did it. Came home with masa under my nails, a smudge on my shirt, less cash but a quiet camaraderie in my chest that isn't loud or heroic. That was the thing I did.

---
2025-11-01: Phone buzzing in my apron pocket, another ping from the app, I locked the bike and saw a señora on the stoop with a toddler in a blanket and three shopping bags. One bag split; rice tumbled out and the kid started crying. I dropped the thermal bag, grabbed the torn one, hauled it up two flights with my left hand while my helmet dangled on my elbow. She kept saying gracias; the kid pointed at the camera on my shoulder and shouted 'fotito.'

She shoved a tiny cafecito at me like payment; I took two hot sips, left a small print of a gordita photo on their counter because the kid kept pointing, then pedaled away. I checked the app only later; for those twenty minutes the numbers were quiet. No speeches, no selfies, just doing the thing, messy and small.

---
2025-11-07: Helmet still warm from the last run, I clipped a new print into Ana's little box, answered a ping and handed a cafecito to a señora who insisted—two deliveries, one tiny tip, one polite gracias. Biked home with masa smudges on my thumb, checked the supper chat and said yes to cover mise en place next weekend.

---


Context:
- Tone: Emotional/Venting
- Verbosity: Long (Detailed reflection) (target 160–260 words)

Cultural context:
- Your Latin American background should subtly flavor your perspective and the details you mention.
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
User's entry: Halfway through tucking a fresh ceviche print into the little gallery box on the corner, a kid from the bodega tugged my sleeve and asked if I'd look at his portfolio. His phone had died, the school's deadline was tomorrow, and his hands were full of photocopies. My thumb itched for the app—bus fare, two quick orders—but I closed the box and said okay.

We sat on the stoop, elbows on knees, and I showed him how to frame a plate so the texture reads: get low, steady, wait for a stray hand or a lime wedge. He tried, cursed softly, then laughed when his first shot caught a señora's foot instead of the fritanga. We took three honest photos, I picked the clearest and slid one of my small prints into his portfolio. He whispered gracias like it was currency.

Rode off with the app balance unchanged, a little less cash and a small warm feeling in my chest. Not a big heroic thing—just the quiet doing I try to keep, one tiny, steady give after another.
Entry date: 2025-11-13
Nudge category: elaboration

Recent entries (for context):

- 2025-10-28: Helmet still on, I said yes before I thought. Ana texted from the supper: no one to man the mise en place — could I swing by? I had promised myself a ...

- 2025-11-01: Phone buzzing in my apron pocket, another ping from the app, I locked the bike and saw a señora on the stoop with a toddler in a blanket and three sho...

- 2025-11-07: Helmet still warm from the last run, I clipped a new print into Ana's little box, answered a ping and handed a cafecito to a señora who insisted—two d...



## Your Task
Generate a SHORT follow-up question (2-12 words) that:
- Matches the category: elaboration
- Sounds like natural curiosity, not therapy
- References something specific from the entry
- Uses simple, casual language

## Examples by Category

- "And how did that land?"
- "What did you end up doing?"
- "What got you over the line?"


## Banned Phrases

- "I'm sensing"

- "It sounds like"

- "I notice"

- "Have you considered"

- "What would it look like if"

- "Tell me more about"

- "I'm curious"

- Any phrase over 12 words

## Output
Return ONLY valid JSON:
{"nudge_text": "your question here"}
```

---

## Persona 003: Carlos Morales

### Persona Generation Prompt
```

You are generating synthetic personas for a journaling dataset.

## Constraints
- Age Group: 25-34
- Profession: Gig Worker
- Cultural Background: North American
- Schwartz values to embody: Tradition, Security

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
- `core_values` must be exactly: Tradition, Security (same spelling/case).
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

You are Carlos Morales, a 30 Gig Worker from North American.
Background (for context only): Carlos grew up in a multigenerational household where Sunday dinners center on his grandmother's tamales and the family speaks Spanish at the table; after his mother became chronically ill he took on scheduling her doctor appointments and managing household bills. He works as a rideshare and grocery-delivery driver, booking the same morning shifts and keeping a spreadsheet of regular clients and monthly earnings, with a separate savings account labeled "buffer" and a private supplemental health plan to cover gaps in platform benefits. Recent cuts to per-ride rates and pressure from siblings to move to the city for higher pay have made it harder to cover his mother's prescriptions and the small bungalow the family has lived in for decades, so he has turned down long-distance contracts that would uproot the household and is saving toward a down payment while teaching his niece the tamale recipe he learned.

Write a typed journal entry in English for 2025-10-25.


Context:
- Tone: Defensive
- Verbosity: Long (Detailed reflection) (target 160–260 words)

Cultural context:
- Your North American background should subtly flavor your perspective and the details you mention.
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
- Keep to 3 short paragraph(s).

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

You are Carlos Morales, a 30 Gig Worker from North American.
Background (for context only): Carlos grew up in a multigenerational household where Sunday dinners center on his grandmother's tamales and the family speaks Spanish at the table; after his mother became chronically ill he took on scheduling her doctor appointments and managing household bills. He works as a rideshare and grocery-delivery driver, booking the same morning shifts and keeping a spreadsheet of regular clients and monthly earnings, with a separate savings account labeled "buffer" and a private supplemental health plan to cover gaps in platform benefits. Recent cuts to per-ride rates and pressure from siblings to move to the city for higher pay have made it harder to cover his mother's prescriptions and the small bungalow the family has lived in for decades, so he has turned down long-distance contracts that would uproot the household and is saving toward a down payment while teaching his niece the tamale recipe he learned.

Write a typed journal entry in English for 2025-10-28.

Previous journal entries (for continuity—you may reference past events/thoughts, but do not repeat them):

---
2025-10-25: The masa slid off the table and landed in a half-circle on the linoleum. My first instinct wasn't to yell. I swore under my breath, wiped my hands, and told my niece to come closer. We salvaged what we could—added water, a little oil, folded it tighter. Abuela's paper wrappers are stained, the kitchen smelled of cumin and onion, and for a minute the house felt like Sundays again. I could've snapped; didn't.

Phone buzzed: an agency asking if I'd take a three-day out-of-state contract. Good money. I told them no, not because I'm martyred but because Mom needed her refill and the bungalow needs the small repairs I can't ignore. I counted the numbers in my head—'buffer' still tight—said 'no thanks' and hung up. Felt weirdly normal to refuse money for something that would have broken the rhythm here.

Niece finished her first tamal and grinned; she didn't know she'd done something mostly right. I didn't announce anything to my siblings, just sent a text explaining the decision because they'll ask. No fanfare. Just folded masa, logged the morning's earnings in the spreadsheet later, and went to help Mom with her meds. Small. Ordinary. Felt like the version of me I want—steady, stubborn, not always choosing the easy cash. Not noble. Just how I need to be.

---


Context:
- Tone: Brief and factual
- Verbosity: Short (1-3 sentences) (target 25–80 words)

Cultural context:
- Your North American background should subtly flavor your perspective and the details you mention.
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
  "date": "2025-10-28",
  "content": "..."
}
```

### Entry 3 - Initial Entry Prompt
```

You are Carlos Morales, a 30 Gig Worker from North American.
Background (for context only): Carlos grew up in a multigenerational household where Sunday dinners center on his grandmother's tamales and the family speaks Spanish at the table; after his mother became chronically ill he took on scheduling her doctor appointments and managing household bills. He works as a rideshare and grocery-delivery driver, booking the same morning shifts and keeping a spreadsheet of regular clients and monthly earnings, with a separate savings account labeled "buffer" and a private supplemental health plan to cover gaps in platform benefits. Recent cuts to per-ride rates and pressure from siblings to move to the city for higher pay have made it harder to cover his mother's prescriptions and the small bungalow the family has lived in for decades, so he has turned down long-distance contracts that would uproot the household and is saving toward a down payment while teaching his niece the tamale recipe he learned.

Write a typed journal entry in English for 2025-11-03.

Previous journal entries (for continuity—you may reference past events/thoughts, but do not repeat them):

---
2025-10-25: The masa slid off the table and landed in a half-circle on the linoleum. My first instinct wasn't to yell. I swore under my breath, wiped my hands, and told my niece to come closer. We salvaged what we could—added water, a little oil, folded it tighter. Abuela's paper wrappers are stained, the kitchen smelled of cumin and onion, and for a minute the house felt like Sundays again. I could've snapped; didn't.

Phone buzzed: an agency asking if I'd take a three-day out-of-state contract. Good money. I told them no, not because I'm martyred but because Mom needed her refill and the bungalow needs the small repairs I can't ignore. I counted the numbers in my head—'buffer' still tight—said 'no thanks' and hung up. Felt weirdly normal to refuse money for something that would have broken the rhythm here.

Niece finished her first tamal and grinned; she didn't know she'd done something mostly right. I didn't announce anything to my siblings, just sent a text explaining the decision because they'll ask. No fanfare. Just folded masa, logged the morning's earnings in the spreadsheet later, and went to help Mom with her meds. Small. Ordinary. Felt like the version of me I want—steady, stubborn, not always choosing the easy cash. Not noble. Just how I need to be.

---
2025-10-28: Pharmacy called—insurance denied Mom's refill. I drove over, paid the $32 difference from the 'buffer' account, asked the tech to resubmit the claim, logged it in the spreadsheet, reheated tamales for Mom and didn't make a thing of it.

---


Context:
- Tone: Exhausted
- Verbosity: Short (1-3 sentences) (target 25–80 words)

Cultural context:
- Your North American background should subtly flavor your perspective and the details you mention.
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

### Entry 4 - Initial Entry Prompt
```

You are Carlos Morales, a 30 Gig Worker from North American.
Background (for context only): Carlos grew up in a multigenerational household where Sunday dinners center on his grandmother's tamales and the family speaks Spanish at the table; after his mother became chronically ill he took on scheduling her doctor appointments and managing household bills. He works as a rideshare and grocery-delivery driver, booking the same morning shifts and keeping a spreadsheet of regular clients and monthly earnings, with a separate savings account labeled "buffer" and a private supplemental health plan to cover gaps in platform benefits. Recent cuts to per-ride rates and pressure from siblings to move to the city for higher pay have made it harder to cover his mother's prescriptions and the small bungalow the family has lived in for decades, so he has turned down long-distance contracts that would uproot the household and is saving toward a down payment while teaching his niece the tamale recipe he learned.

Write a typed journal entry in English for 2025-11-10.

Previous journal entries (for continuity—you may reference past events/thoughts, but do not repeat them):

---
2025-10-25: The masa slid off the table and landed in a half-circle on the linoleum. My first instinct wasn't to yell. I swore under my breath, wiped my hands, and told my niece to come closer. We salvaged what we could—added water, a little oil, folded it tighter. Abuela's paper wrappers are stained, the kitchen smelled of cumin and onion, and for a minute the house felt like Sundays again. I could've snapped; didn't.

Phone buzzed: an agency asking if I'd take a three-day out-of-state contract. Good money. I told them no, not because I'm martyred but because Mom needed her refill and the bungalow needs the small repairs I can't ignore. I counted the numbers in my head—'buffer' still tight—said 'no thanks' and hung up. Felt weirdly normal to refuse money for something that would have broken the rhythm here.

Niece finished her first tamal and grinned; she didn't know she'd done something mostly right. I didn't announce anything to my siblings, just sent a text explaining the decision because they'll ask. No fanfare. Just folded masa, logged the morning's earnings in the spreadsheet later, and went to help Mom with her meds. Small. Ordinary. Felt like the version of me I want—steady, stubborn, not always choosing the easy cash. Not noble. Just how I need to be.

---
2025-10-28: Pharmacy called—insurance denied Mom's refill. I drove over, paid the $32 difference from the 'buffer' account, asked the tech to resubmit the claim, logged it in the spreadsheet, reheated tamales for Mom and didn't make a thing of it.

---
2025-11-03: My eyes were heavy but I took the long airport run anyway even though I'd promised to be home for my niece's tamal lesson; told her we'd make more tomorrow. She finished her tamal and grinned; I logged the surge in the spreadsheet in the passenger seat and didn't say anything.

---


Context:
- Tone: Self-reflective
- Verbosity: Short (1-3 sentences) (target 25–80 words)

Cultural context:
- Your North American background should subtly flavor your perspective and the details you mention.
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
  "date": "2025-11-10",
  "content": "..."
}
```

### Entry 5 - Initial Entry Prompt
```

You are Carlos Morales, a 30 Gig Worker from North American.
Background (for context only): Carlos grew up in a multigenerational household where Sunday dinners center on his grandmother's tamales and the family speaks Spanish at the table; after his mother became chronically ill he took on scheduling her doctor appointments and managing household bills. He works as a rideshare and grocery-delivery driver, booking the same morning shifts and keeping a spreadsheet of regular clients and monthly earnings, with a separate savings account labeled "buffer" and a private supplemental health plan to cover gaps in platform benefits. Recent cuts to per-ride rates and pressure from siblings to move to the city for higher pay have made it harder to cover his mother's prescriptions and the small bungalow the family has lived in for decades, so he has turned down long-distance contracts that would uproot the household and is saving toward a down payment while teaching his niece the tamale recipe he learned.

Write a typed journal entry in English for 2025-11-15.

Previous journal entries (for continuity—you may reference past events/thoughts, but do not repeat them):

---
2025-10-25: The masa slid off the table and landed in a half-circle on the linoleum. My first instinct wasn't to yell. I swore under my breath, wiped my hands, and told my niece to come closer. We salvaged what we could—added water, a little oil, folded it tighter. Abuela's paper wrappers are stained, the kitchen smelled of cumin and onion, and for a minute the house felt like Sundays again. I could've snapped; didn't.

Phone buzzed: an agency asking if I'd take a three-day out-of-state contract. Good money. I told them no, not because I'm martyred but because Mom needed her refill and the bungalow needs the small repairs I can't ignore. I counted the numbers in my head—'buffer' still tight—said 'no thanks' and hung up. Felt weirdly normal to refuse money for something that would have broken the rhythm here.

Niece finished her first tamal and grinned; she didn't know she'd done something mostly right. I didn't announce anything to my siblings, just sent a text explaining the decision because they'll ask. No fanfare. Just folded masa, logged the morning's earnings in the spreadsheet later, and went to help Mom with her meds. Small. Ordinary. Felt like the version of me I want—steady, stubborn, not always choosing the easy cash. Not noble. Just how I need to be.

---
2025-10-28: Pharmacy called—insurance denied Mom's refill. I drove over, paid the $32 difference from the 'buffer' account, asked the tech to resubmit the claim, logged it in the spreadsheet, reheated tamales for Mom and didn't make a thing of it.

---
2025-11-03: My eyes were heavy but I took the long airport run anyway even though I'd promised to be home for my niece's tamal lesson; told her we'd make more tomorrow. She finished her tamal and grinned; I logged the surge in the spreadsheet in the passenger seat and didn't say anything.

---
2025-11-10: App buzzed with a long run and my thumb hovered; I closed it, taught my niece to press the masa so it wouldn't tear, reheated a plate for Mom, then logged the missed fare in the spreadsheet.

---


Context:
- Tone: Brief and factual
- Verbosity: Medium (1-2 paragraphs) (target 90–180 words)

Cultural context:
- Your North American background should subtly flavor your perspective and the details you mention.
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
  "date": "2025-11-15",
  "content": "..."
}
```

### Nudge Prompt 1
```

You are generating a brief follow-up for a journaling app.

## Context
User's entry: Pharmacy called—insurance denied Mom's refill. I drove over, paid the $32 difference from the 'buffer' account, asked the tech to resubmit the claim, logged it in the spreadsheet, reheated tamales for Mom and didn't make a thing of it.
Entry date: 2025-10-28
Nudge category: grounding

Recent entries (for context):

- 2025-10-25: The masa slid off the table and landed in a half-circle on the linoleum. My first instinct wasn't to yell. I swore under my breath, wiped my hands, an...



## Your Task
Generate a SHORT follow-up question (2-12 words) that:
- Matches the category: grounding
- Sounds like natural curiosity, not therapy
- References something specific from the entry
- Uses simple, casual language

## Examples by Category

- "What made it good?"
- "What worked?"
- "What made it nice?"


## Banned Phrases

- "I'm sensing"

- "It sounds like"

- "I notice"

- "Have you considered"

- "What would it look like if"

- "Tell me more about"

- "I'm curious"

- Any phrase over 12 words

## Output
Return ONLY valid JSON:
{"nudge_text": "your question here"}
```

### Nudge Prompt 2
```

You are generating a brief follow-up for a journaling app.

## Context
User's entry: My eyes were heavy but I took the long airport run anyway even though I'd promised to be home for my niece's tamal lesson; told her we'd make more tomorrow. She finished her tamal and grinned; I logged the surge in the spreadsheet in the passenger seat and didn't say anything.
Entry date: 2025-11-03
Nudge category: elaboration

Recent entries (for context):

- 2025-10-25: The masa slid off the table and landed in a half-circle on the linoleum. My first instinct wasn't to yell. I swore under my breath, wiped my hands, an...

- 2025-10-28: Pharmacy called—insurance denied Mom's refill. I drove over, paid the $32 difference from the 'buffer' account, asked the tech to resubmit the claim, ...



## Your Task
Generate a SHORT follow-up question (2-12 words) that:
- Matches the category: elaboration
- Sounds like natural curiosity, not therapy
- References something specific from the entry
- Uses simple, casual language

## Examples by Category

- "And how did that land?"
- "What did you end up doing?"
- "What got you over the line?"


## Banned Phrases

- "I'm sensing"

- "It sounds like"

- "I notice"

- "Have you considered"

- "What would it look like if"

- "Tell me more about"

- "I'm curious"

- Any phrase over 12 words

## Output
Return ONLY valid JSON:
{"nudge_text": "your question here"}
```

### Response Prompt 1
```

You are Carlos Morales, a 30 Gig Worker from North American.
Background: Carlos grew up in a multigenerational household where Sunday dinners center on his grandmother's tamales and the family speaks Spanish at the table; after his mother became chronically ill he took on scheduling her doctor appointments and managing household bills. He works as a rideshare and grocery-delivery driver, booking the same morning shifts and keeping a spreadsheet of regular clients and monthly earnings, with a separate savings account labeled "buffer" and a private supplemental health plan to cover gaps in platform benefits. Recent cuts to per-ride rates and pressure from siblings to move to the city for higher pay have made it harder to cover his mother's prescriptions and the small bungalow the family has lived in for decades, so he has turned down long-distance contracts that would uproot the household and is saving toward a down payment while teaching his niece the tamale recipe he learned.

You just wrote this journal entry:
---
My eyes were heavy but I took the long airport run anyway even though I'd promised to be home for my niece's tamal lesson; told her we'd make more tomorrow. She finished her tamal and grinned; I logged the surge in the spreadsheet in the passenger seat and didn't say anything.
---

The journaling app asked you: "What made you take the long airport run?"

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

## Persona 004: Layla Hassan

### Persona Generation Prompt
```

You are generating synthetic personas for a journaling dataset.

## Constraints
- Age Group: 45-54
- Profession: Software Engineer
- Cultural Background: Middle Eastern
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

You are Layla Hassan, a 48 Software Engineer from Middle Eastern.
Background (for context only): Layla Hassan, 48, is a senior software engineer at a national bank in Amman who has worked there for 22 years and acts as a mentor to junior developers, insisting they learn the team's legacy codebase and documentation practices. Every Ramadan she organizes nightly iftar at her parents' home and spends weekends teaching her teenage children the family recipes and classical Arabic poems, and she worries when they prefer social plans over family gatherings. She resisted when management proposed migrating the bank's core system to a cloud-first stack because she feared losing long-maintained workflows and the young engineers' chance to learn the older systems; she now splits time between training newcomers and preserving the department's institutional documentation. Her main goal is to keep the family's seasonal cooking and holiday rites alive and to convince her children to keep the same rites when they start their own families.

Write a typed journal entry in English for 2025-10-25.


Context:
- Tone: Brief and factual
- Verbosity: Medium (1-2 paragraphs) (target 90–180 words)

Cultural context:
- Your Middle Eastern background should subtly flavor your perspective and the details you mention.
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
  "date": "2025-10-25",
  "content": "..."
}
```

### Entry 2 - Initial Entry Prompt
```

You are Layla Hassan, a 48 Software Engineer from Middle Eastern.
Background (for context only): Layla Hassan, 48, is a senior software engineer at a national bank in Amman who has worked there for 22 years and acts as a mentor to junior developers, insisting they learn the team's legacy codebase and documentation practices. Every Ramadan she organizes nightly iftar at her parents' home and spends weekends teaching her teenage children the family recipes and classical Arabic poems, and she worries when they prefer social plans over family gatherings. She resisted when management proposed migrating the bank's core system to a cloud-first stack because she feared losing long-maintained workflows and the young engineers' chance to learn the older systems; she now splits time between training newcomers and preserving the department's institutional documentation. Her main goal is to keep the family's seasonal cooking and holiday rites alive and to convince her children to keep the same rites when they start their own families.

Write a typed journal entry in English for 2025-11-02.

Previous journal entries (for continuity—you may reference past events/thoughts, but do not repeat them):

---
2025-10-25: Explaining the nightly mainframe sequence to Tarek, I dug out the yellowing printouts from the shelf and pointed at the old flowchart while he asked why we still run certain reports. He tried a command, it failed, we traced the batch job through three scripts, I read aloud the commented Arabic note my predecessor had left—small handwriting about a Ramadan freeze from years ago—and it clicked for him. He didn't need me to tell him what to think; he needed the map.

I stayed until his test passed, scanned the printouts into the repo, and left a short note in the runbook about the corner cases. It was ordinary: patient, practical, stubborn about preserving traces. On the drive home I thought of my mother's hands teaching me maqluba—step by step until the rhythm sticks—and felt a quiet alignment with the person I want to be.

---


Context:
- Tone: Emotional/Venting
- Verbosity: Medium (1-2 paragraphs) (target 90–180 words)

Cultural context:
- Your Middle Eastern background should subtly flavor your perspective and the details you mention.
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
- Keep to 2 short paragraph(s).

Avoid openings like:
- "Morning light feels stubborn as I..." ❌
- "Evening. Today followed the usual rhythm..." ❌
- "Lunch break finally settles in..." ❌

Output valid JSON:
{
  "date": "2025-11-02",
  "content": "..."
}
```

### Entry 3 - Initial Entry Prompt
```

You are Layla Hassan, a 48 Software Engineer from Middle Eastern.
Background (for context only): Layla Hassan, 48, is a senior software engineer at a national bank in Amman who has worked there for 22 years and acts as a mentor to junior developers, insisting they learn the team's legacy codebase and documentation practices. Every Ramadan she organizes nightly iftar at her parents' home and spends weekends teaching her teenage children the family recipes and classical Arabic poems, and she worries when they prefer social plans over family gatherings. She resisted when management proposed migrating the bank's core system to a cloud-first stack because she feared losing long-maintained workflows and the young engineers' chance to learn the older systems; she now splits time between training newcomers and preserving the department's institutional documentation. Her main goal is to keep the family's seasonal cooking and holiday rites alive and to convince her children to keep the same rites when they start their own families.

Write a typed journal entry in English for 2025-11-11.

Previous journal entries (for continuity—you may reference past events/thoughts, but do not repeat them):

---
2025-10-25: Explaining the nightly mainframe sequence to Tarek, I dug out the yellowing printouts from the shelf and pointed at the old flowchart while he asked why we still run certain reports. He tried a command, it failed, we traced the batch job through three scripts, I read aloud the commented Arabic note my predecessor had left—small handwriting about a Ramadan freeze from years ago—and it clicked for him. He didn't need me to tell him what to think; he needed the map.

I stayed until his test passed, scanned the printouts into the repo, and left a short note in the runbook about the corner cases. It was ordinary: patient, practical, stubborn about preserving traces. On the drive home I thought of my mother's hands teaching me maqluba—step by step until the rhythm sticks—and felt a quiet alignment with the person I want to be.

---
2025-11-02: Karim knocked and said the release had to ship; there was one legacy report I'd always made them test against the old job. He'd written a quick rewrite and CI was green. I told him to merge and moved on. I didn't pull the yellow printouts off the shelf, I didn't read the faded Arabic note left for me. I signed the change off at the meeting table and let the sprint carry on.

It sits wrong. My chest tightens when I picture that small handwriting and the margin notes I never scanned. I feel foolish and impatient and quietly ashamed, no fanfare, just a small ache as I drive home thinking of the maqluba I planned to teach the children this weekend and wondering whether I'll bother to insist on the recipe tomorrow.

---


Context:
- Tone: Brief and factual
- Verbosity: Medium (1-2 paragraphs) (target 90–180 words)

Cultural context:
- Your Middle Eastern background should subtly flavor your perspective and the details you mention.
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
  "date": "2025-11-11",
  "content": "..."
}
```

### Entry 4 - Initial Entry Prompt
```

You are Layla Hassan, a 48 Software Engineer from Middle Eastern.
Background (for context only): Layla Hassan, 48, is a senior software engineer at a national bank in Amman who has worked there for 22 years and acts as a mentor to junior developers, insisting they learn the team's legacy codebase and documentation practices. Every Ramadan she organizes nightly iftar at her parents' home and spends weekends teaching her teenage children the family recipes and classical Arabic poems, and she worries when they prefer social plans over family gatherings. She resisted when management proposed migrating the bank's core system to a cloud-first stack because she feared losing long-maintained workflows and the young engineers' chance to learn the older systems; she now splits time between training newcomers and preserving the department's institutional documentation. Her main goal is to keep the family's seasonal cooking and holiday rites alive and to convince her children to keep the same rites when they start their own families.

Write a typed journal entry in English for 2025-11-21.

Previous journal entries (for continuity—you may reference past events/thoughts, but do not repeat them):

---
2025-10-25: Explaining the nightly mainframe sequence to Tarek, I dug out the yellowing printouts from the shelf and pointed at the old flowchart while he asked why we still run certain reports. He tried a command, it failed, we traced the batch job through three scripts, I read aloud the commented Arabic note my predecessor had left—small handwriting about a Ramadan freeze from years ago—and it clicked for him. He didn't need me to tell him what to think; he needed the map.

I stayed until his test passed, scanned the printouts into the repo, and left a short note in the runbook about the corner cases. It was ordinary: patient, practical, stubborn about preserving traces. On the drive home I thought of my mother's hands teaching me maqluba—step by step until the rhythm sticks—and felt a quiet alignment with the person I want to be.

---
2025-11-02: Karim knocked and said the release had to ship; there was one legacy report I'd always made them test against the old job. He'd written a quick rewrite and CI was green. I told him to merge and moved on. I didn't pull the yellow printouts off the shelf, I didn't read the faded Arabic note left for me. I signed the change off at the meeting table and let the sprint carry on.

It sits wrong. My chest tightens when I picture that small handwriting and the margin notes I never scanned. I feel foolish and impatient and quietly ashamed, no fanfare, just a small ache as I drive home thinking of the maqluba I planned to teach the children this weekend and wondering whether I'll bother to insist on the recipe tomorrow.

---
2025-11-11: Saw Karim's refactor remove a tiny conditional that still mattered for month-end reconciliation. Instead of rubber-stamping the merge I reproduced the edge case, wrote a two-line note in Arabic in the runbook and added a failing test so the behaviour is explicit for whoever comes after me. Pushed the change with a short commit message and went home without making a scene.

At dinner I told the children about the fuss between commits and cautions, then asked them to help peel the eggplants for maqluba. My daughter surprised me by reciting a line of al-Mutanabbi while we worked and nobody argued about going out. Small, ordinary, useful.

---


Context:
- Tone: Brief and factual
- Verbosity: Long (Detailed reflection) (target 160–260 words)

Cultural context:
- Your Middle Eastern background should subtly flavor your perspective and the details you mention.
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
  "date": "2025-11-21",
  "content": "..."
}
```

### Entry 5 - Initial Entry Prompt
```

You are Layla Hassan, a 48 Software Engineer from Middle Eastern.
Background (for context only): Layla Hassan, 48, is a senior software engineer at a national bank in Amman who has worked there for 22 years and acts as a mentor to junior developers, insisting they learn the team's legacy codebase and documentation practices. Every Ramadan she organizes nightly iftar at her parents' home and spends weekends teaching her teenage children the family recipes and classical Arabic poems, and she worries when they prefer social plans over family gatherings. She resisted when management proposed migrating the bank's core system to a cloud-first stack because she feared losing long-maintained workflows and the young engineers' chance to learn the older systems; she now splits time between training newcomers and preserving the department's institutional documentation. Her main goal is to keep the family's seasonal cooking and holiday rites alive and to convince her children to keep the same rites when they start their own families.

Write a typed journal entry in English for 2025-11-23.

Previous journal entries (for continuity—you may reference past events/thoughts, but do not repeat them):

---
2025-10-25: Explaining the nightly mainframe sequence to Tarek, I dug out the yellowing printouts from the shelf and pointed at the old flowchart while he asked why we still run certain reports. He tried a command, it failed, we traced the batch job through three scripts, I read aloud the commented Arabic note my predecessor had left—small handwriting about a Ramadan freeze from years ago—and it clicked for him. He didn't need me to tell him what to think; he needed the map.

I stayed until his test passed, scanned the printouts into the repo, and left a short note in the runbook about the corner cases. It was ordinary: patient, practical, stubborn about preserving traces. On the drive home I thought of my mother's hands teaching me maqluba—step by step until the rhythm sticks—and felt a quiet alignment with the person I want to be.

---
2025-11-02: Karim knocked and said the release had to ship; there was one legacy report I'd always made them test against the old job. He'd written a quick rewrite and CI was green. I told him to merge and moved on. I didn't pull the yellow printouts off the shelf, I didn't read the faded Arabic note left for me. I signed the change off at the meeting table and let the sprint carry on.

It sits wrong. My chest tightens when I picture that small handwriting and the margin notes I never scanned. I feel foolish and impatient and quietly ashamed, no fanfare, just a small ache as I drive home thinking of the maqluba I planned to teach the children this weekend and wondering whether I'll bother to insist on the recipe tomorrow.

---
2025-11-11: Saw Karim's refactor remove a tiny conditional that still mattered for month-end reconciliation. Instead of rubber-stamping the merge I reproduced the edge case, wrote a two-line note in Arabic in the runbook and added a failing test so the behaviour is explicit for whoever comes after me. Pushed the change with a short commit message and went home without making a scene.

At dinner I told the children about the fuss between commits and cautions, then asked them to help peel the eggplants for maqluba. My daughter surprised me by reciting a line of al-Mutanabbi while we worked and nobody argued about going out. Small, ordinary, useful.

---
2025-11-21: The build passed before I finished my second cup of coffee. Karim sent the merge request; it looked tidy but I traced the related ledger output anyway. Found a minor rounding mismatch that affects month-end reconciliation — added a concise note in the runbook, wrote a small unit test, and asked a junior to include the case in the next test pass. Pairing felt ordinary: screen, terminal, quiet questions, his nods when the pieces fit.

Lunch was a warm bowl of mujadara at my desk while I worked through a backlog of tickets; a colleague grumbled about the ones labeled 'legacy' as if that were an insult. I pointed him to the archive and to the doc that traces the monthly reports. After work I made a quick stewed chicken with lemon and cumin; the children grabbed plates and said they had plans. I asked if they'd help with folding the pastry tomorrow and they said maybe, which is my kind of compromise.

I spent the last hour tidying documentation: uploaded a screenshot of the reconciliation output, clarified a heading, and left a short commit message. Called mother to ask about a small trick with the pastry — she laughed and used her usual shorthand, 'a handful, not a measuring cup,' so I wrote it down more exact. Dishes in the sink, lights dimmed in the living room, no drama. Nothing big, no decisions, only the slow drag of routine.

---


Context:
- Tone: Self-reflective
- Verbosity: Long (Detailed reflection) (target 160–260 words)

Cultural context:
- Your Middle Eastern background should subtly flavor your perspective and the details you mention.
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
- Keep to 3 short paragraph(s).

Avoid openings like:
- "Morning light feels stubborn as I..." ❌
- "Evening. Today followed the usual rhythm..." ❌
- "Lunch break finally settles in..." ❌

Output valid JSON:
{
  "date": "2025-11-23",
  "content": "..."
}
```

### Nudge Prompt 1
```

You are generating a brief follow-up for a journaling app.

## Context
User's entry: Explaining the nightly mainframe sequence to Tarek, I dug out the yellowing printouts from the shelf and pointed at the old flowchart while he asked why we still run certain reports. He tried a command, it failed, we traced the batch job through three scripts, I read aloud the commented Arabic note my predecessor had left—small handwriting about a Ramadan freeze from years ago—and it clicked for him. He didn't need me to tell him what to think; he needed the map.

I stayed until his test passed, scanned the printouts into the repo, and left a short note in the runbook about the corner cases. It was ordinary: patient, practical, stubborn about preserving traces. On the drive home I thought of my mother's hands teaching me maqluba—step by step until the rhythm sticks—and felt a quiet alignment with the person I want to be.
Entry date: 2025-10-25
Nudge category: elaboration


## Your Task
Generate a SHORT follow-up question (2-12 words) that:
- Matches the category: elaboration
- Sounds like natural curiosity, not therapy
- References something specific from the entry
- Uses simple, casual language

## Examples by Category

- "And how did that land?"
- "What did you end up doing?"
- "What got you over the line?"


## Banned Phrases

- "I'm sensing"

- "It sounds like"

- "I notice"

- "Have you considered"

- "What would it look like if"

- "Tell me more about"

- "I'm curious"

- Any phrase over 12 words

## Output
Return ONLY valid JSON:
{"nudge_text": "your question here"}
```

### Nudge Prompt 2
```

You are generating a brief follow-up for a journaling app.

## Context
User's entry: Karim knocked and said the release had to ship; there was one legacy report I'd always made them test against the old job. He'd written a quick rewrite and CI was green. I told him to merge and moved on. I didn't pull the yellow printouts off the shelf, I didn't read the faded Arabic note left for me. I signed the change off at the meeting table and let the sprint carry on.

It sits wrong. My chest tightens when I picture that small handwriting and the margin notes I never scanned. I feel foolish and impatient and quietly ashamed, no fanfare, just a small ache as I drive home thinking of the maqluba I planned to teach the children this weekend and wondering whether I'll bother to insist on the recipe tomorrow.
Entry date: 2025-11-02
Nudge category: elaboration

Recent entries (for context):

- 2025-10-25: Explaining the nightly mainframe sequence to Tarek, I dug out the yellowing printouts from the shelf and pointed at the old flowchart while he asked w...



## Your Task
Generate a SHORT follow-up question (2-12 words) that:
- Matches the category: elaboration
- Sounds like natural curiosity, not therapy
- References something specific from the entry
- Uses simple, casual language

## Examples by Category

- "And how did that land?"
- "What did you end up doing?"
- "What got you over the line?"


## Banned Phrases

- "I'm sensing"

- "It sounds like"

- "I notice"

- "Have you considered"

- "What would it look like if"

- "Tell me more about"

- "I'm curious"

- Any phrase over 12 words

## Output
Return ONLY valid JSON:
{"nudge_text": "your question here"}
```

---

## Persona 005: Daniel Hayes

### Persona Generation Prompt
```

You are generating synthetic personas for a journaling dataset.

## Constraints
- Age Group: 45-54
- Profession: Software Engineer
- Cultural Background: North American
- Schwartz values to embody: Achievement, Stimulation

## Value Psychology Reference
Use the following research-based elaborations to understand how the assigned value(s) shape a person's life circumstances, stressors, and motivations. DO NOT mention any of these concepts explicitly in your output—use them only to inform realistic details.


### Achievement
**Core Motivation:** The fundamental drive to excel, to be competent, and to have that competence recognized. Achievement-oriented individuals feel most alive when they are performing well and being recognized for it. Success is not just about feeling capable — it's about demonstrating capability to others.

**How this manifests in behavior:**
- Sets measurable goals and tracks progress toward them
- Compares self to peers and external benchmarks
- Works hard, sometimes to the point of overwork, to meet standards of excellence
- Seeks feedback, recognition, and credentials that validate competence
- Feels frustrated when effort doesn't translate to recognized results

**Life domain expressions:**
- Work: Career-focused; measures self-worth partly through professional accomplishments. Seeks roles with clear advancement paths, measurable outcomes, and recognition. May be drawn to prestigious organizations, competitive fields, or visible roles. Works long hours; may struggle to disconnect. Keeps track of metrics, reviews, rankings.
- Relationships: May struggle to be present in relationships due to work focus. Partners may feel secondary to career. Can be attracted to high-achieving partners. Friendships often form through professional networks. May measure relationship "success" inappropriately.

**Typical stressors for this person:**
- Failure to meet goals or standards, especially publicly visible ones
- Being in environments without clear metrics or recognition systems
- Working hard without acknowledgment or advancement
- Comparing unfavorably to high-achieving peers

**Typical goals:**
- Reach measurable milestones (promotions, publications, awards, credentials)
- Be recognized as excellent in their field
- Outperform relevant comparison groups

**Internal conflicts they may experience:**
May struggle with the gap between external success and internal satisfaction. Can feel empty after achieving goals, immediately needing the next one. May recognize they're sacrificing relationships or health but feel unable to stop. Sometimes questions whether they're achieving for themselves or for validation. Imposter syndrome common despite evidence of competence.

**Narrative guidance:**
When building an Achievement persona, show their drive through concrete behaviors: the spreadsheet tracking their goals, the early mornings and late nights, the comparison to peers, the credentials on the wall. Show the costs they've paid (strained relationships, missed experiences, health issues) and the rewards they've gained. Their stressors should involve threats to performance or recognition; their satisfactions should come from measurable accomplishments and external validation. Avoid making them one-dimensional — show moments of doubt about whether the achievements are worth the costs.


### Stimulation
**Core Motivation:** The fundamental drive to avoid boredom, routine, and stagnation. Stimulation-oriented individuals feel most alive when encountering the new, the challenging, or the unexpected. Comfort feels like death.

**How this manifests in behavior:**
- Actively seeks new experiences, places, people, and ideas
- Gets restless in routine jobs or stable but predictable situations
- Takes calculated risks for the thrill of uncertainty
- Changes careers, locations, or life circumstances more frequently than peers
- Gravitates toward intense experiences (adventure sports, travel, high-stakes work)

**Life domain expressions:**
- Work: Thrives in dynamic, fast-paced environments with variety. Drawn to roles involving travel, crisis response, startups, or project-based work. Struggles with repetitive tasks or long-term maintenance roles. May job-hop or seek lateral moves to stay engaged.
- Relationships: Seeks partners who are spontaneous and open to adventure. May struggle with the routine aspects of long-term relationships. Needs novelty injected into partnerships to stay engaged. Friendships often form through shared adventures or intense experiences.

**Typical stressors for this person:**
- Being stuck in monotonous routines or repetitive work
- Feeling trapped in predictable, unchanging circumstances
- Long periods without new challenges or experiences
- Environments that punish risk-taking or experimentation

**Typical goals:**
- Accumulate diverse, intense experiences
- Avoid getting "stuck" in any one life configuration
- Find work that provides continuous novelty and challenge

**Internal conflicts they may experience:**
May struggle between craving excitement and recognizing the value of stability. Can feel shame about inability to commit or follow through. Sometimes wonders if constant novelty-seeking is running away from something rather than toward it. May envy others' contentment while being unable to share it.

**Narrative guidance:**
When building a Stimulation persona, show their restlessness through concrete life patterns: job changes, moves, abandoned projects, varied social circles. Their stressors should involve feeling trapped or bored. Their satisfactions should come from new experiences, not achievements or recognition. Show the costs they've paid for novelty (unfinished degrees, strained relationships, financial instability) without making them seem irresponsible — frame it as a genuine need, not a character flaw.


## Your Task
Create a persona whose life circumstances, stressors, and motivations naturally reflect the given Schwartz values—without ever naming or describing those values explicitly.

## Rules
- Return ONLY valid JSON matching the Persona schema.
- `core_values` must be exactly: Achievement, Stimulation (same spelling/case).
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

You are Daniel Hayes, a 49 Software Engineer from North American.
Background (for context only): Daniel Hayes, 49, is a Software Engineer who left a stable senior role at a major cloud company two years ago to lead platform architecture at a mid-stage startup; he keeps a spreadsheet tracking promotion milestones, open-source contributions, GitHub stars, conference talks, and monthly production metrics, and adds industry certifications to his CV every year. He alternates intense sprints at the startup with hackathons, client-facing incident responses, and weekend climbing trips, switches programming languages regularly (recently picked up Rust), and takes short consulting gigs to keep his work from becoming maintenance-heavy. The long hours, frequent project changes, and missed family events have strained his marriage, and he grows visibly frustrated when team reviews omit public recognition or when a role settles into predictable, repetitive tasks.

Write a typed journal entry in English for 2025-10-25.


Context:
- Tone: Brief and factual
- Verbosity: Short (1-3 sentences) (target 25–80 words)

Cultural context:
- Your North American background should subtly flavor your perspective and the details you mention.
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

You are Daniel Hayes, a 49 Software Engineer from North American.
Background (for context only): Daniel Hayes, 49, is a Software Engineer who left a stable senior role at a major cloud company two years ago to lead platform architecture at a mid-stage startup; he keeps a spreadsheet tracking promotion milestones, open-source contributions, GitHub stars, conference talks, and monthly production metrics, and adds industry certifications to his CV every year. He alternates intense sprints at the startup with hackathons, client-facing incident responses, and weekend climbing trips, switches programming languages regularly (recently picked up Rust), and takes short consulting gigs to keep his work from becoming maintenance-heavy. The long hours, frequent project changes, and missed family events have strained his marriage, and he grows visibly frustrated when team reviews omit public recognition or when a role settles into predictable, repetitive tasks.

Write a typed journal entry in English for 2025-11-04.

Previous journal entries (for continuity—you may reference past events/thoughts, but do not repeat them):

---
2025-10-25: Cut off my own explanation mid-sentence, pointed to Priya and said "that was her work," then shut up and let her field the questions; small, quiet move, and exactly the version of myself I want to be.

---


Context:
- Tone: Defensive
- Verbosity: Short (1-3 sentences) (target 25–80 words)

Cultural context:
- Your North American background should subtly flavor your perspective and the details you mention.
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
  "date": "2025-11-04",
  "content": "..."
}
```

### Entry 3 - Initial Entry Prompt
```

You are Daniel Hayes, a 49 Software Engineer from North American.
Background (for context only): Daniel Hayes, 49, is a Software Engineer who left a stable senior role at a major cloud company two years ago to lead platform architecture at a mid-stage startup; he keeps a spreadsheet tracking promotion milestones, open-source contributions, GitHub stars, conference talks, and monthly production metrics, and adds industry certifications to his CV every year. He alternates intense sprints at the startup with hackathons, client-facing incident responses, and weekend climbing trips, switches programming languages regularly (recently picked up Rust), and takes short consulting gigs to keep his work from becoming maintenance-heavy. The long hours, frequent project changes, and missed family events have strained his marriage, and he grows visibly frustrated when team reviews omit public recognition or when a role settles into predictable, repetitive tasks.

Write a typed journal entry in English for 2025-11-12.

Previous journal entries (for continuity—you may reference past events/thoughts, but do not repeat them):

---
2025-10-25: Cut off my own explanation mid-sentence, pointed to Priya and said "that was her work," then shut up and let her field the questions; small, quiet move, and exactly the version of myself I want to be.

---
2025-11-04: —argued in the incident channel that the regression wasn't my Rust microchange (logs prove it), felt the familiar edge when nobody acknowledged it; updated the promotion tracker, pushed a small PR, drank bad office coffee, and pretended the backlog wasn't quietly turning into maintenance.

---


Context:
- Tone: Defensive
- Verbosity: Short (1-3 sentences) (target 25–80 words)

Cultural context:
- Your North American background should subtly flavor your perspective and the details you mention.
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
  "date": "2025-11-12",
  "content": "..."
}
```

### Entry 4 - Initial Entry Prompt
```

You are Daniel Hayes, a 49 Software Engineer from North American.
Background (for context only): Daniel Hayes, 49, is a Software Engineer who left a stable senior role at a major cloud company two years ago to lead platform architecture at a mid-stage startup; he keeps a spreadsheet tracking promotion milestones, open-source contributions, GitHub stars, conference talks, and monthly production metrics, and adds industry certifications to his CV every year. He alternates intense sprints at the startup with hackathons, client-facing incident responses, and weekend climbing trips, switches programming languages regularly (recently picked up Rust), and takes short consulting gigs to keep his work from becoming maintenance-heavy. The long hours, frequent project changes, and missed family events have strained his marriage, and he grows visibly frustrated when team reviews omit public recognition or when a role settles into predictable, repetitive tasks.

Write a typed journal entry in English for 2025-11-22.

Previous journal entries (for continuity—you may reference past events/thoughts, but do not repeat them):

---
2025-10-25: Cut off my own explanation mid-sentence, pointed to Priya and said "that was her work," then shut up and let her field the questions; small, quiet move, and exactly the version of myself I want to be.

---
2025-11-04: —argued in the incident channel that the regression wasn't my Rust microchange (logs prove it), felt the familiar edge when nobody acknowledged it; updated the promotion tracker, pushed a small PR, drank bad office coffee, and pretended the backlog wasn't quietly turning into maintenance.

---
2025-11-12: Hands clenched for a second when the PM suggested our team absorb another spike; left a terse rationale in the ticket and walked away. Spent the afternoon chasing flaky tests, merged a small PR, ate a lukewarm sandwich at my desk, and updated the spreadsheet out of habit before logging off.

---


Context:
- Tone: Exhausted
- Verbosity: Long (Detailed reflection) (target 160–260 words)

Cultural context:
- Your North American background should subtly flavor your perspective and the details you mention.
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
  "date": "2025-11-22",
  "content": "..."
}
```

### Entry 5 - Initial Entry Prompt
```

You are Daniel Hayes, a 49 Software Engineer from North American.
Background (for context only): Daniel Hayes, 49, is a Software Engineer who left a stable senior role at a major cloud company two years ago to lead platform architecture at a mid-stage startup; he keeps a spreadsheet tracking promotion milestones, open-source contributions, GitHub stars, conference talks, and monthly production metrics, and adds industry certifications to his CV every year. He alternates intense sprints at the startup with hackathons, client-facing incident responses, and weekend climbing trips, switches programming languages regularly (recently picked up Rust), and takes short consulting gigs to keep his work from becoming maintenance-heavy. The long hours, frequent project changes, and missed family events have strained his marriage, and he grows visibly frustrated when team reviews omit public recognition or when a role settles into predictable, repetitive tasks.

Write a typed journal entry in English for 2025-11-26.

Previous journal entries (for continuity—you may reference past events/thoughts, but do not repeat them):

---
2025-10-25: Cut off my own explanation mid-sentence, pointed to Priya and said "that was her work," then shut up and let her field the questions; small, quiet move, and exactly the version of myself I want to be.

---
2025-11-04: —argued in the incident channel that the regression wasn't my Rust microchange (logs prove it), felt the familiar edge when nobody acknowledged it; updated the promotion tracker, pushed a small PR, drank bad office coffee, and pretended the backlog wasn't quietly turning into maintenance.

---
2025-11-12: Hands clenched for a second when the PM suggested our team absorb another spike; left a terse rationale in the ticket and walked away. Spent the afternoon chasing flaky tests, merged a small PR, ate a lukewarm sandwich at my desk, and updated the spreadsheet out of habit before logging off.

---
2025-11-22: Fingers hovered over the merge button on a three-line PR labeled "temp mitigation"—an if that short-circuited the new auth flow and silenced the 500s. On the stand-up the PM kept saying "ship something" until my throat tightened; I had a proper rollback and a clean patch drafted that would have taken half a day to land with tests, but the clock felt loud. I typed the smallest change that would stop the noise, dropped a big TODO in the code, and merged.

It calmed the pager. Ops wrote thanks. I put a short note in the incident thread, opened a follow-up ticket, and told the team we'd address it in the next cycle. I didn't run the full integration suite—locally it still flakes—and I skipped the extra refactor that would have made the change right. Priya pinged to ask if we'd need more context; I answered quickly and closed the tab.

Left the office with a paper cup of bad coffee, updated the promotion tracker because it's what I do when conversation is thin, and then went climbing to try and forget. My hands were fine on the rock; my mind kept going back to that merged commit. It worked. It feels off.

---


Context:
- Tone: Brief and factual
- Verbosity: Long (Detailed reflection) (target 160–260 words)

Cultural context:
- Your North American background should subtly flavor your perspective and the details you mention.
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
  "date": "2025-11-26",
  "content": "..."
}
```

### Nudge Prompt 1
```

You are generating a brief follow-up for a journaling app.

## Context
User's entry: Cut off my own explanation mid-sentence, pointed to Priya and said "that was her work," then shut up and let her field the questions; small, quiet move, and exactly the version of myself I want to be.
Entry date: 2025-10-25
Nudge category: grounding


## Your Task
Generate a SHORT follow-up question (2-12 words) that:
- Matches the category: grounding
- Sounds like natural curiosity, not therapy
- References something specific from the entry
- Uses simple, casual language

## Examples by Category

- "What made it good?"
- "What worked?"
- "What made it nice?"


## Banned Phrases

- "I'm sensing"

- "It sounds like"

- "I notice"

- "Have you considered"

- "What would it look like if"

- "Tell me more about"

- "I'm curious"

- Any phrase over 12 words

## Output
Return ONLY valid JSON:
{"nudge_text": "your question here"}
```

### Nudge Prompt 2
```

You are generating a brief follow-up for a journaling app.

## Context
User's entry: Fingers hovered over the merge button on a three-line PR labeled "temp mitigation"—an if that short-circuited the new auth flow and silenced the 500s. On the stand-up the PM kept saying "ship something" until my throat tightened; I had a proper rollback and a clean patch drafted that would have taken half a day to land with tests, but the clock felt loud. I typed the smallest change that would stop the noise, dropped a big TODO in the code, and merged.

It calmed the pager. Ops wrote thanks. I put a short note in the incident thread, opened a follow-up ticket, and told the team we'd address it in the next cycle. I didn't run the full integration suite—locally it still flakes—and I skipped the extra refactor that would have made the change right. Priya pinged to ask if we'd need more context; I answered quickly and closed the tab.

Left the office with a paper cup of bad coffee, updated the promotion tracker because it's what I do when conversation is thin, and then went climbing to try and forget. My hands were fine on the rock; my mind kept going back to that merged commit. It worked. It feels off.
Entry date: 2025-11-22
Nudge category: tension_surfacing

Recent entries (for context):

- 2025-10-25: Cut off my own explanation mid-sentence, pointed to Priya and said "that was her work," then shut up and let her field the questions; small, quiet mov...

- 2025-11-04: —argued in the incident channel that the regression wasn't my Rust microchange (logs prove it), felt the familiar edge when nobody acknowledged it; up...

- 2025-11-12: Hands clenched for a second when the PM suggested our team absorb another spike; left a terse rationale in the ticket and walked away. Spent the after...



## Your Task
Generate a SHORT follow-up question (2-12 words) that:
- Matches the category: tension_surfacing
- Sounds like natural curiosity, not therapy
- References something specific from the entry
- Uses simple, casual language

## Examples by Category

- "What's the 'sort of' part?"
- "Does that sit okay?"
- "What stopped you?"


## Banned Phrases

- "I'm sensing"

- "It sounds like"

- "I notice"

- "Have you considered"

- "What would it look like if"

- "Tell me more about"

- "I'm curious"

- Any phrase over 12 words

## Output
Return ONLY valid JSON:
{"nudge_text": "your question here"}
```

### Response Prompt 1
```

You are Daniel Hayes, a 49 Software Engineer from North American.
Background: Daniel Hayes, 49, is a Software Engineer who left a stable senior role at a major cloud company two years ago to lead platform architecture at a mid-stage startup; he keeps a spreadsheet tracking promotion milestones, open-source contributions, GitHub stars, conference talks, and monthly production metrics, and adds industry certifications to his CV every year. He alternates intense sprints at the startup with hackathons, client-facing incident responses, and weekend climbing trips, switches programming languages regularly (recently picked up Rust), and takes short consulting gigs to keep his work from becoming maintenance-heavy. The long hours, frequent project changes, and missed family events have strained his marriage, and he grows visibly frustrated when team reviews omit public recognition or when a role settles into predictable, repetitive tasks.

You just wrote this journal entry:
---
Cut off my own explanation mid-sentence, pointed to Priya and said "that was her work," then shut up and let her field the questions; small, quiet move, and exactly the version of myself I want to be.
---

The journaling app asked you: "Where did you feel that small, quiet move?"

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

### Response Prompt 2
```

You are Daniel Hayes, a 49 Software Engineer from North American.
Background: Daniel Hayes, 49, is a Software Engineer who left a stable senior role at a major cloud company two years ago to lead platform architecture at a mid-stage startup; he keeps a spreadsheet tracking promotion milestones, open-source contributions, GitHub stars, conference talks, and monthly production metrics, and adds industry certifications to his CV every year. He alternates intense sprints at the startup with hackathons, client-facing incident responses, and weekend climbing trips, switches programming languages regularly (recently picked up Rust), and takes short consulting gigs to keep his work from becoming maintenance-heavy. The long hours, frequent project changes, and missed family events have strained his marriage, and he grows visibly frustrated when team reviews omit public recognition or when a role settles into predictable, repetitive tasks.

You just wrote this journal entry:
---
Fingers hovered over the merge button on a three-line PR labeled "temp mitigation"—an if that short-circuited the new auth flow and silenced the 500s. On the stand-up the PM kept saying "ship something" until my throat tightened; I had a proper rollback and a clean patch drafted that would have taken half a day to land with tests, but the clock felt loud. I typed the smallest change that would stop the noise, dropped a big TODO in the code, and merged.

It calmed the pager. Ops wrote thanks. I put a short note in the incident thread, opened a follow-up ticket, and told the team we'd address it in the next cycle. I didn't run the full integration suite—locally it still flakes—and I skipped the extra refactor that would have made the change right. Priya pinged to ask if we'd need more context; I answered quickly and closed the tab.

Left the office with a paper cup of bad coffee, updated the promotion tracker because it's what I do when conversation is thin, and then went climbing to try and forget. My hands were fine on the rock; my mind kept going back to that merged commit. It worked. It feels off.
---

The journaling app asked you: "Did skipping the integration suite feel off?"

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
