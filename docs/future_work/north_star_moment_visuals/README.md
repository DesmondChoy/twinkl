# North Star Moment team pitch images

Three concept visuals generated with the built-in image_gen tool on 5 September 2026. These are presentation illustrations; screen excerpts are simplified, and NSM is a proposed addition. The tool returned 1536 × 1024 PNGs (3:2), despite the requested 16:9 framing; preserve their aspect ratio when placing them in slides.

**Scope update:** The specification now covers all five saved Persona demos
and onboarding from scratch. These images predate that expansion: their
saved-replay-only captions and the third image's 7–11-day estimate are
historical. Use the current specification for scope, live execution, and
re-estimation. The before/after card concept still applies to both paths.

All three images were visually inspected for legibility, quotations, chronology, current/proposed status, and the separation between AI review and code checks. The first image received the branding correction and the user-facing wording correction recorded below. Its current version removes both Active Drift badges and the Drift disclaimer. Internal terms in the implementation explainers describe developer concepts, not copy shown to users.

## Suggested presentation order

1. **01-before-after.png** — Show what the user gains: a past supportive action beside the existing Coach Digest. The internal detection state and reflective question remain unchanged; no Drift terminology appears in the screen excerpts.
2. **02-selection-workflow.png** — Explain why a semantically related phrase is insufficient, and distinguish AI review from deterministic source checks.
3. **03-bounded-first-version.png** — Historical pitch for the narrower saved Persona replay version; its scope and estimate no longer describe the complete implementation.

## Sources and evidence boundaries

- [North Star Moment specification](../../north_star/north_star_moment.md), sections 1–7 and technical appendix.
- [Current PRD](../../prd.md), current product loop and product principles.
- [Canonical nouns](../../canonical_nouns.md).
- Current screen structure: [WeeklyExperience.tsx](../../../frontend/onboarding/src/WeeklyExperience.tsx).
- Palette and type direction: [styles.css](../../../frontend/onboarding/src/styles.css).
- Exact existing question and quoted passage: [saved Coach Digest responses](../../../src/demo/coach_digest_responses.json), `active-wei-jun::2025-06-30`.
- Earlier quotation: [Wei Jun saved Persona](../../../frontend/onboarding/public/scenarios/active-wei-jun.json), Journal Entry at `t_index=7`, 22 June 2025.

The examples come from a synthetic saved Persona. They illustrate the design and are not NSM evaluation results or evidence of user benefit. The 90% retrieval criterion is a feasibility gate; 7–11 working days was the earlier replay-only estimate. Evaluation-history selection remains open. The PRD now adopts the expanded scope; implementation remains outstanding.

## Exact generation prompts

Latest edit to image 1, using the previously corrected full image as the edit target:

```text
Use case: text-localization.
Input image: edit target, the existing complete Twinkl before/after presentation image.
Make only these two corrections:
1. Remove both apricot "Active Drift" badges entirely from the app headers. Fill their former areas seamlessly with the surrounding navy. Do not replace them with another badge, status, or phrase.
2. In the pale green North Star Moment card, replace the entire notice "This earlier writing is a reference point for your Core Value. It does not mean the current Drift has ended." with exactly "This earlier writing is a reference point for your Core Value." Keep the font readable and the Open Journal Entry link. Allow the notice to occupy two lines naturally.
Preserve all other content and layout: plain white Twinkl wordmarks, title, subtitle, before/after labels, both Core Value phrases, both exact Coach Digest quotations and reflective questions, the card title, 22 June 2025 date and source, exact quotation "Helped two new guys file their claims.", link, footer, palette, borders, decorative leaves and stars, dimensions 1536x1024. No Drift or drifting language anywhere in the resulting image. Do not invent copy, statuses, advice, or claims of improvement. Preserve the user's quoted wording exactly.
```

The original generation and earlier edit prompts below are retained as historical provenance. The latest edit above supersedes their instructions to display Drift wording.

Earlier branding edit to image 1 (the original generated image was supplied as the edit target):

```text
Use case: precise-object-edit. Edit only the two app mastheads in this image. Replace each white cloud-shaped commercial-looking twinkl logo with the plain text wordmark 'Twinkl' in a clean white Manrope-style sans-serif on the existing navy background. No cloud, no pink star, no other logo. Keep every other pixel, layout, dimension, color, all wording, both Active Drift pills, both questions, quotation, date, notice, footer, illustrations and card unchanged. This is an academic journaling project, not the education company. Preserve the image at its original resolution.
```

### 1

```text
Use case: infographic-diagram
Asset type: polished raster presentation image for Twinkl capstone teammates, landscape 16:9, high resolution.
Style: restrained editorial product storytelling. Twinkl's actual palette: paper #f7faf9, mist #e7edf2, navy #14223b, blue #5576d9, apricot #ff8a5b, verdigris #2e8c82. Elegant Source Serif-style headline and Manrope-style sans-serif body. Clear large type, generous whitespace, crisp card edges, quiet soft shadows, subtle hand-drawn editorial accents. Flat frontal layouts. No stock photos, no robot, no brain, no fake charts, no extra copy or watermarks. Text must be verbatim and readable at presentation size. Use color with restraint. Do not portray benefits as measured outcomes.
Primary request: Before/After product concept explaining exactly what North Star Moment adds. Make the personal quotation the memorable focal point. Two generous vertical app panels side by side, left current and right proposed, on a clean wide canvas. The right panel is taller or its contents fit with balanced hierarchy; do not crop text. These are simplified screen excerpts, not screenshots.
Top headline: "North Star Moment"
Subtitle: "A past action beside a present difficulty"
Left column title: "BEFORE · CURRENT EXPERIENCE"
Right column title: "AFTER · WITH NORTH STAR MOMENT"
Both app panels have the same Twinkl masthead, same apricot "Active Drift" pill, same Core Value phrase "Making the world a fairer, better place", and exactly the same Coach Digest excerpt:
Label: "Coach Digest · excerpt"
Quote: "I've been choosing convenience over doing what I know matters."
Then the identical reflective question in each panel: "When you notice yourself saying “okay” or nodding despite knowing what matters, what feels at stake in speaking or acting differently?"
Only the right app panel adds a highlighted pale verdigris card directly beneath that unchanged excerpt and question. Card has small star symbol, title "A past moment in your own words", phrase "Making the world a fairer, better place", date/source "22 June 2025 · From your Journal Entry", and large exact quotation:
"Helped two new guys file their claims."
Below the quote render exact notice, legible: "This earlier writing is a reference point for your Core Value. It does not mean the current Drift has ended."
A quiet link inside card: "Open Journal Entry"
Under the left panel a short caption: "Notice the pattern. Reflect on it."
Under the right panel a short caption: "Also recall a time you acted on this Core Value."
Small footer: "Illustrative screen excerpts · Wei Jun saved Persona · Proposed first version: saved Persona replay only"
Constraints: Both statuses remain Active Drift, same color. Do not depict a resolved state, replacement Coach Digest, second question, new advice, invented metrics, recovery arrows, or falsely claim current Twinkl only criticizes users. Omit additional UI navigation to keep copy readable.
```

### 2

```text
Use case: infographic-diagram
Asset type: polished raster presentation image for Twinkl capstone teammates, landscape 16:9, high resolution.
Style: restrained editorial product storytelling. Twinkl's actual palette: paper #f7faf9, mist #e7edf2, navy #14223b, blue #5576d9, apricot #ff8a5b, verdigris #2e8c82. Elegant Source Serif-style headline and Manrope-style sans-serif body. Clear large type, generous whitespace, crisp card edges, quiet soft shadows, subtle hand-drawn editorial accents. Flat frontal layouts. No stock photos, no robot, no brain, no fake charts, no extra copy or watermarks. Text must be verbatim and readable at presentation size. Use color with restraint. Do not portray benefits as measured outcomes.
Primary request: A visually engaging high-level implementation explainer showing that searching by meaning finds candidates, then AI review and code checks determine whether one exact quotation can be shown. Four numbered stations left to right across the main upper-middle canvas, with distinctive simple editorial illustrations of journal pages, meaning search, reviewed passage, and a source-linked quotation. Below, two large example cards compare accepted supportive action against rejected topic-only wording. Strong hierarchy, legible sparse labels.
Headline: "How a past moment earns its place"
Subtitle: "Find by meaning. Review the action. Check the source."
Small trigger ribbon above stations: "Closed week → Weekly Drift Detection → Active Drift"
Station 1 title: "1  Select eligible writing"
Body: "Same person. Same Core Value. Written before this Drift began."
Small source label: "Journal Entries + user nudge responses"
Station 2 title: "2  Search by meaning"
Body: "Find earlier writing related to the selected Core Value."
Station 3 title: "3  Review with AI"
Body: "Does it describe the user's supportive action, with no Conflict against that Core Value?"
Station 4 title: "4  Check and save"
Body: "Code checks the exact quote, identity and timing."
Output below station 4: "At most one card"
Make arrows flow between numbered stations. Keep flow clear without crossing lines.
Bottom left green-tinted card title: "SUPPORTIVE ACTION"
Exact quote: "Helped two new guys file their claims."
Explanation: "A concrete action supporting fairness."
Bottom right neutral/apricot card title: "RELATED WORDS ONLY"
Exact quote: "The promotion process here is never fair."
Explanation: "No supportive action described → no card."
Lower slim band with small source/inspection icon and exact text: "Inspect shows the sources, AI decisions and code checks."
Footer: "Proposed design · Results prepared offline for saved Persona replay · If no example passes, the Coach Digest remains available"
Constraints: Do not imply embeddings prove support, that code verifies semantic truth, or that AI-written nudges are user evidence. Do not show all history as eligible, future writing, external quotes, numerical confidence, or access to hidden model reasoning. The positive example illustrates the specification, not a completed benchmark.
```

### 3

```text
Use case: infographic-diagram
Asset type: polished raster presentation image for Twinkl capstone teammates, landscape 16:9, high resolution.
Style: restrained editorial product storytelling. Twinkl's actual palette: paper #f7faf9, mist #e7edf2, navy #14223b, blue #5576d9, apricot #ff8a5b, verdigris #2e8c82. Elegant Source Serif-style headline and Manrope-style sans-serif body. Clear large type, generous whitespace, crisp card edges, quiet soft shadows, subtle hand-drawn editorial accents. Flat frontal layouts. No stock photos, no robot, no brain, no fake charts, no extra copy or watermarks. Text must be verbatim and readable at presentation size. Use color with restraint. Do not portray benefits as measured outcomes.
Primary request: Persuasive practical scope and implementation visual for teammates deciding whether to explore North Star Moment. Show an existing product foundation, a compact optional addition, and three staged build steps. Make it feel achievable and concrete without selling a production launch or guaranteed outcomes. Use layered paper/card imagery and a small star on the added card; no rocket or growth chart.
Headline: "Start with one card in saved Persona replay"
Subtitle: "A visible product addition. A focused research question."
Upper half contains a wide visual composition: a solid foundation of two connected existing cards labelled "Weekly Drift Detection" and "Coach Digest", joined by an additive plus symbol to one highlighted card labelled "North Star Moment". In the highlighted card show the exact quotation "Helped two new guys file their claims." A small source tag "Earlier Journal Entry" below it. A branch from this card points to a compact "Inspect" panel with the words "Source → Review → Checks". Label over the existing pair "EXISTING EXPERIENCE"; label over the new card "OPTIONAL ADDITION".
Under this composition a prominent sentence: "Can we select a supportive past action and quote it faithfully?"
Lower half is three horizontal numbered steps with clear visual progression:
"1  Test retrieval locally"
"Compare 1, 3 and 5 results. Continue only at ≥90% proxy retrieval recall."
Small badge: "No paid calls"
"2  Build the saved replay"
"Prepare retrieval and AI review offline. Add the card, source link and Inspect record."
"3  Evaluate and demonstrate"
"Measure selection accuracy, faithful quotes, correct omissions, cost and failures."
Below the steps, a compact decision strip: "Before paid work: agree the budget and which histories to reserve for evaluation."
Bottom line in two concise parts: "Estimate: 7–11 working days" and "Assumes stable app contracts and an available evaluation provider"
Footer: "First version: saved Persona replay only · User benefit remains a question for future study"
Constraints: 90% is a target for proxy retrieval recall, never an achieved result, NSM accuracy score, or deployment threshold. Do not show manual/live availability, new model training, production storage, background scheduling, guaranteed wellbeing gains or completed feature status. Do not call the roadmap approved implementation. Make the estimate subordinate to the bounded scope.
```
