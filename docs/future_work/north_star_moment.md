# North Star Moment

## Future-Work Design Brief

**Status:** Proposed stretch goal; not implemented

**Recommended scope:** A bounded, fail-closed addition to Experience and Inspect

**Primary trigger:** Active Drift for one Core Value

**Primary user-facing result:** One exact quotation from an earlier Journal Entry in which the user described behaviour supporting that Core Value

**Tracker:** `twinkl-b8w3` documents this design brief only

## 1. Decision Summary

North Star Moment is a proposed extension to the Coach Digest experience. When Weekly Drift Detection reports Active Drift for a Core Value, Twinkl searches the user's earlier Journal Entries for a concrete moment in which the user described behaviour supporting that same Core Value. A separate bounded AI review verifies the earlier Journal Entry. If the Journal Entry passes every eligibility and grounding check, Experience displays its date and an exact quotation in a small card beneath the Coach Digest.

The proposed card does not advise the user what to do. It does not change the Profile, Core Values, Weekly Drift Reviewer Decisions, Drift Detector result, Active Drift, or any Historical Drift Record. It does not claim improvement, recovery, progress, or success. It reminds the user, in their own words, that their present Conflict is not the only evidence in their history.

The idea is intentionally narrower than a habit recommender, action planner, or automatically evolving Profile. It adds the missing directional element in the original Twinkl vision while retaining the project's current evidence-grounded and non-prescriptive design.

The recommended implementation keeps North Star Moment outside the generated Coach Digest response for the first version. Experience renders the verified quotation deterministically beneath the existing Coach Digest, while Inspect exposes how the quotation was retrieved and verified. This boundary avoids changing the Coach Digest's current three-field response and makes the new evidence independently inspectable.

## 2. Product Context

Twinkl's original proposition is to help users notice whether their lived behaviour remains connected to the Core Values they have confirmed. The current capstone proof of concept implements most of that proposition:

1. Onboarding creates a confirmed Profile and Core Values.
2. The user writes chronological Journal Entries.
3. Weekly Drift Detection compares closed-week Journal Entries with those Core Values.
4. The Weekly Drift Reviewer decides Conflict, Not Conflict, or Abstain for each relevant Journal Entry and Core Value.
5. The Drift Detector creates Drift only when two consecutive Conflicts concern the same Core Value.
6. The Coach Digest cites the relevant Journal Entries, explains the current finding, and asks a non-prescriptive reflective question.
7. Inspect exposes the Profile, Journal Entries, Weekly Drift Reviewer Decisions, Drift Detector transitions, Coach Digest response, validations, and provider receipts.

This implementation creates an inspectable path from a confirmed Profile to an Active Drift alert. It is deliberately cautious about what happens next. The Coach Digest avoids advice, action plans, micro-habits, generic praise, and unsupported claims that a user has improved. That design protects the user from an AI-generated prescription, but it also leaves the original product loop incomplete: Twinkl identifies distance from a Core Value without showing any evidence of what movement toward that Core Value has previously looked like for this person.

North Star Moment addresses that gap by using the user's own earlier writing as the directional reference.

## 3. What the Idea Is

### 3.1 Core concept

A North Star Moment is one earlier Journal Entry that meets all of the following conditions:

- It belongs to the same user or Persona as the current Weekly Drift Detection result.
- It was written before the first Conflict in the current Active Drift.
- It describes observable behaviour, a choice, or an action that supports the affected Core Value.
- Its relevance can be established without generation instructions, LLM-Judge VIF Labels, VIF Critic Predictions, biography-only claims, or other information unavailable to the user-facing application.
- The displayed quotation is an exact substring of the earlier Journal Entry.
- A separate North Star Moment review accepts the Journal Entry under a fixed structured contract.
- Every deterministic validation passes.

If no earlier Journal Entry meets these conditions, Twinkl returns no North Star Moment. Absence is an expected result, not an error to be filled with a generic quotation or invented suggestion.

### 3.2 Intended user experience

For an Active Drift result, Experience continues to show the existing Weekly Drift Detection finding and Coach Digest. When a verified earlier Journal Entry exists, Experience adds a compact card with:

- a heading such as **A past moment in your own words**;
- the user-facing Core Value phrase, never an exposed internal Schwartz label;
- the date of the earlier Journal Entry;
- one exact quotation from that Journal Entry;
- an action that opens the complete Journal Entry or focuses its linked Inspect events;
- a short evidence boundary such as: **This earlier entry is a reference point, not proof that the current Drift has ended.**

The first version should not ask a second question. The existing Coach Digest reflective question remains the single open question presented for the week. This avoids competing calls to reflection and keeps North Star Moment as evidence rather than a recommendation.

### 3.3 Why "verified" matters

Semantic similarity alone is insufficient. A Journal Entry can mention fairness, freedom, family, achievement, or security without describing behaviour that supports the corresponding Core Value. It may discuss someone else's behaviour, quote a hypothetical action, express an intention without acting, or contain a Conflict against the same Core Value.

The proposal therefore separates retrieval from verification:

1. **Retrieval** finds a small number of earlier Journal Entries that are semantically related to the Core Value.
2. **North Star Moment review** determines whether one retrieved Journal Entry contains explicit supportive behaviour and identifies the exact quotation.
3. **Deterministic validation** checks identity, chronology, exact quotation, allowed fields, and absence of future information.
4. **Selection** chooses at most one verified Journal Entry by a fixed rule.

This staged design uses semantic retrieval for breadth and a narrow structured decision for precision.

## 4. Gap Filled in the Current Proposition

### 4.1 The missing fourth leg

The original Twinkl vision can be described as four connected legs:

1. establish the user's North Star through a confirmed Profile and Core Values;
2. notice behaviour that conflicts with a Core Value;
3. detect repeated Conflict and alert the user through an evidence-grounded Coach Digest;
4. help the user reorient toward the Core Value.

The capstone proof of concept implements the first three legs. The fourth remains intentionally limited to one reflective question. The current Coach Digest can identify the tension and invite reflection, but it cannot show what the affected Core Value has looked like when expressed by this user.

North Star Moment fills that gap without moving into prescriptive coaching. It changes Twinkl from only saying **this is where your recent behaviour appears distant from your Core Value** to also saying **this is an earlier moment, in your own words, when your behaviour expressed that Core Value**.

### 4.2 Current and proposed behaviour

| Product responsibility | Current capstone behaviour | Proposed addition |
|---|---|---|
| Establish the reference | Onboarding creates a confirmed Profile and Core Values. | No change. |
| Review behaviour | The Weekly Drift Reviewer decides Conflict, Not Conflict, or Abstain. | No change. |
| Decide Drift | The Drift Detector applies the two-consecutive-Conflict rule. | No change. |
| Explain the result | The Coach Digest cites the current evidence and asks one reflective question. | No change to the first-version Coach Digest response. |
| Offer direction | The Coach Digest does not provide advice or a behavioural plan. | Experience may show one earlier, verified Journal Entry supporting the affected Core Value. |
| Preserve contestability | Inspect shows the evidence and stored decisions behind the result. | Inspect also shows retrieval eligibility, North Star Moment review, exact-quotation validation, and selection. |
| Handle weak evidence | The application can return Insufficient Evidence or omit an invalid Coach Digest. | North Star Moment independently fails closed to no card. Weekly Drift Detection and the Coach Digest remain available. |

### 4.3 Why North Star Moment is preferable to generic advice

A generic recommendation asks an AI model to invent an action that might suit the user. That introduces several difficult questions: whether the advice is practical, whether it respects constraints not present in the Journal Entries, whether it feels intrusive, and whether it changes behaviour. Those questions require user testing that is not feasible in the remaining capstone period.

North Star Moment makes a narrower and more testable claim. It says only that Twinkl found an earlier Journal Entry in which the user described behaviour that supports the same Core Value. The evidence already exists in the user's history, the displayed text is traceable, and incorrect retrieval can be measured on a frozen synthetic benchmark.

## 5. Required Behaviour and Boundaries

### 5.1 Trigger conditions

The first version should attempt North Star Moment retrieval only when all of the following are true:

- Weekly Drift Detection completed successfully for a closed week.
- At least one Core Value has Active Drift.
- The Active Drift has a known onset and supporting Journal Entries.
- At least one earlier Journal Entry exists before that onset.
- The relevant Profile and Core Value are still the confirmed reference used by Weekly Drift Detection.

No North Star Moment should be requested for No Active Drift or Insufficient Evidence. Those states do not require a counterpoint to a confirmed Active Drift, and extending the first version to them would add policy branches without strengthening the central proposition.

### 5.2 Multiple Active Drifts

If more than one Core Value has Active Drift, the first version should return at most one North Star Moment for each affected Core Value but display no more than one North Star Moment card by default. A deterministic precedence rule should choose which card appears first. Possible rules include:

1. the Active Drift with the longest current Conflict run;
2. the Active Drift with the most recent second Conflict;
3. the first affected Core Value in the confirmed Profile order.

The implementation context should choose one rule before evaluation and record it in the versioned contract. It should not let the generation model choose which Core Value receives attention.

### 5.3 Chronology boundary

The North Star Moment must precede the onset of the current Active Drift, not merely the current closed week. This prevents Twinkl from retrieving:

- one of the Journal Entries that formed the current Drift;
- a later Journal Entry unavailable at the selected Persona replay point;
- a future Journal Entry accidentally exposed by saved scenario data;
- a Journal Entry written after the pattern began and then misrepresented as an earlier reference point.

The chronology check must use stored dates and `t_index` ordering. When those disagree, the North Star Moment must be omitted and Inspect should show the validation failure.

### 5.4 Evidence boundary

A North Star Moment may describe one earlier supportive choice. It must not be treated as evidence that:

- the current Active Drift has ended;
- the user has recovered or improved;
- the earlier behaviour is typical of the user;
- the Profile is correct or permanent;
- the earlier action is appropriate in the current context;
- repeating the earlier action will produce a desired outcome.

These limits should appear in prompt instructions, deterministic validations, Coach Digest Evals, Inspect, and the Technical Paper.

### 5.5 Separation from existing authority

The proposal must preserve the following ownership boundaries:

- The **Weekly Drift Reviewer** continues to decide only Conflict, Not Conflict, or Abstain for current-week Journal Entries and Core Values.
- The **Drift Detector** continues to decide whether two consecutive Conflicts form Drift and to maintain Historical Drift Records.
- The **Coach Digest** continues to use stored Weekly Drift Detection output and does not decide whether Drift exists.
- The proposed North Star Moment review decides only whether an earlier Journal Entry is suitable to display as a supportive reference for one Core Value.
- The proposed North Star Moment never becomes a Weekly Drift Reviewer Decision and never changes a Drift Detector result.
- The **VIF Critic (Offline)** remains an offline research component and does not receive user-facing authority through this feature.

### 5.6 Explicit non-goals

The first version does not include:

- advice, action plans, implementation intentions, habits, or recommendations;
- a user-authored next-step form;
- automatic Profile evolution or a request to change Core Values;
- a new current Drift state such as `returned` or `recovered`;
- a `supports` verdict added to the Weekly Drift Reviewer;
- a claim that Not Conflict demonstrates supportive behaviour;
- external quotations, books, videos, or an inspiration feed;
- cross-user retrieval;
- VIF Critic Predictions as retrieval or selection authority;
- model training or fine-tuning;
- user testing, usability testing, or a behavioural outcome study;
- production multi-user storage or background scheduling.

## 6. Before-and-After Examples

The following examples are illustrative. They are not current saved Persona responses and must not be reported as evaluated application evidence.

### 6.1 Example A: a verified earlier Journal Entry exists

#### Before

Weekly Drift Detection reports Active Drift for the Core Value expressed to the user as **making the world a fairer, better place**. Two consecutive Journal Entries include the exact phrases:

> "I said okay."

and

> "I nodded and let the meeting move on."

The current Coach Digest might say:

> Across these two meetings, you noticed something felt unfair and let both moments pass without saying so. What felt at stake in speaking differently?

The response identifies the Conflict and asks an appropriate question, but it offers no personal reference point for the Core Value.

#### After

The Weekly Drift Detection result and Coach Digest remain unchanged. Beneath the Coach Digest, Experience adds:

> **A past moment in your own words**
>
> 14 May 2026
>
> "I stayed after the review because the intern had not been given a fair chance to explain."
>
> This earlier Journal Entry is a reference point for your Core Value. It does not mean the current Drift has ended.

The user can open the complete earlier Journal Entry. Inspect shows that it predates the current Drift, was retrieved for the same Core Value, passed North Star Moment review, and passed exact-quotation validation.

### 6.2 Example B: only a semantic near-match exists

#### Before

The Coach Digest explains Active Drift and asks one reflective question.

#### Proposed retrieval result

An earlier Journal Entry says:

> "The promotion process here is never fair."

The entry contains semantically related language, but it does not describe the user's supportive behaviour or choice. Retrieval may rank it highly, but North Star Moment review must reject it.

#### After

Experience shows the same Weekly Drift Detection result and Coach Digest as before. No North Star Moment card appears. Inspect records that no eligible earlier Journal Entry was verified.

This is correct fail-closed behaviour. A missing North Star Moment is safer than an impressive but false personal connection.

### 6.3 Example C: an earlier supportive action concerns another value

#### Before

Active Drift concerns the Core Value expressed as **having the freedom to choose my own path**.

An earlier Journal Entry says:

> "I covered the shift because my teammate had nobody else to ask."

That action may support being dependable toward others, but it is not necessarily evidence of choosing one's own path.

#### After

The North Star Moment review rejects the Journal Entry for the affected Core Value. Experience displays no North Star Moment unless another earlier Journal Entry passes. Inspect shows the Core Value mismatch without exposing an internal Schwartz label in user-facing text.

### 6.4 Example D: a future Journal Entry would be an excellent North Star Moment

#### Before

A saved Persona contains a later Journal Entry that clearly describes the relevant supportive action, but the current replay point occurs before that entry.

#### After

The no-future-data rule excludes the later Journal Entry before semantic retrieval. It cannot appear in Experience, the North Star Moment review request, logs intended to represent the current replay point, or Inspect. North Star Moment is absent until the later Journal Entry becomes chronologically available.

### 6.5 Example E: No Active Drift or Insufficient Evidence

#### Before

Experience displays No Active Drift or Insufficient Evidence and the corresponding Coach Digest policy.

#### After

The first version behaves exactly as it did before. No North Star Moment retrieval runs and no North Star Moment card appears. This keeps the stretch goal focused on the missing guidance step after confirmed Active Drift.

## 7. Relationship to the Academic Requirements

The [Capstone Requirements](../capstone_report/capstone_requirements.pdf) describe Phase 2 as development, coding, testing, validation, implementation, and demonstration. The Phase 2 assessment assigns 20% each to the Technical Paper, Presentation, and System Implementation & Demo, plus 10% to Sponsor/Panel Assessment. The assessment guide distinguishes working application evidence from research evidence and includes complexity, innovativeness, verification and validation, functionality, methodology, and programming among its considerations. It also expects the Technical Paper and reports to show clear writing, substantial depth, technical achievement, and appropriate references.

North Star Moment is attractive as a capstone stretch goal because it contributes to several of these dimensions without requiring the project to make a behavioural-outcome claim.

### 7.1 Intelligent System objective

The feature adds a bounded reasoning capability to an existing Intelligent System. It does not merely add decorative interface content. It must select historical evidence under semantic, chronological, identity, and policy constraints; request a structured AI decision; validate the returned quotation; and make the result inspectable.

### 7.2 Application-oriented assessment

| Assessment consideration | Contribution of the proposed feature | Evidence that can be produced without users |
|---|---|---|
| Complexity | Integrates retrieval, structured AI review, deterministic validation, versioned contracts, Experience, Inspect, saved Persona replay, and failure handling. | Architecture description, code, tests, stored receipts, and a complete professor walkthrough. |
| Innovativeness | Uses the user's own earlier behaviour as the directional reference instead of generic advice or an external quotation. | Before-and-after demonstration across multiple saved Personas and benchmark categories. |
| Verification and validation | Separates retrieval quality, North Star Moment review, exact quotation, chronology, abstention, and presentation. | Frozen synthetic benchmark, deterministic checks, provider-separated AI review, regression tests, and error analysis. |
| Functionality | Adds a visible response to the missing guide-back step after Active Drift. | End-to-end Persona replay showing an accepted North Star Moment, no-card result, provider failure, and no-future-data enforcement. |
| Demonstration | Produces an immediately understandable change in Experience while retaining full traceability in Inspect. | Scripted phone-width and desktop-width walkthrough with linked evidence. |
| Customer satisfaction or feedback | The feature may be meaningful to users, but the capstone cannot establish that without user testing. | No claim should be made. This assessment consideration remains outside the feature's evidence. |

### 7.3 Research-oriented assessment

The feature can also support a small R&D investigation:

> Can semantic retrieval followed by a bounded, evidence-grounded AI review identify an earlier Journal Entry that demonstrates behaviour supporting the same Core Value involved in a current Active Drift?

That question is narrow enough to evaluate without human participants. It creates a clear methodological comparison between retrieval alone and retrieval followed by verification. The contribution is not a claim that North Star Moments change behaviour. It is a measured account of whether Twinkl can select and faithfully present a valid earlier reference from longitudinal Journal Entries.

### 7.4 Practice-module alignment

| Practice module | Proposed contribution |
|---|---|
| Intelligent Reasoning Systems | Applies semantic retrieval, a bounded North Star Moment decision, deterministic eligibility rules, and fail-closed selection to connect an Active Drift with earlier supportive evidence. |
| Pattern Recognition Systems | Uses a frozen text encoder to rank earlier Journal Entries by relevance to one Core Value. Retrieval quality is measured separately from the North Star Moment review. |
| Intelligent Sensing Systems | Treats chronological Journal Entries as a longitudinal behavioural signal while enforcing the exact information available at each weekly cutoff. The feature remains text-only. |
| Architecting AI Systems | Adds a versioned request, receipt, trace events, deterministic validations, Experience card, Inspect detail, saved Persona fixtures, and safe failure without changing existing Drift authority. |

### 7.5 Why assessors may value it

The proposal creates a coherent extension of Twinkl's central argument rather than adding an unrelated late feature. It reuses the confirmed Profile, longitudinal Journal Entries, Active Drift, evidence-grounded explanations, abstention, and Inspect. The demonstration can therefore show both product value and technical discipline in one short sequence.

Its claim is also falsifiable. Twinkl either retrieves an eligible earlier Journal Entry and quotes it faithfully, or it does not. Incorrect North Star Moments, chronology violations, wrong-Core-Value retrieval, and unsupported quotations can all be counted and inspected. That is more academically defensible than demonstrating a generated recommendation and assuming it helped.

## 8. Complexity and Time Estimate

### 8.1 Overall estimate

The recommended first version is **medium complexity**. A realistic implementation and evaluation estimate is **7–10 working days**, assuming the current Experience and Inspect contracts remain stable and no new paid-model procurement is required.

This estimate is larger than a prompt-only change because the feature requires a new evidence contract and an evaluation benchmark. It is smaller than Return Detection or Profile evolution because it does not alter Weekly Drift Reviewer Decisions, the Drift Detector, current Drift states, or Historical Drift Records.

### 8.2 Complexity by work area

| Work area | Expected complexity | Reason |
|---|---|---|
| Eligibility and chronology | Low | Existing Journal Entry identifiers, dates, `t_index` values, Active Drift onset, and replay cutoffs already exist. The main work is consolidating them into one tested rule. |
| Semantic retrieval | Low to medium | Existing encoder code can be reused, but the Embedding Explorer's reduced three-dimensional coordinates must not be used as retrieval vectors. The implementation needs full-dimensional embeddings and a small cache or on-demand computation path. |
| North Star Moment review | Medium | Requires a new prompt, structured response, provider boundary, fail-closed parsing, receipts, and task-specific examples. It must remain separate from the Weekly Drift Reviewer. |
| Deterministic validation | Medium | Must check exact quotation, chronology, identity, Core Value coordinates, requested identifiers, response completeness, and failure behavior. |
| Experience | Low to medium | One optional mobile-first card, a link to the complete Journal Entry, loading and unavailable behavior, keyboard access, and narrow-screen tests. |
| Inspect | Medium | Needs trace events and a readable explanation of retrieval, North Star Moment review, selection, and validation. Hidden provider reasoning must not be exposed. |
| Saved Persona replay | Medium | Requires at least one accepted-moment fixture, one no-card fixture, and no-future-data verification without invalidating current scenario provenance. |
| Evaluation | Medium to high | The most important work is creating a frozen task-specific benchmark and preserving the distinction between LLM-Judge VIF Labels and North Star Moment reference decisions. |
| Technical Paper update | Medium | Requires a method, results, error analysis, limitations, architecture update, and claim boundaries. A feature description alone is insufficient. |

### 8.3 Suggested time allocation

| Stage | Indicative effort | Deliverable |
|---|---:|---|
| Feasibility and benchmark design | 1–2 days | Fixed North Star Moment definition, benchmark categories, baseline retrieval result, and go/no-go decision. |
| Contracts, retrieval, and validation | 2–3 days | Versioned request and receipt, full-dimensional retrieval, North Star Moment review, deterministic validations, and focused tests. |
| Experience, Inspect, and Persona replay | 2–3 days | Optional North Star Moment card, trace view, fixtures, no-future-data behavior, accessibility checks, and integration tests. |
| Evaluation and Technical Paper evidence | 2–3 days | Frozen report, metrics, error analysis, cost and latency, limitations, and report-ready figures or tables. |

The implementation context should resist combining this work with Return Prompt, Return Detection, automatic Profile evolution, or longitudinal Core Value history. Each addition weakens the one-month scope boundary.

## 9. High-Level Implementation Design

### 9.1 Recommended flow

```text
Confirmed Profile and Journal Entries
              |
              v
Existing Weekly Drift Detection
              |
              +---- No Active Drift or Insufficient Evidence
              |             |
              |             +---- No North Star Moment work
              |
              +---- Active Drift for a Core Value
                            |
                            v
              Deterministic earlier-entry filter
                            |
                            v
              Full-dimensional semantic retrieval
                            |
                            v
              Structured North Star Moment review
                            |
                            v
              Exact-quotation and chronology checks
                            |
                   +--------+--------+
                   |                 |
                 Pass              No pass
                   |                 |
                   v                 v
       North Star Moment card          No card
                   |
                   v
          Linked Inspect evidence
```

### 9.2 Step 1: derive the retrieval request

After Weekly Drift Detection is stored, the orchestration code derives one request for each eligible Active Drift. The request should contain only information needed for the North Star Moment task:

- session or Persona identifier;
- reviewed week and replay cutoff;
- affected Core Value;
- user-facing Core Value phrase;
- current Active Drift onset and supporting Journal Entry identifiers;
- earlier Journal Entries eligible at the cutoff;
- prompt name, version, and hash;
- model and reasoning-effort request;
- schema version and creation time.

Current Conflict evidence may be supplied to help distinguish a meaningful contrast, but the North Star Moment decision must remain focused on whether the earlier Journal Entry itself describes supportive behaviour. The prompt should not infer that the same action would be appropriate now.

### 9.3 Step 2: filter the history before retrieval

Deterministic code should remove ineligible Journal Entries before any embedding or provider call. The filter should enforce:

- exact user or Persona match;
- `t_index` lower than the first Conflict in the current Active Drift;
- date no later than the Active Drift onset date;
- availability at the current Persona replay point;
- non-empty displayed Journal Entry text;
- absence from the current Active Drift's supporting Journal Entries;
- no use of removed Journal Entries;
- no hidden generation, labelling, biography, or VIF Critic data.

Filtering first reduces cost and makes future-data leakage mechanically impossible at the model boundary.

### 9.4 Step 3: rank earlier Journal Entries

The retrieval implementation should encode:

1. a query derived from the affected Core Value's internal rubric and user-facing phrase; and
2. each eligible earlier Journal Entry.

It should use full-dimensional embeddings from one fixed encoder. The three-dimensional PCA or t-SNE coordinates in `viz/embedding_explorer.html` are suitable for visualization, not semantic retrieval.

Cosine similarity can rank the eligible Journal Entries. The first version should send only a small fixed number, such as the top three or top five, to North Star Moment review. The exact number is an evaluation parameter and must be fixed before the reported benchmark run.

Retrieval similarity remains internal. Experience must not show a relevance number, rank, or confidence indicator. A high similarity does not establish that the Journal Entry is supportive.

### 9.5 Step 4: perform North Star Moment review

The North Star Moment review should use a new prompt and response schema. It must not reuse the name, receipt schema, or authority of the Weekly Drift Reviewer.

One possible internal response contract is:

```json
{
  "entry_id": "journal-entry-id",
  "decision": "supportive | not_supportive | abstain",
  "evidence_quote": "exact substring or empty",
  "reason_code": "observable_choice | other_value | intent_only | hypothetical | other_person | ambiguous | insufficient_text"
}
```

The exact identifiers may change during implementation. The semantic contract should remain:

- `supportive` means the earlier Journal Entry describes an observable choice or behaviour supporting the requested Core Value;
- `not_supportive` means the available text does not meet that definition;
- `abstain` means the text is too ambiguous to decide without hidden context;
- an accepted result requires a non-empty exact quotation;
- rejected or Abstain results require an empty quotation;
- the model cannot change the requested Journal Entry or Core Value coordinate;
- malformed, refused, missing, or provider-error responses fail closed.

The North Star Moment prompt should include hard cases involving intentions, emotions, someone else's actions, hypotheticals, value trade-offs, and supportive behaviour for another value.

### 9.6 Step 5: validate and select

Deterministic validation should confirm:

- the returned Journal Entry was requested;
- the returned Core Value is the requested Core Value;
- the Journal Entry precedes the current Active Drift;
- the Journal Entry is available at the current cutoff;
- the quotation is an exact substring of the stored Journal Entry;
- an accepted result has a quotation and a permitted reason code;
- a rejected, Abstain, or unavailable result has no quotation;
- user-facing text contains no raw internal Schwartz label;
- no field claims recovery, progress, success, or an end to Active Drift.

If several Journal Entries pass, selection should be deterministic. A reasonable first rule is:

1. preserve semantic-retrieval order;
2. select the first verified Journal Entry;
3. use recency only as a fixed tie-breaker;
4. display at most one Journal Entry for each attempted North Star Moment card.

The report should disclose this rule. It should not describe the selected Journal Entry as the user's best or strongest example unless the evaluation establishes that interpretation.

### 9.7 Step 6: persist a versioned receipt

The receipt should preserve enough information to reproduce and inspect the decision without storing hidden provider reasoning. Suggested fields include:

| Field | Purpose |
|---|---|
| `schema_version` | Supports compatibility-safe evolution. |
| `session_id` or `persona_id` | Links the North Star Moment to the correct history. |
| `reviewed_week` | Links the North Star Moment to one Weekly Drift Detection result. |
| `core_value` | Stores the internal coordinate used by the application. |
| `user_facing_core_value_phrase` | Stores the phrase permitted in Experience. |
| `active_drift_start_t_index` | Makes chronology validation inspectable. |
| `eligible_entry_ids` | Records the deterministic pre-retrieval filter result. |
| `retrieved_entry_ids` | Records the fixed top-k retrieval order. |
| `selected_entry_id` | Identifies the displayed Journal Entry, when present. |
| `evidence_quote` | Stores the exact displayed quotation. |
| `decision` and `reason_code` | Records the bounded North Star Moment review. |
| `validation` | Records exact quotation, chronology, identity, Core Value, and response checks. |
| `prompt_name`, `prompt_version`, and hashes | Supports reproducibility. |
| requested and actual model fields | Supports provider and model provenance. |
| usage, latency, and status | Supports evaluation and operational reporting. |

Similarity values may be recorded for offline evaluation but should not appear in Experience. If retained, they must be documented as retrieval diagnostics rather than certainty.

### 9.8 Step 7: add Experience behavior

Experience should render North Star Moment as a progressive enhancement:

- Weekly Drift Detection remains visible if North Star Moment retrieval fails.
- The Coach Digest remains visible if North Star Moment retrieval fails.
- The North Star Moment card appears only after every required check passes.
- Loading North Star Moment must not block the existing weekly result indefinitely.
- Retry behavior must be explicit and idempotent.
- The complete Journal Entry should be reachable without losing the current week.
- The card must work at the project's primary narrow-screen width.
- Keyboard focus, screen-reader labels, quotation semantics, and reduced-motion behavior should follow the existing Experience conventions.

The first version should render stored Persona North Star Moments without live provider calls. Optional live rerun should remain a separate action if it is implemented at all.

### 9.9 Step 8: add Inspect behavior

Inspect should let an assessor trace:

1. which Active Drift triggered retrieval;
2. which earlier Journal Entries were eligible;
3. which Journal Entries semantic retrieval selected for review;
4. which prompt and model contract ran;
5. which North Star Moment decision was returned;
6. which exact quotation was validated;
7. why no North Star Moment appeared when retrieval or review failed;
8. whether the result came from a saved Persona fixture or a live call.

Inspect should not expose chain-of-thought, hidden provider reasoning, API secrets, or generation metadata. A concise reason code and evidence quotation are sufficient.

### 9.10 Likely repository touchpoints

The implementation context should confirm the current tree before editing. Likely areas include:

- `src/demo/contracts.py` for the versioned Experience and Inspect contract;
- `src/demo/experience_service.py` for orchestration and persistence;
- `src/coach/` for shared provider-boundary and validation patterns, without giving the Coach Digest new Drift authority;
- a new focused module for North Star Moment retrieval, review, and its receipt;
- `frontend/onboarding/src/experienceApi.ts` and the corresponding contract fixtures;
- the Experience component that renders Weekly Drift Detection and Coach Digest results;
- the Inspect components and fixture builders;
- `src/demo/scenarios.py` and saved Persona JSON files for deterministic replay;
- focused Python, React, contract, replay, accessibility, and no-future-data tests;
- an evaluation script and timestamped report under `logs/experiments/reports/`;
- the PRD and Technical Paper only after implementation and evaluation are complete.

The implementation must inspect current contracts rather than treating this list as exhaustive or current by definition.

## 10. Evaluation Without User Testing

### 10.1 Evaluation objective

The evaluation should answer:

> Given an Active Drift for one Core Value and only the Journal Entries available before its onset, how reliably can Twinkl retrieve, verify, and faithfully display one earlier Journal Entry that describes behaviour supporting that Core Value?

It should not answer:

- whether users like North Star Moment;
- whether North Star Moment changes subsequent behaviour;
- whether North Star Moment increases journaling;
- whether North Star Moment improves wellbeing;
- whether the user agrees with the selected Core Value;
- whether the earlier action should be repeated.

Those are user or behavioural outcome questions and remain outside this stretch goal.

### 10.2 Why existing LLM-Judge VIF Labels are insufficient by themselves

The existing corpus contains 16,510 LLM-Judge VIF Labels across 1,651 Journal Entries and ten value dimensions. A `+1` LLM-Judge VIF Label can help find possible supportive Journal Entries, but it was created to train or evaluate the VIF Critic (Offline). It is not a North Star Moment reference decision and does not use the exact runtime task, evidence contract, or chronology conditions proposed here.

The existing LLM-Judge VIF Labels should therefore serve only as one source for benchmark construction or retrieval analysis. They must not silently become user-facing authority. The implementation should create a new, task-specific set of AI-reviewed synthetic reference decisions under the exact North Star Moment definition.

### 10.3 Frozen benchmark design

The benchmark should contain complete history coordinates rather than isolated positive examples. Each benchmark case should include:

- Persona identifier;
- confirmed Core Values;
- reviewed week;
- current Active Drift and onset;
- all Journal Entries available before onset;
- one of the benchmark categories below;
- task-specific AI-reviewed reference decisions;
- provenance for every reference decision and adjudication;
- a frozen manifest with hashes.

Required categories are:

| Category | Expected behavior |
|---|---|
| Clear same-Core-Value support | Retrieve and verify the eligible earlier Journal Entry. |
| Semantic mention without supportive behaviour | Reject or abstain. |
| Intention or emotion without observable action | Reject or abstain. |
| Another person's supportive action | Reject. |
| Support for a different value | Reject for the requested Core Value. |
| Conflict against the requested Core Value | Reject. |
| Mixed or ambiguous behaviour | Abstain unless the supportive choice is explicit. |
| Valid support after Active Drift onset | Exclude before retrieval. |
| Valid support after the replay cutoff | Exclude before retrieval and from provider input. |
| Multiple valid earlier Journal Entries | Select one by the fixed deterministic rule. |
| No valid earlier Journal Entry | Return no North Star Moment. |
| Provider refusal, invalid JSON, or timeout | Return no North Star Moment and preserve the existing weekly result. |

The benchmark should include hard Core Values and ambiguous short Journal Entries rather than only obvious success cases. Cases used to design prompts or selection rules must remain separate from the final reported benchmark cases.

### 10.4 Reference-decision process

Because human review and user testing are excluded, every benchmark conclusion must be labelled as AI-reviewed synthetic reference evidence. A defensible process would:

1. define the North Star Moment task and examples before requesting reference decisions;
2. use a model or provider separated from the runtime North Star Moment reviewer where feasible;
3. request `supportive`, `not_supportive`, or `abstain` plus an exact quotation for supportive decisions;
4. use a second AI review for disagreements or high-risk categories;
5. preserve prompts, model snapshots, reasoning effort, hashes, costs, and timestamps;
6. freeze the final manifest before evaluating the implementation;
7. disclose that provider separation does not create human validation.

### 10.5 Primary and secondary metrics

The primary metric should be **North Star Moment precision**: among displayed North Star Moments, the proportion accepted by the frozen task-specific reference decisions for the requested Core Value.

Precision should outrank coverage because an incorrect personal quotation is more harmful than showing no North Star Moment.

Secondary metrics should include:

| Metric | Meaning |
|---|---|
| North Star Moment coverage | Proportion of eligible Active Drift cases for which Twinkl displays a verified North Star Moment. |
| Incorrect-selection count and rate | Displayed North Star Moments that the frozen reference rejects or marks Abstain. |
| Correct no-card rate | Histories with no valid North Star Moment in which Twinkl correctly omits the card. |
| Chronology violation count | North Star Moments at or after Active Drift onset or beyond the replay cutoff. The required result is zero. |
| Exact-quotation failure count | Displayed text that is not an exact substring of the stored Journal Entry. The required result is zero. |
| Wrong-user count | North Star Moments retrieved from another user or Persona. The required result is zero. |
| Wrong-Core-Value rate | North Star Moments that support another value but not the affected Core Value. |
| Abstention rate | Proportion of reviewed Journal Entries for which the North Star Moment review abstains. |
| Retrieval recall at k | Proportion of cases with a valid reference North Star Moment where at least one valid Journal Entry appears in the fixed top-k retrieval result. |
| Verification lift | Difference between retrieval-only precision and precision after North Star Moment review. |
| Provider failure rate | Refused, invalid, timed-out, or error responses. |
| Latency and calculated model cost | Operational evidence for one North Star Moment attempt and the full benchmark. |

Metric thresholds should be declared before the final benchmark run. The implementation context should not choose thresholds after inspecting final results.

### 10.6 Mechanical validations

The feature should add validations for:

- exact quotation;
- same user or Persona;
- same requested Core Value;
- strict pre-onset chronology;
- replay cutoff;
- selected Journal Entry membership in the reviewed set;
- complete version and provenance fields;
- valid decision and reason-code combinations;
- absence of a North Star Moment on provider failure;
- raw internal Schwartz label leakage in user-facing fields;
- prohibited claims of improvement, recovery, progress, success, or an ended Active Drift;
- length and rendering limits for the card.

These checks should be named separately from Coach Digest Validations unless they are formally added to that contract. The report must not imply that existing Coach Digest Validations already cover the new card.

### 10.7 Coach Digest Evals

If the North Star Moment card is evaluated together with the Coach Digest, the existing Coach Digest Eval dimensions can be extended or accompanied by North Star Moment review criteria:

- correctness of the relationship between Active Drift and the earlier Journal Entry;
- specificity of the displayed reference;
- non-prescriptive tone;
- tension honesty;
- absence of recovery or success claims;
- whether the existing reflective question remains open-ended when read beside North Star Moment.

These results remain AI review, not human validation or evidence of user benefit.

### 10.8 Regression and integration tests

At minimum, automated tests should cover:

- trigger only for Active Drift;
- no trigger for No Active Drift and Insufficient Evidence;
- same-user and same-Core-Value filtering;
- strict pre-onset chronology;
- no-future-data Persona replay;
- semantic retrieval ordering under fixed embeddings;
- accepted, rejected, and Abstain North Star Moment decisions;
- malformed, refused, missing, and provider-error responses;
- exact-quotation rejection;
- idempotent retry and session resume;
- Journal Entry removal and affected-week recomputation;
- no mutation of Weekly Drift Reviewer Decisions or Drift Detector results;
- no mutation of the Profile or Core Values;
- Experience rendering at narrow and wide widths;
- keyboard and screen-reader access;
- Inspect linkage and source disclosure;
- saved Persona hash and manifest behavior;
- migration from sessions created before the North Star Moment fields existed;
- all existing Weekly Drift Detection and Coach Digest tests.

### 10.9 Permitted and prohibited final claims

Permitted claim:

> On a frozen AI-reviewed synthetic benchmark, the proposed retrieval and verification workflow selected earlier Journal Entries that met the North Star Moment definition at the reported precision and coverage, with zero detected chronology and exact-quotation failures.

Permitted implementation claim:

> Saved Persona replay demonstrates that Experience can display a verified earlier Journal Entry for Active Drift and that Inspect can trace its retrieval, review, validation, and source.

Prohibited claims without users:

> North Star Moment helps users return to their Core Values.

> North Star Moment improves decision-making or wellbeing.

> Users find North Star Moment accurate, motivating, or non-judgmental.

> North Star Moment changes later Journal Entries or reduces future Drift.

The product rationale may describe the intended guide-back role, but results and conclusions must retain these evidence boundaries.

## 11. Recommended Implementation Sequence

### Phase 0: feasibility gate

Before changing Experience or Inspect:

1. Freeze the North Star Moment definition and non-goals.
2. Build a small development benchmark containing every required category.
3. Run a retrieval-only baseline using the intended full-dimensional encoder.
4. Test one North Star Moment structured prompt against the hard cases.
5. Measure retrieval recall at k and North Star Moment precision.
6. Decide whether the evidence supports continuing.

If semantic retrieval rarely contains a valid Journal Entry or the North Star Moment review accepts too many near-misses, stop before interface work and adopt the fallback in Section 13.

### Phase 1: contracts and core behavior

1. Create a focused Beads feature with explicit acceptance criteria.
2. Define the versioned request, response, validation, and receipt contracts.
3. Implement deterministic eligibility and chronology filtering.
4. Implement full-dimensional embedding retrieval with a fixed encoder.
5. Implement the North Star Moment provider call and fail-closed parser.
6. Implement deterministic validation and selection.
7. Add focused unit and contract tests.

### Phase 2: Experience, Inspect, and replay

1. Extend the shared session contract compatibly.
2. Add the optional North Star Moment card beneath the Coach Digest.
3. Link the card to the complete Journal Entry and Inspect.
4. Add Inspect events for trigger, filtering, retrieval, review, validation, and selection.
5. Add saved Persona fixtures for accepted, absent, and failed North Star Moments.
6. Verify session resume, retry, recomputation, removal, migration, and no-future-data behavior.
7. Run narrow-screen, keyboard, screen-reader, and reduced-motion checks.

### Phase 3: frozen evaluation

1. Freeze the final benchmark manifest before evaluation.
2. Run deterministic validations.
3. Run the selected provider-separated AI review.
4. Report every metric, failure, exclusion, cost, latency, prompt version, model, reasoning effort, and hash.
5. Perform error analysis by Core Value and benchmark category.
6. Preserve failed responses and validation diagnostics where privacy permits.

### Phase 4: capstone integration

1. Add one accepted-moment and one no-card path to the professor walkthrough.
2. Update the PRD implementation status.
3. Update architecture and evaluation documentation.
4. Add the method, results, limitations, and before-and-after figure to the Technical Paper.
5. State clearly that the evidence is AI-reviewed synthetic evaluation without user testing, a fresh final test, or deployment approval.

## 12. Risks and Mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| Semantic similarity is mistaken for supportive behaviour. | Twinkl shows an irrelevant or false personal reference. | Separate retrieval from North Star Moment review; make precision primary; fail closed. |
| Existing `+1` LLM-Judge VIF Labels are treated as direct North Star Moment reference decisions. | The evaluation inherits a mismatched task definition and overstates validity. | Use them only to help construct cases; create task-specific AI-reviewed synthetic reference decisions. |
| A later Journal Entry leaks into an earlier Persona replay point. | The demonstration violates chronological integrity. | Filter before embedding and provider calls; require zero chronology violations. |
| The quotation is paraphrased or invented. | The most visible evidence is unfaithful. | Require an exact substring and reject the entire North Star Moment on mismatch. |
| An earlier supportive action is presented as proof of recovery. | The card overstates the user's current state. | Keep Active Drift unchanged; add explicit language rules and claim validations. |
| The selected Journal Entry supports another value. | The North Star Moment appears personally resonant but is semantically wrong. | Include wrong-value hard cases and report wrong-Core-Value rate. |
| North Star Moment becomes implicit advice. | The user may feel instructed to repeat an action that does not fit the current context. | Display evidence only; add no new action prompt in the first version. |
| The feature changes the frozen Weekly Drift Reviewer. | Existing evaluation and report claims become stale. | Create a separate North Star Moment review with no Drift authority. |
| The new call fails or adds noticeable latency. | The weekly experience becomes fragile. | Run after the weekly result, preserve existing content, store failure status, and allow safe retry. |
| North Star Moment coverage is low. | Few Active Drift cases receive the proposed card. | Treat coverage as secondary; report it honestly; use the fallback if the feature is not demonstrable. |
| Same-provider evaluation rewards the runtime prompt's style. | AI review appears stronger than it is. | Use provider separation where feasible and disclose that this remains AI review. |
| The feature expands into a recommendation product. | Scope, safety obligations, and evaluation needs exceed the capstone window. | Preserve the explicit non-goals and create separate future work for any action-planning feature. |
| A stale Profile makes North Star Moment irrelevant. | The feature reinforces a Core Value the user no longer endorses. | Use only the confirmed Profile that governed the current Weekly Drift Detection result; do not claim the Profile is permanent. Profile evolution remains separate future work. |

## 13. Fallback: Historical Drift Contrast

If the North Star Moment review does not reach a convincing precision level, the project should not weaken the definition or hide failures to preserve the feature. The recommended fallback is a narrower Historical Drift contrast.

The fallback would:

- use only already stored Weekly Drift Detection output and Historical Drift Records;
- identify an earlier comparable period for the same Core Value;
- cite the exact earlier Conflict evidence and the later Not Conflict decision that ended that Conflict run;
- say only that the earlier repeated pattern did not continue in the same way;
- avoid describing the later Journal Entry as supportive behaviour, improvement, success, or recovery;
- require no new supportive-behaviour decision.

This fallback is less directional than North Star Moment, but it remains truthful and easier to evaluate. It can still help the user see that an Active Drift is a time-bounded pattern rather than a permanent identity claim.

## 14. Open Decisions for the Implementation Context

The following questions should be resolved before implementation begins:

1. Should the first version render North Star Moment only as a separate Experience card, or also supply it to the Coach Digest prompt? The recommended answer is a separate card.
2. Which fixed encoder should produce full-dimensional retrieval embeddings?
3. Should the retrieval query use only the user-facing Core Value phrase or combine it with the internal rubric?
4. What top-k retrieval value gives acceptable recall without unnecessary provider input?
5. Should North Star Moment review examine retrieved Journal Entries in one batch or one at a time?
6. What exact task-specific reference process will resolve ambiguous benchmark cases without human review?
7. Which deterministic rule chooses among multiple verified Journal Entries?
8. Which Active Drift receives the visible card when several Core Values have Active Drift?
9. Should live manual Experience compute embeddings on demand or maintain a versioned per-session embedding cache?
10. How should session migration represent historical weeks that predate the North Star Moment contract?
11. Which saved Personas provide the clearest accepted-moment, no-card, and future-leak demonstrations?
12. What precision, coverage, and provider-failure thresholds will be declared before the final benchmark run?
13. What maximum latency and calculated cost are acceptable for one weekly North Star Moment attempt?
14. Is the feature still worthwhile if high precision produces low coverage?

## 15. Definition of Done for a Future Implementation

The feature is complete only when:

- the current PRD and a focused Beads feature record the adopted scope;
- Weekly Drift Reviewer and Drift Detector contracts remain unchanged;
- the versioned North Star Moment request, response, receipt, and validation contracts are implemented;
- retrieval uses full-dimensional embeddings and excludes ineligible Journal Entries before any provider call;
- invalid, refused, missing, and provider-error responses fail closed to no North Star Moment;
- Experience renders at most one selected North Star Moment card according to the adopted policy;
- Inspect exposes the complete permitted evidence path;
- saved Persona replay demonstrates accepted, absent, failed, and no-future-data cases;
- contract, Python, React, migration, replay, accessibility, and no-future-data tests pass;
- the frozen benchmark and final evaluation report are committed with exact provenance;
- exact-quotation, chronology, and wrong-user failures are zero in the final reported run;
- retrieval and North Star Moment review metrics are reported separately;
- the Technical Paper states that results are AI-reviewed synthetic evidence, not human validation, user testing, a fresh final test, or deployment approval;
- the implementation context reads all changed files, runs proportionate quality checks, inspects the final diff and working-tree status, and records remaining risks.

## 16. Related Documentation

- [Product Requirements Document](../prd.md)
- [Canonical Nouns and Communication Rules](../canonical_nouns.md)
- [Capstone Requirements](../capstone_report/capstone_requirements.pdf)
- [Technical Paper source](../capstone_report/capstone_project_report.md)
- [Experience and Inspect design](../demo/experience_inspect_app.md)
- [Coach Digest explanation quality](../evals/explanation_quality_eval.md)
- [VIF Critic (Offline) concepts and roadmap](../vif/01_concepts_and_roadmap.md)
- [Value evolution concept note](../evolution/01_value_evolution.md)
- [Habit recommendation future work](habit_recommendations.md)
