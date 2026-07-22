# Experience and Inspect React Demo

## Status

This document specifies the capstone demo experience. The React demo is not yet
implemented. Its versioned React-Python boundary, JSON Schema, and canonical
fixtures are implemented in [`src/demo/contracts.py`](../../src/demo/contracts.py)
and [`frontend/onboarding/src/contracts/`](../../frontend/onboarding/src/contracts/).
The existing React onboarding implementation and the
[Onboarding Specification](../onboarding/onboarding_spec.md) remain
authoritative for the Schwartz Values Best-Worst Survey (SVBWS), Profile, and
Core Value contracts.

## 1. Purpose

The demo presents the product experience and the AI architecture from the same
session. A persistent two-option control switches between:

- **Experience** — the user-facing journey through onboarding, Journal Entries,
  displayed nudges and responses, Drift, and the Weekly Digest.
- **Inspect** — the developer-facing explanation of the exact backend work that
  produced the currently selected result.

The two views are not separate demonstrations. They read the same Profile,
Journal Entries, Weekly Drift Reviewer Decisions, Drift state, Weekly Digest,
and run trace. Switching views must preserve the current session, selected
week, selected Journal Entry, and selected backend event.

This design lets a professor assess both user value and Architecting AI Systems
work without waiting for a real week of journaling.

## 2. Product and Evidence Boundaries

- The React implementation in `frontend/onboarding/` remains the single
  onboarding implementation. Do not port it to Shiny or duplicate its SVBWS
  scoring in Python.
- The user-facing Drift path is fixed: Journal Entries and Core Values go to
  the `gpt-5.6-luna` reasoning-effort-`low` Weekly Drift Reviewer without VIF
  Critic input, then the Drift Detector applies the two-consecutive-Conflict
  rule.
- The VIF Critic remains offline research. Inspect may link to separate
  research reports, but it must not imply that VIF Critic Predictions produce
  user-facing Drift.
- Persona replay uses saved Weekly Drift Reviewer Decisions by default. It must
  identify their model contract, input hash, run provenance, and whether the
  source was replayed or generated live.
- AI-reviewed synthetic development evidence is not human validation or
  deployment approval. The source of every displayed decision must remain
  explicit in Inspect.

## 3. View Availability

| Session stage | Experience | Inspect |
|---|---|---|
| Active SVBWS card selection | Enabled | Disabled with “Available after Profile confirmation” |
| Goal and Core Value confirmation | Enabled | Disabled |
| Confirmed Profile handoff | Enabled | Enabled; shows the Profile handoff and validation |
| Journal Entry draft or nudge check | Enabled | Enabled; follows the active Journal Entry event |
| Weekly review or Weekly Coach work | Enabled | Enabled; follows the active run |
| Persona replay | Enabled | Enabled; follows the selected saved run |

The disabled state must explain why it is unavailable. It must not look like a
broken control.

## 4. Shared Session Model

One client-side session store owns:

- the confirmed Profile and Core Values;
- ordered Journal Entries, including displayed nudges and responses;
- the selected persona scenario, week, Journal Entry, and trace event;
- Weekly Drift Reviewer Decisions grouped by calendar week;
- the current Drift Detector result;
- the Weekly Digest and optional Weekly Coach reflection;
- run status and retry state; and
- references to backend trace events.

Changing views changes presentation only. It must not repeat a model call,
reset onboarding, alter replay progress, or create a second copy of the
session.

Every generated nudge, Weekly Drift Reviewer Decision, Drift state, and Weekly
Digest has an **Inspect this run** action. It switches to Inspect and focuses
the event that produced the selected result. Returning to Experience restores
the same screen position and selection where practical.

## 5. Experience View

### 5.1 Manual onboarding

Preserve the complete React onboarding flow:

1. 11 randomized SVBWS groups with one Most and one Least choice;
2. structured goal selection;
3. label-free Core Value confirmation; and
4. first Journal Entry handoff.

Do not add backend telemetry to the SVBWS card screens. Profile JSON, raw
scores, Schwartz labels, and developer terminology remain hidden in
Experience.

### 5.2 Manual journaling

After Profile confirmation, Experience provides:

- an ordered Journal Entry composer;
- clear saving, nudge-checking, reviewing, complete, and failed states;
- one contextual nudge with reply and skip actions when the nudge decision
  requests it;
- the anti-annoyance rule of no more than two displayed nudges in the previous
  three Journal Entries;
- a chronological journal timeline containing the user's words plus any
  displayed nudge and response;
- safe Journal Entry removal for the POC, followed by automatic recomputation;
  and
- a contextual retry action only after a failed backend operation.

A Journal Entry must be held safely while the nudge check runs. A missing key,
refusal, invalid response, or request failure must not discard the Journal
Entry.

### 5.3 Persona simulation

Experience offers a curated **Try a demo persona** shortcut. Selecting a
persona loads its Profile, Core Values, Journal Entries, displayed nudges and
responses, saved Weekly Drift Reviewer Decisions, Drift states, and Weekly
Digests.

Persona simulation is a week-by-week replay rather than an immediate dump of
the final state. Controls provide:

- previous week;
- next week;
- play or pause; and
- restart scenario.

Advancing a week reveals only the Journal Entries and results available by that
week. This preserves the temporal meaning of Drift and lets the professor see
the user experience change from stable to active, recovered, uncertain, or
mixed.

Saved replay is the default because it is fast, deterministic, and free of
provider availability. A separate, clearly labelled **Re-run live** action may
exist in Inspect. Live results must never silently replace the saved reference
run.

### 5.4 User-facing results

Experience shows:

- the user's own Journal Entries, displayed nudges, and responses;
- an ambient per-Core-Value Drift state;
- the Weekly Digest; and
- the optional Weekly Coach reflection and question.

Experience does not show raw Weekly Drift Reviewer prompts, validation
details, run records, or per-entry `Conflict` badges. Those belong in Inspect.
When Experience explains Drift, it cites the relevant Journal Entries without
exposing internal reasoning text.

## 6. Inspect View

### 6.1 Information hierarchy

Inspect opens on a readable event timeline, not a telemetry dump. The first
level answers:

1. What happened?
2. What component did it?
3. What result did it produce?
4. How long did it take?
5. Was it replayed, generated live, reused, refused, invalid, or failed?

Detailed inputs, prompts, responses, and validation expand on demand.

### 6.2 Trace event types

Inspect represents these events when applicable:

1. `profile_confirmed`
   - Profile validation, Core Values, goal category, and Profile provenance.
2. `journal_entry_submitted`
   - Journal Entry date, text reference, ordering validation, and session ID.
3. `nudge_suppression_checked`
   - previous-three-entry window and whether the anti-annoyance rule suppressed
     a nudge.
4. `nudge_decided`
   - sanitized inputs, exact prompt, model, category, reason, response,
     validation, and latency.
5. `nudge_generated`
   - exact prompt, generated question, word-count validation, attempts, and
     latency.
6. `weekly_review_requested`
   - week boundaries, cumulative displayed Journal Entry history, Core Values,
     prompt, fixed model contract, and input hash.
7. `weekly_review_completed`
   - raw provider response, validation result, effective Weekly Drift Reviewer
     Decisions, response ID when available, attempts, and latency.
8. `drift_detected`
   - the ordered Weekly Drift Reviewer Decisions considered, the deterministic
     rule steps, and resulting Drift state.
9. `weekly_digest_built`
   - structured Weekly Digest fields, cited Journal Entries, and source Drift
     state.
10. `weekly_coach_generated`
    - exact prompt, model, response, narrative validation, and latency.

The Inspect copy must use the canonical component names above. In particular,
Weekly Drift Reviewer Decisions are not called predictions.

### 6.3 Required event fields

Each trace event contains:

| Field | Purpose |
|---|---|
| `event_id` | Stable identity within the session |
| `session_id` | Joins Experience and Inspect state |
| `parent_event_id` | Connects cause and effect |
| `event_type` | One of the event types above |
| `status` | queued, running, complete, reused, refused, invalid, or failed |
| `source` | saved replay or live run |
| `started_at` / `completed_at` | Timing and ordering |
| `duration_ms` | Latency shown in Inspect |
| `input_refs` | Profile, Journal Entry, week, or prior-event references |
| `model_contract` | Model and reasoning effort when a model is called |
| `prompt` | Exact rendered prompt when applicable |
| `raw_response` | Provider response before product transformation |
| `validation` | Schema and content validation result |
| `result_refs` | Resulting nudge, decisions, Drift, or Weekly Digest |
| `input_hash` | Idempotency and replay identity |
| `error` | Safe error class and message without secrets |

Prompt reuse or a cache hit may be recorded when implemented, but caching is
not required by this UI contract. Persistent, inspectable provenance is the
requirement.

## 7. Python Boundary

The React app calls a small Python HTTP API. The API framework is an
implementation detail; the request, response, and trace contracts are the
stable boundary.

The Python side owns:

- Profile validation and session creation;
- nudge decision and generation;
- weekly grouping and affected-week selection;
- Weekly Drift Reviewer calls and response validation;
- Drift Detector execution;
- Weekly Digest construction;
- optional Weekly Coach generation;
- idempotent retry behavior; and
- trace creation and retrieval.

The React side owns:

- onboarding interaction and local resumability;
- Experience and Inspect presentation;
- persona replay controls;
- optimistic but recoverable Journal Entry state;
- view selection and focused trace navigation; and
- accessible, responsive status and error presentation.

Provider keys and unredacted provider configuration stay on the Python side.

### 7.1 Version 1 contract

`experience-inspect-v1` defines four framework-neutral operations:

| Operation | Purpose |
|---|---|
| `create_session` | Validate a confirmed Profile and establish shared session state |
| `submit_journal_entry` | Append one ordered Journal Entry using an expected session revision |
| `load_scenario` | Load one deterministic saved persona scenario |
| `read_trace` | Retrieve typed trace events, optionally after a known event |

Python Pydantic models are the schema source. The checked-in JSON Schema and
canonical fixture are generated by
`uv run python -m src.demo.export_contract_schema`. React validates the same
fixture through `frontend/onboarding/src/demoContracts.ts`. The fixture covers
all ten event types and complete, reused, refused, invalid, and failed results.

The following rules are part of the contract rather than a chosen HTTP
framework:

- `create_session` and `submit_journal_entry` carry a 64-character input hash
  as an idempotency key. Repeating the same key and input returns the stored
  result with `reused`; reusing the key for different input returns a safe
  conflict error before any model call.
- `submit_journal_entry` carries `expected_revision`. Python rejects a stale
  revision, duplicate Journal Entry identifier, duplicate `t_index`, or
  non-chronological Journal Entry before nudge or weekly review work begins.
- Event order is represented by timestamps plus `parent_event_id`. Journal
  Entry order is represented by `t_index`; callers must not infer it from
  response array order alone.
- Provider secrets, authorization headers, and unredacted provider
  configuration never cross the boundary. Exact prompts and raw model
  responses may cross only after secret redaction. Errors expose a stable code,
  safe message, and retryable flag.
- Weekly review events require `gpt-5.6-luna` with reasoning effort `low` and
  contain Weekly Drift Reviewer Decisions. VIF Critic Predictions and their
  uncertainty fields are rejected by this contract.
- Saved replay and live results use the same payload shapes and differ through
  `source`. A saved result may use `reused`; caching remains optional.
- Version 1 is strict: unknown fields or incompatible values are rejected. Any
  field or semantic change requires a new contract version and explicit React
  and Python compatibility handling. Existing version 1 fixtures remain valid
  and immutable.

## 8. Review Orchestration

For one manually submitted Journal Entry, the observable sequence is:

```text
Journal Entry submitted
→ nudge suppression check
→ optional nudge decision and generation
→ optional user response or skip
→ Journal Entry finalized
→ affected calendar weeks selected
→ Weekly Drift Reviewer runs with cumulative displayed history
→ response validated into Weekly Drift Reviewer Decisions
→ Drift Detector applies the deterministic rule
→ Weekly Digest is built
→ optional Weekly Coach reflection is generated
```

The backend may reuse an unchanged weekly result by input hash. Reuse must be
visible in Inspect and must return the same saved decisions and provenance. A
cache is an optimization, not a user-facing feature or a capstone result by
itself.

## 9. Persona Scenario Bundles

The capstone demo requires at least these curated scenarios:

| Scenario | Required progression |
|---|---|
| Stable | No confirmed Drift across the replay |
| Active Drift | Two consecutive Conflicts for one Core Value |
| Recovered Drift | Active Drift followed by recovery |
| Uncertain | At least one effective Abstain affecting the displayed state |
| Two Core Values | Independent decisions and Drift state per Core Value |

Each saved scenario bundle contains or references:

- persona and Profile provenance;
- ordered Journal Entries, displayed nudges, and responses;
- calendar-week boundaries;
- rendered Weekly Drift Reviewer requests;
- raw responses and validation results;
- effective Weekly Drift Reviewer Decisions;
- Drift Detector results;
- Weekly Digests and optional Weekly Coach reflections;
- model contract, timestamps, response IDs when available, and input hashes;
  and
- a manifest version and content hash.

Scenario selection must be based on reviewed, reproducible behavior. Do not
rewrite Journal Entries or decisions merely to make the demonstration cleaner.
If a scenario is AI-reviewed synthetic development evidence, say so.

## 10. Privacy and Safety

- Inspect is a capstone and developer view, not a normal user destination.
- The default demo uses synthetic personas. If manual user text is inspected,
  it remains within the current local demo session unless persistence is
  explicitly enabled.
- Never display API keys, authorization headers, hidden environment values, or
  unrelated logs.
- Preserve the banned-term and value-leakage protections in generation and
  labeling work.
- Do not expose synthetic generation metadata to the Weekly Drift Reviewer,
  Drift Detector, Weekly Digest, or Weekly Coach.
- Raw provider responses are visible only in Inspect and must be clearly
  separated from validated product results.

## 11. Responsive and Accessible Behavior

- The Experience/Inspect selector remains reachable at the top of every
  post-onboarding screen.
- On narrow screens, each view occupies the full screen; do not force a
  side-by-side debugger.
- Persona replay controls remain operable with touch and keyboard input.
- Focus moves to the selected Inspect event when using **Inspect this run**.
- Status changes and nudge availability use appropriate live-region behavior.
- Long prompts and responses wrap, preserve whitespace, and expand without
  horizontal page scrolling.
- Reduced-motion preferences disable automatic replay animation while keeping
  explicit previous/next controls.

## 12. Non-Goals

- Reimplementing onboarding in Shiny.
- Making the current Shiny Runtime Demo Review App the mobile-first product.
- Adding VIF Critic Predictions to the Weekly Drift Reviewer or user-facing
  Drift path.
- Presenting LLM-Judge labels as production decisions.
- Claiming human validation, a fresh final test, or deployment approval.
- Building production authentication, multi-tenant storage, notifications, or
  native mobile packaging in the first capstone demo slice.
- Turning every backend log line into Inspect content.

## 13. Professor Demo Acceptance Walkthrough

A release is demo-ready when one uninterrupted walkthrough can:

1. complete or resume React onboarding and confirm Core Values;
2. submit a Journal Entry and observe a relevant nudge or a documented no-nudge
   decision;
3. switch to Inspect and view the exact nudge events without losing Experience
   state;
4. load an active-Drift persona scenario;
5. replay the persona week by week until the two consecutive Weekly Drift
   Reviewer Conflicts produce Drift;
6. inspect the exact weekly request, validated decisions, and deterministic
   Drift Detector steps;
7. return to Experience and read the corresponding Weekly Digest and Weekly
   Coach question;
8. demonstrate one recovered or uncertain scenario; and
9. distinguish saved replay from an optional live run.

## 14. Implementation Order

1. Define the Python API, shared session, scenario bundle, and trace contracts.
2. Build deterministic scenario bundles with provenance checks.
3. Extend the React app with the shared Experience/Inspect shell.
4. Implement manual Journal Entry and nudge behavior through the Python API.
5. Integrate weekly review, Drift, and Weekly Digest behavior.
6. Implement persona replay in Experience.
7. Implement event-linked Inspect timelines and details.
8. Add end-to-end, accessibility, responsive, failure, and replay tests.
9. Prepare the professor walkthrough and update capstone documentation.

Contract work blocks integration work. Scenario replay and live Journal Entry
work may proceed in parallel after the contracts exist. Inspect must consume
real trace events rather than reconstructing backend behavior in the browser.

## 15. Tracked Implementation Work

The parent Beads epic is `twinkl-rklc`. P0 items form the smallest complete
professor demo; P1 items add optional live execution or presentation material.

| Beads issue | Priority | Scope | Blocked by |
|---|---:|---|---|
| `twinkl-rklc.1` | P0 | API, session, scenario, and trace contracts | — |
| `twinkl-rklc.2` | P0 | Deterministic persona scenario bundles | `.1` |
| `twinkl-rklc.3` | P0 | Shared React Experience/Inspect shell | `.1` |
| `twinkl-rklc.4` | P0 | Experience journaling and nudges | `.1`, `.3` |
| `twinkl-rklc.5` | P0 | Weekly review, Drift, and Weekly Digest | `.1`, `.4` |
| `twinkl-rklc.6` | P0 | Week-by-week persona replay | `.2`, `.3`, `.5` |
| `twinkl-rklc.7` | P0 | Event-linked Inspect view | `.1`, `.3` |
| `twinkl-rklc.8` | P1 | Optional live rerun and visible reuse | `.5`, `.7` |
| `twinkl-rklc.9` | P0 | End-to-end demo quality gate | `.4`, `.5`, `.6`, `.7` |
| `twinkl-rklc.10` | P1 | Professor walkthrough and capstone evidence | `.9` |

The P0 quality gate intentionally does not depend on optional live reruns. A
saved, deterministic replay must remain sufficient for the complete demo.

## 16. Verification Requirements

- Unit tests protect onboarding contracts, client state transitions, trace
  serialization, affected-week selection, and deterministic replay.
- Contract tests verify React fixtures against Python request and response
  schemas.
- Integration tests cover successful, reused, refused, invalid, and failed
  model outcomes.
- End-to-end browser tests cover the professor walkthrough, reply and skip,
  Journal Entry removal, active and recovered Drift, and view-state
  preservation.
- Accessibility checks cover keyboard operation, focus, names, status updates,
  and reduced motion.
- Responsive checks cover representative narrow and desktop viewports.
- Saved scenario manifests are reproducible from their recorded inputs and
  reject mismatched hashes or model contracts.
