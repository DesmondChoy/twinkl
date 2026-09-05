# North Star Moment Phase 0B derived audit

Date: 2026-09-05. This audit examines the frozen development run in
[report.json](report.json), [manifest.json](manifest.json), and
[budget.json](budget.json). It makes no new model calls, revises no prompts or
reference decisions, and does not inspect the reserved evaluation histories.
The existing Coach Digest behaviour must remain unchanged because the Phase 0B
gate failed. Selected quotations below are offline candidate outputs, not cards
rendered in the Experience or validated with users.

## Provenance and deterministic replay

The report's manifest hash matches the current manifest. All three frozen input
hashes match their files: the evaluation policy, Phase 0A retrieval output, and
review contract. The report's provider and runner hashes also match the current
code. Audit-input SHA-256 values are:

| File | SHA-256 |
|---|---|
| `manifest.json` | `2c32a09c3ea32a8b45f3810bc9dcb4f7acf3ccd324ac2a9568f66b0c96e71e8d` |
| `report.json` | `5849cb816cc2a44ca39f7a1455d8cfb8f81fdbd4a519cbe947d76ad3c2bb08ec` |
| `budget.json` | `0f22beddc4713a7c2204c48db082cffe6b22f4b0998575312300c7cf0d753998` |
| `src/north_star/provider.py` | `03fc8135b377492d8342adb031703d3fd5086aaa480f281afca9b5eccd85e362` |
| `scripts/experiments/north_star_phase0b.py` | `2e0585a0720a1216d1dc71fbcfd693c7ec9867ee80e5235d6127175f1764edbf` |

Replaying the saved objects through `validate_review` reproduced 27 valid
runtime batches and 28 complete reference batches. Replaying `select_moment`
reproduced every selected identifier and quotation in the frozen retrieval
order. All 19 selected quotations passed the deterministic quotation, requested
Core Value, membership, and selection checks. Those checks establish textual
fidelity and structural correctness; they do not establish semantic support.
This replay did not exercise the browser, session lifecycle, source editing, or
the production chronology filter.

The budget contains 61 actual attempts: 29 OpenAI runtime attempts, 28 Gemini
exhaustive reference attempts, and four Gemini candidate-quotation checks.
Every attempt has calculated usage and cost. The ledger sum reproduces
US$0.20617785: US$0.01425435 for OpenAI and US$0.19192350 for Gemini. Reused
receipts in the case records do not add calls to this count. Recorded median
attempt latencies were 2.073 seconds for OpenAI and 2.271 seconds for Gemini;
these are offline provider-attempt timings, not live user wait times.

## Gate result and failure classification

There were 33 development episode cases from 27 synthetic Personas. Five cases
had no eligible earlier writing and correctly made no provider call. Of 19
selected candidates, 12 were accepted by the frozen AI reference protocol and
seven were rejected or received an abstention. Precision was therefore 12/19
(63.2%). Among nine histories with earlier writing but no reference-accepted
example, five omitted the candidate and four selected one: correct no-card rate
5/9 (55.6%). The five structurally empty histories are separately counted and
are excluded from this denominator.

All seven incorrect selections were disagreements with the primary semantic
reference decision. Three had reference reason `wrong_value`, two `ambiguous`,
and two `same_value_conflict`. None was an exact-quotation code-check failure,
and none arose from a failed candidate-quotation recheck. All four additional
candidate-quotation checks accepted the submitted quotation.

| Core Value | Selected candidates | Incorrect against AI reference |
|---|---:|---:|
| Universalism | 6 | 3 |
| Tradition | 2 | 1 |
| Security | 3 | 1 |
| Hedonism | 2 | 1 |
| Conformity | 2 | 1 |
| Power | 3 | 0 |
| Benevolence | 1 | 0 |
| Self-Direction | 0 | Undefined selection precision |
| Stimulation | 0 | No earlier writing; no paid review |

Three examples illustrate distinct limitations:

1. **A state assertion without a supporting action.** For Security,
   `7ff1d0fb:entry:0` selected “I'm fine. The studio is fine. Rent is covered.”
   The full Journal Entry describes a defensive conversation with an aunt and
   a fabricated excuse to end it; it does not describe the action that secured
   the rent. Gemini returned `not_supportive` / `wrong_value`.
   [Saved case](cases/7ff1d0fb_security_episode_01.json).
2. **Supportive writing selected despite conflicting context.** For Hedonism,
   `bf44e50f:entry:1` selected the passage beginning “I went and made tea and sat
   back on the balcony”. The same Journal Entry subsequently says “I couldn't
   let myself have it for more than five minutes” and describes work rumination
   displacing the enjoyment. Gemini returned `not_supportive` /
   `same_value_conflict`. The runtime preserved the quote exactly but failed
   the reference's whole-entry interpretation.
   [Saved case](cases/bf44e50f_hedonism_episode_01.json).
3. **An observable action whose requested-value relationship is disputed.** For
   Universalism, `152df7a4:entry:0` selected helping a warehouse contact resolve
   paperwork: “I walked them through it, helped them resubmit.” The source
   explains that handling the contact personally was faster. Gemini returned
   `not_supportive` / `wrong_value`; the occurrence of a helpful action alone
   did not establish the requested Core Value under that reference decision.
   [Saved case](cases/152df7a4_universalism_episode_01.json).

The other disagreements were a Tradition reflection about a daughter's
learning, two ambiguous Universalism examples involving teaching/art-workshop
outcomes, and a Conformity example about avoiding a family group-chat argument.
The reference's interpretation of contextual Conflict and value specificity is
itself AI-derived and should not be presented as human adjudication.

The single failed case was `dbe2c53d:universalism:episode_01`. Both its initial
OpenAI response and its one permitted retry were valid JSON, but each paired
`decision="abstain"` with `reason_code="other_actor"`. The contract permits that
reason only with `not_supportive`. Both batches therefore correctly failed
closed with `malformed_decision:results.0:value_error`. The saved generic
`review_contract_invalid` error hides this useful detail; the original payloads
allow it to be recovered without rerunning the model.

Consequently, the reported OpenAI unexpected failure rate of 2/29 (6.9%) counts
contract-invalid model responses, not network outages or JSON parsing failures.
Gemini had 0/32 invalid or failed attempts. The rate exceeds the frozen 5%
criterion even though both providers returned responses. The precision and
no-card failures independently require the same stop decision.

Four histories had a reference-accepted example but no selected candidate:
Noor's second Tradition episode and Lukas's second Self-Direction episode were
retrieval misses; `7c712a0a:conformity:episode_01` retrieved an accepted source but
the runtime abstained; the Universalism case above failed structural validation.
Task-specific retrieval recall was 17/19 (89.5%). This is distinct from the
persisted-label proxy used to pass Phase 0A and does not retroactively change
that gate's measurement.

## Metric interpretation and reporting corrections

The frozen arithmetic reproduces, but the reported `verification_lift` needs a
qualification. The retrieval-only rule displays the entire first-ranked
Journal Entry and grades whether its **source identifier** appears among the
reference's supportive decisions. It does not independently grade that entire
displayed quotation. In contrast, reviewed precision grades the selected
**quotation**, including a separate check when it differs from the primary
reference quotation. Thus `retrieval_only_precision=7/28` is source precision,
whereas `precision=12/19` is candidate-quotation precision.

The subtraction, +38.16 percentage points, is arithmetically correct but should
not be presented as established quotation-level verification lift. Without
additional calls, it can be reported as a clearly labelled source-level
comparison: reviewed source precision also happens to be 12/19 in this run.
Alternatively, omit the lift claim and report both measures with their units.
Include coverage alongside either comparison: retrieval-only selects 28/33
cases, while the reviewed pipeline selects 19/33. The frozen report should
remain preserved; explanatory documentation can identify this measurement
limitation without retrospectively changing the protocol.

Additional distinctions needed in documentation are:

- The three `wrong_value` reference decisions are reason-code counts. Some
  involve absence of an observable action, so they do not alone establish the
  specification's stricter rate of actions supporting a different Core Value.
- The 27 valid runtime batches contain 21 abstentions among 71 reviewed Journal
  Entry decisions (29.6%). This excludes the structurally invalid batches; use
  that denominator explicitly if reporting this derived abstention measure.
- “Reference-confirmed no example” means exhaustive AI reference review found
  no accepted example under this protocol. It is not proof that no supportive
  action occurred in the Persona's history, particularly when references
  abstained because context was ambiguous.
- These are development results on repeated episodes from synthetic Personas.
  They are neither an untouched final benchmark nor independent human
  validation. Repeated episodes can share source writing, so 33 episodes must
  not be represented as 33 independent participants.

## Disposition of all five saved Personas

The saved demo configurations are identified in
[`src/demo/scenarios.py`](../../../../src/demo/scenarios.py). Development
episodes below do not necessarily coincide with each demo's highlighted week.

| Saved Persona | Development evidence | Result and boundary |
|---|---|---|
| Wei Jun Chen (`8f83c818`) | Universalism episode with eight eligible sources | Entry 7 was selected and accepted by the AI reference. This is an offline candidate; no NSM Experience card or browser QC was completed. |
| Marc Vandenberghe (`988d1a65`) | Power episode with five eligible sources | Entry 1 was selected and accepted by the AI reference. The saved demo's highlighted week follows the ended Active Drift; this earlier candidate does not authorize showing a card after that end. |
| Noor (`02fb94f3`) | Two Tradition episodes | The first had no reference-accepted earlier example and correctly omitted. The second had an accepted earlier source, entry 7, outside top-3 retrieval and also omitted. |
| Lukas (`11de77e8`) | Two Self-Direction episodes | The first had no reference-accepted earlier example and correctly omitted. The second had accepted earlier sources, entries 4 and 5, outside top-3 retrieval and omitted. |
| Meera (`23d101f8`) | No episode in this development manifest | The configured saved demo is a No Active Drift case at its highlighted week. That configuration supports non-triggering intent; it is not an executed NSM browser test or a no-example search result. |

Wei Jun's accepted quotation is **“Helped two new guys file their claims.”**
It appears exactly in `8f83c818:entry:7`, describing a worker-center morning.
The runtime rejected earlier-ranked entries 3 and 0 before selecting entry 7;
Gemini independently returned the identical quotation. Entry 7 precedes the
episode onset at `t_index=8` (2025-06-25).
[Saved case](cases/8f83c818_universalism_episode_01.json).

Marc's accepted quotation is **“Ran the quarterly review myself today, no
slides from anyone else, just my numbers, my narrative.”** It is the exact
opening sentence of `988d1a65:entry:1`; the remainder describes the room
listening. The runtime abstained on earlier-ranked entry 0 before selecting
entry 1, and Gemini returned the same quotation. Entry 1 precedes onset at
`t_index=5` (2025-03-10).
[Saved case](cases/988d1a65_power_episode_01.json).

Confidence is high in the reproduced counts, hashes, structural validation,
quotation fidelity, retrieval order, and spending reconciliation. Semantic
acceptance remains bounded by the frozen Gemini reference and its four
candidate-quotation checks. No findings justify proceeding to application
integration, fresh onboarding, browser screenshots, or final evaluation after
this failed feasibility gate.
