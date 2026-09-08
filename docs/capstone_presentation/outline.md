# Twinkl — Final Presentation Outline

**Date:** 15 Oct 2026, 6.30PM · **Audience:** professors · **Assumed:** ~20 min talk + Q&A

**Through-line:** *Can AI hold you accountable to your own values, with evidence?*
Three investigations answer it, and the architecture is what the evidence decided.

**Confirm before building slides:** exact talk length, whether a live demo is
expected, and whether the professors have read the Technical Paper.

---

## Part 1 — Introduction (4 slides, ~4 min)

**1. Title**
Twinkl: Evidence-Grounded Value Accountability. Name, programme, date. One
spoken line: what the system does.

**2. The problem**
Journaling records; it rarely confronts. AI assistants tend toward agreement
rather than testing a user's account. Twinkl's narrower target is
product-level ungrounded affirmation, not sycophancy as a model property.
*Visual:* affirming reply vs. evidence-cited reply, side by side.

**3. What the market already does**
AI journaling apps (Reflection, Mindsera, Insight Journal, Day One, Pixel
Journal, Rosebud) summarise mood and trends but treat each entry as an
isolated item. They start from a blank slate, so they have no declared
reference to compare behaviour against, and they optimise for engagement
loops rather than confrontation. The gap is not journaling or AI reflection.
It is the absence of a user-confirmed reference and cited evidence.
*Visual:* the PRD's Summarizer vs. Twinkl contrast, cut to two or three rows.
Use the shared three-entry example so the difference is visible, not asserted.

**4. What Twinkl does differently**
Confirmed Profile of Core Values → chronological Journal Entries → Weekly
Drift Detection → Coach Digest, with Insufficient Evidence as a valid answer.
Schwartz's Theory of Basic Human Values supplies the vocabulary.

Spend a moment on the Profile, since it is the reference everything later
compares against. Eleven balanced Best-Worst Scaling groups, every object
seen six times and every pair three times, with no Schwartz labels or scores
shown to the user. The scoring is deterministic through to at most two
confirmed Core Values, and Inspect exposes the whole calculation from 22
recorded selections onward. Say the boundary in the same breath: the design
is research-grounded, the Twinkl instrument is not psychometrically
validated, and the user confirms the result rather than receiving a
diagnosis.
*Visual:* `images/onboarding-choice.png` beside the one-line product flow.
Say the roadmap here: three investigations, one application, then limits.

## Part 2 — Approach and evidence (2 slides, ~3.5 min)

**5. Where the hypotheses come from**
Twinkl claims no novelty for journaling, Profile construction, memory, or AI
reflection alone. Each design commitment inherits from prior work, and the
contribution is their integration into a contestable accountability path.
Four supports, one line each:

- **Values reference:** Schwartz's ten-value model and its 19-value
  refinement; Best-Worst Scaling elicits relative importance in less
  respondent time than the traditional survey.
- **Why longitudinal:** personal informatics stages and lapse-and-resume
  self-tracking; reflective informatics names breakdown and inquiry as the
  dimensions Twinkl implements.
- **Why not agreement:** sycophancy research shows models can mirror stated
  beliefs when a truthful response would disagree. This is the risk the
  evidence requirements target.
- **Nearest neighbour:** MindScape personalises prompts from passively
  sensed behaviour, but prompts for well-being rather than comparing against
  a confirmed Profile. Name it, then name the difference.

Also state the boundary once, plainly: synthetic Journal Entries, AI review,
development evidence, no fresh final test. Everything after inherits it.
*Visual:* four supports as a column, each with the commitment it backs.
Consider splitting into two slides if the delivery feels crowded.

**6. Architecture and how each part is evaluated**
Adopted architecture figure. Point out the missing arrow: VIF Critic
Predictions do not feed Weekly Drift Detection. Flag it as a promise to
explain, and pay it off in slide 12.

Then the evaluation map, so coverage is visible before the results start and
nobody has to wonder what went unmeasured. One row per component:

| Component | How it is evaluated | Evidence source |
|---|---|---|
| Profile | Deterministic scoring plus Python validation | Not psychometrically validated |
| Corpus and labels | Agreement, stability, frozen-holdout lifts | Synthetic; project-team annotators |
| VIF Critic | QWK, Conflict recall, three seeds | LLM-Judge VIF Labels, frozen test |
| Weekly Drift Detection | Drift recall, false alerts, coverage | AI-reviewed, development-only |
| Coach Digest | Validations plus four eval scores | Same-model AI review, n=5 |
| Application | Tests, replays, fail-closed, deletion | Implementation, not usability |
| Displayed nudge | Implemented, not separately evaluated | — |

Say the last row out loud rather than letting it pass as a table entry.
*Visual:* `images/adopted-architecture.png`, then the table. Two slides if the
figure needs room.

## Part 3 — Investigation 1: the corpus (2 slides, ~3 min)

**7. Building a longitudinal corpus that can be trusted enough**
204 personas, 1,651 Journal Entries. Parallel across personas, sequential
within one, so narrative continuity holds. Banned terms and no generation
metadata in production-like logic. Neutral labels are 75.92% — the long-tail
problem is visible before any training.
*Visual:* corpus numbers plus the label distribution.

**8. Validation and its limits**
Human-human Fleiss' kappa 0.56; mean LLM-Judge-human Cohen's kappa 0.66,
different rater structures, so not a paired advantage. Frozen-holdout targeted
batches gave local lifts and one regression: more synthetic data was not
automatically better.
*Visual:* `images/label-agreement.png` (drop `synthetic-data-lifts.png` if time
is tight; the regression point survives as a spoken sentence).

## Part 4 — Investigation 2: the VIF Critic (2 slides, ~3.5 min)

**9. A compact model, deliberately**
23,454-parameter MLP over frozen embeddings plus Profile weights. 69 run IDs,
133 configurations across ordinal, long-tail, two-stage, and encoder families.
Three-seed reference: median QWK 0.362, Conflict recall 0.313.
*Visual:* the three-seed table, four rows only.

**10. Where it breaks, and why that matters**
It misses about two thirds of Conflicts, and per-value recall runs from 0.125
(Power) to 0.733 (Self-Direction). A negative result inside a tested design
space. This is the reason the model does not hold user-facing Drift authority.
*Visual:* `images/per-value-conflict-recall.png`, left panel only. Slide 15
returns to the full figure for the cross-value analysis, so introduce the
spread here and save the interpretation for then.

## Part 5 — Investigation 3: the drift path (2 slides, ~3.5 min)

**11. Does the offline model help downstream?**
Hand-off ablation: raw Predictions lowered median Drift recall from 9/33 to
7/33 and the paired interval included zero; early triggering changed delay,
not hits. Honest caveat, one sentence: no matched Luna-low ablation was run.
*Visual:* `images/vif-handoff-ablation.png`

**12. Choosing the operating point, and what it produces**
Reviewing cumulative history is the stronger path. Low reasoning effort: 0.548
recall, 4 false alerts, 0.852 precision, 0.637 coverage. Xhigh buys recall and
pays in false alerts. Low was retained as a documented capstone choice, after
replacing a preregistered rule that no longer fit — say so.

Then land the chosen contract in the running system, so the operating point
is not left as an abstraction: this is the reviewer the professors will see
in the demo.
*Visual:* `images/weekly-drift-tradeoff.png`, then Weekly Drift Reviewer
screenshots.

**Screenshots needed — none of these exist yet.** Capture from a saved replay
so they are reproducible and require no provider key:

- A Weekly Drift Reviewer Decision in Inspect, showing the model, low
  reasoning effort, the returned Conflict or Not Conflict, and the
  justification against cited Journal Entries.
- An Abstain or failed review resolving to Insufficient Evidence, which
  demonstrates the fail-closed path that the coverage number implies.

Prefer the second if only one fits. Recall and coverage are abstract, and
Insufficient Evidence is the visible consequence of choosing low effort.

## Part 6 — The application (2 slides, ~3.5 min)

**13. One traced example**
Wei Jun, Universalism. Conflicts at t8 and t9 confirm Active Drift across a
week boundary; t10 makes the run three. Then Inspect shows model, effort,
justifications, cutoffs, and state transitions behind that exact result.
*Visual:* `images/active-drift-experience.png`, then
`images/inspect-evidence-trail.png`. **This is the slide to demo live if a demo
is expected.** Keep the screenshots as fallback.

**14. Coach Digest, and what verifies the application**
Five saved responses passed validations; same-model evals averaged 4.80
correctness and 4.60 tension honesty. Five same-model reviews, so this is a
calibration target, not an error rate. Say "AI-reviewed" out loud.

Then the verification behind the demo, since the System Implementation and
Demo assessment is judged here too: 196 passing Python tests and 158 passing
React tests at the 30 August checkpoint, five hash-checked replay bundles
with no-future-data projection, fail-closed validation on invalid model
output, and confirmed deletion of browser and temporary Python state. State
the ceiling immediately: this establishes implemented and inspectable
behaviour, not usability, load reliability, or deployment readiness.
*Visual:* verification counts as a short block, not prose.

## Part 7 — Limitations (2 slides, ~3 min)

Own the boundaries here, in full, while the results are still fresh. Slide 16
in the conclusion then only has to point back rather than re-argue. Delivery
matters more than content on these two: state each limit flatly, do not
apologise, and do not soften with "but". Confidence about what the evidence
cannot support is what earns the conclusions that follow.

**15. Not every value is equally detectable**
The most interesting limitation is domain-specific: Twinkl detects some
Schwartz values far better than others, and the weak ones are weak for
different reasons.

- **Weak in both components.** Stimulation stayed at 0.333 Conflict recall in
  the VIF Critic and 0.300 under Luna-low. Tradition ran 0.400 and 0.292.
  These are the honest failures, since neither more history nor a stronger
  model rescued them.
- **Rescued by history.** Power was the VIF Critic's worst value at 0.125 but
  reached 0.567 once the reviewer saw cumulative history. This is the
  clearest evidence that some Conflicts are only visible longitudinally.
- **Lost by it.** Self-Direction moved the other way, 0.733 down to 0.533.
  Worth showing rather than hiding, because it argues against a simple
  more-context-is-better story.
- **Unstable by support.** Universalism rested on three VIF test Conflicts,
  Achievement on two under Luna-low. Recall on that little support is noise,
  not a finding, so read those numbers with the support beside them.

Say clearly that the two panels are not a controlled comparison. Different
data, labels, and inputs, so this locates where the work is, and does not
rank the components. The matched analysis on the stored 106-case union is
exactly what would settle it, which is why it leads future work.

Two domain limits alongside the per-value picture. A Conflict is judged from
what the person wrote, so an unwritten week is invisible and Twinkl cannot
distinguish a value abandoned from one merely unmentioned. And human values
are not independent, whereas the ten dimensions are scored as if they were.
Conflict between competing values is the phenomenon Schwartz's circular
continuum describes, and Twinkl only sees it one dimension at a time.

*Visual:* `images/per-value-conflict-recall.png`, both panels, with the four
groupings called out on the figure itself.

**16. Evidence limits and one honest trade-off**
Four evidence limits, one line each, grouped by what they block. Deliver at
pace; slide 15 is the one that deserves the time.

- **Synthetic corpus.** Designed coverage, not population representativeness.
  The same model family generated the Journal Entries and their original
  labels, so generator and judge errors may correlate. Unmeasured.
- **Not human ground truth.** 115 entries, 19 personas, three project-team
  annotators, not an independent panel.
- **No fresh final test.** The 42-Drift study selected the model contract, so
  reporting it as final-test performance would be leakage.
- **AI reviewing AI.** All drift references are AI-reviewed, and one model
  both wrote and scored five Coach Digest responses.

Scope gaps, briefly: the displayed nudge is unevaluated, the Profile
instrument is not psychometrically validated, and no real user has used the
system. Mechanical checks catch a broken evidence link, not whether a person
found a response helpful or well timed.

Then the trade-off worth volunteering, because a professor may find it
anyway: Luna-low failed its original preregistered coverage gate. The project
replaced that rule with the recall-first hierarchy and then kept low even
though xhigh scored higher on recall. That is a documented capstone judgment,
not a preregistered optimum, and the deck should say so before Q&A does.

Close the part in one line: none of this invalidates the findings, it bounds
them, and the bounds are what the next slides work within.

## Part 8 — Conclusion (3 slides, ~3.5 min)

**17. What the evidence decided**
Corpus is fit for bounded development. The VIF Critic reaches a real but
inadequate frontier and stays offline as a Pattern Recognition Systems
contribution. The Weekly Drift Reviewer plus the deterministic Drift Detector
own user-facing Drift. Architecture followed evidence, not the proposal.

**18. What this supports, and what it does not**
One slide, two columns, no new content. Supported: bounded development use of
the labels, an evidence-driven architecture decision, an implemented and
inspectable assessment path. Not supported: a direct MLP-versus-LLM
performance claim, real-user benefit, deployment readiness. Part 7 already
made the case, so deliver this as a summary and keep moving.

**19. Future work and close**
Matched per-value error analysis on the stored 106-case union, a frozen final
test, human calibration of the AI review, and a five-to-ten-user pilot. Each
one answers a limit from Part 7, so name the pairing rather than listing the
work. Close on the opening question, answered: accountability made testable
and inspectable.

---

## Q&A preparation

Prepare a one-slide answer for each; keep them in the appendix. Part 7 should
pre-empt the first four. If a professor asks one anyway, answer briefly and
point to the slide rather than repeating the whole argument.

1. **Isn't this just an LLM beating an MLP?** No. Different inputs, labels, and
   granularity. Not a controlled model comparison, and the paper says so.
2. **Why trust synthetic data?** It is bounded development evidence.
   Frozen holdouts, label QA, repeated-call stability, human overlap. Not
   ground truth, and no claim that it is.
3. **Why keep the VIF Critic at all?** It answers the research question and
   produces the negative result that justified the architecture.
4. **Isn't AI evaluating AI circular?** Yes, and it is disclosed. Five
   responses, same model, self-enhancement risk unmeasured. Human calibration
   is named future work.
5. **Why 0.548 recall as acceptable?** It is not a deployment threshold. Drift
   requires two consecutive Conflicts and Insufficient Evidence is a valid
   output, so the cost of a miss is a delayed prompt, not a false accusation.
6. **Is this therapy?** No. Assessment-only scope, non-prescriptive language,
   fail-closed validation, explicit non-therapy boundary.
7. **How do you know the Profile is right?** Two separate claims. The scoring
   is deterministic and Python-validated, and Inspect shows every step. The
   instrument itself is not psychometrically validated, which is why the user
   confirms the result and why Drift needs behavioural evidence on top of it.

## Appendix slides (held back)

Drift Detector state diagram, Best-Worst Scaling scoring, full experiment
table, per-value metrics, test and deployment evidence.

## Delivery notes

- 19 main slides, roughly 24 minutes. If the slot is 20, cut slide 8, compress
  slide 5 to two supports, and reduce slide 16 to the four evidence limits with
  the trade-off spoken. Keep slide 15 whole; the per-value analysis is a
  research finding, not boilerplate. Do not cut the evaluation map on slide 6
  or the verification block on slide 14; both are assessed directly.
- Slides 6, 12, and 14 each carry two jobs and may split under rehearsal.
- One claim per slide; the figure carries the number, not the bullet text.
- Say the evidence source aloud whenever it changes the conclusion:
  synthetic, AI-reviewed, development-only.
- Rehearse the three transitions that carry the argument: corpus → model,
  model → drift path, drift path → architecture decision.
- Part 7 is the tone risk in this deck. Rehearse it until it sounds like
  command of the evidence rather than hedging.

## Before the deck is built

- Capture the two Weekly Drift Reviewer screenshots for slide 12 from a saved
  replay. Nothing in `docs/capstone_report/images/` shows the reviewer in
  production today.
- Confirm talk length, whether a live demo is expected, and whether the
  professors have read the Technical Paper.
