# April Project Proposal

This report presents Twinkl's April 2026 milestone as a sponsor-facing project
update. The goal is to show that the team has built a credible technical core,
understands the main delivery risks, and has a realistic path to a demonstrable
product slice.

[**1. Introduction**](#1-introduction)

[Problem Statement](#problem-statement)

[Differentiation and Target Users](#differentiation-and-target-users)

[Related Academic Work](#related-academic-work)

[**2. System Architecture**](#2-system-architecture)

[End-to-end Pipeline](#end-to-end-pipeline)

[**3. Offline Data Generation and Model Training**](#3-offline-data-generation-and-model-training)

[Synthetic Persona Generation](#synthetic-persona-generation)

[LLM-as-Judge Labeling](#llm-as-judge-labeling)

[Human Annotation and Validation](#human-annotation-and-validation)

[Critic (VIF) Model Training](#critic-vif-model-training)

[Training and Experiment Infrastructure](#training-and-experiment-infrastructure)

[Evaluation Framework](#evaluation-framework)

[**4. Online Model Inference**](#4-online-model-inference)

[User Value Profile Construction](#user-value-profile-construction)

[Journal Entry Ordinal Alignment Classification](#journal-entry-ordinal-alignment-classification)

[Journal Entry Uncertainty Quantification](#journal-entry-uncertainty-quantification)

[Behavioral Intelligence: Drift and Evolution](#behavioral-intelligence-drift-and-evolution)

[Explainable Feedback via Coach](#explainable-feedback-via-coach)

[**5. Project Evaluation Metrics**](#5-project-evaluation-metrics)

[Four-Stage Gate Structure](#four-stage-gate-structure)

[Judge Validation with Human Annotation](#judge-validation-with-human-annotation)

[Value Modelling](#value-modelling)

[Drift Detection](#drift-detection)

[Explainable Feedback via Coach](#explainable-feedback-via-coach-1)

[**6. Project Progress and Results**](#6-project-progress-and-results)

[Judge Validation](#judge-validation)

[VIF Experiment Progression](#vif-experiment-progression)

[Current Critic VIF Frontier](#current-critic-vif-frontier)

[What Was Tried Beyond the Frontier](#what-was-tried-beyond-the-frontier)

[Current Findings](#current-findings)

[Annotation Tool](#annotation-tool)

[Phase 1 (April 2026 Scope-Locking Wave)](#phase-1-april-2026-scope-locking-wave)

[Phase 2 (June-September 2026)](#phase-2-june-september-2026)

[Deferred and Out-of-Scope](#deferred-and-out-of-scope)

[**7. Challenges and Open Questions**](#7-challenges-and-open-questions)

[Hard Dimensions Still Set the Ceiling](#hard-dimensions-still-set-the-ceiling)

[Label Stability Improved, But Did Not Fully Solve the Problem](#label-stability-improved-but-did-not-fully-solve-the-problem)

[Statistical Power Remains a Real Constraint](#statistical-power-remains-a-real-constraint)

[What Has Been Ruled Out](#what-has-been-ruled-out)

[Open Questions for the Next Phase](#open-questions-for-the-next-phase)

[**8. Conclusion**](#8-conclusion)

[**Appendix: Program Submodule Mapping**](#appendix-program-submodule-mapping)

## 1. Introduction

### Problem Statement

Ambitious people often know what matters to them, but their weekly behaviour
drifts quietly away from those priorities. Health, relationships, meaningful
work, and personal direction are easy to state and hard to sustain under
ordinary pressure. Existing journaling products help users log feelings, answer
prompts, and review trends, but they rarely maintain an explicit model of what
the user said they care about and then evaluate later behaviour against it.

Twinkl addresses that gap. The product treats declared priorities as system
state, then compares new journal behaviour against that state through the Value
Identity Function (VIF): a multi-dimensional, uncertainty-aware evaluator built
around Schwartz's ten-value framework. The intended result is not a better mood
summary. It is an evidence-grounded weekly reflection on whether the user's
actions still match the direction they claimed to want.

For this milestone, the core question is practical rather than theoretical:
can this alignment engine be built, trained, validated, and run in a way that
is technically credible enough to justify continued investment?

### Differentiation and Target Users

Twinkl is aimed first at knowledge workers in transition: graduate students,
new managers, founders, and other high-agency professionals managing recurring
trade-offs across work, health, relationships, and identity. These users
already journal, seek coaching, or pay for reflective productivity tools, yet
most available products still stop at summarisation, prompting, or general
wellness encouragement.

The product distinction is straightforward: most AI journaling tools explain
what the user has been feeling; Twinkl is designed to ask whether the user has
been living in line with what they said matters.

| Current AI Journals | Twinkl |
| :--- | :--- |
| Summarise mood, topics, and streaks | Compare behaviour against a declared value profile |
| Treat each entry mainly as a stand-alone reflection | Maintain an evolving self-model across entries |
| Offer general prompts or coaching questions | Surface evidence-grounded tensions, trade-offs, and alignment patterns |
| Optimise for engagement and reflection volume | Optimise for honest accountability and explainable weekly review |

This milestone has already de-risked four important questions:

1. The offline teacher-student pipeline can be built at usable scale.
2. Human calibration is strong enough to justify judge supervision for a POC.
3. A compact critic can learn meaningful alignment signal from the available
   data.
4. The project already has an internal runtime path from model checkpoint to
   weekly output.

The main remaining risk is not whether the concept is interesting. It is
whether the value model is reliable enough to support trustworthy user-facing
feedback beyond synthetic and internal review settings.

![Sponsor milestone scorecard](sponsor_milestone_scorecard.png)

**Current delivery state**

| Delivery state | What it means in this milestone |
| :--- | :--- |
| Implemented now | Synthetic persona generation, judge labeling, annotation workflow, critic training, experiment archive, and an internal checkpoint-to-digest review flow are all working today. |
| Implemented experimentally / not yet validated | Runtime inference, weekly aggregation, crash/rut routing, weekly digest generation, and first-pass narrative validation can run, but they are not yet calibrated or sponsor-ready. |
| Specified but not built | Live onboarding, onboarding-to-runtime integration, real-user journaling orchestration, calibrated weekly detection benchmarks, and a true end-user product loop are not yet delivered. |

This milestone should therefore be read as a technically coherent internal
prototype, not as a user-ready product.

Although this report is written for a sponsor audience, the project still spans
intelligent sensing, pattern recognition, reasoning, and AI systems
orchestration; the formal program mapping is provided in the appendix.

### Related Academic Work

#### Schwartz Theory of Basic Human Values

Twinkl uses Schwartz's ten-value framework because it provides a practical and
well-studied structure for modelling competing priorities rather than treating
alignment as a single scalar score (Schwartz, 1992; Schwartz et al., 2012). In
this project, the framework is not included as background theory alone; it is
the operational schema for persona generation, judge rubrics, and model
targets.

#### Best-Worst Scaling for Value Elicitation

Best-Worst Scaling was chosen for onboarding because it produces clearer
trade-offs than Likert-style self-rating and reduces the "everything matters"
problem common in value elicitation (Louviere et al., 2015). That matters for
Twinkl because the critic is intended to interpret journal behaviour in light
of the user's declared priorities, not in light of generic values alone.

#### Synthetic Data & LLM-as-Judge

Twinkl uses a generate-annotate-learn pipeline because there is no public
dataset that pairs longitudinal journal behaviour with per-dimension value
alignment labels at the level of detail required here. Synthetic generation
provides controlled trajectories, while rubric-guided LLM judging provides
scalable supervision that can then be distilled into a smaller runtime model
(He et al., 2022; Zheng et al., 2023). Because this supervision is subjective,
the project validates it explicitly against human annotation rather than
treating synthetic labels as self-justifying.

## 2. System Architecture

### End-to-end Pipeline

![End-to-end pipeline](end_to_end_pipeline.png)

Twinkl has two linked but distinct system layers. The first is an offline
teacher-student pipeline that creates data, labels it, and trains the critic.
The second is an experimental runtime path that turns a saved critic checkpoint
into weekly alignment signals and coach-ready artifacts. The important sponsor
point is that the offline core is already real, while the runtime path is
usable internally but not yet fully validated as a product experience.

| System layer | Role | Current state |
| :--- | :--- | :--- |
| Offline teacher-student pipeline | Generate journal trajectories, label them, and train the critic | Implemented |
| Runtime scoring and weekly review | Rebuild state, score entries, aggregate weekly signals, route drift modes, and generate digest artifacts | Implemented experimentally |
| Onboarding and live product loop | Collect user priorities, feed them into runtime state, and support live journaling behaviour | Specified only |

At a high level, the online path is organised into three layers:

1. **Critic**: produces per-dimension alignment scores and uncertainty from a
   compact state representation.
2. **Weekly routing**: aggregates scores into weekly signals and applies
   uncertainty-gated crash/rut-style logic.
3. **Coach**: converts those signals into evidence-grounded weekly reflections.

Each layer operates over a different scope of information: entry-level state
for scoring, weekly aggregation for routing, and broader journal history for
explanation.

## 3. Offline Data Generation and Model Training

Twinkl's offline core is no longer conceptual. The project now has a complete
teacher-student pipeline covering data generation, labeling, human validation,
critic training, and experiment review. As of this milestone, that pipeline
operates over **204 synthetic personas** and **1,651 judged journal entries**.

### Synthetic Persona Generation

The project requires paired training data: journal entries and per-dimension
value-alignment labels. Because no public dataset exists for this problem, the
team built a synthetic generation pipeline that produces longitudinal journal
trajectories rather than isolated text snippets.

Each persona combines a demographic profile with 1-2 emphasized Schwartz
dimensions. Generation runs in parallel across personas but sequentially within
each persona so later entries can refer to earlier events. Entries vary in tone
and reflection style, and the pipeline can optionally generate conversational
nudge-response turns to better reflect how users interact with modern
journaling products.

The pipeline also includes leakage controls. Explicit Schwartz terminology is
banned from generated text, generation context is separated from labeling
context, and targeted augmentation for weak dimensions is accepted only after
judged outcome checks rather than prompt intent alone.

Targeted generation has been especially important for rare and difficult
dimensions. Instead of relying only on generic "misaligned" scenarios, the
pipeline can produce value-specific tension cases designed to probe the critic's
weakest semantic boundaries.

### LLM-as-Judge Labeling

The judge labels each journal entry across all ten Schwartz dimensions on a
three-way ordinal scale:

| Label | Meaning |
| :---: | :--- |
| \-1 | Misaligned with the value |
| 0 | Neutral or no relevant evidence |
| \+1 | Aligned with the value |

The judge reads the persona biography, prior journal history, and the current
session content. This broader context lets it interpret vague entries as part
of a trajectory rather than as isolated one-off statements. Each label is
accompanied by a short rationale to support later review.

The labeling path is auditable: raw markdown is wrangled into structured
entries, labels are produced with schema validation, and the outputs are then
consolidated into a deterministic training dataset. A consensus re-judging path
also exists as a diagnostic branch so the team can test label stability across
multiple independent judge passes.

### Human Annotation and Validation

Because the supervision problem is subjective, Twinkl validates judge labels
against human scoring rather than assuming that synthetic labels are reliable by
construction. The team built a custom annotation application that presents
entries sequentially within a persona context and collects per-dimension
\{-1, 0, +1\} judgments.

For the current benchmark, three annotators labeled a shared **115-entry**
subset spanning **19 personas**, producing roughly **380 total annotations**.
This gives the project a credible human reference point for judging whether the
LLM supervision is usable at POC scale.

![Annotation tool](annotation_tool.png)

### Critic (VIF) Model Training

The critic is a compact multi-layer perceptron that learns to approximate the
judge's per-dimension alignment decisions while remaining cheap enough for
runtime inference. This is the core distillation step that turns an expensive,
context-heavy teacher into a practical student model.

The current default setup uses:

- a frozen sentence encoder based on Nomic Embed Text v1.5
- Matryoshka truncation to **256 dimensions**
- a compact state representation with **window size = 1**
- a **10-dimensional value profile vector**
- ordinal decoding into \{-1, 0, +1\}
- **MC Dropout** for uncertainty estimation

The current state intentionally favors reliability over architectural
ambition. Larger history windows were explored but overfit badly at current
data scale, so the active baseline keeps the runtime state compact and treats
short-horizon context as a targeted next-step question rather than a default.

A Bayesian baseline is also implemented to provide a secondary uncertainty
reference, but the main sponsor story remains the same: the project already has
a working student model, and the open question is how far that model can be
pushed before target quality becomes the real bottleneck.

### Training and Experiment Infrastructure

The training workflow is disciplined rather than ad hoc. It includes:

- persona-level splitting to avoid leakage across correlated histories
- fixed holdout manifests for comparable follow-up experiments
- guardrailed checkpoint selection based on ranking quality and minority-signal
  recovery
- logging of configurations, metrics, and run commentary
- post-hoc threshold tuning on validation data only

This matters for sponsor confidence because the team is no longer making claims
from isolated best-seed results. The project now has a repeatable evaluation
regime, an archived experiment history, and explicit promotion criteria for new
challengers.

The codebase also maintains automated tests across the core pipeline, including
critic training and evaluation, judge labeling, data wrangling, coaching and
runtime behavior, and local integration smoke checks. For a sponsor, that is a
practical sign that the engineering workflow is mature enough to support
continued iteration without treating every change as a manual regression risk.

### Evaluation Framework

The evaluation suite tracks both aggregate and tail-sensitive behaviour:

| Metric | Purpose |
| :--- | :--- |
| QWK (Quadratic Weighted Kappa) | Primary ranking metric for ordinal agreement |
| Recall\_-1 | Recovery rate for misalignment labels |
| MinR (Minority Recall) | Sensitivity to the non-neutral classes overall |
| Hedging rate | How often the model collapses to neutral predictions |
| Calibration | Whether reported confidence matches actual correctness |

To avoid over-interpreting noise, configurations are compared across three
training seeds and reviewed at the family level rather than on single runs.
Persona-cluster bootstrap intervals are then used to test whether apparent
gains survive holdout uncertainty.

## 4. Online Model Inference

### User Value Profile Construction

Twinkl is designed to address the cold-start problem through a Best-Worst
Scaling onboarding flow that elicits an initial value profile from forced
trade-offs. The proposed flow presents six 4-item sets, a mid-flow mirror, and
a goal-selection step so the system can capture both value priorities and the
reason the user joined.

That design remains important, but it is not yet live. In the current POC, the
onboarding flow is fully specified but not deployed, and graded onboarding
weights are not yet wired into the active runtime path. The working runtime
still uses declared core values already present in persona data to condition
inference.

This distinction matters. The product vision depends on live onboarding, but
the technical milestone delivered so far is the profile-conditioned critic path,
not the onboarding-to-runtime loop itself.

### Journal Entry Ordinal Alignment Classification

At runtime, the critic outputs per-dimension alignment scores for each journal
entry in \{-1, 0, +1\}. The current baseline uses ordinal heads because the
classes have meaningful order: misaligned, neutral, and aligned are not merely
three unrelated buckets.

| Value | Score | Interpretation | Example |
| :--- | :---: | :--- | :--- |
| Benevolence | \+1 | The entry supports the value | "I dropped everything to help my neighbour move." |
| Hedonism | 0 | The value is not materially active | "I worked extra hours today." |
| Tradition | \-1 | The entry conflicts with the value | "I snapped at my parents over something they expected me to do." |

The current operating point uses BalancedSoftmax because the main practical
failure mode in earlier models was excessive neutral hedging. For Twinkl, that
failure mode is especially damaging: a critic that suppresses rare \-1 signals
cannot support any downstream drift logic.

It is also important to distinguish the synthetic-data pipeline from the live
runtime. Conversational nudge logic exists and is used in synthetic data
generation, but the current runtime is not yet a fully wired live journaling
experience. The project can model entry-plus-response behaviour internally, but
it does not yet provide a complete session-time product loop for end users.

### Journal Entry Uncertainty Quantification

MC Dropout remains active at inference time so the critic can produce both a
point estimate and an uncertainty estimate for each dimension. In the current
setup, **50 stochastic forward passes** are used to estimate epistemic
uncertainty.

This is operationally important because Twinkl should not surface confident
critiques when the model is unsure. In the current design, uncertainty acts as
a gating signal: high uncertainty should suppress strong claims and route the
system toward clarification or lighter-touch feedback instead.

### Behavioral Intelligence: Drift and Evolution

The runtime aggregates per-entry critic outputs into weekly signals so that the
system can reason about patterns rather than overreacting to single entries. In
the active POC, the core routing logic is a simple crash/rut framing:

- **Crash**: a sharp weekly drop on important dimensions
- **Rut**: sustained low alignment on declared priorities
- **High uncertainty**: the model is too unsure to support a strong critique

This layer exists and runs today, but it should still be treated as
experimental. The thresholds are not yet calibrated against a weekly benchmark,
and the quality of the routing logic is still constrained by the upstream
critic's current frontier quality.

The project also contains two related but different extensions:

| Capability | Current role |
| :--- | :--- |
| Multi-detector comparison | Evaluation surface used to compare alternative heuristic families offline |
| Evolution analysis | Experimental layer used to test whether sustained divergence may reflect changing priorities rather than failure |

Neither should be presented as committed production behaviour yet. The sponsor
story for this milestone is narrower: weekly crash/rut-style routing is
implemented experimentally, but calibration and benchmark validation are still
pending.

### Explainable Feedback via Coach

The coach layer converts critic and weekly-routing signals into a digest that is
clear, evidence-grounded, and non-judgmental. In the current POC, this takes
the form of a weekly artifact that:

- selects focus tensions and strengths
- cites representative journal evidence
- builds a full-context weekly prompt
- can optionally generate a narrative response
- runs first-pass checks on groundedness, non-circularity, and length

This is further along than a design sketch, but it is not a finished product
experience. Structured digest generation is implemented, narrative-generation
hooks exist, and internal validation checks exist, but user-facing explanation
quality has not yet been proven through real evaluation.

## 5. Project Evaluation Metrics

### Four-Stage Gate Structure

![VIF evaluation pipeline](vif_evaluation_pipeline.png)

The project uses a sequential evaluation model because every downstream layer
depends on the reliability of the layer before it. User-facing weekly feedback
is only as trustworthy as the supervision, critic modelling, and weekly routing
that feed it.

| Evaluation | What it validates | Key metric | Current status |
| :--- | :--- | :--- | :--- |
| Judge validation | Are the labels consistent enough to train on? | Judge-human Cohen's κ | **Operational**: κ = 0.66 on the shared benchmark |
| Value modeling | Can the critic recover meaningful ordinal alignment signal? | QWK, Recall\_-1, Minority Recall | **In progress**: current frontier QWK = 0.362 vs 0.40 target |
| Drift detection | Can weekly routing identify crash/rut patterns reliably? | Hit rate, precision, recall | **Partial**: runtime path exists, benchmark still pending |
| Explanation quality | Can the system produce grounded, useful feedback? | User-rated usefulness / accuracy | **Partial**: digest generation exists, user-facing evaluation still pending |

### Judge Validation with Human Annotation

Judge validation is the strongest completed evaluation gate in the project.
Three annotators independently labeled a shared subset of entries using the
custom annotation tool, blind to the judge outputs and with chronology
preserved within each persona.

On the shared benchmark:

- **Fleiss' κ = 0.56** for human-human agreement
- **Average Cohen's κ = 0.66** for judge-human agreement
- the judge exceeds human-human consistency on **9 of 10 dimensions**

This is strong enough to support judge supervision for a POC, but it is not the
end of the story. Follow-up audits showed that agreement alone is not enough:
the hardest dimensions can still be unstable or unreachable from the student's
available context. That is why the project now treats supervision quality as an
active design problem rather than as a solved preprocessing step.

### Value Modelling

Value modelling is evaluated at two levels.

At the **entry level**, the key question is whether the critic can classify
individual reflections credibly enough to support downstream logic. QWK is the
main ranking metric because it respects the ordinal structure of \{-1, 0, +1\},
while minority-signal metrics reveal whether the model can recover the rare but
important misalignment cases Twinkl exists to surface.

At the **persona level**, the question is whether aggregated entry scores can
recover the broader value profile implied by a journal trajectory. That remains
an important downstream goal, but the current bottleneck is still entry-level
reliability. Until that improves further, persona-level ranking is better
treated as a follow-on evaluation than as the main proof point for this
milestone.

### Drift Detection

Drift detection is currently evaluated as a weekly crash/rut-style routing task
over aggregated critic outputs and uncertainty estimates. The implementation is
in place, but the benchmark is not. The team still needs synthetic
crisis-injection timelines, calibrated thresholds, and measured hit
rate/precision/recall before this layer can be treated as validated.

### Explainable Feedback via Coach

Explanation quality is evaluated by asking whether weekly feedback is grounded,
clear, and useful. The project already supports structured digest generation and
first-pass automated checks, but user-facing quality targets remain ahead of the
current evidence. For this milestone, the honest claim is partial progress: the
explanation path is implemented enough to inspect and improve internally, but it
is not yet validated with user ratings.

## 6. Project Progress and Results

This milestone moved Twinkl from concept and architecture into a measured,
auditable technical core. The project now has a complete offline supervision
pipeline, a validated human benchmark, a serious experiment archive, and an
internal runtime path that can turn a saved checkpoint into weekly review
artifacts.

**What is runnable today**

- Generate synthetic personas and longitudinal journal data
- Produce judge labels with rationales and consolidate them into a training set
- Run human annotation and agreement analysis on a shared benchmark
- Train and compare critic families on fixed holdouts
- Load a saved checkpoint, score a persona timeline, aggregate weekly signals,
  route crash/rut modes, and generate weekly digest artifacts for internal
  review

### Judge Validation

Before relying on LLM-generated supervision at scale, the team expanded the
human benchmark from a small initial sample to **115 shared entries across 19
personas**.

| Metric | Value | Interpretation |
| :--- | ---: | :--- |
| Shared annotation subset | 115 entries across 19 personas | Final like-for-like benchmark |
| Human-Human Agreement (Fleiss' κ) | 0.56 | Moderate |
| Judge-Human Agreement (Avg Cohen's κ) | 0.66 | Substantial |
| Dimensions where Judge \> Human-Human | 9 / 10 | Strong enough for POC supervision |

The project's later audits made this story more precise rather than weaker:

| Follow-up audit | Scope | Key result | Practical implication |
| :--- | :--- | :--- | :--- |
| Reachability audit | 50 hard cases | Security labels, especially positive ones, were often not reproducible from student-visible context | Some of the bottleneck sits in the target, not just the student |
| Consensus re-judging | 5 judge passes over all 1,651 entries | Judge self-consistency was strong, but the relabeled holdout changed too much for clean replacement of the persisted-label benchmark | Consensus labels are useful diagnostics, not yet a direct frontier replacement |

The sponsor-relevant takeaway is simple: judge supervision is usable overall,
but not equally trustworthy across all dimensions.

### VIF Experiment Progression

The model search is now large enough to support disciplined comparison rather
than isolated intuition. The archive covers **50 run IDs** and **114 persisted
configurations**, spanning multiple ordinal losses, calibration strategies,
representation choices, and reformulations.

| Stage | Main work completed | Main insight |
| :--- | :--- | :--- |
| Baseline search | Compared multiple ordinal and long-tail training families | The early bottleneck was neutral-class hedging and weak recovery of misalignment |
| Evaluation reset | Corrected the validation/test split regime | Earlier leaderboard claims were invalidated; the project reset onto a fairer benchmark |
| Frontier discovery | Established a BalancedSoftmax family as the first convincing corrected-split baseline | The first credible gains came from reducing neutral collapse without abandoning QWK entirely |
| Systematic follow-up | Tested weighting, encoder changes, two-stage reformulation, and relabeling diagnostics | Several ideas improved specific metrics, but none yet replaced the incumbent cleanly |

This matters because the team is no longer moving randomly through model ideas.
The frontier is now narrow, the remaining questions are better isolated, and
the project has evidence for why several tempting alternatives did not become
the default.

### Current Critic VIF Frontier

The active reference point is the current BalancedSoftmax family under the
corrected-split, persisted-label regime.

| Metric | Median (3 seeds) |
| :--- | ---: |
| QWK | 0.362 |
| Recall\_-1 | 0.313 |
| Minority Recall | 0.448 |
| Hedging | 62.1% |
| Calibration | 0.713 |

This is not yet good enough to support strong sponsor claims about end-user
reliability. It is, however, strong enough to justify the next round of focused
work. The critic is clearly learning meaningful signal, but hard dimensions and
target quality still cap the ceiling.

### What Was Tried Beyond the Frontier

The main post-frontier challengers clarified what still does and does not move
the system:

| Challenger | Best result | Why it did not replace the incumbent |
| :--- | :--- | :--- |
| BalancedSoftmax with per-dimension weighting | Best tail-sensitive recovery on misalignment | Improved minority recovery, but overall ranking quality was too unstable |
| Qwen-based encoder swap | Slightly higher surface QWK and lower hedging | Hard dimensions, especially Hedonism and Power, remained too weak |
| Two-stage reformulation | Competitive QWK and strong calibration | Became too conservative overall; misalignment recall fell too much |
| Consensus-label retrain | Better within-regime QWK under relabeled evaluation | Changed the holdout labels, so the result is not a clean like-for-like replacement |

The most important methodological improvement from this phase is not a single
metric gain. It is that the project now compares challengers using family-level
medians and bootstrap confidence intervals rather than point estimates alone.

### Current Findings

The project is now in a healthier position than it was earlier in the semester.
The main conclusions are:

1. **The supervision pipeline is defensible at POC scale**, but hard dimensions
   still expose target-design weaknesses.
2. **The current critic is credible as a baseline, not as a finished model**.
3. **Several nearby alternatives have now been tested and ruled out as default
   replacements**, which narrows the remaining search space.
4. **The remaining ceiling is better understood**: part representation problem,
   part target-quality problem, with Security the clearest example of the
   latter.

The next phase is therefore no longer "try more model ideas." It is focused
work on hard-dimension targets, compact context, and better alignment between
training signals and frontier metrics.

### Annotation Tool

The project's support tooling is also materially stronger than a typical first
milestone. Beyond the core training stack, the team has a functioning
annotation workflow and an internal review surface that can inspect a saved
model checkpoint all the way through to weekly outputs.

![Internal review surface](demo_review.png)

This matters because it turns the project from a paper pipeline into something
the team can actually inspect, debug, and demonstrate. A sponsor should read
this as evidence of delivery momentum, even though the current review surface
is still internal rather than end-user facing.

The team also has an interactive embedding explorer for qualitative inspection
of what the critic has learned across all **1,651** labeled entries. It
supports multiple projection and coloring modes, lets reviewers inspect
individual points in detail, and traces persona trajectories through the learned
representation over time. This is valuable not as proof on its own, but as an
internal review tool that helps the team inspect model behavior beyond
aggregate metrics.

### Phase 1 (April 2026 Scope-Locking Wave)

As of **6 April 2026**, the project has moved from broad exploration into a
scope-locking phase. The central question is now narrower: is the remaining VIF
ceiling driven mainly by unreachable hard-dimension targets, by missing
short-horizon context, or by limits in the frozen representation?

| Workstream | Status | Purpose | Decision it informs |
| :--- | :--- | :--- | :--- |
| Hard-dimension reachability audit | Completed | Test whether the hardest labels are actually learnable from student-visible context | Whether the current frontier is imitating an inaccessible target |
| Consensus re-judging diagnostic | Completed | Measure repeated-call stability across the full corpus | Whether label instability is the dominant remaining bottleneck |
| Matched counterfactual hard-set | Open | Build sharper boundary cases for Hedonism, Security, and Stimulation | Whether the critic still fails on near-counterfactual distinctions after target refinement |
| Compact history / context prototype | Open | Reintroduce short-horizon context without recreating severe overfitting | Whether the single-entry state is now an artificial ceiling |
| Training-signal divergence analysis | Open | Test whether current selection signals promote the wrong checkpoints | Whether later training changes should focus on model choice or selection policy |

The most decisive completed result is the reachability audit. It showed that
aggregate judge-human agreement was not enough to guarantee that the hardest
stored labels are clean student targets. That finding changes how the current
frontier should be read: it remains the active baseline, but it is partly a
pre-target-redesign reference point rather than the final capstone endpoint.

#### Integration Milestone

In parallel, the team now has a clearer picture of what a sponsor-facing
demonstration could look like. The core path remains:

**onboarding -> critic -> weekly digest -> coach**

The practical constraint is that the current milestone only supports the middle
of that path directly. A messaging-native wrapper remains a plausible Phase 2
delivery option, but it should be treated as a conditional packaging decision
rather than as part of the current proven core.

| Integration component | Current status | Remaining gap |
| :--- | :--- | :--- |
| Onboarding / BWS flow | Documented | Not yet live and not yet wired into active runtime |
| Critic runtime path | Implemented experimentally | Hard-dimension target redesign still needed before stronger claims |
| Weekly digest generation | Initial implementation complete | Upstream calibration and evaluation still in progress |
| Drift-aware routing | Partial | Weekly benchmark calibration remains incomplete |
| Coach explanation path | Partial | Stronger explanation-to-signal linkage and user-facing evaluation still needed |
| Confidence-gated triggering | Partial concept plus supporting pieces | Policy layer is not yet finalized as a product behaviour |
| Messaging-native wrapper | Conditional future direction | No implementation yet; only worth pursuing once core outputs are stable enough to show users |

The sponsor-relevant point is that the project can now show a technically
coherent path from declared priorities to scored evidence to weekly reflective
output, even though the full end-user journey is not yet complete.

### Phase 2 (June-September 2026)

Phase 2 should be treated as a conditional deepening phase, not an automatic
continuation of broad model search. The emphasis should depend on what the
current scope-locking work proves.

| Phase 2 track | Planned work | Why it belongs in Phase 2 | Dependency / Gate |
| :--- | :--- | :--- | :--- |
| Representation follow-up | Add a gated parameter-efficient adaptation path if the frozen encoder is still the main bottleneck | Keeps the model search focused on the strongest remaining technical question | Only after target cleanup and compact-context testing |
| Drift calibration | Generate synthetic crisis-injection timelines and report hit rate / precision / recall | Converts the current runtime bridge into a properly evaluated weekly routing layer | Requires a stronger upstream critic and a benchmark pass |
| Narrow messaging-native pilot | Prototype chat-based journaling, cadence checks, and digest delivery | Offers a sponsor-friendly demonstration path without building a separate full application | Only if the core critic and digest outputs are stable enough to surface |
| External user pilot | Run a small structured validation study with real users to test whether the current alignment and digest outputs transfer beyond synthetic personas and internal review | Converts the largest remaining product risk, external validity, into a measured outcome | Requires stable core outputs and one narrow end-to-end path that is safe to demonstrate |
| Evolution decision | Decide whether evolution gating remains experimental or enters active scope | Prevents overclaiming long-horizon behaviour change | Depends on the calibrated weekly-routing story |
| Full end-to-end demonstration | Connect onboarding, scoring, weekly routing, digest generation, and coach output into one clear flow | Moves the project from component validation to system demonstration | Depends on the final scope choice and Phase 1 outcomes |

The intended Phase 2 output is therefore twofold: a more defensible final
system path, and a clearer final capstone claim about what Twinkl can and
cannot yet do credibly.

### Deferred and Out-of-Scope

Several ideas remain attractive, but they are not the right focus for the
current capstone milestone. The guiding rule is simple: do not let downstream
product richness dilute the validation of the core judge -> critic -> coach
loop.

| Deferred area | Current status | Why deferred |
| :--- | :--- | :--- |
| Full multi-channel sensing and long-running observability | Conceptual future work | Adds security, privacy, token-cost, and persistence complexity beyond a scoped POC |
| Habit recommendation system | Parked | Introduces a second intervention problem and requires longitudinal user evidence |
| Goal-aligned inspiration feed | Not started | Requires external search integration and separate evaluation that do not strengthen the current milestone claim |
| Multimodal fusion | Future work | Adds major sensing and dataset complexity beyond the text-first capstone scope |
| Offline RL for nudge or intervention policies | Later-stage idea | Requires stable rewards and longer user trajectories than the project currently has |
| Adaptive onboarding and dynamic profile refinement | Conceptually important | Depends on validated value-evolution logic and real-user calibration |

The project has deliberately chosen depth over breadth. That remains the right
choice.

## 7. Challenges and Open Questions

The project's remaining challenges are now clearer than they were at the start
of the semester. The main technical bottleneck is concentrated in a small
number of hard dimensions and in the quality of their supervision targets.
Separately, the biggest product risk is still external validity: most of the
current evidence comes from synthetic personas and internal review rather than
from real users. Closing that gap through a small external pilot is an explicit
Phase 2 deliverable.

### Hard Dimensions Still Set the Ceiling

The current frontier stabilises around a BalancedSoftmax-based student that is
better than earlier baselines at recovering rare active labels. Even so,
aggregate performance remains capped by three difficult dimensions:
**Security**, **Hedonism**, and **Stimulation**.

These dimensions do not fail for the same reason:

- **Security** appears to be the clearest teacher-student reachability problem.
  Many labels are not reproducible from student-visible context alone.
- **Hedonism** remains primarily a semantic polarity problem. The model still
  confuses healthy rest and boundary-setting with avoidance or guilt.
- **Stimulation** is both semantically difficult and statistically weak under
  the current evaluation regime.

The project is therefore no longer asking only which loss function works best.
It is asking which parts of the current supervision signal are genuinely
learnable from the available inputs.

### Label Stability Improved, But Did Not Fully Solve the Problem

Recent work showed clearly that the hard-dimension ceiling is partly a
target-quality problem, not only a modelling problem. Consensus re-judging was
useful because it proved that the judge can be internally stable on the hard
dimensions, but it also showed that greater stability does not automatically
produce a better training target.

That is an important maturity signal for the project. The team is no longer
assuming that more relabeling is automatically better. The current challenge is
to improve target quality without erasing the rare, disagreement-heavy edge
cases that matter most operationally.

### Statistical Power Remains a Real Constraint

The corrected holdout is much more credible than the earlier regime, but it is
still small enough that moderate model differences can disappear under proper
uncertainty analysis. This means the project must be careful not to overclaim
incremental gains, especially on dimensions that remain close to chance-level
behaviour.

That constraint is frustrating, but it is not a failure. It is exactly why the
project now relies on family medians, bootstrap intervals, and more disciplined
promotion criteria.

### What Has Been Ruled Out

This phase has already ruled out several simpler explanations:

- The bottleneck is **not only** loss design.
- It is **not only** generic model capacity.
- It is **not only** class imbalance.
- It is **not only** task formulation.

Those are useful negative results. They narrow the remaining search space and
make the next milestone more focused.

### Open Questions for the Next Phase

The most important remaining questions are:

1. **What is the right student-reachable supervision target for the hardest
   dimensions, especially Security?**
2. **Can a compact history representation recover missing signal without making
   the critic impractical?**
3. **Will a matched hard-set provide a cleaner test of the critic's semantic
   boundaries than further broad data generation?**
4. **Are current selection signals still misaligned with the frontier metrics
   the project actually cares about?**
5. **How much of the current synthetic-persona performance transfers to real
   journaling behaviour once the project begins external validation?**

## 8. Conclusion

Twinkl has now cleared the most important first-milestone hurdle: the project
is no longer just a product idea with an architecture diagram. It has a
functioning offline supervision pipeline, a defensible human benchmark, a
measured critic-training program, and an internal runtime path that can produce
weekly review artifacts from saved checkpoints.

At the same time, the report should be honest about what remains unfinished.
The project has not yet proven a live onboarding-to-runtime loop, a calibrated
weekly drift benchmark, or real-user explanation quality. The central remaining
risk is whether the value model can be made reliable enough on the hardest
dimensions to support trustworthy user-facing feedback.

The next milestone must therefore do three things clearly: improve the quality
of the hardest supervision targets, calibrate the weekly routing layer against
a real benchmark, and demonstrate one narrow, sponsor-ready path from declared
priorities to scored evidence to weekly reflective output. If those three steps
succeed, Twinkl will have a strong claim not just as an interesting capstone,
but as a viable product concept with a technically credible core.

## Appendix: Program Submodule Mapping

The following table maps the project's main technical contributions to the
practice module's four capability areas.

| Submodule | Twinkl Mapping |
| :---- | :---- |
| Pattern Recognition | Ordinal value classification, uncertainty estimation, and sequence-aware weekly aggregation |
| Intelligent Sensing | Text-derived signals from journal content, temporal patterns, and declared priority profiles |
| Intelligent Reasoning | Weekly routing logic, evidence-grounded explanation, and profile-conditioned interpretation |
| Architecting AI Systems | End-to-end orchestration from offline supervision to runtime scoring, weekly review, and future user-facing delivery |
