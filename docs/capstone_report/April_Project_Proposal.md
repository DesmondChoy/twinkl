---
title: "Twinkl: An inner compass that helps users align behavior with long-term values"
subtitle: "Capstone 5 - Project Proposal (Apr 2026)"
format:
  pdf:
    toc: true
    toc-depth: 3
    documentclass: scrartcl
    papersize: a4
    geometry:
      - top=25mm
      - bottom=25mm
      - left=25mm
      - right=25mm
    mainfont: "TeX Gyre Pagella"
    sansfont: "TeX Gyre Heros"
    monofont: "TeX Gyre Cursor"
    colorlinks: true
    linkcolor: "black"
    urlcolor: "blue"
    header-includes:
      - \addtokomafont{disposition}{\rmfamily}
      - \AtBeginDocument{\author{Choy Yong Yi Desmond [A0315402W] \\ Leong Kay Mei [A0188702Y] \\ Loke Yuen Ying, Jodie [A0310555M]}}
      - \usepackage{float}
      - \floatplacement{figure}{H}
      - \usepackage{needspace}
      - \usepackage{etoolbox}
      - \BeforeBeginEnvironment{longtable}{\Needspace{0.3\textheight}}
---

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
around Schwartz's ten-value framework. The intended result is an
evidence-grounded weekly reflection on whether the user's actions still match
the direction they claimed to want, rather than a better mood summary.

For this milestone, the core question is whether this alignment engine can be
built, trained, validated, and run in a way that justifies continued
investment.

### Milestone Status

This milestone already answers four practical questions:

1. The offline teacher-student pipeline can be built at usable scale.
2. Human calibration provides a benchmark for judge supervision at POC scale.
3. A compact critic can learn meaningful alignment signal from the available
   data.
4. The project already has an internal runtime path from model checkpoint to
   weekly output.

The main remaining risk is whether the value model is reliable enough to
support trustworthy user-facing feedback beyond synthetic and internal review
settings.

![Milestone scorecard](sponsor_milestone_scorecard.png)

**Current delivery state**

| Delivery state | What it means in this milestone |
| :--- | :--- |
| Implemented now | Synthetic persona generation, judge labeling, annotation workflow, critic training, experiment archive, and an internal checkpoint-to-digest review flow are all working today. |
| Implemented experimentally / not yet validated | Runtime inference, weekly aggregation, crash/rut routing, weekly digest generation, and first-pass narrative validation can run, but they are not yet calibrated or ready for external demonstration. |
| Specified but not built | Live onboarding, onboarding-to-runtime integration, real-user journaling orchestration, calibrated weekly detection benchmarks, and a true end-user product loop are not yet delivered. |

At this stage, Twinkl remains an internal prototype rather than a user-ready
product.

### Differentiation and Target Users

Twinkl is aimed first at knowledge workers in transition: graduate students,
new managers, founders, and other high-agency professionals managing recurring
trade-offs across work, health, relationships, and identity. These users
already journal, seek coaching, or pay for reflective productivity tools, yet
most available products still stop at summarisation, prompting, or general
wellness encouragement.

The product distinction is straightforward: most AI journaling tools explain
what the user has been feeling; Twinkl is designed to ask whether the user has
been living in line with what they said matters. Several funded competitors
illustrate the gap:

| App | Core Approach | Gap vs Twinkl |
| :--- | :--- | :--- |
| Rosebud | CBT/ACT-based therapeutic reflection with habit tracking | No declared-value model; analyses mental patterns, not behavioural drift from stated priorities |
| Reflection | AI companion with mood tracking and retrospective theme discovery | Values discovered post-hoc from entries, not proactively declared or tracked over time |
| Mindsera | 50+ generic mental frameworks (CBT, Stoic, Ikigai, etc.) with multi-perspective AI | External frameworks applied uniformly, not adapted to user-specific value trade-offs |
| Reflectly | Mood tracking with conversational AI and daily challenges | Purely emotional pattern analysis; no values framework or alignment assessment |
| Life Note | 1,000+ AI mentors modelled on historical figures | Mentor philosophies are fixed; no mechanism to track the user's own evolving priorities |
| Stoic | Stoic philosophy daily rituals with GPT-4 | Single prescriptive philosophy, not a user-declared values model |

The addressable market spans digital journal apps (USD 5.69 billion in 2025,
11.5% CAGR), mental health apps (USD 7.48 billion, 14.6% CAGR), and personal
development tools (USD 46.1 billion, 8.0% CAGR). Twinkl sits at the
intersection: structured value-alignment coaching delivered through journaling.

![Market context](market_context.png)

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
into weekly alignment signals and coach-ready artifacts. The offline core is
already real, while the runtime path is usable internally but not yet fully
validated as a product experience.

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

Twinkl's offline core is now implemented. The project has a complete
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
and reflection style, and the pipeline includes a conversational nudge system
that uses a three-category decision engine (clarification, elaboration,
tension-surfacing) with anti-annoyance throttling (maximum two nudges per three
entries). Nudge-response pairs are integrated across 62% of the current corpus
and better reflect how users interact with modern journaling products.

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
This gives the project a human reference point for judging whether the LLM
supervision is usable at POC scale.

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

A Bayesian baseline is also implemented as a secondary uncertainty reference.
The broader takeaway is unchanged: the project already has a working student
model, and the open question is how far that model can be pushed before target
quality becomes the real bottleneck.

### Training and Experiment Infrastructure

The training workflow is disciplined rather than ad hoc. It includes:

- persona-level splitting to avoid leakage across correlated histories
- fixed holdout manifests for comparable follow-up experiments
- guardrailed checkpoint selection based on ranking quality and minority-signal
  recovery
- logging of configurations, metrics, and run commentary
- post-hoc threshold tuning on validation data only

These choices reduce the risk of over-reading isolated best-seed results and
make follow-up experiments more comparable. The project now has a repeatable
evaluation regime, an archived experiment history, and explicit promotion
criteria for new challengers.

The codebase maintains **53 test modules** covering all nine ordinal loss
functions (CORAL, CORN, EMD, SoftOrdinal, CDW-CE, BalancedSoftmax, Two-Stage
BalancedSoftmax, LDAM-DRW, and SLACE), gradient flow and numerical stability,
evaluation metrics, coach runtime, nudge decision logic, data wrangling, the
demo workbench, and local integration smoke tests. That level of automated
coverage supports continued iteration without turning every change into a
manual regression risk.

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

### BWS Onboarding

Twinkl addresses cold-start with a Best-Worst Scaling (BWS) onboarding flow.
Six 4-item sets — adapted from PVQ-21 portrait items (Schwartz et al., 2001)
as concise first-person mobile cards — produce an ordinal ranking of the 10
Schwartz value dimensions. The top two are designated **core values**
(monitored for decline); the remaining eight are **peripheral** (monitored for
emergence). A mid-flow mirror presents the emerging ranking for optional
refinement, and a goal-selection step (work-life balance, life transition,
relationships, health/wellbeing, direction, meaningful work) anchors the
Coach's initial framing.

For the Critic, core/peripheral designation is converted into a binary weight
vector (equal weight on core, zero on peripheral) concatenated with the text
embedding in the State Encoder, conditioning alignment predictions so that
ambiguous entries are interpreted against declared priorities.

The BWS flow is fully specified but not yet deployed. A planned extension
derives a continuous 10-dimensional weight vector from raw BWS scores, giving
every dimension a signal proportional to its rank.

### Journal Entry Alignment Classification

#### Conversational Nudges

Within a session, nudges operate on observable text signals:
**clarification** (ambiguous entries), **elaboration** (terse entries), and
**tension surfacing** (Critic-detected misalignment on a core value). An
anti-annoyance gate suppresses nudges after two have fired in the last three
entries.

#### Per-Dimension Scoring

For each journal entry, the Critic outputs per-dimension alignment scores in
\{−1, 0, +1\}:

| Value | Score | Interpretation | Example |
|---|---|---|---|
| Benevolence | +1 | Lived this value | "I dropped everything to help my neighbour move" |
| Hedonism | 0 | Not relevant to entry | "I worked extra hours today" |
| Tradition | −1 | Acted against this value | "I snapped at my parents over something they expected me to do" |

Each dimension uses an **ordinal classification head** enforcing −1 < 0 < +1,
preventing the model from assigning high probability to both extremes while
giving low probability to the neutral class.

**Balanced Softmax loss** corrects class priors in the softmax denominator,
addressing severe imbalance (75.9% neutral, 7.1% misaligned) without
resampling. This shifts the decision boundary so rare misalignment signals
surface at inference without inflating false positives — critical because a
Critic that suppresses −1 predictions would blind downstream drift detection.

**MC Dropout** (50 forward passes) estimates epistemic uncertainty,
suppressing critiques when confidence is low and triggering clarifying
questions instead.

### Drift & Evolution

Drift is the gap between declared values and the temporal trend of
Critic-observed behaviour, sustained beyond noise thresholds — not a single
bad entry but a pattern persisting across weeks and surviving uncertainty
gating.

The drift layer operates on **weekly aggregates** of raw per-entry Critic
scores, smoothing within-week noise while preserving trends.

#### Core Value Signals (decline)

| Signal | Pattern |
|---|---|
| **Crash** | Weekly profile-weighted mean drops >0.5 from previous week |
| **Rut** | Alignment stays below −0.4 for ≥3 consecutive low-uncertainty weeks |

#### Peripheral Value Signals (emergence)

| Signal | Pattern |
|---|---|
| **Rise** | Sustained 0→+1 shift — a previously peripheral value consistently appears, suggesting genuine value evolution |

#### Detection Pipeline

The detector makes a single weekly decision, evaluating triggers in priority
order (first match wins):

1. **High Uncertainty Gate** — If overall model uncertainty exceeds 0.3,
   return immediately. The Coach asks for clarification rather than judging on
   unreliable scores.

2. **Evolution Classification** — Over a 6-week lookback, compute
   per-dimension residual (observed minus expected alignment) and volatility
   (s.d. of scores). Residual < 0.4 → stable; residual ≥ 0.4 with volatility <
   0.5 → evolution (genuine shift); residual ≥ 0.4 with volatility ≥ 0.5 →
   volatile (divergent but unsettled). Evolution dimensions are excluded from
   crash/rut checks and routed to Coach messaging inviting profile updates.

3. **Rut Detection** — For each non-evolution core dimension (weight ≥ 0.15),
   flag as rut if all of the last 3 weeks show low alignment.

4. **Crash Detection** — If this week's profile-weighted mean drops >0.5 from
   last week, flag each non-evolution core dimension where alignment
   decreased.

5. **Stable** — No trigger fired.

The dual-trigger design was chosen for interpretability: crash maps to
"something shifted this week," rut maps to "this has been weighing on you for a
while."

#### Experimental Detector Ensemble

To evaluate whether the dual-trigger approach has blind spots, complementary
rule-based detectors were benchmarked:

```{=latex}
\Needspace{0.55\textheight}
```

| Detector | Mechanism | Catches |
|---|---|---|
| **EMA** | Running average of misalignment (recent-weighted, negative scores only); alerts when average exceeds threshold | Slow fades (+1→0) that never reach −1 |
| **CUSUM** | Cumulative sum with slack parameter (k=0.3) absorbing minor noise; jar grows on persistent misalignment, resets to zero on recovery | Intermittent dips (−1,+1,−1,+1) that individually recover but accumulate |
| **Cosine** | Cosine similarity between profile weight vector and alignment vector; alerts when vectors oppose | Multi-value reversals where several dimensions shift moderately negative |
| **Control Chart** | Per-dimension baseline from early entries; lower control limit = mean − 2 s.d. | Scores nominally neutral but abnormally low for that user (e.g., habitual +1 dropping to 0) |
| **KL Divergence** | KL divergence between baseline and rolling 3-entry score distributions (Laplace-smoothed) | Distributional shape changes (e.g., consistent +1 shifting to a 0/−1 mix) |

These detectors run over the full synthetic corpus and are visualised in a
comparison dashboard. In the absence of human-labelled crisis weeks,
cross-detector consensus (≥4 of 6 flagging the same timestep) serves as
pseudo-ground-truth.

A direction under exploration is replacing the dual-trigger with
consensus-based firing (multiple detectors must agree), reducing blind spots at
the cost of interpretability. Proper evaluation requires synthetic
crisis-injection timelines with ground-truth labels, which is planned but not
yet completed.

### Explainable Feedback via Coach

The coach layer converts critic and weekly-routing signals into a digest that is
clear, evidence-grounded, and non-judgmental. In the current POC, this takes
the form of a weekly artifact that:

- selects focus tensions and strengths
- cites representative journal evidence
- builds a full-context weekly prompt
- can optionally generate a narrative response
- runs first-pass checks on groundedness, non-circularity, and length

The coach selects from four active behavioural modes — **high uncertainty**,
**rut**, **mixed state**, and **background strain** — plus a **stable**
fallback when no tension is detected. Each mode shapes the tone, evidence
selection, and question framing of the weekly digest so the feedback matches
the user's current behavioural pattern rather than applying a generic template.

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

Judge validation currently includes a shared benchmark in the project. Three
annotators independently labeled a shared subset of entries using the custom
annotation tool, blind to the judge outputs and with chronology preserved
within each persona.

On the shared benchmark:

- **Fleiss' κ = 0.56** for human-human agreement
- **Average Cohen's κ = 0.66** for judge-human agreement
- the judge exceeds human-human consistency on **9 of 10 dimensions**

This benchmark is the basis for continuing to use judge supervision in the
current POC, but it does not settle the question. Follow-up audits showed that
agreement alone is not enough: the hardest dimensions can still be unstable or
unreachable from the student's available context. That is why the project now
treats supervision quality as an active design problem rather than as a solved
preprocessing step.

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
current evidence. For this milestone, the explanation path is implemented
enough to inspect and improve internally, but it is not yet validated with user
ratings.

## 6. Project Progress and Results

This milestone adds a complete offline supervision pipeline, a validated human
benchmark, an experiment archive, and an internal runtime path that can turn a
saved checkpoint into weekly review artifacts.

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
| Dimensions where Judge \> Human-Human | 9 / 10 | Current benchmark used for POC supervision |

Later audits refined that conclusion:

| Follow-up audit | Scope | Key result | Practical implication |
| :--- | :--- | :--- | :--- |
| Reachability audit | 50 hard cases | Security labels, especially positive ones, were often not reproducible from student-visible context | Part of the bottleneck sits in the target as well as the student |
| Consensus re-judging | 5 judge passes over all 1,651 entries | Judge self-consistency was strong, but the relabeled holdout changed too much for clean replacement of the persisted-label benchmark | Consensus labels are useful diagnostics, not yet a direct frontier replacement |

Judge-human agreement varies across dimensions, so supervision quality still
needs to be interpreted dimension by dimension.

### VIF Experiment Progression

The model search is now large enough to support disciplined comparison rather
than isolated intuition. The archive covers **50 run IDs** and **114 persisted
configurations**, spanning multiple ordinal losses, calibration strategies,
representation choices, and reformulations.

| Stage | Main work completed | Main insight |
| :--- | :--- | :--- |
| Baseline search | Compared multiple ordinal and long-tail training families | The early bottleneck was neutral-class hedging and weak recovery of misalignment |
| Evaluation reset | Corrected the validation/test split regime | Earlier leaderboard claims were invalidated; the project reset onto a fairer benchmark |
| Frontier discovery | Established a BalancedSoftmax family as the corrected-split reference baseline | Early gains came from reducing neutral collapse without abandoning QWK entirely |
| Systematic follow-up | Tested weighting, encoder changes, two-stage reformulation, and relabeling diagnostics | Several ideas improved specific metrics, but none yet replaced the incumbent cleanly |

The frontier is narrower, the remaining questions are better isolated, and the
project has evidence for why several alternatives did not become the default.

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

The current frontier remains below the report's target range for end-user
reliability, but the critic is clearly learning signal. The next round of work
therefore remains focused on hard dimensions and target quality.

The per-dimension breakdown shows where the critic already captures meaningful
signal and where hard dimensions drag the aggregate:

![Per-dimension QWK frontier](per_dimension_qwk_frontier.png)

### What Was Tried Beyond the Frontier

The main post-frontier challengers clarified what still does and does not move
the system:

| Challenger | Best result | Why it did not replace the incumbent |
| :--- | :--- | :--- |
| BalancedSoftmax with per-dimension weighting | Best tail-sensitive recovery on misalignment | Improved minority recovery, but overall ranking quality was too unstable |
| Qwen-based encoder swap | Slightly higher surface QWK and lower hedging | Hard dimensions, especially Hedonism and Power, remained too weak |
| Two-stage reformulation | Competitive QWK and strong calibration | Became too conservative overall; misalignment recall fell too much |
| Consensus-label retrain | Better within-regime QWK under relabeled evaluation | Changed the holdout labels, so the result is not a clean like-for-like replacement |

One methodological change in this phase is that the project now compares
challengers using family-level medians and bootstrap confidence intervals
rather than point estimates alone.

### Current Findings

The main conclusions at this stage are:

1. **The supervision pipeline is in active use at POC scale**, but hard
   dimensions still expose target-design weaknesses.
2. **The current critic is the active baseline, but it is not yet a finished
   model**.
3. **Several nearby alternatives have now been tested and ruled out as default
   replacements**, which narrows the remaining search space.
4. **The remaining ceiling appears to include both representation and
   target-quality issues**, with Security as the most visible example of the
   latter.

The next phase should focus on hard-dimension targets, compact context, and
better alignment between training signals and frontier metrics, instead of
returning to broad model search.

### Annotation Tool

Beyond the core training stack, the project also includes a functioning
annotation workflow and an internal review surface that can inspect a saved
model checkpoint all the way through to weekly outputs.

![Internal review surface](demo_review.png)

This makes the pipeline inspectable, debuggable, and demonstrable, even though
the current review surface is still internal rather than end-user facing.

The team also built an interactive embedding explorer using Three.js for
qualitative inspection of what the critic has learned across all **1,651**
labeled entries. The tool offers 3D visualization with multiple projection
modes (PCA and t-SNE), per-dimension coloring, individual point inspection, and
persona trajectory tracing through the learned representation over time. It
serves as an internal review tool for auditing model behaviour beyond aggregate
metrics.

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

A key completed result is the reachability audit. It showed that
aggregate judge-human agreement was not enough to guarantee that the hardest
stored labels are clean student targets. The current frontier therefore remains
the active baseline, but it is partly a pre-target-redesign reference point
rather than the final capstone endpoint.

#### Integration Milestone

In parallel, the team has outlined a plausible demonstration path. The core
path remains:

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

The project can now show a coherent path from declared priorities to scored
evidence to weekly reflective output, even though the full end-user journey is
not yet complete.

### Phase 2 (June-September 2026)

Phase 2 should be treated as a conditional deepening phase, not an automatic
continuation of broad model search. The emphasis should depend on what the
current scope-locking work proves.

```{=latex}
\Needspace{0.85\textheight}
```

| Phase 2 track | Planned work | Why it belongs in Phase 2 | Dependency / Gate |
| :--- | :--- | :--- | :--- |
| Representation follow-up | Add a gated parameter-efficient adaptation path if the frozen encoder is still the main bottleneck | Keeps the model search focused on one remaining technical question | Only after target cleanup and compact-context testing |
| Drift calibration | Generate synthetic crisis-injection timelines and report hit rate / precision / recall | Converts the current runtime bridge into a properly evaluated weekly routing layer | Requires a stronger upstream critic and a benchmark pass |
| Narrow messaging-native pilot | Prototype chat-based journaling, cadence checks, and digest delivery | Offers a clear demonstration path without building a separate full application | Only if the core critic and digest outputs are stable enough to surface |
| External user pilot | Run a small structured validation study with real users to test whether the current alignment and digest outputs transfer beyond synthetic personas and internal review | Converts the largest remaining product risk, external validity, into a measured outcome | Requires stable core outputs and one narrow end-to-end path that is safe to demonstrate |
| Evolution decision | Decide whether evolution gating remains experimental or enters active scope | Prevents overclaiming long-horizon behaviour change | Depends on the calibrated weekly-routing story |
| Full end-to-end demonstration | Connect onboarding, scoring, weekly routing, digest generation, and coach output into one clear flow | Moves the project from component validation to system demonstration | Depends on the final scope choice and Phase 1 outcomes |

Phase 2 output should include a final system path and a final capstone claim
about what Twinkl can and cannot yet do based on the completed work.

### Delivery Surfaces and Cost Sketch

The critic model contains approximately **23,000 parameters** and runs **50
MC Dropout forward passes** per inference. At this scale, model footprint and
entry-level scoring cost are not the main commercialization constraint. The
more significant recurring cost sits in weekly narrative generation rather than
in alignment inference itself.

That makes the next packaging question less about raw model deployability and
more about delivery surface. Two Phase 2 paths remain credible:

```{=latex}
\Needspace{0.45\textheight}
```

| Delivery surface | Why it fits Twinkl | Main trade-off | Current status |
| :--- | :--- | :--- | :--- |
| Standalone mobile app | Provides a fully owned surface for onboarding, journaling, nudges, and weekly coaching | Highest product and UX implementation burden | Plausible Phase 2 path; not yet built |
| OpenClaw-based conversational surface | Preserves the same Twinkl core while delivering journaling and weekly review through existing messaging channels, with a natural place for scheduled check-ins and orchestration | Adds a dependency on an external orchestration layer and still requires a scoped wrapper around the current core | Researched integration option; not yet implemented |

The project does not need to commit to either path yet. The sponsor-relevant
point is that the alignment engine is lightweight enough to support more than
one credible delivery surface, while the real Phase 2 decision is which user
experience should carry the proven core into a demonstrable end-to-end product.

### Deferred and Out-of-Scope

Several ideas remain attractive, but they are not the right focus for the
current capstone milestone. The guiding rule is to avoid letting downstream
product richness dilute validation of the core judge -> critic -> coach loop.

```{=latex}
\Needspace{0.6\textheight}
```

| Deferred area | Current status | Why deferred |
| :--- | :--- | :--- |
| Full multi-channel sensing and long-running observability | Conceptual future work | Adds security, privacy, token-cost, and persistence complexity beyond a scoped POC |
| Habit recommendation system | Parked | Introduces a second intervention problem and requires longitudinal user evidence |
| Goal-aligned inspiration feed | Not started | Requires external search integration and separate evaluation that do not strengthen the current milestone claim |
| Multimodal fusion | Future work | Adds major sensing and dataset complexity beyond the text-first capstone scope |
| Offline RL for nudge or intervention policies | Later-stage idea | Requires stable rewards and longer user trajectories than the project currently has |
| Adaptive onboarding and dynamic profile refinement | Conceptually important | Depends on validated value-evolution logic and real-user calibration |

The project currently prioritizes depth over breadth.

## 7. Challenges and Open Questions

The project's remaining challenges are concentrated in a small number of hard
dimensions and in the quality of their supervision targets.
Separately, the biggest product risk is still external validity: most of the
current evidence comes from synthetic personas and internal review rather than
from real users. Closing that gap through a small external pilot is an explicit
Phase 2 deliverable.

### Hard Dimensions Still Set the Ceiling

The current frontier is a BalancedSoftmax-based student. Aggregate performance
still remains capped by three difficult dimensions:
**Security**, **Hedonism**, and **Stimulation**.

These dimensions do not fail for the same reason:

- **Security** appears to be a teacher-student reachability problem.
  Many labels are not reproducible from student-visible context alone.
- **Hedonism** remains primarily a semantic polarity problem. The model still
  confuses healthy rest and boundary-setting with avoidance or guilt.
- **Stimulation** is both semantically difficult and statistically weak under
  the current evaluation regime.

The project has moved past asking only which loss function works best. The more
important question now is which parts of the current supervision signal are
genuinely learnable from the available inputs.

### Label Stability Improved, But Did Not Fully Solve the Problem

Recent work showed clearly that the hard-dimension ceiling is partly a
target-quality problem, not only a modelling problem. Consensus re-judging was
useful because it proved that the judge can be internally stable on the hard
dimensions, but it also showed that greater stability does not automatically
produce a better training target.

This changes how the team interprets relabeling results. The team is no longer
assuming that more relabeling is automatically better. The current challenge is
to improve target quality without erasing the rare, disagreement-heavy edge
cases that matter most operationally.

### Statistical Power Remains a Real Constraint

The corrected holdout addresses problems in the earlier regime, but it is still
small enough that moderate model differences can disappear under proper
uncertainty analysis. This means the project must be careful not to overclaim
incremental gains, especially on dimensions that remain close to chance-level
behaviour.

That constraint also explains why the project now relies on family medians,
bootstrap intervals, and promotion criteria.

### What Has Been Ruled Out

This phase has already ruled out several simpler explanations. The remaining
bottleneck cannot be explained by loss design, generic model capacity, class
imbalance, or task formulation alone.

Those negative results narrow the remaining search space and make the next
milestone more focused.

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

By this milestone, the project has delivered a functioning offline supervision
pipeline, a human benchmark, a disciplined critic-training program, and an
internal runtime path that produces weekly review artifacts from saved
checkpoints. The engineering foundation — 53 test modules, nine ordinal loss
variants, six drift detectors, and a four-mode coaching engine — is deeper than
a typical POC and designed to support continued iteration.

The honest gaps remain: the critic has not yet reached the QWK target on the
hardest dimensions, the weekly drift benchmark is not yet calibrated, and no
real user has seen the output. These are the right problems to have at this
stage — they are measurable, scoped, and addressed by the Phase 2 plan.

Three concrete steps define the path forward:

1. **Improve hard-dimension supervision targets** so the critic trains against
   labels that are actually reachable from student-visible context.
2. **Calibrate the weekly routing layer** against synthetic crisis-injection
   timelines with measured hit rate, precision, and recall.
3. **Run a narrow external pilot** to test whether alignment signals and weekly
   digests transfer from synthetic personas to real journaling behaviour.

Risk mitigation is built into the timeline. If hard-dimension targets do not
improve sufficiently, the project will narrow the active value set to the
dimensions where the critic already performs credibly — preserving a
demonstrable end-to-end path while being transparent about scope. If the
external pilot reveals systematic gaps, the team will treat those findings as
the capstone's concluding contribution rather than as a failure.

The team is committed to delivering a working end-to-end demonstration by the
final milestone: from declared priorities to scored evidence to a weekly
reflective output that a real user can evaluate.

## Appendix: Program Submodule Mapping

The following table maps the project's main technical contributions to the
practice module's four capability areas.

| Submodule | Twinkl Mapping |
| :---- | :---- |
| Pattern Recognition | Ordinal value classification, uncertainty estimation, and sequence-aware weekly aggregation |
| Intelligent Sensing | Text-derived signals from journal content, temporal patterns, and declared priority profiles |
| Intelligent Reasoning | Weekly routing logic, evidence-grounded explanation, and profile-conditioned interpretation |
| Architecting AI Systems | End-to-end orchestration from offline supervision to runtime scoring, weekly review, and future user-facing delivery |
