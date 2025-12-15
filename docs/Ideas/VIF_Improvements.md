# VIF Improvements – Conceptual Design & Brainstorm

## 1. Executive Summary

The **Value Identity Function (VIF)** is Twinkl’s internal "compass" that estimates how aligned a user’s recent behaviour is with their long‑term values. It already has a strong conceptual and technical foundation:

- **Teacher–Student architecture**: Generator → Judge (LLM‑as‑Judge) → Critic (MLP).
- **Vector‑valued evaluation** across value dimensions derived from Schwartz’s theory.
- **Temporal awareness** via sliding windows and history statistics.
- **Uncertainty‑aware critiques** using MC Dropout and dual crash/rut triggers.
- **Critic vs Coach separation**, where the Critic scores and the Coach explains.

This document:

- Summarises the **current VIF architecture and behaviour**.
- Identifies **strengths and limitations** across data, modeling, uncertainty, coaching, evaluation, and safety.
- Proposes **concrete, prioritised improvements**, separated into capstone‑friendly extensions and future‑leaning directions.

The goal is not to change the core philosophy of VIF, but to sharpen its **personalisation, robustness, safety, and evaluation story** while staying realistic about the project’s time‑boxed scope.

---

## 2. Current Architecture Analysis

### 2.1 What VIF Does Today

Based on `docs/VIF/VIF_01_Concepts_and_Roadmap.md`, `VIF_02_System_Architecture.md`, `VIF_03_Model_Training.md`, `VIF_04_Uncertainty_Logic.md`, `VIF_Example.md`, and `docs/PRD.md`:

- **Conceptual role**
  - VIF acts as a **temporal critic**: it tracks alignment between what a user says/does (journals, voice notes) and what they previously claimed to value.
  - It outputs a **vector of alignment estimates** per value dimension rather than a single scalar, preserving tensions like "Career up, Relationships down".
  - It is **trajectory‑aware**: one bad day is not a crisis; multi‑week patterns matter.

- **Inputs and state**
  - Inputs per step \(t\): text \(T_{u,t}\), optional audio \(A_{u,t}\), optional physio \(H_{u,t}\), time deltas, and user profile \(z_u\).
  - Core state \(s_{u,t}\) (Tier 1 / POC):
    - Text embedding for current entry (and optionally a small sliding window of recent entries).
    - Time deltas between entries.
    - Simple history statistics (EMA, rolling standard deviation, counts).
    - User profile embedding \(z_u\).

- **Reward modeling pipeline**
  - **Generator**: synthetic personas/journals created using Schwartz‑grounded value elaborations from `config/schwartz_values.yaml` and structured prompts (`docs/Ideas/Synthetic_data.md`, `notebooks/journal_gen.ipynb`).
  - **Judge (LLM‑as‑Judge)**:
    - Scores entries per value dimension using a **categorical rubric**: Misaligned (−1), Neutral (0), Aligned (+1).
    - Produces a vector like `[Health: -1, Career: +1, Family: 0]` as ground truth for the student.
  - **Critic (VIF)**:
    - MLP regressor that learns to map \(s_{u,t}\) → Judge scores.
    - Designed to be fast enough for many MC Dropout forward passes.

- **Inference and memory architecture**
  - For each new user entry:
    1. Build state \(s_{u,t}\) from current and recent history.
    2. Run the Critic with MC Dropout to obtain per‑dimension means and variances.
    3. Aggregate over a coarser time scale (e.g. weekly) for stability.
    4. Apply **crash and rut rules** per dimension.
    5. If confident and a pattern is detected, trigger the **Coach**.
  - The **Critic** uses strictly sequential, recent history; the **Coach** uses semantic retrieval over the long journal history to construct explanations.

- **Uncertainty and critique logic**
  - **MC Dropout** estimates epistemic uncertainty by keeping dropout active and sampling \(N\) outputs.
  - **Crash trigger**: large negative delta vs previous aggregate \(V_{t-1}^{(j)} - V_{t}^{(j)} > \delta_j\).
  - **Rut trigger**: values below a low threshold \(\tau^{(j)}_{low}\) for \(C_{min}\) consecutive periods.
  - **Uncertainty gate**: only trigger critiques when uncertainty \(\sigma_{V_t}^{(j)} < \epsilon_j\); otherwise prefer clarifying questions or presence.

- **Critic vs Coach**
  - Critic: numeric, sequential, conservative in when it speaks.
  - Coach: retrieval‑augmented, narrative, makes sense of patterns the Critic has flagged, and explicitly avoids gamification and generic praise.

### 2.2 Strengths

- **Clear conceptual separation** between data generation, value scoring, numeric evaluation, and conversational coaching.
- **Vector‑valued framing** matches the product’s core promise of surfacing trade‑offs rather than collapsing them.
- **Safety‑aware by design**:
  - Explicit uncertainty gating and examples (e.g. acute grief) where the system should refuse to "score".
  - Dual‑trigger logic focuses on **patterns over time** instead of reacting to noise.
- **Strong ontology grounding** in Schwartz’s values, with detailed behavioural and cultural elaborations in `schwartz_values.yaml`.
- **Capstone‑friendly** architecture: synthetic data + LLM‑as‑Judge + simple MLP Critic hits a sweet spot of conceptual richness and implementation tractability.

### 2.3 Limitations and Gaps

These are natural next frontiers rather than flaws:

- **Personalisation and cold start**
  - Critic is globally trained; user profile \(z_u\) is present but there is no explicit mechanism for per‑user adaptation beyond conditioning.
  - Crash/rut thresholds are currently developer‑driven with limited user‑specific calibration.

- **Data realism and coverage**
  - Synthetic pipeline is strong, but particularly messy or ethically sensitive scenarios (grief, moral injury, trauma, complex caregiving) are under‑specified.
  - Cultural/linguistic variation is handled at the persona level, but not yet stress‑tested against real accents, code‑switching, or local idioms.

- **Uncertainty calibration and OOD detection**
  - MC Dropout is selected but **calibration procedures** (reliability diagrams, empirical thresholds) are not yet concretely documented.
  - OOD detection is described qualitatively (scattered predictions → high uncertainty) but not operationalised on the embedding space.

- **Temporal modeling expressiveness**
  - Sliding window + simple statistics is robust and simple, but may miss:
    - Longer‑range patterns beyond the window.
    - Differential importance of events (e.g., major crisis vs routine update).

- **Coach behaviour and UX**
  - Example dialogues are excellent, but a **formal rubric** for what counts as an acceptable Coach response (and how often to speak) is implicit rather than codified.

- **Evaluation of behavioural impact**
  - Current evals focus on **felt accuracy** and tagging quality, less on whether feedback **changes behaviour** (e.g., reduces drift or regret over time).

---

## 3. Improvement Categories & Proposals

This section groups improvement ideas into categories and, where relevant, distinguishes **capstone‑friendly** steps from **future‑leaning** extensions.

### 3.1 Data Quality & Ontology

**Current foundation**

- Rich Schwartz value elaborations (`config/schwartz_values.yaml`).
- Synthetic persona and journal generation pipeline with configurable personas, tones, verbosity, and reflection modes (`docs/Ideas/Synthetic_data.md`).

**Proposals**

- **Persona conflict coverage (capstone‑friendly)**
  - Explicitly generate personas that sit in classic value conflicts that Twinkl cares about (e.g. Benevolence vs Achievement, Self‑Direction vs Security, Stimulation vs Security).
  - For each conflict type, ensure the dataset includes:
    - Sustained alignment stories.
    - Slow drift stories.
    - Sudden crisis weeks (e.g., work crunch vs family commitments).

- **Edge‑case scenario pack (capstone‑friendly)**
  - Curate a small, manually‑designed set of high‑stakes or ambiguous scenarios that are likely to be mishandled if not considered:
    - Acute grief or diagnosis news (like Sarah’s Stage 4 scenario).
    - Caregiving burnout and moral fatigue.
    - Moral injury (e.g., being forced to compromise on ethics at work).
  - Use these scenarios primarily to **test** that the system behaves conservatively (high uncertainty, presence‑only Coach) rather than as training data to score grief.

- **Cultural nuance checks (future‑leaning)**
  - For a subset of personas, intentionally vary culture, socioeconomic context, and family expectations while holding underlying conflict structure constant.
  - Inspect whether Judge and Critic outputs systematically differ in ways that reveal bias (e.g. penalising collectivist, family‑focused decisions as "misaligned").

### 3.2 Model Architecture & Personalisation

**Current foundation**

- MLP Critic mapping state \(s_{u,t}\) to vector scores.
- User profile \(z_u\) included in state; thresholds \(\delta_j\), \(\tau^{(j)}_{low}\), \(C_{min}\) set by developers.

**Proposals**

- **Multi‑horizon state features (capstone‑friendly)**
  - Enrich \(s_{u,t}\) with additional statistics computed from historical Critic outputs per value dimension:
    - Short‑term EMA (e.g. 1‑week window).
    - Medium‑term EMA (e.g. 1‑month window).
    - Count of days/weeks with no entries (silence) in the last \(H\) days.
  - This preserves the simple MLP architecture while making VIF more sensitive to different temporal scales and journaling lapses.

- **User‑baseline normalisation (capstone‑friendly)**
  - For each user and value dimension, maintain running estimates of mean and variance of VIF outputs.
  - Express drift in terms of **distance from user’s own baseline**, not just absolute thresholds. For example:
    - A dimension whose absolute score is moderate but significantly below this user’s usual level can still be flagged.
  - This supports critiques like "you’re lower than your usual here" instead of "you’re objectively low".

- **Per‑user adapters (future‑leaning)**
  - Add small, low‑rank adapter layers or FiLM‑style conditioning layers that are parameterised by \(z_u\) and lightly updated as more data arrives.
  - Keeps a shared global backbone Critic while allowing personalised refinements without per‑user full retraining.

- **Temporal attention encoders (future‑leaning)**
  - Replace simple concatenation of last \(N\) embeddings with a small attention mechanism over the window.
  - This lets the model focus on salient events (e.g. unusual language or high emotional intensity) rather than treating all recent entries equally.

### 3.3 Uncertainty, Calibration & OOD Detection

**Current foundation**

- MC Dropout to approximate epistemic uncertainty.
- Dual‑trigger logic gated by a per‑dimension uncertainty threshold \(\epsilon_j\).

**Proposals**

- **Concrete calibration protocol (capstone‑friendly)**
  - On a held‑out synthetic test set:
    - Compute predicted mean and variance from MC Dropout for each sample.
    - Plot **reliability diagrams** of variance vs squared prediction error.
    - Estimate simple mappings from raw variance to calibrated "confidence" buckets.
  - Use this to choose \(\epsilon_j\) such that "below this threshold,  e.g. 80–90% of predictions fall within an acceptable error band".

- **Embedding‑space OOD detector (capstone‑friendly)**
  - Train a one‑class model (e.g. One‑Class SVM or Mahalanobis distance) on text embeddings \(\phi_{text}(T)\) from the synthetic training corpus.
  - At inference:
    - Compute a distance or anomaly score for each new entry.
    - If anomaly score exceeds a threshold, mark the state as **OOD**.
    - When OOD:
      - Inflate effective uncertainty and prevent crash/rut triggers.
      - Route to a **presence‑only** or "curiosity" Coach template.

- **Conformal prediction wrapper (future‑leaning)**
  - On top of the Critic, fit a conformal regressor that outputs prediction intervals per value dimension with guaranteed coverage (e.g. 90%).
  - Use interval width as an additional, better‑calibrated uncertainty signal than variance alone.

- **Multi‑teacher distillation (future‑leaning)**
  - Use multiple Judge prompts or multiple LLMs and distill their consensus into the Critic.
  - This can reduce idiosyncratic biases in any single Judge and lead to more stable uncertainty estimates.

### 3.4 Feedback & Coach Behaviour

**Current foundation**

- High‑quality example dialogues in `VIF_Example.md` and `PRD.md`.
- Strong constraints against gamification, generic praise, or harsh judgement.

**Proposals**

- **Coach response rubric (capstone‑friendly)**
  - Make explicit a small checklist every Coach response should satisfy:
    - **Evidence‑based**: reference (quote or paraphrase) at least one concrete phrase or situation from the user’s history.
    - **Value‑anchored**: link the reflection to what the user said matters (their value profile), not to generic ideals.
    - **Non‑prescriptive by default**: avoid direct advice unless the user asks or explicitly signals readiness.
    - **Non‑comparative**: no social comparison or gamified scores.
  - Encode this rubric in prompt templates and in qualitative evaluation forms.

- **Critique frequency & prioritisation (capstone‑friendly)**
  - Define simple policies like:
    - At most **one major critique** and **one acknowledgment** per week.
    - If multiple dimensions trigger:
      - Prioritise dimensions with highest user‑reported importance \(w_{u,t}\).
      - Or prioritise the largest **change from baseline**, not just lowest absolute level.
  - This avoids overwhelming users and keeps the feedback focused.

- **Tone and challenge calibration (future‑leaning)**
  - Infer user’s comfort level with directness and challenge from the onboarding mini‑assessment and early interactions.
  - Condition Coach prompts on a small set of tone profiles (e.g. gentle/reflective, direct/coach‑like, highly validating) and allow the user to adjust.

### 3.5 Evaluation Framework

**Current foundation**

- PRD already outlines evaluations for tagging, value profile modeling, drift detection, explanation quality, and nudge relevance.

**Proposals**

- **Scenario‑based Critic + Coach tests (capstone‑friendly)**
  - For Sarah‑like worked examples and additional fictional personas, create clear expected behaviours:
    - In crash weeks, Benevolence (or relevant value) should be flagged while others are not over‑praised or scapegoated.
    - In rut weeks, repeated low Self‑Direction with low uncertainty should eventually trigger a rut critique.
    - In high‑uncertainty grief weeks, **no numeric critique** should be issued; Coach should use presence language.
  - For each scenario, define a checklist:
    - Did the Critic’s scores move in the expected direction?
    - Were crash/rut triggers fired only when intended?
    - Did the Coach response satisfy the response rubric?

- **User‑study‑ready metrics (capstone‑friendly)**
  - Extend evaluation tables with:
    - **Felt alignment**: user‑rated accuracy of weekly digests and critiques.
    - **Perceived safety**: "I felt judged" vs "I felt understood".
    - **Behavioural follow‑through**: fraction of suggested experiments or reflections that users engage with.

- **Longitudinal impact (future‑leaning)**
  - For a pilot cohort, track whether users report reduced regret or drift over time, or improved alignment between stated values and weekly behaviour.
  - Compare against a journaling‑only baseline (no VIF feedback) if feasible.

### 3.6 Safety, Ethics & Guardrails

**Current foundation**

- Strong qualitative guidance in docs about not pathologising grief, not using gamification, and remaining cautious with uncertain inputs.

**Proposals**

- **Guardrail templates (capstone‑friendly)**
  - Define explicit Coach templates for:
    - **High‑uncertainty / OOD**: acknowledge uncertainty, offer presence, invite more context without analysis.
    - **Sensitive domains** (self‑harm, abuse, trauma): supportive, non‑diagnostic language and signposting to professional help where appropriate.
  - Define clear rules like: "we never interpret grief as misalignment" and document them with examples.

- **Bias and fairness probes (capstone‑friendly to future‑leaning)**
  - Create mirrored persona pairs (e.g. same scenario, different gender or culture labels) and test whether Judge and Critic scores systematically differ.
  - Use this as an internal diagnostic, not necessarily to solve bias fully within the capstone, but to show awareness and early mitigation steps.

---

## 4. Prioritisation Matrix

This matrix roughly orders ideas by **impact** on user safety/experience and **effort** within a 6–9 month capstone.

| Priority | Category | Proposal | Impact | Effort | Notes |
|---------|----------|----------|--------|--------|-------|
| P1 | Data & Ontology | Persona conflict coverage, edge‑case scenario pack | High | Low–Med | Strengthens behavioural diversity and safety stories. |
| P1 | Uncertainty & OOD | Embedding‑space OOD detection + high‑uncertainty templates | High | Med | Makes "we don’t know" operational and testable. |
| P1 | Coach Behaviour | Response rubric + critique frequency policy | High | Low | Clarifies what "good" looks like and avoids over‑nudging. |
| P1 | Evaluation | Scenario‑based tests for Critic + Coach | High | Med | Ties math to narrative examples for XRAI storytelling. |
| P2 | Model Architecture | Multi‑horizon EMAs + silence counts in state | Med–High | Low | Improves temporal sensitivity without new architectures. |
| P2 | Personalisation | User‑baseline normalisation | Med | Low–Med | Makes feedback feel more personal and fair. |
| P2 | Uncertainty | Calibration protocol for MC Dropout | Med | Med | Turns a qualitative safety claim into a quantitative one. |
| P3 | Model Architecture | Per‑user adapters, temporal attention encoders | High | High | Strong research angle but heavier lift. |
| P3 | Evaluation | Longitudinal behavioural impact | High | High | Requires multi‑week pilots and careful design. |
| P3 | Safety & Ethics | Systematic bias probes and mitigation | High | Med–High | Important for production, can be outlined even if partially implemented. |

For the capstone, focusing on **P1 and P2** is likely sufficient to tell a strong, coherent story. **P3** items can be framed as future work and stretch goals.

---

## 5. Open Questions & Research Directions

A few questions to decide as you refine and implement the VIF:

1. **Personalisation vs simplicity**
   - How far do we want to go beyond \(z_u\) as a static embedding during the capstone? Is user‑baseline normalisation enough, or do we want to prototype a tiny per‑user adapter layer?

2. **Time horizon for alignment**
   - For this POC, should the Critic focus on **immediate alignment** (Option A in `VIF_03_Model_Training.md`) with smoothing, or is there capacity to experiment with **short‑horizon forecasts** (Option B)?

3. **Role of real user data**
   - How much real, opt‑in user data (if any) will be available to validate synthetic assumptions? Are we comfortable keeping all training synthetic for the capstone and using real data only for evals?

4. **Tolerance for "silence"**
   - Philosophically, how should VIF interpret gaps in journaling? As neutral, as weak misalignment, or as a separate "engagement" signal? This affects how silence counts in state are used.

5. **Coach intervention philosophy**
   - Should the Coach ever make concrete behavioural suggestions ("micro‑experiments") unprompted, or always stick to reflection unless invited? The answer will shape prompt design and risk profile.

6. **Scope boundaries**
   - Which domains are explicitly **out of scope** for VIF scoring (e.g. acute trauma, complex psychiatric issues), and how will the system recognise and handle them conservatively?

These questions do not block implementation, but clarifying them will help keep the VIF’s design coherent as you move from documentation to code and experiments.
