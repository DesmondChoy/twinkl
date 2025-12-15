# VIF System Architecture Diagrams

This document provides visual system architecture diagrams that complement the existing VIF documentation:

- `VIF_01_Concepts_and_Roadmap.md`
- `VIF_02_System_Architecture.md`
- `VIF_03_Model_Training.md`
- `VIF_04_Uncertainty_Logic.md`
- `VIF_Example.md`

The diagrams are descriptive, not prescriptive code; they are meant to help readers quickly understand how the offline training pipeline, online inference loop, and Critic–Coach interaction fit together.

---

## 1. High-Level VIF Ecosystem

This diagram shows the main components and how they relate over offline training and online usage.

```mermaid
flowchart LR
  subgraph offline[Offline_Training]
    generator[Generator]
    judge[Judge_LLM]
    criticTrain[Critic_Training]
  end

  subgraph online[Online_Inference]
    userApp[Twinkl_App]
    critic[VIF_Critic]
    coach[Coach_Agent]
    historyStore[User_History_Store]
  end

  generator -->|synthetic_journals| judge
  judge -->|"labeled_alignment_vectors"| criticTrain
  criticTrain -->|trained_model| critic

  userApp -->|journals,voice| critic
  critic -->|alignment_scores+uncertainty| historyStore
  historyStore --> coach
  critic -->|"triggers(crash/rut/uncertainty)"| coach
  coach -->|digests,reflections| userApp
```

**Key idea:**
- The **Generator–Judge–Critic** loop runs offline to produce a trained Critic.
- The **Critic–Coach–UserApp** loop runs online for real users.

---

## 2. Offline Training: Generator–Judge–Critic

This diagram expands the offline pipeline described in `VIF_03_Model_Training.md` and `docs/Ideas/Synthetic_data.md`.

```mermaid
flowchart TD
  config["Config_Files (schwartz_values,synthetic_data)"]
  personaGen[Persona_Generator]
  journalGen[Journal_Generator]
  judgeLLM[Judge_LLM]
  dataset[Training_Dataset]
  criticModel[Critic_MLP]

  config --> personaGen
  personaGen --> journalGen
  journalGen -->|synthetic_journals| judgeLLM
  judgeLLM -->|"alignment_labels(-1,0,1)^K"| dataset
  dataset --> criticModel
  criticModel -->|trained_critic| criticModel
```

**Notes:**
- `Config_Files` includes `config/schwartz_values.yaml` and `config/synthetic_data.yaml`.
- The Judge turns rich LLM reasoning into discrete alignment vectors; the Critic distills this into a fast numeric model.

---

## 3. Online Inference: State Construction & Critic Flow

This diagram summarises Section 2 of `VIF_02_System_Architecture.md`.

```mermaid
flowchart TD
  userInput[User_Entry]
  textEnc[Text_Encoder]
  audioEnc[Audio_Encoder]
  physioEnc[Physio_Encoder]
  profile[z_u_Profile]
  historyStats[History_Stats]
  stateBuilder[State_Builder]
  critic[VIF_Critic]
  weeklyAgg[Weekly_Aggregator]
  triggerLogic[Trigger_Logic]

  userInput --> textEnc
  userInput --> audioEnc
  userInput --> physioEnc

  textEnc --> stateBuilder
  audioEnc --> stateBuilder
  physioEnc --> stateBuilder
  profile --> stateBuilder
  historyStats --> stateBuilder

  stateBuilder --> critic
  critic -->|alignment_means+variances| weeklyAgg
  weeklyAgg --> triggerLogic
```

**Notes:**
- `State_Builder` implements the sliding window design (current + recent entries, time gaps, history statistics, profile).
- `Trigger_Logic` is where crash/rut and uncertainty thresholds are applied.

---

## 4. Critic vs Coach Separation

This diagram emphasizes the separation of numeric evaluation (Critic) and explanation (Coach), as described in `VIF_02_System_Architecture.md` and `VIF_Example.md`.

```mermaid
flowchart LR
  subgraph criticSide[Critic_Path]
    state[s_u_t]
    critic[VIF_Critic]
    weeklyAgg[Weekly_Aggregator]
    triggers[Crash_Rut_Triggers]
  end

  subgraph coachSide[Coach_Path]
    retriever[Journal_Retriever]
    explainer[Coach_LLM]
  end

  state --> critic --> weeklyAgg --> triggers
  triggers -->|dimension,pattern,confidence| retriever
  retriever -->|evidence_snippets| explainer
  explainer -->|message| userOut[User_View]
```

**Key properties:**
- The **Critic** only uses recent sequential state and outputs numeric signals.
- The **Coach** pulls from the full history using retrieval and turns signals into human‑friendly reflections.

---

## 5. Uncertainty & Dual-Trigger Logic

This diagram captures the logic from `VIF_04_Uncertainty_Logic.md`.

```mermaid
flowchart TD
  critic["VIF_Critic (MC_Dropout)"]
  mcSamples[MC_Samples]
  stats[Mean_And_Variance]
  weeklyAgg[Weekly_Aggregator]
  crashCheck[Crash_Check]
  rutCheck[Rut_Check]
  uncertaintyGate[Uncertainty_Gate]
  decision[Decision]

  critic --> mcSamples
  mcSamples --> stats
  stats --> weeklyAgg

  weeklyAgg --> crashCheck
  weeklyAgg --> rutCheck

  crashCheck --> uncertaintyGate
  rutCheck --> uncertaintyGate
  stats --> uncertaintyGate

  uncertaintyGate --> decision
```

**Outcomes at `Decision`:**
- **Critique**: crash or rut is detected and uncertainty is below threshold.
- **Clarifying_Question**: patterns are unclear and uncertainty is high.
- **No_Action**: no significant pattern or user has been contacted recently.

---

## 6. Sarah’s Journey View (Lifecycle Overview)

Finally, this diagram ties the stages from `VIF_Example.md` together.

```mermaid
flowchart LR
  offline[Offline_Training]
  onboard[Onboarding]
  stable[Stable_Alignment]
  crash[Crash_Week]
  rut[Rut_Weeks]
  grief[High_Uncertainty_Grief]

  offline --> onboard --> stable --> crash --> rut --> grief

```

**Interpretation:**
- The **components** (Generator, Judge, Critic, Coach) stay the same; what changes by stage is **which ones are active** and how the triggers fire.
- This mirrors the table at the end of `VIF_Example.md` and can be used in slides to explain the system lifecycle.
