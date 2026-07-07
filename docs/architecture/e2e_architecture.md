# Twinkl E2E Architecture

This is the high-level product/system map. It intentionally sits outside
`docs/vif/` because the end-to-end story is broader than the VIF Critic.
For the detailed training/runtime dataflow, see
[docs/vif/current_system_architecture.mmd](../vif/current_system_architecture.mmd).

Status legend (node colors):

- **Implemented** (green): working repo capability
- **Partial / experimental** (amber): working slice, not ready to claim as product behavior
- **Specified** (blue): documented, not wired into the active runtime
- **??? Decision** (dashed grey): team decision or ambiguity to resolve

Solid arrows are paths that are wired in the repo today. Dashed arrows are
benchmark, intended, or undecided connections.

```mermaid
%%{init: {
  'theme': 'base',
  'themeVariables': {
    'fontFamily': 'ui-sans-serif, system-ui, sans-serif',
    'lineColor': '#5B6B88',
    'edgeLabelBackground': '#F8F8F6',
    'clusterBkg': '#F8F8F6',
    'clusterBorder': '#9CA39A'
  }
}}%%
flowchart TB
    classDef implemented fill:#E8F2F0,stroke:#3F7C73,color:#16302D;
    classDef partial fill:#FFF4D9,stroke:#B38600,color:#3B2C00;
    classDef specified fill:#EAF0FF,stroke:#4767B3,color:#15223E;
    classDef decision fill:#F8F8F6,stroke:#9CA39A,color:#36413B,stroke-dasharray: 4 4;

    subgraph training["Training core"]
        personas["Synthetic personas"]
        judge["LLM Judge labels"]
        annotation["Human labels<br/>agreement benchmark"]
        critic_train["Critic training +<br/>experiment archive"]
        checkpoint["Selected Critic checkpoint"]
    end

    subgraph product["Product shell"]
        onboarding["Values onboarding survey<br/>(BWS)"]
        profile["Value profile<br/>(self-model)"]
        journaling["Journaling UI"]
        nudges["Conversational nudging"]
        d_surface["??? Where does the<br/>product ship?"]
    end

    subgraph runtime["Scoring runtime"]
        state["Runtime state builder<br/>(entry + profile → Critic input)"]
        scores["VIF Critic scores<br/>+ uncertainty"]
        weekly["Weekly aggregation"]
        drift["Drift detection"]
        d_trigger["??? When to trust<br/>drift alerts?"]
        d_evolution["??? Value evolution:<br/>in scope or future work?"]
    end

    subgraph coach["Coach + review"]
        digest["Weekly digest"]
        narrative["Coach prompt / narrative"]
        review["Internal review app +<br/>evaluation reports"]
        d_boundary["??? What is the Coach<br/>allowed to do?"]
        d_feedback["??? Does user feedback<br/>update the profile?"]
    end

    %% Training flow (wired)
    personas --> judge --> critic_train --> checkpoint

    %% Human benchmark, not label production
    annotation -. "benchmark comparison" .-> review

    %% Product shell (specified, not wired to real users yet)
    onboarding -.-> profile
    journaling <-.-> nudges
    journaling -.-> state
    profile -. "??? How onboarding scores<br/>become the runtime profile" .-> state

    %% What actually feeds the runtime today
    personas -->|"synthetic journals stand in<br/>for real users today"| state

    %% Scoring runtime (wired, experimental)
    checkpoint --> scores
    state --> scores --> weekly --> drift

    %% Coach + review (wired, experimental)
    drift --> digest
    weekly --> digest
    digest --> narrative
    digest --> review

    %% Open decisions attached to where they bite
    journaling -.- d_surface
    drift -.- d_trigger
    weekly -.- d_evolution
    narrative -.- d_boundary
    narrative -.- d_feedback

    class personas,judge,annotation,critic_train,checkpoint implemented;
    class state,scores,weekly,drift,digest,narrative,review,nudges partial;
    class onboarding,profile,journaling specified;
    class d_surface,d_trigger,d_evolution,d_boundary,d_feedback decision;
```

## Read This As

The dashed grey `???` nodes and edge labels mark team decisions that still
need calls. Read this as a product/system map, not a literal runtime sequence.

Twinkl's proven spine runs top to bottom: generated and judged data trains a
Critic, and a trained checkpoint then scores each journal entry, rolls the
scores up into weekly signals, flags drift, and packages everything into a
weekly digest the Coach can narrate. Today that spine runs on synthetic
persona journals, which stand in for real user journals — that is the solid
edge from the training core into the runtime.

The product shell is designed on paper but not built: where the product ships
(app, web, something else), the journaling UI itself, and how a user's
onboarding answers get turned into the value profile the runtime reads. One
exception inside it: the conversational nudging engine already exists as an
experimental slice, even though the journaling UI it would attach to does not.

The remaining open decisions are when drift alerts are reliable enough to act
on, what the Coach is allowed to do or say, whether user feedback should
update the profile over time, and whether telling genuine value change apart
from behavioral drift ("value evolution") is in scope now or left for future
work.
