# Canonical Nouns and Communication Rules

Use this document for Twinkl product discussions, explanations, plans, issues,
reports, and maintained documentation. The [PRD](prd.md) remains authoritative
for product behavior; this document standardizes how that behavior is named.

These rules govern prose. They do not require renaming code identifiers, data
fields, file paths, or historical records.

## How to Use This Glossary

- Use the canonical noun whenever its definition applies.
- Do not invent synonyms for variety.
- Name the actual component, data, experiment setup, or output instead of
  using an umbrella word.
- Add technical jargon only when a real distinction requires it. Define it in
  plain English on first use.

## Canonical Nouns

| Noun | Definition | Avoid in prose |
|---|---|---|
| **Journal Entry** | One chronological journal entry written by the user. When the evaluated text also contains a displayed nudge and response, say so explicitly. | runtime state, trajectory cell |
| **Core Value** | A value explicitly selected by the user. Only Core Values can produce Drift. | `top_values` or declared-core set outside code-level discussion |
| **LLM-Judge** | The offline LLM that creates training or reference labels. It is not the VIF Critic, Weekly Drift Reviewer, or a human reviewer. | Judge, teacher, oracle, labeling system |
| **LLM-Judge VIF Label** | An LLM-Judge label of `-1`, `0`, or `+1` for one Journal Entry and value, created to train or evaluate the VIF Critic. | older label, legacy label, LLM-Judge output |
| **LLM-Judge Conflict Label** | An LLM-Judge reference label of **Conflict**, **Not Conflict**, or **Uncertain** for one Journal Entry and Core Value under the current displayed-behavior definition. A resolved label is **Conflict** or **Not Conflict**; **Uncertain** remains unresolved. | newer label, resolved judge label when resolution is not the distinction, LLM-Judge output |
| **VIF Critic** | The trained model, currently an MLP, that predicts `-1`, `0`, or `+1` for each Journal Entry and value, plus uncertainty. | system, scorer, or model when the VIF Critic is specifically meant |
| **VIF Critic Prediction** | The VIF Critic's predicted `-1`, `0`, or `+1` for one Journal Entry and value, plus uncertainty. | VIF Critic output, score when prediction is meant |
| **Conflict (`-1`)** | One Journal Entry that clearly shows behavior or a choice against one value. When the source matters, name the LLM-Judge VIF Label, LLM-Judge Conflict Label, VIF Critic Prediction, or Weekly Drift Reviewer Decision. | negative evidence, misalignment signal |
| **Drift** | Two consecutive Conflicts for the same Core Value. A longer uninterrupted run is still one Drift. | sustained-conflict episode, meaningful two-entry pattern, reference event |
| **Weekly Drift Reviewer** | The LLM that reviews weekly Journal Entries for Conflict. The fixed model contract is `gpt-5.6-luna` with reasoning effort `low`, without VIF Critic predictions. Historical experiments used other setups. It is not the VIF Critic. | weekly verifier, LLM arm, system |
| **Weekly Drift Reviewer Decision** | The Weekly Drift Reviewer's decision of **Conflict**, **Not Conflict**, or **Abstain** for one Journal Entry and Core Value. | reviewer output, weekly label |
| **Drift Detector** | The deterministic rule that decides whether two consecutive Weekly Drift Reviewer Conflicts for the same Core Value form Drift. The approved rule is not wired or deployment-approved yet. | trigger layer, episode engine |
| **Weekly Digest** | The structured weekly record containing values, evidence, Drift state, and inputs for the Weekly Coach. | artifact, payload, packet |
| **Weekly Coach** | The component that turns the Weekly Digest into the user-facing reflection and question. | narrative layer, delivery engine |

Use **resolved** only when distinguishing a **Conflict** or **Not Conflict**
LLM-Judge Conflict Label from an **Uncertain** one. Use **consecutive** to
describe the relationship between two labels; it is not part of either label's
name.

## Use Concrete Experiment and Data Names

Do not replace one ambiguous experiment word with another.

- Instead of **arm** or **condition**, name the exact experiment setup, such
  as **Weekly Drift Reviewer without VIF Critic input** and **Weekly Drift
  Reviewer with VIF Critic input**.
- An **experiment setup** is the exact model and inputs being compared. A
  **run** or **repeat** is one execution of that setup.
- Instead of **surface**, **population**, or a bare **set**, name the data's
  role: **development set**, **final test set**, **42 reviewed cases**, or
  another concrete description.
- Use **final test set** instead of **promotion surface**.
- Use **deployment approval** instead of **promotion** when discussing whether
  a model is ready for use.
- Replace **artifact** with the actual output: report, labels, JSON file,
  checkpoint, chart, or another specific name.
- Replace **system**, **pipeline**, **scorer**, and **candidate** with the exact
  component, workflow, model, or experiment setup whenever possible.
- Use **Drift recall** instead of **episode recall**, and **false Drift alert**
  instead of **false episode**.

## Example

Avoid:

> On the development surface, the system's median episode recall was `0.40`.

Prefer:

> On the development set, the Weekly Drift Reviewer without VIF Critic input
> found two of five known Drifts, so its median Drift recall was `0.40`.
