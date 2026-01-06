# Annotation Guidelines: Nudge Effectiveness Study

## Objective

Determine whether nudging improves signal quality for VIF training.

**The question we're answering**: Do journal entries with nudge follow-ups provide more useful training data than entries without?

---

## What You're Labeling

Each sample is either:
- **A standalone journal entry** (no nudge)
- **A journal entry + nudge + response** (nudged session)

Your job: Label each sample as **Scorable** or **Not Scorable**.

---

## The One Question to Ask

> "Could I tell a friend what values this person is honoring or compromising based on this entry?"

- **Yes** → Mark as **Scorable**
- **No / I'd be guessing** → Mark as **Not Scorable**

---

## Examples

### Scorable

| Entry | Why |
|-------|-----|
| "Said yes to covering Jake's shift again even though I had plans. Whatever." | Clear action, clear trade-off, hints at how they feel about it |
| "Finally told Sarah I couldn't keep lending her money. She wasn't happy but I feel lighter." | Concrete decision, emotional response, self-reflection |
| "Skipped the gym to help my sister move. Didn't even think twice." | Action + implicit priority (family over self) |

### Not Scorable

| Entry | Why |
|-------|-----|
| "Rough day." | No information about what happened or how they feel |
| "Work was busy. Tired now." | Generic, no values visible |
| "Feeling off lately." | Abstract emotion, no behavioral content |

---

## For Nudged Sessions

When a session includes a nudge and response, read the **whole exchange** before labeling.

**Example nudged session:**

> **Entry**: "Meeting went fine, I guess."
>
> **Nudge**: "The 'I guess' part?"
>
> **Response**: "I didn't push back when they shot down my idea. Felt like I should have said something but didn't."

Label based on the combined content. This session is **Scorable** — the response reveals a value tension (speaking up vs. going along).

---

## Process

1. Read the entry (and response if present)
2. Ask: "Can I identify what values are at play?"
3. Label: Scorable or Not Scorable
4. Move on — don't overthink it

---

## Output Format

Record your labels in a spreadsheet or CSV:

| id | group | scorable |
|----|-------|----------|
| 1 | nudged | yes |
| 2 | no_nudge | no |
| 3 | nudged | yes |
| ... | ... | ... |

---

## Success Criteria

After labeling all samples, compare:

- **% Scorable in nudged group**
- **% Scorable in no-nudge group**

If nudged entries are meaningfully more scorable → nudges improve signal quality.
If similar → nudges add complexity without benefit.
