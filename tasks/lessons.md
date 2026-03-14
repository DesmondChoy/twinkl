# Lessons

## 2026-03-11

- When promoting a script out of a notebook path, avoid carrying forward
  "notebook-derived" wording into the canonical name or docs unless the user
  explicitly wants that provenance preserved. Prefer the clearest operational
  label for fresh-session discoverability.
- When a new training lever is intended to drive the active frontier path,
  default it on in that path instead of relying on a per-run opt-in flag that
  can be forgotten. Keep low-level APIs explicit, but avoid operational
  footguns in the primary experiment flow.
- In experiment reviews, separate the intervention recommendation from the
  paper citation used to justify it. Reuse the repo's actual tooling and open
  issue path first, and only cite external work that matches this setup's
  class count, training regime, and model architecture closely enough to avoid
  misleading extrapolation.
- For this repo's Git workflow, stay on the current branch unless the user
  explicitly asks for a new one. If branch isolation seems helpful, ask first
  instead of creating or switching branches unilaterally.
- Before recommending a "next experiment," explicitly verify that it is not
  already completed, de-scoped, or closed in `logs/experiments/index.md`,
  recent report docs, and `bd`. Do not re-suggest a finished experiment as
  if it were untested.
- In this VIF stack, be precise about what counts as "representation" versus
  "head" changes: the sentence encoder is already frozen and the state encoder
  is deterministic, so "head-only retraining" means changing the MLP critic
  layers, not toggling an encoder-freeze path that does not exist.
- Before recommending a "next experiment," verify from the latest report, open
  issues, and close reasons that it has not already been run, explicitly
  de-scoped, or superseded. Treat stale recommendation text in older reports as
  historical context, not current guidance.
