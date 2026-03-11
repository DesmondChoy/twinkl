# Lessons

## 2026-03-11

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
