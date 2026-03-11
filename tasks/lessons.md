# Lessons

## 2026-03-11

- When a new training lever is intended to drive the active frontier path,
  default it on in that path instead of relying on a per-run opt-in flag that
  can be forgotten. Keep low-level APIs explicit, but avoid operational
  footguns in the primary experiment flow.
