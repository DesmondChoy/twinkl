# Judge Labeling Validation Report

## Run Summary
- **Personas processed:** 5/5
- **Entries labeled:** 37/37
- **Timestamp:** 2026-01-14_23-03-35

## Score Distribution

| Value Dimension | -1 | 0 | +1 |
|-----------------|-----|-----|-----|
| Self-Direction  | 14 | 19 | 4 |
| Stimulation     | 1 | 36 | 0 |
| Hedonism        | 6 | 24 | 7 |
| Achievement     | 2 | 23 | 12 |
| Power           | 7 | 23 | 7 |
| Security        | 1 | 16 | 20 |
| Conformity      | 2 | 14 | 21 |
| Tradition       | 0 | 26 | 11 |
| Benevolence     | 2 | 12 | 23 |
| Universalism    | 1 | 36 | 0 |

## Quality Flags

### All-Zero Entries
- Count: 0 (0%)
- Note: All entries have at least one non-zero score

### Sparse Dimensions
- **Stimulation:** 97% neutral (36/37) - minimal signal
- **Universalism:** 97% neutral (36/37) - minimal signal

### Active Dimensions
- **Benevolence:** 68% non-neutral (25/37) - strongest signal
- **Conformity:** 62% non-neutral (23/37)
- **Security:** 57% non-neutral (21/37)

## Sample Labels (First 3 Entries)

### Persona 1, Entry 0 (2025-12-01)
Gabriela defends her choice to stay home, asserting control over household.
```json
{"self_direction": 1, "achievement": 1, "power": 1, "conformity": -1}
```

### Persona 1, Entry 1 (2025-12-09)
Parents' committee meeting, prefers husband's absence for control.
```json
{"achievement": 1, "power": 1, "security": 1}
```

### Persona 1, Entry 2 (2025-12-11)
Comparing herself to sister Lucia, unable to relax.
```json
{"hedonism": -1, "achievement": 1, "power": 1, "security": 1}
```

## Persona-Level Patterns

| Persona | Dominant Alignments | Dominant Misalignments |
|---------|---------------------|------------------------|
| Gabriela (Power) | Power +6, Achievement +6 | Conformity -1, Universalism -1 |
| Wei-Lin (Power, Self-Direction) | Conformity +2, Achievement +2 | Self-Direction -2, Power -2 |
| Tarek (Benevolence) | Benevolence +9, Tradition +6 | Self-Direction -3, Hedonism -3 |
| Marcus (Conformity) | Conformity +8, Security +8 | Self-Direction -4, Power -4 |
| Karen (Conformity) | Benevolence +7, Conformity +7 | Self-Direction -4, Power -2 |

## Notes
- Stimulation and Universalism show very sparse signals - may need more targeted journal prompts
- Security and Conformity show high alignment across personas - may reflect synthetic data generation bias toward routine/stable contexts
- Self-Direction frequently misaligned - consistent with personas facing constraints on autonomy
