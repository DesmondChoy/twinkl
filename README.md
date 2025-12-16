# Twinkl

Twinkl is a voice-first "inner compass" that helps users align their daily behavior with their long-term values. Unlike traditional journaling apps that summarize moods and topics, Twinkl maintains a dynamic self-model of the user's declared priorities and surfaces tensions when behavior drifts from intent. It answers the question: *"Am I living in line with what I said I value?"*

## Value Identity Function (VIF)

The VIF is Twinkl's core evaluative engine. It compares what users *do* (daily journal entries) against what they *value* (their declared priorities) across multiple dimensions like Health, Relationships, and Growth.

Key properties:
- **Vector-valued**: Tracks multiple life dimensions simultaneously, preserving trade-offs (e.g., "work goals crushed, but sleep suffered")
- **Uncertainty-aware**: Holds back judgment when the situation is complex or data is sparse
- **Trajectory-aware**: Detects patterns over time rather than reacting to single entries

When the VIF detects significant misalignment with high confidence, it triggers the Coach layer to surface evidence-based feedback. See `docs/VIF/` for architecture details.

## Synthetic Data Generation

Bootstraps training data for value tagging and reward modeling. Generates realistic, longitudinal journal entries from synthetic personas with known Schwartz value profiles.

Key features:
- Personas with 1-2 assigned values expressed through concrete life details (not labels)
- Longitudinal entries that exhibit value drift, conflicts, and ambiguity
- Parallel async pipeline for efficient generation
- Configurable tone, verbosity, and reflection mode per entry

See `docs/Ideas/Synthetic_data.md` and `notebooks/journal_gen.ipynb` for implementation.

# Setup

This repo uses `uv` and `pyproject.toml` for dependency management.

1. Install `uv` (see https://docs.astral.sh/uv/).
2. Create the virtual environment:
   ```sh
   uv venv
   ```
3. Activate it (Fish shell):
   ```sh
   source .venv/bin/activate.fish
   ```
4. Create a `.env` file in the project root with your OpenAI API key:
   ```sh
   OPENAI_API_KEY=your-api-key-here
   ```

## Installing dependencies

Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`.

- Install everything from the lockfile:
  ```sh
  uv sync
  ```
- If/when a dev group is added later:
  ```sh
  uv sync --dev
  ```

## Adding a dependency

Use `uv add` to both install into the environment and record it in
`pyproject.toml`:

```sh
uv add <package>
```

Pin an exact version if desired:

```sh
uv add "<package>==<version>"
```

After adding, `uv` updates `uv.lock` automatically.

## Exporting requirements.txt (optional)

Only needed for legacy tooling or platforms that require it:

```sh
uv export -o requirements.txt
```
