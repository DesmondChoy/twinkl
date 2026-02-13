# One-Word Commands
Quick shortcuts for common tasks:

- `$craft`: Generate high-quality conventional commit messages for this session’s changes (do not commit; user reviews first).
  - Behavior:
    - Inspect staged/unstaged changes and summarize what changed and why.
    - Always propose a single commit message combining all changes.
  - Output format (no extra prose; emit only commit message text in code fences):
    - Single commit:
      ```
      <type>(<scope>): <summary>
      
      <body>
      
      - <bullet describing change>
      - <bullet describing change>
      
      Affected: <file1>, <file2>, ...
      Test Plan:
      - <how you verified>
      Revert plan:
      - <how to undo safely>
      ```

  - Allowed types: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert.
  - Conventions:
    - Subject ≤ 50 chars, imperative mood; wrap body at ~72 chars.
    - Use BREAKING CHANGE: in body when applicable.
    - Add Refs:/Closes: lines for issues/PRs when available.
  - If context is missing, ask one concise question; otherwise proceed with best assumption and note it in the body.
- `$parallel-x`: Run x sub-agents in parallel (not sequentially) where x is the number specified.

## Commands
- **Virtual Environment**: ALWAYS activate `source .venv/bin/activate.fish` before running Python commands
- **Dependency Workflow**: Use `uv` (`uv sync`, `uv add`, `uv pip install`) instead of plain `pip`

## Documentation
- **Source of Truth**: `docs/prd.md` is the definitive specification
- **Other docs/**: Brainstorming ideas and features under consideration (not authoritative)
- **Architecture Guide**: `CLAUDE.md` is the detailed project execution playbook and should stay aligned with this file

## Code Style
- **Imports**: Standard library first, then third-party (streamlit, cv2, numpy, etc.)
- **Naming**: snake_case for variables/functions, PascalCase for classes (VideoTransformer)
- **Style**: Clean, readable code with good spacing and comments
- **Implementation Notes**: After every code change, report in chat whether the solution feels over-engineered for the academic, time-boxed POC scope. Also comment if there are any simpler alternatives or noted gaps. Keep this as a conversational summary rather than an inline code comment.

## Operating Principles
- Keep solutions scoped to the academic, time-boxed POC; avoid heavy abstractions unless justified
- Prefer small, testable iterations over broad rewrites
- Preserve registry pipeline stage semantics (`stage_synthetic`, `stage_wrangled`, `stage_labeled`) when touching pipeline code
- Maintain value-leakage safeguards in synthetic data generation and prompt logic

## Quality Expectations
- Before running `git commit`, run `.claude/skills/quality/SKILL.md`
- Review complete changed files before commit, not only patch hunks
- Run targeted tests/lint checks for touched modules before proposing commit text
- Remove debug artifacts and obvious dead code before handoff

Use 'bd' for task tracking
