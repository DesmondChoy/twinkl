# Lukas Coach Digest voice pilot — 9 September 2026

Only Lukas’s week of 16–22 June 2025 was refreshed: one of the 22 saved comparison pairs. The two responses now begin with the situation involving Sandra, use an earlier remembered action without a dated timeline, and ask one short question. All other pair objects remain identical to the preserved baseline. The other four exported Persona files change only their source-manifest hashes because the shared comparison fixture changed.

## Method and adoption boundary

Generation retained `gpt-5.6-luna`, reasoning effort `none`, and the existing bounded runner budget. Coach prompt `4.5` and comparison extension `1.1` add situation-led openings, direct language, natural chronology, integrated quotations, and repair guidance that preserves those choices. Archived `4.4`/`1.0` templates preserve historical validation. New narrow checks reject date-led openings and writing-process commentary; they do not establish factual fidelity or overall writing quality.

Both arms have identical initial instructions and identical final repair requirements. Their initial data differ only in `north_star_context`. The Weekly Drift Detection input hash, North Star Moment input hash, selected context, and context hash match the prior saved pair. The response text is exact provider output, not hand-edited. The saved question is identical in both arms; independent generation does not require every field to differ. Replay tests still verify each exact response, prompt, source binding, and provider receipt.

Earlier pilot responses passed deterministic checks but were rejected by AI editorial review for repetitive assessment language, unsupported shared motives, invented meeting context, or incorrect chronology. The first revision also put one new guidance paragraph inside the template’s data section, so request assembly omitted it. That paragraph was moved into trusted instructions; a regression assertion now checks the rendered request. Every trial retains its source snapshot, request, raw output, diagnostics, and usage. These are development iterations under an unadopted prompt revision, not independent evaluation runs.

The selected lineage consists of `coach_voice_pilot_verified_20260909`, `coach_voice_pilot_reviewed_20260909`, and this directory. Its two bounded repairs made six calls in total. Final AI source review found no blocking factual issue. Minor limitations remain: the with-context opening unnecessarily emphasizes that the roadmap review is still ahead; both use a generic sentence about recent choices; the baseline infers retrospective preference from expressed regret and the confirmed Profile. This is a limited local pilot, not human validation, evidence of user benefit, or approval for a roster-wide refresh. The repaired outputs do not establish that the shared prompt alone reliably achieves this quality.

## Exact selected responses

### Without North Star Moment

When you were with Sandra, you considered her seniority and changed the subject rather than speaking up. You wrote, "Could have said something. Should have, maybe."

Making the world a fairer, better place matters to you, yet it has not shaped several recent choices in the way you would have wanted. Earlier, you helped Amira untangle the intake form, showing that this value also has a place in how you spend an ordinary evening.

What felt at stake for you when you were deciding whether to speak up with Sandra?

### With North Star Moment

When you were with Sandra, you considered her seniority and changed the subject, leaving the Q3 roadmap review still ahead of you. You wrote, "Could have said something. Should have, maybe."

Wanting to make the world a fairer, better place has not shaped several recent choices, including this one. Earlier, you helped Amira debug the intake form at the nonprofit, a practical way of supporting someone who was stuck.

What felt at stake for you when you were deciding whether to speak up with Sandra?

## API usage

Each row counts diagnostics created in that directory only, avoiding double-counting inherited attempts. Costs use the repository’s recorded rate calculation. No seed was supplied; frozen inputs and receipts support audit, not exact stochastic regeneration.

| Run directory | New calls | Calculated USD |
| --- | ---: | ---: |
| `coach_voice_pilot_20260909` | 2 | 0.00113155 |
| `coach_voice_pilot_current_20260909` | 2 | 0.00114391 |
| `coach_voice_pilot_final_20260909` | 2 | 0.00136384 |
| `coach_voice_pilot_focused_20260909` | 5 | 0.00306545 |
| `coach_voice_pilot_repair_20260909` | 2 | 0.00128458 |
| `coach_voice_pilot_reviewed_20260909` | 2 | 0.00125650 |
| `coach_voice_pilot_selected_20260909` | 2 | 0.00125164 |
| `coach_voice_pilot_verified_20260909` | 2 | 0.00120046 |

Total: **19 calls**, **US$0.01169793**. The selected lineage cost US$0.00370860.

## Reproduction and verification

Run from the repository root after activating the virtual environment. Paid generation requires the authorized provider environment. The final apply must include its prior-run and editorial-repair arguments so the frozen plan remains identical.

```sh
source .venv/bin/activate
export UV_CACHE_DIR=/tmp/twinkl-uv-cache
export MPLCONFIGDIR=/tmp/twinkl-mpl
uv run python -m scripts.coach.compare_scenario_coach --output logs/experiments/reports/coach_voice_pilot_verified_20260909 --case persistent-lukas::2025-06-16 --execute
uv run python -m scripts.coach.compare_scenario_coach --output logs/experiments/reports/coach_voice_pilot_reviewed_20260909 --prior-run logs/experiments/reports/coach_voice_pilot_verified_20260909 --editorial-repairs logs/experiments/reports/coach_voice_pilot_verified_20260909/editorial_repairs.json --execute
uv run python -m scripts.coach.compare_scenario_coach --output logs/experiments/reports/coach_voice_pilot_selected_20260909 --prior-run logs/experiments/reports/coach_voice_pilot_reviewed_20260909 --editorial-repairs logs/experiments/reports/coach_voice_pilot_reviewed_20260909/editorial_repairs.json --execute
uv run python -m scripts.coach.compare_scenario_coach --output logs/experiments/reports/coach_voice_pilot_selected_20260909 --prior-run logs/experiments/reports/coach_voice_pilot_reviewed_20260909 --editorial-repairs logs/experiments/reports/coach_voice_pilot_reviewed_20260909/editorial_repairs.json --apply
uv run python -m src.demo.scenarios
uv run python -m src.demo.export_contract_schema
uv run pytest tests/coach tests/demo -q -o addopts=''
```

The frozen generation plans describe pre-application inputs. After exporting updated scenarios, a generation resume can correctly reject changed source hashes; use the preserved source snapshots to reconstruct the original environment. Do not override that guard.

Validation completed:

- 282 Coach/demo tests passed, including all saved replay pairs, exact prompt documentation, source-bound export, repair/resume, and partial-apply preservation.
- Scoped Ruff passed for changed Python implementation and tests.
- 366 frontend tests passed; TypeScript and Vite build passed. The existing large-bundle advisory remains.
- MyPy 2.3.0 with silent imported-module reporting found six existing Polars union errors in `weekly_digest.py`, identical on the pre-change HEAD baseline; the other changed implementation modules passed the isolated check. Type checking is not fully green. The broader imported graph also retains baseline errors.
- Browser review at the local Vite preview confirmed both new responses, the exact original North Star Moment quotation and source link, and both prompt/response receipts in Inspect. Reloading was required to replace the already-loaded scenario. The earlier no-selection explanation behavior remains covered by frontend tests and the prior browser check.
- Comparison against the preserved fixture confirms only the selected pair changed, with unchanged evidence and selected context. Other Persona export changes are source-manifest hashes only.
- `git diff --check` passed. Full repository tests were not run. No commit, push, tracker synchronization, or deployment was performed.
