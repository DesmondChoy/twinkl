# Saved Persona Coach Digest comparison — 9 September 2026

## Scope and evidence

This implementation adds a saved comparison to the five frontend demo Personas.
For each of the 22 weeks with an accepted North Star Moment, it generates two
complete Coach Digest responses from the same weekly input: one without selected
source context and one with it. The 27-week roster contains five weeks without
a selection; those keep their existing Coach Digest and a disabled comparison
button. The selection records and frozen selection experiment are unchanged.
Beads: `twinkl-rklc.48`.

The Experience toggle replaces both narrative paragraphs and the reflective
question, and shows or hides the original exact source panel. Inspect records
both prompts with their associated structured and raw responses. Initial
instructions are identical between arms, with only `north_star_context` changing
in the JSON data. Later repair instructions can differ and are exposed alongside
the accepted prompt. Onboarding from scratch keeps its existing behavior.

These are AI-generated responses to synthetic Persona writing. The source review
described below is AI editorial review, not human validation or a new Coach
Digest Eval. Neither the wording differences nor the toggle establish user
benefit or a causal effect of the selected context.

## Inputs and generation contract

The source roster is Nisha (five eligible weeks), Noor (six), Lukas (four), Wei
Jun (three), and Meera (four). The accepted selections comprise 17 encouragement,
four reflection, and one reminder; 21 quote Journal Entries and one quotes a
nudge response. A nudge response remains distinct from its parent Journal Entry.

The common instructions combine Coach prompt `4.4` and the demo comparison
extension `1.0`. The model is OpenAI `gpt-5.6-luna`, reasoning effort `none`, service
tier `default`, with a 2,048-token output limit and `store: false`. No seed is set.
Each Persona is processed sequentially by week, with Personas processed in
parallel. Both arms retain the initial and accepted prompt, raw and structured
output, deterministic checks, model settings, hashes, and every call receipt.

The exact documented instructions and both example JSON inputs were compared
with the implementation and matched. Full source text remains in untrusted JSON
data. Persona biographies, generation targets, assessment rationale, and
evaluation judgments are excluded from the Coach input.

The generation plans preserve file hashes and source snapshots. Each request
has a saved JSON receipt, each attempt has a diagnostic, and each accepted arm
is bound to its original input and response hashes. The original
`src/demo/coach_digest_responses.json` remains unchanged; paired responses are
stored separately in `src/demo/coach_digest_comparisons.json`.

## Generation and editorial repair

The initial run accepted ten arms. Its first repair run retained those ten and
reached 39 accepted arms across 83 cumulative provider calls, at a calculated
cost of US$0.04470574. Most deterministic failures concerned missing verbatim
weekly quotations: quoting only the added source did not satisfy the unchanged
weekly-evidence requirement. Retry instructions were made more explicit without
relaxing that check or changing the initial prompt.

AI editorial review compared all 39 accepted responses with their exact supplied
inputs. It identified nine material source or chronology problems despite the
deterministic passes:

| Response | Correction required |
| --- | --- |
| Nisha, 17 February, with context | Distinguish prior-week school and family events from the reviewed week. |
| Nisha, 24 February, without context | Remove the invented intended duration of the call with Didi. |
| Noor, 28 April, with context | Attribute laughter to the actual event, rather than a truncated quotation fragment. |
| Lukas, 16 June, with context | Make the earlier supportive action and the later workplace event temporally distinct. |
| Lukas, 7 July, without context | Attach the silence quotation to the earlier onboarding event, not the later coffee deliberation. |
| Wei Jun, 9 June, with context | Distinguish speech to the child from a private explanation, and avoid inventing a supportive meaning for unexplained silence. |
| Wei Jun, 9 June, without context | Describe silence without inventing an unkind occasion or supportive non-participation. |
| Wei Jun, 16 June, both arms | Do not infer an environmental cleanup purpose from a child's collection of mixed shells and plastic. |

The final repair preserves 30 accepted responses and replaces those nine while
completing the five outstanding arms. Every editorial rejection is bound to its
accepted response hash and recorded with its reason and repair requirements.
Original responses and provider receipts remain available. No generated
response is manually rewritten.

The next review found one additional causal attribution error in Nisha's
10 March treatment: the required weekly quotation was presented as the reason
for the later lesson plan, although the selected full source supplied a different
reason. A single hash-bound repair retained the other 43 responses and corrected
that attribution in one call. All 44 final responses were then covered by AI
editorial review against their exact supplied inputs, with no remaining material
source or chronology blocker identified. The retained 30 responses were also
checked against their earlier hashes. The final change retained the other 43
accepted arms unchanged.

The final per-week review covered both arms. The with-context checks below
identify the added source relationship; the without-context arm was checked
against its own weekly input without using the added source to fill gaps.

| Persona · week beginning | Added-source relationship checked | AI editorial outcome |
| --- | --- | --- |
| Nisha · 10 February | Fatima's 16 February interaction and the hour spent helping her. | Both pass. |
| Nisha · 17 February | Earlier 12/16 February experiences remain separate from Ravi's 23 February conversation. | Both pass. |
| Nisha · 24 February | Ravi's 23 February source precedes the 2 March Didi call. | Both pass. |
| Nisha · 3 March | Desk help remains earlier than the later choices and does not cancel Drift. | Both pass. |
| Nisha · 10 March | Earlier desk help is separate; Ravi's example motivates future lesson planning. | Both pass. |
| Noor · 14 April | Nighttime care, medicine knowledge, and the 5am handover match the source. | Both pass. |
| Noor · 21 April | The extension is requested, without inventing its approval or completed work. | Both pass. |
| Noor · 28 April | Dress choice is grounded; Sami's tug on the embroidery causes laughter. | Both pass. |
| Noor · 5 May | Napkin sketch, village customs, and earlier payment/park moments remain grounded. | Both pass. |
| Noor · 12 May | Accepted work remains to be fitted around existing demands. | Both pass. |
| Noor · 19 May | Conversation with Tariq and independent choices remain unresolved. | Both pass. |
| Lukas · 16 June | Explicit dates separate 10 June help from later repeated choices. | Both pass. |
| Lukas · 23 June | Historical help for Amira remains separate from the current pattern. | Both pass. |
| Lukas · 30 June | Amira is remembered alongside more recent silence. | Pass; baseline wording caveat below. |
| Lukas · 7 July | Conversation with Katharina is a specific action, without general recovery. | Both pass. |
| Wei Jun · 2 June | Letter drafting matches the source; earlier workplace withdrawal remains visible. | Both pass. |
| Wei Jun · 9 June | Speech to the child and private explanation remain distinct; unexplained silence stays unresolved. | Both pass. |
| Wei Jun · 16 June | Collecting is not assigned an invented environmental purpose. | Pass; treatment wording caveat below. |
| Meera · 17 November | Indigo work and family customs remain distinct from earlier constrained choices. | Both pass. |
| Meera · 24 November | Teaching and dye work match 30 November; no completed business outcome is invented. | Both pass. |
| Meera · 1 December | The selected nudge response remains distinct from its parent Journal Entry. | Pass; treatment emphasis caveat below. |
| Meera · 8 December | Order acceptance remains a beginning. | Pass; treatment attribution caveat below. |

| Run | New calls | Accepted arms available after run | Cumulative calculated cost (USD) |
| --- | ---: | ---: | ---: |
| Initial generation | 32 | 10 | 0.0161806 |
| Deterministic repair | 51 | 39 | 0.04470574 |
| Editorial repair and remaining generation | 20 | 44 | 0.05946405 |
| Final causal-attribution repair | 1 | 44 | 0.0603628 |

All 104 calls have usage receipts. Total calculated cost is **US$0.0603628**;
this is the runner's token-based calculation, not an invoice reconciliation.
The cumulative reservation was US$1.872, below the original US$3.168 bound.
The saved final comparison contains **22 pairs and 44 accepted responses**.
Every pair differs in both narrative paragraphs and its reflective question.

Final artifacts:

- [Final generation plan](../demo_coach_nsm_comparison_editorial_20260909/plan.json)
- [Final cumulative usage summary](../demo_coach_nsm_comparison_editorial_20260909/summary.json)
- [Nine initial editorial repair requirements](../demo_coach_nsm_comparison_final_20260909/editorial_repair_requirements.json)
- [Final causal-attribution repair requirements](../demo_coach_nsm_comparison_editorial_20260909/editorial_repair_requirements.json)
- [Installed paired responses](../../../../src/demo/coach_digest_comparisons.json)
- [Exact prompts and comparison specification](../../../../docs/north_star/demo_coach_comparison.md)

## Commands and verification

Activate the project environment and direct the `uv` cache to its writable
location before running the commands:

```sh
source .venv/bin/activate
export UV_CACHE_DIR=/tmp/twinkl-uv-cache
```

The executed generation sequence used the following output directories and
source-bound predecessors. Each run had a preceding dry run without execution
or apply flags. The first two commands stopped at their validation bounds before
applying anything. Each run preserved the earlier receipts; the successful
apply occurred only after AI source review. A reproduction requires the frozen
source versions recorded in the plans. Reapplying an old plan to changed scenario
bundles is rejected by the source-hash checks.

```sh
uv run python -m scripts.coach.compare_scenario_coach --execute --apply
uv run python -m scripts.coach.compare_scenario_coach --output logs/experiments/reports/demo_coach_nsm_comparison_repair_20260909 --prior-run logs/experiments/reports/demo_coach_nsm_comparison_20260909 --execute --apply
uv run python -m scripts.coach.compare_scenario_coach --output logs/experiments/reports/demo_coach_nsm_comparison_final_20260909 --prior-run logs/experiments/reports/demo_coach_nsm_comparison_repair_20260909 --editorial-repairs logs/experiments/reports/demo_coach_nsm_comparison_final_20260909/editorial_repair_requirements.json --execute
uv run python -m scripts.coach.compare_scenario_coach --output logs/experiments/reports/demo_coach_nsm_comparison_editorial_20260909 --prior-run logs/experiments/reports/demo_coach_nsm_comparison_final_20260909 --editorial-repairs logs/experiments/reports/demo_coach_nsm_comparison_editorial_20260909/editorial_repair_requirements.json --execute
uv run python -m scripts.coach.compare_scenario_coach --output logs/experiments/reports/demo_coach_nsm_comparison_editorial_20260909 --prior-run logs/experiments/reports/demo_coach_nsm_comparison_final_20260909 --editorial-repairs logs/experiments/reports/demo_coach_nsm_comparison_editorial_20260909/editorial_repair_requirements.json --apply
python -m src.demo.export_contract_schema
python -m src.demo.scenarios
```

Verification completed against the exported bundles:

| Check | Result |
| --- | --- |
| `uv run pytest tests/demo tests/coach tests/north_star -o addopts='' -q` | 551 passed; one stale packaging-test fixture omitted the new comparison file. |
| `uv run pytest tests/demo/test_experience_deployment.py -o addopts='' -q` after updating that fixture | All four packaging checks passed, including loading all 22 pairs without the full selection-study artifacts. No application-code change was needed for that failure; the image already copies `src/`. |
| Focused comparison generation and validation tests | 31 passed; also included in the broader Python run. |
| Ruff on nine changed Python implementation/test files | Passed. |
| MyPy 2.3.0 with `--follow-imports=silent` on the four changed implementation files | Passed. The full repository MyPy baseline was not claimed clean. |
| `npm test -- --run` in `frontend/onboarding` | 354 tests passed across 22 files after updating five stale projection expectations. Original input hashes and event lineage remain asserted. |
| `npm run typecheck` and `npm run build` | Passed. Vite retains its existing large-chunk advisory: 973.63 kB minified, 260.13 kB gzip. |
| Schema, canonical fixture, five scenario bundles, and catalog export | Passed; catalog/source verification and all 27 reviewed weeks were checked. |
| Markdown inputs, local links, canonical terminology, and `git diff --check` | Passed. |

The browser checks used the local Vite preview. Nisha's first week starts without
context; switching replaces both paragraphs and the question and reveals the
exact selected quote. Switching back restores the saved baseline. Inspect shows
one exact accepted prompt and one raw response per labeled arm, alongside each
associated narrative and question. Retry feedback is separately accessible.
Wei Jun's sixth week shows the disabled control and the no-moment explanation.
Meera's 1 December week displays the selected nudge response with the correct
2 December source link; switching to another week resets to without context.

Wide and narrow layouts were checked, including an effective CSS viewport of
391 × 844 for the phone checks. The phone toggle stacks above the heading and
fits within the card; the Experience and Inspect documents stay within the
viewport width. The temporary viewport override was reset afterward. Browser
error logs were empty. These checks did not deploy the application or run new
provider calls from the interface.

Implementation verification finished on the existing branch before publication.
The user subsequently authorized a Git commit and push. Remote tracker sync and
deployment were outside the implementation checks. Full-repository Python tests
and a Docker image build were not run; verification covered the affected
component suites, the packaging contract, frontend build, and local browser behavior.

## Limits

Deterministic validation checks source eligibility and whether each quotation
matches supplied text. It does not establish the truth of every prose
attribution, chronology, intention, or generalization. The editorial failures
above illustrate that difference. AI editorial review reduces identified errors
but cannot establish that all remaining semantic errors have been found.

Some unchanged weekly excerpts end in ellipses, and accepted responses can quote
those fragments. The existing interface can expand an eligible quotation to its
source sentence while Inspect preserves the raw generated wording. This display
behavior does not imply that the generator saw the expanded sentence. Small
wording caveats remain in the reviewed copy: Lukas's 30 June baseline infers an
impact on Tomasz from his visible reaction, and Meera's 8 December treatment
refers broadly to the boys where the relevant activity specifically names
Vihaan. Wei Jun's 16 June treatment retains awkward corrective wording about
not reinterpreting the collecting activity, and Meera's 1 December treatment
summarizes the selected morning activities broadly while emphasizing afternoon
ownership. These are recorded as limitations rather than hidden by a generic pass.

The generation is stochastic, and some accepted prompts include different
repair feedback. These pairs support a qualitative walkthrough of the same
Persona and week, not a controlled causal comparison. The frozen North Star
Moment experiment retains its own methodology, partitions, and evidence limits.
