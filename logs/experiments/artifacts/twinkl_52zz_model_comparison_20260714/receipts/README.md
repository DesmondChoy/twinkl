# OpenAI API Cost Receipt

The raw OpenAI Platform Costs export is
[`openai_cost_2026-07-14_utc.csv`](openai_cost_2026-07-14_utc.csv). It was
downloaded after the `twinkl-52zz` comparison and retains the original values;
the file was renamed only to make its UTC scope explicit.

## Recorded amount and scope

- UTC interval: `2026-07-14T00:00:00` to `2026-07-15T00:00:00`
- OpenAI project: `Default project`
- Billed project-day amount: `$12.01916515 USD`
- SHA-256:
  `a9ac1538f57dea938aaee6be85f3ea3938b32f242312e9d953c5bd348b8a8d41`

The export contains one project-day row. It does not separate models, API
keys, line items, or `twinkl-52zz` calls from other activity in the same OpenAI
project and UTC interval. It is therefore billing evidence for the whole
project day, not an exact billed cost for the model comparison.

## How to cite the cost

The preregistered token estimate was `$11.976405`, under a `$15` cap. The
recorded response-token totals produce a `$10.86520875` standard-rate
calculation when every input token is charged at the full uncached rate. The
OpenAI export reports `$12.01916515` for the whole project day, which is below
the cap but `$0.04276015` above the preregistered estimate. The two amounts are
not directly comparable because the export's scope is broader than the
comparison.

The response receipts retain aggregate input tokens but not the cached-input
token split. [OpenAI API pricing](https://developers.openai.com/api/docs/pricing)
lists separate cached-input rates, so the `$10.86520875` calculation is not an
official billed amount. A project report may cite the CSV as the official
project-day receipt and the experiment report as the study-specific token
calculation, but should not describe either number as the exact billed cost of
`twinkl-52zz`.
