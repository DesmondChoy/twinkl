# Experience and Inspect Contracts

`experience_inspect_v1.schema.json` and
`experience_inspect_v1.fixture.json` are generated from the strict Python
contracts in `src/demo/`.

Regenerate them from the repository root:

```sh
uv run python -m src.demo.export_contract_schema
```

Do not edit the generated JSON files directly. Python tests verify data
equivalence with the generator, while React tests validate the same
Profile, shared session, saved scenario, API examples, and Inspect trace events.
