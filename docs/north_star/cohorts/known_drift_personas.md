# Retained Personas with known Drift

[Back to the methodology](../nsm_experiment_methodology.md#selected-persona-index)

**35 Personas, 311 Journal Entries.** All IDs from the existing known-Drift
cohort are retained. The table records the existing 27 development and eight
reserved assignments. The eight reserved histories enter final evaluation only
if they pass the prior NSM exposure audit; otherwise, move exposed histories
to development and revisit the proposed split. Reserved writing was not
inspected during cohort selection; the exposure audit remains pending.

| Persona ID | Name | Core Values, in stored order | Journal Entries | Existing assignment |
| --- | --- | --- | ---: | --- |
| `02fb94f3` | Noor Haddad | Self-Direction, Tradition | 12 | Development |
| `110fcd4f` | Lukas Bergmann | Conformity, Self-Direction | 3 | Development |
| `11de77e8` | Lukas Vermeer | Self-Direction, Conformity | 12 | Development |
| `127aefff` | Tariq Haddad | Stimulation, Conformity | 9 | Development |
| `1496159d` | Soo-jin Park | Universalism | 6 | Development |
| `152df7a4` | Joon-ho Park | Universalism | 8 | Development |
| `1f86f569` | Erik Lindqvist | Self-Direction, Conformity | 4 | Reserved |
| `2541429a` | Farid Al-Khatib | Tradition, Self-Direction | 9 | Development |
| `3a3b15e4` | Nasser Haddad | Tradition, Hedonism | 11 | Development |
| `5fa8b540` | Nisha Agarwal | Universalism | 8 | Development |
| `66ced716` | Layla Haddad | Universalism | 6 | Development |
| `742c98d6` | Margot van den Berg | Universalism | 11 | Development |
| `775c7d71` | Yoon Jihye | Benevolence, Self-Direction | 8 | Development |
| `7adc5866` | Meena Krishnamurthy | Benevolence, Conformity | 9 | Development |
| `7c712a0a` | Ricardo Mendoza | Stimulation, Conformity | 10 | Development |
| `7ff1d0fb` | Seo-yun Park | Tradition, Security | 7 | Development |
| `87e92805` | Roberto Estrada | Security, Hedonism | 11 | Development |
| `8f83c818` | Wei Jun Chen | Universalism | 11 | Development |
| `8fcd7947` | Marcus Chen | Security, Benevolence | 4 | Development |
| `961a4e3f` | Meera Krishnamurthy | Self-Direction, Tradition | 12 | Development |
| `988d1a65` | Marc Vandenberghe | Power | 12 | Development |
| `9d126412` | Kenneth Goh | Power | 8 | Development |
| `a24b8d8f` | Lukas Vetter | Universalism | 10 | Reserved |
| `abf1ce49` | Jordan Reeves | Security | 10 | Reserved |
| `ad378991` | Lim Wei Keong | Universalism | 7 | Reserved |
| `bf44e50f` | Tariq Haddad | Hedonism | 6 | Development |
| `cd7c2a11` | Roshni Malik | Power | 10 | Development |
| `d7a0683f` | Lukas Engström | Universalism | 11 | Reserved |
| `dab3e9b9` | Nate Bergeron | Universalism | 6 | Reserved |
| `dbe2c53d` | Luciana Ferreira Vidal | Universalism, Conformity | 11 | Development |
| `e448979e` | Nadia Hussain | Power | 10 | Reserved |
| `e5c9d9e5` | Henrik van den Berg | Power | 12 | Development |
| `e6838e16` | Rohan Kulkarni | Security | 10 | Development |
| `ed67c9cc` | Lim Sook Yin | Hedonism | 9 | Development |
| `f418b74b` | Rohan Shetty | Hedonism | 8 | Reserved |

Sources: [Persona registry](../../../logs/registry/personas.parquet),
[upstream AI-reviewed Drift records](../../../logs/experiments/artifacts/twinkl_qtwz_complete_development_review_20260714/results/complete_development_drift_episodes.parquet),
and [existing cohort assignment](../../../config/evals/north_star_cohort.json).
These define cohort membership, not NSM support labels. Names are not unique;
use `persona_id` as the key and preserve the listed Core Value order.

SHA-256 at selection:

| Source | SHA-256 |
| --- | --- |
| Registry | `c6b975711471140b6cd8faf4d730a3b9b3078c2cc37704d38cdc143526be91fd` |
| Drift records | `a2065c1217776071a5ce4f317b4a7afd21d0f0b08644ceebd38dbdafc7cb1c22` |
| Existing assignment | `085a92a540d71d78b02888b2ce315b5717ad636fb85dba51893df7a180d537f6` |
