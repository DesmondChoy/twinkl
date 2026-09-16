# Reproduce the legacy validation comparison

This is the original offline comparison command, with the original `HEAD`
references pinned to `c6b5edb1a30e1f18f23a5f1752fa545abc386c6f` and output moved
to `/tmp`. It compares eight constructed narratives under 24 settings each;
it does not assess saved model responses. No provider calls or random seed
are involved.

The baseline validator function executes with the current module's globals.
The shared helper implementations were unchanged in the reviewed patch. This
comparison therefore covers the extracted validator function, rather than
an isolated execution of the entire historical repository.

From the repository root:

```sh
source .venv/bin/activate
UV_CACHE_DIR=/tmp/twinkl-qa-uv-cache uv run --no-sync python - <<'PY'
import ast
import hashlib
import itertools
import json
import subprocess
from pathlib import Path

import src.coach.weekly_digest as module
from src.coach.schemas import CoachNarrative, EvidenceSnippet, WeeklyDigest

baseline = 'c6b5edb1a30e1f18f23a5f1752fa545abc386c6f'
source = subprocess.check_output(
    ['git', 'show', f'{baseline}:src/coach/weekly_digest.py'], text=True,
)
node = next(
    item for item in ast.parse(source).body
    if isinstance(item, ast.FunctionDef)
    and item.name == 'validate_weekly_digest_narrative'
)
namespace = vars(module).copy()
exec(compile(ast.Module(body=[node], type_ignores=[]), '<baseline validator>', 'exec'), namespace)
legacy = namespace['validate_weekly_digest_narrative']
digest = WeeklyDigest(
    persona_id='fictional-qa', week_start='2026-09-14', week_end='2026-09-20',
    response_mode='no_active_drift', mode_source='drift_detection',
    mode_rationale='QA fixture', n_entries=1, overall_mean=0.0,
    top_tensions=[], top_strengths=[], dimensions=[],
    evidence=[EvidenceSnippet(
        date='2026-09-18', t_index=0, direction='context',
        dimensions=['self_direction'],
        excerpt='I finished the drawing. I saw an improvement in the shading. What can I try next?',
    )],
)
base = CoachNarrative(
    weekly_mirror='You wrote, "I finished the drawing".',
    tension_explanation='You made time to try a technique you had chosen.',
    reflective_question='What mattered to you about choosing that exercise?',
)
cases = {
    'normal': base,
    'source_improvement': base.model_copy(update={
        'tension_explanation': 'You wrote, "I saw an improvement in the shading."',
    }),
    'extra_question': base.model_copy(update={
        'tension_explanation': 'You tried a new technique. What drew you to it?',
    }),
    'invented_quote': base.model_copy(update={
        'tension_explanation': 'You wrote, "I saw improvement yesterday".',
    }),
    'source_question': base.model_copy(update={
        'reflective_question': 'When you wrote "What can I try next?", what possibility did you have in mind?',
    }),
    'too_long': base.model_copy(update={'tension_explanation': 'detail ' * 180}),
    'whitespace': CoachNarrative(**{
        key: ' \n' + value + '\n ' for key, value in base.model_dump().items()
    }),
    'empty': base.model_copy(update={'reflective_question': ''}),
}
results = []
for (name, narrative), (old_policy, new_policy), (voice, version), minimum, maximum in itertools.product(
    cases.items(), [('historical', 'historical'), ('current', '4.6')],
    [(False, '4.5'), (True, '4.4'), (True, '4.5')], [None, 10], [180, 20],
):
    options = {
        'validate_voice': voice, 'voice_version': version,
        'min_words': minimum, 'max_words': maximum,
    }
    old = legacy(digest, narrative, validation_policy=old_policy, **options).model_dump(mode='json')
    new = module.validate_weekly_digest_narrative(
        digest, narrative, validation_policy=new_policy, **options,
    ).model_dump(mode='json')
    results.append({
        'case': name, 'legacy_policy': old_policy, 'current_policy': new_policy,
        **options, 'identical_complete_receipt': old == new,
        'original': old, 'replayed': new,
    })
assert all(item['identical_complete_receipt'] for item in results)
report = {
    'git_head': baseline,
    'original_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
    'comparison_count': len(results), 'all_identical': True,
    'note': 'Synthetic fixture/settings comparison against the pinned baseline validator function; no live calls.',
    'results': results,
}
output = Path('/tmp/twinkl-legacy-validation-equivalence.json')
output.write_text(json.dumps(report, indent=2) + '\n')
print(f'Complete legacy receipts identical in {len(results)}/{len(results)} cases.')
print(output)
PY
```

Expected result for this change: `192/192`. The generated JSON contains both
complete validation outputs for every comparison, including check details,
pass/fail results, quoted phrases, and word counts.
