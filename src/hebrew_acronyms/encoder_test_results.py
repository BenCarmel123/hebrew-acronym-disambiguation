"""Validate and read saved encoder test predictions without loading a model."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from hebrew_acronyms.models.common.pairs import load_rows
from hebrew_acronyms.test_cohort import scored_ids
from hebrew_acronyms.test_evaluation import _hash


def load_encoder_test(directory, test_path, *, expected_run_id, expected_identity, expected_predictions_sha256):
    """Check a saved encoder run and score it on the items of the current test file.

    The run's own saved inputs give every collected row; a run saved before the
    document-overlap exclusion covers 395 items. Its scored rows must equal the
    current 381-item test file.
    """
    root = Path(directory)
    manifest = json.loads((root / 'manifest.json').read_text())
    identity = manifest['identity']
    if (manifest['run_id'] != expected_run_id or manifest['identity_sha256'] != expected_identity
            or _hash(identity) != expected_identity):
        raise ValueError('Encoder run identity mismatch')
    raw = (root / 'predictions.jsonl').read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_predictions_sha256:
        raise ValueError('Encoder predictions differ from the identified saved artifact')
    records = [json.loads(line) for line in raw.decode().splitlines()]
    rows = json.loads((root / 'inputs.json').read_text())
    if (len(records) != len(rows)
            or [r['item_id'] for r in rows] != identity['item_ids']
            or [r['item_id'] for r in records] != identity['item_ids']):
        raise ValueError('Encoder predictions do not preserve all ordered test IDs')
    # Every saved record is checked; only the scored items count towards the metric.
    scored = set(scored_ids(identity['item_ids']))
    current = load_rows(test_path)
    saved = [row for row in rows if row['item_id'] in scored]
    if (len(current) != len(saved)
            or any(any(old.get(k) != v for k, v in new.items()) for new, old in zip(current, saved))):
        raise ValueError('Encoder test source differs from comparison cohort')
    correct, completed = 0, 0
    for row, record in zip(rows, records):
        if record['run_id'] != expected_run_id or record['task'] != 'select':
            raise ValueError('Encoder record provenance mismatch')
        if record['status'] == 'ok':
            if record['selected_candidate'] not in [c.strip() for c in row['candidates'].split('|')]:
                raise ValueError('Encoder choice is outside the original candidates')
            completed += row['item_id'] in scored
        recomputed = record['status'] == 'ok' and record['selected_candidate'] == row['gold_expansion']
        if record['correct'] is not recomputed:
            raise ValueError('Encoder saved score disagrees with exact candidate equality')
        correct += recomputed and row['item_id'] in scored
    reconstruction = json.loads((root / 'checkpoint-reconstruction.json').read_text())
    if reconstruction['weights_sha256'] != identity['weights_sha256']:
        raise ValueError('Encoder checkpoint provenance mismatch')
    metric = {'system': 'dictabert', 'task': 'select', 'run_id': expected_run_id,
              'n_items': len(scored), 'n_completed': completed, 'n_correct': correct,
              'accuracy': correct / len(scored), 'score': 'exact_candidate_match',
              'n_valid': completed,
              'status_counts': dict(Counter(r['status'] for r in records if r['item_id'] in scored))}
    return {'manifest': manifest, 'reconstruction': reconstruction, 'records': records, 'rows': rows, 'metric': metric,
            'source_path': str(root), 'predictions_sha256': expected_predictions_sha256}
