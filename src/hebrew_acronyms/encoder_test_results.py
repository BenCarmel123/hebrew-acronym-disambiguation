"""Validate and read saved encoder test predictions without loading a model."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from hebrew_acronyms.models.common.pairs import load_rows
from hebrew_acronyms.test_evaluation import _hash


def load_encoder_test(directory, test_path, *, expected_run_id, expected_identity, expected_predictions_sha256):
    root = Path(directory)
    manifest = json.loads((root / 'manifest.json').read_text())
    identity = manifest['identity']
    if (manifest['run_id'] != expected_run_id or manifest['identity_sha256'] != expected_identity
            or _hash(identity) != expected_identity):
        raise ValueError('Encoder run identity mismatch')
    if hashlib.sha256(Path(test_path).read_bytes()).hexdigest() != identity['source_sha256']:
        raise ValueError('Encoder test source differs from comparison cohort')
    raw = (root / 'predictions.jsonl').read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_predictions_sha256:
        raise ValueError('Encoder predictions differ from the identified saved artifact')
    records = [json.loads(line) for line in raw.decode().splitlines()]
    rows = load_rows(test_path)
    if (len(rows) != 395 or len(records) != 395
            or [r['item_id'] for r in rows] != identity['item_ids']
            or [r['item_id'] for r in records] != identity['item_ids']):
        raise ValueError('Encoder predictions do not preserve all ordered test IDs')
    correct, completed = 0, 0
    for row, record in zip(rows, records):
        if record['run_id'] != expected_run_id or record['task'] != 'select':
            raise ValueError('Encoder record provenance mismatch')
        if record['status'] == 'ok':
            if record['selected_candidate'] not in [c.strip() for c in row['candidates'].split('|')]:
                raise ValueError('Encoder choice is outside the original candidates')
            completed += 1
        recomputed = record['status'] == 'ok' and record['selected_candidate'] == row['gold_expansion']
        if record['correct'] is not recomputed:
            raise ValueError('Encoder saved score disagrees with exact candidate equality')
        correct += recomputed
    reconstruction = json.loads((root / 'checkpoint-reconstruction.json').read_text())
    if reconstruction['weights_sha256'] != identity['weights_sha256']:
        raise ValueError('Encoder checkpoint provenance mismatch')
    metric = {'system': 'dictabert', 'task': 'select', 'run_id': expected_run_id,
              'n_items': 395, 'n_completed': completed, 'n_correct': correct,
              'accuracy': correct / 395, 'score': 'exact_candidate_match',
              'n_valid': completed, 'status_counts': dict(Counter(r['status'] for r in records))}
    return {'manifest': manifest, 'reconstruction': reconstruction, 'records': records, 'metric': metric,
            'source_path': str(root), 'predictions_sha256': expected_predictions_sha256}
