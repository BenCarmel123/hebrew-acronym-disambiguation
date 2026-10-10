"""Identified inference from the approved Colab checkpoint on all test items.

No training or downloads. Predictions are persisted incrementally; technical
failures remain in the full denominator. Existing run directories are preserved.
"""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

from hebrew_acronyms.models.common.pairs import explicit_span, load_rows, validate_ids
from hebrew_acronyms.test_cohort import check_scored_cohort
from hebrew_acronyms.test_evaluation import _atomic_json, _code_hashes, _hash, _now


def validate_test_inputs(source_path, derived_path):
    source, rows = load_rows(source_path), load_rows(derived_path)
    validate_ids(source)
    validate_ids(rows)
    check_scored_cohort([r['item_id'] for r in source])
    if len(rows) != len(source):
        raise ValueError('Derived inputs must cover every test item')
    for original, row in zip(source, rows):
        if any(row.get(key) != value for key, value in original.items()):
            raise ValueError('Derived inputs changed original test fields or item order')
        explicit_span(row)
    return rows


def run_encoder_test(source_path, derived_path, output_dir, *, checkpoint, snapshot_path,
                     expected_sha256, code_revision, device='cpu', threads=4):
    """Validate three predictions, then persist all test items using the same loaded model."""
    import torch
    from hebrew_acronyms.models.dictabert_cross_encoder.colab import load_colab_finetuned
    from hebrew_acronyms.models.dictabert_cross_encoder.eval import evaluate

    rows = validate_test_inputs(source_path, derived_path)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(threads)
    relevant_sources = {name: digest for name, digest in _code_hashes().items()
                        if name.startswith('models/dictabert_cross_encoder/') or name in {
                            'test_encoder_inputs.py', 'test_encoder_inference.py', 'test_evaluation.py',
                            'models/common/pairs.py', 'data_processing/common/csv_io.py'}}
    identity = {'code_revision': code_revision, 'code_sha256': relevant_sources,
                'source_sha256': hashlib.sha256(Path(source_path).read_bytes()).hexdigest(),
                'derived_sha256': hashlib.sha256(Path(derived_path).read_bytes()).hexdigest(),
                'weights_sha256': expected_sha256, 'device': device, 'threads': threads,
                'item_ids': [r['item_id'] for r in rows],
                'target_policies': dict(Counter(r.get('span_basis') for r in rows)),
                'task': 'select', 'n_items': len(rows)}
    manifest = {'run_id': root.name, 'created_utc': _now(), 'identity': identity,
                'identity_sha256': _hash(identity)}
    _atomic_json(root / 'manifest.json', manifest)
    _atomic_json(root / 'inputs.json', rows)
    started = time.monotonic()
    print('Loading identified existing checkpoint', flush=True)
    tok, model, opened, closed = load_colab_finetuned(
        checkpoint, snapshot_path=snapshot_path, expected_sha256=expected_sha256, device=device)
    _atomic_json(root / 'checkpoint-reconstruction.json', model.colab_reconstruction)
    print('Strict checkpoint load passed; predicting first 3 test items', flush=True)
    records = []
    with (root / 'predictions.jsonl').open('x', encoding='utf-8') as stream:
        for index, row in enumerate(rows):
            tick = time.monotonic()
            # Gold is used only after prediction to compute exact candidate accuracy.
            model_input = {k: v for k, v in row.items() if k != 'gold_expansion'}
            record = evaluate([model_input], tok, model, opened, closed, device)[0]
            record.update(task='select', run_id=root.name, elapsed_seconds=time.monotonic() - tick,
                          correct=record['status'] == 'ok' and record['selected_candidate'] == row['gold_expansion'])
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
            records.append(record)
            if index == 2:
                passed = all(r['status'] == 'ok' for r in records)
                _atomic_json(root / 'validation.json', {'n_items': 3, 'technical_pass': passed,
                    'seconds': sum(r['elapsed_seconds'] for r in records)})
                if not passed:
                    raise RuntimeError('Initial technical prediction check failed; saved records retained')
                print('3/3 technically valid; continuing without retraining', flush=True)
            if (index + 1) % 25 == 0:
                print(f'{index + 1}/{len(rows)} predictions persisted; elapsed {time.monotonic() - started:.1f}s', flush=True)
    summary = {'run_id': root.name, 'identity_sha256': manifest['identity_sha256'],
               'n_items': len(rows), 'n_records': len(records), 'task': 'select',
               'statuses': dict(Counter(r['status'] for r in records)),
               'n_correct': sum(r['correct'] for r in records),
               'accuracy': sum(r['correct'] for r in records) / len(rows),
               'metric': 'exact_candidate_match', 'human_judgments': 0,
               'elapsed_seconds_including_load': time.monotonic() - started,
               'api_cost_ils': 0, 'training_performed': False,
               'target_policy_human_validated': False,
               'limitations': model.colab_reconstruction['unknowns']}
    _atomic_json(root / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary
