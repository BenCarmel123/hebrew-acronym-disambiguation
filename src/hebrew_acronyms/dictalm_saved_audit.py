"""Audit the preserved DictaLM run without admitting it to the pooled comparison."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

from . import test_evaluation as evaluation
from .saved_test_comparison import (
    _test_signature, _scored_metric, _finish_reason, COMPATIBLE_COLLECTOR_SHA256,
)
from .test_cohort import is_scored

RUN_ID = 'c6cc64b4017547d884540a16af02a066'
MODEL = 'hf.co/dicta-il/dictalm2.0-instruct-GGUF:Q4_K_M'
DIGEST = '2ec04b12dad1823bdc36d8317c8515346dd2cebef40ca9e4300a8ea319acabc0'
SOURCE = 'local-runs/dictalm-20261010'


def audit_saved_dictalm(saved_root, reference_manifest):
    """Recompute journals, match inputs/prompts, retain strict and diagnostic fields.

    A leading-label diagnostic reproduces the supplied CSV, never the official
    parser. The missing template newline is a protocol concern, not a demonstrated
    cause of any error. No sources or historical judgments are written here.
    """
    root = Path(saved_root) / SOURCE
    session = json.loads((root / 'session.json').read_text())
    if evaluation._hash(session['identity']) != session['identity_sha256']:
        raise ValueError('DictaLM session identity mismatch')
    reference = json.loads(Path(reference_manifest).read_text())
    if (session['identity']['code_sha256']['test_evaluation.py'] not in COMPATIBLE_COLLECTOR_SHA256
            or hashlib.sha256(Path(evaluation.__file__).read_bytes()).hexdigest() not in COMPATIBLE_COLLECTOR_SHA256):
        raise ValueError('Unreviewed DictaLM collector')
    sources, runs = [], {}
    for stage in ('dev-pilot', 'full-test'):
        directory = root / 'dictalm' / stage
        manifest = evaluation._load_manifest(directory)
        identity = manifest['identity']
        if (identity['metadata']['session_identity'] != session['identity_sha256']
                or identity['code_sha256'] != session['identity']['code_sha256']
                or identity['code_revision'] != session['identity']['code_revision']
                or identity['source_sha256'] != session['identity']['sources'][stage.replace('-', '_')]['sha256']):
            raise ValueError('DictaLM run/session mismatch')
        model = identity['metadata']['model_identity']
        if (model['model'] != MODEL or model['digest'] != DIGEST
                or model['options'] != {'temperature': 0, 'seed': 42, 'num_predict': 512}
                or model['details']['quantization_level'] != 'Q4_K_M'):
            raise ValueError('DictaLM model identity mismatch')
        _, _, tails = evaluation._read_events(directory, manifest)
        if tails:
            raise ValueError('Truncated DictaLM journal')
        summary = evaluation.summarize_evaluation(directory)
        if (identity['rates_usd_per_million'] != {'dictalm': {'input': 0.0, 'output': 0.0}}
                or identity['reserve_per_call_usd'] != {'dictalm': 0.0}
                or summary['charged_or_reserved_usd'] != 0):
            raise ValueError('DictaLM local accounting changed')
        stored = json.loads((directory / 'summary.json').read_text())
        for field in ('records', 'by_system_task', 'n_calls', 'n_completed', 'n_records',
                      'n_ambiguous', 'n_incomplete', 'n_identity_unverified', 'charged_or_reserved_usd'):
            if summary[field] != stored[field]:
                raise ValueError(f'Saved DictaLM summary differs: {field}')
        for record in summary['records']:
            metadata = record['response_metadata']
            if (metadata['digest_before'] != DIGEST or metadata['digest_after'] != DIGEST
                    or metadata['request_settings']['options'] != model['options']):
                raise ValueError('DictaLM response identity/settings mismatch')
        runs[stage] = (manifest, summary)
    manifest, summary = runs['full-test']
    if manifest['run_id'] != RUN_ID:
        raise ValueError('Unexpected DictaLM test run')
    if _test_signature(manifest['identity']) != _test_signature(reference['identity']):
        raise ValueError('DictaLM scored inputs, prompts or orders differ')
    pilot_reference = json.loads((Path(reference_manifest).parent.parent / 'dev-pilot/manifest.json').read_text())['identity']
    pilot_identity = runs['dev-pilot'][0]['identity']
    for field in ('rows', 'protocol', 'seed', 'target_policy', 'target_occurrences'):
        if pilot_identity[field] != pilot_reference[field]:
            raise ValueError('DictaLM pilot inputs differ')
    fields = ('item_id', 'task', 'prompt', 'prompt_sha256', 'shown_order')
    if ([{k: r[k] for k in fields} for r in pilot_identity['requests']]
            != [{k: r[k] for k in fields} for r in pilot_reference['requests']]):
        raise ValueError('DictaLM pilot prompts or orders differ')
    if runs['dev-pilot'][0]['identity']['metadata']['model_identity'] != manifest['identity']['metadata']['model_identity']:
        raise ValueError('DictaLM pilot/test model identity differs')
    for name in ('models/common/eval.py', 'models/common/pairs.py'):
        current = Path(evaluation.__file__).parent / name
        if hashlib.sha256(current.read_bytes()).hexdigest() != manifest['identity']['code_sha256'][name]:
            raise ValueError('DictaLM scoring source differs')
    for path in sorted(root.rglob('*')):
        if path.is_file():
            sources.append({'path': str(path.relative_to(saved_root)),
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'bytes': path.stat().st_size})
    with (root / 'dictalm_test_select_details.csv').open(encoding='utf-8-sig', newline='') as handle:
        details = list(csv.DictReader(handle))
    if len(details) != 395 or len({r['item_id'] for r in details}) != 395:
        raise ValueError('DictaLM CSV coverage differs')
    details = {r['item_id']: r for r in details}
    rows = {r['item_id']: r for r in manifest['identity']['rows']}
    records = []
    for record in summary['records']:
        row = rows[record['item_id']]
        choice = None
        if record['task'] == 'select':
            match = re.match(r'^\s*([A-Z]+)\b', record['response'])
            index = evaluation.scoring.parse_letter_choice(match[1], len(record['shown_order'])) if match else None
            choice = record['shown_order'][index] if index is not None else None
            detail = details[record['item_id']]
            # The supplied review CSV trims boundary whitespace. Journals remain
            # authoritative; derived bindings retain the original leading newline.
            if (detail['response'] != record['response'].strip()
                    or detail['lenient_choice'] != (choice or '')
                    or detail['lenient_correct'] != str(choice == row['gold_expansion'])
                    or detail['auto_correct'] != str(record['correct'])
                    or detail['valid'] != str(record['valid'])
                    or detail['human_judgment'] != ''):
                raise ValueError('DictaLM CSV diagnostic differs from raw response')
        if is_scored(record['item_id']):
            records.append(record | {'sentence': row['sentence'], 'gold': row['gold_expansion'],
                'response_sha256': hashlib.sha256(record['response'].encode()).hexdigest(),
                'run_id': RUN_ID, 'diagnostic_leading_label_candidate': choice,
                'diagnostic_leading_label_correct': choice == row['gold_expansion'] if record['task'] == 'select' else None,
                'human_judgment': None})
    metrics = [_scored_metric(g, summary['records']) | {'run_id': RUN_ID, 'comparison_status': 'held_for_protocol_review'}
               for g in summary['by_system_task']]
    selections = [r for r in records if r['task'] == 'select']
    return {'comparison_status': 'held_for_protocol_review', 'run_id': RUN_ID,
        'protocol_concern': 'Saved template ends with [/INST] without the publisher-required following newline; causal effect unknown; tokenized request not logged.',
        'publisher_format_url': 'https://huggingface.co/dicta-il/dictalm2.0-instruct-GGUF#instruction-format',
        'publisher_format_inspected_date': '2026-10-10',
        'accounting_concern': 'Independent zero-cost local session records zero prior; cannot append to the paid chronology without an explicit accounting decision.',
        'session_identity': session['identity_sha256'], 'model_identity': manifest['identity']['metadata']['model_identity'],
        'source_revision': manifest['identity']['code_revision'], 'sources': sources,
        'scored_signature_matches_reference': True, 'collected_items': len(rows),
        'excluded_items': sorted(k for k in rows if not is_scored(k)), 'metrics': metrics,
        'pilot': {k: runs['dev-pilot'][1][k] for k in ('run_id', 'n_records', 'n_calls', 'n_completed')},
        'test_calls': summary['n_calls'], 'finish_reasons': dict(Counter(_finish_reason(r) for r in summary['records'])),
        'attempt_counts': dict(Counter(str(r['attempts']) for r in summary['records'])),
        'local_increment_ils': 0.0, 'local_prior_ils': session['identity']['prior_spend_ils'],
        'diagnostic_selection': {'n_items': len(selections),
            'csv_boundary_whitespace_trimmed': sum(details[r['item_id']]['response'] != r['response'] for r in selections),
            'leading_label_parsed': sum(r['diagnostic_leading_label_candidate'] is not None for r in selections),
            'leading_label_matches_gold': sum(r['diagnostic_leading_label_correct'] for r in selections),
            'human_judgments': 0, 'official_score_unchanged': True}, 'records': records}
