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
from .saved_human_review import load_review_table, REVIEW, PROVENANCE, _verified_bytes

RUN_ID = 'c6cc64b4017547d884540a16af02a066'
MODEL = 'hf.co/dicta-il/dictalm2.0-instruct-GGUF:Q4_K_M'
DIGEST = '2ec04b12dad1823bdc36d8317c8515346dd2cebef40ca9e4300a8ea319acabc0'
SOURCE = 'local-runs/dictalm-20261010'
BEN_TAGS_FILE = 'dictalm_generate_human_tags.csv'
BEN_TAGS_SHA256 = 'c6cfec23d97d9b2f29eac135454c4d609f19e6aba9280060a4c8ab406c51cbf0'
BEN_TAGS_COMMIT = 'b5cabe8ed749b5eff14de6cda48aa5ac838dc684'
GENERATION_DETAILS_SHA256 = 'd4c95c4ad035b1a32f7d0515e86e7c91e720412d535c957c24c915dfd92bc6ef'


def _bind_ben_tags(tags, generation_records, provenance):
    """Validate semantic bindings before constructing a separate judgment layer."""
    if (len(tags) != 381 or len({r['item_id'] for r in tags}) != 381
            or not all(is_scored(r['item_id']) for r in tags)):
        raise ValueError('Ben annotations require 381 unique scored test items')
    if (len(generation_records) != 381
            or len({r['item_id'] for r in generation_records}) != 381
            or any(r['run_id'] != RUN_ID or r['system'] != 'dictalm'
                   or r['task'] != 'generate' for r in generation_records)):
        raise ValueError('Ben annotations require the original DictaLM generation run')
    records = {r['item_id']: r for r in generation_records}
    if {r['item_id'] for r in tags} != set(records):
        raise ValueError('Ben annotation cohort differs from original responses')
    judgments = []
    for tag in tags:
        record = records[tag['item_id']]
        if tag['label'] not in ('yes', 'no'):
            raise ValueError('Ben annotation must be yes or no')
        if (tag['displayed_response'] != record['response'].strip()
                or tag['response_sha256'] != record['response_sha256']):
            raise ValueError('Ben annotation response differs from original journal')
        response_sha256 = hashlib.sha256(record['response'].encode()).hexdigest()
        if response_sha256 != record['response_sha256']:
            raise ValueError('Ben annotation journal response hash mismatch')
        # IDs bind to the original raw answer, not its boundary-trimmed CSV copy.
        identity = [BEN_TAGS_SHA256, RUN_ID, 'dictalm', 'generate',
                    tag['item_id'], response_sha256]
        judgment_id = hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()
        expected = dict(layer='ben_dictalm', judgment_id=judgment_id, answer_id='',
            item_id=record['item_id'], run_id=RUN_ID, system='dictalm', task='generate',
            response_sha256=response_sha256, label=tag['label'], kind='human_ben', round='',
            group_id='', event_index='', annotator='Ben', label_updated_at='', confirmed_at='',
            confirmed_by='', source_id='ben_dictalm', displayed_response=record['response'].strip())
        if tag != expected:
            raise ValueError('Ben annotation identity or attribution differs')
        judgments.append({
            'judgment_id': judgment_id, 'annotator': 'Ben',
            'source_path': f'{SOURCE}/{BEN_TAGS_FILE}',
            'source_sha256': BEN_TAGS_SHA256, 'source_commit': BEN_TAGS_COMMIT,
            'run_id': RUN_ID, 'system': 'dictalm', 'task': 'generate',
            'item_id': tag['item_id'], 'gold': record['gold'],
            'response': record['response'], 'response_sha256': response_sha256,
            'csv_response': tag['displayed_response'],
            'response_match': ('exact' if tag['displayed_response'] == record['response']
                               else 'boundary_whitespace_only'),
            'label': tag['label'], 'human_correct': tag['label'] == 'yes',
            'automatic_correct': record['correct'],
        })
    counts = Counter(j['label'] for j in judgments)
    matches = Counter(j['response_match'] for j in judgments)
    cross_tab = {
        'both_accept': sum(j['automatic_correct'] and j['human_correct'] for j in judgments),
        'ben_only': sum(not j['automatic_correct'] and j['human_correct'] for j in judgments),
        'automatic_only': sum(j['automatic_correct'] and not j['human_correct'] for j in judgments),
        'both_reject': sum(not j['automatic_correct'] and not j['human_correct'] for j in judgments),
    }
    if counts != {'yes': 193, 'no': 188}:
        raise ValueError('Ben annotation label counts differ')
    if matches != {'exact': 234, 'boundary_whitespace_only': 147}:
        raise ValueError('Ben annotation raw response correspondence differs')
    if cross_tab != {'both_accept': 121, 'ben_only': 72, 'automatic_only': 4, 'both_reject': 184}:
        raise ValueError('Ben annotation automatic/human cross-tab differs')
    expected_provenance = {
                'annotator': 'Ben', 'source_path': f'{SOURCE}/{BEN_TAGS_FILE}',
                'source_sha256': BEN_TAGS_SHA256, 'source_commit': BEN_TAGS_COMMIT,
                'source_run_id': RUN_ID,
                'source_generation_csv': f'{SOURCE}/dictalm_test_generate_details.csv',
                'source_generation_csv_sha256': GENERATION_DETAILS_SHA256,
                'recorded_rule': 'The CSV records yes/no under same_as_gold for each generation response.',
                'limitations': 'Annotation times, blinding and a detailed decision rubric are not documented; no protocol is reconstructed.',
            }
    if provenance != expected_provenance:
        raise ValueError('Ben historical provenance differs')
    return {'provenance': provenance,
            'validation': {'n_unique_items': len(records), 'all_in_scored_cohort': True,
                           'source_csv_exact_matches': len(judgments),
                           'journal_exact_matches': matches['exact'],
                           'journal_boundary_whitespace_only': matches['boundary_whitespace_only'],
                           'original_response_hashes_preserved': True},
            'summary': {'n_items': len(judgments), 'n_yes': counts['yes'],
                        'n_no': counts['no'], 'accuracy': counts['yes'] / len(judgments)},
            'cross_tab': cross_tab, 'judgments': judgments}


def load_ben_dictalm_tags(saved_root, generation_records):
    """Load Ben's consolidated rows, preserving original IDs and no writes.

    Repeated calls return the same judgment layer rather than appending to any
    review store. Only the original run's generation responses are accepted.
    """
    saved = Path(saved_root)
    directory = saved / SOURCE / 'dictalm/full-test'
    inventory = json.loads((saved / 'files.json').read_text(encoding="utf-8"))['files']
    for path in (directory / 'manifest.json', *sorted(directory.glob('attempts*.jsonl'))):
        relative = str(path.relative_to(saved))
        _verified_bytes(saved, relative, inventory[relative])
    manifest = evaluation._load_manifest(directory)
    if manifest['run_id'] != RUN_ID:
        raise ValueError('Ben annotations require the original DictaLM generation run')
    rows = {r['item_id']: r for r in manifest['identity']['rows']}
    original = {r['item_id']: r | {'gold': rows[r['item_id']]['gold_expansion'], 'run_id': RUN_ID}
                for r in evaluation.summarize_evaluation(directory)['records'] if r['task'] == 'generate'}
    for record in generation_records:
        source = original.get(record['item_id'])
        if source is None or any(record.get(k) != value for k, value in source.items()):
            raise ValueError('Ben annotation record differs from original DictaLM generation run journal')
    table, provenance = load_review_table(saved)
    return _bind_ben_tags([r for r in table if r['layer'] == 'ben_dictalm'],
                          generation_records, provenance['ben'])


def audit_saved_dictalm(saved_root, reference_manifest):
    """Recompute journals, match inputs/prompts, retain strict and diagnostic fields.

    A leading-label diagnostic reproduces the supplied CSV, never the official
    parser. The missing template newline is a protocol concern, not a demonstrated
    cause of any error. No sources or historical judgments are written here.
    """
    root = Path(saved_root) / SOURCE
    session = json.loads((root / 'session.json').read_text(encoding="utf-8"))
    if evaluation._hash(session['identity']) != session['identity_sha256']:
        raise ValueError('DictaLM session identity mismatch')
    reference = json.loads(Path(reference_manifest).read_text(encoding="utf-8"))
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
    pilot_reference = json.loads((Path(reference_manifest).parent.parent / 'dev-pilot/manifest.json').read_text(encoding="utf-8"))['identity']
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
    retained = [root / 'session.json']
    for stage in ('dev-pilot', 'full-test'):
        directory = root / 'dictalm' / stage
        retained.extend([directory / 'manifest.json', *sorted(directory.glob('attempts*.jsonl'))])
    for path in sorted(retained):
        sources.append({'path': str(path.relative_to(saved_root)),
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'bytes': path.stat().st_size})
    for relative in (REVIEW, PROVENANCE):
        path = Path(saved_root) / relative
        sources.append({'path': relative, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'bytes': path.stat().st_size})
    rows = {r['item_id']: r for r in manifest['identity']['rows']}
    records = []
    for record in summary['records']:
        row = rows[record['item_id']]
        choice = None
        if record['task'] == 'select':
            match = re.match(r'^\s*([A-Z]+)\b', record['response'])
            index = evaluation.scoring.parse_letter_choice(match[1], len(record['shown_order'])) if match else None
            choice = record['shown_order'][index] if index is not None else None
        if is_scored(record['item_id']):
            records.append(record | {'sentence': row['sentence'], 'gold': row['gold_expansion'],
                'response_sha256': hashlib.sha256(record['response'].encode()).hexdigest(),
                'run_id': RUN_ID, 'diagnostic_leading_label_candidate': choice,
                'diagnostic_leading_label_correct': choice == row['gold_expansion'] if record['task'] == 'select' else None,
                'human_judgment': None})
    metrics = [_scored_metric(g, summary['records']) | {'run_id': RUN_ID, 'comparison_status': 'held_for_protocol_review'}
               for g in summary['by_system_task']]
    selections = [r for r in records if r['task'] == 'select']
    ben_annotation = load_ben_dictalm_tags(saved_root, [r for r in records if r['task'] == 'generate'])
    reported_metrics = [
        {'metric': 'automatic_generation', 'task': 'generate', 'judgment_layer': 'automatic',
         'n_correct': sum(r['correct'] for r in records if r['task'] == 'generate')},
        {'metric': 'ben_generation', 'task': 'generate', 'judgment_layer': 'human_ben',
         'n_correct': ben_annotation['summary']['n_yes']},
        {'metric': 'strict_selection', 'task': 'select', 'judgment_layer': 'automatic',
         'n_correct': sum(r['correct'] for r in selections)},
        {'metric': 'diagnostic_leading_label_selection', 'task': 'select', 'judgment_layer': 'diagnostic',
         'n_correct': sum(r['diagnostic_leading_label_correct'] for r in selections)},
    ]
    for metric in reported_metrics:
        metric.update({'system': 'dictalm', 'run_id': RUN_ID, 'n_items': 381,
                       'accuracy': metric['n_correct'] / 381})
    return {'comparison_status': 'held_for_protocol_review', 'run_id': RUN_ID,
        'protocol_concern': 'Saved template ends with [/INST] without the publisher-required following newline; causal effect unknown; tokenized request not logged.',
        'publisher_format_url': 'https://huggingface.co/dicta-il/dictalm2.0-instruct-GGUF#instruction-format',
        'publisher_format_inspected_date': '2026-10-10',
        'accounting_concern': 'The local Ollama run adds zero API cost; its historical session records zero prior. This does not block reporting the result. Cumulative paid expenditure is accounted for separately; historical manifests and paid-run accounting gates remain unchanged.',
        'session_identity': session['identity_sha256'], 'model_identity': manifest['identity']['metadata']['model_identity'],
        'source_revision': manifest['identity']['code_revision'], 'sources': sources,
        'scored_signature_matches_reference': True, 'collected_items': len(rows),
        'excluded_items': sorted(k for k in rows if not is_scored(k)), 'metrics': metrics,
        'ben_annotation': ben_annotation, 'reported_metrics': reported_metrics,
        'pilot': {k: runs['dev-pilot'][1][k] for k in ('run_id', 'n_records', 'n_calls', 'n_completed')},
        'test_calls': summary['n_calls'], 'finish_reasons': dict(Counter(_finish_reason(r) for r in summary['records'])),
        'attempt_counts': dict(Counter(str(r['attempts']) for r in summary['records'])),
        'local_increment_ils': 0.0, 'local_prior_ils': session['identity']['prior_spend_ils'],
        'diagnostic_selection': {'n_items': len(selections),
            'csv_boundary_whitespace_trimmed': sum(r['response'].strip() != r['response'] for r in selections),
            'leading_label_parsed': sum(r['diagnostic_leading_label_candidate'] is not None for r in selections),
            'leading_label_matches_gold': sum(r['diagnostic_leading_label_correct'] for r in selections),
            'human_judgments': 0, 'official_score_unchanged': True}, 'records': records}
