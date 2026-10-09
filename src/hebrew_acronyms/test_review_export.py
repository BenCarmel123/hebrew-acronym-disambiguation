"""Export saved test answers to the existing local human-review application.

One review item represents one original answer. Its identity binds the collection
run, task, item and exact response; no historical judgments are imported.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from hebrew_acronyms import test_evaluation as evaluation
from hebrew_acronyms.data_processing.prepare_encoder_inputs import QUOTES


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def export_review(run_sources, output_path):
    """Write a fresh immutable input bundle for (directory, expected run ID) pairs."""
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    items, systems, sources, seen = [], [], [], set()
    for directory, expected_id in run_sources:
        directory = Path(directory)
        manifest = evaluation._load_manifest(directory)
        if manifest['run_id'] != expected_id or manifest['identity']['cohort'] != 'full_test':
            raise ValueError('Expected identified full-test run')
        if expected_id in seen:
            raise ValueError('Duplicate run')
        seen.add(expected_id)
        summary = evaluation.summarize_evaluation(directory)
        rows = {r['item_id']: r for r in manifest['identity']['rows']}
        if len(rows) != 395 or len(summary['records']) != 790:
            raise ValueError('Full test denominators are required')
        system = manifest['identity']['systems'][0]
        for task in ('generate', 'select'):
            systems.append({'id': system['name'] + '_' + task,
                            'blind_id': 'מערכת ' + str(len(systems) + 1),
                            'name': system['model'] + (' — יצירה' if task == 'generate' else ' — בחירה'),
                            'task': 'generation' if task == 'generate' else 'selection',
                            'response_type': 'text' if task == 'generate' else 'letter'})
        for path in [directory / 'manifest.json', *sorted(directory.glob('attempts*.jsonl'))]:
            sources.append({'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'bytes': path.stat().st_size})
        for record in summary['records']:
            row = rows[record['item_id']]
            raw = record['response']
            binding = {'item_id': row['item_id'], 'task': record['task'], 'run_id': expected_id,
                       'response_sha256': hashlib.sha256(raw.encode()).hexdigest()}
            answer_id = digest(binding)
            technical = record['status'] != 'response_received'
            stratum = 'technical_failure' if technical else 'automatic_positive' if record['correct'] else 'automatic_nonpositive'
            shown = record['shown_order'] or []
            flags = [stratum]
            answer = {'id': answer_id, 'system_id': system['name'] + '_' + record['task'],
                      'task': 'generation' if record['task'] == 'generate' else 'selection',
                      'response_type': 'text' if record['task'] == 'generate' else 'letter',
                      'raw': raw, 'decoded': record['selected_candidate'] if shown else raw,
                      'option_mapping': [{'letter': evaluation.scoring.candidate_labels(len(shown))[i], 'candidate': c} for i, c in enumerate(shown)],
                      'shown_order': shown, 'omitted_candidates': [], 'missing': not bool(raw),
                      'status': record['status'], 'technical_failure': technical,
                      'auto_score': record['correct'], 'auto_valid': record['valid'],
                      'auto_score_rule': 'historical_quote_normalized_gold_substring' if not shown else 'strict_candidate_label_v1',
                      'mechanical_flags': flags, 'source_file': str(directory / 'attempts.jsonl'),
                      'binding': binding, 'prompt_sha256': record['prompt_sha256']}
            spans = [{'start': m.start(), 'end': m.end()} for m in re.finditer(
                re.escape(row['acronym'].translate(QUOTES)), row['sentence'].translate(QUOTES))]
            items.append({'id': answer_id, 'original_item_id': row['item_id'],
                          'sentence': row['sentence'], 'acronym': row['acronym'], 'source': row.get('source', ''),
                          'origin': 'מחובר ב־AI' if row.get('label_origin') == 'claude_authored' else 'לפי מקור שמור',
                          'source_metadata': row, 'gold': row['gold_expansion'],
                          'candidates': row['candidates'].split('|'), 'answers': [answer],
                          'findings': flags, 'selection_reasons': ['בדיקת תשובה חדשה; אין העברת שיפוטים היסטוריים'],
                          'prior_exposure': 'unknown', 'target_spans': spans, 'target_span_verified': False,
                          'target_warning': 'סימון מכני של כל ההתאמות; אינו הכרעה בהופעת המטרה.',
                          'acronym_type_proxy': '', 'sampling_stratum': stratum})
    # Early timing sample mixes negative and positive generation answers; all
    # answers remain accessible afterward, including selection and technical failures.
    ranked = sorted(items, key=lambda x: x['id'])
    negative = [x['id'] for x in ranked if not x['answers'][0]['auto_score']]
    positive = [x['id'] for x in ranked if x['answers'][0]['auto_score']]
    neg_gen = [x['id'] for x in ranked if x['answers'][0]['task'] == 'generation' and x['sampling_stratum'] == 'automatic_nonpositive']
    pos_gen = [x['id'] for x in ranked if x['answers'][0]['task'] == 'generation' and x['sampling_stratum'] == 'automatic_positive']
    calibration = []
    for i in range(min(4, len(pos_gen), len(neg_gen) // 4)):
        calibration.extend(neg_gen[4*i:4*i+4] + pos_gen[i:i+1])
    priority = {v: i for i, v in enumerate(calibration + [v for v in negative if v not in calibration] + [v for v in positive if v not in calibration])}
    items.sort(key=lambda x: priority[x['id']])
    provenance = {'repository': 'https://github.com/BenCarmel123/hebrew-acronym-disambiguation',
                  'files': sources, 'run_ids': sorted(seen),
                  'limits': ['Automatic scores are not human judgments.', 'Technical failures remain distinct.',
                             'All negatives and positives are queued; incomplete review cannot exclude remaining false negatives.',
                             'No previous judgments were transferred.']}
    source_id = digest(provenance)
    queues = {'calibration': calibration, 'diagnosis': negative, 'evaluation': positive}
    bundle = {'schema_version': 'human-review-data-v2', 'dataset_id': source_id, 'source_identity': source_id,
              'sampling_plan_id': digest(queues), 'provenance': provenance, 'systems': systems,
              'queues': queues, 'sampling': {'warning': '20 תשובות למדידת קצב; לאחריהן תור כל השליליים וביקורת כל החיוביים. זה אינו מדגם מייצג.',
                                            'size': len(calibration), 'stage': 'calibration'},
              'items': items, 'coverage': dict(Counter(x['sampling_stratum'] for x in items))}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('x', encoding='utf-8') as handle:
        json.dump(bundle, handle, ensure_ascii=False, indent=2)
    return bundle
