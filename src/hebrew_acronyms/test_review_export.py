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
from hebrew_acronyms.test_cohort import is_scored, scored_ids


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
        scored_ids(rows)
        if len(summary['records']) != 2 * len(rows):
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
            # Items that share a document with training are never scored, so never queued.
            if not is_scored(record['item_id']):
                continue
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
    return _write_review_bundle(items, systems, sources, seen, output_path)


def _write_review_bundle(items, systems, sources, seen, output_path):
    # Early timing sample mixes negative and positive generation answers; all
    # answers remain accessible afterward, including selection and technical failures.
    ranked = sorted(items, key=lambda x: x['id'])
    negative = [x['id'] for x in ranked if not x['answers'][0]['auto_score']]
    positive = [x['id'] for x in ranked if x['answers'][0]['auto_score']]
    neg_gen = [x['id'] for x in ranked if x['answers'][0]['task'] == 'generation' and x['sampling_stratum'] == 'automatic_nonpositive']
    pos_gen = [x['id'] for x in ranked if x['answers'][0]['task'] == 'generation' and x['sampling_stratum'] == 'automatic_positive']
    if not neg_gen and not pos_gen:
        neg_gen, pos_gen = negative, positive
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


def review_coverage(data_path, annotations_path):
    """Read explicit completed judgments, keeping automatic and human labels apart."""
    dataset = json.loads(Path(data_path).read_text())
    path = Path(annotations_path)
    annotations = json.loads(path.read_text()) if path.exists() else None
    if annotations is not None and (
            annotations.get('schema_version') != 'human-review-v2'
            or annotations.get('dataset_id') != dataset['dataset_id']
            or annotations.get('provenance') != dataset['provenance']):
        raise ValueError('Human review source or schema mismatch')
    records = annotations.get('records', {}) if annotations else {}
    items = {item['id']: item for item in dataset['items']}
    if len(items) != len(dataset['items']) or set(records) - set(items):
        raise ValueError('Duplicate or unknown review item')
    groups, timestamps, starts = {}, [], []
    for item_id, item in items.items():
        # Earlier bundles queued all 395 items; only scored items count towards coverage.
        if not is_scored(item['original_item_id']):
            continue
        record = records.get(item_id, {})
        complete = record.get('completion', {}).get('status') == 'complete'
        reviewed = record.get('reviewed') or {}
        if complete and (not reviewed.get('annotator') or reviewed != record.get('draft')):
            raise ValueError('Completed review lacks a matching explicit snapshot')
        for answer in item['answers']:
            binding = answer['binding']
            if (binding['response_sha256'] != hashlib.sha256(answer['raw'].encode()).hexdigest()
                    or answer['id'] != digest(binding)):
                raise ValueError('Review answer identity differs from response text')
            key = (answer['system_id'], answer['task'], binding['run_id'])
            group = groups.setdefault(key, Counter(system=key[0], task=key[1], run_id=key[2]))
            group['total'] += 1
            group['technical_failures'] += int(answer['technical_failure'])
            positive = bool(answer['auto_score'])
            group['automatic_positive_total' if positive else 'automatic_nonpositive_total'] += 1
            judgment = reviewed.get('answers', {}).get(answer['id'], {}) if complete else {}
            quality = judgment.get('quality', '')
            if quality and (judgment.get('system_id') != answer['system_id']
                            or quality not in {'correct', 'wrong', 'partial', 'undecidable', 'no_answer'}):
                raise ValueError('Human judgment does not match its answer')
            if not quality:
                group['not_reviewed'] += 1
                continue
            group['reviewed'] += 1
            group['human_' + quality] += 1
            group['automatic_positive_reviewed' if positive else 'automatic_nonpositive_reviewed'] += 1
            if not answer['technical_failure']:
                if not positive and quality == 'correct':
                    group['automatic_nonpositive_human_correct'] += 1
                if positive and quality == 'wrong':
                    group['automatic_positive_human_wrong'] += 1
            timestamps.append(reviewed['updated_at'])
            initial = record.get('initial_interpretation') or {}
            if initial.get('captured_at'):
                starts.append(initial['captured_at'])
    fields = ('automatic_positive_total', 'automatic_nonpositive_total', 'technical_failures',
              'reviewed', 'not_reviewed', 'human_correct', 'human_wrong', 'human_partial',
              'human_undecidable', 'human_no_answer', 'automatic_positive_reviewed',
              'automatic_nonpositive_reviewed', 'automatic_nonpositive_human_correct',
              'automatic_positive_human_wrong')
    rows = []
    for group in groups.values():
        row = dict(group)
        row.update({field: group[field] for field in fields})
        row['unresolved'] = group['not_reviewed'] + group['human_undecidable']
        rows.append(row)
    pace = None
    if len(timestamps) >= 20 and starts:
        from datetime import datetime
        end = sorted(timestamps)[19]
        elapsed = (datetime.fromisoformat(end) - datetime.fromisoformat(min(starts))).total_seconds()
        if elapsed > 0:
            pace = {'first_20_wall_minutes': elapsed / 60,
                    'remaining_wall_minutes_estimate': elapsed / 20 / 60 * sum(r['not_reviewed'] for r in rows),
                    'note': 'Wall-clock estimate includes breaks; the initial queue is not a representative sample.'}
    return {'rows': rows, 'pace': pace, 'annotations_present': annotations is not None,
            'source_identity': dataset['source_identity']}


def export_encoder_review(directory, test_path, output_path, *, expected_run_id,
                          expected_identity, expected_predictions_sha256):
    """Queue identified encoder candidate choices; this is selection, not generation."""
    from hebrew_acronyms.encoder_test_results import load_encoder_test
    output_path, directory = Path(output_path), Path(directory)
    if output_path.exists():
        raise FileExistsError(output_path)
    saved = load_encoder_test(directory, test_path, expected_run_id=expected_run_id,
                              expected_identity=expected_identity,
                              expected_predictions_sha256=expected_predictions_sha256)
    items = []
    for row, record in zip(saved['rows'], saved['records']):
        if not is_scored(row['item_id']):
            continue
        raw = record['selected_candidate'] or ''
        binding = {'item_id': row['item_id'], 'task': 'select', 'run_id': expected_run_id,
                   'response_sha256': hashlib.sha256(raw.encode()).hexdigest()}
        answer_id = digest(binding)
        technical = record['status'] != 'ok'
        stratum = 'technical_failure' if technical else 'automatic_positive' if record['correct'] else 'automatic_nonpositive'
        answer = {'id': answer_id, 'system_id': 'dictabert_select', 'task': 'selection',
                  'response_type': 'text', 'raw': raw, 'decoded': raw,
                  'option_mapping': [], 'shown_order': [], 'omitted_candidates': [],
                  'missing': not bool(raw), 'status': record['status'], 'technical_failure': technical,
                  'auto_score': record['correct'], 'auto_valid': not technical,
                  'auto_score_rule': 'exact_candidate_match', 'mechanical_flags': [stratum],
                  'source_file': str(directory / 'predictions.jsonl'), 'binding': binding,
                  'prompt_sha256': None}
        spans = [{'start': m.start(), 'end': m.end()} for m in re.finditer(
            re.escape(row['acronym'].translate(QUOTES)), row['sentence'].translate(QUOTES))]
        items.append({'id': answer_id, 'original_item_id': row['item_id'],
                      'sentence': row['sentence'], 'acronym': row['acronym'], 'source': row.get('source', ''),
                      'origin': 'מחובר ב־AI' if row.get('label_origin') == 'claude_authored' else 'לפי מקור שמור',
                      'source_metadata': row, 'gold': row['gold_expansion'],
                      'candidates': row['candidates'].split('|'), 'answers': [answer], 'findings': [stratum],
                      'selection_reasons': ['בחירת מועמד של המודל המאומן; אין העברת שיפוטים היסטוריים'],
                      'prior_exposure': 'unknown', 'target_spans': spans[:1], 'target_span_verified': False,
                      'target_warning': 'הופעה ראשונה לפי מדיניות ההרצה האוטומטית; לא תיוג מיקום אנושי.',
                      'acronym_type_proxy': '', 'sampling_stratum': stratum})
    sources = [{'path': str(p.resolve()), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                'bytes': p.stat().st_size} for p in
               (directory/'manifest.json', directory/'predictions.jsonl', Path(test_path))]
    systems = [{'id': 'dictabert_select', 'blind_id': 'מערכת 1', 'name': 'DictaBERT — בחירה',
                'task': 'selection', 'response_type': 'text'}]
    return _write_review_bundle(items, systems, sources, {expected_run_id}, output_path)
