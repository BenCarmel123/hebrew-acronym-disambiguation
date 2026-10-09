"""Small post-experiment qualitative protocol; original annotations remain untouched."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import threading
from .human_review_server import atomic_json, now, ReviewStore

PROTOCOL = 'qualitative-generation-v1'
MODELS = ('qwen', 'gemini')
LABELS = {'fits', 'not_fits', 'unsure'}
MAPPING = {'correct': 'fits', 'wrong': 'not_fits', 'undecidable': 'unsure'}
GROUPS = {'generation_fail_selection_pass': 'כשל ביצירה והצלחה בבחירה של אותו מודל',
          'both_fail': 'כשל ביצירה ובבחירה של אותו מודל',
          'both_generation_pass': 'שתי תשובות היצירה קיבלו ציון נכון'}
LIMITATIONS = 'בדיקה איכותנית אבחונית לאחר הניסוי ובהצגת הייחוס. אינה בדיקה עצמאית ועיוורת או מדגם מייצג; אין לחשב ממנה דיוק מתוקן לכל הבנצ׳מרק. פער בציונים אינו מסביר כשלעצמו את סיבתו. הפרוטוקול אינו דורש חיפוש חיצוני; אין שיפוט סמנטי אוטומטי.'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':')).encode()).hexdigest()


def eligible(item):
    scores = {a['system_id']: a.get('auto_score') for a in item['answers']}
    a = [m for m in MODELS if scores.get(m + '_generate') is False and scores.get(m + '_select') is True]
    b = [m for m in MODELS if scores.get(m + '_generate') is False and scores.get(m + '_select') is False]
    c = all(scores.get(m + '_generate') is True for m in MODELS)
    return {'generation_fail_selection_pass': a, 'both_fail': b, 'both_generation_pass': list(MODELS) if c else []}


def build_plan(dataset, legacy):
    """Select by saved scores and metadata only, never by answer semantics."""
    targets = dict(zip(GROUPS, (12, 4, 4)))
    items = {i['id']: i for i in dataset['items']}
    pools = {g: [i for i in items.values() if eligible(i)[g]] for g in GROUPS}
    selected, reasons = [], {}
    counts, acronym_counts, sources, models, types = (Counter() for _ in range(5))

    def add(item, group, retained=False):
        witnesses = eligible(item)[group]
        witness = min(witnesses, key=lambda m: (models[m], m))
        selected.append(item['id'])
        reasons[item['id']] = {'group': group, 'description': GROUPS[group], 'qualifying_models': witnesses,
                               'balance_witness': witness, 'retained_prior_work': retained}
        counts[group] += 1; acronym_counts[item['acronym']] += 1
        sources[item.get('source', '')] += 1; models[witness] += 1; types[item.get('acronym_type_proxy', '')] += 1

    # Preserve eligible previously reviewed items before selecting anything new.
    for item_id in sorted(legacy.get('records', {})):
        if item_id not in items or len(selected) == 20:
            continue
        choices = [g for g in GROUPS if eligible(items[item_id])[g] and counts[g] < targets[g]]
        if choices:
            add(items[item_id], min(choices, key=lambda g: (len(pools[g]) / targets[g], g)), True)
    # Reserve the small double-failure group before filling the larger group.
    for group in ('both_fail', 'both_generation_pass', 'generation_fail_selection_pass'):
        while counts[group] < targets[group]:
            candidates = [i for i in pools[group] if i['id'] not in reasons]
            if not candidates:
                break
            def key(i):
                return (acronym_counts[i['acronym']], min(models[m] for m in eligible(i)[group]),
                        sources[i.get('source', '')], types[i.get('acronym_type_proxy', '')],
                        digest([PROTOCOL, i['id']]))
            add(min(candidates, key=key), group)
    # Deterministically interleave groups so an early stop is not one contiguous stratum.
    retained = [i for i in selected if reasons[i]['retained_prior_work']]
    pending = {g: [i for i in selected if reasons[i]['group'] == g and i not in retained] for g in GROUPS}
    queue = retained[:]
    while any(pending.values()):
        for g in GROUPS:
            if pending[g]: queue.append(pending[g].pop(0))
    plan = {'protocol_version': PROTOCOL, 'queue': queue, 'selection_reasons': reasons,
            'targets': targets, 'actual': dict(counts), 'eligible_pool_counts': {g: len(v) for g, v in pools.items()},
            'source_counts': dict(sources), 'distinct_acronyms': len(acronym_counts),
            'balance_witness_counts': dict(models), 'type_proxy_counts': dict(types),
            'selection_method': 'Deterministic saved-score eligibility; prior eligible work retained; no duplicate item; prefer unused acronyms, less represented model witness/source/length proxy; hash tie-break; interleave groups.',
            'adjustments': [f'{g}: requested {targets[g]}, selected {counts[g]}' for g in GROUPS if counts[g] != targets[g]],
            'limitations': LIMITATIONS, 'legacy_retained_ids': retained,
            'source_identity': dataset.get('source_identity', dataset['dataset_id'])}
    plan['plan_id'] = PROTOCOL + '-' + digest(plan)[:16]
    return plan


class ShortStore:
    def __init__(self, dataset, path, legacy, legacy_history=''):
        self.dataset = dataset
        self.plan = dataset['short_plan']
        self.path = Path(path)
        self.items = {i['id']: i for i in dataset['items']}
        self.lock = threading.RLock()
        self.answers = {}
        for item_id in self.plan['queue']:
            generations = {a['system_id']: a for a in self.items[item_id]['answers']}
            self.answers[item_id] = {digest([PROTOCOL, item_id, m])[:16]: generations[m + '_generate'] for m in MODELS}
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            self.validate(self.state)
        else:
            names = {r.get('draft', {}).get('annotator') for r in legacy.get('records', {}).values()} - {None, ''}
            reviewer = next(iter(names)) if len(names) == 1 else 'מתייג מקומי'
            self.state = {'protocol_version': PROTOCOL, 'plan_id': self.plan['plan_id'], 'plan': copy.deepcopy(self.plan),
                          'source_identity': dataset.get('source_identity', dataset['dataset_id']),
                          'provenance': copy.deepcopy(dataset['provenance']), 'revision': 0, 'records': {},
                          'reviewer': reviewer, 'legacy_snapshot': copy.deepcopy(legacy), 'legacy_history': legacy_history,
                          'legacy_sha256': digest(legacy), 'summary_exposures': []}
            for item_id in self.plan['queue']:
                prior = legacy.get('records', {}).get(item_id, {})
                annotation = prior.get('draft') or prior.get('reviewed') or {}
                if not annotation: continue
                record = self.blank()
                record.update(note=annotation.get('note', ''), example=bool(annotation.get('example')),
                              suspect=bool(set(annotation.get('item_problems', [])) - {'valid'}))
                record['inherited_from'] = {'schema': legacy.get('schema_version'), 'item_id': item_id,
                                            'updated_at': annotation.get('updated_at'), 'annotator': annotation.get('annotator')}
                record['unmapped_legacy'] = {}
                for opaque, answer in self.answers[item_id].items():
                    j = annotation.get('answers', {}).get(answer['id'], {})
                    if j.get('quality') in MAPPING:
                        record['judgments'][opaque] = {'label': MAPPING[j['quality']], 'origin': 'legacy_explicit_judgment',
                                                     'original_quality': j['quality'], 'annotator': annotation.get('annotator'),
                                                     'judged_at': annotation.get('updated_at')}
                    elif j.get('quality'):
                        record['unmapped_legacy'][opaque] = copy.deepcopy(j)
                self.state['records'][item_id] = record
            self.validate(self.state)
            atomic_json(self.path, self.state)

    @staticmethod
    def blank():
        return {'judgments': {}, 'suspect': False, 'example': False, 'note': '', 'exposure': {}}

    def validate(self, state):
        if state.get('protocol_version') != PROTOCOL or state.get('plan_id') != self.plan['plan_id']:
            raise ValueError('מסלול או גרסת פרוטוקול אינם תואמים; אין לדרוס עבודה קודמת')
        if ReviewStore.manifest(state.get('provenance')) != ReviewStore.manifest(self.dataset['provenance']):
            raise ValueError('Source provenance mismatch')
        if state.get('source_identity') != self.dataset.get('source_identity', self.dataset['dataset_id']):
            raise ValueError('Source identity mismatch')
        if type(state.get('revision')) is not int or state['revision'] < 0:
            raise ValueError('Invalid revision')
        for item_id, r in state['records'].items():
            if item_id not in self.answers: raise ValueError('Unknown short-route item')
            for aid, j in r['judgments'].items():
                if aid not in self.answers[item_id] or j.get('label') not in LABELS:
                    raise ValueError('Invalid short judgment')
            if type(r['suspect']) is not bool or type(r['example']) is not bool or not isinstance(r['note'], str):
                raise ValueError('Invalid optional fields')
        if digest(state['legacy_snapshot']) != state['legacy_sha256']:
            raise ValueError('Original annotation snapshot changed')

    def completion(self, record):
        n = len(record['judgments'])
        return {'status': 'complete' if n == 2 else ('partial' if n else 'unreviewed'), 'judged_answers': n, 'total_answers': 2}

    def counts(self):
        completions = [self.completion(r) for r in self.state['records'].values()]
        return {'complete_items': sum(c['status'] == 'complete' for c in completions),
                'partial_items': sum(c['status'] == 'partial' for c in completions),
                'reviewed_items': sum(c['judged_answers'] > 0 for c in completions),
                'judged_answers': sum(c['judged_answers'] for c in completions),
                'total_items': len(self.plan['queue']), 'total_answers': len(self.plan['queue']) * 2}

    def snapshot(self):
        with self.lock:
            # Deliberate allowlist: legacy scores, system IDs and sampling reasons never reach judgment UI.
            records = {}
            for item_id, r in self.state['records'].items():
                records[item_id] = {k: copy.deepcopy(r[k]) for k in ('suspect', 'example', 'note')}
                records[item_id].update(judgments={k: {'label': v['label']} for k, v in r['judgments'].items()},
                                       completion=self.completion(r), prior_reused=bool(r.get('inherited_from')))
            return {'protocol_version': PROTOCOL, 'plan_id': self.plan['plan_id'], 'revision': self.state['revision'],
                    'queue': self.plan['queue'], 'records': records, 'counts': self.counts(), 'reviewer': self.state['reviewer']}

    def commit(self, candidate, action, item_id=None):
        candidate['revision'] += 1; candidate['saved_at'] = now()
        self.validate(candidate)
        event = {'time': now(), 'action': action, 'item_id': item_id, 'revision': candidate['revision'],
                 'protocol_version': PROTOCOL, 'record': candidate['records'].get(item_id),
                 'summary_exposure': candidate['summary_exposures'][-1:] if action == 'summary' else []}
        # Atomic durable state is authoritative; append-only history preserves previous judgments.
        import os
        with self.path.with_suffix('.history.jsonl').open('a', encoding='utf-8') as out:
            out.write(json.dumps(event, ensure_ascii=False) + '\n'); out.flush(); os.fsync(out.fileno())
        atomic_json(self.path, candidate); self.state = candidate
        return self.snapshot()

    def transact(self, action, payload):
        with self.lock:
            if payload.get('revision') != self.state['revision']:
                raise ValueError('חלון אחר שינה את העבודה. יש לטעון מחדש לפני שמירה.')
            candidate = copy.deepcopy(self.state)
            if action == 'summary':
                candidate['summary_exposures'].append({'at': now(), 'items': list(candidate['records']),
                                                       'content': ['model_identities', 'automatic_scores', 'selection_reasons'],
                                                       'reason': 'explicit_summary_request'})
                state = self.commit(candidate, action)
                return {'state': state, 'summary': self.summary()}
            item_id = payload.get('item_id')
            if item_id not in self.answers: raise ValueError('פריט אינו במסלול הקצר')
            r = candidate['records'].setdefault(item_id, self.blank())
            if action == 'open':
                r['exposure'].setdefault('reference_and_generation', {'at': now(), 'protocol': PROTOCOL,
                         'independent_attempt': False, 'judgment_context': 'reference_presented_before_judgment'})
            elif action == 'save':
                if 'reference_and_generation' not in r['exposure']: raise ValueError('יש לפתוח את המשפט לפני שמירה')
                incoming = payload.get('judgments', {})
                if not isinstance(incoming, dict) or not set(incoming).issubset(self.answers[item_id]):
                    raise ValueError('Unknown short answer')
                judgments = {}
                for aid, label in incoming.items():
                    if label == '': continue
                    if label not in LABELS: raise ValueError('Invalid short label')
                    old = r['judgments'].get(aid)
                    judgments[aid] = old if old and old['label'] == label else {'label': label, 'origin': PROTOCOL,
                                                   'annotator': self.state['reviewer'], 'judged_at': now()}
                r.update(judgments=judgments, suspect=payload.get('suspect', False), example=payload.get('example', False),
                         note=payload.get('note', ''), updated_at=now())
            elif action == 'details':
                if 'reference_and_generation' not in r['exposure']: raise ValueError('יש לפתוח את המשפט תחילה')
                r['exposure'].setdefault('case_details', {'at': now(), 'content': ['model_identities', 'automatic_scores', 'selection_reasons', 'candidates', 'other_answers'], 'reason': 'explicit_details_request'})
            else: raise ValueError('Unknown short action')
            state = self.commit(candidate, action, item_id)
            if action == 'open': return {'state': state, 'item': self.public_item(item_id)}
            if action == 'details': return {'state': state, 'details': self.details(item_id)}
            return state

    def public_item(self, item_id):
        i = self.items[item_id]
        return {'id': item_id, 'sentence': i['sentence'], 'acronym': i['acronym'], 'target_spans': i.get('target_spans', []),
                'gold': i['gold'], 'answers': [{'id': aid, 'code': 'אב'[idx], 'text': a.get('raw') or 'לא נשמרה תשובה'}
                                             for idx, (aid, a) in enumerate(self.answers[item_id].items())],
                'prior_reused': bool(self.state['records'].get(item_id, {}).get('inherited_from'))}

    def details(self, item_id):
        i = self.items[item_id]
        return {'id': item_id, 'candidates': i['candidates'], 'source': i['source'], 'source_metadata': i['source_metadata'],
                'selection_reason': self.plan['selection_reasons'][item_id],
                'answers': [{'model': a['system_id'], 'task': a['task'], 'raw': a.get('raw'), 'decoded': a.get('decoded'),
                             'auto_score': a.get('auto_score')} for a in i['answers']],
                'legacy_record': self.state['legacy_snapshot'].get('records', {}).get(item_id)}

    def case(self, item_id):
        i, r = self.items[item_id], self.state['records'][item_id]
        answers = []
        for idx, (aid, a) in enumerate(self.answers[item_id].items()):
            j = r['judgments'].get(aid, {})
            label, score = j.get('label', ''), a.get('auto_score')
            mismatch = ((label == 'fits') != score) if label in {'fits', 'not_fits'} and type(score) is bool else None
            answers.append({'code': 'אב'[idx], 'model': a['system_id'], 'text': a.get('raw'), 'label': label,
                            'auto_score': score, 'comparison': 'disagreement' if mismatch else ('agreement' if mismatch is False else 'undetermined'),
                            'judgment_origin': j.get('origin'), 'original_quality': j.get('original_quality')})
        return {'id': item_id, 'sentence': i['sentence'], 'gold': i['gold'], 'note': r['note'], 'answers': answers,
                'suspect': r['suspect'], 'example': r['example'], 'source': i['source'],
                'group': self.plan['selection_reasons'][item_id]['group'], 'completion': self.completion(r),
                'inherited_from': r.get('inherited_from'), 'legacy_record': self.state['legacy_snapshot'].get('records', {}).get(item_id)}

    def summary(self):
        with self.lock:
            cases = [self.case(i) for i in self.plan['queue'] if i in self.state['records']]
            reviewed = [c for c in cases if c['completion']['judged_answers']]
            return {'protocol_version': PROTOCOL, 'counts': self.counts(), 'composition': {
                'selected': self.plan['actual'], 'reviewed': dict(Counter(c['group'] for c in reviewed)),
                'completed': dict(Counter(c['group'] for c in reviewed if c['completion']['status'] == 'complete')),
                'sources_selected': self.plan['source_counts'], 'sources_reviewed': dict(Counter(c['source'] for c in reviewed)),
                'distinct_acronyms': self.plan['distinct_acronyms'], 'adjustments': self.plan['adjustments']},
                'group_labels': GROUPS, 'disagreements': [c for c in reviewed if any(a['comparison'] == 'disagreement' for a in c['answers'])],
                'suspicions': [c for c in cases if c['suspect']], 'examples': [c for c in cases if c['example']],
                'unresolved': [c for c in cases if any(a['label'] == 'unsure' for a in c['answers'])],
                'incomplete_ids': [i for i in self.plan['queue'] if i not in self.state['records'] or self.completion(self.state['records'][i])['status'] != 'complete'],
                'cases': reviewed, 'limitations': LIMITATIONS,
                'evidence_note': 'הספירות, הרכב המדגם והפערים הם עובדות מחושבות; התוויות, הסימונים וההערות הם החלטות המתייג. חשדות אינם תיקון מאומת של הייחוס. אין הסבר סמנטי מוסק אוטומטית.'}

    def export(self):
        with self.lock:
            result = copy.deepcopy(self.state)
            result['summary'] = self.summary()
            history = self.path.with_suffix('.history.jsonl')
            result['short_history'] = history.read_text() if history.exists() else ''
            return result

    def markdown(self):
        s = self.summary(); c = s['counts']
        lines = ['# בדיקה איכותנית ממוקדת — יצירה חופשית', '', s['limitations'], '', s['evidence_note'], '',
                 f"נבדקו {c['reviewed_items']} משפטים: {c['complete_items']} מלאים ו־{c['partial_items']} חלקיים במסלול הקצר; {c['judged_answers']} מתוך {c['total_answers']} תשובות סומנו. השלמה כאן מתייחסת לשתי תשובות יצירה בלבד.", '', '## הרכב בפועל', '']
        for group, label in GROUPS.items():
            lines.append(f"- {label}: נבחרו {s['composition']['selected'].get(group, 0)}, נבדקו {s['composition']['reviewed'].get(group, 0)}, הושלמו {s['composition']['completed'].get(group, 0)}.")
        lines.extend(['', 'מקורות שנבחרו: ' + json.dumps(s['composition']['sources_selected'], ensure_ascii=False), ''])
        label_names = {'fits': 'מתאימה להקשר', 'not_fits': 'לא מתאימה', 'unsure': 'לא בטוח', '': 'טרם נשפטה'}
        for key, title in [('disagreements', 'פערים בין שיפוט אנושי לניקוד'), ('suspicions', 'חשדות לייחוס או להקשר'), ('examples', 'דוגמאות שסומנו למאמר'), ('unresolved', 'מקרים לא מוכרעים'), ('cases', 'כל המשפטים שנבדקו')]:
            lines.extend(['## ' + title, ''])
            if not s[key]: lines.extend(['אין כרגע.', ''])
            for case in s[key]:
                lines.extend(['### ' + case['id'], '', 'משפט: ' + case['sentence'], '', 'ייחוס שמור (עשוי להיות שגוי): ' + case['gold'], ''])
                for a in case['answers']:
                    lines.extend([f"- {a['model']} / {a['code']}: {a['text']}", f"  שיפוט אנושי: {label_names[a['label']]}; ציון אוטומטי שמור: {a['auto_score']}; מקור השיפוט: {a['judgment_origin'] or 'טרם נשפט'}."])
                if case['note']: lines.extend(['', 'הערת המתייג: ' + case['note']])
                if case['inherited_from']: lines.extend(['', 'שיפוטים תואמים נשמרו מהפרוטוקול הקודם; אין כאן תיוג עצמאי נוסף.'])
                lines.append('')
        lines.extend(['משפטים שלא הושלמו במסלול הקצר: ' + ', '.join(s['incomplete_ids']), ''])
        return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True); parser.add_argument('--legacy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Output exists; choose a fresh bundle path')
    dataset = json.loads(args.data.read_text()); legacy = json.loads(args.legacy.read_text())
    if ReviewStore.manifest(legacy['provenance']) != ReviewStore.manifest(dataset['provenance']): parser.error('Source mismatch')
    dataset['short_plan'] = build_plan(dataset, legacy)
    atomic_json(args.output, dataset)
    print(json.dumps(dataset['short_plan'], ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
