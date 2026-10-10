"""Persisted per-item masking and qualitative tags; previous protocols are immutable inputs."""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import copy
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import threading
import unicodedata

from .human_review_server import ReviewStore, atomic_json, now
from .human_review_short import LABELS, MODELS, PROTOCOL as PREVIOUS_PROTOCOL, digest

PROTOCOL = 'qualitative-generation-v2'
TAGS = {'gibberish', 'inflection', 'punctuation', 'spelling', 'equivalent', 'extra_text'}
PHASES = {'prior_protocol', 'before_reveal', 'after_reveal', 'unknown'}
LIMITATIONS = ('בדיקה איכותנית אבחונית עם ייחוס מוצג, עד 20 משפטים. אינה מדגם מייצג. '
               'זהות המודל והניקוד הוסתרו בממשק החדש לפני חשיפה; סגנון התשובה עשוי לרמוז לזהות. '
               'חשיפות קודמות נשמרו ונלקחות בחשבון. אין כאן טענה לעיוורון כפול או להעדר הטיה. '
               'תגית מתארת תופעה ואינה קובעת נכונות או מסבירה לבדה פער בניקוד.')


class MaskedStore:
    protocol = PROTOCOL
    limitations = LIMITATIONS

    def __init__(self, dataset, path, previous_short_bundle, previous_short_history=''):
        self.dataset = copy.deepcopy(dataset)
        self.plan = self.dataset['short_plan']
        self.path = Path(path)
        self.lock = threading.RLock()
        self.items = {i['id']: i for i in dataset['items']}
        queue = self.plan['queue']
        if len(queue) > 20 or len(queue) != len(set(queue)):
            raise ValueError('Masked route must contain at most 20 distinct items')
        self.answers = {item_id: {a['id']: a for a in self.items[item_id]['answers']
                                 if a['system_id'] in {m + '_generate' for m in MODELS}} for item_id in queue}
        if any(len(a) != 2 for a in self.answers.values()):
            raise ValueError('Each item must have exactly two generation answers')
        if self.path.exists():
            self.state = json.loads(self.path.read_text(encoding='utf-8'))
            self.validate(self.state)
        else:
            previous = copy.deepcopy(previous_short_bundle)
            if previous.get('protocol_version') != PREVIOUS_PROTOCOL or previous.get('plan_id') != self.plan['plan_id']:
                raise ValueError('Previous protocol/plan mismatch')
            if (ReviewStore.manifest(previous.get('provenance')) != ReviewStore.manifest(dataset['provenance'])
                    or previous.get('source_identity') != dataset.get('source_identity', dataset['dataset_id'])):
                raise ValueError('Previous source identity mismatch')
            presentation = {}
            for item_id, answers in self.answers.items():
                order = list(answers)
                secrets.SystemRandom().shuffle(order)
                presentation[item_id] = [{'answer_id': aid, 'token': secrets.token_hex(16)} for aid in order]
            self.state = {'protocol_version': self.protocol, 'plan_id': self.plan['plan_id'],
                          'source_identity': dataset.get('source_identity', dataset['dataset_id']),
                          'provenance': copy.deepcopy(dataset['provenance']), 'revision': 0,
                          'reviewer': previous.get('reviewer', 'מתייג מקומי'), 'records': {}, 'backup_secret': secrets.token_hex(32),
                          'presentation': presentation, 'previous_protocol': previous,
                          'previous_history': previous_short_history, 'previous_sha256': digest(previous),
                          'previous_history_sha256': hashlib.sha256(previous_short_history.encode()).hexdigest(),
                          'revealed': False, 'reveal_events': [], 'pre_reveal_snapshot': None,
                          'summary_exposures': []}
            for item_id, old in previous.get('records', {}).items():
                if item_id not in self.answers:
                    continue
                r = self.blank()
                r.update(suspect=old.get('suspect', False), example=old.get('example', False),
                         previous_note=old.get('note', ''), previous_exposure=copy.deepcopy(old.get('exposure', {})),
                         prior_reused=bool(old.get('judgments') or old.get('inherited_from')))
                for aid, answer in self.answers[item_id].items():
                    model = answer['system_id'].removesuffix('_generate')
                    old_token = digest([PREVIOUS_PROTOCOL, item_id, model])[:16]
                    old_j = old.get('judgments', {}).get(old_token)
                    if old_j and old_j.get('label') in LABELS:
                        r['judgments'][aid] = {'label': old_j['label'], 'tags': [], 'tag_status': 'not_marked',
                                              'label_phase': 'prior_protocol', 'tags_phase': None,
                                              'label_updated_at': old_j.get('judged_at'), 'tags_updated_at': None,
                                              'annotator': old_j.get('annotator'), 'origin': old_j.get('origin', PREVIOUS_PROTOCOL),
                                              'previous_judgment': copy.deepcopy(old_j),
                                              'exposure_evidence': self.prior_evidence(item_id, self.state, at=old_j.get('judged_at'), historical=True)}
                self.state['records'][item_id] = r
            self.validate(self.state)
            atomic_json(self.path, self.state)

    @staticmethod
    def blank():
        return {'judgments': {}, 'suspect': False, 'example': False, 'note': '', 'exposure': {}}

    def prior_evidence(self, item_id, state=None, at=None, historical=False):
        state = state or self.state
        previous = state['previous_protocol']
        record = previous.get('records', {}).get(item_id, {})
        legacy = previous.get('legacy_snapshot', {}).get('records', {}).get(item_id, {})
        uncertain = False
        future = 0
        def relevant(events):
            nonlocal uncertain, future
            if not historical:
                return copy.deepcopy(events)
            result = []
            for event in events:
                stamp = event if isinstance(event, str) else event.get('at', event.get('time')) if isinstance(event, dict) else None
                try:
                    cutoff, moment = datetime.fromisoformat(at), datetime.fromisoformat(stamp)
                    if cutoff.tzinfo is None or moment.tzinfo is None:
                        raise ValueError('Timezone required')
                    if moment <= cutoff:
                        result.append(copy.deepcopy(event))
                    else:
                        future += 1
                except (TypeError, ValueError):
                    uncertain = True
            return result
        global_events = relevant(previous.get('summary_exposures', []))
        detail_events = relevant([r.get('exposure', {}).get('case_details') for r in previous.get('records', {}).values()
                                  if r.get('exposure', {}).get('case_details')])
        legacy_identity_events = relevant([{'at': r['exposure']['identities'], 'kind': 'identities'}
                                          for r in previous.get('legacy_snapshot', {}).get('records', {}).values()
                                          if 'identities' in r.get('exposure', {})])
        local_scores = relevant([legacy['exposure']['auto_scores']] if 'auto_scores' in legacy.get('exposure', {}) else [])
        prior = (legacy.get('draft') or legacy.get('reviewed') or {}).get('prior_exposure')
        if historical and (not at or prior == 'yes'):
            uncertain = True
        evidence = {'previous_global_summary': global_events, 'previous_global_details': detail_events,
                    'legacy_global_identity_exposure': legacy_identity_events, 'legacy_item_score_exposure': local_scores,
                    'legacy_prior_exposure': prior, 'evaluated_at_judgment_time': at if historical else None,
                    'future_events_excluded': future,
                    'previous_item_exposure': copy.deepcopy(record.get('exposure', {})) if not historical else {},
                    'legacy_item_exposure': copy.deepcopy(legacy.get('exposure', {})) if not historical else {}}
        known = bool(global_events or detail_events or legacy_identity_events or local_scores or (prior == 'yes' and not historical))
        prior_open = relevant([record['exposure']['reference_and_generation']] if 'reference_and_generation' in record.get('exposure', {}) else [])
        evidence['previous_reference_exposure'] = prior_open
        incomplete = uncertain or 'summary_exposures' not in previous or prior == 'unknown' or bool(prior_open) or historical
        evidence['status'] = 'after_reveal' if known else ('unknown' if incomplete else 'before_reveal')
        return evidence

    def phase(self, item_id):
        return 'after_reveal' if self.state['revealed'] else self.prior_evidence(item_id)['status']

    def current_evidence(self, item_id):
        evidence = self.prior_evidence(item_id)
        evidence['current_protocol_reveal'] = copy.deepcopy(self.state['reveal_events'])
        evidence['status'] = self.phase(item_id)
        return evidence

    def validate(self, state):
        if state.get('protocol_version') != self.protocol or state.get('plan_id') != self.plan['plan_id']:
            raise ValueError('Masked protocol or sampling plan mismatch')
        if (ReviewStore.manifest(state.get('provenance')) != ReviewStore.manifest(self.dataset['provenance'])
                or state.get('source_identity') != self.dataset.get('source_identity', self.dataset['dataset_id'])):
            raise ValueError('Source identity mismatch')
        if type(state.get('revision')) is not int or state['revision'] < 0:
            raise ValueError('Invalid revision')
        if digest(state['previous_protocol']) != state['previous_sha256']:
            raise ValueError('Previous protocol snapshot changed')
        if hashlib.sha256(state['previous_history'].encode()).hexdigest() != state['previous_history_sha256']:
            raise ValueError('Previous history changed')
        presentation = state.get('presentation', {})
        if set(presentation) != set(self.answers):
            raise ValueError('Presentation items mismatch')
        tokens = []
        for item_id, slots in presentation.items():
            if len(slots) != len(self.answers[item_id]) or {s['answer_id'] for s in slots} != set(self.answers[item_id]):
                raise ValueError('Presentation answer mapping mismatch')
            for slot in slots:
                token = slot.get('token')
                if not isinstance(token, str) or len(token) != 32 or any(c not in '0123456789abcdef' for c in token):
                    raise ValueError('Invalid opaque answer token')
                tokens.append(token)
        if len(set(tokens)) != len(tokens):
            raise ValueError('Duplicate opaque token')
        if not isinstance(state.get('records'), dict):
            raise ValueError('Invalid records')
        for item_id, r in state['records'].items():
            if item_id not in self.answers or not isinstance(r.get('judgments'), dict):
                raise ValueError('Unknown item or invalid judgments')
            if type(r.get('suspect')) is not bool or type(r.get('example')) is not bool or not isinstance(r.get('note'), str):
                raise ValueError('Invalid optional fields')
            for aid, j in r['judgments'].items():
                if aid not in self.answers[item_id] or j.get('label', '') not in LABELS | {''}:
                    raise ValueError('Invalid judgment')
                tags = j.get('tags', [])
                if not isinstance(tags, list) or any(not isinstance(t, str) or t not in TAGS for t in tags) or len(set(tags)) != len(tags):
                    raise ValueError('Invalid tags')
                if j.get('tag_status') != ('marked' if tags else 'not_marked'):
                    raise ValueError('Invalid tag status')
                for field in ('label_phase', 'tags_phase'):
                    if j.get(field) is not None and j[field] not in PHASES:
                        raise ValueError('Invalid exposure phase')
        if type(state.get('revealed')) is not bool:
            raise ValueError('Invalid reveal status')
        frozen = state.get('pre_reveal_snapshot')
        if state['revealed'] and (not frozen or digest(frozen) != state.get('pre_reveal_sha256')):
            raise ValueError('Pre-reveal snapshot missing or changed')

    @staticmethod
    def completion(record):
        n = sum(j.get('label') in LABELS for j in record.get('judgments', {}).values())
        return {'status': 'complete' if n == 2 else ('partial' if n else 'unreviewed'), 'judged_answers': n, 'total_answers': 2}

    def counts(self, records=None):
        records = self.state['records'] if records is None else records
        cs = [self.completion(r) for r in records.values()]
        return {'complete_items': sum(c['status'] == 'complete' for c in cs),
                'partial_items': sum(c['status'] == 'partial' for c in cs),
                'reviewed_items': sum(c['judged_answers'] > 0 for c in cs),
                'judged_answers': sum(c['judged_answers'] for c in cs),
                'total_items': len(self.plan['queue']), 'total_answers': len(self.plan['queue']) * 2}

    def public_records(self, records):
        result = {}
        fields = ('label', 'tags', 'tag_status', 'label_phase', 'tags_phase', 'label_updated_at', 'tags_updated_at')
        for item_id, r in records.items():
            judgments = {}
            for slot in self.state['presentation'][item_id]:
                j = r.get('judgments', {}).get(slot['answer_id'], {})
                judgments[slot['token']] = {key: copy.deepcopy(j.get(key, [] if key == 'tags' else 'not_marked' if key == 'tag_status' else '' if key == 'label' else None)) for key in fields}
            result[item_id] = {k: copy.deepcopy(r[k]) for k in ('suspect', 'example', 'note')}
            result[item_id].update(judgments=judgments, completion=self.completion(r),
                                   prior_reused=bool(r.get('prior_reused')), has_previous_note=bool(r.get('previous_note')))
        return result

    def snapshot(self):
        with self.lock:
            return {'protocol_version': self.protocol, 'plan_id': self.plan['plan_id'], 'revision': self.state['revision'],
                    'queue': copy.deepcopy(self.plan['queue']), 'records': self.public_records(self.state['records']),
                    'counts': self.counts(), 'reviewer': self.state['reviewer'], 'revealed': self.state['revealed'],
                    'exposed': self.state['revealed'], 'prior_exposure': any(self.prior_evidence(i)['status'] != 'before_reveal' for i in self.plan['queue'])}

    def public_item(self, item_id, include_all=False):
        i = self.items[item_id]
        return {'id': item_id, 'sentence': i['sentence'], 'acronym': i['acronym'], 'gold': i['gold'],
                'target_spans': i.get('target_spans', []), 'prior_reused': bool(self.state['records'].get(item_id, {}).get('prior_reused')),
                'answers': [{'id': slot['token'], 'code': 'אב'[index], 'text': self.answers[item_id][slot['answer_id']].get('raw') or 'לא נשמרה תשובה'}
                            for index, slot in enumerate(self.state['presentation'][item_id])]}

    def commit(self, candidate, action, item_id=None):
        candidate['revision'] = self.state['revision'] + 1
        candidate['saved_at'] = now()
        self.validate(candidate)
        # Immutable original material and the first-reveal snapshot cannot be replaced by any operation.
        for key in ('previous_protocol', 'previous_history', 'presentation'):
            if candidate[key] != self.state[key]:
                raise ValueError('Immutable protocol material changed')
        if self.state.get('pre_reveal_snapshot') and candidate.get('pre_reveal_snapshot') != self.state['pre_reveal_snapshot']:
            raise ValueError('Pre-reveal snapshot is immutable')
        event = {'time': now(), 'action': action, 'item_id': item_id, 'revision': candidate['revision'],
                 'protocol_version': self.protocol, 'record': candidate['records'].get(item_id),
                 'reveal_event': candidate['reveal_events'][-1:] if action == 'reveal' else []}
        if action == 'authorize-continuation':
            event['manual_session'] = copy.deepcopy(candidate['manual_session'])
        if action.startswith('reuse-'):
            event['historical_reuse'] = copy.deepcopy(candidate.get('historical_reuse', {}))
            event['historical_reuse_exposures'] = copy.deepcopy(candidate.get('historical_reuse_exposures', []))
        if action in {'batch-open', 'batch-save'}:
            event['batch'] = copy.deepcopy(candidate['active_batch'])
            event['records'] = {key: copy.deepcopy(candidate['records'][key]) for key in candidate['active_batch']['item_ids']}
        if action in {'group-open', 'group-save', 'group-details'}:
            group_event = candidate['cross_context_events'][-1]
            event['group_event'] = copy.deepcopy(group_event)
            event['records'] = {key: copy.deepcopy(candidate['records'][key]) for key in group_event['item_ids']}
            event['group_session'] = copy.deepcopy(candidate['cross_context_sessions'][group_event['open_id']])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix('.history.jsonl').open('a', encoding='utf-8') as out:
            out.write(json.dumps(event, ensure_ascii=False) + '\n'); out.flush(); os.fsync(out.fileno())
        atomic_json(self.path, candidate)
        self.state = candidate
        return self.snapshot()

    def transact(self, action, payload):
        with self.lock:
            if payload.get('revision') != self.state['revision']:
                raise ValueError('חלון אחר שינה את העבודה. יש לטעון מחדש לפני שמירה.')
            candidate = copy.deepcopy(self.state)
            if action == 'reveal':
                if not candidate['revealed']:
                    candidate['pre_reveal_snapshot'] = {'captured_at': now(), 'revision': candidate['revision'],
                                                         'records': copy.deepcopy(candidate['records']),
                                                         'presentation': copy.deepcopy(candidate['presentation'])}
                    candidate['pre_reveal_sha256'] = digest(candidate['pre_reveal_snapshot'])
                    candidate['revealed'] = True
                    candidate['reveal_events'].append({'at': now(), 'scope': 'all_route_items', 'reason': 'explicit_finish_and_reveal',
                                                       'content': ['model_identities', 'original_scores', 'source_and_selection']})
                state = self.commit(candidate, action)
                return {'state': state, 'summary': self.summary(masked=False)}
            if action == 'summary':
                candidate['summary_exposures'].append({'at': now(), 'masked': not candidate['revealed']})
                state = self.commit(candidate, action)
                return {'state': state, 'summary': self.summary()}
            item_id = payload.get('item_id')
            if item_id not in self.answers:
                raise ValueError('Unknown masked-route item')
            r = candidate['records'].setdefault(item_id, self.blank())
            if action == 'open':
                r['exposure'].setdefault('reference_and_generation', {'at': now(), 'protocol': self.protocol,
                                  'independent_attempt': False, 'phase': self.phase(item_id)})
            elif action == 'details':
                if 'reference_and_generation' not in r['exposure']:
                    raise ValueError('Open the item before viewing details')
                r['exposure'].setdefault('candidates', {'at': now(), 'phase': self.phase(item_id), 'reason': 'explicit_masked_details'})
            elif action == 'save':
                if 'reference_and_generation' not in r['exposure']:
                    raise ValueError('Open the item before saving')
                mapping = {slot['token']: slot['answer_id'] for slot in candidate['presentation'][item_id]}
                labels, tags = payload.get('judgments', {}), payload.get('tags', {})
                if not isinstance(labels, dict) or not isinstance(tags, dict) or not (set(labels) | set(tags)).issubset(mapping):
                    raise ValueError('Unknown answer token')
                for token in set(labels) | set(tags):
                    aid = mapping[token]
                    old = r['judgments'].get(aid, {})
                    label = labels.get(token, old.get('label', ''))
                    selected = tags.get(token, old.get('tags', []))
                    if not isinstance(label, str) or label not in LABELS | {''}:
                        raise ValueError('Invalid judgment')
                    if not isinstance(selected, list) or any(not isinstance(t, str) or t not in TAGS for t in selected):
                        raise ValueError('Invalid tags')
                    selected = sorted(set(selected))
                    j = copy.deepcopy(old)
                    j.setdefault('label_phase', None); j.setdefault('tags_phase', None)
                    j.setdefault('label_updated_at', None); j.setdefault('tags_updated_at', None)
                    if label != old.get('label', ''):
                        j.update(label_phase=self.phase(item_id), label_updated_at=now(), annotator=self.state['reviewer'],
                                 origin=self.protocol, exposure_evidence=self.current_evidence(item_id))
                    if selected != old.get('tags', []):
                        j.update(tags_phase=self.phase(item_id), tags_updated_at=now(), tags_exposure_evidence=self.current_evidence(item_id))
                    j.update(label=label, tags=selected, tag_status='marked' if selected else 'not_marked')
                    r['judgments'][aid] = j
                r.update(suspect=payload.get('suspect', r['suspect']), example=payload.get('example', r['example']),
                         note=payload.get('note', r['note']), updated_at=now(), edit_phase=self.phase(item_id))
            else:
                raise ValueError('Unknown masked-route action')
            state = self.commit(candidate, action, item_id)
            if action == 'open':
                return {'state': state, 'item': self.public_item(item_id)}
            if action == 'details':
                i = self.items[item_id]
                return {'state': state, 'details': {'id': item_id, 'sentence': i['sentence'], 'gold': i['gold'],
                         'candidates': copy.deepcopy(i.get('candidates', [])), 'masked': True,
                         'exposure_note': 'פתיחת המועמדים תועדה; זהויות וניקוד אינם מוצגים.'}}
            return state

    def case(self, item_id, masked=True):
        r = self.state['records'][item_id]
        public = self.public_item(item_id, include_all=True)
        public_j = self.public_records({item_id: r})[item_id]['judgments']
        answers = []
        for slot, shown in zip(self.state['presentation'][item_id], public['answers']):
            aid = slot['answer_id']; a = self.answers[item_id][aid]; j = r['judgments'].get(aid, {})
            answer = dict(shown, **public_j[slot['token']])
            if not masked:
                score, label = a.get('auto_score'), j.get('label', '')
                mismatch = (label == 'fits') != score if label in {'fits', 'not_fits'} and type(score) is bool else None
                answer.update(original_answer_id=aid, answer_id=aid, raw=a.get('raw'), model=a['system_id'], auto_score=score,
                              comparison='disagreement' if mismatch else 'agreement' if mismatch is False else 'undetermined',
                              original_score_rule=a.get('auto_score_rule'), exposure_evidence=j.get('exposure_evidence'),
                              tags_exposure_evidence=j.get('tags_exposure_evidence'),
                              origin=j.get('origin'), previous_judgment=j.get('previous_judgment'))
            answers.append(answer)
        case = {'id': item_id, 'sentence': public['sentence'], 'gold': public['gold'], 'answers': answers,
                'suspect': r['suspect'], 'example': r['example'], 'note': r['note'], 'completion': self.completion(r),
                'prior_reused': bool(r.get('prior_reused'))}
        if not masked:
            case.update(source=self.items[item_id].get('source'), previous_note=r.get('previous_note', ''),
                        edit_phase=r.get('edit_phase'), updated_at=r.get('updated_at'),
                        selection_reason=self.plan['selection_reasons'][item_id])
        return case

    def summary(self, masked=None):
        with self.lock:
            masked = not self.state['revealed'] if masked is None else masked
            if not masked and not self.state['revealed']:
                raise ValueError('Explicit reveal required before full results')
            cases = [self.case(i, masked) for i in self.plan['queue'] if i in self.state['records']]
            tag_counts = Counter(t for r in self.state['records'].values() for j in r['judgments'].values() for t in j.get('tags', []))
            result = {'protocol_version': self.protocol, 'masked': bool(masked), 'counts': self.counts(), 'cases': cases,
                      'examples': [c for c in cases if c['example']], 'suspicions': [c for c in cases if c['suspect']],
                      'unresolved': [c for c in cases if any(a['label'] == 'unsure' for a in c['answers'])],
                      'tagged': [c for c in cases if any(a['tags'] for a in c['answers'])],
                      'tag_counts': dict(tag_counts), 'limitations': self.limitations,
                      'incomplete_ids': [i for i in self.plan['queue'] if self.completion(self.state['records'].get(i, {}))['status'] != 'complete']}
            if not masked:
                reviewed = [c for c in cases if c['completion']['judged_answers']]
                completed = [c for c in reviewed if c['completion']['status'] == 'complete']
                group = lambda c: self.plan['selection_reasons'][c['id']]['group']
                result['composition'] = {
                    'selected': dict(Counter(self.plan['selection_reasons'][i]['group'] for i in self.plan['queue'])),
                    'reviewed': dict(Counter(group(c) for c in reviewed)),
                    'completed': dict(Counter(group(c) for c in completed)),
                    'sources_selected': dict(Counter(self.items[i].get('source', '') for i in self.plan['queue'])),
                    'sources_reviewed': dict(Counter(c.get('source', '') for c in reviewed)),
                    'adjustments': copy.deepcopy(self.plan.get('adjustments', []))}
                result['accepted_human_rejected_auto'] = [c for c in cases if any(a['label'] == 'fits' and a['auto_score'] is False for a in c['answers'])]
                result['rejected_human_accepted_auto'] = [c for c in cases if any(a['label'] == 'not_fits' and a['auto_score'] is True for a in c['answers'])]
                result['pre_reveal_snapshot_preserved'] = True
            return result

    def export(self, masked=True):
        with self.lock:
            if not masked:
                if not self.state['revealed']:
                    raise ValueError('Explicit reveal required before full export')
                result = copy.deepcopy(self.state)
                result['summary'] = self.summary(False)
                history = self.path.with_suffix('.history.jsonl')
                result['history'] = history.read_text(encoding='utf-8') if history.exists() else ''
                return result
            result = self.snapshot()
            result['masked'] = True
            result['summary'] = self.summary(True)
            frozen = self.state.get('pre_reveal_snapshot')
            result['pre_reveal_snapshot'] = ({'captured_at': frozen['captured_at'], 'revision': frozen['revision'],
                                              'records': self.public_records(frozen['records'])} if frozen else None)
            result['backup_mac'] = hmac.new(self.state['backup_secret'].encode(), digest(result).encode(), hashlib.sha256).hexdigest()
            return result

    def restore_masked(self, bundle, revision):
        """Restore only against the existing server-side random mapping and immutable source history."""
        with self.lock:
            if revision != self.state['revision']:
                raise ValueError('State changed before restore')
            if (bundle.get('protocol_version') != self.protocol or bundle.get('plan_id') != self.state['plan_id']
                    or bundle.get('queue') != self.plan['queue'] or bundle.get('masked') is not True):
                raise ValueError('Masked backup protocol or queue mismatch')
            unsigned = copy.deepcopy(bundle)
            signature = unsigned.pop('backup_mac', '')
            expected = hmac.new(self.state['backup_secret'].encode(), digest(unsigned).encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(signature, expected):
                raise ValueError('Masked backup signature mismatch')
            candidate = copy.deepcopy(self.state)
            for item_id, public in bundle.get('records', {}).items():
                if item_id not in self.answers:
                    raise ValueError('Unknown restore item')
                mapping = {s['token']: s['answer_id'] for s in self.state['presentation'][item_id]}
                if set(public.get('judgments', {})) != set(mapping):
                    raise ValueError('Masked backup random handles do not match this store')
                r = candidate['records'].setdefault(item_id, self.blank())
                for token, supplied in public['judgments'].items():
                    aid = mapping[token]
                    j = r['judgments'].setdefault(aid, {})
                    j.update({k: copy.deepcopy(supplied.get(k)) for k in ('label', 'tags', 'tag_status', 'label_phase', 'tags_phase', 'label_updated_at', 'tags_updated_at')})
                    if self.state['revealed']:
                        j['restored_after_reveal_at'] = now()
                for key in ('note', 'suspect', 'example'):
                    r[key] = copy.deepcopy(public[key])
                r.update(edit_phase=self.phase(item_id), updated_at=now(), restored_at=now())
            self.validate(candidate)
            backup = self.path.with_name(self.path.name + '.before-restore-' + secrets.token_hex(6) + '.json')
            atomic_json(backup, self.state)
            return self.commit(candidate, 'restore_masked')

    def markdown(self, masked=True):
        summary = self.summary(masked)
        label_names = {'fits': 'מתאימה להקשר', 'not_fits': 'לא מתאימה', 'unsure': 'לא בטוח', '': 'טרם סומן'}
        tag_names = {'gibberish': 'ג׳יבריש / טקסט לא מובן', 'inflection': 'יחיד–רבים / נטייה',
                     'punctuation': 'רווחים / מקפים / גרשיים', 'spelling': 'הבדל כתיב',
                     'equivalent': 'ניסוח חלופי שקול', 'extra_text': 'טקסט עודף / סותר'}
        phase_names = {'prior_protocol': 'פרוטוקול קודם', 'before_reveal': 'לפני חשיפה',
                       'after_reveal': 'לאחר חשיפה מתועדת', 'unknown': 'מצב החשיפה אינו ידוע'}
        lines = ['# סיכום מוסתר' if masked else '# סיכום לאחר חשיפה', '', self.limitations, '',
                 f"{summary['counts']['complete_items']} פריטים הושלמו; {summary['counts']['judged_answers']} תשובות סומנו.", '']
        if not masked:
            from .human_review_short import GROUPS
            composition = summary['composition']
            lines += ['## הרכב המדגם שנבחר והחלק שנבדק', '']
            for group, label in GROUPS.items():
                lines += [f"- {label}: נבחרו {composition['selected'].get(group, 0)}, נבדקו {composition['reviewed'].get(group, 0)}, הושלמו {composition['completed'].get(group, 0)}."]
            lines += ['', 'מקורות שנבחרו: ' + json.dumps(composition['sources_selected'], ensure_ascii=False),
                      'מקורות שנבדקו: ' + json.dumps(composition['sources_reviewed'], ensure_ascii=False),
                      'החלוקה מתארת את המדגם האבחוני ואת החלק שנבדק בלבד; אינה אומדן לבנצ׳מרק.', '']
            if composition['adjustments']:
                lines += ['התאמות בפועל: ' + '; '.join(composition['adjustments']), '']
        for case in summary['cases']:
            lines += ['## ' + case['id'], '', case['sentence'], '', 'ייחוס: ' + str(case['gold']),
                      'חשד לבעיה בייחוס/במשפט: ' + ('כן' if case['suspect'] else 'לא סומן'),
                      'דוגמה למאמר: ' + ('כן' if case['example'] else 'לא סומן'), '']
            if not masked:
                lines += ['מקור: ' + str(case.get('source', '')), '']
            for answer in case['answers']:
                lines += [f"- תשובה {answer['code']}: {answer['text']}",
                          '  שיפוט: ' + label_names[answer['label']] + '; תגיות: ' + (', '.join(tag_names[t] for t in answer['tags']) or 'לא סומן') + '.',
                          '  שלב שיפוט: ' + phase_names.get(answer['label_phase'], 'טרם נשפט') + '; מועד: ' + str(answer['label_updated_at'] or 'לא זמין') + '.',
                          '  שלב תגיות: ' + phase_names.get(answer['tags_phase'], 'לא סומן') + '; מועד: ' + str(answer['tags_updated_at'] or 'לא סומן') + '.']
                if not masked:
                    lines += [f"  מזהה: {answer['original_answer_id']}; מודל: {answer['model']}; ציון מקורי: {answer['auto_score']}.",
                              '  חשיפה לפי הראיות בעת השיפוט: ' + phase_names.get((answer.get('exposure_evidence') or {}).get('status'), 'לא ידוע') + '.']
            if case['note']:
                lines += ['', 'הערה: ' + case['note']]
            if not masked and case.get('previous_note'):
                lines += ['', 'הערה מהפרוטוקול הקודם: ' + case['previous_note']]
            if not masked and case.get('edit_phase'):
                lines += ['', 'שלב עריכת הפריט האחרונה: ' + phase_names[case['edit_phase']] + '; מועד: ' + str(case.get('updated_at'))]
            lines.append('')
        return '\n'.join(lines)


CONTINUATION_PROTOCOL = 'qualitative-generation-v3'
FILTER_VERSION = 'generation-filter-v1'
QUOTE_EQUIVALENTS = str.maketrans({'׳': "'", '‘': "'", '’': "'", '״': '"', '“': '"', '”': '"'})


def technical_normalize(text):
    """Only NFC, whitespace collapse and equivalent quote forms; preserve all other text."""
    return ' '.join(unicodedata.normalize('NFC', text).translate(QUOTE_EQUIVALENTS).split())


def contains_foreign_letters(text):
    return any(unicodedata.category(c).startswith('L') and 'HEBREW' not in unicodedata.name(c, '') for c in text)


def mechanical_status(answer, gold):
    """A work-queue filter, never an assertion of human correctness or a rescore."""
    raw = answer.get('raw')
    explicit_failure = answer.get('status') in {'missing_record', 'empty_answer', 'technical_failure', 'request_failed', 'api_error'}
    if raw is None or not isinstance(raw, str) or not raw.strip() or answer.get('missing') is True or explicit_failure:
        return 'missing', 'empty_or_explicit_source_failure'
    if isinstance(gold, str) and raw.strip() == gold.strip():
        return 'exacttrim', 'whole_response_trim_equal'
    if isinstance(gold, str) and technical_normalize(raw) == technical_normalize(gold):
        return 'technical', 'whole_response_nfc_whitespace_quote_equal'
    return 'pending', None


class ContinuationStore(MaskedStore):
    """The same reviewer with answer-level continuation; the original study snapshot is immutable."""
    protocol = CONTINUATION_PROTOCOL
    limitations = ('תור המשך לכיסוי תשובות היצירה שלא הוצאו לפי כללים מפורשים, ולא מדגם אקראי. '
                   'המדגם ההיסטורי ושיפוטיו נשמרו בנפרד. סינון מכני אינו אישור אנושי ואינו משנה ציון. '
                   'הייחוס מוצג ועשוי להיות שגוי; חשיפות קודמות נשמרות וסגנון עשוי לרמוז לזהות. '
                   'אין לערבב אישורים אנושיים וסינון מכני בחישוב דיוק אנושי. תגית אינה הסבר סיבתי לפער.')
    views = {'all', 'foreign', 'unsure', 'suspicions', 'filtered'}

    def __init__(self, dataset, path, previous_v2_bundle, previous_v2_history=''):
        self.dataset = copy.deepcopy(dataset)
        self.plan = self.dataset['continuation_plan']
        self.path = Path(path)
        self.lock = threading.RLock()
        self.items = {i['id']: i for i in dataset['items']}
        if set(self.plan['queue']) != set(self.items) or len(self.plan['queue']) != len(self.items):
            raise ValueError('Continuation queue must cover every source item exactly once')
        self.answers = {item_id: {a['id']: a for a in self.items[item_id]['answers']
                                 if a['system_id'] in {m + '_generate' for m in MODELS}} for item_id in self.plan['queue']}
        if any(len(a) != 2 for a in self.answers.values()):
            raise ValueError('Exactly two generation answers required per item')
        self._view = 'all'
        if self.path.exists():
            self.state = json.loads(self.path.read_text(encoding='utf-8'))
            self.validate(self.state)
            return
        previous = copy.deepcopy(previous_v2_bundle)
        if previous.get('protocol_version') != PROTOCOL or previous.get('plan_id') != dataset['short_plan']['plan_id']:
            raise ValueError('Previous masked study mismatch')
        if (ReviewStore.manifest(previous.get('provenance')) != ReviewStore.manifest(dataset['provenance'])
                or previous.get('source_identity') != dataset.get('source_identity', dataset['dataset_id'])):
            raise ValueError('Previous source identity mismatch')
        # Preserve existing judgments, notes, tags and presentations exactly; extend only the working route.
        state = copy.deepcopy(previous)
        state.update(protocol_version=self.protocol, plan_id=self.plan['plan_id'], revision=0,
                     previous_v2_snapshot=previous, previous_v2_history=previous_v2_history,
                     previous_v2_sha256=digest(previous), previous_v2_history_sha256=hashlib.sha256(previous_v2_history.encode()).hexdigest(),
                     historical_plan=copy.deepcopy(dataset['short_plan']), continuation_plan=copy.deepcopy(self.plan),
                     filter_version=FILTER_VERSION, restored_answers={}, continuation_events=[],
                     historical_revealed=bool(previous.get('revealed')), revealed=False, reveal_events=[],
                     pre_reveal_snapshot=None, summary_exposures=[], backup_secret=secrets.token_hex(32))
        state.pop('pre_reveal_sha256', None)
        for item_id, answers in self.answers.items():
            if item_id in state['presentation']:
                continue
            order = list(answers); secrets.SystemRandom().shuffle(order)
            state['presentation'][item_id] = [{'answer_id': aid, 'token': secrets.token_hex(16)} for aid in order]
        self.state = state
        state['inventory'] = self.make_inventory(state)
        state['initial_counts'] = self.counts()
        self.validate(state)
        atomic_json(self.path, state)

    def prior_evidence(self, item_id, state=None, at=None, historical=False):
        state = state or self.state
        evidence = super().prior_evidence(item_id, state, at, historical)
        previous = state.get('previous_v2_snapshot', {})
        events = copy.deepcopy(previous.get('reveal_events', []))
        events += [e for e in previous.get('summary_exposures', []) if e.get('masked') is False]
        evidence['previous_v2_reveal_events'] = events
        evidence['previous_v2_revealed'] = bool(previous.get('revealed'))
        if previous.get('revealed') or events:
            evidence['status'] = 'after_reveal'
        return evidence

    def validate(self, state):
        super().validate(state)
        if (digest(state['previous_v2_snapshot']) != state['previous_v2_sha256']
                or hashlib.sha256(state['previous_v2_history'].encode()).hexdigest() != state['previous_v2_history_sha256']):
            raise ValueError('Historical v2 study changed')
        if state.get('historical_plan') != self.dataset['short_plan'] or state.get('continuation_plan') != self.plan:
            raise ValueError('Historical or continuation plan mismatch')
        if state.get('filter_version') != FILTER_VERSION or state.get('inventory') != self.make_inventory(state):
            raise ValueError('Continuation inventory mismatch')

    def make_inventory(self, state):
        inventory = {}
        for item_id, answers in self.answers.items():
            judgments = state['records'].get(item_id, {}).get('judgments', {})
            inventory[item_id] = {}
            for aid, answer in answers.items():
                automatic, rule = mechanical_status(answer, self.items[item_id].get('gold'))
                human = judgments.get(aid, {}).get('label') in LABELS
                restored = aid in state.get('restored_answers', {}).get(item_id, {})
                status = 'human' if human else ('pending' if restored else automatic)
                inventory[item_id][aid] = {'status': status, 'filter_rule': rule if automatic != 'pending' else None,
                    'rules_version': FILTER_VERSION, 'mechanical_status': automatic, 'restored': restored,
                    'foreign_letters': contains_foreign_letters(answer.get('raw') or ''),
                    'original_answer': copy.deepcopy(answer), 'gold': self.items[item_id].get('gold'),
                    'source': self.items[item_id].get('source')}
        return inventory

    def selected_ids(self, item_id, view, state=None):
        state = state or self.state
        inventory = state['inventory'][item_id]
        judgments = state['records'].get(item_id, {}).get('judgments', {})
        if view == 'suspicions':
            return set(inventory) if state['records'].get(item_id, {}).get('suspect') else set()
        if view == 'unsure':
            return {aid for aid in inventory if judgments.get(aid, {}).get('label') == 'unsure'}
        if view == 'filtered':
            return {aid for aid, entry in inventory.items() if entry['status'] in {'exacttrim', 'technical', 'missing'}}
        return {aid for aid, entry in inventory.items() if entry['status'] == 'pending' and (view != 'foreign' or entry['foreign_letters'])}

    def groups(self, item_id, view='all', state=None):
        state = state or self.state
        selected = self.selected_ids(item_id, view, state)
        slots = [slot for slot in state['presentation'][item_id] if slot['answer_id'] in selected]
        if len(slots) == 2:
            first, second = [slot['answer_id'] for slot in slots]
            a, b = [self.answers[item_id][aid].get('raw') for aid in (first, second)]
            js = state['records'].get(item_id, {}).get('judgments', {})
            shared = js.get(first, {}).get('decision_id') and js.get(first, {}).get('decision_id') == js.get(second, {}).get('decision_id')
            both_pending = all(state['inventory'][item_id][aid]['status'] == 'pending' for aid in (first, second))
            if isinstance(a, str) and a == b and (both_pending or shared):
                return [slots]
        return [[slot] for slot in slots]

    def queues(self):
        return {view: [i for i in self.plan['queue'] if self.selected_ids(i, view)] for view in self.views}

    def group_key(self, item_id, answer_id):
        item = self.items[item_id]
        return (item.get('acronym'), item.get('gold'), self.answers[item_id][answer_id].get('raw'))

    def group_id(self, key):
        return hmac.new(self.state['backup_secret'].encode(), digest(['exact-cross-context-v1', key]).encode(), hashlib.sha256).hexdigest()[:32]

    def group_index(self, view='all'):
        result = {}
        for item_id in self.plan['queue']:
            for slot in self.state['presentation'][item_id]:
                aid = slot['answer_id']
                entry = self.state['inventory'][item_id][aid]
                if entry['status'] != 'pending' or (view == 'foreign' and not entry['foreign_letters']):
                    continue
                key = self.group_key(item_id, aid)
                gid = self.group_id(key)
                group = result.setdefault(gid, {'key': key, 'members': {}})
                group['members'].setdefault(item_id, []).append(aid)
        return result

    def _member_fingerprint(self, judgment):
        return {key: copy.deepcopy(judgment.get(key)) for key in (
            'label', 'tags', 'label_updated_at', 'tags_updated_at', 'cross_context')}

    def public_group(self, session):
        acronym, gold, raw = session['key']
        contexts = []
        for item_id, aids in session['members'].items():
            r = self.state['records'].get(item_id, self.blank())
            tokens = {s['answer_id']: s['token'] for s in self.state['presentation'][item_id]}
            contexts.append({'id': item_id, 'sentence': self.items[item_id]['sentence'],
                'target_spans': copy.deepcopy(self.items[item_id].get('target_spans', [])),
                'occurrence_count': len(aids), 'answer_ids': [tokens[aid] for aid in aids],
                'suspect': r['suspect'], 'example': r['example'], 'note': r['note']})
        return {'id': session['group_id'], 'open_id': session['open_id'], 'acronym': acronym, 'gold': gold, 'text': raw,
                'contexts': contexts, 'form': copy.deepcopy(session['form']), 'grouping_version': 'exact-cross-context-v1'}

    def _group_session(self, payload):
        gid, opened = payload.get('group_id'), payload.get('open_id')
        session = self.state.get('cross_context_sessions', {}).get(opened)
        if (not session or session['group_id'] != gid
                or self.state.get('cross_context_active', {}).get(gid) != opened):
            raise ValueError('Open this group again before saving; its displayed membership has changed')
        return session

    def group_transact(self, action, payload):
        view = payload.get('view', 'all')
        if view not in {'all', 'foreign'}:
            raise ValueError('Unknown grouped view')
        candidate = copy.deepcopy(self.state)
        if action == 'group-open':
            gid = payload.get('group_id')
            group = self.group_index(view).get(gid)
            if not group:
                raise ValueError('This group has no pending answers in the selected view')
            opened = secrets.token_hex(16)
            session = {'open_id': opened, 'group_id': gid, 'version': 'exact-cross-context-v1', 'view': view,
                       'key': list(group['key']), 'members': copy.deepcopy(group['members']), 'opened_at': now(),
                       'form': {'label': '', 'tags': [], 'note': '', 'exceptions': {}}, 'expected': {}, 'expected_context': {},
                       'context_phases': {i: self.phase(i) for i in group['members']}}
            for item_id, aids in session['members'].items():
                r = candidate['records'].setdefault(item_id, self.blank())
                # Every displayed context/reference is explicitly logged, before any common decision.
                r['exposure'].setdefault('reference_and_generation', {'at': now(), 'protocol': self.protocol,
                    'independent_attempt': False, 'phase': self.phase(item_id)})
                r['exposure'].setdefault('cross_context_groups', {})[opened] = {
                    'at': now(), 'group_id': gid, 'phase': self.phase(item_id), 'reference_presented': True}
                session['expected'][item_id] = {aid: self._member_fingerprint(r['judgments'].get(aid, {})) for aid in aids}
                session['expected_context'][item_id] = {key: copy.deepcopy(r[key]) for key in ('note', 'suspect', 'example')}
            candidate.setdefault('cross_context_sessions', {})[opened] = session
            candidate.setdefault('cross_context_active', {})[gid] = opened
        else:
            original_session = self._group_session(payload)
            opened = original_session['open_id']; gid = original_session['group_id']
            session = candidate['cross_context_sessions'][opened]
            if action == 'group-details':
                for item_id in session['members']:
                    candidate['records'][item_id]['exposure'].setdefault('candidates', {
                        'at': now(), 'phase': self.phase(item_id), 'reason': 'explicit_group_details'})
            elif action == 'group-save':
                exceptions = payload.get('exceptions', {})
                updates = payload.get('context_updates', {})
                if (not isinstance(exceptions, dict) or not isinstance(updates, dict)
                        or not (set(exceptions) | set(updates)).issubset(session['members'])):
                    raise ValueError('Unknown or undisplayed context')
                if 'members' in payload:
                    raise ValueError('Group membership is fixed by the saved open session')
                label = payload.get('common_label', '')
                tags = payload.get('common_tags', session['form']['tags'])
                note = payload.get('common_note', session['form']['note'])
                if not isinstance(label, str) or label not in LABELS | {''}:
                    raise ValueError('Invalid common judgment')
                if not isinstance(tags, list) or any(not isinstance(t, str) or t not in TAGS for t in tags) or not isinstance(note, str):
                    raise ValueError('Invalid common tags or note')
                tags = sorted(set(tags))
                apply_tags = tags != session['form']['tags'] or payload.get('apply_tags') is True
                apply_note = note != session['form']['note']
                common_decision = session.get('common_decision_id') if label == session['form']['label'] else None
                common_decision = common_decision or secrets.token_hex(16)
                action_id = secrets.token_hex(16)
                stamp = now()
                for item_id, aids in session['members'].items():
                    r = candidate['records'][item_id]
                    if any(r[key] != session['expected_context'][item_id][key] for key in ('note', 'suspect', 'example')):
                        raise ValueError('Context notes or flags changed outside this group; reload before saving')
                    exception = exceptions.get(item_id, {})
                    update = updates.get(item_id, {})
                    if not isinstance(exception, dict) or not isinstance(update, dict):
                        raise ValueError('Invalid context override')
                    if not set(exception).issubset({'label', 'tags', 'note', 'suspect', 'example'}) or not set(update).issubset({'note', 'suspect', 'example'}):
                        raise ValueError('Unknown context override field')
                    choice = exception.get('label', label)
                    if not isinstance(choice, str) or choice not in LABELS | {'', 'defer'}:
                        raise ValueError('Invalid context judgment')
                    context_tags = exception.get('tags', tags)
                    if not isinstance(context_tags, list) or any(not isinstance(t, str) or t not in TAGS for t in context_tags):
                        raise ValueError('Invalid context tags')
                    context_tags = sorted(set(context_tags))
                    kind = 'defer' if choice == 'defer' else 'exception' if 'label' in exception else 'common'
                    effective = '' if choice == 'defer' else choice
                    old_first = r['judgments'].get(aids[0], {})
                    prior_link = (old_first.get('cross_context') or {})
                    decision = (old_first.get('decision_id') if old_first.get('label', '') == effective
                                and prior_link.get('kind') == kind else None) or secrets.token_hex(16)
                    for aid in aids:
                        current = self.state['records'].get(item_id, {}).get('judgments', {}).get(aid, {})
                        if self.group_key(item_id, aid) != tuple(session['key']):
                            raise ValueError('Source context/answer no longer matches this exact group')
                        if self._member_fingerprint(current) != session['expected'][item_id][aid]:
                            raise ValueError('A displayed answer was edited outside this group; reload without overwriting it')
                        if (current.get('label') in LABELS and (current.get('cross_context') or {}).get('open_id') != opened):
                            raise ValueError('Existing human judgment cannot be overwritten by a new group')
                        if self.state['inventory'][item_id][aid]['status'] not in {'pending', 'human'}:
                            raise ValueError('Filtered answer cannot be judged through a group')
                        j = r['judgments'].setdefault(aid, {})
                        j.setdefault('label', ''); j.setdefault('tags', []); j.setdefault('tag_status', 'not_marked')
                        for field in ('label_phase', 'tags_phase', 'label_updated_at', 'tags_updated_at'):
                            j.setdefault(field, None)
                        # Blank common form is a draft; only explicit defer can withdraw this session's own label.
                        write_label = bool(choice) or choice == 'defer'
                        if write_label and j.get('label', '') != effective:
                            j.update(label=effective, label_phase=self.phase(item_id), label_updated_at=stamp,
                                     annotator=self.state['reviewer'], origin=self.protocol,
                                     exposure_evidence=self.current_evidence(item_id))
                        if (apply_tags or 'tags' in exception) and j['tags'] != context_tags:
                            j.update(tags=context_tags, tag_status='marked' if context_tags else 'not_marked',
                                     tags_phase=self.phase(item_id), tags_updated_at=stamp,
                                     tags_exposure_evidence=self.current_evidence(item_id))
                        if write_label:
                            j.update(decision_id=decision, occurrence_count=len(aids), applied_to_original_answers=list(aids),
                                     decision_origin='explicit_cross_context_common_with_exceptions',
                                     cross_context={'version': 'exact-cross-context-v1', 'group_id': gid, 'open_id': opened,
                                         'shared_action_id': action_id, 'common_decision_id': common_decision,
                                         'context_decision_id': decision, 'context_id': item_id, 'kind': kind,
                                         'at': stamp, 'displayed_context_count': len(session['members'])})
                        session['expected'][item_id][aid] = self._member_fingerprint(j)
                    if apply_note and 'note' not in exception and 'note' not in update:
                        r['note'] = note
                    for source in (exception, update):
                        for field in ('note', 'suspect', 'example'):
                            if field in source:
                                r[field] = copy.deepcopy(source[field])
                    r.update(updated_at=stamp, edit_phase=self.phase(item_id))
                    session['expected_context'][item_id] = {key: copy.deepcopy(r[key]) for key in ('note', 'suspect', 'example')}
                session.update(form={'label': label, 'tags': tags, 'note': note, 'exceptions': copy.deepcopy(exceptions)},
                               common_decision_id=common_decision, saved_at=stamp)
            else:
                raise ValueError('Unknown grouped action')
        candidate.setdefault('cross_context_events', []).append({'at': now(), 'action': action,
            'group_id': gid, 'open_id': opened, 'version': 'exact-cross-context-v1', 'item_ids': list(session['members'])})
        state = self.commit(candidate, action)
        if action == 'group-details':
            return {'state': state, 'details': {'masked': True, 'contexts': [
                {'id': item_id, 'sentence': self.items[item_id]['sentence'], 'candidates': copy.deepcopy(self.items[item_id].get('candidates', []))}
                for item_id in session['members']]}}
        return {'state': state, 'group': self.public_group(self.state['cross_context_sessions'][opened])}

    def counts(self, records=None):
        inventory = self.state['inventory']
        statuses = Counter(entry['status'] for answers in inventory.values() for entry in answers.values())
        pending = statuses['pending']
        decisions = sum(len(self.groups(i)) for i in self.plan['queue'])
        base = super().counts(records)
        base.update(source_answers=sum(statuses.values()), human_answers=statuses['human'],
                    exacttrim_answers=statuses['exacttrim'], technical_answers=statuses['technical'], missing_answers=statuses['missing'],
                    pending_answers=pending, pending_decisions=decisions, duplicate_savings=pending - decisions,
                    foreign_pending_answers=sum(e['foreign_letters'] and e['status'] == 'pending' for a in inventory.values() for e in a.values()),
                    foreign_total_answers=sum(e['foreign_letters'] for a in inventory.values() for e in a.values()),
                    unsure_answers=sum(j.get('label') == 'unsure' for r in self.state['records'].values() for j in r['judgments'].values()),
                    pending_items=sum(bool(self.selected_ids(i, 'all')) for i in self.plan['queue']))
        historical_answers = 0
        decisions = set()
        continuation_decisions = set()
        shared_common_actions, group_context_judgments, group_exceptions = set(), set(), set()
        for item_id, record in self.state['records'].items():
            for aid, judgment in record['judgments'].items():
                if judgment.get('label') not in LABELS:
                    continue
                old = self.state['previous_v2_snapshot']['records'].get(item_id, {}).get('judgments', {}).get(aid, {})
                inherited = old.get('label') == judgment.get('label') and old.get('label_updated_at') == judgment.get('label_updated_at')
                historical_answers += inherited
                cross = (judgment.get('cross_context') or {})
                if cross.get('kind') in {'common', 'exception'}:
                    group_context_judgments.add((item_id, cross.get('context_decision_id')))
                    if cross['kind'] == 'common':
                        shared_common_actions.add(cross.get('common_decision_id'))
                    else:
                        group_exceptions.add((item_id, cross.get('context_decision_id')))
                key = ('cross_context_common', cross['common_decision_id']) if cross.get('kind') == 'common' and cross.get('common_decision_id') else (item_id, judgment.get('decision_id') or aid)
                decisions.add(key)
                if not inherited:
                    continuation_decisions.add(key)
        base.update(judged_answers=statuses['human'], historical_human_answers=historical_answers,
                    continuation_human_answers=statuses['human'] - historical_answers,
                    human_decisions=len(decisions), continuation_human_decisions=len(continuation_decisions),
                    grouped_pending_decisions=len(self.group_index()),
                    cross_context_savings=base['pending_decisions']-len(self.group_index()),
                    shared_common_actions=len(shared_common_actions), group_context_judgments=len(group_context_judgments),
                    group_exception_decisions=len(group_exceptions))
        return base

    def snapshot(self):
        with self.lock:
            result = super().snapshot()
            result.update(queues=self.queues(), queue=self.queues()['all'], rules_version=FILTER_VERSION,
                          historical_revealed=self.state['historical_revealed'],
                          prior_exposure=result['prior_exposure'] or self.state['historical_revealed'],
                          initial_counts=copy.deepcopy(self.state['initial_counts']),
                          group_queue=list(self.group_index()),
                          group_queues={view: list(self.group_index(view)) for view in ('all', 'foreign')},
                          grouping_version='exact-cross-context-v1')
            return result

    def public_records(self, records):
        result = super().public_records(records)
        historical = self.state['previous_v2_snapshot']['records']
        for item_id, r in records.items():
            for slot in self.state['presentation'][item_id]:
                aid, token = slot['answer_id'], slot['token']
                j = r['judgments'].get(aid, {})
                old = historical.get(item_id, {}).get('judgments', {}).get(aid, {})
                inherited = bool(j.get('label') in LABELS and old.get('label') == j.get('label')
                                 and old.get('label_updated_at') == j.get('label_updated_at'))
                result[item_id]['judgments'][token].update(
                    human_origin='historical' if inherited else 'continuation' if j.get('label') in LABELS else None,
                    decision_id=j.get('decision_id'), occurrence_count=j.get('occurrence_count', 1),
                    label_exposure_status=(j.get('exposure_evidence') or {}).get('status'),
                    tags_exposure_status=(j.get('tags_exposure_evidence') or {}).get('status'),
                    cross_context=copy.deepcopy(j.get('cross_context')))
        return result

    def export(self, masked=True):
        with self.lock:
            result = super().export(masked)
            if not masked:
                return result
            ledger = {}
            public_records = self.public_records(self.state['records'])
            for item_id, slots in self.state['presentation'].items():
                ledger[item_id] = {}
                for slot in slots:
                    entry = self.state['inventory'][item_id][slot['answer_id']]
                    human = public_records.get(item_id, {}).get('judgments', {}).get(slot['token'], {})
                    ledger[item_id][slot['token']] = {
                        'review_status': entry['status'], 'filter_rule': entry['filter_rule'], 'rules_version': FILTER_VERSION,
                        'original_text': self.answers[item_id][slot['answer_id']].get('raw'), 'gold': self.items[item_id].get('gold'),
                        'restored': entry['restored'], 'foreign_letters': entry['foreign_letters'],
                        'human_origin': human.get('human_origin'), 'decision_id': human.get('decision_id'),
                        'occurrence_count': human.get('occurrence_count', 1), 'cross_context': copy.deepcopy(human.get('cross_context'))}
            public_sessions = copy.deepcopy(self.state.get('cross_context_sessions', {}))
            for session in public_sessions.values():
                for item_id, aids in session['members'].items():
                    tokens = {slot['answer_id']: slot['token'] for slot in self.state['presentation'][item_id]}
                    session['members'][item_id] = [tokens[aid] for aid in aids]
                    session['expected'][item_id] = {tokens[aid]: value for aid, value in session['expected'][item_id].items()}
            result.update(answer_ledger=ledger, source_identity=self.state['source_identity'],
                          cross_context_sessions=public_sessions,
                          cross_context_active=copy.deepcopy(self.state.get('cross_context_active', {})))
            result.pop('backup_mac', None)
            result['backup_mac'] = hmac.new(self.state['backup_secret'].encode(), digest(result).encode(), hashlib.sha256).hexdigest()
            return result

    def restore_masked(self, bundle, revision):
        with self.lock:
            if revision != self.state['revision']:
                raise ValueError('State changed before restore')
            unsigned = copy.deepcopy(bundle)
            signature = unsigned.pop('backup_mac', '')
            expected = hmac.new(self.state['backup_secret'].encode(), digest(unsigned).encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(signature, expected):
                raise ValueError('Masked backup signature mismatch')
            if (bundle.get('protocol_version') != self.protocol or bundle.get('plan_id') != self.plan['plan_id']
                    or bundle.get('source_identity') != self.state['source_identity'] or bundle.get('masked') is not True
                    or set(bundle.get('answer_ledger', {})) != set(self.answers)):
                raise ValueError('Masked continuation backup mismatch')
            candidate = copy.deepcopy(self.state)
            for item_id, entries in bundle['answer_ledger'].items():
                mapping = {s['token']: s['answer_id'] for s in self.state['presentation'][item_id]}
                if set(entries) != set(mapping):
                    raise ValueError('Backup random handles do not match this store')
                for token, entry in entries.items():
                    if entry.get('restored'):
                        candidate['restored_answers'].setdefault(item_id, {}).setdefault(mapping[token], {'at': now(), 'reason': 'restore_masked_backup', 'rules_version': FILTER_VERSION})
                public = bundle.get('records', {}).get(item_id)
                if public is None:
                    continue
                if set(public.get('judgments', {})) != set(mapping):
                    raise ValueError('Backup judgment mapping mismatch')
                r = candidate['records'].setdefault(item_id, self.blank())
                for token, supplied in public['judgments'].items():
                    aid = mapping[token]
                    j = r['judgments'].setdefault(aid, {})
                    current_judgment = copy.deepcopy(j)
                    historical_judgment = self.state['previous_v2_snapshot']['records'].get(item_id, {}).get('judgments', {}).get(aid, {})
                    j.update({key: copy.deepcopy(supplied.get(key)) for key in (
                        'label', 'tags', 'tag_status', 'label_phase', 'tags_phase', 'label_updated_at', 'tags_updated_at',
                        'decision_id', 'occurrence_count', 'cross_context')})
                    for evidence_key, fields, status_key in (
                            ('exposure_evidence', ('label', 'label_phase', 'label_updated_at'), 'label_exposure_status'),
                            ('tags_exposure_evidence', ('tags', 'tags_phase', 'tags_updated_at'), 'tags_exposure_status')):
                        matching = next((source for source in (current_judgment, historical_judgment)
                                         if source and all(source.get(key) == supplied.get(key) for key in fields)), None)
                        if matching is not None:
                            if evidence_key in matching:
                                j[evidence_key] = copy.deepcopy(matching[evidence_key])
                            else:
                                j.pop(evidence_key, None)
                        else:
                            if evidence_key in current_judgment:
                                j.setdefault('evidence_before_restores', []).append({
                                    'at': now(), 'field': evidence_key,
                                    'evidence': copy.deepcopy(current_judgment[evidence_key])})
                            j[evidence_key] = {'status': supplied.get(status_key), 'restored_from_masked_backup': True}
                    j['restored_at'] = now()
                for aid, j in r['judgments'].items():
                    if j.get('decision_id'):
                        j['applied_to_original_answers'] = [slot['answer_id'] for slot in self.state['presentation'][item_id] if r['judgments'].get(slot['answer_id'], {}).get('decision_id') == j['decision_id']]
                for key in ('note', 'suspect', 'example'):
                    r[key] = copy.deepcopy(public[key])
                r.update(edit_phase=self.phase(item_id), updated_at=now(), restored_at=now())
            for opened, supplied_session in bundle.get('cross_context_sessions', {}).items():
                restored_session = copy.deepcopy(supplied_session)
                for item_id, tokens in restored_session['members'].items():
                    mapping = {slot['token']: slot['answer_id'] for slot in self.state['presentation'][item_id]}
                    restored_session['members'][item_id] = [mapping[token] for token in tokens]
                    restored_session['expected'][item_id] = {mapping[token]: value for token, value in restored_session['expected'][item_id].items()}
                candidate.setdefault('cross_context_sessions', {})[opened] = restored_session
            candidate.setdefault('cross_context_active', {}).update(bundle.get('cross_context_active', {}))
            candidate['inventory'] = self.make_inventory(candidate)
            self.validate(candidate)
            atomic_json(self.path.with_name(self.path.name + '.before-restore-' + secrets.token_hex(6) + '.json'), self.state)
            return self.commit(candidate, 'restore_masked')

    def public_item(self, item_id, include_all=False):
        item = super().public_item(item_id, include_all=True)
        if include_all:
            return item
        by_token = {a['id']: a for a in item['answers']}
        answers = []
        for slots in self.groups(item_id, self._view):
            representative = slots[0]
            answer = copy.deepcopy(by_token[representative['token']])
            entry = self.state['inventory'][item_id][representative['answer_id']]
            answer.update(occurrence_count=len(slots), review_status=entry['status'], foreign_letters=entry['foreign_letters'])
            if entry['status'] in {'exacttrim', 'technical', 'missing'}:
                answer.update(filter_rule=entry['filter_rule'], rules_version=FILTER_VERSION)
            answers.append(answer)
        item.update(answers=answers, view=self._view)
        return item

    def commit(self, candidate, action, item_id=None):
        if candidate['previous_v2_snapshot'] != self.state['previous_v2_snapshot'] or candidate['previous_v2_history'] != self.state['previous_v2_history']:
            raise ValueError('Previous study archive is immutable')
        if action == 'save':
            for aid, judgment in candidate['records'][item_id]['judgments'].items():
                old = self.state['records'].get(item_id, {}).get('judgments', {}).get(aid, {})
                link = old.get('cross_context')
                if link and old.get('label') != judgment.get('label'):
                    judgment.setdefault('cross_context_history', []).append(copy.deepcopy(link))
                    judgment['cross_context'] = dict(link, kind='individual_edit', individual_edit_at=now())
                elif link and old.get('tags') != judgment.get('tags'):
                    judgment['cross_context'] = dict(link, individual_tags_edited_at=now())
            for slots in getattr(self, '_decision_groups', {}).values():
                aids = [slot['answer_id'] for slot in slots]
                judgments = candidate['records'][item_id]['judgments']
                if all(judgments.get(aid, {}).get('label') in LABELS for aid in aids):
                    previous_ids = {self.state['records'].get(item_id, {}).get('judgments', {}).get(aid, {}).get('decision_id') for aid in aids}
                    decision_id = next(iter(previous_ids)) if len(previous_ids) == 1 and None not in previous_ids else secrets.token_hex(16)
                    for aid in aids:
                        judgments[aid].update(decision_id=decision_id, occurrence_count=len(aids),
                                             applied_to_original_answers=aids, decision_origin='single_judgment_exact_same_item_duplicates')
        candidate['inventory'] = self.make_inventory(candidate)
        candidate['continuation_events'].append({'action': action, 'item_id': item_id, 'at': now()})
        return super().commit(candidate, action, item_id)

    def transact(self, action, payload):
        with self.lock:
            if payload.get('revision') != self.state['revision']:
                raise ValueError('חלון אחר שינה את העבודה. יש לטעון מחדש לפני שמירה.')
            view = payload.get('view', 'all')
            if view not in self.views:
                raise ValueError('Unknown continuation view')
            self._view = view
            if action in {'group-open', 'group-save', 'group-details'}:
                return self.group_transact(action, payload)
            if action == 'restore':
                item_id = payload.get('item_id')
                if item_id not in self.answers:
                    raise ValueError('Unknown item')
                token = payload.get('answer_id')
                aid = next((s['answer_id'] for s in self.state['presentation'][item_id] if s['token'] == token), None)
                if not aid or self.state['inventory'][item_id][aid]['status'] not in {'exacttrim', 'technical', 'missing'}:
                    raise ValueError('Only mechanically filtered answers may be restored')
                candidate = copy.deepcopy(self.state)
                candidate['restored_answers'].setdefault(item_id, {})[aid] = {'at': now(), 'rules_version': FILTER_VERSION,
                     'previous_status': self.state['inventory'][item_id][aid]['status']}
                return self.commit(candidate, action, item_id)
            if action == 'summary':
                candidate = copy.deepcopy(self.state)
                candidate['summary_exposures'].append({'at': now(), 'masked': True})
                state = self.commit(candidate, action)
                return {'state': state, 'summary': self.summary(True)}
            if action == 'save':
                payload = copy.deepcopy(payload)
                mapping = {s['token']: s['answer_id'] for s in self.state['presentation'][payload['item_id']]}
                # A displayed exact duplicate is one decision applied to its original occurrences.
                groups = self.groups(payload['item_id'], view)
                grouped = {}
                for slots in groups:
                    if len(slots) == 2:
                        for slot in slots:
                            grouped[slot['token']] = slots
                saved = self.state['records'].get(payload['item_id'], {}).get('judgments', {})
                for slot in self.state['presentation'][payload['item_id']]:
                    decision_id = saved.get(slot['answer_id'], {}).get('decision_id')
                    if decision_id:
                        peers = [s for s in self.state['presentation'][payload['item_id']] if saved.get(s['answer_id'], {}).get('decision_id') == decision_id]
                        if len(peers) > 1:
                            grouped[slot['token']] = peers
                expanded = {}
                for token in set(payload.get('judgments', {})) | set(payload.get('tags', {})):
                    if token not in mapping:
                        raise ValueError('Unknown answer token')
                    aid = mapping[token]
                    if self.state['inventory'][payload['item_id']][aid]['status'] in {'exacttrim', 'technical', 'missing'}:
                        raise ValueError('Restore filtered answer before judging')
                    slots = grouped.get(token)
                    if slots:
                        expanded[token] = slots
                        for slot in slots:
                            for field in ('judgments', 'tags'):
                                if token in payload.get(field, {}):
                                    if slot['token'] in payload[field] and payload[field][slot['token']] != payload[field][token]:
                                        raise ValueError('Conflicting labels or tags for one duplicate decision')
                                    payload[field][slot['token']] = copy.deepcopy(payload[field][token])
                self._decision_groups = expanded
                try:
                    return super().transact(action, payload)
                finally:
                    self._decision_groups = {}
            return super().transact(action, payload)

    def case(self, item_id, masked=True):
        case = super().case(item_id, True)
        for shown, slot in zip(case['answers'], self.state['presentation'][item_id]):
            entry = self.state['inventory'][item_id][slot['answer_id']]
            j = self.state['records'][item_id]['judgments'].get(slot['answer_id'], {})
            shown.update(review_status=entry['status'], foreign_letters=entry['foreign_letters'],
                         decision_id=j.get('decision_id'), occurrence_count=j.get('occurrence_count', 1),
                         cross_context=copy.deepcopy(j.get('cross_context')))
            if entry['status'] in {'exacttrim', 'technical', 'missing'}:
                shown.update(filter_rule=entry['filter_rule'], rules_version=FILTER_VERSION)
            if not masked:
                answer = self.answers[item_id][slot['answer_id']]
                label, score = shown['label'], answer.get('auto_score')
                mismatch = (label == 'fits') != score if label in {'fits', 'not_fits'} and type(score) is bool else None
                shown.update(answer_id=slot['answer_id'], original_answer_id=slot['answer_id'], model=answer['system_id'],
                             raw=answer.get('raw'), auto_score=score, original_score_rule=answer.get('auto_score_rule'), origin=j.get('origin'),
                             exposure_evidence=j.get('exposure_evidence'), tags_exposure_evidence=j.get('tags_exposure_evidence'),
                             comparison='disagreement' if mismatch else 'agreement' if mismatch is False else 'undetermined')
        if not masked:
            r = self.state['records'][item_id]
            case.update(source=self.items[item_id].get('source'), previous_note=r.get('previous_note', ''), edit_phase=r.get('edit_phase'), updated_at=r.get('updated_at'))
        return case

    def summary(self, masked=None):
        with self.lock:
            masked = True if masked is None else masked
            if not masked and not self.state['revealed']:
                raise ValueError('Explicit continuation reveal required')
            result = super().summary(True)
            result.update(masked=bool(masked), inventory_counts=self.counts(), initial_counts=copy.deepcopy(self.state['initial_counts']), rules_version=FILTER_VERSION, incomplete_ids=self.queues()['all'], tag_count_unit='answer_occurrences',
                          grouping_version='exact-cross-context-v1',
                          tag_count_note='ספירת תגיות לפי מופעי תשובה; שיפוט משותף מפורש עשוי לחול על כמה מופעים והקשרים.',
                          grouping_note='קיבוץ רק לפי טקסט תשובה, קיצור וייחוס זהים בדיוק; השיפוט המשותף הוא החלטה אנושית לאחר הצגת כל ההקשרים. חריגים מתועדים בנפרד, ואין לספור הקשרים כהכרעות עצמאיות.')
            if self.state.get('cross_context_sessions'):
                result['limitations'] += ' ' + result['grouping_note']
            if not masked:
                cases = [self.case(i, False) for i in self.plan['queue'] if i in self.state['records']]
                result['cases'] = cases
                result['examples'] = [c for c in cases if c['example']]
                result['suspicions'] = [c for c in cases if c['suspect']]
                result['unresolved'] = [c for c in cases if any(a['label'] == 'unsure' for a in c['answers'])]
                result['accepted_human_rejected_auto'] = [c for c in cases if any(a['label'] == 'fits' and a['auto_score'] is False for a in c['answers'])]
                result['rejected_human_accepted_auto'] = [c for c in cases if any(a['label'] == 'not_fits' and a['auto_score'] is True for a in c['answers'])]
                result['historical_sample_preserved'] = True
            return result

    def markdown(self, masked=True):
        # Reuse the established report's masked human fields, then add the continuation accounting.
        text = super().markdown(True)
        counts = self.counts()
        text += '\n\n## תור ההמשך\n\n' + '\n'.join(f'- {label}: {counts[key]}' for key, label in (
            ('source_answers', 'תשובות מקור'), ('human_answers', 'שיפוטים אנושיים קיימים'), ('exacttrim_answers', 'סוננו בהתאמה מלאה'),
            ('technical_answers', 'סוננו בנרמול טכני מצומצם'), ('missing_answers', 'אין תשובה או כשל מתועד'),
            ('pending_answers', 'תשובות שנותרו'), ('duplicate_savings', 'מופעים זהים שחוסכים הכרעה נוספת'), ('pending_decisions', 'הכרעות לפי הקשרים שנותרו'),
            ('grouped_pending_decisions', 'קבוצות מדויקות שנותרו'), ('shared_common_actions', 'הכרעות משותפות שנשמרו'),
            ('group_context_judgments', 'הקשרים שאליהם הוחלו הכרעות קבוצתיות'), ('group_exception_decisions', 'הכרעות חריגות להקשר')))
        text += '\n\n' + self.summary(True)['grouping_note'] + '\n' + self.summary(True)['tag_count_note']
        if not masked:
            full = self.summary(False)
            text += '\n\n## תוצאות לאחר חשיפה\n'
            for case in full['cases']:
                for answer in case['answers']:
                    text += f"\n- {case['id']} / {answer['answer_id']}; מקור: {case['source']}; טקסט: {answer['raw']}; ייחוס: {case['gold']}; מודל: {answer['model']}; ציון מקורי: {answer['auto_score']}; שיפוט: {answer['label'] or 'טרם נשפט'}; תגיות: {', '.join(answer['tags']) or 'לא סומן'}; שלב: {answer['label_phase']}"
        return text
