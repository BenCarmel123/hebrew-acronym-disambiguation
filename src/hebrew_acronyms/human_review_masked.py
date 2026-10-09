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
            self.state = {'protocol_version': PROTOCOL, 'plan_id': self.plan['plan_id'],
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
        if state.get('protocol_version') != PROTOCOL or state.get('plan_id') != self.plan['plan_id']:
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
            if len(slots) != 2 or {s['answer_id'] for s in slots} != set(self.answers[item_id]):
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
            return {'protocol_version': PROTOCOL, 'plan_id': self.plan['plan_id'], 'revision': self.state['revision'],
                    'queue': copy.deepcopy(self.plan['queue']), 'records': self.public_records(self.state['records']),
                    'counts': self.counts(), 'reviewer': self.state['reviewer'], 'revealed': self.state['revealed'],
                    'exposed': self.state['revealed'], 'prior_exposure': any(self.prior_evidence(i)['status'] != 'before_reveal' for i in self.plan['queue'])}

    def public_item(self, item_id):
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
                 'protocol_version': PROTOCOL, 'record': candidate['records'].get(item_id),
                 'reveal_event': candidate['reveal_events'][-1:] if action == 'reveal' else []}
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
                r['exposure'].setdefault('reference_and_generation', {'at': now(), 'protocol': PROTOCOL,
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
                                 origin=PROTOCOL, exposure_evidence=self.current_evidence(item_id))
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
        public = self.public_item(item_id)
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
            result = {'protocol_version': PROTOCOL, 'masked': bool(masked), 'counts': self.counts(), 'cases': cases,
                      'examples': [c for c in cases if c['example']], 'suspicions': [c for c in cases if c['suspect']],
                      'unresolved': [c for c in cases if any(a['label'] == 'unsure' for a in c['answers'])],
                      'tagged': [c for c in cases if any(a['tags'] for a in c['answers'])],
                      'tag_counts': dict(tag_counts), 'limitations': LIMITATIONS,
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
            if (bundle.get('protocol_version') != PROTOCOL or bundle.get('plan_id') != self.state['plan_id']
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
        lines = ['# סיכום מוסתר' if masked else '# סיכום לאחר חשיפה', '', LIMITATIONS, '',
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
