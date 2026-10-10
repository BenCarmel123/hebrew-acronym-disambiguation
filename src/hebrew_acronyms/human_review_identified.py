"""Bind the existing compact reviewer to immutable collected test responses.

No historical judgments are imported. Exact response groups are confined to one
item/context/task, retaining every original answer and candidate mapping.
"""
from collections import Counter
import copy
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import secrets
import threading
from datetime import datetime, timezone, timedelta

from .human_review_masked import MaskedStore, LABELS, contains_foreign_letters, now
from .human_review_server import atomic_json
from .human_review_short import digest

PROTOCOL = 'identified-test-review-v1'


def build_bundle(paths, cohort_path, reviewer):
    import csv
    with Path(cohort_path).open(encoding='utf-8-sig') as handle:
        rows = {r['item_id']: r for r in csv.DictReader(handle)}
    if len(rows) != 381:
        raise ValueError('Expected the approved 381-item cohort')
    groups, sources, seen = {}, [], set()
    for path in map(Path, paths):
        dataset = json.loads(path.read_text())
        sources.append({'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size})
        for item in dataset['items']:
            original = item['original_item_id']
            if original not in rows:
                continue
            row = rows[original]
            if (item['sentence'], item['acronym'], item['gold']) != (row['sentence'], row['acronym'], row['gold_expansion']):
                raise ValueError('Review input differs from scored cohort')
            for answer in item['answers']:
                binding = answer['binding']
                if binding['item_id'] != original or binding['response_sha256'] != hashlib.sha256(answer['raw'].encode()).hexdigest():
                    raise ValueError('Response binding mismatch')
                if answer['id'] in seen:
                    raise ValueError('Duplicate source answer')
                seen.add(answer['id'])
                key = digest([original, item['sentence'], item['acronym'], item['gold'], answer['task'],
                              answer['raw'], answer.get('decoded'), answer.get('option_mapping'), answer.get('technical_failure')])
                if key not in groups:
                    group = copy.deepcopy(item)
                    group.update(id=key, answers=[copy.deepcopy(answer)], occurrences=[])
                    groups[key] = group
                groups[key]['occurrences'].append(copy.deepcopy(answer))
    items = list(groups.values())
    frequency = Counter((i['acronym'], i['answers'][0]['raw']) for i in items)
    for item in items:
        a = item['answers'][0]
        item['priority_evidence'] = {
            'similarity_to_gold': SequenceMatcher(None, a['raw'], item['gold']).ratio(),
            'repeated_variant_count': frequency[(item['acronym'], a['raw'])],
            'positive_with_additions': a['auto_score'] is True and a['raw'].strip() != item['gold'].strip(),
            'not_human_judgment': True,
        }
    def rank(i):
        p = i['priority_evidence']
        return (-p['similarity_to_gold'], -p['repeated_variant_count'], i['id'])
    negative = sorted([i for i in items if i['answers'][0]['task']=='generation' and not i['answers'][0]['technical_failure'] and not i['answers'][0]['auto_score']], key=rank)
    positive = sorted([i for i in items if i['answers'][0]['task']=='generation' and not i['answers'][0]['technical_failure'] and i['answers'][0]['auto_score']], key=lambda i: (not i['priority_evidence']['positive_with_additions'], *rank(i)))
    selection = sorted([i for i in items if i['answers'][0]['task']=='selection' and not i['answers'][0]['technical_failure']], key=lambda i: (bool(i['answers'][0].get('auto_valid')), i['answers'][0]['auto_score'], i['id']))
    # Interleave system-specific queues; every fifth negative comes from the hash-
    # ordered remainder, so high string similarity does not monopolize the hour.
    pools = {}
    for i in negative:
        pools.setdefault(i['answers'][0]['system_id'], []).append(i)
    prioritized = []
    while any(pools.values()):
        for name in sorted(pools):
            pool = pools[name]
            if pool:
                idx = min(range(len(pool)), key=lambda x: pool[x]['id']) if len(prioritized)%5==4 else 0
                prioritized.append(pool.pop(idx))
    calibration = []
    for start in range(0, 16, 4):
        calibration += prioritized[start:start+4] + positive[start//4:start//4+1]
    calibration += selection[:2]
    order = list(dict.fromkeys(i['id'] for i in calibration + prioritized + positive + selection + items))
    provenance = {'files': sources, 'cohort_sha256': hashlib.sha256(Path(cohort_path).read_bytes()).hexdigest(),
                  'exclusion_timing': 'After original 395-item response collection', 'judgment_transfer': 'None',
                  'grouping': 'Same item, context, acronym, reference, task, exact raw and decoded response, candidate mapping, failure status',
                  'prior_exposure': 'unknown; historical reviews retained separately'}
    plan = {'queue': order, 'selection_reasons': {i['id']: {'group':'diagnostic', **i['priority_evidence']} for i in items},
            'calibration': [i['id'] for i in calibration], 'prioritized': [i['id'] for i in prioritized],
            'positives': [i['id'] for i in positive], 'selection': [i['id'] for i in selection]}
    plan['plan_id'] = digest(plan)
    return {'schema_version':'human-review-data-v2', 'short_protocol':PROTOCOL, 'short_plan':plan,
            'dataset_id':digest([provenance, items]), 'source_identity':digest([provenance, items]),
            'provenance':provenance, 'items':items, 'reviewer':reviewer}


class IdentifiedStore(MaskedStore):
    """Reuse masking, persistence, keyboard UI and signed backups; one exact group per card."""
    protocol = PROTOCOL
    limitations = ('בדיקה אבחונית חלקית של 381 פריטים, עם ייחוס מוצג וחשיפה קודמת שאינה ידועה. '
                   'בטבלה ברירת מחדל שלילית מאושרת במפורש ועלולה להטות את השיפוט. תעדוף מכני אינו שיפוט אנושי. אין להסיק דיוק כולל או שלילת כל false negatives. '
                   'שיפוט משותף חל רק על תשובות זהות באותו פריט, הקשר ומשימה; כל מקורותיהן נשמרים.')

    def __init__(self, dataset, path):
        self.dataset=copy.deepcopy(dataset); self.plan=dataset['short_plan']; self.path=Path(path)
        self.lock=threading.RLock(); self.items={i['id']:i for i in dataset['items']}
        self.answers={i: {a['id']:a for a in item['answers']} for i,item in self.items.items()}
        self._view='all'
        if self.path.exists():
            self.state=json.loads(self.path.read_text()); self.validate(self.state); return
        # Empty predecessor is explicit: this source has no inherited judgments.
        previous={'records':{}, 'summary_exposures':[], 'legacy_snapshot':{'records':{}}}
        self.state={'protocol_version':PROTOCOL, 'plan_id':self.plan['plan_id'],
                    'source_identity':dataset['source_identity'], 'provenance':copy.deepcopy(dataset['provenance']),
                    'revision':0, 'reviewer':dataset['reviewer'], 'records':{}, 'backup_secret':secrets.token_hex(32),
                    'presentation':{i:[{'answer_id':next(iter(a)), 'token':secrets.token_hex(16)}] for i,a in self.answers.items()},
                    'previous_protocol':previous, 'previous_history':'', 'previous_sha256':digest(previous),
                    'previous_history_sha256':hashlib.sha256(b'').hexdigest(),
                    'revealed':False, 'reveal_events':[], 'pre_reveal_snapshot':None, 'summary_exposures':[]}
        self.validate(self.state); atomic_json(self.path,self.state)

    def validate(self, state):
        super().validate(state)
        if state['provenance'] != self.dataset['provenance']:
            raise ValueError('Changed source provenance')
        if self.dataset['source_identity'] != digest([self.dataset['provenance'], self.dataset['items']]):
            raise ValueError('Changed response group content')

    def prior_evidence(self, item_id, state=None, **kwargs):
        return {'status':'unknown', 'source':'Historical exposure not reset; no judgments transferred'}

    @staticmethod
    def completion(record):
        n=sum(j.get('label') in LABELS for j in record.get('judgments',{}).values())
        return {'status':'complete' if n else 'unreviewed','judged_answers':n,'total_answers':1}

    def queues(self):
        result={k:[] for k in ('all','calibration','prioritized','hebrew','positives','selection','foreign','unsure','suspicions','filtered')}
        for i in self.plan['queue']:
            item=self.items[i]; a=item['answers'][0]
            r=self.state['records'].get(i,{}); j=r.get('judgments',{}).get(a['id'],{})
            if a['technical_failure']: result['filtered'].append(i)
            elif j.get('label') not in LABELS:
                result['all'].append(i)
                for view in ('calibration','prioritized','positives','selection'):
                    if i in self.plan[view]: result[view].append(i)
                if contains_foreign_letters(a['raw']): result['foreign'].append(i)
                elif i in self.plan['prioritized']: result['hebrew'].append(i)
            if j.get('label')=='unsure': result['unsure'].append(i)
            if r.get('suspect'): result['suspicions'].append(i)
        return result

    def counts(self, records=None):
        records=self.state['records'] if records is None else records
        total=human=missing=unsure=decisions=pending_groups=0
        for i,item in self.items.items():
            n=len(item['occurrences']); total+=n; a=item['answers'][0]
            label=records.get(i,{}).get('judgments',{}).get(a['id'],{}).get('label')
            if label in LABELS: human+=n; decisions+=1; unsure+=n*(label=='unsure')
            elif a['technical_failure']: missing+=n
            else: pending_groups+=1
        return {'total_items':len(self.items),'total_answers':total,'source_answers':total,
                'human_answers':human,'judged_answers':human,'human_decisions':decisions,
                'complete_items':decisions,'reviewed_items':decisions,'partial_items':0,
                'missing_answers':missing,'pending_answers':total-human-missing,
                'pending_decisions':pending_groups,'duplicate_savings':total-human-missing-pending_groups,
                'exacttrim_answers':0,'technical_answers':0,'unsure_answers':unsure}

    def snapshot(self):
        result=super().snapshot(); result.update(queues=self.queues(), manual_session=self.state.get('manual_session'))
        return result

    def public_item(self,item_id,include_all=False):
        item=super().public_item(item_id,include_all)
        source=self.items[item_id]; a=source['answers'][0]
        if a['task']=='selection':
            item['answers'][0]['text']='פלט מקורי: '+a['raw']+'\nפירוש מפוענח: '+str(a.get('decoded') or 'לא פוענח')
        item['answers'][0].update(occurrence_count=len(source['occurrences']),review_status='missing' if a['technical_failure'] else 'pending', can_restore=False)
        item['task_label']='בחירה — בדיקת פירוש ופענוח' if a['task']=='selection' else 'יצירה'
        return item

    def commit(self, candidate, action, item_id=None):
        if action in {'save', 'batch-save'} and not candidate.get('manual_session'):
            if any(j.get('label') in LABELS for r in candidate['records'].values() for j in r['judgments'].values()):
                start = datetime.now(timezone.utc)
                candidate['manual_session'] = {'started_at':start.isoformat(),
                    'deadline':(start+timedelta(minutes=60)).isoformat(),
                    'clock_basis':'60-minute wall-clock cap from first real judgment; preparation excluded'}
        return super().commit(candidate, action, item_id)

    def stop_if_due(self):
        with self.lock:
            session=self.state.get('manual_session')
            if not session or datetime.now(timezone.utc) < datetime.fromisoformat(session['deadline']):
                return False
            backup=self.path.parent/'verified-backup-60min'
            if not backup.exists():
                backup.mkdir()
                manifest={}
                for name, content in ((self.path.name,self.path.read_bytes()),
                        ('review-data.json',json.dumps(self.dataset,ensure_ascii=False,indent=2).encode())):
                    target=backup/name;target.write_bytes(content)
                    signature=hashlib.sha256(content).hexdigest()
                    if hashlib.sha256(target.read_bytes()).hexdigest()!=signature: raise OSError('Backup mismatch')
                    manifest[name]=signature
                history=self.path.with_suffix('.history.jsonl')
                if history.exists():
                    content=history.read_bytes();(backup/history.name).write_bytes(content)
                    manifest[history.name]=hashlib.sha256(content).hexdigest()
                    if hashlib.sha256((backup/history.name).read_bytes()).hexdigest()!=manifest[history.name]: raise OSError('History backup mismatch')
                atomic_json(backup/'sha256.json',manifest)
            return True

    def summary(self, masked=None):
        result = super().summary(masked)
        tags = Counter()
        for item_id, record in self.state['records'].items():
            for judgment in record['judgments'].values():
                for tag in judgment.get('tags', []):
                    tags[tag] += len(self.items[item_id]['occurrences'])
        result['tag_counts'] = dict(tags)
        result['tag_count_unit'] = 'answer_occurrences'
        return result

    def transact(self,action,payload):
        if action in {'save','open','batch-open','batch-save'} and self.stop_if_due():
            raise ValueError('הסתיימו 60 דקות התיוג. העבודה וגיבוי מאומת נשמרו; אפשר לפתוח סיכום.')
        if action.startswith('group-') or action=='restore':
            raise ValueError('Cross-context transfer and technical-failure relabeling are disabled')
        self._view=payload.get('view','all')
        if self._view not in self.queues(): raise ValueError('Unknown queue')
        if action=='save' and self.items[payload['item_id']]['answers'][0]['technical_failure']:
            raise ValueError('Technical failures stay separate')
        if action in {'batch-open','batch-save'}:
            return self.transact_batch(action,payload)
        result=super().transact(action,payload)
        return result

    def transact_batch(self, action, payload):
        """One explicit confirmation, independent labels bound to displayed rows.

        Opening a table never creates judgments. Blank labels skip unread rows;
        the proposed negative default is a UI aid, recorded only on confirmation.
        """
        with self.lock:
            if payload.get('revision') != self.state['revision']:
                raise ValueError('העבודה השתנתה. יש לטעון מחדש לפני אישור הטבלה.')
            if self.stop_if_due():
                raise ValueError('הסתיימו 60 דקות התיוג; העבודה וגיבוי מאומת נשמרו.')
            candidate = copy.deepcopy(self.state)
            if action == 'batch-open':
                ids = payload.get('item_ids', [])
                allowed = self.queues()[self._view]
                if not isinstance(ids, list) or not 1 <= len(ids) <= 20 or len(set(ids)) != len(ids):
                    raise ValueError('A table must contain 1–20 distinct rows')
                if any(i not in allowed or self.items[i]['answers'][0]['technical_failure'] for i in ids):
                    raise ValueError('Invalid table membership')
                opened = secrets.token_hex(16)
                candidate['active_batch'] = dict(id=opened, item_ids=ids, view=self._view, opened_at=now(),
                    proposed_default='not_fits', interaction_mode='explicit_table_confirmation')
                for i in ids:
                    r = candidate['records'].setdefault(i, self.blank())
                    r['exposure'].setdefault('reference_and_generation', dict(at=now(), protocol=self.protocol,
                        independent_attempt=False, phase=self.phase(i)))
            else:
                session = candidate.get('active_batch', {})
                if payload.get('batch_id') != session.get('id') or session.get('confirmed_at') or payload.get('confirmed') is not True:
                    raise ValueError('Explicit confirmation of the opened table is required')
                if session.get('view') != self._view:
                    raise ValueError('Table queue changed')
                ids = session['item_ids']
                labels = payload.get('labels', {})
                if not isinstance(labels, dict) or set(labels) != set(ids) or any(v not in LABELS | {''} for v in labels.values()):
                    raise ValueError('Labels must match the displayed rows; blank skips a row')
                if not any(labels.values()):
                    raise ValueError('אין הכרעות לאישור; אפשר לדלג בלי לשמור תיוגים.')
                session['confirmed_at'] = now()
                session['judged_item_ids'] = [i for i in ids if labels[i]]
                for i in session['judged_item_ids']:
                    r = candidate['records'][i]
                    aid = self.items[i]['answers'][0]['id']
                    j = copy.deepcopy(r['judgments'].get(aid, {}))
                    j.update(label=labels[i], label_phase=self.phase(i), label_updated_at=now(),
                        annotator=self.state['reviewer'], origin=self.protocol, exposure_evidence=self.current_evidence(i),
                        interaction_mode='explicit_table_confirmation', batch_id=session['id'])
                    j.setdefault('tags', []); j.setdefault('tags_phase', None); j.setdefault('tags_updated_at', None)
                    j['tag_status'] = 'marked' if j['tags'] else 'not_marked'
                    r['judgments'][aid] = j
                    r.update(updated_at=now(), edit_phase=self.phase(i))
            state = self.commit(candidate, action)
            return dict(state=state, batch_id=candidate['active_batch']['id'],
                        items=[self.public_item(i) for i in ids] if action == 'batch-open' else [])


def coverage_rows(dataset, state):
    groups = {}
    for item in dataset['items']:
        representative = item['answers'][0]
        judgment = state['records'].get(item['id'], {}).get('judgments', {}).get(representative['id'], {})
        label = judgment.get('label')
        for answer in item['occurrences']:
            key = (answer['system_id'], answer['task'], answer['binding']['run_id'])
            row = groups.setdefault(key, dict(system=key[0], task=key[1], run_id=key[2], total=0,
                reviewed=0, positive_total=0, positive_reviewed=0, negative_total=0, negative_reviewed=0,
                technical_failures=0, unsure=0, fits=0, not_fits=0, automatic_negative_human_fits=0,
                automatic_positive_human_not_fits=0, decisions=set()))
            row['total'] += 1
            if answer['technical_failure']:
                row['technical_failures'] += 1
            else:
                polarity = 'positive' if answer['auto_score'] else 'negative'
                row[polarity+'_total'] += 1
                if label in LABELS:
                    row['reviewed'] += 1; row[polarity+'_reviewed'] += 1; row[label] += 1
                    row['decisions'].add(item['id'])
                    row['automatic_negative_human_fits'] += polarity=='negative' and label=='fits'
                    row['automatic_positive_human_not_fits'] += polarity=='positive' and label=='not_fits'
    for row in groups.values():
        row['decisions'] = len(row['decisions'])
        row['unreviewed'] = row['total']-row['reviewed']
        row['unresolved'] = row['unreviewed']+row['unsure']
    return sorted(groups.values(), key=lambda r:(r['system'],r['task']))


def export_judgments(dataset, state, output):
    """Portable, source-bound decisions; excludes private masking keys and mappings."""
    decisions=[]
    for item in dataset['items']:
        answer=item['answers'][0]
        record=state['records'].get(item['id'],{})
        judgment=record.get('judgments',{}).get(answer['id'],{})
        if judgment.get('label') not in LABELS:
            continue
        decisions.append({'decision_id':item['id'], 'original_item_id':item['original_item_id'],
            'answer_ids':[a['id'] for a in item['occurrences']],
            'bindings':[a['binding'] for a in item['occurrences']],
            'judgment':copy.deepcopy(judgment),
            'suspect':record.get('suspect',False),'example':record.get('example',False),'note':record.get('note','')})
    result={'protocol_version':PROTOCOL,'dataset_id':dataset['dataset_id'],'plan_id':dataset['short_plan']['plan_id'],
            'source_files':[{'path':'/'.join(Path(f['path']).parts[-2:]),'sha256':f['sha256'],'bytes':f['bytes']} for f in dataset['provenance']['files']],
            'cohort_sha256':dataset['provenance']['cohort_sha256'], 'manual_session':state.get('manual_session'),
            'annotation_sha256':digest(state),'reviewer':state['reviewer'],'decisions':decisions,
            'coverage':coverage_rows(dataset,state),'limitations':IdentifiedStore.limitations}
    atomic_json(Path(output),result)
    return result


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    build=sub.add_parser('build');build.add_argument('--source',type=Path,action='append',required=True)
    build.add_argument('--cohort',type=Path,required=True);build.add_argument('--reviewer',required=True)
    build.add_argument('--output',type=Path,required=True)
    export=sub.add_parser('export');export.add_argument('--data',type=Path,required=True)
    export.add_argument('--annotations',type=Path,required=True);export.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='build':
        bundle=build_bundle(args.source,args.cohort,args.reviewer)
        with args.output.open('x',encoding='utf-8') as handle: json.dump(bundle,handle,ensure_ascii=False,indent=2)
    else:
        dataset=json.loads(args.data.read_text())
        if not args.annotations.exists(): raise FileNotFoundError(args.annotations)
        store=IdentifiedStore(dataset,args.annotations)
        result=export_judgments(dataset,store.state,args.output)
        print(json.dumps({'decisions':len(result['decisions']),'covered_answers':sum(r['reviewed'] for r in result['coverage'])}))


if __name__=='__main__': main()
