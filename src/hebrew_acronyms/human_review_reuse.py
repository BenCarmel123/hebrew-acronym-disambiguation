"""Verify explicitly proposed historical reuse without migrating judgments."""
import copy
import hashlib
import json
from pathlib import Path

from .human_review_masked import LABELS


def verified_reuse_proposals(report_path, source_root, dataset):
    report_path, source_root = Path(report_path), Path(source_root)
    report = json.loads(report_path.read_text())
    sources = {}
    for relative, expected in report['source_sha256'].items():
        path = source_root / relative
        if not path.resolve().is_relative_to(source_root.resolve()):
            raise ValueError('Historical source outside source root')
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError('Historical source hash changed')
        sources[path.name] = json.loads(content)
    old = sources['review-data-continuation-v1.json']
    annotations = sources['annotations.continuation-v1.json']
    if sources['review-data-381.json'] != dataset:
        raise ValueError('Historical proposal refers to another current dataset')
    original = {i['id']: i for i in old['items']}
    current = {i['id']: i for i in dataset['items']}
    proposals = {}
    for match in report['historical_exact_context_response_matches']:
        group = current[match['group_id']]
        item = original[match['original_item_id']]
        answer = group['answers'][0]
        if group['original_item_id'] != item['id'] or any(group.get(k) != item.get(k)
                for k in ('sentence', 'acronym', 'gold', 'target_spans')):
            raise ValueError('Historical context or target mismatch')
        if [a['binding'] for a in group['occurrences']] != match['new_bindings']:
            raise ValueError('New response bindings changed')
        prior = []
        record = annotations['records'].get(item['id'], {})
        for old_answer in item['answers']:
            judgment = record.get('judgments', {}).get(old_answer['id'], {})
            if (old_answer['task'], old_answer['raw']) != (answer['task'], answer['raw']):
                continue
            if answer['task'] == 'selection' and any(old_answer.get(k) != answer.get(k)
                    for k in ('decoded', 'option_mapping')):
                continue
            if judgment.get('label') in LABELS:
                prior.append(dict(answer=copy.deepcopy(old_answer), judgment=copy.deepcopy(judgment),
                                  record_exposure=copy.deepcopy(record.get('exposure', {}))))
        summaries = [dict(answer_id=p['answer']['id'], **{k:p['judgment'].get(k)
                     for k in ('label', 'origin', 'label_updated_at', 'annotator')}) for p in prior]
        if sorted(summaries, key=lambda p:p['answer_id']) != sorted(match['previous_judgments'], key=lambda p:p['answer_id']):
            raise ValueError('Historical judgments differ from reviewed proposal')
        if len({p['judgment']['label'] for p in prior}) != 1:
            raise ValueError('Conflicting or missing historical judgments')
        if any(not p['judgment'].get('label_updated_at') or not p['judgment'].get('origin') for p in prior):
            raise ValueError('Missing historical judgment provenance')
        proposals[group['id']] = dict(group_id=group['id'], original_item_id=item['id'],
            label=prior[0]['judgment']['label'], historical_sources=prior,
            source_sha256=copy.deepcopy(report['source_sha256']),
            coordinator_review_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
            historical_source_identity=old['source_identity'],
            new_bindings=copy.deepcopy(match['new_bindings']),
            new_answer_ids=[a['id'] for a in group['occurrences']])
    if len(proposals) != report['match_count'] or sum(len(p['new_bindings']) for p in proposals.values()) != report['new_answer_occurrences']:
        raise ValueError('Historical proposal totals differ')
    return proposals
