"""Select a bounded diagnostic review; unselected contexts receive no judgment.

Formatting-normalized variants are sampling strata only. They never transfer a
label between sentences, models, or response bindings.
"""
from collections import Counter, defaultdict
import copy
from difflib import SequenceMatcher
import re

from .human_review_short import digest
from .human_review_masked import contains_foreign_letters


def variant_key(item):
    a = item['answers'][0]
    normalize = lambda text: re.sub(r'\W+', '', text or '', flags=re.UNICODE)
    return (a['task'], item['acronym'], normalize(item['gold']),
            normalize(a.get('decoded') if a['task'] == 'selection' else a['raw']))


def build_focus(dataset, previous_dataset, previous_state, *, limit=200, minutes=20):
    if not 1 <= limit <= 200 or not 1 <= minutes <= 20:
        raise ValueError('Focused review permits at most 200 decisions and 20 minutes')
    if dataset['provenance']['cohort_sha256'] != previous_dataset['provenance']['cohort_sha256']:
        raise ValueError('Different scored cohorts')
    old = {i['id']: i for i in previous_dataset['items']}
    current = {i['id']: i for i in dataset['items']}
    # Validate that every old occurrence still has the exact original binding.
    old_answers = {a['id']: a for i in old.values() for a in i['occurrences']}
    current_answers = {a['id']: a for i in current.values() for a in i['occurrences']}
    if any(current_answers.get(k) != a for k, a in old_answers.items()):
        raise ValueError('Prior answer content or binding changed')
    completed, uncertain, seen_variants = set(), set(), set()
    for i, item in old.items():
        a = item['answers'][0]
        label = previous_state['records'].get(i, {}).get('judgments', {}).get(a['id'], {}).get('label')
        if i in previous_state.get('historical_reuse', {}):
            label = previous_state['historical_reuse'][i]['proposal']['label']
        if label == 'unsure':
            uncertain.add(i)
        elif label in {'fits', 'not_fits'}:
            completed.add(i); seen_variants.add(variant_key(item))
    eligible = [i for i in dataset['items'] if not i['answers'][0]['technical_failure']
                and i['id'] not in completed and
                (i['id'] in uncertain or variant_key(i) not in seen_variants)]

    def gemini(item):
        return any(a['system_id'].startswith('gemini_') for a in item['occurrences'])

    def rank(item):
        a = item['answers'][0]; text = a.get('decoded') or a['raw']
        similarity = SequenceMatcher(None, variant_key(item)[-1], re.sub(r'\W+', '', item['gold'])).ratio()
        return (contains_foreign_letters(text), -similarity, item['id'])

    selected, reasons, chosen_variants = [], {}, set()

    def take(name, pool, count):
        pools = defaultdict(list)
        for item in sorted(pool, key=rank):
            pools[item['answers'][0]['system_id']].append(item)
        taken = 0
        while pools and taken < count and len(selected) < limit:
            for system in sorted(list(pools)):
                pool = pools[system]
                while pool:
                    item = pool.pop(0); key = variant_key(item)
                    if item['id'] not in reasons and key not in chosen_variants:
                        selected.append(item); reasons[item['id']] = name
                        chosen_variants.add(key); taken += 1
                        break
                if not pool: del pools[system]
                if taken >= count or len(selected) >= limit: break

    negative = lambda i: i['answers'][0]['task'] == 'generation' and not i['answers'][0]['auto_score']
    addition = lambda i: i['answers'][0]['task'] == 'generation' and i['priority_evidence']['positive_with_additions']
    selection = lambda i: i['answers'][0]['task'] == 'selection' and not i['answers'][0]['auto_score']
    take('previous_unsure_recheck', [i for i in eligible if i['id'] in uncertain], 2)
    take('gemini_generation_negative', [i for i in eligible if gemini(i) and negative(i)], 70)
    take('other_generation_negative', [i for i in eligible if not gemini(i) and negative(i)], 70)
    take('gemini_positive_additions', [i for i in eligible if gemini(i) and addition(i)], 15)
    take('other_positive_additions', [i for i in eligible if not gemini(i) and addition(i)], 15)
    take('gemini_selection_negative', [i for i in eligible if gemini(i) and selection(i)], 10)
    take('other_selection_negative', [i for i in eligible if not gemini(i) and selection(i)], 10)
    # A small control component does not make this a representative sample.
    controls = sorted(eligible, key=lambda i: digest(['control', i['id']]))[:30]
    take('diagnostic_control', controls, 8)
    take('additional_novel_generation', [i for i in eligible if negative(i)], limit-len(selected))
    take('additional_novel_case', eligible, limit-len(selected))
    # Interleave strata so an early stop still samples Gemini and positive audits.
    strata = defaultdict(list)
    for item in selected: strata[reasons[item['id']]].append(item)
    selected = []
    while strata:
        for reason in list(strata):
            selected.append(strata[reason].pop(0))
            if not strata[reason]: del strata[reason]

    focused = copy.deepcopy(dataset)
    focused['items'] = copy.deepcopy(selected)
    ids = [i['id'] for i in selected]
    policy = dict(minutes=minutes, max_decisions=limit, selected_decisions=len(ids),
                  full_source_answers=len(current_answers), full_source_groups=len(current),
                  full_dataset_id=dataset['dataset_id'], previous_dataset_id=previous_dataset['dataset_id'],
                  previous_annotation_sha256=digest(previous_state),
                  selection_method='Deterministic mechanical diagnostic triage; no AI or human labels added',
                  unselected_status='Unreviewed; no judgment is transferred from a selected representative',
                  reasons=reasons, reason_counts=dict(Counter(reasons.values())),
                  start_policy='Explicit user start; preparation excluded')
    focused['provenance']['focused_review'] = policy
    plan = dict(queue=ids, calibration=[], prioritized=ids, positives=[], selection=[],
                selection_reasons={i: {'group': reasons[i], 'not_human_judgment': True} for i in ids},
                focus=policy)
    plan['plan_id'] = digest(plan)
    focused['short_plan'] = plan
    focused['dataset_id'] = focused['source_identity'] = digest([focused['provenance'], focused['items']])
    return focused
