"""Offline P1 proposals; never build model inputs or alter source/decision values.

Run the installed package from the repository root::
    python -m hebrew_acronyms.data_processing.prepare_study_review --repo .

All writes are confined to data/study_v1/review. No model implementations, network, or research pipeline are imported; only
the existing dependency-free span helper is reused. CSV record numbers are 1-based data records,
not physical line numbers (source text can contain newlines).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

from hebrew_acronyms.models.common.pairs import find_span

SCHEMA = 'p1-proposal-v1'
IDENTITY = 'ראשי תיבות שם פרטי ומשפחה (זהות חסויה)'
CONTENT_FIELDS = ('acronym', 'sentence', 'gold_expansion', 'candidates')
NATURAL = {'knesset', 'wiki_natural'}
QUOTE_MAP = str.maketrans({'״': '"', '“': '"', '”': '"', '׳': "'", '‘': "'", '’': "'"})
DECISION_FIELDS = ['review_id', 'reviewer', 'decision', 'label_decision', 'corrected_label',
                   'target_decision', 'approved_span_start', 'approved_span_end',
                   'inventory_decision', 'approved_aliases', 'evidence_reference',
                   'reason', 'decided_at', 'elapsed_seconds', 'problem_types']
RISK = {'inventory_conflict': 8, 'label_missing_independent_inventory': 10,
        'missing_label': 10, 'target_missing': 10, 'target_multiple': 8,
        'uncertain_review': 10, 'adverse_review': 10, 'scope_exclusion_proposed': 8,
        'document_overlap': 9, 'target_prefix_or_embedded': 6, 'quote_variant': 3,
        'definition_or_expansion_in_sentence': 7, 'missing_provenance': 8,
        'review_not_proven': 5, 'duplicate_text': 7, 'document_key_variant': 6,
        'legacy_id_collision': 6, 'candidate_count_mismatch': 7,
        'label_missing_row_inventory': 10, 'natural_dev_proposal': 5,
        'historical_adverse_review': 9}


def dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def content_key(row):
    return tuple(row.get(k, '') for k in CONTENT_FIELDS)


def document_key(row):
    source, title = row.get('source', ''), row.get('page_title', '')
    return (source, title) if source.strip() and title.strip() else None


def candidates(row):
    # Matches stored-pair cardinality; repeated values remain repeated pairs.
    return [c.strip() for c in row.get('candidates', '').split('|') if c.strip()]


def scope_reason(expansion):
    if 'גימטר' in expansion:
        return 'gematria_string_rule_proposed'
    if expansion == IDENTITY:
        return 'exact_concealed_identity_string_rule_proposed'
    return ''


def in_scope(row):
    return not scope_reason(row.get('gold_expansion', ''))


def quote_fold(value):
    return value.translate(QUOTE_MAP)


def spans(sentence, acronym):
    if not acronym:
        return []
    return [[m.start(), m.end()] for m in re.finditer(re.escape(quote_fold(acronym)), quote_fold(sentence))]


def row_id(row):
    return 'item-' + digest(dump(content_key(row)))[:24]


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def csv_text(rows, fields):
    import io
    handle = io.StringIO(newline='')
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def docset(rows):
    return {d for r in rows if (d := document_key(r)) is not None}


def summary(rows):
    return {'rows': len(rows), 'types': len({r['acronym'] for r in rows}),
            'documents': len(docset(rows)), 'missing_document_rows': sum(document_key(r) is None for r in rows), 'categories': dict(sorted(Counter(r['category'] for r in rows).items())),
            'observed_type_label_pairs': len({(r['acronym'], r['gold_expansion']) for r in rows}),
            'stored_candidate_pairs': sum(len(candidates(r)) for r in rows),
            'historical_pair_builder_pairs': sum(len(candidates(r)) for r in rows
                                               if len(candidates(r)) >= 2 and find_span(r['sentence'], r['acronym']) is not None)}


def structural(splits):
    train, dev, test, all_rows = (splits[k] for k in ('train', 'dev', 'test', 'all'))
    union = Counter(content_key(r) for role in ('train', 'dev', 'test') for r in splits[role])
    remaining = union.copy()
    extra = []
    for r in all_rows:
        k = content_key(r)
        if remaining[k]:
            remaining[k] -= 1
        else:
            extra.append(r)
    docs = {role: docset(splits[role]) for role in ('train', 'dev', 'test')}
    dev_types = {r['acronym'] for r in dev}
    dev_raw = [r for r in extra if r['category'] == 'knesset' and r['acronym'] in dev_types]
    dev_scope = [r for r in dev_raw if in_scope(r)]
    natural_dev = [r for r in dev_scope if document_key(r) not in docs['test']]
    natural_docs = docset(natural_dev)
    blocked = docs['test'] | docs['dev'] | natural_docs
    overlap_test = [r for r in train if document_key(r) in docs['test']]
    overlap_old_dev = [r for r in train if document_key(r) in docs['dev'] - docs['test']]
    overlap_new_dev = [r for r in train if document_key(r) in natural_docs - docs['test'] - docs['dev']]
    retained = [r for r in train if document_key(r) not in blocked]
    natural_test = [r for r in test if r['category'] in NATURAL and in_scope(r)]
    target_types = {r['acronym'] for r in natural_test}
    extension = [r for r in extra if r['acronym'] in target_types and document_key(r) not in blocked]
    dev_inventory = defaultdict(set)
    for r in dev:
        dev_inventory[r['acronym']].update(candidates(r))
    metrics = {
        'split_counts': {k: summary(v) for k, v in splits.items()},
        'aggregate_minus_split_union': summary(extra),
        'split_union_records_absent_from_aggregate': sum(remaining.values()),
        'natural_diagnostic_all': summary([r for r in all_rows if r['category'] in NATURAL and in_scope(r)]),
        'natural_diagnostic_test': summary(natural_test),
        'extra_knesset_dev_types_before_scope': summary(dev_raw),
        'extra_knesset_dev_types_in_scope': summary(dev_scope),
        'natural_dev_proposed': summary(natural_dev),
        'natural_dev_old_inventory_label_coverage': sum(r['gold_expansion'] in dev_inventory[r['acronym']] for r in natural_dev),
        'natural_dev_scope_before_test_block_old_inventory_coverage': sum(r['gold_expansion'] in dev_inventory[r['acronym']] for r in dev_scope),
        'natural_dev_also_blocking_train_documents': summary([r for r in natural_dev if document_key(r) not in docs['train']]),
        'train_overlap_test_rows': len(overlap_test), 'train_overlap_old_dev_additional_rows': len(overlap_old_dev),
        'train_overlap_natural_dev_additional_rows': len(overlap_new_dev),
        'train_after_document_proposals': summary(retained),
        'extension_available': summary(extension),
        'historical_document_overlap_train_dev': len(docs['train'] & docs['dev']),
        'historical_document_overlap_train_test': len(docs['train'] & docs['test']),
        'missing_document_keys': {k: sum(document_key(r) is None for r in v) for k, v in splits.items()},
        'content_duplicate_records': {k: len(v) - len({content_key(r) for r in v}) for k, v in splits.items()},
        'acronym_sentence_duplicate_records': {k: len(v) - len({(r['acronym'], r['sentence']) for r in v}) for k, v in splits.items()},
        'legacy_id_collisions': {k: sum(n - 1 for n in Counter(r['item_id'] for r in v).values() if n > 1) for k, v in splits.items()},
        'scope_proposal_counts': {k: dict(Counter(scope_reason(r['gold_expansion']) for r in v if scope_reason(r['gold_expansion']))) for k, v in splits.items()},
    }
    return metrics, natural_dev, retained, extension, natural_test, docs, blocked


def extension_scenarios(retained, extension, natural_test):
    control = [r for r in retained if r['category'] == 'wiki_substituted']
    substituted = [r for r in extension if r['category'] == 'wiki_substituted']
    control_buckets = defaultdict(list)
    match_key = lambda r: (r['source'], len(candidates(r)))
    for r in control:
        control_buckets[match_key(r)].append(r)
    # Availability-only ranking: never inspect a gold label, candidate string or score.
    order = lambda r: (r['source'], r['page_title'], r['item_id'], digest(r['sentence']))
    for bucket in control_buckets.values():
        bucket.sort(key=order)
    matchable = [r for r in substituted if match_key(r) in control_buckets]
    by_type = defaultdict(list)
    for r in matchable:
        by_type[r['acronym']].append(r)
    capped = []
    for acronym in sorted(by_type):
        rows = by_type[acronym]
        by_doc = defaultdict(list)
        for r in rows:
            if document_key(r) is not None:
                by_doc[document_key(r)].append(r)
        if len(by_doc) < 2:
            continue
        # Round-robin distinct documents guarantees >=2 selected documents.
        buckets = [sorted(by_doc[d], key=order) for d in sorted(by_doc)]
        selected = []
        while buckets and len(selected) < 5:
            for bucket in buckets:
                if bucket and len(selected) < 5:
                    selected.append(bucket.pop(0))
            buckets = [b for b in buckets if b]
        capped.extend(selected)
    scenarios = [('all_available', extension), ('same_source_substituted', substituted),
                 ('exact_candidate_count', matchable), ('capped_5_min_2_documents', capped)]
    output, memberships, summaries = [], [], {}
    for name, rows in scenarios:
        types = {r['acronym'] for r in rows}
        target = [r for r in natural_test if r['acronym'] in types]
        observed = {(r['acronym'], r['gold_expansion']) for r in rows}
        need = Counter(match_key(r) for r in rows if r['category'] == 'wiki_substituted')
        unmatched = sum(max(0, n-len(control_buckets[k])) for k, n in need.items())
        unmatched += sum(r['category'] != 'wiki_substituted' for r in rows)
        selected_controls = [r for k, n in sorted(need.items()) for r in control_buckets[k][:n]]
        pairs = sum(len(candidates(r)) for r in rows)
        controls_pairs = sum(len(candidates(r)) for r in selected_controls)
        report = summary(rows)
        full_pairs = summary(retained)['historical_pair_builder_pairs']
        replacement_pairs = summary(rows)['historical_pair_builder_pairs']
        removed_pairs = summary(selected_controls)['historical_pair_builder_pairs']
        report.update({'shared_test_items_by_type_availability': len(target),
                       'shared_test_type_label_pairs': len({(r['acronym'], r['gold_expansion']) for r in target}),
                       'test_items_with_observed_extension_label': sum((r['acronym'], r['gold_expansion']) in observed for r in target),
                       'test_type_label_pairs_with_observed_extension_label': len({(r['acronym'], r['gold_expansion']) for r in target} & observed),
                       'unmatched_rows': unmatched, 'control_rows': len(selected_controls),
                       'control_pairs': controls_pairs,
                       'retained_full_training_usable_pairs': full_pairs,
                       'replacement_condition_full_usable_pairs': full_pairs - removed_pairs + replacement_pairs if unmatched == 0 else None,
                       'illustrative_unapproved_batch16_epoch1_updates_core': math.ceil(full_pairs / 16),
                       'illustrative_unapproved_batch16_epoch1_updates_extension': math.ceil((full_pairs - removed_pairs + replacement_pairs) / 16) if unmatched == 0 else None,
                       'illustration_assumptions': 'one epoch, effective batch 16, no drop_last, no accumulation remainder change; proposed not approved',
                       'source_candidate_histogram': {dump(k): n for k,n in sorted(need.items())},
                       'exact_source_row_pair_match': (unmatched == 0 and pairs == controls_pairs
                                                     and replacement_pairs == removed_pairs == pairs
                                                     and all(document_key(r) is not None for r in rows + selected_controls)),
                       'training_update_budget': 'conditional: identical full pair totals, effective batch, epochs, drop_last, accumulation and checkpoint rule required; no training run',
                       'selection_rule': 'source/document/legacy-ID/text-hash; document round-robin for cap; no label or system-score selection'})
        summaries[name] = report
        for acronym in ['__ALL__'] + sorted(types):
            rs = rows if acronym == '__ALL__' else [r for r in rows if r['acronym'] == acronym]
            ts = target if acronym == '__ALL__' else [r for r in target if r['acronym'] == acronym]
            obs = {(r['acronym'], r['gold_expansion']) for r in rs}
            output.append({'scenario': name, 'acronym': acronym, 'proposal_status': 'protocol pending',
                           'rows': len(rs), 'types': len({r['acronym'] for r in rs}), 'documents': len(docset(rs)),
                           'source_categories': dump(dict(Counter(r['category'] for r in rs))),
                           'stored_candidate_pairs': sum(len(candidates(r)) for r in rs),
                           'candidate_count_histogram': dump(dict(Counter(len(candidates(r)) for r in rs))),
                           'shared_test_items': len(ts), 'observed_extension_senses': len(obs),
                           'test_items_label_covered_descriptive_only': sum((r['acronym'], r['gold_expansion']) in obs for r in ts),
                           'control_match_status': ('available' if report['exact_source_row_pair_match'] else 'not_exact') if acronym == '__ALL__' else 'see_scenario_aggregate',
                           'unmatched_rows': unmatched if acronym == '__ALL__' else '',
                           'update_budget': report['training_update_budget'],
                           'full_training_usable_pairs_core': full_pairs if acronym == '__ALL__' else '',
                           'full_training_usable_pairs_extension': report['replacement_condition_full_usable_pairs'] if acronym == '__ALL__' else '',
                           'illustrative_unapproved_batch16_epoch1_updates_core': math.ceil(full_pairs / 16) if acronym == '__ALL__' else '',
                           'illustrative_unapproved_batch16_epoch1_updates_extension': report['illustrative_unapproved_batch16_epoch1_updates_extension'] if acronym == '__ALL__' else '',
                           'limitations': 'stored inventories only; semantic validity, final spans, inventory reconciliation, lengths and training budget remain unapproved'})
        for role, rs in [('extension_proposed', rows), ('control_replacement_proposed', selected_controls)]:
            for r in rs:
                memberships.append({'scenario': name, 'role': role, 'item_id': row_id(r),
                                    'acronym': r['acronym'], 'candidate_pairs': len(candidates(r)),
                                    'proposal_status': 'not an active training input'})
    return output, memberships, summaries


def prepare(repo):
    repo = Path(repo).resolve()
    output = repo / 'data/study_v1/review'
    # Do not follow a destination symlink outside the only authorized output directory.
    if output.resolve() != output or any(p.is_symlink() for p in [output, *output.parents[:3]]):
        raise ValueError('Review output must not be a symlink')
    split_paths = {s: f'data/splits/{s}_items.csv' for s in ('train', 'dev', 'test', 'all')}
    source_paths = sorted(set(p.relative_to(repo).as_posix() for p in (repo/'data/mined').rglob('*') if p.is_file()) |
                          set(p.relative_to(repo).as_posix() for p in (repo/'data/splits').rglob('*') if p.is_file()) |
                          {'data/README.md', 'src/hebrew_acronyms/models/common/pairs.py'})
    source_hashes = {p: hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in source_paths}
    tables = {p: read_csv(repo/p) for p in source_paths if p.endswith('.csv')}
    splits = {s: tables[p] for s, p in split_paths.items()}
    metrics, natural_dev, retained, extension, natural_test, docs, blocked = structural(splits)
    ext_rows, ext_members, ext_summary = extension_scenarios(retained, extension, natural_test)
    metrics['extension_scenarios'] = ext_summary
    metrics['control_substituted_after_document_proposals'] = summary([r for r in retained if r['category']=='wiki_substituted'])

    sources_by_key = defaultdict(list)
    split_roles = defaultdict(set)
    row_by_key = {}
    for role in ('all', 'train', 'dev', 'test'):
        path = split_paths[role]
        for record, row in enumerate(splits[role], 1):
            key = content_key(row)
            sources_by_key[key].append({'path': path, 'record': record})
            row_by_key.setdefault(key, row)
            if role != 'all':
                split_roles[key].add(role)
    if any(n > 1 for n in Counter(content_key(r) for r in splits['all']).values()):
        raise ValueError('Exact duplicate aggregate content needs an explicit occurrence identity policy')
    # Preserve differences rather than assuming aggregate metadata equals split metadata.
    metadata_differences = defaultdict(list)
    for role in ('train', 'dev', 'test'):
        for n, row in enumerate(splits[role], 1):
            diff = {k: v for k, v in row.items() if v != row_by_key[content_key(row)].get(k)}
            if diff:
                metadata_differences[content_key(row)].append({'path': split_paths[role], 'record': n, 'values': diff})

    central = defaultdict(set)
    evidence = defaultdict(list)
    for path in ('data/mined/candidate_table.csv', 'data/mined/merged_counts.csv',
                 'data/mined/wiktionary/wiktionary_counts.csv', 'data/mined/wikipedia/bullet_counts.csv'):
        for n, r in enumerate(tables[path], 1):
            pair = (r['acronym'], r['expansion'])
            evidence[pair].append({'path': path, 'record': n, 'source': r.get('sources', r.get('source', '')), 'page_title': r.get('page_title', '')})
            if path.endswith('candidate_table.csv'):
                central[r['acronym']].add(r['expansion'])
    row_inventories = defaultdict(set)
    offered = defaultdict(set)
    observed = defaultdict(set)
    candidate_refs = defaultdict(list)
    for key, row in row_by_key.items():
        acronym = row['acronym']
        row_inventories[acronym].add(tuple(sorted(set(candidates(row)))))
        offered[acronym].update(candidates(row))
        observed[acronym].add(row['gold_expansion'])
        for c in set(candidates(row)):
            candidate_refs[(acronym, c)].append(row_id(row))
    inventory = []
    for acronym, expansion in sorted(set(evidence) | set(candidate_refs)):
        variants = sorted(e for a, e in (set(evidence) | set(candidate_refs))
                          if a == acronym and e != expansion and ' '.join(quote_fold(e).split()) == ' '.join(quote_fold(expansion).split()))
        inventory.append({'inventory_id': 'inv-'+digest(dump([acronym, expansion]))[:24],
                          'acronym': acronym, 'expansion_raw': expansion, 'aliases_proposed_json': dump(variants),
                          'alias_proposal_basis': 'quote/whitespace equivalence only; human approval required' if variants else '',
                          'independent_export_evidence_json': dump(evidence[(acronym, expansion)]),
                          'in_central_inventory': str(expansion in central[acronym]).lower(),
                          'offered_by_item_ids_json': dump(candidate_refs[(acronym, expansion)]),
                          'observed_label_in_aggregate': str(expansion in observed[acronym]).lower(),
                          'central_plus_item_candidate_strings_for_type': len(central[acronym] | offered[acronym]),
                          'all_source_hypothesis_strings_for_type': len({e for a,e in set(evidence) | set(candidate_refs) if a==acronym}),
                          'observed_label_strings_for_type': len(observed[acronym]),
                          'scope_proposal': scope_reason(expansion),
                          'proposal': 'review_exclusion' if scope_reason(expansion) else 'verify_independent_evidence' if evidence[(acronym, expansion)] else 'unsupported_row_candidate_review',
                          'human_decision': ''})
    # Reviews match multiple fields; historical item_id alone is never sufficient.
    review_lookup = defaultdict(list)
    historical_reviews = defaultdict(list)
    aggregate_by_legacy_type = defaultdict(list)
    for row in row_by_key.values():
        aggregate_by_legacy_type[(row['item_id'], row['acronym'])].append(row)
    unmatched_reviews = []
    for path in ('data/mined/dev_review.csv', 'data/mined/knesset/knesset_reviewed.csv'):
        for n, r in enumerate(tables[path], 1):
            key = (r.get('item_id'), r.get('acronym'), r.get('expansion', r.get('gold_expansion')))
            entry = {'path': path, 'record': n, 'verdict': r.get('verdict', r.get('review_verdict')), 'note': r.get('note', ''),
                     'source_record': r, 'link_basis': 'legacy_id+acronym+label; inspect source sentence where available'}
            possible = aggregate_by_legacy_type[(r.get('item_id'), r.get('acronym'))]
            exact = [x for x in possible if x['gold_expansion']==key[2] and ('sentence' not in r or r['sentence']==x['sentence'])]
            if len(exact)==1:
                review_lookup[key].append(entry)
            elif len(possible)==1 and 'sentence' not in r:
                entry['link_basis'] = 'unique legacy_id+acronym; label_changed; historical evidence not current verdict'
                entry['current_label'] = possible[0]['gold_expansion']
                historical_reviews[content_key(possible[0])].append(entry)
            else:
                unmatched_reviews.append({'path': path, 'record': n, 'reason': 'absent_or_ambiguous_item_link', 'source_record_json': dump(r)})
    type_review = defaultdict(list)
    for path in ('data/mined/duplicate_review.csv', 'data/mined/merge_review.csv', 'data/mined/wikipedia/unmined_triage.csv'):
        for n, r in enumerate(tables[path], 1):
            type_review[r['acronym']].append({'path': path, 'record': n, 'source_record': r})
    id_keys, text_keys, normalized_docs = defaultdict(set), defaultdict(set), defaultdict(set)
    for key, r in row_by_key.items():
        id_keys[r['item_id']].add(key)
        text_keys[(r['acronym'], quote_fold(' '.join(r['sentence'].split())))].add(key)
        if document_key(r):
            normalized_docs[(r['source'].strip().lower(), quote_fold(' '.join(r['page_title'].split())))].add(document_key(r))
    nd_keys = {content_key(r) for r in natural_dev}
    extension_keys = {content_key(r) for r in extension}
    audits = []
    for key, r in row_by_key.items():
        acronym, sentence, label = (r[k] for k in ('acronym', 'sentence', 'gold_expansion'))
        matches = spans(sentence, acronym)
        flags = []
        def flag(condition, name):
            if condition:
                flags.append(name)
        flag(len(row_inventories[acronym]) > 1 or set(candidates(r)) != central[acronym], 'inventory_conflict')
        flag(not evidence[(acronym, label)], 'label_missing_independent_inventory')
        flag(not label.strip(), 'missing_label')
        flag(label not in candidates(r), 'label_missing_row_inventory')
        flag(not matches, 'target_missing')
        flag(len(matches) > 1, 'target_multiple')
        flag(any((s > 0 and '\u05d0' <= sentence[s-1] <= '\u05ea') or (e < len(sentence) and '\u05d0' <= sentence[e] <= '\u05ea') for s, e in matches), 'target_prefix_or_embedded')
        flag(any(sentence[s:e] != acronym for s, e in matches), 'quote_variant')
        flag(bool(label and len(label) > 2 and label in sentence) or 'ראשי תיבות' in sentence or any(re.match(r'\s*\(', sentence[e:]) for _, e in matches), 'definition_or_expansion_in_sentence')
        review_refs = review_lookup[(r['item_id'], acronym, label)]
        verdicts = {r.get('review_verdict', '')} | {ref['verdict'] for ref in review_refs}
        flag(bool(verdicts & {'unsure', 'uncertain'}), 'uncertain_review')
        flag(bool(verdicts & {'wrong_sense', 'broken', 'reject'}), 'adverse_review')
        flag(any(ref['verdict'] in {'wrong_sense', 'broken', 'reject'} for ref in historical_reviews[key]), 'historical_adverse_review')
        flag(not r.get('provenance') or document_key(r) is None or not (r.get('label_origin') or r.get('label_status') or review_refs), 'missing_provenance')
        flag(not review_refs and r.get('review_verdict') not in {'clean', 'human_review'}, 'review_not_proven')
        flag(bool(scope_reason(label)), 'scope_exclusion_proposed')
        flag('train' in split_roles[key] and document_key(r) in blocked, 'document_overlap')
        flag(len(text_keys[(acronym, quote_fold(' '.join(sentence.split())))]) > 1, 'duplicate_text')
        norm_doc = (r['source'].strip().lower(), quote_fold(' '.join(r['page_title'].split())))
        flag(len(normalized_docs[norm_doc]) > 1, 'document_key_variant')
        flag(len(id_keys[r['item_id']]) > 1, 'legacy_id_collision')
        flag(r.get('n_candidates', '') != str(len(candidates(r))), 'candidate_count_mismatch')
        flag(key in nd_keys, 'natural_dev_proposal')
        role = '+'.join(sorted(split_roles[key])) or 'aggregate_only'
        doc = document_key(r)
        audit = {'stable_item_id': row_id(r), 'source_refs_json': dump(sources_by_key[key]),
                 'historical_roles': role, 'natural_dev_proposed': str(key in nd_keys).lower(),
                 'extension_pool_proposed': str(key in extension_keys).lower(),
                 'doc_id': 'doc-'+digest(dump(doc))[:24] if doc else '',
                 'doc_key_json': dump(doc) if doc else '',
                 'span_start_proposed': matches[0][0] if len(matches) == 1 else '',
                 'span_end_proposed': matches[0][1] if len(matches) == 1 else '',
                 'all_target_spans_json': dump(matches), 'span_status': 'unique_substring_pending_boundary_review' if len(matches)==1 else 'human_target_selection_required',
                 'flags': '|'.join(sorted(flags)), 'risk_score': sum(RISK[f] for f in flags),
                 'scope_proposal': scope_reason(label),
                 'review_evidence_json': dump(review_refs), 'historical_label_changed_review_evidence_json': dump(historical_reviews[key]), 'type_review_evidence_json': dump(type_review[acronym]),
                 'source_metadata_differences_json': dump(metadata_differences[key]),
                 'proposal_status': 'protocol pending'}
        audit.update({'raw_'+k: v for k, v in r.items()})
        audits.append(audit)
    audits.sort(key=lambda r: r['stable_item_id'])
    queue = []
    for r in sorted(audits, key=lambda r: (-r['risk_score'], r['stable_item_id'])):
        queue.append({'review_id': r['stable_item_id'], 'review_kind': 'item', 'risk_score': r['risk_score'],
                      'flags': r['flags'], 'historical_roles': r['historical_roles'],
                      'natural_dev_proposed': r['natural_dev_proposed'],
                      'lookup_file': 'item_audit.csv', 'human_decision': ''})
    for r in inventory:
        queue.append({'review_id': r['inventory_id'], 'review_kind': 'inventory',
                      'risk_score': 10 if r['proposal']=='unsupported_row_candidate_review' else 8 if r['scope_proposal'] else 4,
                      'flags': r['proposal'], 'historical_roles': 'type_inventory', 'natural_dev_proposed': '',
                      'lookup_file': 'inventory_proposed.csv', 'human_decision': ''})
    queue.sort(key=lambda r: (-r['risk_score'], r['review_kind'], r['review_id']))
    for rank, row in enumerate(queue, 1):
        row['queue_rank'] = rank
    eligible_pilot = [r for r in audits if 'test' not in r['historical_roles'] and
                      (r['historical_roles'] in {'train', 'dev'} or r['natural_dev_proposed']=='true')]
    eligible_pilot.sort(key=lambda r: (-r['risk_score'], r['stable_item_id']))
    chosen = []
    # Coverage of roles and risk classes, followed by deterministic risk order.
    for criterion in [lambda r: r['natural_dev_proposed']=='true', lambda r: r['historical_roles']=='dev', lambda r: r['historical_roles']=='train'] + [lambda r, f=f: f in r['flags'].split('|') for f in RISK]:
        match = next((r for r in eligible_pilot if criterion(r) and r not in chosen), None)
        if match is not None and len(chosen)<16:
            chosen.append(match)
    for r in eligible_pilot:
        if len(chosen) >= 16:
            break
        if r not in chosen:
            chosen.append(r)
    pilot = [{'pilot_order': i, 'review_id': r['stable_item_id'], 'historical_roles': r['historical_roles'],
              'natural_dev_proposed': r['natural_dev_proposed'], 'acronym': r['raw_acronym'],
              'sentence': r['raw_sentence'], 'recorded_label': r['raw_gold_expansion'],
              'stored_candidates': r['raw_candidates'], 'flags': r['flags'],
              'source_refs_json': r['source_refs_json'], 'decision_location': 'review_decisions.csv',
              'human_pilot_status': 'not performed'} for i, r in enumerate(chosen, 1)]
    decisions_path = output/'review_decisions.csv'
    old_decisions = read_csv(decisions_path) if decisions_path.exists() else []
    ids = {r['review_id'] for r in queue}
    if len(ids) != len(queue):
        raise ValueError('Review IDs are not unique')
    old_by_id = {r['review_id']: r for r in old_decisions}
    if len(old_by_id) != len(old_decisions) or set(old_by_id) - ids:
        raise ValueError('Existing decision IDs duplicated/orphaned; explicit reconciliation required')
    if old_decisions and list(old_decisions[0]) != DECISION_FIELDS:
        raise ValueError('Existing decisions schema differs; refusing to overwrite')
    old_manifest_path = output/'source_manifest.json'
    if old_decisions and not old_manifest_path.exists():
        raise ValueError('Existing decisions lack source manifest; explicit reconciliation required')
    if old_decisions and old_manifest_path.exists():
        old_hashes = {r['path']: r['sha256'] for r in json.loads(old_manifest_path.read_text())['sources']}
        if old_hashes != source_hashes:
            raise ValueError('Sources changed after review preparation; explicit reconciliation required')
    decisions = [old_by_id.get(r['review_id'], {k: r['review_id'] if k=='review_id' else '' for k in DECISION_FIELDS}) for r in queue]
    metrics.update({'item_audit_rows': len(audits), 'inventory_proposals': len(inventory),
                    'review_queue_rows': len(queue), 'risk_flags': dict(sorted(Counter(f for r in audits for f in r['flags'].split('|') if f).items())),
                    'metadata_difference_items': sum(bool(v) for v in metadata_differences.values()),
                    'historical_label_changed_review_links': sum(len(v) for v in historical_reviews.values()),
                    'unmatched_review_records': len(unmatched_reviews),
                    'pilot': {'prepared_items': len(pilot), 'human_status': 'not performed', 'measured_seconds': None,
                              'review_completion_estimate': None, 'estimate_rule': 'After human pilot: report median/p75 seconds by risk/source, remaining stratum counts, measured adjudication overhead; no fabricated estimate.'}})
    try:
        commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = 'unavailable_fixture_or_non_git_directory'
    manifest = {'schema_version': SCHEMA, 'proposal_status': 'protocol pending', 'source_repository_commit': commit,
                'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'sources': [{'path': p, 'sha256': source_hashes[p], 'bytes': (repo/p).stat().st_size,
                             'csv_records': len(tables[p]) if p in tables else None} for p in source_paths],
                'definitions': {'item_id': 'SHA256 of exact JSON [acronym,sentence,gold_expansion,candidates], first 24 hex; raw metadata differences preserved; not historical item_id',
                                'document': 'exact nonempty (source,page_title); normalized-key variants flagged only, not merged',
                                'natural': sorted(NATURAL), 'diagnostic_scope': {'contains_string_excluded_proposal': 'גימטר', 'exact_string_excluded_proposal': IDENTITY},
                                'span': 'zero-based Python Unicode code-point offsets, end-exclusive; length-preserving quote normalization; unique substring only, pending human boundary validation',
                                'independent_inventory': 'stored candidate/source exports independent of this per-item label field; not proof of semantic correctness or of upstream independence',
                                'exposure': 'aggregate structural label/candidate/document/span inspection; no semantic test adjudication and no system outputs or scores',
                                'extension_selection': 'availability-only; final type subset not approved; label coverage calculated only after selection',
                                'source_row_number': 'one-based CSV data record, header excluded'},
                'limits': ['No benchmark freeze, approved labels or inventory, new train, model invocation or network.',
                           'Literal string scope rule does not identify every nonliteral label or candidate.',
                           'Ben evidence preserved as source records; AI preparation is not second human annotation.',
                           'Document normalization is diagnostic; exact keys cannot establish real-world document identity.',
                           'Pair counts and matched updates must be recomputed after human inventory/span decisions.']}
    outputs = {'source_manifest.json': json.dumps(manifest, ensure_ascii=False, indent=2)+'\n',
               'diagnostic_summary.json': json.dumps(metrics, ensure_ascii=False, indent=2)+'\n',
               'item_audit.csv': csv_text(audits, list(audits[0]) if audits else ['stable_item_id']),
               'inventory_proposed.csv': csv_text(inventory, list(inventory[0]) if inventory else ['inventory_id']),
               'review_queue.csv': csv_text(queue, list(queue[0]) if queue else ['review_id']),
               'review_decisions.csv': csv_text(decisions, DECISION_FIELDS),
               'extension_feasibility.csv': csv_text(ext_rows, list(ext_rows[0]) if ext_rows else ['scenario']),
               'extension_membership_proposed.csv': csv_text(ext_members, list(ext_members[0]) if ext_members else ['scenario']),
               'pilot_items.csv': csv_text(pilot, list(pilot[0]) if pilot else ['review_id']),
               'pilot_rubric.md': RUBRIC,
               'unmatched_review_evidence.csv': csv_text(unmatched_reviews, ['path', 'record', 'reason', 'source_record_json'])}
    if any((output/name).is_symlink() for name in outputs):
        raise ValueError('Refusing output symlink; no outputs refreshed')
    output.mkdir(parents=True, exist_ok=True)
    for name, contents in outputs.items():
        target = output/name
        if target.is_symlink():
            raise ValueError(f'Refusing output symlink: {name}')
        # Existing decision file is byte-preserved if no new IDs are needed.
        if name == 'review_decisions.csv' and old_decisions and set(old_by_id)==ids:
            continue
        target.write_text(contents, encoding='utf-8')
    if source_hashes != {p: hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in source_paths}:
        raise RuntimeError('Source bytes changed during preparation')
    return metrics


RUBRIC = '''# P1 human data-review pilot

These are proposals, not approved model inputs. The human pilot has not been
performed. No elapsed times, human decisions, or agreement estimates are inferred.
Use `pilot_items.csv` for 16 deterministic dev/train or proposed natural-dev items;
it contains no historical test item. Use `item_audit.csv` and source references for
evidence, and record all decisions in `review_decisions.csv` using `review_id`.

For each item, start a timer and inspect the sentence before consulting its recorded
label. Record a defensible literal expansion or `uncertain`; uncertainty is retained.
Then inspect the recorded label and evidence. Enter label_decision (`accept`, `correct`,
`uncertain`, `exclude_proposed`) and corrected_label only when justified. Select the
intended target occurrence and enter zero-based, end-exclusive character offsets in
the unchanged raw sentence. Check attached prefixes, quote variants, repeated targets,
and an explanation in the sentence. A proposed span is not an annotated span.

Review the type inventory independently of this sentence: follow candidate/source
export references, Ben's recorded review and raw provenance. Do not add an answer
solely because it is the sentence label; do not invent a distractor or require two
expansions. Distinguish possible meanings from observed label strings. Review numeral
and concealed-identity exclusions as proposals. Alias equivalence needs an evidence
reference and must apply consistently to the type, not just one item. Record
inventory decisions on the corresponding `inv-...` review_id. Existing source values
and Ben's decisions are evidence and are never overwritten.

Complete reviewer, decision, reason, evidence_reference and decided_at, plus
elapsed_seconds and problem_types (pipe-separated label/target/inventory/provenance/
document/definition/other). A held/uncertain case remains open. Decisions and pilot
measurements must be entered by Shaked; the current blank cells are intentional.

After the pilot, summarize median and 75th-percentile seconds by risk/source where
sample sizes permit, issue categories and unresolved fraction. Estimate remaining
work from stratum counts times measured review times, adding measured adjudication
and inventory overhead. This risk-enriched pilot is not representative; report wide
ranges and do not invent precision for unobserved strata. Time limits do not replace
quality gates. Review proposed natural dev, critical flags and Wikipedia without
proven review, then a fixed sample of low-risk Knesset. Final sampling and closure
require Shaked. No inter-annotator agreement is available. The full queue is a searchable universe
of item and source-hypothesis proposals, not a requirement to review every row;
Shaked approves the required scope and unresolved cases remain open.

Reproduction: from the repository root, use the installed package:
`.venv/bin/python -m hebrew_acronyms.data_processing.prepare_study_review --repo .`.
Only `data/study_v1/review` is written; existing decision bytes are preserved on repeat
runs with identical source hashes. Changed sources or orphan decision IDs stop the
run for explicit reconciliation. CSV source references count records, not lines.

Local environment caveat observed during P1: the existing editable installation's
`.venv/lib/python3.12/site-packages/__editable__.hebrew_acronym_disambiguation-0.1.0.pth`
intermittently acquires the macOS `hidden` flag, causing `ModuleNotFoundError`.
If this recurs, inspect that exact file with `ls -lO`; when `hidden` is present,
clear only that flag with `chflags nohidden` on the same file immediately before
running the command. An offline editable reinstall used `PIP_NO_INDEX=1` and
`--no-deps --no-build-isolation`; no dependency or model downloads were performed.
The source of the recurring flag was not diagnosed. This local workaround is not
a verified fresh-environment installation or a change to the research protocol.

Extension scenarios and membership are inspectable availability proposals, never
active training sets. The cap uses at most five rows/type, at least two documents,
and document round-robin ordered by source, page title, historical ID and sentence
hash. No gold or score selects these rows. Coverage is described after selection.
Source/row/candidate-pair matching is necessary, not sufficient: after inventory and
span review, recompute both full pair totals and use the same batch, epochs,
accumulation, drop-last, seed and checkpoint policy. No schedule is executed here.
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.repo)
    # Aggregate-only console output; never print research sentences.
    print(json.dumps({k: result[k] for k in ('item_audit_rows', 'inventory_proposals', 'review_queue_rows', 'natural_dev_proposed')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
