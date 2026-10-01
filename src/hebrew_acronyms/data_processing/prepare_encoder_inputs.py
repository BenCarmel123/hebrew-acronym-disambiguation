"""Qualify saved train/natural-dev exports, without models, services or test content.

Only the audit's role/source metadata is inspected before the access gate. Test
rows contribute opaque type/document IDs only. Historical split files remain
immutable; source record numbers are one-based CSV data records, not lines.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import unicodedata

from hebrew_acronyms.models.common.pairs import (
    build_pairs, candidates_for, explicit_span, validate_ids,
)

VERSION = 'encoder-inputs-v1'
CONTENT_FIELDS = ('acronym', 'sentence', 'gold_expansion', 'candidates')
QUOTES = str.maketrans({'״': '"', '“': '"', '”': '"', '׳': "'", '‘': "'", '’': "'"})
CONSTRUCTION = {'wiki_natural': 'natural', 'knesset': 'natural',
                'wiki_substituted': 'substituted', 'wiki_deglossed': 'deglossed',
                'manual': 'authored'}
INPUT_FIELDS = ['item_id', 'acronym', 'sentence', 'target_raw', 'span_start', 'span_end',
                'candidates', 'gold_expansion', 'split', 'source_file', 'source_record',
                'source_row_sha256', 'audit_item_id', 'audit_record', 'construction',
                'source', 'page_title', 'label_origin', 'label_status', 'label_evidence',
                'doc_id', 'type_id', 'decision_id', 'span_basis']
TRACE_FIELDS = INPUT_FIELDS + ['status', 'reasons', 'historical_roles', 'raw_item_id',
    'original_gold_expansion', 'original_candidates', 'source_refs_json',
    'source_metadata_json', 'review_evidence_json', 'applied_decision_json',
    'audit_flags', 'duplicate_group']


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError('Missing or duplicate CSV header')
        for number, row in enumerate(reader, 1):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f'Malformed CSV data record {number}')
            yield number, row


def audit_id(row):
    return 'item-' + digest(canonical([row.get(k, '') for k in CONTENT_FIELDS]))[:24]


def type_id(acronym):
    # Exact historical type only: no quote/prefix/alias merging.
    return 'type-' + digest(acronym)


def document_id(row):
    source, title = row.get('source', ''), row.get('page_title', '')
    return ('doc-' + digest(canonical([source, title]))[:24]
            if source.strip() and title.strip() else '')


def role_gate(row):
    """Return role and parsed refs; never inspect sentence/labels/candidates here."""
    roles = set(re.split(r'[+|,;\s]+', row.get('historical_roles', '').strip()))
    try:
        refs = json.loads(row.get('source_refs_json', ''))
        if not isinstance(refs, list) or not refs or any(
            not isinstance(ref, dict) or not isinstance(ref.get('path'), str)
            or type(ref.get('record')) is not int or ref['record'] < 1 for ref in refs
        ):
            return 'unknown', []
    except (ValueError, TypeError):
        return 'unknown', []
    if any('test' in role.lower() for role in roles) or any(
        'test' in ref['path'].lower() for ref in refs
    ):
        return 'test', refs
    paths = {ref['path'] for ref in refs}
    known = {'data/splits/all_items.csv', 'data/splits/train_items.csv', 'data/splits/dev_items.csv'}
    if not paths <= known:
        return 'unknown', refs
    if roles == {'train'} and 'data/splits/train_items.csv' in paths and 'data/splits/dev_items.csv' not in paths:
        return 'train', refs
    if roles == {'dev'} and 'data/splits/dev_items.csv' in paths and 'data/splits/train_items.csv' not in paths:
        return 'historical_dev', refs
    if (roles == {'aggregate_only'} and row.get('natural_dev_proposed') == 'true'
            and paths == {'data/splits/all_items.csv'}):
        return 'dev', refs
    if roles == {'aggregate_only'} and row.get('natural_dev_proposed') == 'false' and paths == {'data/splits/all_items.csv'}:
        return 'out_of_scope', refs
    return 'unknown', refs


def read_scoped_audit(path):
    """Project reserved metadata immediately; unsafe payload never leaves this gate."""
    safe, reserved_types, reserved_docs, access = [], set(), set(), Counter()
    for number, row in read_rows(path):
        role, refs = role_gate(row)
        access[role] += 1
        if role == 'test':
            # These two metadata fields alone are permitted for overlap checking.
            if not row.get('raw_acronym') or not row.get('doc_id'):
                # Missing docs are counted, never replaced using test content.
                access['reserved_missing_type_or_document'] += 1
            if row.get('raw_acronym'):
                reserved_types.add(type_id(row['raw_acronym']))
            if row.get('doc_id'):
                reserved_docs.add(row['doc_id'])
            continue
        if role in {'unknown', 'out_of_scope'}:
            continue
        safe.append((number, role, refs, row))
    return safe, reserved_types, reserved_docs, dict(sorted(access.items()))


def read_scoped_decisions(path, allowed_ids):
    result = {}
    for number, row in read_rows(path):
        if row.get('review_id') not in allowed_ids:
            continue  # Never inspect or return non-allowed decision payload.
        key = row['review_id']
        if key in result:
            raise ValueError('Duplicate scoped review decision ID')
        result[key] = {'source_record': number, 'values': row}
    return result


def word_character(char):
    return char == '_' or unicodedata.category(char)[0] in {'L', 'N', 'M'}


def locate_target(sentence, acronym):
    """Quote folding is length-preserving; embedded/attached boundaries are held.

    We do not infer whether adjacent letters are proclitics or part of a type.
    Only a saved human span can resolve an attached boundary or repeated target.
    """
    if not acronym:
        return None, 'target_missing'
    folded, needle = sentence.translate(QUOTES), acronym.translate(QUOTES)
    matches = [(m.start(), m.start() + len(needle))
               for m in re.finditer('(?=' + re.escape(needle) + ')', folded)]
    if not matches:
        return None, 'target_missing'
    if len(matches) != 1:
        return None, 'target_multiple'
    start, end = matches[0]
    if ((start > 1 and folded[start - 1] in {'\"', "'"} and word_character(sentence[start - 2]))
            or (end + 1 < len(sentence) and folded[end] in {'\"', "'"} and word_character(sentence[end + 1]))
            or (start and word_character(sentence[start - 1]))
            or (end < len(sentence) and word_character(sentence[end]))):
        return None, 'target_boundary_ambiguous'
    return (start, end), 'unique_bounded_quote_folded_match'


def exact_review_evidence(audit, source):
    """Reuse only sentence-bearing human evidence matching content and document."""
    linked, adverse = [], False
    for evidence in json.loads(audit.get('review_evidence_json', '[]')):
        record = evidence.get('source_record', {})
        if evidence.get('path') != 'data/mined/knesset/knesset_reviewed.csv':
            continue
        if not all(record.get(k) == source.get(k) for k in
                   ('item_id', 'acronym', 'sentence', 'gold_expansion', 'candidates', 'source', 'page_title')):
            continue
        verdict = evidence.get('verdict', '')
        if verdict in {'clean', 'human_review'}:
            linked.append(evidence)
        elif verdict in {'unsure', 'uncertain', 'wrong_sense', 'broken', 'reject'}:
            adverse = True
    return linked, adverse


def qualify(source, audit, decision, local_decision=None):
    reasons, output = [], dict(source)
    output['gold_expansion'] = source.get('gold_expansion', '')
    output['construction'] = CONSTRUCTION.get(source.get('category'), 'unknown')
    output['label_evidence'] = 'stored_training_label_not_independently_verified'
    output['decision_id'], output['span_basis'] = '', ''
    matched, adverse = exact_review_evidence(audit, source)
    if matched:
        output['label_evidence'] = 'historical_human_review_attributed_to_Ben'
    if source.get('review_verdict') in {'unsure', 'uncertain', 'wrong_sense', 'broken', 'reject'} or adverse:
        reasons.append('adverse_or_uncertain_historical_review')
    span = None
    d = decision.get('values', {})
    active = any(d.get(k) for k in ('decision', 'label_decision', 'target_decision'))
    if active:
        # Unknown reviewers/AI are not promoted into human authority.
        if (d.get('reviewer') != 'שקד' or not d.get('decided_at')
                or not d.get('reason') or not d.get('evidence_reference')):
            reasons.append('unverified_decision_authority')
        else:
            output['decision_id'] = d['review_id']
            overall = d.get('decision')
            label = d.get('label_decision')
            if overall == 'exclude_proposed':
                reasons.append('human_exclusion_recorded')
            if overall not in {'accept', 'correct', 'exclude_proposed', 'uncertain'}:
                reasons.append('invalid_human_decision_status')
            if ((overall == 'accept' and label != 'accept')
                    or (overall == 'exclude_proposed' and label != 'exclude_proposed')
                    or (overall == 'uncertain' and label != 'uncertain')):
                reasons.append('inconsistent_human_decision_status')
            if label == 'correct' and d.get('corrected_label'):
                output['gold_expansion'] = d['corrected_label']
                output['label_evidence'] = 'Shaked_review_with_AI_assistance'
            elif label == 'accept':
                output['label_evidence'] = 'Shaked_review_with_AI_assistance'
            elif label == 'exclude_proposed':
                reasons.append('human_exclusion_recorded')
            else:
                reasons.append('human_label_unresolved')
            if d.get('decision') == 'uncertain':
                reasons.append('human_label_unresolved')
            target = d.get('target_decision')
            if target in {'accept', 'correct'}:
                try:
                    start, end = int(d['approved_span_start']), int(d['approved_span_end'])
                    raw = source['sentence'][start:end]
                    explicit_span({'sentence': source['sentence'], 'target_raw': raw,
                                   'span_start': start, 'span_end': end})
                    if source['acronym'].translate(QUOTES) not in raw.translate(QUOTES):
                        raise ValueError('Approved target does not contain source type')
                    span = (start, end)
                    output['span_basis'] = 'saved_human_span'
                except (ValueError, KeyError):
                    reasons.append('invalid_saved_human_span')
            else:
                reasons.append('human_target_unresolved')
    if not active:
        span, basis = locate_target(source.get('sentence', ''), source.get('acronym', ''))
        if span is None:
            reasons.append(basis)
        else:
            output['span_basis'] = basis
    local_decision = local_decision or {}
    if local_decision:
        if (local_decision.get('audit_item_id') != audit['stable_item_id']
                or local_decision.get('source_row_sha256') != digest(canonical(source))):
            raise ValueError('Local decision is stale or linked to different content')
        human = local_decision.get('human_decision', {})
        approvals = [human, *local_decision.get('additional_human_approvals', [])]
        if any(a.get('status') != 'approved' for a in approvals):
            reasons.append('AI_proposal_pending_human_decision')
        elif any(a.get('reviewer') != 'Shaked' or not a.get('response') for a in approvals):
            reasons.append('unverified_local_decision_authority')
        else:
            proposal = local_decision['proposal']
            replacement = proposal.get('candidate_replacement')
            if replacement:
                old, new = replacement['old'], replacement['new']
                entries = source['candidates'].split('|')
                if sum(entry.strip() == old for entry in entries) != 1:
                    raise ValueError('Approved candidate replacement does not match source inventory')
                output['candidates'] = ' | '.join(new if entry.strip() == old else entry.strip() for entry in entries)
            if 'gold_expansion' in proposal:
                output['gold_expansion'] = proposal['gold_expansion']
                reasons = [r for r in reasons if r != 'human_label_unresolved']
            if 'span' in proposal:
                start, end = proposal['span']
                explicit_span({'sentence': source['sentence'], 'target_raw': source['sentence'][start:end],
                               'span_start': start, 'span_end': end})
                if source['acronym'].translate(QUOTES) not in source['sentence'][start:end].translate(QUOTES):
                    raise ValueError('Approved local target does not contain source type')
                span = (start, end)
                reasons = [r for r in reasons if r not in {'target_multiple', 'target_boundary_ambiguous', 'human_target_unresolved'}]
                output['span_basis'] = 'local_human_approved_AI_span_proposal'
            output['decision_id'] = local_decision['decision_id']
            if 'gold_expansion' in proposal:
                output['label_evidence'] = 'Shaked_review_with_AI_assistance'
    if span:
        output['span_start'], output['span_end'] = span
        output['target_raw'] = source['sentence'][span[0]:span[1]]
    else:
        output.update(target_raw='', span_start='', span_end='')
    if span:
        try:
            explicit_span(output)
        except ValueError:
            reasons.append('invalid_encoder_span')
    output['doc_id'] = document_id(source)
    output['type_id'] = type_id(source.get('acronym', ''))
    if not output['doc_id']:
        reasons.append('missing_document_key')
    if output['construction'] == 'unknown':
        reasons.append('unknown_text_construction')
    if not source.get('source') or not source.get('label_origin'):
        reasons.append('missing_text_or_label_provenance')
    try:
        candidates = candidates_for(output)
        if len(set(candidates)) != len(candidates):
            reasons.append('duplicate_candidates')
        if sum(c == output['gold_expansion'].strip() for c in candidates) != 1:
            reasons.append('gold_not_exactly_one_candidate')
        if len(candidates) < 2:
            reasons.append('singleton_not_pair_loss')
        if source.get('n_candidates') != str(len(candidates)):
            reasons.append('candidate_count_mismatch')
    except ValueError:
        reasons.append('invalid_candidates')
    if not output['gold_expansion'].strip():
        reasons.append('missing_label')
    return output, sorted(set(reasons)), matched


def csv_bytes(rows, fields):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n', extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def apply_dr_bdr_separation(rows, decisions):
    """Apply the approved exact pair only; never infer or strip Hebrew prefixes."""
    applied = {}
    for decision in decisions:
        if ((decision.get('train_type'), decision.get('dev_type'), decision.get('action'))
                != ('ד״ר', 'בד״ר', 'hold') or decision['decision_id'] in applied):
            raise ValueError('Only the explicit dr/bdr separation decision is supported')
        human = decision.get('human_decision', {})
        if (human.get('status') != 'approved' or human.get('reviewer') != 'Shaked'
                or not human.get('response')):
            raise ValueError('Split separation requires saved human approval')
        held_ids = []
        if any(r['split'] == 'dev' and r['acronym'] == 'בד״ר' for r in rows):
            for row in rows:
                if row['split'] == 'train' and row['acronym'] == 'ד״ר' and not row['reasons']:
                    row['reasons'].append('approved_dr_bdr_split_overlap')
                    evidence = json.loads(row['applied_decision_json'])
                    evidence['split_separation'] = decision
                    row['applied_decision_json'] = canonical(evidence)
                    held_ids.append(row['item_id'])
        applied[decision['decision_id']] = sorted(held_ids)
    return applied


def prepare_inputs(*, train_path, historical_dev_path, audit_path, decisions_path,
                   policy_path, output_dir):
    """Explicit saved inputs -> deterministic derivatives. No upstream file traversal."""
    paths = {'train_export': Path(train_path), 'historical_dev_export': Path(historical_dev_path),
             'scoped_audit_source': Path(audit_path), 'human_decisions': Path(decisions_path),
             'local_policy': Path(policy_path)}
    if any(p.name in {'test_items.csv', 'all_items.csv'} or p.resolve().name in {'test_items.csv', 'all_items.csv'} for p in paths.values()):
        raise ValueError('Direct test/aggregate source access is forbidden; use the scoped saved audit')
    output_dir = Path(output_dir)
    if output_dir.is_symlink():
        raise ValueError('Symlink output directory is not allowed')
    for name in ('train.csv', 'dev.csv', 'dev_singletons.csv', 'trace.csv', 'manifest.json'):
        if (output_dir / name).is_symlink():
            raise ValueError('Symlink output file is not allowed')
    # Never allow an output to overwrite an input or a synced/source directory.
    if any(output_dir.resolve() == p.resolve().parent for k, p in paths.items() if k != 'local_policy'):
        raise ValueError('Output directory must be separate from historical inputs')
    policy = json.loads(paths['local_policy'].read_text(encoding='utf-8'))
    if policy.get('rules_version') != VERSION:
        raise ValueError('Local policy does not match preparation rules')
    source_hashes = {k: file_hash(p) for k, p in paths.items()}
    scoped, reserved_types, reserved_docs, access = read_scoped_audit(audit_path)
    if access.get('unknown', 0):
        raise ValueError('Audit rows lack unambiguous role/reference metadata; reserved blockers may be incomplete')
    selected_ids = {r['stable_item_id'] for _, _, _, r in scoped}
    if len(selected_ids) != len(scoped):
        raise ValueError('Duplicate scoped audit identity')
    decisions = read_scoped_decisions(decisions_path, selected_ids)
    tables = {'train': dict(read_rows(train_path)), 'historical_dev': dict(read_rows(historical_dev_path))}
    coverage = defaultdict(set)
    local_decisions = {}
    for entry in policy.get('item_decisions', []):
        key = entry['audit_item_id']
        if key not in selected_ids or key in local_decisions:
            raise ValueError('Local decision is duplicated or outside authorized scope')
        local_decisions[key] = entry
    traces = []
    for audit_record, role, refs, audit in scoped:
        raw = {k[4:]: v for k, v in audit.items() if k.startswith('raw_')}
        if audit_id(raw) != audit['stable_item_id']:
            raise ValueError('Audit content identity mismatch in allowed row')
        if role in tables:
            source_file = 'data/splits/' + ('train_items.csv' if role == 'train' else 'dev_items.csv')
            matching = [r for r in refs if r['path'] == source_file]
            # A content identity can represent multiple source rows; retain each reference.
            for ref in matching:
                if ref['record'] in coverage[role]:
                    raise ValueError('Source record assigned more than once in audit')
                coverage[role].add(ref['record'])
            sources = [(ref['record'], tables[role].get(ref['record'])) for ref in matching]
            if not sources or any(s is None for _, s in sources):
                raise ValueError('Audit source record is missing')
        else:
            source_file = 'data/study_v1/review/item_audit.csv'
            sources = [(audit_record, raw)]
        for source_record, source in sources:
            if audit_id(source) != audit['stable_item_id']:
                raise ValueError('Source content differs from audit; never join on historical ID alone')
            decision = decisions.get(audit['stable_item_id'], {})
            local = local_decisions.get(audit['stable_item_id'], {})
            output, reasons, matched = qualify(source, audit, decision, local)
            if source != raw:
                # Metadata differences are material to evidence linkage; do not guess precedence.
                reasons.append('audit_source_metadata_mismatch')
            if output['doc_id'] != audit.get('doc_id', ''):
                reasons.append('audit_document_mismatch')
            output.update(item_id='enc-' + digest(canonical([source_file, source_record, source]))[:32],
                split='train' if role == 'train' else 'dev', source_file=source_file,
                source_record=source_record, source_row_sha256=digest(canonical(source)),
                audit_item_id=audit['stable_item_id'], audit_record=audit_record)
            if role == 'historical_dev':
                reasons.append('historical_dev_reference_only')
            elif role == 'dev':
                if output['construction'] != 'natural':
                    reasons.append('dev_not_natural')
                if output['label_evidence'] not in {'historical_human_review_attributed_to_Ben', 'Shaked_review_with_AI_assistance'}:
                    reasons.append('dev_without_matching_human_label')
            traces.append({**output, 'status': '', 'reasons': sorted(set(reasons)),
                'historical_roles': audit['historical_roles'], 'raw_item_id': source.get('item_id', ''),
                'original_gold_expansion': source.get('gold_expansion', ''),
                'original_candidates': source.get('candidates', ''),
                'source_refs_json': canonical(refs), 'source_metadata_json': canonical({k: v for k, v in source.items() if k not in CONTENT_FIELDS}),
                'review_evidence_json': canonical([{k: v for k, v in e.items() if k != 'source_record'} | {'source_record_sha256': digest(canonical(e['source_record']))} for e in matched]), 'applied_decision_json': canonical({'saved_review': decision, 'local': local}),
                'audit_flags': audit.get('flags', ''), 'duplicate_group': ''})
    # Missing audit membership is unsafe, not permission to infer a split from a filename.
    for role, table in tables.items():
        if coverage[role] != set(table):
            raise ValueError('Train/dev source rows lack unambiguous authorized audit membership')
    # Reserve all proposed natural dev + historical dev, even if later held/excluded.
    dev_rows = [r for r in traces if r['split'] == 'dev']
    dev_types = {r['type_id'] for r in dev_rows}
    dev_docs = {r['doc_id'] for r in dev_rows if r['doc_id']}
    dev_sentences = {' '.join(r['sentence'].translate(QUOTES).split()) for r in dev_rows}
    groups = defaultdict(list)
    for row in traces:
        if row['type_id'] in reserved_types:
            row['reasons'].append('reserved_type_overlap')
        if row['doc_id'] and row['doc_id'] in reserved_docs:
            row['reasons'].append('reserved_document_overlap')
        if row['split'] == 'train':
            if row['type_id'] in dev_types:
                row['reasons'].append('dev_type_overlap')
            if row['doc_id'] and row['doc_id'] in dev_docs:
                row['reasons'].append('dev_document_overlap')
            if ' '.join(row['sentence'].translate(QUOTES).split()) in dev_sentences:
                row['reasons'].append('dev_text_overlap')
        # Conservative duplicate grouping within each split; never select a preferred label.
        key = canonical([row['split'], ' '.join(row['sentence'].translate(QUOTES).split())])
        groups[key].append(row)
    for key, rows in groups.items():
        if len(rows) > 1:
            for row in rows:
                row['duplicate_group'] = 'dup-' + digest(key)[:24]
                row['reasons'].append('duplicate_text_group_held')
    split_holds = apply_dr_bdr_separation(traces, policy.get('split_decisions', []))
    train, dev, singleton = [], [], []
    for row in traces:
        reasons = sorted(set(row['reasons']))
        if reasons == ['singleton_not_pair_loss'] and row['split'] == 'dev':
            row['status'] = 'dev_singleton_prediction_only'
            singleton.append(row)
        elif reasons:
            row['status'] = ('reference_only' if 'historical_dev_reference_only' in reasons
                             else 'excluded' if 'human_exclusion_recorded' in reasons else 'held')
        else:
            row['status'] = 'included_pair_loss'
            (train if row['split'] == 'train' else dev).append(row)
        row['reasons'] = '|'.join(reasons)
    for rows in (traces, train, dev, singleton):
        validate_ids(rows)
    pair_counts = {}
    for name, rows in (('train', train), ('dev', dev)):
        pair_counts[name] = len(build_pairs(rows))
    for row in singleton:
        explicit_span(row)
    traces.sort(key=lambda r: (r['source_file'], r['source_record']))
    for rows in (train, dev, singleton):
        rows.sort(key=lambda r: r['item_id'])
    outputs = {'train.csv': csv_bytes(train, INPUT_FIELDS), 'dev.csv': csv_bytes(dev, INPUT_FIELDS),
               'dev_singletons.csv': csv_bytes(singleton, INPUT_FIELDS),
               'trace.csv': csv_bytes(traces, TRACE_FIELDS)}
    def counts(rows):
        return {'rows': len(rows), 'types': len({r['type_id'] for r in rows}),
                'documents': len({r['doc_id'] for r in rows if r['doc_id']}),
                'construction': dict(sorted(Counter(r['construction'] for r in rows).items())),
                'source': dict(sorted(Counter(r['source'] for r in rows).items())),
                'label_origin': dict(sorted(Counter(r['label_origin'] for r in rows).items())),
                'label_evidence': dict(sorted(Counter(r['label_evidence'] for r in rows).items()))}
    populations = {'train_export': [r for r in traces if r['split'] == 'train'],
                   'natural_dev_proposed': [r for r in traces if r['source_file'].endswith('item_audit.csv')],
                   'historical_dev_reference': [r for r in traces if r['source_file'].endswith('dev_items.csv')]}
    qualification_flow = {}
    for name, rows in populations.items():
        qualification_flow[name] = {'input_rows': len(rows),
            'status': dict(sorted(Counter(r['status'] for r in rows).items())),
            'nonexclusive_reasons': dict(sorted(Counter(reason for r in rows for reason in r['reasons'].split('|') if reason).items())),
            'source_construction_status': [dict(zip(('source', 'construction', 'status', 'rows'), (*key, count)))
                for key, count in sorted(Counter((r['source'], r['construction'], r['status']) for r in rows).items())]}
    dev_items_by_type = Counter(r['acronym'] for r in dev)
    dev_labels_by_type = defaultdict(set)
    for row in dev:
        dev_labels_by_type[row['acronym']].add(row['gold_expansion'])
    pending_local = sum(any(a.get('status') != 'approved'
        for a in [d['human_decision'], *d.get('additional_human_approvals', [])])
        for d in policy.get('item_decisions', []))
    module_path = Path(__file__)
    from hebrew_acronyms.models.common import pairs
    manifest = {'schema_version': VERSION, 'reference_commit': policy['reference_commit'],
        'code': {'module': 'src/hebrew_acronyms/data_processing/prepare_encoder_inputs.py',
                 'sha256': file_hash(module_path), 'contract_sha256': file_hash(pairs.__file__)},
        'sources': {k: {'path': policy['source_paths'][k], 'sha256': source_hashes[k]} for k in paths},
        'rules': policy['rules'], 'authority': policy['authority'], 'access_gate_counts': access,
        'reserved_metadata': {'type_count': len(reserved_types), 'document_count': len(reserved_docs),
                              'type_ids_sha256': digest(canonical(sorted(reserved_types))),
                              'document_ids_sha256': digest(canonical(sorted(reserved_docs)))},
        'qualification_flow': qualification_flow,
        'split_separation': {'decisions': policy.get('split_decisions', []), 'held_item_ids': split_holds},
        'dev_distribution': {
            'items_by_type': dict(sorted(dev_items_by_type.items())),
            'observed_gold_counts_by_type': {k: len(v) for k, v in sorted(dev_labels_by_type.items())},
            'types_with_one_item': sum(n == 1 for n in dev_items_by_type.values()),
            'types_with_one_observed_gold': sum(len(v) == 1 for v in dev_labels_by_type.values())},
        'counts': {'train': counts(train), 'dev': counts(dev), 'dev_singletons': counts(singleton),
                   'all_scoped_rows': len(traces), 'pair_counts': pair_counts,
                   'status_by_split': {s: dict(sorted(Counter(r['status'] for r in traces if r['split'] == s).items())) for s in ('train', 'dev')},
                   'reasons_by_split': {s: dict(sorted(Counter(reason for r in traces if r['split'] == s for reason in r['reasons'].split('|') if reason).items())) for s in ('train', 'dev')}},
        'applied_human_decision_ids': sorted({r['decision_id'] for r in traces if r['decision_id']}
            | {key for key, d in decisions.items() if d['values'].get('decision') and d['values'].get('reviewer') == 'שקד'}),
        'new_AI_semantic_judgments': [{'decision_id': d['decision_id'], 'audit_item_id': d['audit_item_id'], 'human_status': d['human_decision']['status']} for d in policy.get('item_decisions', [])],
        'readiness': {'train_pair_contract': bool(train), 'dev_pair_contract': bool(dev),
                      'pending_local_decisions': pending_local,
                      'manager_acceptance_pending': True, 'training_performed': False},
        'limits': policy['limits'],
        'outputs': {name: {'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)} for name, payload in outputs.items()}}
    outputs['manifest.json'] = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()
    if any(file_hash(p) != source_hashes[k] for k, p in paths.items()):
        raise ValueError('Source changed during preparation; no outputs written')
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in outputs.items():
        target = output_dir / name
        if target.resolve() in {p.resolve() for p in paths.values()}:
            raise ValueError('Output would overwrite an input')
        target.write_bytes(payload)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('train', 'historical-dev', 'audit', 'decisions', 'policy', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = prepare_inputs(train_path=args.train, historical_dev_path=args.historical_dev,
        audit_path=args.audit, decisions_path=args.decisions, policy_path=args.policy,
        output_dir=args.output)
    print(json.dumps(result['counts'], ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
