"""Derive explicit test spans without changing any original input field.

Only sentence/acronym determine target locations. The default holds ambiguous
matches for human review. The separately selected legacy policy uses the first
quote-normalized occurrence; it is not evidence of human target validation.
"""
import csv
import hashlib
import json
from pathlib import Path
import re

from hebrew_acronyms.data_processing.prepare_encoder_inputs import locate_target, QUOTES
from hebrew_acronyms.models.common.pairs import explicit_span, find_span, validate_ids
from hebrew_acronyms.test_cohort import check_scored_cohort


def span_candidates(sentence, acronym):
    choices = []
    for m in re.finditer('(?=' + re.escape(acronym.translate(QUOTES)) + ')', sentence.translate(QUOTES)):
        start, end = m.start(), m.start() + len(acronym)
        # A proposal, never an automatic decision: include directly attached Hebrew letters.
        while start > 0 and '\u05d0' <= sentence[start - 1] <= '\u05ea':
            start -= 1
        choices.append({'span_start': start, 'span_end': end, 'target_raw': sentence[start:end]})
    return choices


def text_identity(row):
    return hashlib.sha256(json.dumps({k: row[k] for k in ('item_id', 'sentence', 'acronym')},
                         ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def qualify_test(source, decisions=None, *, policy="qualified"):
    """Return original rows with policy-identified spans and unresolved exceptions."""
    if policy not in {"qualified", "legacy_first_occurrence"}:
        raise ValueError("Unknown target-location policy")
    if policy == "legacy_first_occurrence" and decisions:
        raise ValueError("Do not mix human target decisions with the legacy policy")
    with Path(source).open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    validate_ids(rows)
    check_scored_cohort([r['item_id'] for r in rows])
    decisions = decisions or {}
    if set(decisions) - {r['item_id'] for r in rows}:
        raise ValueError('Unexpected span decision ID')
    output, exceptions = [], []
    for row in rows:
        span, basis = locate_target(row['sentence'], row['acronym'])
        if policy == 'legacy_first_occurrence':
            span = find_span(row['sentence'], row['acronym'])
            basis = 'legacy_first_occurrence_not_human_validated'
        entry = dict(row)
        if span is None:
            decision = decisions.get(row['item_id'])
            if decision:
                if (decision.get('text_sha256') != text_identity(row) or not decision.get('reviewer')
                        or not decision.get('decided_at')):
                    raise ValueError('Human decision must bind reviewer, date and original text')
                entry.update({k: decision[k] for k in ('span_start', 'span_end', 'target_raw')})
                span = explicit_span(entry)
                basis = 'explicit_human_decision'
            else:
                exceptions.append({k: row[k] for k in ('item_id', 'sentence', 'acronym')} |
                                  {'text_sha256': text_identity(row), 'reason': basis,
                                   'choices': span_candidates(row['sentence'], row['acronym'])})
        if span is not None:
            entry.update(span_start=span[0], span_end=span[1], target_raw=row['sentence'][span[0]:span[1]])
            explicit_span(entry)
        entry['span_basis'] = basis
        assert all(entry[k] == v for k, v in row.items())
        output.append(entry)
    return output, exceptions


def write_qualified(source, output, decisions=None, *, policy="qualified"):
    rows, unresolved = qualify_test(source, decisions, policy=policy)
    if unresolved:
        raise ValueError(f'{len(unresolved)} test spans still require human decisions')
    fields = list(rows[0])
    with Path(output).open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows
