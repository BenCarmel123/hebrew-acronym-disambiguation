"""Build a local, immutable-source review dataset; never judge answer semantics.

Run with explicit --root and --output paths. The export is deterministic for the
same source files, seed and sample size. It performs no model or network calls.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

EXPORT_VERSION = 'human-review-data-v2'
SAMPLING_VERSION = 'human-review-sampling-v2'
SEED = 'human-review-20261009-v1'
SYSTEMS = [
    ('dictabert', 'מערכת א', 'DictaBERT ללא אימון משימתי', 'selection', 'candidate', 'results/dictabert/test_details.csv'),
    ('dictabertx', 'מערכת ב', 'DictaBERT מאומן', 'selection', 'candidate', 'results/dictabertx/test_details.csv'),
    ('qwen_generate', 'מערכת ג', 'Qwen — יצירה', 'generation', 'text', 'results/qwen/generate_details.csv'),
    ('qwen_select', 'מערכת ד', 'Qwen — בחירה', 'selection', 'letter', 'results/qwen/select_details.csv'),
    ('gemini_generate', 'מערכת ה', 'Gemini — יצירה', 'generation', 'text', 'results/gemini/test_generate_details.csv'),
    ('gemini_select', 'מערכת ו', 'Gemini — בחירה', 'selection', 'letter', 'results/gemini/test_select_details.csv'),
]
QUOTE_MAP = str.maketrans({'״': '"', '“': '"', '”': '"', '׳': "'", '‘': "'", '’': "'"})


def normalise_quotes(value: str) -> str:
    """Mirror the source scorer's length-preserving quotation folding only."""
    return value.translate(QUOTE_MAP)


def split_candidates(value: str) -> list[str]:
    return [part.strip() for part in value.split('|') if part.strip()]


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def index_rows(rows: list[dict], description: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        item_id = row.get('item_id', '')
        if not item_id.strip() or item_id in result:
            raise ValueError(f'{description}: empty or duplicate item_id {item_id!r}')
        result[item_id] = row
    return result


def csv_bool(value: str, description: str) -> bool:
    if value not in ('True', 'False'):
        raise ValueError(f'{description}: expected True/False, got {value!r}')
    return value == 'True'


def decode_letter(raw: str, shown: list[str]) -> str | None:
    answer = raw.strip()
    if len(shown) >= 1 and len(answer) == 1 and 'A' <= answer <= chr(64 + min(26, len(shown))):
        return shown[ord(answer) - 65]
    return None


def mechanical_spans(sentence: str, acronym: str) -> list[dict]:
    """All quote-normalized literal matches, NOT certified target annotations."""
    if not acronym:
        return []
    return [{'start': match.start(), 'end': match.end()} for match in
            re.finditer(re.escape(normalise_quotes(acronym)), normalise_quotes(sentence))]


def make_answer(item: dict, system: tuple, row: dict | None) -> dict:
    system_id, _, _, task, response_type, path = system
    base = {'id': f"{item['item_id']}:{system_id}", 'system_id': system_id,
            'task': task, 'response_type': response_type, 'source_file': path}
    if row is None:
        return dict(base, raw=None, decoded=None, option_mapping=[], auto_score=None,
                    auto_valid=None, missing=True, status='missing_record',
                    auto_score_rule=None, mechanical_flags=['missing_record'])
    for detail_key, input_key in [('acronym', 'acronym'), ('gold', 'gold_expansion'), ('candidates', 'candidates')]:
        if row[detail_key] != item[input_key]:
            raise ValueError(f'{path}: {item["item_id"]}: {detail_key} differs from test input')
    raw = row['response']
    shown = split_candidates(row['shown_order'])
    candidates = split_candidates(item['candidates'])
    if response_type in ('letter', 'candidate') and Counter(shown) != Counter(candidates):
        raise ValueError(f'{path}: {item["item_id"]}: shown_order is not a candidate permutation')
    if response_type == 'text' and shown:
        raise ValueError(f'{path}: unexpected candidate exposure in generation record')
    score = csv_bool(row['correct'], path)
    valid = csv_bool(row['valid'], path)
    flags = []
    if response_type == 'letter':
        decoded = decode_letter(raw, shown)
        recomputed_valid = decoded is not None and len(shown) <= 26
        recomputed = recomputed_valid and decoded == item['gold_expansion'].strip()
        if len(shown) > 26:
            flags.append('over_26_candidates')
        rule = 'source parser rejects candidate lists >26; otherwise strict uppercase single letter and exact candidate equality'
        if decoded is None:
            flags.append('letter_format_invalid')
    elif response_type == 'text':
        decoded = raw or None
        recomputed = normalise_quotes(item['gold_expansion'].strip()) in normalise_quotes(raw)
        recomputed_valid = any(normalise_quotes(c) in normalise_quotes(raw) for c in candidates)
        rule = 'gold substring in raw response after quote normalization (not semantic or exact match)'
        if recomputed and normalise_quotes(item['gold_expansion'].strip()) != normalise_quotes(raw):
            flags.append('substring_not_exact')
        if len(raw) > 60:
            flags.append('long_answer')
    else:
        decoded = raw or None
        recomputed = raw == item['gold_expansion'].strip()
        recomputed_valid = raw in candidates
        rule = 'selected candidate text equals gold; preserved original CSV score'
    if score != recomputed or valid != recomputed_valid:
        flags.append('stored_score_recomputation_mismatch')
    if not raw:
        flags.append('empty_answer')
    return dict(base, raw=raw, decoded=decoded,
                option_mapping=[{'letter': chr(65 + i), 'candidate': c} for i, c in enumerate(shown[:26])] if response_type == 'letter' else [],
                omitted_candidates=shown[26:] if response_type == 'letter' else [],
                mapping_note=('מיפוי לפי shown_order השמור. קוד ההנחיה הנוכחי מציג רק A–Z; מועמדים מעבר ל־26 אינם מוצגים, והמפענח המקורי פוסל רשימות מעל 26. הטקסט המלא שנשלח אינו שמור ב־CSV.' if response_type == 'letter' else None),
                shown_order=shown, auto_score=score, auto_valid=valid,
                auto_score_rule=rule, missing=not bool(raw), status='recorded' if raw else 'empty_answer',
                mechanical_flags=flags,
                mechanical_recomputed_score=recomputed, mechanical_recomputed_valid=recomputed_valid)


def select_sample(items: list[dict], size: int, seed: str) -> list[str]:
    """Balanced source × orthographic-length proxy, outcome-independent hash rank.

    Round-robin strata, prioritizing unrepresented acronym types within each
    stratum. This deliberately disproportionate coverage sample is not a simple
    random benchmark sample; an unweighted overall rate is not representative.
    """
    strata = defaultdict(list)
    for item in items:
        strata[item['sampling_stratum']].append(item)
    for values in strata.values():
        values.sort(key=lambda item: hashlib.sha256(f"{seed}|{item['id']}".encode()).hexdigest())
    selected, types = [], set()
    while len(selected) < min(size, len(items)):
        for stratum in sorted(strata):
            available = [item for item in strata[stratum] if item['id'] not in selected]
            if not available:
                continue
            choice = next((item for item in available if item['acronym'] not in types), available[0])
            selected.append(choice['id'])
            types.add(choice['acronym'])
            if len(selected) == min(size, len(items)):
                break
    return selected


def document_overlap(test: list[dict], train: list[dict]) -> dict:
    documents = {(r.get('source'), r.get('page_title')) for r in train if r.get('page_title')}
    matches = [r for r in test if r.get('page_title') and (r.get('source'), r['page_title']) in documents]
    return {'test_items': len(matches), 'documents': len({(r['source'], r['page_title']) for r in matches}),
            'item_ids': [r['item_id'] for r in matches],
            'key': ['source', 'page_title'], 'limitation': 'Exact nonempty source/document metadata overlap; not proof of checkpoint training exposure.'}


def source_identity(provenance: dict) -> str:
    """Identity of original sources, independent of local paths or queue plans."""
    files = sorted([{key: f[key] for key in ('path', 'sha256', 'bytes')}
                    for f in provenance['files']], key=lambda f: f['path'])
    payload = {'repository': provenance.get('repository', ''), 'files': files}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True).encode()).hexdigest()


def sampling_identity(queues: dict, seed: str, size: int) -> str:
    payload = {'version': SAMPLING_VERSION, 'queues': queues, 'seed': seed, 'size': size}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True).encode()).hexdigest()


def build_dataset(root: Path, sample_size: int = 20, seed: str = SEED) -> dict:
    root = root.resolve()
    test_path = 'data/splits/test_items.csv'
    source_paths = [test_path] + [system[-1] for system in SYSTEMS]
    rows = read_rows(root / test_path)
    indexed = index_rows(rows, test_path)
    result_rows = {}
    for system in SYSTEMS:
        source = index_rows(read_rows(root / system[-1]), system[-1])
        extra = set(source) - set(indexed)
        if extra:
            raise ValueError(f'{system[-1]}: unexpected item IDs {sorted(extra)}')
        result_rows[system[0]] = source
    items = []
    for row in rows:
        answers = [make_answer(row, s, result_rows[s[0]].get(row['item_id'])) for s in SYSTEMS]
        spans = mechanical_spans(row['sentence'], row['acronym'])
        findings = []
        if len(spans) != 1:
            findings.append('multiple_target_matches' if spans else 'no_target_match')
        scores = [answer['auto_score'] for answer in answers]
        if False in scores:
            findings.append('automatic_error_present')
        if True in scores and False in scores:
            findings.append('automatic_scores_disagree')
        for answer in answers:
            findings.extend(answer['mechanical_flags'])
        source = row['source']
        origin = 'מחובר ב־AI' if row['label_origin'] == 'claude_authored' else 'טבעי לפי מטא־נתוני המקור' if row['category'] in ('knesset', 'wiki_natural') else 'לא ידוע'
        length = len(re.sub('[^א-ת]', '', row['acronym']))
        type_proxy = '2 אותיות או פחות' if length <= 2 else '3 אותיות או יותר'
        items.append({'id': row['item_id'], 'sentence': row['sentence'], 'acronym': row['acronym'],
                      'source': source, 'origin': origin, 'source_metadata': dict(row),
                      'gold': row['gold_expansion'], 'candidates': split_candidates(row['candidates']),
                      'answers': answers, 'findings': sorted(set(findings)), 'selection_reasons': [],
                      'prior_exposure': 'unknown', 'target_spans': spans, 'target_span_verified': False,
                      'target_warning': 'התאמות טקסט מכניות בלבד, לאחר נרמול מירכאות; אין סימון מוסמך של הופעת המטרה.',
                      'acronym_type_proxy': type_proxy, 'sampling_stratum': f'{source}|{type_proxy}'})
    evaluation = select_sample(items, sample_size, seed)
    diagnosis = [item['id'] for item in sorted(items, key=lambda x: (-len(x['findings']), x['id']))
                 if item['findings'] and item['id'] not in evaluation]
    for item in items:
        if item['id'] in evaluation:
            item['selection_reasons'].append(f"מדגם כיול קבוע: שכבה {item['sampling_stratum']}; דירוג SHA-256 עם seed מתועד, ללא שימוש בתוצאות לבחירה")
        elif item['id'] in diagnosis:
            item['selection_reasons'].append('תור אבחון מכני: ' + ', '.join(item['findings']))
    audit = {'item_count': len(items), 'acronym_count': len({r['acronym'] for r in rows}),
             'source_counts': dict(Counter(r['source'] for r in rows)),
             'ai_authored_items': sum(r['label_origin'] == 'claude_authored' for r in rows),
             'systems': {}, 'overlap': {}, 'target_match_counts': dict(Counter(str(len(item['target_spans'])) for item in items)),
             'semantic_judgments': 0, 'previous_human_exposure': 'unknown',
             'limits': ['CSV records establish saved responses and scores, not exact service/checkpoint identities or sentence-level prompt provenance. The test notebook dynamically fetches main; current code does not prove the immutable execution version.',
                        'Sentence joins use stable item IDs and validate stored acronym, gold and ordered candidate strings. Response CSVs do not contain sentence text.',
                        'No semantic judgments are supplied; diagnostic selection and code review are not human annotation.']}
    for s in SYSTEMS:
        a = [item['answers'][[t[0] for t in SYSTEMS].index(s[0])] for item in items]
        audit['systems'][s[0]] = {'records': len(result_rows[s[0]]), 'missing': sum(x['missing'] for x in a),
                                 'original_correct': sum(x['auto_score'] is True for x in a),
                                 'original_invalid': sum(x['auto_valid'] is False for x in a),
                                 'substring_not_exact': sum('substring_not_exact' in x['mechanical_flags'] for x in a),
                                 'recomputation_mismatches': sum('stored_score_recomputation_mismatch' in x['mechanical_flags'] for x in a)}
    for path in ['data/splits/train_items.csv', 'data/study_v1/encoder_inputs/train.csv']:
        if (root / path).exists():
            source_paths.append(path)
            train = read_rows(root / path)
            audit['overlap'][path] = dict(document_overlap(rows, train), train_rows=len(train))
    for path in ['src/hebrew_acronyms/models/common/eval.py', 'src/hebrew_acronyms/models/common/pairs.py',
                 'notebooks/run_test_eval_colab.ipynb', 'paper/sections/results.tex', 'paper/sections/evaluation.tex',
                 'paper/draft.pdf', 'results/test_results.md']:
        if (root / path).exists():
            source_paths.append(path)
    files = [{'path': p, 'sha256': hashlib.sha256((root / p).read_bytes()).hexdigest(),
              'bytes': (root / p).stat().st_size} for p in source_paths]
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    source_id = source_identity({'repository': 'https://github.com/BenCarmel123/hebrew-acronym-disambiguation', 'files': files})
    queues = {'calibration': evaluation, 'evaluation': [], 'diagnosis': diagnosis}
    plan_id = sampling_identity(queues, seed, sample_size)
    sample = [item for item in items if item['id'] in evaluation]
    return {'schema_version': EXPORT_VERSION, 'dataset_id': source_id, 'source_identity': source_id, 'sampling_plan_id': plan_id,
            'provenance': {'repository': 'https://github.com/BenCarmel123/hebrew-acronym-disambiguation',
                           'commit': commit, 'source_commit': subprocess.check_output(['git', '-C', str(root), 'log', '-1', '--format=%H', '--', *source_paths], text=True).strip(),
                           'source_root': str(root), 'files': files, 'audit': audit},
            'systems': [{'id': s[0], 'blind_id': s[1], 'name': s[2], 'task': s[3], 'response_type': s[4]} for s in SYSTEMS],
            'queues': queues,
            'sampling': {'version': SAMPLING_VERSION, 'plan_id': plan_id, 'stage': 'calibration', 'evaluation_status': 'not_decided', 'seed': seed, 'size': sample_size, 'method': select_sample.__doc__,
                         'calibration_is_evaluation_sample': False, 'diagnosis_excludes_calibration': True,
                         'strata': dict(Counter(item['sampling_stratum'] for item in sample)),
                         'sample_acronym_types': len({item['acronym'] for item in sample}),
                         'sample_items_with_automatic_success': sum(any(a['auto_score'] for a in item['answers']) for item in sample),
                         'warning': f'כעת כיול בלבד: {sample_size} פריטים במדגם כיסוי לא יחסי. הערכת ההמשך טרם נקבעה. אין להסיק שיעור שגיאה כולל מתור הכיול או האבחון. הכיוונון נעשה לאחר חשיפה למבחן.'},
            'items': items}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample-size', type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.sample_size <= 395:
        parser.error('--sample-size must be between 1 and 395')
    dataset = build_dataset(args.root, args.sample_size)
    output = args.output.resolve()
    if output.is_relative_to(args.root.resolve()):
        parser.error('--output must be outside the source repository to preserve original files')
    content = json.dumps(dataset, ensure_ascii=False, indent=2) + '\n'
    if output.exists() and output.read_text(encoding='utf-8') != content:
        parser.error('--output already contains different data; choose a fresh output path')
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        with output.open('x', encoding='utf-8') as handle:
            handle.write(content)
    print(json.dumps({'output': str(args.output), 'dataset_id': dataset['dataset_id'],
                      'items': len(dataset['items']), 'systems': len(dataset['systems'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
