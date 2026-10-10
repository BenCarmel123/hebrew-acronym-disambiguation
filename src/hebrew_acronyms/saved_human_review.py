"""Reconstruct the closed, partially reviewed 381-item experiment offline.

The two signed exports retain decision history. Only their explicit response
bindings receive labels; group similarity never extends a judgment to new runs.
"""
from collections import Counter
import csv
from datetime import datetime
import hashlib
from html import escape
import json
from pathlib import Path

from .test_review_export import digest, identified_review_coverage, identified_group_id

EXPORTS = ('human-review-381-20261010/review-export.json',
           'human-review-focus-200-20261010/review-export.json')
FINAL = 'human-review-final-20261010'
NEW = 'new_human_judgment'
REUSE = 'explicit_historical_reuse_not_new_judgment'
CASES = ('kn4-0525', 'kn-0229', 'manual-0023', 'manual-0001')


def _read(path):
    return json.loads(Path(path).read_text())


def _merge_decisions(exports, answers, rechecks):
    """Accept only the two explicitly documented later rechecks, once each."""
    judged, events, actual_rechecks = {}, [], []
    expected = {r['answer_id']: r for r in rechecks}
    for round_name, source, export in zip(('previous', 'focused'), EXPORTS, exports):
        for decision in export['decisions']:
            j = decision['judgment']
            events.append({'round': round_name, 'source_export': source, **decision})
            for aid in decision['answer_ids']:
                item, answer = answers[aid]
                if identified_group_id(item, answer) != decision['decision_id']:
                    raise ValueError('Decision group differs from exact context and response')
                current = dict(answer_id=aid, group_id=decision['decision_id'], label=j['label'],
                               kind=NEW, round=round_name, decision_id=decision['decision_id'],
                               source_export=source, label_updated_at=j['label_updated_at'])
                if aid in judged:
                    old = judged[aid]
                    evidence = expected.get(aid)
                    if not evidence or any(
                        entry[k] != evidence[stage][k]
                        for stage, entry in [('previous', old), ('latest', current)]
                        for k in ('answer_id', 'group_id', 'label', 'kind', 'round', 'label_updated_at')):
                        raise ValueError('Undocumented or changed repeated judgment')
                    if datetime.fromisoformat(current['label_updated_at']) <= datetime.fromisoformat(old['label_updated_at']):
                        raise ValueError('Recheck must be later than the superseded decision')
                    actual_rechecks.append(dict(answer_id=aid, previous=old, latest=current))
                judged[aid] = current
    if {r['answer_id'] for r in actual_rechecks} != set(expected) or len(actual_rechecks) != len(expected):
        raise ValueError('Missing or repeated recheck')
    return judged, events, actual_rechecks


def load_final_review(saved_root, cohort_path):
    """Validate source hashes, contexts, bindings, approvals and aggregate coverage."""
    saved = Path(saved_root)
    inventory = _read(saved / 'files.json')['files']
    exports = [_read(saved / path) for path in EXPORTS]
    audit = _read(saved / FINAL / 'combined-review-coverage.json')
    proof = _read(saved / FINAL / 'reuse-context-evidence.json')
    sources = {s['path']: s for e in exports for s in e['source_files']}
    for relative in [*EXPORTS, *sources, f'{FINAL}/combined-review-coverage.json',
                     f'{FINAL}/reuse-context-evidence.json', f'{FINAL}/selection-manifest.json']:
        content = (saved / relative).read_bytes()
        entry = inventory[relative]
        if len(content) != entry['bytes'] or hashlib.sha256(content).hexdigest() != entry['sha256']:
            raise ValueError('Saved review evidence differs: ' + relative)
    selection = _read(saved / FINAL / 'selection-manifest.json')
    identified_review_coverage(saved / EXPORTS[0], saved, cohort_path)
    identified_review_coverage(saved / EXPORTS[1], saved, cohort_path,
                               selected_group_ids=selection['reasons'])
    with Path(cohort_path).open(encoding='utf-8-sig') as handle:
        cohort = {r['item_id']: r for r in csv.DictReader(handle)}
    answers = {}
    for relative in sources:
        for item in _read(saved / relative)['items']:
            if item['original_item_id'] not in cohort:
                continue  # Original 395-item bundles preserve the 14 excluded records.
            row = cohort[item['original_item_id']]
            if (item['sentence'], item['acronym'], item['gold']) != (row['sentence'], row['acronym'], row['gold_expansion']):
                raise ValueError('Review context differs from the scored cohort')
            for answer in item['answers']:
                binding = answer['binding']; aid = answer['id']
                if (aid in answers or aid != digest(binding) or binding['item_id'] != row['item_id']
                        or binding['response_sha256'] != hashlib.sha256(answer['raw'].encode()).hexdigest()):
                    raise ValueError('Duplicate or invalid response binding')
                if binding['task'] != {'selection': 'select', 'generation': 'generate'}[answer['task']]:
                    raise ValueError('Task binding differs')
                answers[aid] = (item, answer)
    judged, events, rechecks = _merge_decisions(exports, answers, audit['rechecks'])
    reuse_groups = exports[0]['historical_reuse']
    if exports[1]['historical_reuse']:
        raise ValueError('Unexpected second-round historical reuse')
    for gid, reuse in reuse_groups.items():
        p = reuse['proposal']; context = proof['groups'][gid]
        if (reuse['kind'] != REUSE or not reuse.get('confirmed_at') or not reuse.get('confirmed_by')
                or p['source_sha256'] != proof['source_sha256']
                or context['historical_context'] != context['current_context']
                or p['label'] not in {'fits', 'not_fits', 'unsure'}
                or len(p['new_answer_ids']) != len(p['new_bindings']) or not p['historical_sources']):
            raise ValueError('Historical reuse lacks explicit approval or matching context')
        for aid, binding in zip(p['new_answer_ids'], p['new_bindings']):
            item, answer = answers[aid]
            if (aid in judged or binding != answer['binding'] or identified_group_id(item, answer) != gid
                    or item['original_item_id'] != context['original_item_id']
                    or any(item.get(k) != v for k, v in context['current_context'].items())
                    or answer['technical_failure']):
                raise ValueError('Reused answer overlaps or differs from approved binding/context')
            for old in p['historical_sources']:
                old_answer, old_judgment = old['answer'], old['judgment']
                fields = ('task', 'raw') + (('decoded', 'option_mapping') if answer['task'] == 'selection' else ())
                if (any(old_answer.get(k) != answer.get(k) for k in fields)
                        or old_judgment['label'] != p['label'] or not old_judgment.get('label_updated_at')
                        or not old_judgment.get('annotator') or not old_judgment.get('origin')):
                    raise ValueError('Historical judgment or exact response differs')
            judged[aid] = dict(answer_id=aid, group_id=gid, label=p['label'], kind=REUSE,
                               round='historical_reuse', source_export=EXPORTS[0],
                               confirmed_at=reuse['confirmed_at'], confirmed_by=reuse['confirmed_by'])
    audit_judged = {r['answer_id']: r for r in audit['response_judgments']}
    if set(judged) != set(audit_judged) or any(
            j[k] != audit_judged[aid][k] for aid, j in judged.items()
            for k in ('group_id', 'label', 'kind')):
        raise ValueError('Reconstructed judgments differ from final audit')
    groups, records = {}, []
    for aid, (item, answer) in answers.items():
        key = (answer['system_id'], answer['task'], answer['binding']['run_id'])
        r = groups.setdefault(key, Counter(system=key[0], task=key[1], run_id=key[2]))
        r['total'] += 1
        technical = bool(answer['technical_failure']); positive = bool(answer['auto_score'])
        r['technical_failures'] += technical
        if not technical:
            r['positive_total' if positive else 'negative_total'] += 1
        judgment = judged.get(aid)
        if judgment:
            if technical:
                raise ValueError('Technical failure received a semantic judgment')
            r['reviewed'] += 1; r[judgment['label']] += 1
            r['new_human_reviewed' if judgment['kind'] == NEW else 'historical_reused'] += 1
            r['positive_reviewed' if positive else 'negative_reviewed'] += 1
            r['automatic_negative_human_fits'] += not positive and judgment['label'] == 'fits'
            r['automatic_positive_human_not_fits'] += positive and judgment['label'] == 'not_fits'
        records.append(dict(answer_id=aid, item_id=item['original_item_id'], sentence=item['sentence'],
            acronym=item['acronym'], gold=item['gold'], system=key[0], task=key[1], run_id=key[2],
            response_sha256=answer['binding']['response_sha256'], raw=answer['raw'], decoded=answer.get('decoded'),
            option_mapping=answer.get('option_mapping'), automatic_score=answer['auto_score'],
            automatic_rule=answer['auto_score_rule'], technical_failure=technical, status=answer['status'],
            human_label=judgment['label'] if judgment else '',
            judgment_kind=judgment['kind'] if judgment else 'unreviewed',
            judgment_source=judgment['source_export'] if judgment else '',
            group_id=identified_group_id(item, answer), pending_clarification=(positive and bool(judgment) and judgment['label']=='not_fits')))
    rows = []
    for key, counts in sorted(groups.items()):
        template = next(r for r in audit['coverage'] if (r['system'], r['task'], r['run_id']) == key)
        row = {k: counts[k] for k in template}
        row['groups'] = len({judged[aid]['group_id'] for aid,(i,a) in answers.items()
                            if aid in judged and (a['system_id'],a['task'],a['binding']['run_id']) == key})
        row['unreviewed'] = row['total'] - row['reviewed']
        row['valid_unreviewed'] = row['unreviewed'] - row['technical_failures']
        row['unresolved'] = row['unreviewed'] + row['unsure']
        if row != template:
            raise ValueError('Reconstructed per-system coverage differs from final audit')
        rows.append(row)
    totals = {k: sum(r[k] for r in rows) for k in audit['totals'] if k in rows[0] and k != 'groups'}
    totals.update(unique_groups_with_review=len({j['group_id'] for j in judged.values()}),
        full_source_groups=len({identified_group_id(i,a) for i,a in answers.values()}), new_decision_events=len(events),
        new_unique_groups=len({j['group_id'] for j in judged.values() if j['kind']==NEW}),
        historical_reuse_groups=len(reuse_groups), rechecked_responses=len(rechecks))
    if totals != audit['totals']:
        raise ValueError('Reconstructed final totals differ from audit')
    pending = [r for r in records if r['pending_clarification']]
    if {r['item_id'] for r in pending} != set(CASES) or len(pending) != 9:
        raise ValueError('Pending clarification cases differ from the approved list')
    return dict(rows=rows, totals=totals, records=records, decisions=events, rechecks=rechecks,
                reuse= reuse_groups, cases=[r for r in records if r['item_id'] in CASES and
                    (r['pending_clarification'] or (r['item_id']=='kn4-0525' and r['system']=='gemini_generate'))])


def write_csv(path, rows):
    """Write a UTF-8 table with structured fields retained as JSON strings."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError('Cannot export an empty table without a schema')
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerows({k: json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v
                         for k,v in row.items()} for row in rows)


def clarification_html(cases):
    sections = []
    for item_id in CASES:
        rows = [r for r in cases if r['item_id'] == item_id]; first = rows[0]
        body = ''.join('<tr>' + ''.join('<td dir="auto">'+escape(str(r[k]))+'</td>'
               for k in ('system','task','raw','decoded','human_label','run_id'))+'</tr>' for r in rows)
        sections.append(f'<section><h3>{escape(item_id)} — pending clarification</h3>'
            f'<p dir="rtl">{escape(first["sentence"])}</p><p dir="auto">Target: {escape(first["acronym"])}; '
            f'reference: {escape(first["gold"])}</p><div style="overflow-x:auto"><table><thead><tr>'
            '<th>System</th><th>Task</th><th>Raw answer</th><th>Decoded</th><th>Saved judgment</th><th>Run</th>'
            f'</tr></thead><tbody>{body}</tbody></table></div></section>')
    return ('<p>Nine automatic-positive / not_fits occurrences across four items remain unresolved. '
           'Labels and automatic scores are unchanged; these are not nine confirmed independent false positives.</p>'+''.join(sections))


def export_review_analysis(review, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    write_csv(output/'human_review_coverage.csv', review['rows'])
    write_csv(output/'response_review_bindings.csv', review['records'])
    write_csv(output/'clarification_cases.csv', review['cases'])
    write_csv(output/'review_summary.csv', [review['totals']])
    for name in ('decisions', 'rechecks'):
        (output/f'human_review_{name}.json').write_text(json.dumps(review[name],ensure_ascii=False,indent=2)+'\n')
    html = '<!doctype html><meta charset="utf-8"><title>Human review: clarification cases</title><style>'
    html += 'body{font-family:Arial,sans-serif;max-width:1200px;margin:40px auto;padding:20px;color:#222}'
    html += 'td,th{padding:10px;border:1px solid #ddd;text-align:start}table{border-collapse:collapse}section{margin:40px 0}p{line-height:1.7}</style>'
    (output/'clarification_cases.html').write_text(html+'<h1>Cases for discussion</h1>'+clarification_html(review['cases']))


def review_coverage_figure(review, model_labels, output):
    """Plot response coverage, not semantic accuracy of a selected subset."""
    import matplotlib.pyplot as plt
    rows = sorted(review['rows'], key=lambda r: (r['task'], r['system']))
    with plt.rc_context({'font.size':9, 'pdf.fonttype':42, 'svg.fonttype':'none'}):
        fig, ax = plt.subplots(figsize=(9,6.5), layout='constrained')
        left = [0]*len(rows)
        for key,label,color in [('new_human_reviewed','New judgments','#416788'),
                                ('historical_reused','Approved reuse','#88AB75'),
                                ('valid_unreviewed','Technically complete, unreviewed','#DCDCDC'),
                                ('technical_failures','Technical failures','#BD6B57')]:
            values = [r[key] for r in rows]
            ax.barh(range(len(rows)), values, left=left, label=label, color=color, height=.65)
            left = [a+b for a,b in zip(left,values)]
        ax.set_yticks(range(len(rows)),[model_labels[r['system'].rsplit('_',1)[0]]+' / '+r['task'] for r in rows])
        ax.invert_yaxis(); ax.set_xlim(0,381); ax.set_xticks([0,100,200,300,381])
        ax.set_xlabel('Answer occurrences (381 per system and task)')
        ax.set_title('Human review coverage: prioritized diagnostic review',loc='left',pad=15)
        ax.spines[['top','right']].set_visible(False)
        ax.legend(loc='upper center',bbox_to_anchor=(.35,-.13),ncols=2,frameon=False,fontsize=8)
        for extension in ('png','pdf','svg'):
            fig.savefig(Path(output)/f'human_review_coverage.{extension}',dpi=200,bbox_inches='tight')
    return fig
