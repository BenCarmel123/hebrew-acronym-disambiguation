"""Offline comparison of identified runs, preserving every original arm run ID."""
from collections import Counter
from html import escape
import hashlib
import json
from pathlib import Path

from hebrew_acronyms.experimental_study import validate_artifact
from hebrew_acronyms.models.common.eval import summarize_selection


def compare_saved_runs(sources):
    """sources is a list of (study.json path, exact run ID); no model/API imports."""
    if not sources:
        raise ValueError('Provide identified saved sources')
    snapshots = [Path(path).read_bytes() for path, run_id in sources]
    artifacts = [validate_artifact(json.loads(data), expected_run_id=run_id)
                 for data, (path, run_id) in zip(snapshots, sources)]
    base = artifacts[0]
    if any(a['input_identity'] != base['input_identity'] for a in artifacts):
        raise ValueError('Comparison requires identical complete input snapshots')
    selected, origins = {}, {}
    for a in artifacts:
        for arm in ('dictabert', 'qwen_generate', 'qwen_select', 'gemini_generate', 'gemini_select'):
            records = [r for r in a['records'] if r['condition'] == arm]
            if not records or all(r['status'] == 'not_run' for r in records):
                continue
            if arm in selected:
                raise ValueError(f'Multiple attempted sources for {arm}; choose one explicitly')
            selected[arm], origins[arm] = records, a['run_id']
    # A cross-run view is not a synthetic run. Original records remain unchanged.
    records = [r for arm in selected.values() for r in arm]
    for task in ('generate', 'select'):
        q = {r['item_id']: r for r in selected.get('qwen_' + task, [])}
        for g in selected.get('gemini_' + task, []):
            if g.get('prompt') and g['item_id'] in q:
                if any(q[g['item_id']].get(k) != g.get(k) for k in ('prompt', 'prompt_sentence', 'shown_order')):
                    raise ValueError('LLM prompts or shown candidate orders differ')
    metrics = []
    for arm in ('dictabert', 'qwen_select', 'gemini_select'):
        if arm in selected:
            metric = summarize_selection(base['items'], selected[arm], arm)
            metric['source_run_id'] = origins[arm]
            metric['n_correct'] = sum(d['correct'] is True for d in metric['details'])
            metrics.append(metric)
    return {'input_identity': base['input_identity'], 'items': base['items'], 'records': records,
            'arm_run_ids': origins, 'metrics': metrics, 'generation_score_status': 'manual_review_unscored',
            'sources': [{'path': str(Path(p).resolve()), 'run_id': r,
                         'sha256': hashlib.sha256(data).hexdigest()}
                        for data, (p, r) in zip(snapshots, sources)]}


def display_saved_comparison(sources):
    result = compare_saved_runs(sources)
    def t(value):
        return escape('—' if value is None else str(value))
    def pct(value):
        return 'unavailable' if value is None else f'{value:.2%}'
    html = '''<!doctype html><html lang="he"><meta charset="utf-8">
<title>Five-arm dev comparison</title><style>
body{font:16px system-ui;max-width:1180px;margin:32px auto;padding:0 20px;color:#172536;background:#f7f9fc}
h1,h2{line-height:1.2}table{border-collapse:collapse;background:white;width:100%;margin:14px 0}
th,td{padding:10px;border:1px solid #dbe1e8;text-align:left;vertical-align:top}
details{background:white;border:1px solid #dbe1e8;border-radius:8px;padding:14px;margin:12px 0}
summary{cursor:pointer;font-weight:650}pre{white-space:pre-wrap;overflow-wrap:anywhere}
.notice{padding:14px;background:#fff0cf;border-radius:8px} .answer{white-space:pre-wrap;overflow-wrap:anywhere}
code{overflow-wrap:anywhere} .muted{color:#526476;font-size:13px}</style>
<h1>השוואת חמש הזרועות · 62 פריטי dev</h1>
<p dir="rtl">תצוגה שמורה בלבד. פתיחתה אינה טוענת מודלים או שולחת בקשות. כל התשובות והציונים משויכים למזהי הריצות המקוריים.</p>
<div class="notice" dir="rtl">ציוני הבחירה מחושבים מול התוויות הקיימות לצורך אבחון. ידועות בעיית תווית במט״ח וחסר של ״חומרים מסוכנים״ במועמדי חמ״ס. כל 62 הפריטים נשארו ללא שינוי. יצירה טרם נשפטה אנושית ואינה מקבלת ציון סמנטי. תשובת מודל אינה תווית אמת. חמישה מתוך 11 סוגים כוללים פריט יחיד.</div>'''
    html += '<h2>Selection · existing labels</h2><table><tr><th>System</th><th>Correct / all</th><th>Micro</th><th>Macro (11 types)</th><th>Attempted</th><th>Failures / statuses</th></tr>'
    for m in result['metrics']:
        html += '<tr>' + ''.join('<td>' + t(v) + '</td>' for v in (
            m['system'], f"{m['n_correct']}/{m['n_items']}", pct(m['micro_accuracy']), pct(m['macro_accuracy']),
            f"{m['n_attempted']}/{m['n_items']}", json.dumps(m['status_counts'], ensure_ascii=False))) + '</tr>'
    html += '</table><p>Failures and unrun items remain in the 62-item denominator once an arm is attempted. Partial results are provisional.</p>'
    html += '<h2>Arms and source runs</h2><table><tr><th>Arm</th><th>Run ID</th><th>Response status counts</th><th>Scoring</th></tr>'
    arms = ('dictabert', 'qwen_generate', 'qwen_select', 'gemini_generate', 'gemini_select')
    by_key = {(r['condition'], r['item_id']): r for r in result['records']}
    for arm in arms:
        counts = Counter(r['status'] for r in result['records'] if r['condition'] == arm)
        html += '<tr>' + ''.join('<td>'+t(v)+'</td>' for v in (arm, result['arm_run_ids'].get(arm, 'not_run'),
            json.dumps(counts), 'Unjudged; no semantic score' if arm.endswith('generate') else 'Exact selection'))+'</tr>'
    html += '</table><h2>All items for review with Ben</h2>'
    scored = {(m['condition'], d['item_id']): d for m in result['metrics'] for d in m['details']}
    for i, row in enumerate(result['items'], 1):
        item_id = row['item_id']
        flags = []
        if row['acronym'] == 'חמ״ס': flags.append('Review candidate inventory: חומרים מסוכנים absent')
        if item_id == 'enc-73ed2da155b891b95861f67deb04b9f8': flags.append('Review gold/context mismatch')
        if row['acronym'] == 'רמב״ם': flags.append('Review institutional uses versus person-expansion gold; annotation convention unresolved')
        if row['acronym'] == 'נ״ר': flags.append('Review נ״ר interpretations and explicit expansion in context')
        html += f'<details id="{t(item_id)}"><summary>{i}. {t(row["acronym"])} · {t(item_id)}'+(' · REVIEW' if flags else '')+'</summary>'
        a,b=int(row['span_start']),int(row['span_end'])
        marked=t(row['sentence'][:a])+'<mark>'+t(row['sentence'][a:b])+'</mark>'+t(row['sentence'][b:])
        html += f'<p dir="rtl">{marked}</p><p>Target: <b dir="auto">{t(row["target_raw"])}</b> · span [{a}, {b})</p>'
        html += f'<p>Gold: <b dir="auto">{t(row["gold_expansion"])}</b></p><p dir="auto">Candidates: {t(row["candidates"])}</p>'
        if flags: html += '<p class="notice">'+'; '.join(map(t,flags))+'</p>'
        html += '<table><tr><th>Arm</th><th>Answer</th><th>Status / label match</th></tr>'
        for arm in arms:
            r=by_key.get((arm,item_id), {'status':'not_run'})
            d=scored.get((arm,item_id), {})
            answer=d.get('selected_candidate') if arm == 'dictabert' or arm.endswith('select') else r.get('raw_response')
            html += f'<tr><td>{t(arm)}</td><td dir="auto" class="answer">{t(answer)}'
            if arm.endswith('select'):
                html += f'<p class="muted">Raw: {t(r.get("raw_response"))}</p>'
                mapping=' | '.join(f'{chr(65+j)}: {c}' for j,c in enumerate(r.get('shown_order') or []))
                html += f'<p class="muted">{t(mapping)}</p>'
            html += '</td><td>'+t(d.get('status',r['status']))
            if d and d.get('status') == 'ok':
                html += ' · '+('matches gold' if d.get('correct') else 'does not match gold')
            elif d:
                html += ' · no valid prediction; included in denominator'
            if r.get('error'): html += '<p>'+t(r['error'])+'</p>'
            html += '</td></tr>'
        html += '</table><p class="muted">Human review: pending. Resolve label and candidate inventory with Ben; do not infer truth from model agreement.</p></details>'
    html += '<h2>Input identity</h2><code>'+t(result['input_identity']['sha256'])+'</code>'
    html += '<p class="muted">Source artifact hashes and original run identities are retained in comparison.json. No synthetic run ID is assigned.</p></html>'
    return html, result


def export_saved_comparison(sources, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    html, result = display_saved_comparison(sources)
    (output_dir/'saved_results.html').write_text(html, encoding='utf-8')
    (output_dir/'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return output_dir/'saved_results.html'
