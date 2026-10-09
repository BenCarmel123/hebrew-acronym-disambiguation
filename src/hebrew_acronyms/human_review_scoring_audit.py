"""Read-only evidence and proposed scoring checks, never replacement scores."""
from __future__ import annotations
import ast
import copy
import hashlib
from pathlib import Path


def audit_original_rule(dataset):
    """Verify the pure saved-score functions and original file hashes without inference."""
    root = Path(dataset['provenance']['source_root'])
    manifest = {f['path']: f for f in dataset['provenance']['files']}
    paths = ['src/hebrew_acronyms/models/common/eval.py', 'src/hebrew_acronyms/models/common/pairs.py',
             'notebooks/run_test_eval_colab.ipynb']
    for path in paths:
        content = (root / path).read_bytes()
        if hashlib.sha256(content).hexdigest() != manifest[path]['sha256']:
            raise ValueError('Scoring source hash changed: ' + path)
    evaluation = ast.parse((root / paths[0]).read_text())
    pairs = ast.parse((root / paths[1]).read_text())
    pure = [n for n in pairs.body if (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_EQUIV' for t in n.targets)) or (isinstance(n, ast.FunctionDef) and n.name == '_normalise')]
    pure += [n for n in evaluation.body if isinstance(n, ast.FunctionDef) and n.name == 'is_correct']
    if len(pure) != 3:
        raise ValueError('Expected quotation map, normalization and scoring functions')
    # These inspected definitions contain only string normalization and substring testing.
    namespace = {}
    exec(compile(ast.Module(body=pure, type_ignores=[]), '<verified-pure-scorer>', 'exec'), namespace)
    tested, mismatches = 0, []
    for item in dataset['items']:
        for answer in item['answers']:
            if answer['system_id'] not in {'qwen_generate', 'gemini_generate'}: continue
            if type(answer.get('auto_score')) is not bool or not isinstance(answer.get('raw'), str): continue
            source = answer['source_file']
            if hashlib.sha256((root / source).read_bytes()).hexdigest() != manifest[source]['sha256']:
                raise ValueError('Original response source changed')
            reproduced = namespace['is_correct'](answer['raw'], item['gold'].strip())
            tested += 1
            if reproduced != answer['auto_score']:
                mismatches.append({'item_id': item['id'], 'answer_id': answer['id']})
    return {'verified': True, 'rule': 'quote_normalized_gold_substring',
            'description': 'במסלול היצירה נבדקת הכלת הייחוס בתשובה לאחר נרמול גרשיים ומירכאות; אין זו התאמת Exact Match מלאה או בדיקת משמעות.',
            'normalization': 'גרשיים עבריים ומירכאות מסולסלות מוחלפים בסימני ASCII המתאימים. הייחוס עובר strip בגבולות; אין איחוד רווחים פנימיים, מקפים, כתיב או נטייה.',
            'evidence': [{'path': p, 'sha256': manifest[p]['sha256']} for p in paths],
            'saved_generation_answers_checked': tested, 'score_mismatches': mismatches,
            'provenance_limit': 'הפונקציות שנבדקו משחזרות את הציונים השמורים והמחברת מפנה למסלול זה. המחברת טוענת קוד מ־main; אין בכך הוכחה בלתי תלויה לגרסת הקוד המדויקת בזמן ההרצה המקורית.',
            'official_scores_modified': False}


def recommendation_rows(summary):
    """Point to annotated cases as reasons to investigate, not proof of causality."""
    cases = summary.get('cases', [])
    rows = []
    for case in cases:
        for a in case.get('answers', []):
            rows.append(dict(a, item_id=case['id'], suspect=case.get('suspect', False)))
    def refs(tags):
        return [{'item_id': r['item_id'], 'answer_id': r.get('answer_id', r.get('id')), 'label': r.get('label'),
                 'auto_score': r.get('auto_score'), 'tags': r.get('tags', [])}
                for r in rows if set(r.get('tags', [])) & set(tags)]
    specs = [
        ('technical_normalization', ['punctuation'], 'בדיקת נרמול טכני של רווחים וסימני פיסוק',
         'עשוי לטפל בדחייה שנגרמה להבדלי רווחים, מקפים או גרשיים בלבד, אם המשמעות אושרה בהקשר.',
         'מחיקה רחבה עלולה לחבר מילים שונות או לקבל תשובה ארוכה/סותרת. יש להגדיר במפורש אילו סימנים מנורמלים ולבדוק גם קבלה שגויה.'),
        ('controlled_spelling', ['spelling'], 'בדיקת חלופות כתיב מוגדרות ומבוקרות',
         'עשוי לטפל בחלופות כתיב מסוימות שאושרו כאותה משמעות בהקשר.',
         'מחיקה גורפת של י׳/ו׳ או דמיון מחרוזות עלולים לאחד מילים שונות. יש לאשר זוגות חלופות מסוימים, ללא כלל מחיקה גורף.'),
        ('human_approved_equivalents', ['equivalent', 'inflection'], 'חלופות משמעות ונטייה לאישור אנושי',
         'עשוי לטפל בניסוח שקול שנדחה, רק לאחר אישור שהמשמעות נשמרת במשפט המסוים.',
         'נושא דומה אינו אותה משמעות; יחיד ורבים עשויים לשנות אותה. אין לאחד נטיות אוטומטית או ללמוד שקילות מדמיון מחרוזות.'),
        ('contradictory_or_extra_text', ['extra_text', 'gibberish'], 'בדיקת טקסט עודף, סותר או משובש גם כשיש הכלת ייחוס',
         'עשוי לזהות קבלה שגויה של שלילה, סתירה או רשימת ניחושים שמכילה גם את הייחוס; תגית לבדה אינה הוכחה לכך.',
         'סינון אוטומטי לפי אורך או שיבוש עלול לדחות פירוש נכון עם טקסט עודף. ג׳יבריש חלקי אינו קובע את השיפוט הסמנטי.')]
    result = []
    for key, tags, title, benefit, risk in specs:
        evidence = refs(tags)
        result.append({'id': key, 'title': title, 'evidence': evidence,
                       'status': 'מועמד לבדיקה בעקבות תגיות אנושיות; לא כלל מאושר' if evidence else 'אין עדיין מקרים מתויגים התומכים בכלל הזה; לא הופקה המלצה אמפירית',
                       'possible_failure': benefit, 'false_acceptance_risk': risk,
                       'causal_limit': 'הופעת תגית לצד פער אינה מוכיחה שהיא סיבת הפער.'})
    return result


def enrich_summary(summary, dataset):
    result = copy.deepcopy(summary)
    if result.get('masked', True): return result
    result['scoring_rule'] = dataset.get('scoring_audit', {'verified': False, 'description': 'כלל הניקוד לא אומת בחבילת נתונים זו.'})
    result['recommendations'] = recommendation_rows(result)
    result['next_step'] = 'ההצעות מיועדות להחלטת שקד ובן בלבד. בדיקת כלל חדש תשתמש באותן תשובות שמורות, תשמור את הציון המקורי ותציג כל שינוי לשני הכיוונים. יש לבדוק גם מקרי ביקורת שהתקבלו אוטומטית. הצלחה על מקרי פיתוח הכלל אינה אימות עצמאי, והמדגם אינו אומדן לדיוק סמנטי בכלל הבנצ׳מרק.'
    return result


def scoring_markdown(summary):
    if summary.get('masked', True): return ''
    rule = summary.get('scoring_rule', {})
    lines = ['\n## כלל הניקוד המקורי שנבדק\n', rule.get('description', ''), rule.get('normalization', ''),
             f"נבדקו {rule.get('saved_generation_answers_checked', 0)} תשובות שמורות; נמצאו {len(rule.get('score_mismatches', []))} אי־התאמות לשחזור הכלל.",
             rule.get('provenance_limit', ''), '\n## כללים אפשריים לבדיקה — לא שינוי תוצאות\n']
    for r in summary.get('recommendations', []):
        lines += ['### ' + r['title'], r['status'], r['possible_failure'], 'סיכון: ' + r['false_acceptance_risk'],
                  'מקרים לבחינה: ' + ', '.join(str(x['item_id']) + '/' + str(x['answer_id']) for x in r['evidence']), r['causal_limit'], '']
    lines += [summary.get('next_step', '')]
    return '\n\n'.join(lines)
