"""Apply the human dev review to data/splits/dev_items.csv.

Three kinds of change, all recorded in new columns rather than silently:
  - relabel: reviewer supplied the correct expansion in their note
  - drop:    text is broken by substitution; no correct answer exists
  - flag:    reviewer marked unsure; row is KEPT, flagged for later
"""
import csv, sys, shutil

DEV = 'data/splits/dev_items.csv'
REVIEW = sys.argv[1]
OUT = sys.argv[2]

# reviewer note -> corrected expansion. Added to candidates when absent,
# because validate_data.py requires gold to appear in its own candidate list.
RELABEL = {
    'ha-00475': 'בית דין משותף',
    'ha-00476': 'בית דין משותף',
    'ha-02321': 'סמל משנה',
    'ha-01377': 'יעקב בן צבי',
    'ha-01378': 'יעקב בן צבי',
    'ha-01379': 'יעקב בן צבי',
    'ha-02612': 'פיתוח חינוך',
    'ha-02048': 'מפקד כוחות',
    'ha-02857': "ר' משה בן מימון",
    'ha-02858': "ר' משה בן מימון",
    'ha-02859': "ר' משה בן מימון",
    'ha-01928': 'מערך ציוד ודרג',
    'ha-01929': 'מערך ציוד ודרג',
    'ha-01930': 'מערך ציוד ודרג',
}
# broken text with no recoverable label
DROP = {
    'ha-02613': 'substitution split a personal name (יצחק פ״ח רוזנטל)',
    'ha-02269': 'non-prose product listing; acronym is a Latin-script brand',
    'ha-01530': 'metalinguistic mention of the term; no context determines the sense',
    'ha-01770': 'reviewer judged label wrong; correct rabbi not determinable',
}

rev = {r['item_id']: r for r in csv.DictReader(open(REVIEW, encoding='utf-8-sig'))}
rows = list(csv.DictReader(open(DEV, encoding='utf-8-sig')))
fields = list(rows[0].keys())
for c in ('review_verdict', 'review_note', 'label_origin'):
    if c not in fields: fields.append(c)

# A corrected expansion becomes a candidate for EVERY row of that acronym type,
# not only the rows it corrects. Otherwise the new option would appear exactly
# where it is the answer, which leaks which rows were reviewed; as a type-wide
# candidate it is a genuine distractor on the rows where it is wrong.
new_by_type = {}
for i, exp in RELABEL.items():
    a = next(r['acronym'] for r in rows if r['item_id'] == i)
    new_by_type.setdefault(a, set()).add(exp)

out, dropped, relabelled, flagged, widened = [], [], [], [], 0
for r in rows:
    i = r['item_id']
    v = rev.get(i, {})
    r['review_verdict'] = v.get('verdict', 'unreviewed')
    r['review_note'] = v.get('note', '')
    r['label_origin'] = 'substitution'

    if i in DROP:
        dropped.append((i, r['acronym'], DROP[i]))
        continue

    extra = new_by_type.get(r['acronym'], set())
    if extra:
        cands = [c.strip() for c in r['candidates'].split('|') if c.strip()]
        added = [e for e in sorted(extra) if e not in cands]
        if added:
            cands.extend(added)
            r['candidates'] = ' | '.join(cands)
            r['n_candidates'] = str(len(cands))
            widened += 1

    if i in RELABEL:
        old = r['gold_expansion']
        r['gold_expansion'] = RELABEL[i]
        r['provisional_expansion'] = RELABEL[i]
        r['label_status'] = 'verified'
        r['label_origin'] = 'human_review'
        relabelled.append((i, r['acronym'], old, RELABEL[i]))

    if r['review_verdict'] == 'unsure':
        flagged.append(i)
        if not r['review_note']:
            r['review_note'] = 'reviewer unsure: sense not clearly determined by the sentence'

    out.append(r)

with open(OUT, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); w.writerows(out)

print(f'in {len(rows)} -> out {len(out)}')
print(f'\nrelabelled {len(relabelled)}:')
for i,a,o,n in relabelled: print(f'  {i} {a}: {o} -> {n}')
print(f'\ndropped {len(dropped)}:')
for i,a,why in dropped: print(f'  {i} {a}: {why}')
print(f'\ncandidate list widened on {widened} rows (type-wide)')
print(f'flagged unsure (kept): {len(flagged)}')
