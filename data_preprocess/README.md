# Acronym candidate tables

Builds the candidate-expansion inventory for the Hebrew acronym disambiguation
project, with a corpus-frequency count attached to every candidate.

This is **exploration evidence, not gold data.** Nothing here is filtered on
plausibility: person senses and initials mismatches are *flagged in their own
columns* so the whole distribution stays visible and thresholds can be chosen
from the data. Every row is candidate evidence requiring human adjudication.

## Why two sources

`research/source_evidence/REPORT.md` established that the two wikis carry
different evidence: **Wiktionary states literal expansions, Wikipedia
disambiguation pages mostly list referents.** Since the ratified prediction
target is the literal expansion string (`planning/DECISIONS.md` §1), Wiktionary
is the higher-value inventory — but the overlap is only partial, so the union is
strictly larger than either.

| Source | Types | Polysemous |
|---|---:|---:|
| Wikipedia `קטגוריה:פירושון ראשי תיבות` | 117 | ~80 |
| Wiktionary `קטגוריה:ראשי תיבות` | 3,678 | 701 |
| Shared types | 50 | |

The 3,678 / 701 counts reproduce the source probe's independently-derived
3,680 / 710 to within 1.3%, using different code.

## Usage

    # one row per Wikipedia disambiguation bullet
    python -m data_preprocess wikipedia   --out data/mined/bullet_counts.csv

    # one row per Wiktionary sense (--min-senses 2 = polysemous types only)
    python -m data_preprocess wiktionary  --min-senses 2 \
        --out data/mined/wiktionary_counts.csv

    # union of the two
    python -m data_preprocess merge --out data/mined/merged_counts.csv

Run from the repository root. Each command also writes `<out>.summary.json`.
All three CSVs share one column set.

## Columns

| Column | Meaning |
|---|---|
| `acronym` | normalised surface, gershayim (`מ״מ`) |
| `page_title` | source page the candidate came from |
| `expansion` | the candidate expansion |
| `hits` | **Hebrew Wikipedia** articles containing the phrase |
| `script` | `hebrew` / `latin` / `other` |
| `initials_match` | expansion's initials match the acronym; empty for non-Hebrew |
| `looks_like_person` | biographical-sense heuristic |
| `source` | `wikipedia`, `wiktionary`, or `wikipedia+wiktionary` in the merge |
| `domain` | Wiktionary register label (`מתמטיקה`, `צה"ל`); empty for Wikipedia |
| `raw_line` | the original line, for auditing any row |

`hits` is always measured against Hebrew **Wikipedia**, including for Wiktionary
senses, because that is the corpus contexts would be drawn from.

## Known limitations

- **`hits` counts articles containing the phrase, not occurrences**, and does not
  account for Hebrew proclitics (`כמפקד מחלקה` does not match `"מפקד מחלקה"`).
  Treat it as a lower bound and a ranking signal, not a token frequency.
- **`looks_like_person` is a crude proxy.** What actually matters for building
  contexts is `hits`, which measures usability directly.
- **Parser precision is unmeasured** against a human reading, for both sources.
- Wiktionary senses are semi-structured free text; a trailing definition clause
  is trimmed heuristically.
- **`mine_substituted` filters honorifics inconsistently.** Every other gate calls
  `is_numeral_reference()`, which exempts the honorific letters `ר ד ע`; the substitution
  path calls `HEBREW_NUMERAL_REF_RE.search()` directly, so a sentence containing `ר׳`
  (rabbi) is dropped before substitution is attempted. Confirmed: for
  `"כתב על כך ר׳ משה…"` the helper returns `False` and the bare regex `True`. This
  **reduces recall in exactly the rabbinic articles the rarer acronyms live in**; it does
  not affect the correctness of rows already mined. Fix it, with a regression test, before
  any further network sweep.
- **`is_clean_sentence` does not enforce one occurrence of the target.** Its final check
  compares sets, so it rejects a different acronym type but keeps repeats of the target.
  See `data/splits/DATASET_CARD.md`.

