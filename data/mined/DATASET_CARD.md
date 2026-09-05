# Dataset card — Hebrew acronym disambiguation

How the benchmark data was obtained, what it contains, and where it is known to
be weak. Mining is complete: all 638 acronym types with two or more candidate
expansions have been swept. What remains is annotation.

**Status: unlabelled.** No item has a human-verified gold expansion yet. Nothing
here is evaluation data. This directory (`data/splits/`) also holds the
frozen benchmark once annotation exists — see that directory's `README.md` for
why an unlabelled snapshot and the frozen benchmark are not the same thing and
must never be confused for one another.

## Task

Given a Hebrew sentence containing an acronym, predict the expansion. Two arms:
free generation, and selection from a candidate list. Only acronym types with
two or more valid literal expansions are in scope.

## Source

Hebrew Wikipedia, via the MediaWiki action API (`he.wikipedia.org/w/api.php`),
namespace 0 only. Sentences come from `prop=extracts&explaintext` — the page's
rendered plain text — never from search snippets.

That choice is load-bearing. An earlier module mined `list=search` snippets,
which are fixed-width excerpts cut from anywhere on a page: reference lists,
tables, image captions, infoboxes. Roughly 32% were usable after ten successive
filters, and the failures were not fixable by more cleaning, because the text
was never prose to begin with. Fetching the article's own plain text solved it
at the source. That module was removed; do not reintroduce snippet mining.

Wiktionary supplies candidate expansions and a small number of lexicographer-
written usage examples.

## Pipeline

`src/data/data_preprocess/`, driven by `python -m data_preprocess <command>`.

### 1. Candidate expansions → `candidate_table.csv`

Acronym disambiguation pages give the candidate expansions; each was counted
for Wikipedia hits (`count_bullets`), merged with Wiktionary sense counts
(`count_wiktionary`, `merge`), and near-duplicate expansions were flagged and
resolved by a human through `review_duplicates_cli`. (Three commit hashes cited here
previously — `caffc54`, `6f89c3b`, `5f27b78` — do not exist in this history; they
predate a rebase. The decisions themselves are recorded in
`data/mined/duplicate_review.csv`.) Latin-script types and the single-letter `א'`
were dropped.

2,264 rows over 680 acronym types.

### 2. Sentence mining → `data/mined/*_by_sense.csv`

Command: `mine-by-sense --acronyms <types.txt>`. Three strategies exist in
`mine_sentences.py`; the mining runs use the latter two.

**`mine_sentences`** (flat, not used for the benchmark) searches the acronym
surface and keeps clean sentences. Fully natural, but it returns whichever
sense dominates the corpus and records no sense at all. On an early run it produced
239 sentences in which every single `מ"מ` example meant מילימטר — the type
looked productive at 10 sentences and was useless for disambiguation.

**`mine_by_expansion`** searches for pages carrying the acronym *and* a given
expansion, giving each sense its own page pool. Text stays natural and a
provisional sense is attached. Yield is low: 2 of 17 expansions for `מ"מ`.

**`mine_substituted`** finds sentences with the expansion spelled out and
rewrites it to the acronym, preserving any attached proclitic
(`לממלא מקום` → `למ"מ`). Yield is far higher — 14 of 17 for `מ"מ` — because a
full form is ordinary Hebrew with none of the acronym's search problems, and
the sense is known by construction. See *Limitations*.

## Filters

Applied to every sentence, whatever the strategy, and re-applied to substituted
text after rewriting:

- **Length** 40–200 characters.
- **Acronym present as a whole token.** Substring matching is wrong: `ב"ש`
  occurs inside `ב"שירות`, which is a proclitic plus a quoted word. The match
  may not be followed by more Hebrew letters, and may only be preceded by a
  proclitic (ה, ו, ב, כ, ל, מ, ש).
- **Exactly one acronym.** Any second acronym-shaped token disqualifies the
  sentence.
- **No inline gloss, in either direction.** Both `ח"ש (חודר שריון)` and
  `חודר שריון (ח"ש)` put the answer in the input. A *spaced* dash also glosses
  (`ד"ש – התנועה הדמוקרטית`), while a tight hyphen joins a compound name
  (`ש"ס-העבודה`) and is kept — the spacing is the signal, not the dash.
- **No explicit gloss phrase** — `ראשי תיבות`, `קיצור של`, `נוטריקון`.
- **No non-prose residue** — URLs, image extensions, table markup, template or
  link braces.
- **No Hebrew-numeral chapter reference** (`פרק ב'`, `משניות י' - י"ג`). The
  honorific `ר'` (Rabbi) uses the same geresh and is exempted, since it is
  ordinary prose and frequent in exactly the rabbinic articles where the rarer
  acronyms occur.

Substitution additionally requires that the source sentence does *not* already
contain the acronym, and is not defining the abbreviation
(`ממלא מקום (בראשי תיבות: מ"מ)`), which would rewrite to `מ"מ (בראשי תיבות: מ"מ)`.

## Exclusions

- Acronym types with fewer than two candidate expansions.
- Latin-script acronym types; the single-letter type `א'`.
- Candidate rows whose "expansion" itself contains an acronym — `סיכת מ"מ`,
  `מפלגת אח"י`. These are phrases *named after* the acronym, not readings of
  it, so there is nothing to substitute into. 230 such rows remain in
  `candidate_table.csv` and are skipped at mining time; they still inflate
  `n_candidates` and should be dropped from the table.

## Coverage

`acronym_items.csv` — 3,386 items over 546 acronym types, after removing 24
exact `(acronym, sentence)` duplicates. Every acronym type with two or more
candidate expansions has been swept: 638 selected, 546 produced at least one
sentence, 92 produced none.

| | count |
|---|---|
| items | 3,386 |
| acronym types | 546 |
| types with ≥2 senses (≥2 rows each) | 295 |
| items in those types | 2,664 |
| natural / substituted | 128 / 3,258 |

**Only 295 of 546 types support disambiguation.** A type needs two senses with
at least two examples each to be usable at all, and the rest fall short —
overwhelmingly the low-ranked types, where the second sense is barely attested
in the corpus. The `multi_sense_type` column marks the ones that clear the bar,
covering 2,664 of the 3,386 items. The remainder cannot support a
disambiguation comparison and should be excluded from evaluation, though they
stay usable as training material.

**The data is 96% synthetic.** Natural acronym usage yielded 128 items against
3,258 from substitution, and the gap widened as mining moved down the hit-count
ranking. Types were selected by `hits`, and for many of them those hits are
prefix collisions rather than acronym occurrences: `ב"ש` reports thousands that
are almost all `ב"שלום` / `ב"שיטת`. Substitution is unaffected, because it
searches the spelled-out expansion; natural mining is not. Hit count remains a
poor proxy for real yield.

## The two files, and how they differ

`processed/` holds two files that describe the same acronyms at different
levels. Confusing them is easy and consequential, because one contains
expansions that are *not* labels.

| | `candidate_table.csv` | `acronym_items.csv` |
|---|---|---|
| one row per | acronym × candidate expansion | sentence |
| size | 2,002 rows, 638 types | 3,386 items, 546 types |
| answers | what *could* this acronym mean? | what does this sentence mean? |
| built from | Wikipedia disambiguation pages + Wiktionary | mining |
| expansions are | **hypotheses** — dictionary senses, mostly unverified against text | one provisional sense per sentence |

**`candidate_table.csv` is the input, not a result.** It lists every expansion a
disambiguation page or Wiktionary offers, whether or not the corpus supports it.
762 of its 2,002 rows produced no items at all — those senses exist in a
dictionary and effectively not in Hebrew Wikipedia. Its `mined_items` and
`mined_natural` columns record how many sentences each expansion actually
yielded, which is the honest measure; the older `hits` column counts string
matches and is a poor proxy (`ב"ש` reports thousands of `ב"שלום`). Use the table
for the *candidate list* an item offers the model, never as evidence that a
sense is real.

**`acronym_items.csv` is the data itself** — one row per sentence. Its
`candidates` column is copied from the candidate table, so it deliberately
includes unattested expansions: a model choosing among candidates should face
the full dictionary-plausible set, not a set pre-filtered by what we managed to
mine. Its `sense_id` column gives each (acronym, expansion) pair a stable
identifier — ids are assigned by descending item count, ties broken on the
expansion text, and **must not be renumbered once annotation starts**, since a
gold label refers to a sense id rather than to a position in a list.

A consequence worth stating: `n_candidates` counts what the model chooses among,
while the number of distinct `sense_id` values for a type counts what the corpus
actually attests. These disagree for most types, and that is correct.

## Limitations

**Substitution dominates the data and is not natural text.** A writer who
spelled a phrase out might not have abbreviated it in that position, so
substituted items can read stiffly or carry thinner context than genuine
abbreviated usage. Two failure modes recur:

- expansions that are *common word sequences* rather than fixed terms
  (`מכל מקום`) match text where the words were not functioning as that unit —
  `מכל מקום בו הרכב נמצא` is compositional "from any place where", and the
  rewrite produces nonsense;
- expansions that are *fragments of longer terms* (`מרחב מכפלה`, from
  `מרחב מכפלה פנימית`) rewrite the fragment and leave a dangling remainder.

Fixed terminology (`מפקד מחלקה`, `מכונאי מוטס`, `ממלא מקום`) substitutes
cleanly. A manual spot-check of one type suggested roughly a third of
substituted rows are damaged; **this rate has not been measured across the
data.** Automated flagging caught 5 of 311 rows in a spot-check and should not be
relied on.

**Every `expansion` value is provisional.** For substituted rows it is correct
by construction, which makes it useless as a training target — a model scoring
well on it has learned the substitution rule, not disambiguation. For natural
rows it means "the source page also mentioned this expansion", which is
evidence, not proof: a page about ממלא מקום can still use `מ"מ` for something
else. Gold labels require human annotation.

**Sense distribution is skewed and not corrected.** Some acronyms are
overwhelmingly one sense in real text. Per-expansion mining gives each sense
its own quota, which surfaces rare senses but does not reflect their true
frequency.

**Single source.** Everything is Hebrew Wikipedia — encyclopaedic register
only. Hebrew Wikisource was validated as a fallback for rabbinic and
liturgical types that Wikipedia lacks prose for, but is not wired up.

**No split.** Deliberate: splitting before labelling would bake unverified data
into the test set.

## Reproducing

```
cd project
.venv/bin/python3 -m src.data.data_preprocess mine-by-sense \
    --acronyms data/mined/retry_types.txt \
    --out data/mined/retry_by_sense.csv \
    --per-expansion 3 --pages-per-expansion 12
```

Output is streamed and flushed per row, so an interrupted run keeps whatever it
had already written. The API client throttles between calls and backs off on
429/5xx.

## Columns

`acronym_items.csv`:

| column | meaning |
|---|---|
| `item_id` | stable id, `ha-00001` … |
| `acronym` | target acronym, gershayim form |
| `sentence` | input text; contains the acronym. **Not always exactly once** — 3,260 rows hold one occurrence under mark-folding, 126 hold two or more (see Known limitations) |
| `provisional_expansion` | machine-assigned sense — **not a gold label** |
| `sense_id` | stable id for the (acronym, expansion) pair |
| `gold_expansion` | empty; the human-verified label |
| `label_status` | `unverified` for every row |
| `n_candidates`, `candidates` | choice set for the constrained arm, ` \| `-separated |
| `provenance` | `natural` or `substituted` |
| `multi_sense_type` | `yes` if the type has ≥2 senses with ≥2 rows each |
| `source`, `page_title` | provenance of the text |

Raw mining output stays in `data/mined/*_by_sense.csv`.

## Known limitations found during model-axis integration (2026-08-29)

Recorded here because they affect how this snapshot may be used, not to diminish it.

- **A sentence may contain the target more than once.** `is_clean_sentence` rejects a
  *different* acronym type but compares a **set**, so repeats of the target itself
  survive. Measured on this snapshot under mark-folding: 3,260 rows with one occurrence,
  **126 with two or more** (up to seven). The annotated occurrence is not recorded, so for
  those rows the target is undetermined. Changing the miner cannot repair rows already
  committed here; downstream conversion quarantines them rather than guessing.
- **Quote variants.** The `acronym` column is always gershayim (U+05F4) while sentences
  use gershayim, ASCII `"` or both. An exact search misses 77 rows that are plainly
  present; a mark-folding search finds every one. Consumers must match leniently and keep
  the original bytes.
- **`sense_id` is not the current rank.** Only 1,662 of 3,386 equal
  `{acronym}.{rank:02d}` — the ids were assigned against an earlier ordering, before the
  near-duplicate review re-ranked the table. They are stable and collision-free and must
  be preserved, never re-derived.
