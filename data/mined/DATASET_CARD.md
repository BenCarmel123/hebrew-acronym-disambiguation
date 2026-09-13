# Dataset card — Hebrew acronym disambiguation

How the benchmark data was obtained, what it contains, and where it is known to
be weak. Mining is complete: all 638 acronym types with two or more candidate
expansions have been swept. What remains is annotation.

**Status: dev reviewed, train unlabelled.** Every substituted row of the dev split
has been judged by a human (2026-09-12); the verdicts are in `dev_review.csv` and the
result is the reviewed `data/splits/dev_items.csv`. The original 2,966 train rows
remain weak-labelled — correct by construction, never judged. See *Substitution
damage, measured* below for what the review found, and *Deglossed and authored rows*
for two smaller additions made after the review, both in the splits rather than here.

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
cleanly. An early spot-check of one type suggested roughly a third of substituted
rows were damaged. **That guess was too pessimistic** — see below.

### Substitution damage, measured

All 285 rows of the dev split were reviewed by hand (2026-09-12), each judged
`clean` / `wrong_sense` / `broken` / `unsure`. Raw verdicts: `dev_review.csv`.

| verdict | rows |
|---|---|
| clean | 227 |
| wrong_sense (words matched, but not as that term) | 11 |
| broken (dangling fragment or mangled grammar) | 7 |
| unsure | 38 |
| unreviewed | 2 |

**Damage rate: 7.3% (18 of 245 decided), 95% CI 4.1–10.6%** — four to five times
lower than the one-type guess. Dev is 100% substituted, so this measures the
substitution strategy directly.

Two qualifications matter more than the headline:

- **The 38 `unsure` rows are not damage.** 29 of them are rabbinic-name types
  (`מהרי״א`, `מהר״ש`, `מהרי״ץ`, `מהר״י`) whose candidates are different rabbis
  sharing initials. The sentence often does not determine which — `למד בעיקר אצל
  מהר״ש` names no distinguishing detail. These are a *knowledge* task rather than
  a context task, and they flatter models with memorised biographical detail. They
  are kept, flagged in `review_verdict`, not silently dropped.
- **The reviewer found failure modes this card did not document.** Three are new:
  an expansion used as a **proper name** (`דו״ד` is a radio programme named after
  `דין ודברים`, not a reading of it); a **candidate table defect**, where
  Wiktionary supplied a *description* instead of an expansion (`יעב״ץ` →
  `מבעלי התוספות`, "one of the Tosafists", where the expansion is `יעקב בן צבי`);
  and **zero-evidence candidates** — 49 candidate slots in dev name a sense with
  0 corpus hits and 0 mined items, leaving 17 rows whose choice is between one
  real option and a dictionary ghost.

14 rows were relabelled from the reviewer's corrections and 4 dropped as
unrecoverable, leaving dev at 281 reviewed rows (283 after two `בא״ח` rows described
below). Corrected expansions were added to the candidate list of **every** row of
that acronym type, not only the corrected rows, so the new option is a genuine
distractor where it is wrong rather than a marker of which rows were reviewed.
`label_origin` records `substitution` vs `human_review`; `review_verdict` and
`review_note` carry the judgment.

Automated flagging caught 5 of 311 rows in an earlier spot-check and should not be
relied on. The review above supersedes it.

### Deglossed and authored rows

Two more provenances were added to both splits after the review, in a separate pass
(2026-09-12), neither counted in the damage-rate measurement above since they are
new rows rather than judgments of existing ones.

**`deglossed`.** `is_clean_sentence` drops any sentence that glosses the acronym
inline (`בא"ח (בסיס אימונים חטיבתי)`), since the answer would sit in the input. But
that is also the one place the corpus proves a rare sense is real *and* shows a
writer actually abbreviating it. `mine_deglossed()` in `mine_sentences.py` finds such
sentences and strips the parenthetical, keeping genuine abbreviated usage with a
known label — more natural than substitution, which invents the abbreviation rather
than finding one. A full sweep over the 368 expansions in `candidate_table.csv` with
corpus hits but zero mined items returned 18 usable rows over 12 acronym types (5%
yield): 6 installed in dev under a new type, `להב״ה` — two real, unrelated
organizations sharing the acronym, a genuine disambiguation case the corpus never
attested before — and 12 installed in train, 10 filling out existing types and 2
introducing new ones. One row was discarded by hand: a `מס״ב` gloss that named a
*place* after a person rather than reading the acronym, the same proper-name defect
documented above for `דו״ד`. `NAMED_AFTER_RE` in `mine_sentences.py` now filters the
common phrasing of this automatically for future runs.

**`authored`.** Two documented real senses — `בא״ח` = `בסיס אימונים חטיבתי`
("brigade training base") and `אז״ר` = `אויב זרק רימון` (a military radio warning,
"enemy threw a grenade") — have zero corpus attestation even after deglossing: every
Wikipedia sentence containing the phrase glosses it, and the grenade-warning sense is
spoken slang unlikely to appear in encyclopedic prose at all. Four sentences were
hand-written to attest these two senses (2 each, in dev for `בא״ח` and train for
`אז״ר`), `label_status=verified` since the writer is also the labeller.
`review_note` says so on every such row; they should never be read as mined text.

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
    --acronyms data/mined/wikipedia/retry_types.txt \
    --out data/mined/wikipedia/retry_by_sense.csv \
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

## Knesset corpus, natural-text pool, and the frozen test split (2026-09-13)

Everything above this section describes the original Wikipedia/Wiktionary
pipeline and the dev review that followed it. This section documents a
substantially larger change: adding a second real-world text source, pulling
in previously-unused natural Wikipedia usage, and — for the first time —
constructing a genuinely held-out **test** split, disjoint from both `dev`
and `train`.

### Why: dev alone could not answer the project's core question

Every row in `train`/`dev` up to this point was `substituted` or
`deglossed` — Wikipedia text either mechanically rewritten from a spelled-out
phrase, or edited to remove an inline gloss. A `substituted` sentence was
*written about* its answer before the acronym was inserted into it, so
context clues can leak the sense without any real disambiguation happening.
This directly bears on the question the project proposal was asked to answer
(course/mor_feedback.md): does a big LLM already solve this task from
context? On flattered, substituted-only data, "yes" can look true regardless
of whether it would hold on real usage. `dev` was reviewed and found to be
94% `weak`-labelled by construction — useful for iteration, but not a fair
final measurement.

### New source: Knesset Proceedings Corpus

`data_preprocess/knesset/` mines `HaifaCLGroup/KnessetCorpus`
(huggingface.co), a public, ungated dataset of ~35M pre-segmented sentences
from Knesset plenary and committee protocols, 1992–2024. Unlike Wikipedia
mining, sentences here need no page-to-sentence splitting — each shard's
`protocol_sentences` are already clean units — so only the acronym/prose
filters in `common/filters.py` apply. Register is formal parliamentary
speech: acronyms recur in genuinely disambiguating context (a speaker uses
`בע"מ` or `ש"ח` the way a reader must actually resolve it from the debate),
unlike Wikipedia's encyclopedic prose.

All 1,000 available plenary shards were downloaded and scanned (`--per-acronym
15` cap) against the full 549-type inventory: **1,045 rows mined, 193 types
attested** (committee protocols — ~9,000 further shards — were evaluated and
explicitly not pursued; see "Scope decisions" below). Every mined row was
manually reviewed by the user (via a purpose-built HTML labelling tool,
`db`-capability-backed for save/resume) against each acronym's existing
candidate list:

| Verdict | Rows |
|---|---:|
| clean (gold already correct) | 941 |
| human_review (corrected — see below) | 100 |
| dropped (unresolvable) | 4 |
| **total usable** | **1,041** |

The 4 dropped rows are genuine defects, not disagreements: two mis-tokenized
clitic-attachment artifacts (a sentence about "מ.י." parsed as if `ב"מ` were
the acronym; a mangled newspaper name parsed as an acronym), one
unit-confusion transcription error ("750 מ"מ" for a wine bottle, where the
source clearly meant מ"ל), one garbled/unparseable token ("ב"ב"ה").

**Two systematic gaps found during review, fixed at the candidate-table
level rather than row-by-row:**

- **Gematria.** Every 2-letter Hebrew acronym is *also* a valid gematria
  number (ת"ק = 500, כ"א = 21, ח"י = 18), and Knesset protocols cite bill
  numbers, page/verse references, and dates this way constantly.
  `hebrew_text.gematria_value()`/`gematria_value_hebrew()` were added and a
  gematria candidate was generated for **all 638** `candidate_table.csv`
  types in one pass, not just the ones a first review round happened to hit
  — the phenomenon is general, not type-specific.
- **Privacy redaction.** A large share of Knesset protocol acronyms turn out
  to be two-initial placeholders for a named person whose full name was
  withheld (from Knesset's own committee-privacy conventions) — `א"א`,
  `ד"א`, `מ"ס`, `ע"מ`, `ש"ב`, and many more, confirmed by cross-referencing
  the surrounding sentence's syntax (a title like "מר"/"גברת" immediately
  before the acronym). A generic candidate,
  `"ראשי תיבות שם פרטי ומשפחה (זהות חסויה)"`, was added to every type where
  this pattern was confirmed, rather than treating each as a one-off.

**Known recurring mining defect, caught by manual review only:** the mining
tokenizer cannot distinguish a genuine standalone acronym that happens to
start with a clitic letter (מ/ב/ו/כ/ל/ש/ה/ד) — e.g. `בע"מ`, `מד"א`, `דמ"צ`
are real, independent acronyms — from an actual מ-/ב- clitic prefix
attached to a *different* real acronym (`מד"ר`/`בד"ר` are both just
`ד"ר` with a prefix; `בר"מ` is `ר"מ` with one). Two such cases were caught
and corrected during review; a general regex fix was considered and
rejected as unreliable without the full type inventory as context —
this remains something a reviewer must catch by eye.

### Candidate-table consolidation

A near-duplicate scan across all `candidate_table.csv` expansions per type
(normalized string similarity) found 3 genuine accidental duplicates —
all self-inflicted by this session's own additions restating a sense
Wiktionary already had, differently formatted (`ח"י`, `ל"ב`'s gematria
entries; `י"ל`'s "יצחק לייבוש" name). These were merged, with the 17
already-reviewed rows pointing at the newer duplicate repointed to the
canonical form. ~39 other superficially-similar pairs the same scan found
(`ר"א` "רבי אלעזר" vs. "רבי אליעזר", `גב"ש` "גבעת שמואל" vs. "גבעת שאול", …)
are genuinely distinct senses that only look alike in spelling — left
untouched; merging them would have been a correctness error, not a cleanup.

### wiki_natural: previously-unused natural Wikipedia rows

`data/mined/acronym_items.csv` has always carried 128 `provenance=natural`
rows (real, unedited Wikipedia acronym usage, mined by `mine_by_expansion`)
that never made it into any split. 124 of these have an attested acronym
type; of those, 4 types were already in `dev`'s frozen type set and were
dropped (dev already owns those types for evaluation; adding the same type
elsewhere would blur what each split measures). A further 12 rows carry
`label_status=unverified` with no `gold_expansion` at all — a pre-existing
gap, not introduced here — and were dropped after confirming every affected
type (`ע"ש`, `ר"מ`, `מהר"ם`, `אב"י`, `ר"י`, `ש"ש`) still has other rows with
valid gold elsewhere. **112 wiki_natural rows were added to `train`.**

### manual: authored rows disclosed by source

Stratified test-type selection (below) surfaced 29 test types where every
reviewed Knesset row shared the same gold sense — no real disambiguation
signal, even with several rows. For each, one alternative candidate already
present in `candidate_table.csv` was chosen and **3 Hebrew sentences per
type (87 total) were written by Claude (model: claude-sonnet-5)** using that
alternative sense, disclosed here and in the data itself
(`provenance=authored`, `source=claude-sonnet-5`,
`label_origin=claude_authored`). These are the only rows in this dataset
that were not mined from a real corpus; every gold sense was checked against
`candidate_table.csv` before being accepted (0 mismatches after fixing an
ASCII-quote-vs-gershayim key bug during construction). **Every one of the 29
target test types now has ≥2 distinct gold senses attested in `test`.**

### The `category` column and the six-way provenance breakdown

`data/splits/all_items.csv` (every row, all sources, one file) and
`data/splits/by_category/*.csv` (one file per bucket) add a `category`
column distinguishing:

| category | rows | meaning |
|---|---:|---|
| `wiki_substituted` | 3,247 | Wikipedia, mechanically rewritten from a spelled-out phrase |
| `knesset` | 1,041 | Knesset Corpus, human-reviewed |
| `wiki_natural` | 112 | Wikipedia, real unedited usage |
| `manual` | 87 | Claude-authored (see above) |
| `wiki_deglossed` | 16 | Wikipedia, real usage with an inline gloss removed |
| `wiktionary` | 0 | **placeholder — no such rows exist in this dataset.** Wiktionary supplies *candidate expansions* and was used to mine lexicographer usage examples earlier in the project, but zero such rows survived into any split. `by_category/wiktionary.csv` is written with a header only, so the gap is visible rather than silently absent. |

`wiki_substituted` vs. `wiki_deglossed`: both start from Wikipedia prose, but
substitution *invents* the abbreviated form (a sentence that never used the
acronym is rewritten to use it), while deglossing finds a sentence where a
Wikipedia author **already used the acronym** and only removes a
parenthetical explanation next to it — the abbreviated usage itself is real,
not manufactured. Deglossed rows are therefore closer in kind to natural
usage than to substitution, despite both starting as edited Wikipedia text.

### The frozen test split

`test_items.csv` (395 rows) is built from the Knesset + wiki_natural + manual
pool, and is **type-disjoint from both `train` and `dev`** — checked
directly. This required an unusual step: every one of the 191 reviewed
Knesset types turned out to already be in the existing 549-type train+dev
inventory (checked, zero exceptions), so a disjoint test set could not be
built by simply adding new-source rows for some subset of types while
leaving `train` unchanged — the selected test types' *existing* Wikipedia
rows had to be actively removed from `train`. `data_preprocess/build_splits.py`
does this: 60 types (35 stratified by ambiguity/frequency + the 29
manual-forced types) were carved out of `train`'s existing 514 rows for
those types and rebuilt from the new-source pool instead; `train`'s
remaining contribution from the pool is capped at 8 rows/type so a few
easy/common types (`ד"ר`, `בע"מ`, `רש"י`, …) don't dominate.

`dev` was **not modified** — it stays exactly the file the earlier
substitution-damage review produced. Its 55 types are excluded from all
test/train allocation decisions in `build_splits.py`.

**Known, accepted leakage channel: page-level overlap.**
`pipeline/validate_data.py` now also checks `page_title` overlap across
splits (previously only `sentence` and acronym-type were checked). This
surfaced 24 shared page_titles between `train`/`dev` (pre-existing — short
Wikipedia articles, real if minor topical leakage) and **64 shared
page_titles between `train`/`test`** — Knesset protocol transcripts, not
Wikipedia articles. The latter is judged much lower severity: a Knesset
protocol is a single long multi-topic session (42% of scanned protocols
mention more than one acronym type), so two different acronym mentions
from the same protocol share far less real content than two sentences
from the same short Wikipedia article would. Accepted as a known condition
rather than engineered around, given the cost of enforcing document-level
disjointness on an already-thin stratified type pool.

### `multi_sense_type` is not populated for new rows

This field (see Columns, above: "`yes` if the type has ≥2 senses with ≥2
rows each") was checked directly against a simpler hypothesis — "does this
type have more than one distinct `gold_expansion` anywhere in the file" —
and found NOT equivalent (40/289 mismatches on `dev`), meaning it encodes a
human annotation call made when `train`/`dev` were first built, not
something mechanically re-derivable from the data alone. It is left blank
for every `knesset`, `wiki_natural`, and `manual` row rather than guessed.

### Scope decisions made and not revisited

- **Committee protocols** (~9,000 further shards, a much larger corpus than
  the 1,000 plenary shards used) were listed and one partial download
  attempted, but not pursued — a deliberate capacity call given the review
  volume already produced by plenary alone.
- **Sefaria** (rabbinic/Talmudic text, intended to cover the ~38 dev rows
  marked `unsure` because they're rabbinic-name types unanswerable from
  Wikipedia context) was built and tested (`data_preprocess/sefaria/`) but
  abandoned: real yield was ~3% of search hits after full-text fetch, far
  below Knesset's yield, because Sefaria's per-ref `he` text segments are
  not reliably sentence-granular the way a Wikipedia plaintext extract or a
  Knesset Corpus pre-segmented sentence is. The client code remains in the
  repo (exploratory, not wired into any pipeline) in case a future session
  finds a workable extraction strategy.

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
