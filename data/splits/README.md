# `data/splits/` — what training and eval read

**Layer 3 of the data lifecycle.** See [`../README.md`](../README.md) for all three layers.

The frozen splits an experiment actually consumes. Nothing else belongs here.

| File | Rows | Acronym types |
|---|---|---|
| `train_items.csv` | 3,115 | 435 |
| `dev_items.csv` | 289 | 55 |
| `test_items.csv` | 395 | 60 |
| `all_items.csv` | 4,649 | 549 | every row above, one file, plus a `category` column |
| `by_category/*.csv` | — | — | the same rows, one file per `category` value |

## The split rule

**Disjoint by acronym type**, checked between every pair: train/dev, train/test,
dev/test. No acronym appears on both sides of any pair. Splitting by row instead would
put the same acronym — often drawn from the same source article or protocol — on both
sides, and a model that had merely memorised its dominant expansion would score as
though it had learned to disambiguate. Type-disjointness makes an eval split measure
generalisation to acronyms never seen in training, which is the actual task.
`pipeline/validate_data.py --train --dev --test` enforces this on every change — run it
after editing any of the three files.

`dev` is **frozen**: it was reviewed once (see Labels, below) and is never rewritten by
later work, including the test-split construction described next. Its 55 types are
excluded from every later allocation decision, so nothing can accidentally violate its
disjointness from train.

## test: why it needed an unusual construction

`test_items.csv` is a later addition (2026-09-13), built from human-reviewed Knesset
Corpus rows, previously-unused natural Wikipedia rows, and a small number of disclosed
AI-authored rows (see Provenance below) — the goal being a held-out set of **real**
(not mechanically rewritten) acronym usage, so it can measure something dev's
substituted-only rows cannot.

The complication: every one of the 191 human-reviewed Knesset acronym types turned out
to already be somewhere in the existing 549-type train+dev inventory — checked directly,
zero exceptions. So a genuinely type-disjoint test set could not be built by simply
adding new-source rows for some types while leaving `train` unchanged; the selected test
types' *existing* rows had to be actively removed from `train` and rebuilt from the
new-source pool instead. `data_preprocess/build_splits.py` does this — see its
docstring and `../mined/DATASET_CARD.md`'s "Knesset corpus, natural-text pool, and the
frozen test split" section for the full rationale, the stratification rule, and a known
accepted leakage condition (page-title overlap between train and test — Knesset
protocols are long multi-topic transcripts, so this is judged much lower severity than
the same condition would be for two Wikipedia articles).

## Provenance: the `category` column

Every row carries one of six values:

| category | train | dev | test | meaning |
|---|---:|---:|---:|---|
| `wiki_substituted` | 2,456 | 281 | 0 | Wikipedia, mechanically rewritten from a spelled-out phrase |
| `knesset` | 455 | 0 | 267 | Knesset Corpus, human-reviewed |
| `manual` | 142 | 2 | 87 | Disclosed AI-authored (`source=claude-sonnet-5`) |
| `wiki_natural` | 54 | 0 | 41 | Wikipedia, real unedited usage |
| `wiki_deglossed` | 8 | 6 | 0 | Wikipedia, real usage with an inline gloss removed |
| `wiktionary` | 0 | 0 | 0 | Placeholder — no such rows exist in this dataset |

`wiki_substituted` vs. `wiki_deglossed`: both start from Wikipedia prose, but
substitution *invents* the abbreviated form (a sentence that never used the acronym is
rewritten to use it), while deglossing finds a sentence where a Wikipedia author
**already used the acronym** and only removes a nearby parenthetical explanation — the
abbreviated usage itself is real, not manufactured.

`manual` rows exist for two disclosed reasons, both recorded in `review_note`:
- **Test-type disambiguation gap**: stratified test-type selection surfaced 29 types
  where every reviewed Knesset row shared one gold sense — no real disambiguation
  signal even with several rows. 87 sentences (3/type) were written using a different,
  already-cataloged candidate sense.
- **Train thin-sense gap**: 58 train types had no sense with 3+ examples at all. 142
  sentences were written to bring every sense up to 3+ (some real Wikipedia re-mining
  was tried first at deeper search depth; only the disclosed-authored route was
  actually merged — see the DATASET_CARD for what mining found).

Every `manual` row's gold sense is a real, pre-existing entry in `candidate_table.csv` —
these rows fill an attestation gap for a known sense, they do not introduce new senses.

## Labels

Most `wiki_substituted` rows are **weak-labelled**: the mining pipeline substituted the
acronym into a sentence that spelled the expansion out, so the label is correct by
construction rather than by human judgment. That is adequate training signal and is
*not* held-out evaluation data on its own — a model scoring well on it may have learned
the substitution rule rather than disambiguation.

`dev_items.csv` carries a full **human review** (2026-09-12) of every substituted row:
judged `clean` / `wrong_sense` / `broken` / `unsure`, recorded in `review_verdict` and
`review_note` (raw pass in [`../mined/dev_review.csv`](../mined/dev_review.csv)). 14
rows were relabelled from reviewer corrections, 4 dropped as unrecoverable. Measured
damage rate: 7.3% (95% CI 4.1–10.6%) — see `../mined/DATASET_CARD.md` for the full
breakdown, including why the 38 `unsure` rows are a distinct, larger issue (mostly
rabbinic-name types where the sentence doesn't determine the answer at all) and were
kept rather than dropped.

`knesset` rows carry `label_status=verified` and `review_verdict` values from a
from-scratch human review against each type's candidate list (not an audit of a
mechanical label) — see `../mined/knesset_reviewed.csv` for the raw review and the
DATASET_CARD for the full verdict breakdown (941 clean, 100 corrected, 4 dropped as
genuinely unresolvable).

## What is not here

The full occurrence tables and the candidate inventory are mining output and live in
[`../mined/`](../mined/). `acronym_items.csv` (Wikipedia) and `knesset/knesset_reviewed.csv`
(Knesset Corpus) are the sources these splits were drawn from; `candidate_table.csv` is
an input to mining, not a result. See [`../mined/DATASET_CARD.md`](../mined/DATASET_CARD.md)
for how all of it was built and where it is known to be weak.
