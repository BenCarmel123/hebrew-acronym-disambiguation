# `data/splits/` — what training reads

**Layer 3 of the data lifecycle.** See [`../README.md`](../README.md) for all three layers.

The frozen splits an experiment actually consumes. Nothing else belongs here.

| File | Rows | Acronym types |
|---|---|---|
| `train_items.csv` | 2,978 | 494 |
| `dev_items.csv` | 289 | 55 |

## The split rule

**Disjoint by acronym type.** No acronym appears in both files. Splitting by row would
put the same acronym — often drawn from the same source article — on both sides, and a
model that had merely memorised its dominant expansion would score as though it had
learned to disambiguate. Type-disjointness makes dev measure generalisation to acronyms
never seen in training, which is the actual task. `pipeline/validate_data.py` enforces
this on every change — run it after editing either file.

Both files started as 90/10 slices of the `label_status=weak` rows in
[`../mined/acronym_items.csv`](../mined/acronym_items.csv), seeded for reproducibility,
then grew by the review and mining work below.

## Provenance

| | `train_items.csv` | `dev_items.csv` |
|---|---|---|
| `substituted` (original mining) | 2,966 | 281 |
| `deglossed` (new — see below) | 10 | 6 |
| `authored` (new — see below) | 2 | 2 |

`substituted` rows are the original mining output — the acronym inserted where a
sentence spelled the expansion out. Two later additions extend both splits:

- **`deglossed`** — mined from sentences that *use the acronym and gloss it inline*
  (`בא"ח (בסיס אימונים חטיבתי)`), with the parenthetical gloss stripped. This is
  genuine abbreviated usage, not a rewrite, and it attests senses substitution
  cannot reach — a phrase spelled out nowhere in the corpus outside its own gloss
  is unfindable by the original miner. Built by
  [`mine_deglossed`](../../data_preprocess/mine_sentences.py) and the sweep script
  that ran it; see `../mined/DATASET_CARD.md` for the yield and a known residual
  defect (proper names built from the acronym, e.g. a place named after a person,
  are not real gloss removals and were filtered out by hand).
- **`authored`** — a small number of hand-written sentences, added only where a
  documented real sense has **zero** corpus attestation even after deglossing
  (Hebrew Wikipedia simply never uses it in a matching context). Flagged so they are
  never mistaken for mined text; `label_status=verified` throughout, since the writer
  is also the labeller.

## Labels

Most rows are **weak-labelled**: the mining pipeline substituted the acronym into a
sentence that spelled the expansion out, so the label is correct by construction
rather than by human judgment. That is adequate training signal and is *not*
evaluation data on its own — a model scoring well on it may have learned the
substitution rule rather than disambiguation.

`dev_items.csv` additionally carries a full **human review** (2026-09-12): every
substituted dev row was judged `clean` / `wrong_sense` / `broken` / `unsure`, recorded
in `review_verdict` and `review_note` (raw pass in
[`../mined/dev_review.csv`](../mined/dev_review.csv)). 14 rows were relabelled from
reviewer corrections, 4 dropped as unrecoverable. Measured damage rate: 7.3%
(95% CI 4.1–10.6%) — see `../mined/DATASET_CARD.md` for the full breakdown, including
why the 38 `unsure` rows are a distinct, larger issue (mostly rabbinic-name types
where the sentence doesn't determine the answer at all) and were kept rather than
dropped. `train_items.csv` carries the same three review columns for schema
consistency, but every row there is `review_verdict=unreviewed` — train has not been
reviewed.

The 115 hand-verified `natural`-provenance rows live in
`../mined/acronym_items.csv` under `label_status=verified` and are still unused by
either split — see `../mined/DATASET_CARD.md` for why (type overlap with train). The
held-out test set is tracked outside this repository.

## What is not here

The full occurrence table and the candidate inventory are mining output and live in
[`../mined/`](../mined/). `acronym_items.csv` is the source these splits were drawn
from; `candidate_table.csv` is an input to mining, not a result. See
[`../mined/DATASET_CARD.md`](../mined/DATASET_CARD.md) for how both were built and
where they are known to be weak.
