# Data lifecycle

> **Open this to find out where a file belongs.**

Data moves through three layers, and the one-way arrow is the point: **1 → 2 → 3**.
Nothing skips a layer.

| # | Layer | Where | What it is |
|---|---|---|---|
| 1 | **External sources** | not in this repository | Hebrew Wikipedia, Wiktionary, and the [Knesset Proceedings Corpus](https://huggingface.co/datasets/HaifaCLGroup/KnessetCorpus), under their own licences |
| 2 | **Mined candidates** | [`mined/`](mined/) | Machine-extracted acronym types, candidate expansions and contexts. Weak labels — **not gold labels** — except the human-reviewed Knesset rows, which are |
| 3 | **Splits** | [`splits/`](splits/) | The frozen train/dev/test files an experiment actually reads |

API responses are not retained: re-running `data_preprocess` fetches them again. The
mined tables are committed instead, because a full sweep is thousands of throttled calls
over hours.

## The distinction that matters

**A weak label is not a gold label.** `mined/acronym_items.csv` carries a
`provisional_expansion` for every row, produced by the rule that built it — the pipeline
substituted the acronym into a sentence that spelled the expansion out, so the label is
correct *by construction*, not because a person judged it.

That is adequate training signal and it is not evaluation data. A model scoring well on
substituted text may have learned the substitution rule rather than disambiguation. The
`label_status` column keeps the two separable:

| status | meaning |
|---|---|
| `weak` | Correct by construction, from substitution |
| `verified` | Judged by a person |
| `unverified` | Neither — left unlabelled rather than guessed |

The splits carry a `category` column with six values beyond plain substitution:
`wiki_substituted`, `wiki_deglossed` (real sentences that named the acronym and glossed
it; the gloss was stripped rather than the acronym inserted), `wiki_natural` (real,
unedited Wikipedia usage), `knesset` (human-reviewed Knesset Proceedings Corpus rows),
`manual` (a disclosed small number of AI-authored sentences, `source=claude-sonnet-5`,
used only to fill a specific attested-sense gap — never passed off as mined), and
`wiktionary` (currently a header-only placeholder — no such rows exist). See
`splits/README.md` for counts per split and `mined/DATASET_CARD.md` for how each
category was built.

## What lives where

**[`mined/`](mined/)** — everything `data_preprocess` produced:

| File | What |
|---|---|
| `acronym_items.csv` | 3,386 Wikipedia acronym occurrences with candidate lists — the corpus |
| `candidate_table.csv` | Acronym → expansion inventory, 638 types. An **input** to mining, not a result |
| `wikipedia/bullet_counts.csv` | Wikipedia disambiguation-page candidates with corpus hit counts |
| `wiktionary/wiktionary_counts.csv` | The same from Wiktionary senses |
| `merged_counts.csv` | Union of the two sources |
| `duplicate_review.csv` | Near-duplicate expansions and the human merge decisions |
| `dev_review.csv` | Human judgment (2026-09-12) of every substituted row in `splits/dev_items.csv` |
| `knesset/knesset_mined.csv` | 1,045 raw Knesset Proceedings Corpus mining hits, unlabeled |
| `knesset/knesset_reviewed.csv` | 1,041 of those, human-reviewed against each type's candidate list |
| `DATASET_CARD.md` | How all of it was built, every filter, and the known limitations |

**[`splits/`](splits/)** — `train_items.csv`, `dev_items.csv`, `test_items.csv`,
disjoint by acronym type between every pair. See that directory's README for why the
split is by type rather than by row, and for how test's construction differs from a
simple 90/10 slice.

Eval outputs (per-item predictions, accuracy tables) are **not** a fourth layer here —
they are a result of reading layer 3, not an input to any layer, so they live in
[`results/`](../results/) at the repo root instead.

## Reading a count from any of this

`mined/DATASET_CARD.md` documents where the data is weak — that it is still
overwhelmingly substituted rather than natural, that only a minority of acronym types
have two senses attested well enough to support disambiguation at all, and what a full
human review of the dev split found: a measured 7.3% substitution damage rate (down
from an earlier one-type guess of "roughly a third"), plus several failure modes the
review surfaced that were not previously documented. Read it before quoting any
number.
