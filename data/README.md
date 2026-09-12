# Data lifecycle

> **Open this to find out where a file belongs.**

Data moves through three layers, and the one-way arrow is the point: **1 → 2 → 3**.
Nothing skips a layer.

| # | Layer | Where | What it is |
|---|---|---|---|
| 1 | **External sources** | not in this repository | Hebrew Wikipedia and Wiktionary, under their own licences |
| 2 | **Mined candidates** | [`mined/`](mined/) | Machine-extracted acronym types, candidate expansions and contexts. Weak labels — **not gold labels** |
| 3 | **Splits** | [`splits/`](splits/) | The frozen train/dev files an experiment actually reads |

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

## What lives where

**[`mined/`](mined/)** — everything `data_preprocess` produced:

| File | What |
|---|---|
| `acronym_items.csv` | 3,386 acronym occurrences with candidate lists — the corpus |
| `candidate_table.csv` | Acronym → expansion inventory. An **input** to mining, not a result |
| `bullet_counts.csv` | Wikipedia disambiguation-page candidates with corpus hit counts |
| `wiktionary_counts.csv` | The same from Wiktionary senses |
| `merged_counts.csv` | Union of the two sources |
| `duplicate_review.csv` | Near-duplicate expansions and the human merge decisions |
| `DATASET_CARD.md` | How all of it was built, every filter, and the known limitations |

**[`splits/`](splits/)** — `train_items.csv` and `dev_items.csv`, disjoint by acronym
type. See that directory's README for why the split is by type rather than by row.

Eval outputs (per-item predictions, accuracy tables) are **not** a fourth layer here —
they are a result of reading layer 3, not an input to any layer, so they live in
[`results/`](../results/) at the repo root instead.

## Reading a count from any of this

`mined/DATASET_CARD.md` documents where the data is weak — that it is 96% substituted
rather than natural, that only 295 of 546 acronym types have two senses attested well
enough to support disambiguation at all, and that a manual spot-check suggested roughly a
third of substituted rows may read awkwardly. Read it before quoting any number.
