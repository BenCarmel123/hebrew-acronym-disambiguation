# `data/splits/` — what training reads

**Layer 5 of the data lifecycle.** See [`../README.md`](../README.md) for all seven layers.

The frozen splits an experiment actually consumes. Nothing else belongs here.

| File | Rows | Acronym types |
|---|---|---|
| `train_items.csv` | 2,966 | 491 |
| `dev_items.csv` | 292 | 55 |

## The split rule

**Disjoint by acronym type.** No acronym appears in both files. Splitting by row would
put the same acronym — often drawn from the same source article — on both sides, and a
model that had merely memorised its dominant expansion would score as though it had
learned to disambiguate. Type-disjointness makes dev measure generalisation to acronyms
never seen in training, which is the actual task.

Both files are 90/10 slices of the `label_status=weak` rows in
[`../interim/acronym_items.csv`](../interim/acronym_items.csv), seeded for
reproducibility.

## Labels

Every row here is **weak-labelled**: the mining pipeline substituted the acronym into a
sentence that spelled the expansion out, so the label is correct by construction rather
than by human judgment. That is adequate training signal and is *not* evaluation data —
a model scoring well on it may have learned the substitution rule rather than
disambiguation.

The 115 hand-verified rows live in `../interim/acronym_items.csv` under
`label_status=verified`, and the held-out test set is tracked outside this repository.

## What is not here

The full occurrence table and the candidate inventory are mining output and live in
[`../interim/`](../interim/). `acronym_items.csv` is the source these splits were drawn
from; `candidate_table.csv` is an input to mining, not a result. See
[`../interim/DATASET_CARD.md`](../interim/DATASET_CARD.md) for how both were built and
where they are known to be weak.
