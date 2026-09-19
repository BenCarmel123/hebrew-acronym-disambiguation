# Data inventory and interpretation

These files preserve existing research evidence. Organization does not approve their
labels, split rules or inclusion in the next benchmark. Counts below were read directly
from CSVs at `63b90acfae36e6b7fee8114760506868e29c681f` on 2026-09-19; no data were rebuilt.
[Construction and review history](mined/DATASET_CARD.md) records how they arose.

## Locations

| Layer | Files | Role |
|---|---|---|
| Source exports | `mined/wikipedia/`, `mined/wiktionary/`, `mined/knesset/knesset_mined.csv`; locally downloaded shards under ignored `raw/` | Source-specific evidence. Original API responses were not systematically retained; live retrieval does not guarantee reconstruction. |
| Candidate inventory | [candidate_table.csv](mined/candidate_table.csv), `mined/merged_counts.csv` and summaries | Possible expansions and mining signals; an input to mining, not per-sentence gold. |
| Mined items | [acronym_items.csv](mined/acronym_items.csv) | Wikipedia contexts with provisional labels and provenance. |
| Review evidence | [duplicate_review.csv](mined/duplicate_review.csv), [dev_review.csv](mined/dev_review.csv), [merge_review.csv](mined/merge_review.csv), [unmined_triage.csv](mined/wikipedia/unmined_triage.csv), [knesset_reviewed.csv](mined/knesset/knesset_reviewed.csv) | Recorded corrections, verdicts and applied review output. Preserve the records and their attribution. |
| Historical inputs | [train_items.csv](splits/train_items.csv), [dev_items.csv](splits/dev_items.csv), [test_items.csv](splits/test_items.csv) | Existing allocation for training/development/test; not a newly approved protocol. |
| Aggregate exports | [all_items.csv](splits/all_items.csv), [by_category/](splits/by_category/) | Separate artifacts, not interchangeable copies of the split union. |

Later deglossed/authored additions did not all pass through the original mining tables.
The retained construction scripts do not alone reproduce every committed addition.
Invented engineering inputs live in [tests/fixtures/](../tests/fixtures/), outside research data.

## Current contents

| File | Rows | Acronym types |
|---|---:|---:|
| `splits/train_items.csv` | 3,115 | 435 |
| `splits/dev_items.csv` | 289 | 55 |
| `splits/test_items.csv` | 395 | 60 |
| `splits/all_items.csv` | 4,649 | 550 |
| `mined/candidate_table.csv` | 2,702 | 638 |
| `mined/acronym_items.csv` | 3,386 | 546 |
| `mined/knesset/knesset_reviewed.csv` | 1,041 | 191 |

| Text category | Train | Dev | Test | Meaning |
|---|---:|---:|---:|---|
| `wiki_substituted` | 2,456 | 281 | 0 | A spelled-out expansion was replaced with the acronym. |
| `knesset` | 455 | 0 | 267 | Observed parliamentary text with recorded review. |
| `manual` | 142 | 2 | 87 | Authored text; authorship must be read from `source` and `label_origin`. |
| `wiki_natural` | 54 | 0 | 41 | Observed Wikipedia acronym usage. |
| `wiki_deglossed` | 8 | 6 | 0 | Observed usage with an adjacent explanation removed. |
| `wiktionary` | 0 | 0 | 0 | Candidate source; no items in these splits. |

All 142 train and 87 test `manual` rows declare `claude-sonnet-5` / `claude_authored`.
The two dev rows declare `human` / `human_authored`. The aggregate holds 233 authored
rows: 229 declaring Claude and four declaring human authorship. Its category export
`by_category/manual.csv` contains only the 229 Claude rows. Do not silently synchronize them.

Under the content key `(acronym, sentence, gold_expansion, candidates)`, the aggregate
contains 850 rows beyond the split union: 510 substituted, 319 Knesset, 17 natural,
two deglossed and two human-authored. This comparison does not imply identical metadata.
The category exports other than `manual` match the aggregate under that content key;
`wiktionary.csv` is header-only. These differences explain why the exports are retained.

## Fields and label evidence

| Fields | Interpretation |
|---|---|
| `item_id`, `acronym`, `sentence` | Row identifier, target type and context. IDs are not globally unique across files; a sentence can contain repeated targets. |
| `source`, `page_title`, `provenance`, `category` | Source/document and text construction, separate from label reliability or split role. |
| `provisional_expansion`, `gold_expansion`, `label_status`, `label_origin` | Mechanically proposed or recorded target and its stated origin. A field named `gold` or `verified` is not independent validation. |
| `review_verdict`, `review_note` | Review decisions and explanations. Shaked attributes the human review work to Ben; these fields do not establish two independent annotations. |
| `sense_id` | Historical identifier for an acronym–expansion pair, not the current candidate rank. Do not renumber it. |
| `candidates`, `n_candidates` | Offered expansions, separated by ` \| `, and their count. Candidate hypotheses need not all be attested in the corpus. |
| `multi_sense_type` | Historical mining flag: originally ≥2 senses with ≥2 rows each. It is blank for many later additions and cannot be inferred merely from distinct gold counts. |

A natural sentence can have an uncertain label; reviewed text can be substituted;
authored text is not observed usage. Neither filenames nor categories settle these questions.

## Structural limitations

- Acronym types and exact `(acronym, sentence)` pairs are disjoint across the three splits.
  Documents are not: nonempty `(source, page_title)` overlap is 24 for train/dev and 64
  for train/test. This is a fact about the stored files, not approval of that split rule.
- Seven IDs occur in both train and dev for different content: `ha-03333`, `ha-03334`,
  `ha-03335`, `ha-03337`, `ha-03344`, `ha-03349`, `ha-03355`. The aggregate also has seven
  duplicated IDs. Do not join these files on `item_id` alone.
- Candidates, ambiguity and label evidence vary across items. Existing review decisions,
  including uncertain cases, remain preserved; changes require a scientific task.
- [Historical predictions](../results/all_arms_summary.md) refer to an older dev snapshot.
  They must not be joined to current rows or cited as current scores without reconciliation.

See [data processing commands](../data_preprocess/README.md) for code responsibilities
and retained execution limits. Preserve source exports and review work even where the
original retrieval or full reconstruction is unavailable.
