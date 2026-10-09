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
| Aggregate exports | [all_items.csv](splits/all_items.csv), [by_category/](splits/by_category) | Separate artifacts, not interchangeable copies of the split union. |

Later deglossed/authored additions did not all pass through the original mining tables.
The retained construction scripts do not alone reproduce every committed addition.
Invented engineering inputs live in [tests/fixtures/](../tests/fixtures), outside research data.

## Current contents

| File | Rows | Acronym types |
|---|---:|---:|
| `splits/train_items.csv` | 3,115 | 435 |
| `splits/dev_items.csv` | 289 | 55 |
| `splits/test_items.csv` | 381 | 60 |
| `splits/all_items.csv` | 4,649 | 550 |
| `mined/candidate_table.csv` | 2,702 | 638 |
| `mined/acronym_items.csv` | 3,386 | 546 |
| `mined/knesset/knesset_reviewed.csv` | 1,041 | 191 |

| Text category | Train | Dev | Test | Meaning |
|---|---:|---:|---:|---|
| `wiki_substituted` | 2,456 | 281 | 0 | A spelled-out expansion was replaced with the acronym. |
| `knesset` | 455 | 0 | 253 | Observed parliamentary text with recorded review. |
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
  Documents are not: nonempty `(source, page_title)` overlap is 24 for train/dev and 60
  for train/test (0 for dev/test). This is a fact about the stored files, not approval of that split rule.
- Seven IDs occur in both train and dev for different content: `ha-03333`, `ha-03334`,
  `ha-03335`, `ha-03337`, `ha-03344`, `ha-03349`, `ha-03355`. The aggregate also has seven
  duplicated IDs. Do not join these files on `item_id` alone.
- Candidates, ambiguity and label evidence vary across items. Existing review decisions,
  including uncertain cases, remain preserved; changes require a scientific task.
- [Historical predictions](../results/all_arms_summary.md) refer to an older dev snapshot.
  They must not be joined to current rows or cited as current scores without reconciliation.

See [data processing commands](../docs/data_processing.md) for code responsibilities
and retained execution limits. Preserve source exports and review work even where the
original retrieval or full reconstruction is unavailable.

## Qualified encoder inputs (1 October 2026)

[Encoder inputs](study_v1/encoder_inputs/) are derivatives of the saved exports,
prepared under Shaked's explicit authorization to use existing training labels,
including weak labels. Technical qualification does not turn those labels or all
negative candidates into independently verified truth. Sources and the historical
review decisions remain unchanged. Project-manager acceptance and authorization
for training are separate next steps.

| Derivative | Items | Types | Documents | Candidate pairs |
|---|---:|---:|---:|---:|
| [train.csv](study_v1/encoder_inputs/train.csv) | 2,829 | 434 | 1,864 | stale — see below |
| [dev.csv](study_v1/encoder_inputs/dev.csv) | 62 | 11 | 35 | 331 |
| [dev_singletons.csv](study_v1/encoder_inputs/dev_singletons.csv) | 0 | 0 | 0 | Not a pair-loss input |

**train.csv update (3 October 2026, Ben):** grew from 2,665 to 2,829 rows after
reviewing a subset of the rows `manifest.json` had held. 164 rows were released
back into train: 126 Claude-authored thin-sense-gap rows held only for a
missing document key that doesn't apply to them (no source document by
construction); 10 rows with a resolved span-boundary issue (an attached Hebrew
prefix letter); and 28 natural `target_multiple` rows (the acronym repeated in
one sentence) truncated to a single occurrence plus trailing context, removing
the ambiguity by construction. Left held, unchanged: 93 substituted
`target_multiple` rows and 6 further span-ambiguous rows. `manifest.json`
(hashes, candidate-pair count, full per-row decision trace, the paragraph
below) still reflects the pre-release 2,665-row state and needs to be
regenerated by Shaked's pipeline script, not hand-edited — the row/type/
document counts above are a directly-counted correction, everything else in
this section is pending regeneration.

Training contains 2,316 substituted, 341 natural (294 Knesset, 47 Wikipedia) and
8 deglossed items. Of 3,115 training source rows, 448 are held and 2 have recorded
human exclusions. Nonexclusive reasons include 142 missing document keys (the
historically declared AI-authored rows), 121 repeated targets, 30 ambiguous target
boundaries, 114 reserved-document overlaps, 60 development-document overlaps,
2 duplicate-text records, 1 development-text overlap, 1 unresolved human label
and 11 holds under the approved `ד״ר`/`בד״ר` separation rule.
Reasons overlap; do not add them to obtain an exclusion total. All 62 proposed
natural development rows are retained. The 289 historical dev rows are reference
only and are not substituted for this development set. No quotas from earlier
proposals were adopted.

The explicitly approved `ד״ר`/`בד״ר` separation holds all 11 otherwise eligible
training `ד״ר` rows, regardless of their label. This split-only decision is stored
separately from label decisions and changes no item content or target span.
No general prefix removal or additional type-family merging is applied.

The 62 dev items cover 11 types and 35 documents, with one observed gold answer
per type; five types have just one item. This supports limited development but
does not alone establish context-dependent discrimination between different
senses of the same acronym.

The preparation uses each row's **stored candidate inventory**, with only the
explicit corrections below; it does not adopt the proposed global inventory or
a historical dev type-level union. Development labels reuse exact sentence,
type, label, candidate and document matches to Knesset review attributed to Ben,
or explicit Shaked decisions. A clean flag alone is insufficient. Of the final
dev labels, 42 retain that historical human evidence and 20 use Shaked's saved
AI-assisted decisions (18 new corrections and 2 earlier acceptances). This is a
curated development sample, not blind reannotation or evidence of annotator agreement.

Shaked approved these item-bounded changes in this preparation chat:

- For all 7 proposed dev `בד״ר` items, replace the metalinguistic candidate
  `ד"ר (ב- קליטי)` and label with `דוקטור`.
- For all 11 proposed dev `רמב״ם` items, use `רבי משה בן מימון` as gold and replace
  the existing abbreviated candidate `ר' משה בן מימון` with its full form. This
  accepts expansion of the proper name in these hospital/street contexts; it is a
  bounded clarification of the task, not a general entity or etymology policy.
- Select the first target in the two repeated `א״ח` voting sentences, and retain
  the full quoted tokens `ב"רמב"ם"` / `וב"רמב"ם"` in two prefixed contexts.

No candidate was added to make a positive-label check pass, no type was merged,
and no source sentence was rewritten. [Local decisions](study_v1/encoder_inputs/local_decisions.json)
contain exact input snapshots, proposals, rationale/evidence, assistant identity
as known, questions and verbatim human approvals. The independent code reviewer
is not a human annotator. Earlier pilot uncertainty/exclusion records remain
visible; the later item-bounded approvals supersede uncertainty only where explicit.

### Qualification and reproduction

The [preparation module](../src/hebrew_acronyms/data_processing/prepare_encoder_inputs.py)
filters mixed audit records by roles and source references **before** exposing
content. It uses reserved exact type/document identifiers from audit metadata;
it does not open `test_items.csv`, inspect test text/labels/candidates or rerun the
review generator. Unknown role/reference metadata stops export. Train and dev
are checked against reserved identifiers; training is also blocked against all
historical and proposed dev types/documents, including held rows. Duplicate text
groups are held without selecting a preferred label. Missing document keys are held.

Targets require a unique quote-folded match with unambiguous word boundaries or
an explicit saved human span. Quote folding only locates a target; output preserves
the exact sentence slice, including approved prefixes. Letters outside surrounding
quotes also require a boundary decision. Candidates and labels must satisfy the
unchanged encoder pair contract. Any otherwise-qualified singleton dev would be
exported separately for description/prediction, never pair-loss selection.

Run from the repository root using an environment with this package installed:

```bash
python -B -m hebrew_acronyms.data_processing.prepare_encoder_inputs \
  --train data/splits/train_items.csv \
  --historical-dev data/splits/dev_items.csv \
  --audit data/study_v1/review/item_audit.csv \
  --decisions data/study_v1/review/review_decisions.csv \
  --policy data/study_v1/encoder_inputs/local_decisions.json \
  --output data/study_v1/encoder_inputs
python -B -m unittest tests.test_prepare_encoder_inputs -v
```

Use another output directory to compare a fresh reconstruction. No LLM call or
model load is needed. [manifest.json](study_v1/encoder_inputs/manifest.json) records
source hashes, the baseline commit, preparation/contract code hashes, rules,
applied decisions, population/source/reason counts and output hashes. Code hashes
identify the exact implementation independently of later documentation commits.
[trace.csv](study_v1/encoder_inputs/trace.csv) records every authorized source row,
its disposition, raw label/candidates and metadata, content-linked review references
and decisions. Review evidence references include hashes of the matched saved
records; the immutable audit retains their full payloads.

IDs bind the source path, one-based CSV data record and complete source-row hash;
legacy IDs alone are never join keys. Natural dev starts at the saved audit row,
which retains the original aggregate reference. A record number is not a physical
line number. Repeated runs with the same sources, decisions and code produce the
same bytes. Reproduction starts at retained exports, not original Internet mining.

Separation uses exact historical types and source/title document keys, plus the
explicitly approved `ד״ר`/`בד״ר` pair.
Document aliases, linguistic type families, test-text duplicate comparisons and
pretraining exposure are not resolved. The pair contract passed without model
loading; tokenizer length checks and Colab training belong to the next package.
