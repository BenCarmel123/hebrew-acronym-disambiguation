# Data

See the [dataset card](mined/DATASET_CARD.md) for construction, authorship and review
limits. Source exports remain separate from derived encoder inputs and saved responses.

| Retained input | Rows | Role |
|---|---:|---|
| [test_items.csv](splits/test_items.csv) | 381 | Scored test: 253 Knesset, 41 natural Wikipedia, 87 Claude-authored. |
| [train_items.csv](splits/train_items.csv) | 3,115 | Historical source for preparation, not proven historical training input. |
| [dev_items.csv](splits/dev_items.csv) | 289 | Historical development; the collection pilot uses its first ten rows. |
| [all_items.csv](splits/all_items.csv) | 4,649 | Aggregate export, not the union of the current splits. |
| [candidate_table.csv](mined/candidate_table.csv) | 2,702 | Candidate expansions and mining evidence; not per-sentence gold. |
| [knesset_reviewed.csv](mined/knesset/knesset_reviewed.csv) | 1,041 | Applied historical review, with document provenance. |
| [train.csv](study_v1/encoder_inputs/train.csv) | 2,829 | Stored qualified encoder training inputs. |
| [dev.csv](study_v1/encoder_inputs/dev.csv) | 62 | Curated natural encoder development items, 11 types/35 documents. |

`source`, `page_title`, `category` and `provenance` describe text origin/construction;
`label_origin`, `review_verdict` and `review_note` describe label evidence. Neither
`gold` nor `verified` proves independent validation. Candidates are pipe-separated;
`sense_id` is historical, not current rank. The old `multi_sense_type` mining flag is
blank for many later additions and cannot be inferred from distinct gold counts.

Legacy item IDs collide across files and sometimes within the aggregate. Join on
content and source references, not those IDs alone. The aggregate has 850 content
records beyond the current split union. Historical splits have disjoint exact types
and acronym/sentence pairs but overlapping document keys: 24 train/dev and 60
train/test, zero dev/test. Those 60 historical train/test protocols cover 97 test
items. Current encoder train/dev files have no non-empty document key in common
with the retained test. Historical checkpoint inputs remain unknown; neither
observation establishes actual training exposure. Exact-string type separation
also does not imply unseen acronym families: final-letter variants אג״ם/אג״מ
and מל״ם/מל״מ cross the split in seven test items, alongside prefix variants.
No absence from model pretraining is established.
Fourteen earlier test items from four overlapping Knesset protocols were removed;
[test_cohort.py](../src/hebrew_acronyms/test_cohort.py) preserves the exclusion list.
Saved 395-item collection records remain unchanged; scoring uses 381 items.
Gemini was collected directly on those 381 items; this does not apply to every
saved collection run.

## Candidate inventory and measurement limits

The 2,702 candidate rows comprise 1,685 Wiktionary, 270 Wikipedia, 47 both,
636 gematria and 64 human-review additions. Additions were made during Knesset
review, including test-related review. Thus 100% oracle coverage is coverage of
the completed inventory, not an independently complete untouched dictionary.
Intended mining filters avoided explicit definitions and additional acronyms;
retained data, later authored additions and observed exceptions do not universally
satisfy those filters. The 87 authored test sentences cover 29 types, of which
28 have retained Knesset test examples. The 221 confirmed and 32 corrected
Knesset test labels are recorded review categories: the original labels for the
32 are not recoverable, so verified before/after corrections cannot be asserted.

Three known reference errors remain frozen in the primary evaluation:
`kn4-0299` explicitly defines מח״ל as overseas volunteers, and `nat-0041` also
concerns those volunteers; both store the political-party expansion
`מפלגת חרות לאומית`. In `kn4-0207`, `דף ס״ב` means page 62, not the stored
`סודי ביותר`. A recorded clean/confirmed review is not proof of correctness.
A separate, post-hoc three-reference alternative, using original strict
selection parsing, gives encoder 274/381 and Qwen14 269/381, versus primary
273/381 and 266/381. The encoder is competitive with the tested local Qwen
configurations and close to Qwen14; this is not stable superiority. Suspected
עכו״ם variants remain unresolved; alternate-label scenarios indicate ranking
fragility rather than established corrections. No historical labels were changed.

A post-hoc wording diagnostic identifies 67/381 references with parentheses.
Across the six main LLMs, these yield 1/402 automatic generation matches and
339/402 selection matches: 402 response occurrences on 67 shared items, not
402 independent items. References include meta-descriptions such as
`ראשי תיבות שם פרטי ומשפחה (זהות חסויה)` and gematria wording. This documents
reference-wording/scoring mismatch; parentheses neither make matching impossible
nor establish semantic invalidity or a causal share of the generation–selection
gap. On the other 314 items selection still exceeds generation for all six
systems (Gemini: 294 versus 207). The full-cohort table remains primary; no
aggregate semantic-accuracy measure is inferred.

The most-mined reference remains 183/381 (48.0%). It is optimistic and
non-held-out: mining counts are influenced by test-containing data. The direct
count-equality check matches pairs for 32 of the 41 Wikipedia test items; the
original count producer is absent, so inclusion of every test sentence is not
proven. Count-subtraction diagnostics of 156/381 and 135/381 are counterfactuals,
not recovered leakage-free baselines, and do not replace the plotted reference.

Stored data-generation records identify `claude-sonnet-5` only for the records
it labels. The author's retrospective clarification identifies Claude Opus 4.6
through Claude Science for pre-proposal source discovery and feasibility work;
it is a distinct attribution. AI assisted scraping, but the scraping model
identity was not recovered. The [review guide](../saved-results/README.md)
separates reference-relative human judgments from contextual correctness and
preserves the historical exposure record alongside the author's clarification.

## Encoder preparation

The retained preparation produces 2,665 train and 62 dev items. Ben's later addition
of 164 training rows produced the stored 2,829: 136 authored rows and 28
repeated-target contexts truncated to one occurrence. The historical dataset
card's 142 authored rows describes its earlier file, not the current train file.
The earlier manifest/trace are archived; they do not document those later changes or prove the actual historical training inputs. Do not overwrite stored
train/dev to conceal this reconstruction limit.

Preparation reads [item_audit.csv](study_v1/review/item_audit.csv), the 16 populated
[review records](study_v1/review/review_decisions.csv) and exact
[local decisions](study_v1/encoder_inputs/local_decisions.json). Its access gate uses
audit roles/references before content; reserved test metadata supplies opaque exact
type/document identifiers. It does not open test text. Unknown membership stops export.
Training overlaps with all historical/proposed dev types/documents and authorized text
are blocked, including held dev rows. Missing documents, unresolved spans, adverse
decisions and duplicate text groups are held. Matching uses quote folding but retains
raw text and explicit half-open Unicode target offsets.

The 62 dev labels reuse 42 content/document-matched reviews attributed to Ben and
20 Shaked decisions with AI assistance. Approvals replace the label/existing candidate
in seven `בד״ר` items with `דוקטור` and eleven `רמב״ם` hospital/street items with
`רבי משה בן מימון`; the latter is a bounded linguistic-expansion decision. Two repeated
`א״ח` contexts select the first target; two quoted `רמב״ם` contexts retain their prefixes.
No global alias/type merge or new gold candidate is inferred. The separate approved
`ד״ר`/`בד״ר` separation holds eleven otherwise eligible training rows regardless of label.
Five dev types have one item and each type has one observed gold; this limits claims
about within-type contextual discrimination. At least two retained dev מט״ח
items have defective references; small dev is not proof of label correctness.
Earlier uncertainty survives unless explicitly superseded. The 18
`human_question` strings in the local policy are restored verbatim from commit
`6baa8f2`, including the rule against AI-only dev gold; decisions and other
fields are unchanged. This changes input/trace hashes, not prepared train/dev
rows; fresh receipts retain the actual hashes.

From the repository root, with the package installed, write a fresh external output:

```bash
ENCODER_OUTPUT=$(mktemp -d /tmp/hebrew-encoder-inputs-XXXXXXXX)
python -B -m hebrew_acronyms.data_processing.prepare_encoder_inputs \
  --train data/splits/train_items.csv --historical-dev data/splits/dev_items.csv \
  --audit data/study_v1/review/item_audit.csv \
  --decisions data/study_v1/review/review_decisions.csv \
  --policy data/study_v1/encoder_inputs/local_decisions.json --output "$ENCODER_OUTPUT"
python -B -m unittest tests.test_prepare_encoder_inputs -v
```

Outputs include train/dev, a separate singleton file, trace and manifest with input/code
hashes, applied decisions and reasons. Source references use one-based CSV records,
not physical lines. Stable identities bind source path, record and complete row hash.
Repeated preparation is deterministic; it starts from saved exports, not Internet mining.

## Collection tools

`python -m hebrew_acronyms.data_processing --help` lists collection commands:
`wikipedia`, `wiktionary`, `merge`, `flag-duplicates`, `apply-review`, `mine-sentences`,
`mine-by-sense`, `build-annotation-table`, `knesset-download`, `knesset-mine`.
Each accepts `--help`. Standalone `apply_dev_review` and `build_splits` also expose
module help; the interactive `review_duplicates_cli` instead takes a CSV path and
writes decisions to it (`q` leaves the current pair undecided).

New mining must produce or explicitly receive its intermediate inputs; removed
historical CSVs are not bundled dependencies. These tools may access live sources,
download corpora and write data. Their historical limitations remain: resume may
re-mine the last type or omit zero-yield completion; annotation input expects plain
UTF-8 while mining output has a BOM, which can cause `KeyError('acronym')`.
No live collection or review application is needed for saved-results analysis.
