# P1 human data-review pilot

These are proposals, not approved model inputs. Current registration counts and
recorded timings are derived from decision rows in `diagnostic_summary.json`;
registration does not establish that a review was validated or scientifically approved.
Use `pilot_items.csv` for the deterministic pilot and `pilot_coverage.csv` for quota
coverage (4 natural-dev, 4 train Wikipedia natural, 4 routine reviewed literal train
Knesset, 4 central-risk cases; shortages are reported and filled where possible);
it contains no historical test item. Use `item_audit.csv` and source references for
evidence, and record all decisions in `review_decisions.csv` using `review_id`.

For each item, start a timer and inspect the sentence before consulting its recorded
label. Record a defensible literal expansion or `uncertain`; uncertainty is retained.
Then inspect the recorded label and evidence. Enter label_decision (`accept`, `correct`,
`uncertain`, `exclude_proposed`) and corrected_label only when justified. Select the
intended target occurrence and enter zero-based, end-exclusive character offsets in
the unchanged raw sentence. Check attached prefixes, quote variants, repeated targets,
and an explanation in the sentence. A proposed span is not an annotated span.

Review the type inventory independently of this sentence: follow candidate/source
export references, Ben's recorded review and raw provenance. Do not add an answer
solely because it is the sentence label; do not invent a distractor or require two
expansions. Distinguish possible meanings from observed label strings. Review numeral
and concealed-identity exclusions as proposals. Alias equivalence needs an evidence
reference and must apply consistently to the type, not just one item. Record
inventory decisions on the corresponding `inv-...` review_id. Existing source values
and Ben's decisions are evidence and are never overwritten.

Complete reviewer, decision, reason, evidence_reference and decided_at, plus
elapsed_seconds and problem_types (pipe-separated label/target/inventory/provenance/
document/definition/other). A held/uncertain case remains open. Decisions and pilot
measurements must be entered by Shaked; blank cells mean no value has been recorded.
A partial answer or timing alone counts as an entry, not complete registration.


## Decision field dictionary

Blank means pending or not applicable, never implicit acceptance. Values below are
case-sensitive. Required fields describe the human recording contract; this generator
preserves values and reports presence/timing only, and does not apply or validate
scientific decisions. Complete registration is explicitly separate from complete
adjudication. Record an unresolved case rather than force a choice.

| Field | Values, applicability and conditional requirements |
|---|---|
| `review_id` | Existing `item-...` or `inv-...` identity; never edit. Inventory and item decisions have separate rows. |
| `reviewer` | Human name; required for a recorded decision. |
| `decision` | `accept`, `correct`, `alias`, `merge`, `exclude_proposed`, `uncertain`; overall disposition. Item rows use accept/correct/exclude_proposed/uncertain. Inventory rows may use all six. `uncertain` remains unresolved. |
| `initial_blind_expansion` | Item only: free-text independent expansion or literal `uncertain`, recorded before viewing stored label/candidates/evidence; required for a newly reviewed item. If already exposed, leave blank and disclose exposure in `reason`; never backfill a blind answer. |
| `label_decision` | Item only: `accept`, `correct`, `exclude_proposed`, `uncertain`; required when the label was reviewed. |
| `corrected_label` | Item only: exact proposed replacement label; required iff label_decision=`correct`, otherwise blank. This does not add a candidate automatically. |
| `target_decision` | Item only: `accept`, `correct`, `missing`, `uncertain`, `not_reviewed`. `accept` confirms the proposed occurrence, `correct` selects a different occurrence/boundary, `missing` states no defensible target, `uncertain` leaves it open. Required for a newly reviewed item. |
| `approved_span_start`, `approved_span_end` | Item only: nonnegative integers, zero-based Unicode code-point offsets, end-exclusive, with end>start within unchanged sentence; both required when target_decision is `accept` or `correct`, both blank otherwise. Names are retained for compatibility; recording them does not approve the benchmark. |
| `inventory_decision` | Inventory only: `accept`, `correct`, `alias`, `merge`, `exclude_proposed`, `uncertain`; must match overall decision. For an item, leave blank and use linked inventory rows. |
| `canonical_expansion` | Inventory only: exact proposed canonical wording. Required for accept/correct/alias/merge. `correct` changes canonical wording without silently identifying two senses; reasons and independent evidence are required. |
| `approved_aliases` | Inventory only: JSON array of exact alias strings, e.g. `["fictional spelling"]`; required and nonempty for `alias`, optional for accept/correct, blank for merge/exclusion/uncertain. An alias is an equivalent form for this canonical sense, not a separate competing sense. |
| `merge_target_inventory_id` | Inventory only: existing, different `inv-...` ID for the same acronym type; required only for `merge`. Set canonical_expansion to that target's proposed canonical wording. A merge proposes unifying two inventory entries; it is not an alias spelling declaration. Never invent an ID. |
| `evidence_reference` | Source path and one-based CSV record, stable reference, or exact review ID; required for accept/correct/alias/merge/exclude_proposed. For uncertain, cite what was checked where available. A sentence label alone cannot validate inventory membership. |
| `reason` | Free-text rationale, required for every recorded decision; explain corrections, exposure, exclusions, uncertainty and relations to separately recorded inventory decisions. |
| `decided_at` | ISO 8601 timestamp with timezone, required for a recorded decision. |
| `elapsed_seconds` | Finite nonnegative number of seconds actually spent on this row; required for timed pilot registration. Never infer elapsed time from a timestamp or fill it for the reviewer. |
| `problem_types` | Pipe-separated subset of `label`, `target`, `inventory`, `provenance`, `document`, `definition`, `other`, or `none` alone. Required for a completed pilot entry; uncertainty still receives its relevant issue categories. |

On an item, overall accept requires accepted label and target; correct records at
least one correction. Overall exclude_proposed/uncertain leaves the relevant question
open for approval/adjudication. On an inventory row, overall and inventory dispositions
agree. Every proposal remains pending application to a future benchmark.

Invented example, not a research decision: sentence `הסמל א״ב הופיע.` has target
[5,8). For fictional item `item-EXAMPLE`, record initial_blind_expansion=`אור בהיר`,
reviewer=`Example Reviewer`, decision=`correct`, label_decision=`correct`,
corrected_label=`אור בהיר`, target_decision=`accept`, approved_span_start=`5`,
approved_span_end=`8`, evidence_reference=`fictional-source.csv record 1`,
reason=`Recorded label differed from independently supported expansion`,
decided_at=`2026-09-28T12:00:00+03:00`, elapsed_seconds=`42`, problem_types=`label`.
Leave its inventory fields blank. Separately, fictional inventory `inv-EXAMPLE-A`
may propose `merge` into existing same-type `inv-EXAMPLE-B`, with canonical_expansion
=`אור בהיר` and an independent evidence reference/reason. An `alias` proposal instead
keeps this inventory ID and supplies an explicit JSON alias list; no merge ID is set.
Example IDs and evidence are explanatory and must not be entered in live decisions.

Before reading stored labels in `pilot_items.csv`, view only its sentence and acronym
columns and record the initial answer in `review_decisions.csv`. This CSV workflow
cannot enforce blindness; any premature exposure must be disclosed rather than
claimed absent. Then reveal the remaining fields and inspect the audit evidence.

Selection is deterministic: prefer types not already selected across strata. Within
natural/risk strata, maximize newly covered flags, then prefer less repeated
within-stratum flag profiles before higher risk score; routine
Knesset uses lower risk score. Stable item ID breaks ties. The risk quota allows at
most one proposed nonliteral exclusion and one historical-adverse case. Reported
fallback fills quota shortages from the eligible pool with the same diversity/risk
ranking. No test item is eligible. This is an intentionally stratified review pilot,
not a representative error estimate or approval of any research sample.

Existing decision values, unknown extra columns and row order survive additive schema
migration. New fields start blank. Invalid/ambiguous CSV schemas, orphan IDs or changed
sources stop generation; an unchanged expanded schema preserves decision file bytes.
Unknown columns are retained without assigning them a meaning.

After the pilot, summarize median and 75th-percentile seconds by risk/source where
sample sizes permit, issue categories and unresolved fraction. Estimate remaining
work from stratum counts times measured review times, adding measured adjudication
and inventory overhead. This risk-enriched pilot is not representative; report wide
ranges and do not invent precision for unobserved strata. Time limits do not replace
quality gates. Review proposed natural dev, critical flags and Wikipedia without
proven review, then a fixed sample of low-risk Knesset. Final sampling and closure
require Shaked. No inter-annotator agreement is available. The full queue is a searchable universe
of item and source-hypothesis proposals, not a requirement to review every row;
Shaked approves the required scope and unresolved cases remain open.

Reproduction: from the repository root, use the installed package:
`.venv/bin/python -m hebrew_acronyms.data_processing.prepare_study_review --repo .`.
Only `data/study_v1/review` is written; existing decision values are preserved through additive schema migration, and
bytes are preserved on repeat runs with the same expanded schema and source hashes. Changed sources or orphan decision IDs stop the
run for explicit reconciliation. CSV source references count records, not lines.

Local environment caveat observed during P1: the existing editable installation's
`.venv/lib/python3.12/site-packages/__editable__.hebrew_acronym_disambiguation-0.1.0.pth`
intermittently acquires the macOS `hidden` flag, causing `ModuleNotFoundError`.
If this recurs, inspect that exact file with `ls -lO`; when `hidden` is present,
clear only that flag with `chflags nohidden` on the same file immediately before
running the command. An offline editable reinstall used `PIP_NO_INDEX=1` and
`--no-deps --no-build-isolation`; no dependency or model downloads were performed.
The source of the recurring flag was not diagnosed. This local workaround is not
a verified fresh-environment installation or a change to the research protocol.

Extension scenarios and membership are inspectable availability proposals, never
active training sets. The cap uses at most five rows/type, at least two documents,
and document round-robin ordered by source, page title, historical ID and sentence
hash. No gold or score selects these rows. Coverage is described after selection.
Source/row/candidate-pair matching is necessary, not sufficient: after inventory and
span review, recompute both full pair totals and use the same batch, epochs,
accumulation, drop-last, seed and checkpoint policy. No schedule is executed here.
