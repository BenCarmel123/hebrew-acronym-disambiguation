# P1 human data-review pilot

These are proposals, not approved model inputs. The human pilot has not been
performed. No elapsed times, human decisions, or agreement estimates are inferred.
Use `pilot_items.csv` for 16 deterministic dev/train or proposed natural-dev items;
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
measurements must be entered by Shaked; the current blank cells are intentional.

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
Only `data/study_v1/review` is written; existing decision bytes are preserved on repeat
runs with identical source hashes. Changed sources or orphan decision IDs stop the
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
