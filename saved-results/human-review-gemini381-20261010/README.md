# Gemini human-review add-on

`review-data.json` contains 762 original answer occurrences from test run
`6658d323657f4df3aced950bfe6abb07`, session
`evaluation-gemini-381-20261010-79b767a`, collection revision
`79b767a3f04bfadc9536d505b11f73ac70a99b5c`. The pilot is not queued.
Paths in this bundle resolve relative to `saved-results/`. File hashes, complete
response text and bindings preserve item, task, run and response identity.

| Task | Occurrences | Automatic positives | Complete nonpositives | Technical failures | New judgments | Approved reuse | Unreviewed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Generation | 381 | 207 | 172 | 2 | 0 | 0 | 381 |
| Selection | 381 | 357 | 24 | 0 | 0 | 0 | 381 |

This is an **inactive add-on**, not a replacement for the current annotator input.
All nonpositive answers and positive controls remain available; the two technical
failures are separate from semantic disagreements. The existing schema's optional
calibration queue does not authorize a new timed window. No judgment, draft or
clock is created here. Together with the existing 4,191 scored occurrences, the
combined inventory contains 4,953 occurrences; that is not human-review coverage.

Before actual integration, finish the existing labeling window and verify an
export and backup against their hashes and answer bindings. Preserve the existing
input and decisions. Do not infer that a missing export means nobody has labeled
answers; this release reports only evidence bundled with it.

Reuse may be proposed only when the original item, sentence, target, task and exact
answer match; selection also requires identical decoded answer and option mapping.
An explicit approval is required before transfer. Retain the original judgment,
annotator, timestamp and exposure, the reuse approval and its time, and all source
and destination run bindings. No reuse has been proposed or applied in this bundle.
All other answers remain unreviewed, and drafts do not count as completed review.

Regenerate the add-on offline with `test_review_export.export_review`, passing
the identified full-test directory and run ID, a fresh output path and
`source_root=Path("saved-results")`. This operation does not launch the annotator
or send model requests. The main notebook displays its coverage without loading it
into the live application. Existing historical judgments remain at their sources.
