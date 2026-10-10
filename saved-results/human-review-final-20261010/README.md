# Closed human-review evidence

The [main notebook](../../notebooks/experimental_study.ipynb) reconstructs 638
reviewed groups / 741 answer occurrences out of 4,953: 707 newly judged occurrences
and 34 explicitly approved historical reuses. This is partial diagnostic review,
not benchmark-wide semantic accuracy.

The immutable [first export](../human-review-381-20261010/review-export.json) and
[focused export](../human-review-focus-200-20261010/review-export.json) preserve all
620 new decision events and historical approval provenance. Two later `fits`
rechecks supersede earlier `unsure` judgments only in the aggregate; both events
remain available. No judgment extends beyond its recorded response bindings.
The focused export's coverage denominator is its selected 200 groups / 239
occurrences. The combined analysis restores all 381 items per system and task.

`combined-review-coverage.json` is the original closing audit. Its totals are
independently reconstructed, not adopted as new labels. `selection-manifest.json`
records mechanical prioritization; it is not human judgment. The receipt explains
the final backup's export hash: only `annotation_sha256` differs from the working
export referenced by the audit, not any decision or coverage value.

`reuse-context-evidence.json` is a minimal derived projection of the 20 historical
contexts and their current counterparts. Original source hashes were verified
before projection. The first export retains each original judgment, time,
annotator, exposure, approval time and destination bindings. Live application
state, access tokens, drafts and operational logs are not distributed. Both exports
preserve the complete confirmed decision history needed for this analysis.

All listed backup files were verified against `backup-sha256.json`; only the
necessary safe evidence is bundled. `saved-results/files.json` pins the included
bytes. Absolute paths inside original records describe provenance; execution uses
only bundled relative paths. No private artifacts, server or weights are required.

Interpretation must retain diagnostic prioritization, unknown/prior exposure and
the negative default with explicit table confirmation. Unreviewed responses are
not implicitly approved. Nine positive-score/rejection occurrences across four
items remain pending clarification, with labels unchanged. See the
[exported cases](../../results/submission-20261010/clarification_cases.html).
