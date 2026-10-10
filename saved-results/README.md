# Identified evaluation evidence

These files support the saved-result analysis in
[the main notebook](../notebooks/experimental_study.ipynb). They require neither
API credentials nor model weights. `files.json` records SHA-256, size and original
artifact location for each byte-preserved input; the notebook verifies these
hashes before comparison. Original local/Colab paths inside immutable records
identify collection provenance and are not required analysis paths.

- `colab-runs/` retains separate sessions, complete request/response journals and
  manifests, including failed pilots and the separately accounted HTTP diagnostic.
  The original allowance and carried expenditure are counted once, not by adding
  cumulative session totals.
- `study-runs/` contains fresh trained-encoder test predictions, input identities,
  loading evidence and the comparison with the separately preserved historical CSV.
- `encoder-test-inputs-20261009/` contains the one-to-one derived test input and its
  explicitly automatic target-position policy. These spans are not human labels.
- `human-review-*/` contains immutable answer queues and the two closed review
  exports. Each answer binds the original item, task, run and response hash.
  [Final review evidence](human-review-final-20261010/README.md) distinguishes new
  judgments, explicitly approved historical reuse and unreviewed answers.
- `collection-notebooks/` contains execution snapshots as provenance appendices.
  They document separate collection sessions and are not the analysis entry point.
  Their saved credentials are names only; actual secrets are not included.

The main notebook recomputes the unchanged task-specific automatic scores from
these responses and retains every test item, failure and incomplete answer.
Technical pilots are development evidence and are excluded from test scores.
Historical results under the repository's `results/` directory remain distinct.
Gemini collection is finished on 381 items in each task, with two truncated
generation responses retained. Its 20-answer pilot is separate. The original
collector status `incomplete` is preserved. The local two-call diagnostic is
accounted once before the new session: 21.6620734 ILS carried prior plus
1.465605 ILS new expenditure and allowances equals 23.1276784 ILS.
These are usage-based estimates and uncertainty allowances, not invoices.
The closed review windows cover 638 groups / 741 of 4,953 responses, including
Gemini. Coverage is partial: 707 newly judged occurrences and 34 approved historical
reuses. The original Gemini preparation bundle remains unchanged; the final review
uses the actual focused-window source with its original hash and exact bindings.

To repeat collection, use the identified collector revision in each manifest and
[the collection guide](../docs/reproducible_evaluation.md). To obtain the authorized
checkpoint for a separate inference run, see
[checkpoint retrieval](../docs/checkpoints.md#retrieving-the-identified-test-checkpoint).
Neither operation is necessary to reproduce this analysis.
