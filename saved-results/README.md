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
- `local-runs/dictalm-20261010/` holds the untrained DictaLM 2.0 instruct (Q4_K_M)
  run through Ollama with seed 42, collected on all 395 items before the 14
  document-overlap items were removed, with its journals and manifests. The two
  `*_details.csv` files are review sheets with an empty `human_judgment` column;
  `scored_in_381` marks the scored items. The selection answers have the form
  "A. text", so the strict letter rule marks all invalid; `lenient_*` columns take the
  leading letter and are a diagnostic, not the reported score.
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

<a id="ben-dictalm-annotations"></a>

## Ben's DictaLM generation annotations

The byte-preserved [CSV](local-runs/dictalm-20261010/dictalm_generate_human_tags.csv)
was added by Ben in commit `b5cabe8ed749b5eff14de6cda48aa5ac838dc684`.
Its SHA-256 is `c6cfec23d97d9b2f29eac135454c4d609f19e6aba9280060a4c8ab406c51cbf0`.
It contains 381 unique scored test IDs with 193 `yes` and 188 `no` labels, only for
DictaLM generation run `c6cc64b4017547d884540a16af02a066`.

The [loader](../src/hebrew_acronyms/dictalm_saved_audit.py) verifies all golds and
responses against the saved generation CSV and the identified original journal.
There are 234 exact raw matches and 147 matches only after boundary whitespace
trimming. Raw response hashes always use the journal text, never the trimmed CSV.
Stable judgment identities make reloading idempotent. No labels transfer to other
models or to selection; the historical review exports are unchanged.

The documented rule is Ben's yes/no acceptance of the saved generation. The CSV has
no per-item rationale, detailed rubric, annotation timestamp or documented blinding;
these are not inferred. The automatic rule stays 125/381 and Ben's distinct result
is 193/381. Their cross-tab is 121 accepted by both, 72 by Ben only, 4 automatically
only and 184 by neither. Diagnostic leading-label selection stays 243/381 and strict
selection stays 0/381. The missing template newline remains a protocol limitation.

The original closed review remains 638 groups / 741 responses. The separate agent
assessments cover five groups / ten already-reviewed responses and add no human
decisions. Historical “resolved” wording in their immutable source records an agent
operational status only; human disagreements and pending flags remain unchanged.

The source inventory also references the existing
[similarity predictions](../results/dictabert/test_details.csv) by relative path,
commit and hash, without copying them. The 245/381 result is verified from saved
predictions, golds and candidate inventories. The source lacks a full sentence
snapshot and collection manifest; the verification does not reconstruct them.
