# Saved evidence

The [main notebook](../notebooks/experimental_study.ipynb) reconstructs the analysis
without models, weights or API credentials. [files.json](files.json) identifies
112 sources by size and SHA-256, including the existing similarity predictions
referenced outside this directory. Historical paths and hashes identify the
original collection; runtime loading uses the retained local sources.

| Sources | Role |
|---|---|
| `colab-runs/` | 97 session identities, run manifests, request/response journals and diagnostic records; failures and historical attempts remain separate. |
| `study-runs/dictabert-test-20261009-f49c3b5/` | Original encoder inputs, predictions, run identity and checkpoint reconstruction evidence. |
| `encoder-test-inputs-20261009/` | Automatic first-occurrence policy; these spans are not human annotations. |
| `local-runs/dictalm-20261010/` | Original DictaLM session, pilot/test manifests and raw journals, including all 395 collected items. |
| `gemini-diagnostic-20261010-after-billing/` | Two-call diagnostic and accounting receipt, separate from test quality scores. |
| [review.csv](human-review/review.csv), [provenance.json](human-review/provenance.json) | Consolidated final human judgments and their source identities, complete decision history, approvals and selection evidence. |
| [Similarity predictions](../results/dictabert/test_details.csv) | Existing 245/381 result; no full sentence snapshot or collection manifest is available. |

The primary review covers 638 groups / 741 of 4,953 responses: 707 newly judged
occurrences and 34 explicitly approved historical reuses. Provenance retains all
620 decision events, two later rechecks, annotators, times, exposure information
and focused-selection reasons. All responses, including the unreviewed ones, are
reconstructed from experiment sources. The primary review had one annotator, Shaked Schnarch (S.S.). System identity
was visible and a negative default required explicit confirmation. Coverage of technically complete automatic negatives was selective: about
24–79% across the six main generation systems. The historical
provenance records prior exposure as unknown. In a retrospective author
clarification on 12 October 2026, Shaked states that he had not seen AI
evaluations of the model answers before annotating; the historical record is
preserved. He clarifies that `fits` meant matching the reference answer shown,
not independently judging contextual correctness. Consequently, the 338
automatic-negative / `fits` response occurrences are diagnostic, not global
accuracy or cross-system human rankings, and review cannot validate erroneous
references. Selection exceeds generation under the automatic metrics, not under
an established common human semantic evaluation.

Nine automatic-positive / `not_fits` response judgments across four items remain
unresolved; six share a grouped decision on `kn4-0525`. These are not nine
independently adjudicated gold errors. Default-confirmed judgments and grouped
decisions remain in the provenance; batch timestamps do not measure per-item
reading time. These judgments do not estimate overall human accuracy. The final
backup's export differed from the working export only in `annotation_sha256`, not
in decisions or coverage; both historical source identities are retained in the
external conversion archive.

<a id="ben-dictalm-annotations"></a>

## Ben's separate DictaLM annotations

The `ben_dictalm` layer in the consolidated table preserves 381 generation labels:
193 `yes`, 188 `no`. Its original CSV was introduced at commit
`b5cabe8ed749b5eff14de6cda48aa5ac838dc684`, with SHA-256
`c6cfec23d97d9b2f29eac135454c4d609f19e6aba9280060a4c8ab406c51cbf0`.
Judgment IDs retain that historical source hash and bind run
`c6cc64b4017547d884540a16af02a066`, item and raw response hash. They never transfer
to other runs or selection.

The original displayed responses match 234 raw journal responses exactly and
147 after boundary trimming with `response.strip()`. Raw whitespace is preserved.
The source records yes/no under `same_as_gold`; a detailed rubric, per-item
rationale, annotation time and blinding are undocumented. This is a separate
layer, not an additional primary-review sample. DictaLM retains automatic
generation 125/381, Ben generation 193/381, strict selection 0/381 and diagnostic
leading-label selection 243/381. The template's missing newline remains a
protocol limitation with unknown causal effect. The empty `human_judgment`
column in the DictaLM response export is a retained schema field; the actual
judgments are in [review.csv](human-review/review.csv), under the separate
`ben_dictalm` layer.

See [reproduction instructions](../docs/reproducible_evaluation.md) for offline
analysis and separately authorized collection, and [checkpoint evidence](../docs/checkpoints.md)
for the encoder's incomplete historical training provenance.
