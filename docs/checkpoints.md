# Checkpoints

Use the [main study notebook](../notebooks/experimental_study.ipynb) for checkpoint
inspection and dev prediction. Local installation and service setup are in the
[README](../README.md#install-and-check). The [training appendix](../notebooks/train_dictabert.ipynb)
remains separate; checkpoint loading does not train the model.

Each checkpoint consists of a weights file and its original `<checkpoint>.json`, which records:

- Training settings, initialization seed, pooling/dropout and library versions.
- Model identifier/revision or local snapshot hashes, plus tokenizer and marker identities.
- Ordered train/dev item IDs and hashes of the complete supplied rows.
- Selected epoch, development pair loss, selection rule and weights SHA-256.

The [loader](../src/hebrew_acronyms/models/dictabert_cross_encoder/model.py) restores
saved settings, adds `[ACR]` and `[/ACR]`, resizes embeddings and verifies weights,
model, tokenizer and library identities. `snapshot_path` can explicitly relocate a
snapshot whose complete local file hashes match the recorded source; the original
JSON remains unchanged. Missing sidecars, weight-only legacy files and incompatible
artifacts are rejected. Loading restores inference weights, not optimizer state.

In the main study, `train_path` and `input_path` identify the qualified train and full
dev files. Before prediction, their independently computed `input_identity` values
are compared with the manifest's `inputs` and passed to `load_finetuned` as
`expected_inputs`. The same comparison precedes reuse of saved encoder predictions.
A three-item validation run therefore checks the full 62-item dev identity, not the
three-item subset. Reloading an existing study result uses its saved contents and
does not reread train/dev files. The notebook's compact checkpoint summary reports
training settings, selected epoch, input counts and match status; full metadata is
retained in the result file.

Pooling options remain `cls` (pair summary), `marker` (opening marker), `span_mean`
(target subwords) and `concat` (summary plus span). Initialization seeds Python and
PyTorch before constructing the encoder, marker embeddings and scoring head.
Checkpoint selection still requires strictly lower development pair loss. The tiny
engineering sanity check reuses invented train/dev rows for 180 epochs; its temporary
weights and learning behavior are separate from research training.

[Shared evaluation](../src/hebrew_acronyms/models/common/eval.py) now aggregates
preliminary dev selection accuracy: micro over requested items and unweighted macro
across `type_id` groups, with failures retained in the denominators. Generation
answers remain for manual review. Fixtures cover loading, input matching, snapshot
relocation and scoring; actual supplied checkpoints and live-service performance
require a manual validation run.

## Shaked's earlier Colab run — separate implementation

The old `ShakedSchnarch/nlp-hw-team` repository records a completed run on
30 August 2026 in `project/docs/MODEL_CACHE.md`, inspected at old commit
`3bae9130a6a7086e3861235afc5360e4d08bc7ea` (source SHA-256
`eb267b1350bd6c6e540a8da8f180842740dd4b49147f6e6155cfbf751fe18f9a`).
This is documentary evidence, not a new reproduction or a claim that its weights
are present in this repository.

The record identifies training-code commit `d64d51d28d3a23bc0ad445914ebc718311933550`:
2,552 silver (provisionally labelled) items, 11,172 candidate pairs and 295 types;
three fixed epochs, seed 42 and learning rate 2e-5. Recorded training loss decreases
from 0.5545 to 0.3322 to 0.2271. `final.pt` and `last.pt` were saved to Drive and
read back according to the record. No dev/test inputs were evaluated, no accuracy
was measured and no checkpoint was selected by an evaluation metric.

Training completed; the subsequent held-out scientific evaluation did not complete
in that recorded workflow. Development source groups had not been reserved, so the
later proposed annotation pack cannot retrospectively become held-out data for that
checkpoint. The training machinery and implementation experience remain useful.
Its checkpoint is not interchangeable with Ben's checkpoints or approved for the
current study. Current Drive availability, GPU repeatability and this historical
run's reproduction have not been verified here.

## Ben's historical checkpoint records

These values are preserved from `63b90acfae36e6b7fee8114760506868e29c681f:weights/README.md`.
They are documentation claims, not verified checkpoint evaluations. Named artifacts
may not be available locally; the row previously called “current” had no filename.
Historical names used `dictabert-crossenc-<YYYYMMDD-HHMMSS>.pt`.

| Recorded checkpoint | Configuration | Recorded dev item accuracy |
|---|---|---:|
| `dictabert-crossenc-20260905-181754.pt` | cls, 3 epochs/epoch 1 selected, lr 2e-5, batch 16, unseeded | 0.815 |
| Not kept | cls, 1 epoch, lr 2e-5, batch 16, unseeded | 0.781 |
| Not kept | cls, 1 epoch, lr 2e-5, batch 16, seed 42 | 0.770 |
| Not kept | span_mean, 1 epoch, lr 2e-5, batch 16, seed 42 | <0.80 |
| Filename unspecified | cls, 1 epoch, lr 2e-5, batch 16, seed 42; described as corrected dev | 0.786 |

The earlier notes distinguish a 292-item dev version from a corrected 285-item version
and report 0.855 on 248 items after excluding five rabbinic-name types. These notes do
not establish a common run manifest or reconcile the [other recorded scores](../results/all_arms_summary.md).
The recorded first run's dev losses were 0.42/0.53/0.54 and train losses
0.178/0.106/0.072, with epoch 1 selected. `marker` and `concat` were described as untried.
Earlier claims that pooling made no difference, that seed 42 guaranteed repeatability,
or that a 4.5-point observed range defined a general noise floor are not established
by these records and are not adopted as current findings.

## Identified test inference

`test_encoder_inputs.qualify_test` retains strict target qualification by default.
The explicitly selected `legacy_first_occurrence` policy instead uses the existing
`models.common.pairs.find_span`: first quote-normalized acronym occurrence, with
attached letters outside the marked span. Shaked authorized this deterministic
choice on 2026-10-09 after the unresolved target positions were reported. It is
not a human annotation and does not establish the provenance of the historical
`results/dictabertx/test_details.csv` predictions.

`test_encoder_inference.run_encoder_test` verifies a one-to-one, ordered mapping
to all original test rows, preserving every source field. It loads the approved
local checkpoint with the existing strict Colab loader, validates three predictions,
and then records all test selection predictions incrementally. It does not train,
download weights, or perform generation. Exact candidate accuracy retains all
test items in the denominator; raw scores, failures, checkpoint reconstruction and
source identities are saved together. An existing run directory is never overwritten.


## Retrieving the identified test checkpoint

Saved-result analysis requires no checkpoint. To reproduce inference separately,
request the existing `dictabert-crossenc-study_v1-newtrain-seed43-20261003.pt`
artifact from the project authors through their authorized storage channel. No
public weight download is included or implied by this repository. Verify SHA-256
`23bbff0324a7ce4d9d4a24a9ebfb87a4f6ac296bf1f1ce1bd25126c09260267f`
before using the existing Colab state-dictionary loader; the filename does not
verify the historical training seed.

The fresh inference used `dicta-il/dictabert` snapshot
`8884c6db002aba4002ee638fe4070c92e9ffbbf1`. Its tokenizer/configuration hashes,
strict-load evidence and unresolved training fields are retained in
[saved checkpoint reconstruction](../saved-results/study-runs/dictabert-test-20261009-f49c3b5/checkpoint-reconstruction.json).
Use the separately supplied, hash-matched local snapshot and checkpoint with the
[training/inference appendix](../notebooks/train_dictabert.ipynb); do not substitute
unidentified weights. Original absolute paths in this evidence document collection
provenance only and are not analysis dependencies.
