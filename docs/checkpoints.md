# Checkpoints

Weight files are local and ignored by Git. The [training appendix](../notebooks/train_dictabert.ipynb)
defaults to smoke and writes no checkpoint. Authorized training requires explicit inputs,
a cached DictaBERT snapshot and a new output path; see [setup](../README.md#training-appendix).
Historical weights, full training and GPU execution have not been reproduced.

The shared [checkpoint loader](../src/hebrew_acronyms/models/dictabert_cross_encoder/model.py) adds `[ACR]` and `[/ACR]`
and resizes embeddings before loading the saved state. Pooling is selected through
`TrainingConfig.pooling`: `cls` uses the pair summary, `marker` the opening marker,
`span_mean` the target subwords, and `concat` combines summary and span vectors.
The existing seed is applied after model initialization; a fixed seed alone does not
establish reproducible initialization.

## Historical record

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
