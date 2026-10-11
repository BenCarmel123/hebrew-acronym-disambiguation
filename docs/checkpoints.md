# Checkpoints

Saved analysis needs no weights. To repeat the identified test inference, request
`dictabert-crossenc-study_v1-newtrain-seed43-20261003.pt` from the project authors
through their authorized storage channel. No public download is provided. Verify
SHA-256 `23bbff0324a7ce4d9d4a24a9ebfb87a4f6ac296bf1f1ce1bd25126c09260267f`.
The filename does not verify the training seed.

The inference used `dicta-il/dictabert` snapshot
`8884c6db002aba4002ee638fe4070c92e9ffbbf1`. Obtain the matching local snapshot and
use the strict [Colab state-dictionary loader](../src/hebrew_acronyms/models/dictabert_cross_encoder/colab.py).
[Checkpoint reconstruction](../saved-results/study-runs/dictabert-test-20261009-f49c3b5/checkpoint-reconstruction.json)
records architecture, tokenizer/configuration hashes, strict-load checks and unknown
training fields. Historical absolute paths there describe collection provenance;
they are not dependencies of saved analysis.

The historical architecture reference is preserved in Git at
`5c6980efee34b18e3fa63e53f7fd5713938571c0:notebooks/train_dictabert_colab.ipynb`,
SHA-256 `a57e668d20ba5c5cd8f6bcacd5b52fe8d67288cf1ea9a09cdb67e7108f0b010c`.
That source sets its seed after model initialization. Original training rows,
seed, selected epoch and library versions remain incompletely documented; neither
the state dictionary nor a fresh inference establishes reproducible historical training.

The identified test run explicitly used the researcher-approved
`legacy_first_occurrence` target policy: the first quote-normalized acronym match,
with attached letters outside the marked span. These derived spans are not human
annotations. Inference preserves every test item, raw scores and failures; it performs
selection only and does not train.

The [optional training example](../notebooks/train_dictabert.ipynb) uses a separate
package checkpoint format: weights plus `<checkpoint>.json` containing model,
tokenizer, library, input, seed and selection identities. Its
[loader](../src/hebrew_acronyms/models/dictabert_cross_encoder/model.py) rejects missing
sidecars and incompatible identities. New training seeds initialization before
constructing the model and selects strictly lower development pair loss. This format
must not be confused with the historical state dictionary. Invented-input checks
are engineering evidence, not benchmark performance.
