# Historical development results

This table is retained unchanged from the baseline record. It is **not a result on the
current splits** and has no complete run manifest. It first appeared in the old
repository at `8b8d741` (then `data/mined/all_arms_summary.md`) and was later
moved; the inherited table is also preserved at new-repository baseline `eb2e778`.
Model names and scores below are recorded historical labels, not current service claims.

| Arm | Accuracy | Invalid rate | Items |
|---|---|---|---|
| random | 0.338 | — | 285 |
| most_frequent | 0.488 | — | 285 |
| most_mined | 0.533 | — | 285 |
| oracle | 1.000 | — | 285 |
| dictabert (untrained) | 0.702 | — | 285 |
| dictabertX (fine-tuned) | 0.825 | — | 285 |
| qwen (generate) | 0.021 | 0.979 | 285 |
| qwen (select) | 0.628–0.635 | ~0.005 | 285 |
| gemini-flash-lite-latest (generate) | 0.477–0.502 | 0.425–0.467 | 285 |
| gemini-flash-lite-latest (select) | 0.867 | 0.000 | 285 |
| gemini-3.6-flash, thinking on (generate) | 0.723 | 0.242 | 285 |
| gemini-3.6-flash, thinking off (generate) | 0.698 | 0.284 | 285 |
| gemini-3.6-flash, thinking on (select) | 0.951 | 0.000 | 285 |
| gemini-3.6-flash, thinking off (select) | 0.954 | 0.000 | 285 |

## Conflicting records and predictions

The earlier root README reported **0.692** for untrained DictaBERT and **0.786** for the
fine-tuned cross-encoder, rather than the **0.702 / 0.825** above. Its narrative associated
0.692 with an earlier 292-item dev set and 0.786 with a corrected 285-item set. It also
claimed a 12.3-point improvement, which does not match that displayed pair. No result
has been selected as authoritative. Recover the complete narrative with:

```bash
git show 63b90acfae36e6b7fee8114760506868e29c681f:README.md
```

The old-repository commit `a728fc1` records the 0.786 claim; the 12.3-point paragraph
already appears in `55e1aba`, before the later summary table. These are historical
attributions, not enough evidence to identify one common run. Other training observations
are retained in [checkpoint notes](../docs/checkpoints.md).

The eight CSVs under [gemini/](gemini) and [qwen/](qwen) each retain 285 per-item
predictions. Their IDs/acronyms/gold/candidates match the old dev snapshot at `8b8d741`.
Against current dev, 281 IDs remain and 35 of those have a changed acronym, gold or
candidate field. Prediction files omit sentence text and a run manifest; do not join
them to current dev by ID and call the outcome a current score. All original response,
correctness and validity fields remain preserved without re-scoring.
