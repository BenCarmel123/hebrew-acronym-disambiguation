# Test-set results (data/splits/test_items.csv)

The 14 test items whose Knesset document also supplies training items were removed, leaving
381 items; all numbers are recomputed from the per-item CSVs with those rows removed.
No system was re-run.

All numbers below are from fresh local runs (not Colab), confirmed directly
from the per-item detail CSVs in `results/{dictabert,dictabertx,qwen,gemini}/`.
An earlier Colab-produced table for dictabertX and gemini disagreed with these
(e.g. dictabertX 0.704 vs. 0.717 here) — two independent local implementations
agree with the numbers below, so they supersede the Colab run.

| Arm | Accuracy | Invalid rate | Items |
|---|---|---|---|
| random | 0.235 | — | 381 |
| most_frequent | 0.370 | — | 381 |
| most_mined | 0.480 | — | 381 |
| oracle | 1.000 | — | 381 |
| dictabert (untrained) | 0.643 | — | 381 |
| dictabertX (fine-tuned) | 0.717 | — | 381 |
| qwen (generate) | 0.071 | 0.924 | 381 |
| qwen (select) | 0.598 | 0.008 | 381 |
| gemini (generate) | 0.520 | 0.454 | 381 |
| gemini (select) | 0.934 | 0.005 | 381 |
