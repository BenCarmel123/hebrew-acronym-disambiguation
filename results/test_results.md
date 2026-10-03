# Test-set results (data/splits/test_items.csv)

All numbers below are from fresh local runs (not Colab), confirmed directly
from the per-item detail CSVs in `results/{dictabert,dictabertx,qwen,gemini}/`.
An earlier Colab-produced table for dictabertX and gemini disagreed with these
(e.g. dictabertX 0.704 vs. 0.724 here) — two independent local implementations
agree with the numbers below, so they supersede the Colab run.

| Arm | Accuracy | Invalid rate | Items |
|---|---|---|---|
| random | 0.234 | — | 395 |
| most_frequent | 0.382 | — | 395 |
| most_mined | 0.489 | — | 395 |
| oracle | 1.000 | — | 395 |
| dictabert (untrained) | 0.651 | — | 395 |
| dictabertX (fine-tuned) | 0.724 | — | 395 |
| qwen (generate) | 0.068 | 0.927 | 395 |
| qwen (select) | 0.595 | 0.008 | 395 |
| gemini (generate) | 0.532 | 0.443 | 395 |
| gemini (select) | 0.937 | 0.005 | 395 |
