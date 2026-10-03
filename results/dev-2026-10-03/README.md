# Preliminary development results — 3 October 2026

This is a frozen discussion snapshot, not a final benchmark or paper result.
No model requests were made to prepare this package.

Open [the offline item comparison](saved_results.html) or [the figure](selection_overview.png).
The [PDF figure](selection_overview.pdf), [per-type counts](by_type.csv),
[item-level predictions](predictions.csv), and [provenance summary](summary.json) are portable.
The input rows are data/study_v1/encoder_inputs/dev.csv; their full content was independently
compared with the saved run snapshot and matched.

| Complete selection arm | Correct / items | Micro | Macro (11 types) |
|---|---:|---:|---:|
| DictaBERT | 41/62 | 66.13% | 70.98% |
| Qwen 2.5 7B | 38/62 | 61.29% | 80.92% |

DictaBERT alone matches 19 items; Qwen alone matches 16. The net difference is three
items, with no established significance or general ranking. Five types have one item.

Gemini is incomplete: selection has 13 responses, 20 service-error items and 29
unattempted items; generation has two service-error items and 60 unattempted items.
The 13 responses match the existing labels, but are not a representative complete
evaluation. Generation has no semantic score for either LLM. The saved service summary
records 83 HTTP attempts: 13 HTTP 200, 63 HTTP 429 and seven HTTP 503. The precise
quota and reset time are unknown; token exhaustion specifically is not established.

## Interpretation for the discussion

- נ״ר: DictaBERT 14/15 versus Qwen 1/15. This is a useful specific strength, not proof
  that fine-tuning caused the difference; an untrained baseline was not compared here.
- רמב״ם: DictaBERT 0/11 versus Qwen 9/11 against the existing expansion convention.
  DictaBERT selected the hospital description for all 11, including one street context.
  The earlier human decision required the name expansion even for institutional uses.
  Discuss expansion versus entity identification; do not automatically mark all 11 as
  annotation errors or retrospectively change the rule to favor a model.
- חמ״ס: DictaBERT 9/9 versus Qwen 4/9 matches defective labels. Hazardous materials is
  absent from all nine candidate inventories, including explicitly stated contexts.
- מט״ח: at least one explicit educational-center context carries a foreign-currency gold.
  Audit labels and candidate inventories before interpreting any model ranking.
- בד״ר: DictaBERT 2/7 versus Qwen 7/7 warrants error inspection.

Full original run artifacts and request histories remain outside Git under
artifacts/study-runs/ in the project folder. The summary identifies each original
run and artifact hash. This compact publication copy does not replace them or
invent a synthetic run identity. No weights, credentials or large raw artifacts
are included. AI assisted preparation and verification of this snapshot.
