# Saved analysis exports

The [main notebook](../../notebooks/experimental_study.ipynb) regenerates these
results from [identified evidence](../../saved-results/README.md), without models
or API calls. Ten exported files are tracked:

| File | Contents |
|---|---|
| [automatic_metrics.csv](automatic_metrics.csv) | Task-specific automatic scores, with all 381 items and failures in each denominator. |
| [baselines.csv](baselines.csv) | Four selection baselines. |
| [similarity_metric.csv](similarity_metric.csv) | Existing DictaBERT similarity result, 245/381. |
| [models_and_settings.csv](models_and_settings.csv) | LLM identities, runs and collection settings. |
| [technical_failures.csv](technical_failures.csv) | 17 failed or truncated response occurrences. |
| [human_review_coverage.csv](human_review_coverage.csv) | New judgments, approved reuse and unreviewed responses by system/task. |
| [response_review_bindings.csv](response_review_bindings.csv) | All 4,953 answers with original identities and separate automatic/human fields. |
| [dictalm_reported_metrics.csv](dictalm_reported_metrics.csv) | Separate automatic generation, Ben generation, strict selection and diagnostic selection measurements. |
| [dictalm_responses_381.csv](dictalm_responses_381.csv) | 762 raw DictaLM answers, prompts, candidate orders and original scores. |
| [dictalm_audit.json](dictalm_audit.json) | Run/source identities, protocol limitations and separate annotation provenance. |

The notebook also produces untracked coverage, event/recheck, agreement, cost,
pilot and similarity-provenance exports, plus figures and clarification views.
There are 23 generated CSV/JSON analysis exports, of which 10 are tracked.
Public reproduction can compare the 10 tracked references with copies saved
before execution or retrieved from the same fixed Git revision. The other 13
are generated without distributed reference copies. Our release audit compared
all 23 byte-for-byte against an external accepted baseline; this is separate
from public reproduction. The runner records provenance but does not automatically
compare exports. The agent-assessment export has been retired. Execution receipts are
saved outside Git and identify their actual notebook and package sources.

The primary review remains 741/4,953 responses: 707 new judgments and 34 approved
reuses. It includes 338 automatic-negative / `fits` occurrences; nine
positive / `not_fits` occurrences across four items remain unresolved. Ben's 381
DictaLM labels are a separate layer in
[the consolidated table](../../saved-results/human-review/review.csv), with
[historical identities and limitations](../../saved-results/README.md#ben-dictalm-annotations).
The empty `human_judgment` column in `dictalm_responses_381.csv` is retained
for schema/byte preservation; DictaLM judgments live in the consolidated
`review.csv` linked above, not in that column. `fits` in the primary review means
matching the displayed reference. These selective reviews do not establish
overall contextual accuracy or cross-system human rankings.

For execution, use the [offline reproduction guide](../../docs/reproducible_evaluation.md#reproduce-the-saved-test-analysis).
