# Hebrew Acronym Disambiguation

A Tel Aviv University NLP course project investigating how models resolve Hebrew
acronyms from sentence context. The study compares **free generation** of an
expansion with **candidate selection** from a supplied inventory. Free generation
receives neither candidates nor the reference answer (gold). The comparison asks
how the task formulation and model family affect disambiguation; it does not
isolate the causal effect of model size.

## Data

The corpus combines observed Wikipedia and Knesset text, Wikipedia sentences with
substituted acronyms or removed explanations, and authored examples. Wiktionary
also contributes candidate expansions. Text origin and label evidence are retained
separately: natural text is not automatically a verified label, and authored text
is not observed usage.

The stored test split contains **381 items across 60 acronym types**: 253 Knesset,
41 Wikipedia and 87 Claude-authored sentences. Fourteen earlier test items from four
Knesset protocols that also supply training items were removed; new runs never
collect them, and runs saved before the removal are scored without them
([test cohort](src/hebrew_acronyms/test_cohort.py)). The research/encoder dev cohort
(`data/study_v1/encoder_inputs/dev.csv`) contains 62 items; the test notebook's
pilot uses the first ten of the 289 items in `data/splits/dev_items.csv`. The
[data inventory](data/README.md) and [dataset card](data/mined/DATASET_CARD.md)
document training inputs, construction, review evidence and limitations, including
incomplete reconstruction of later additions.

## Method and interpretation

LLMs perform both tasks on the same items. Candidate order is fixed per item and
recorded for reuse and recovery; labels beyond Z preserve all candidates, including
the two test items with 30 candidates. The evaluation package includes Qwen2.5 7B and 14B
and DictaLM through Ollama, Gemini, GPT-4.1 mini, Claude Haiku 5.5 and Grok 4.7 adapters.
DictaLM, a Hebrew-specialised LLM, runs locally without fine-tuning; see the
[local DictaLM run](docs/reproducible_evaluation.md#local-dictalm-run). Exact model IDs,
settings, returned versions and response usage are recorded in each run.

Encoder comparisons distinguish trained DictaBERT candidate scoring from a
pretrained DictaBERT similarity baseline without task adaptation. Four deterministic
baselines are also available. The [checkpoint documentation](docs/checkpoints.md) records loading
requirements and the existing weights' incomplete training provenance. The new
checkpoint inference uses an explicitly recorded first-occurrence target
policy; these derived spans are not human annotations.

## Saved results

Open [experimental_study.ipynb](notebooks/experimental_study.ipynb) for the current
methods, tables, figures, coverage and limitations. Its committed outputs are
readable without executing cells. Default execution loads the bundled
[saved evidence](saved-results/README.md); it makes no model requests and requires
neither API accounts nor weights. The optional development demonstration is disabled
and contributes no test observations.

The test analysis preserves separate collection runs and full 381-item denominators,
including technical failures. Gemini now includes all 381 items in both tasks:
357/381 selection and 207/381 generation under the existing automatic rules,
with two truncated generation responses retained. The closed human-review exports
cover **638 groups / 741 of 4,953 responses**: 707 new judgments and 34 explicitly
approved historical reuses. Human coverage remains partial and diagnostically
prioritized; automatic scores do not replace it. Four conflicting contexts now have
separate [delegated agent assessments](saved-results/agent-clarifications-20261010/assessments.json)
at the user's request: five groups / ten already-reviewed responses. These are
unblinded agent assessments, with no new human decision or additional human coverage.
Historical labels and pending flags remain unchanged.
[Reusable tables and figures](results/submission-20261010/README.md) include full
response bindings, technical coverage and human-review gaps.
Earlier measurements under [results/](results/test_results.md) remain historical
sources and are not silently pooled with the identified collection.

The notebook reports four distinct DictaLM measurements on 381 items: automatic
generation **125/381 (32.8%)**, generation accepted by **Ben 193/381 (50.7%)**, strict
selection **0/381 (0%)**, and diagnostic leading-label selection **243/381 (63.8%)**.
The last is not a replacement score. Ben's labels are bound to the original generation
responses and loaded once per identity; they do not propagate to selection or other
models. [Annotation provenance and limits](saved-results/README.md#ben-dictalm-annotations)
include the source commit, hash, raw-response binding and partially documented rule.
The saved chat template omits the publisher-specified newline; its causal effect is
unknown. The zero local API increment does not prevent reporting results: cumulative
accounted expenditure remains 23.1276784 ILS and historical manifests remain intact.
The paper reports these four measurements in a separate supplementary table,
without a common ranking.

[Metric coverage](results/submission-20261010/metric_coverage.csv) separates technical
failures, stored format validity and automatic matches; human judgments are separate.
Qwen14B generation has **10/381 (2.6%) technical failures**. The notebook also verifies
the saved similarity baseline (**245/381, 64.3%**) against its
[historical predictions](results/dictabert/test_details.csv), without loading weights
or rerunning inference. Its missing run manifest and sentence snapshot limit provenance.

## Project structure

| Location | Purpose |
|---|---|
| [src/hebrew_acronyms/](src/hebrew_acronyms/) | Data processing, model adapters, prompts, scoring and resumable evaluation. |
| [data/](data/README.md) | Source exports, candidate inventories, review evidence, splits and qualified encoder inputs. |
| [run_test_eval_colab.ipynb](notebooks/run_test_eval_colab.ipynb) | Pilot/test collection; separate from reading or reproducing saved analysis. |
| [experimental_study.ipynb](notebooks/experimental_study.ipynb) | Primary saved-test analysis; optional dev appendix disabled by default. |
| [train_dictabert.ipynb](notebooks/train_dictabert.ipynb) | Explicit package-backed encoder training and inference. |
| [train_dictabert_colab.ipynb](notebooks/train_dictabert_colab.ipynb) | Historical Colab training route and checkpoint configuration source. |
| [run_qwen_scale_colab.ipynb](notebooks/run_qwen_scale_colab.ipynb) | Optional scale-collection route; existence does not establish completed runs. |
| [tests/](tests/) | Engineering checks with invented inputs and mocked services. |
| [results/](results/) | Historical predictions and summaries, preserved with their provenance. |
| [paper/](paper/README.md) | Working LaTeX manuscript, bibliography and build/export instructions. |

<a id="install-and-check"></a>

## Basic setup and execution

Create an isolated environment from the repository root:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -e . jupyterlab ipykernel
.venv/bin/python -m pip check
.venv/bin/python -m jupyter lab notebooks/experimental_study.ipynb
```

Register it with `.venv/bin/python -m ipykernel install --user --name hebrew-acronym-analysis`,
then select `hebrew-acronym-analysis` in Jupyter. The main notebook validates and analyzes the bundled saved results.
Missing or altered evidence raises an error; there is no private-directory fallback. Dependencies are pinned in `requirements.txt` through
`pyproject.toml`; the full transitive environment is not locked. Model execution
requires explicit settings and the corresponding local model or API credentials.
Keep secrets in local environment configuration or Colab Secrets, outside Git.

For a guarded portable replay and a new execution receipt, follow
[reproduce the saved analysis](docs/reproducible_evaluation.md#reproduce-the-saved-test-analysis).
The historical receipt remains unchanged.

For Colab evaluation, follow the [reproducible execution guide](docs/reproducible_evaluation.md).
It covers a commit-identified archive, persistent Drive output, provider setup,
per-system pilot inspection, cumulative costs and recovery after interruption.
Full test requires a successful matching pilot for each selected system. The guide
also documents local development runs, checkpoint inference and training; training
is a separate experiment rather than an implicit evaluation step.

Run the focused evaluation checks without live API calls:

```bash
.venv/bin/python -B -m unittest tests.test_test_eval_notebook tests.test_test_evaluation tests.test_staged_evaluation tests.test_provider_adapters -v
```

For data preparation and additional checks, see [data processing](docs/data_processing.md),
[local pipelines](docs/pipelines.md) and the execution guide's offline checks.
AI assistance contributed code, checks and draft prose; it does not establish
human annotation or scientific validation.
