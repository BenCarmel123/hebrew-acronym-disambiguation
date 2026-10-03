# Hebrew Acronym Disambiguation

A TAU NLP course project comparing five arms: trained DictaBERT candidate selection,
Qwen free expansion and selection, and Gemini free expansion and selection. All arms
use the sentence and the same identified target occurrence. The two LLMs receive the
same prompt per task and the same displayed candidates for selection; generation
receives neither candidates nor gold. This compares systems, without isolating the
causal effect of model size or context. No final research results are reported.

## Code and reading map

| Location | Purpose |
|---|---|
| [Training appendix](notebooks/train_dictabert.ipynb) | Explicit inputs, candidate pairs, DictaBERT initialization or loading, optional training and item predictions. Start here to inspect executable model code. |
| [Main dev study](notebooks/experimental_study.ipynb) | Runnable local preview, saved-result inspection, manual validation and full-dev prediction across five arms; no training. |
| [Cross-encoder source](src/hebrew_acronyms/models/dictabert_cross_encoder/) | `encoding.py`: length-bounded inputs; `model.py`: scoring and checkpoints; `training.py`: optimization; `eval.py`: item records; `workflow.py`: small local setup helpers. |
| [Input contract](src/hebrew_acronyms/models/common/pairs.py) | Exact target spans, IDs, candidate pairs and input identities. |
| [Shared study evaluation](src/hebrew_acronyms/models/common/eval.py) | Strict letter parsing, preliminary selection micro/macro accuracy and item inspection. Historical scoring functions remain separate from the current study. |
| [LLM backends](src/hebrew_acronyms/models/) | Qwen via local Ollama and Gemini via its API, with bounded requests and response provenance; other retained model code is historical context. |
| [Data preparation](docs/data_processing.md) | Source collection, review application and split-construction functions and commands. |
| [Local checks](docs/pipelines.md) | Environment check and explicit data validation; no combined model runner. |
| [Tests](tests/) | Invented fixtures, tiny learning and reconstruction checks; no Git history needed. |

The [manuscript](paper/manuscript.md), [bibliography](paper/references.bib) and
[course sources](docs/course/README.md) describe the research context. The ID-named PDF
is the submitted proposal; the other PDF is an earlier draft. The [data inventory](data/README.md)
and [dataset card](data/mined/DATASET_CARD.md) preserve source and construction details.
[Historical results](results/all_arms_summary.md) and [training records](docs/checkpoints.md)
refer to earlier inputs and include unresolved score discrepancies. They are not
results for the current study. Earlier work also exists in `ShakedSchnarch/nlp-hw-team`;
its checkpoints and implementation are not interchangeable with this repository.

## Install and check

From the repository root:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -I -B -c "import hebrew_acronyms.models.dictabert_cross_encoder.workflow"
.venv/bin/python -m pip check
```

Dependencies are declared once in `requirements.txt` through `pyproject.toml`.
Selected versions are pinned; the complete transitive environment is not locked.
If an editable installation fails to import in a fresh interpreter, create a clean
external environment and install the package there. When reusing already installed
local dependencies, `pip install --no-index --no-deps --no-build-isolation -e .` avoids
downloads. Do not use `sys.path` or `PYTHONPATH` overrides to hide installation failures.

The essential suite runs on temporary invented data and tiny CPU models. It checks
input contracts, length limits, pooling, controlled initialization, one small learning
exercise, strict checkpoint selection, identity-checked reload, prediction failures,
notebook execution and data-review protections. It requires neither a model cache
nor Git history. These checks establish engineering behavior, not model quality.

## Run and inspect the local dev study

After the isolated installation above, add the notebook tools to that environment:

```bash
.venv/bin/python -m pip install jupyterlab ipykernel
.venv/bin/python -m jupyter lab notebooks/experimental_study.ipynb
```

Select its kernel. The notebook opens in `mode="preview"`, using invented examples
without loading models, contacting services, reading research inputs or saving output.
It contains the detailed setup and explicit input → encoder → LLM → save → inspect flow.
Use its centralized settings for these values:

| Setting | Value or required input |
|---|---|
| `input_path` | Qualified `data/study_v1/encoder_inputs/dev.csv`, expected 62 items; no historical-dev fallback. |
| `checkpoint` | Ben’s exact weights file under `<project>/artifacts/dictabert/2026-10-03/`; its adjacent `<weights>.json` must be the original. No file is selected automatically. |
| `snapshot_path` | Explicit local base-model/tokenizer snapshot if relocated; otherwise `None`. Loading verifies identical content and the recorded library identities. |
| `device` | `"cpu"` by default; choose a supported `"mps"` or `"cuda"` explicitly if needed. |
| `enable_encoder`, `enable_qwen`, `enable_gemini` | Independent switches, all `False` by default. |
| `qwen_model`, `ollama_url` | `"qwen2.5:7b"`, `"http://localhost:11434"`; the user provisions the service/model separately. |
| `gemini_model`, `gemini_generation_config` | Selected `"gemini-3.8-flash"`, `{"thinkingConfig": {"thinkingLevel": "low"}}`. Low thinking is not disabled thinking. |
| `output_root` | `<project>/artifacts/study-runs`, outside Git; every run gets a fresh ID and directory. |
| `saved_run`, `saved_run_id` | Exact saved JSON path and run ID when using `mode="reload"`. |

Set `GEMINI_API_KEY` only in the environment inherited by the kernel. Do not put it
in notebook cells, settings or saved files; no `.env` file is loaded automatically.
Qwen-only and encoder-only runs need no Gemini key, and Gemini-only runs need no
Ollama service. Each enabled request has a 120-second timeout and no automatic retries.
Qwen records server-reported digest evidence; Gemini records the returned `modelVersion`,
finish reason and usage without claiming digest verification. Model availability and
account access still require manual validation. See the [official Gemini model page](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash).

Start with `mode="run"`, `run_kind="validation"`: three dev items for DictaBERT and
one request per task for each enabled LLM, four LLM requests when both are enabled.
After success, restart the settings and run all cells with `run_kind="full_dev"` for
all 62 items per enabled arm, **248 LLM requests** with both providers. Validation and
full-dev outputs stay separate. `mode="reload"` inspects saved evidence without
models, services or credentials; older three-arm artifacts remain readable without
inventing Gemini results.

The source-backed results view shows context, gold, raw answers, candidate mapping,
decoded selections, failures and disagreements. Selection uses strict uppercase-letter
parsing and exact trimmed candidate–gold equality. Micro accuracy includes every
requested item; macro is the unweighted mean of within-`type_id` accuracies. Failures
and unrun items remain in denominators; disabled systems and partial/subset runs are
labelled explicitly. Generation remains unscored for human review. These are preliminary
dev measures, not test results or proof of compatibility with an actual checkpoint.

Focused offline checks use invented fixtures and mocked service responses; they do
not train or call real models/services:

```bash
.venv/bin/python -B -m unittest tests.test_experimental_study tests.test_five_arm_study tests.test_study_evaluation tests.test_qwen_study tests.test_gemini_study tests.test_checkpoint_relocation -v
```

Network and research-file guards exist only in those test processes. The wider
repository suite also includes the small learning exercise described above; it is
separate from the study checks.

## Training appendix

Open [train_dictabert.ipynb](notebooks/train_dictabert.ipynb) in a notebook editor using
the installed environment. Imports and settings appear first, then inspectable steps:
input rows → candidate pairs → model → optional training → predictions.
Jupyter and a notebook kernel are not installed by the project dependencies.

The default (`TRAIN=False`, `LOAD_CHECKPOINT=None`) uses two invented rows and an
existing local DictaBERT snapshot, selected with `SNAPSHOT` or discovered in local
cache. No training, research-file reads or services occur. A missing snapshot raises
an explicit error; no model is downloaded. Initial rankings come from a random head.
For a cache-free engineering execution of all cells with a tiny injected model:

```bash
.venv/bin/python -B -m tests.run_notebook
```

The test runner blocks sockets and research-file access in its own process and
executes unchanged cells. It performs inference only. The single tiny learning check
runs separately in the contract suite. Ordinary notebook setup does not install a
permanent socket blocker.

For separately authorized training, choose explicit `TRAIN_PATH`, `DEV_PATH` and a
fresh `CHECKPOINT_PATH`, then set `TRAIN=True`. The notebook validates inputs, builds
pairs, initializes the model, trains and reloads the best development-loss checkpoint.
For checkpoint inference, set `LOAD_CHECKPOINT` with `TRAIN=False`; saved settings
are restored. Change the explicit prediction input when using qualified data. No test
file or other research input is selected automatically.

Items require unique `item_id`, original `sentence`, exact `target_raw` including any
prefix, half-open `span_start`/`span_end`, and pipe-separated `candidates`. Training
requires `gold_expansion` to match exactly one candidate after surrounding whitespace
is trimmed, with at least two distinct candidates. Prediction preserves singleton
items and identified failure records. Context may be cropped; targets, markers and
candidates are retained or the input fails explicitly. No aliases or metrics are inferred.

## Technical and scientific limits

The encoder uses the existing pooling, BCE loss, AdamW and strict development-pair-loss
checkpoint selection. Defaults in [TrainingConfig](src/hebrew_acronyms/models/dictabert_cross_encoder/training.py)
are implementation settings, not a finalized research protocol. Checkpoints require a
JSON manifest and matching model/tokenizer, inputs when supplied, and library identities;
see [checkpoint details](docs/checkpoints.md). Legacy weight-only files are rejected.
An explicit relocated snapshot is accepted only after content identity verification;
the original manifest is not rewritten. GPU training and research performance have
not been verified by the tiny CPU checks.

Qualified dev inputs and preliminary selection scoring are available. Final benchmark
runs and generation judgment rules remain separate work. Natural, substituted and
AI-authored material must remain identifiable; see the data documentation. AI assistance contributed code, checks and
draft prose and does not constitute human annotation or scientific validation.
Weights, caches, environments, secrets and generated outputs stay outside version control.
