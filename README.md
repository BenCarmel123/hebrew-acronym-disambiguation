# Hebrew Acronym Disambiguation

A TAU NLP course project comparing free expansion of Hebrew acronyms with selection
from a candidate inventory. The planned study compares the same LLM with and without
context in both formats, alongside a task-trained Hebrew encoder. Detailed data and
evaluation choices remain open; no final results exist for this study.

## Code and reading map

| Location | Purpose |
|---|---|
| [Training appendix](notebooks/train_dictabert.ipynb) | Explicit inputs, candidate pairs, DictaBERT initialization or loading, optional training and item predictions. Start here to inspect executable model code. |
| [Methods and analysis](notebooks/experimental_study.ipynb) | Study design, source documentation and analysis outline; not a completed experiment. |
| [Cross-encoder source](src/hebrew_acronyms/models/dictabert_cross_encoder/) | `encoding.py`: length-bounded inputs; `model.py`: scoring and checkpoints; `training.py`: optimization; `eval.py`: item records; `workflow.py`: small local setup helpers. |
| [Input contract](src/hebrew_acronyms/models/common/pairs.py) | Exact target spans, IDs, candidate pairs and input identities. |
| [Other models](src/hebrew_acronyms/models/) | Baselines, encoder similarity and LLM components; their historical scoring rules are not yet unified. |
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
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B -m unittest discover -s tests -v
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
see [checkpoint details](docs/checkpoints.md). Legacy weight-only files and relocation
of snapshot paths need separate handling. GPU training and research performance have
not been verified by the tiny CPU checks.

The shared benchmark, final scoring rules, qualified inputs and final research runs
remain incomplete. Natural, substituted and AI-authored material must remain
identifiable; see the data documentation. AI assistance contributed code, checks and
draft prose and does not constitute human annotation or scientific validation.
Weights, caches, environments, secrets and generated outputs stay outside version control.
