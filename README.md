# Hebrew Acronym Disambiguation

A TAU NLP course project on interpreting Hebrew acronyms in context: generating an
expansion freely or selecting one from a supplied candidate list. The repository
contains data construction code, existing model implementations and a thin training
notebook. Scientific choices and the final experiment protocol remain open.

The code is organized for local development and inspection. Fixture tests and CPU
smoke checks establish execution and structural compatibility, not model quality.
[Historical results](results/all_arms_summary.md) are not findings on the current data.
AI-assisted code organization and reviews are not human approval of the protocol;
AI-authored examples remain disclosed in the [data documentation](data/README.md).

## Repository map

| Location | Responsibility |
|---|---|
| [data_preprocess/](data_preprocess/README.md) | Source collection, candidate tables, sentence mining, review application and split construction. Source-specific code lives under `wikipedia/`, `wiktionary/` and `knesset/`; shared text/table helpers live under `common/`. |
| [model/](model/) | `common/` reads items, builds pairs and implements LLM prompting/scoring; `dictabert/` loads the base encoder and evaluates similarity; `dictabertX/` owns cross-encoder encoding, model, training and workflow; `gemini/` and `qwen/` connect LLM backends. `baselines.py` contains reference methods. |
| [pipeline/](pipeline/README.md) | Safe environment check, data validation and separate research evaluation entry points. |
| [notebooks/train_dictabert.ipynb](notebooks/train_dictabert.ipynb) | Training code appendix: explanations and calls to source functions; default is offline smoke. |
| [data/](data/README.md) | Candidate/mining exports, review evidence and historical split/aggregate inputs. |
| [results/](results/all_arms_summary.md), [weights/](weights/README.md) | Preserved historical predictions and summaries; checkpoint files are local and ignored by Git. |
| [tests/](tests/) | Baseline-equivalence tests and invented fixtures, separate from research data. |
| [course/](course/README.md) | Proposal, course guidelines, instructor feedback and document provenance. |

[AGENTS.md](AGENTS.md) is the single source of operating instructions for agents.
The coordinator maintains the existing project plan outside this repository.

## Install and check

Use the reviewed `review-handoff` branch with its **full Git history**. It is currently
local; remote `main` does not contain the organization work. To check a separate copy:

```bash
git clone --no-hardlinks --branch review-handoff /path/to/reviewed/local/repository hebrew-acronym-check
cd hebrew-acronym-check
git rev-parse HEAD
git status --short
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Confirm the expected commit and a clean tree. `requirements.txt` pins selected
packages; it is not a complete transitive lockfile. A ZIP or shallow clone may omit
reference commits required by the equivalence tests; missing history is a failure,
not a skipped pass. The old `nlp-hw-team` repository remains a separate historical source.

Run from the repository root, each command in a fresh process:

```bash
.venv/bin/python -m pip check
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B -m unittest discover -s tests -v
.venv/bin/python -B -m pipeline.check_environment
.venv/bin/python -B -m tests.run_notebook
```

The test suite uses temporary files, invented rows and tiny encoders. It compares
encoding, pooling, optimizer updates, checkpoint selection, data commands and loading
against preserved Git baselines. It does not evaluate research data.

The two smoke checks require an **existing local DictaBERT snapshot**, run on CPU and
block network access. They do not download, train or write research results. Missing
cache means **NOT RUN**: environment-check exit code 2; other failures return nonzero.
For explicit cache selection, pass `--snapshot /path/to/local/snapshot` to the environment
check and set `DICTABERT_SNAPSHOT` for the notebook runner. The base check uses two
invented sentences; the notebook uses [test fixtures](tests/fixtures/).

The notebook runner executes every Python code cell in a new process. It does not test
the Jupyter interface; Jupyter is not installed by `requirements.txt`. GPU execution,
full training and historical checkpoint reproduction remain unverified. Newly initialized
pooler/marker/scoring parameters in smoke checks are expected and imply no learned result.

## Training appendix

The notebook's first code cell holds imports, paths and mode settings. Later cells
load inputs, prepare pairs, load the model, call the selected action and summarize it.
`MODE="smoke"` performs inference only. Authorized `train` mode requires explicit local
train/development inputs, a cached snapshot and a new checkpoint output path.
Both modes block network access for the lifetime of the process; restart before
unrelated network work. Settings live in [TrainingConfig](model/dictabertX/training.py).

The existing behavior is retained: seeding occurs after model initialization and
checkpoint selection requires strict improvement in development pair loss. Pair
accuracy is not item-level candidate-selection accuracy. Existing length-budget edge
cases and differing evaluator scoring/skip rules have not been changed by organization.

Research training, evaluation, mining and service calls require a separately authorized
task. **`pipeline/run_pipeline.sh` evaluates test automatically when present; `--skip-llm`
does not disable test.** Use only the safe commands above for structural verification.
