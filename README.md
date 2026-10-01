# Hebrew Acronym Disambiguation

A TAU NLP course project on interpreting Hebrew acronyms in context: generating an
expansion freely or selecting one from a supplied candidate list. The repository
contains data construction code, existing model implementations, a training notebook
and a preliminary companion for experimental methods and analysis. Scientific choices
and the final experiment protocol remain open.

The code is organized for local development and inspection. Fixture tests and CPU
smoke checks establish execution and structural compatibility, not model quality.
[Historical results](results/all_arms_summary.md) are not findings on the current data.
AI-assisted code organization and reviews are not human approval of the protocol;
AI-authored examples remain disclosed in the [data documentation](data/README.md).

## Repository map

| Location | Responsibility |
|---|---|
| [src/hebrew_acronyms/](src/hebrew_acronyms/) | Importable Python package, installed into the project environment. |
| ↳ `data_processing/` | Source collection, candidate tables, mining, review application and split construction. See [commands](docs/data_processing.md). |
| ↳ `models/` | Shared input/scoring helpers, baselines, `dictabert_similarity`, `dictabert_cross_encoder`, Gemini and Qwen implementations. |
| ↳ `pipelines/` | Environment check, data validation and research evaluation entry points. See [execution reference](docs/pipelines.md). |
| [data/](data/README.md) | Source exports, review evidence and historical split/aggregate inputs. |
| [notebooks/](notebooks/) | [Experimental methods and analysis](notebooks/experimental_study.ipynb): environment check and study outline. [Training appendix](notebooks/train_dictabert.ipynb): runnable source calls; default is offline smoke. |
| [results/](results/all_arms_summary.md) | Preserved historical predictions and summaries. |
| [docs/](docs/) | [Course sources](docs/course/README.md), command references and [checkpoint notes](docs/checkpoints.md). |
| [tests/](tests/) | Baseline-equivalence tests and invented fixtures, separate from research data. |

The working [manuscript](paper/manuscript.md) and [bibliography](paper/references.bib)
live in `paper/`. The draft describes the approved research direction and pending
method settings; it contains no current experimental results. Checkpoint files
remain local and ignored by Git.

[AGENTS.md](AGENTS.md) is the single source of operating instructions for agents.
The coordinator maintains the existing project plan at `../PROJECT_PLAN.md`, alongside
this repository. The next phase is research clarification with Shaked; existing
experiment proposals are not approved execution settings.

## Research history and current authority

Start with the [submitted proposal and its status](docs/course/README.md), then the
active section of the [central plan](../PROJECT_PLAN.md). Shaked confirmed the
ID-named proposal on 1 October 2026; his Downloads copy is byte-identical to the
preserved PDF. The other proposal PDF is an earlier draft of the same project.

Two repositories contributed work. `ShakedSchnarch/nlp-hw-team`, under `project/`,
contains earlier modelling, planning and research evidence and is now read-only.
`BenCarmel123/hebrew-acronym-disambiguation`, this repository, is the selected
submission base, developed from Ben's work. The two implementations and their old
plans are not interchangeable. The old 600-type HeAcro concept is not an active
delivery requirement.

| Work | What it establishes | Current use |
|---|---|---|
| Shaked's 30 August Colab DictaBERT run | Recorded completion of three training epochs on 2,552 provisionally labelled examples; no dev/test accuracy | Training feasibility history; see [distinct run records](docs/checkpoints.md) |
| Ben's encoder experiments and historical LLM predictions | Development exploration on earlier data; some scores and input versions conflict | Preserve [historical results](results/all_arms_summary.md); do not report them as current findings |
| Code organization, fixtures and notebook preparation | Structural and execution checks within their recorded scope | Reusable implementation, not scientific validation |
| P1/P1-R, 16-item review and later preparation packages | Completed preparation and recorded occurrence judgments; inventories remain unapproved | [Working manuscript](paper/manuscript.md) and proposed review artifacts |
| Focused 100–120 natural-occurrence direction | Approved research scope, including four LLM conditions and task-unseen encoder types | Detailed protocol, qualified inputs and final results remain pending |

As of 1 October 2026, Shaked requests documentation reconciliation before another
planning conversation. No next research package has been authorized by that request.
The central plan distinguishes approved direction, completed work, pending choices
and historical prompts. Prior results exist; final results for the current study do not.

## Install and check

Use the reviewed `review-handoff` branch with its **full Git history**. The 19 September
handoff records a push of an earlier revision of this branch; later work is local.
Do not assume a remote checkout contains the latest handoff or use remote `main`
as its substitute. Check the actual local HEAD and upstream before copying. To check a separate copy:

```bash
git clone --no-hardlinks --branch review-handoff /path/to/reviewed/local/repository hebrew-acronym-check
cd hebrew-acronym-check
git rev-parse HEAD
git status --short
python3.12 -m venv .venv
.venv/bin/python -m pip install -e .
```

Confirm the expected commit and a clean tree. The editable installation uses
`pyproject.toml`, which reads dependencies from `requirements.txt`; there is one
dependency list. It pins selected packages, not the complete transitive environment. A ZIP or shallow clone may omit
reference commits required by the equivalence tests; missing history is a failure,
not a skipped pass. The old `nlp-hw-team` repository remains a separate historical source.

Run from the repository root, each command in a fresh process:

```bash
.venv/bin/python -m pip check
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B -m unittest discover -s tests -v
.venv/bin/python -B -m hebrew_acronyms.pipelines.check_environment
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
invented sentences; the notebook uses [test fixtures](tests/fixtures).

The notebook runner executes every Python code cell in a new process. It does not test
the Jupyter interface; Jupyter is not installed by `requirements.txt`. GPU execution,
full training and historical checkpoint reproduction remain unverified. Newly initialized
pooler/marker/scoring parameters in smoke checks are expected and imply no learned result.

## Project notebooks

Start with [experimental_study.ipynb](notebooks/experimental_study.ipynb),
**Experimental Methods and Analysis**, for the sequence from corpus preparation to
the manuscript's results. Its environment check is executable; the remaining sections
outline methods pending protocol finalization. They will be completed alongside the
approved data, evaluation and analysis work, rather than deferred to final packaging.
It does not yet execute the complete experiment or reproduce research results.

### Training appendix

The notebook's first code cell holds imports, paths and mode settings. Later cells
load inputs, prepare pairs, load the model, call the selected action and summarize it.
`MODE="smoke"` performs inference only. Authorized `train` mode requires explicit local
train/development inputs, a cached snapshot and a new checkpoint output path.
Both modes block network access for the lifetime of the process; restart before
unrelated network work. Settings live in [TrainingConfig](src/hebrew_acronyms/models/dictabert_cross_encoder/training.py).

The existing behavior is retained: seeding occurs after model initialization and
checkpoint selection requires strict improvement in development pair loss. Pair
accuracy is not item-level candidate-selection accuracy. Existing length-budget edge
cases and differing evaluator scoring/skip rules have not been changed by organization.

Research training, evaluation, mining and service calls require a separately authorized
task. **`src/hebrew_acronyms/pipelines/run_pipeline.sh` evaluates test automatically when present; `--skip-llm`
does not disable test.** Use only the safe commands above for structural verification.
