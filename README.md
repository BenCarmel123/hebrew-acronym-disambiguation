# Hebrew Acronym Disambiguation

A TAU NLP course project studying Hebrew acronym expansion through free generation
and candidate selection. Saved test results cover Qwen, Gemini, trained DictaBERT
selection and a DictaBERT similarity baseline. Historical development and test
runs have different execution and scoring paths; they must not be combined without
checking their inputs and provenance. Generation receives neither candidates nor
gold. Comparisons between systems do not isolate the causal effect of model size.

## Execution status and entry points

The saved [test results](results/test_results.md) are historical measurements.
Human review and a new reproducible evaluation are in progress. The repaired
runner and four LLM adapters have offline recovery and provider tests. Live pilot
and Colab validation remain separate acceptance steps; no new full-test result is
claimed here. See the [operating guide](docs/reproducible_evaluation.md).

| Entry point | Role and current status |
|---|---|
| [Test evaluation in Colab](notebooks/run_test_eval_colab.ipynb) | Installs an identified commit/archive, uses Secrets and Drive, runs a fixed dev pilot, and saves each attempt for identity-checked recovery. Full test follows pilot inspection and the shared budget check. |
| [Experimental study](notebooks/experimental_study.ipynb) | Methods, development experiments and analysis companion. Its documented development workflow below is distinct from the test runner. |
| [Training appendix](notebooks/train_dictabert.ipynb) | Package-backed training route. Retraining is a separate experiment, not an automatic part of test evaluation. |
| [Historical Colab training](notebooks/train_dictabert_colab.ipynb) | Reference for the existing checkpoint; its seed is set after model initialization. See the recorded provenance limitations before reusing it. |
| [Qwen scale experiment](notebooks/run_qwen_scale_colab.ipynb) | Optional 32B/72B experiment with substantial GPU requirements; not the default full-test runner. |

Preserve saved outputs when preparing a new run. Human decisions, mechanical
normalization and original automatic scores are different evidence types. The
paper is a working draft; its pending values and figures require reconciliation
with the selected results before submission.

## Code and reading map

| Location | Purpose |
|---|---|
| [Training appendix](notebooks/train_dictabert.ipynb) | Explicit inputs, candidate pairs, DictaBERT initialization or loading, optional training and item predictions. Start here to inspect executable model code. |
| [Main dev study](notebooks/experimental_study.ipynb) | Runnable local preview, saved-result inspection, manual validation and full-dev prediction across five arms; no training. |
| [Cross-encoder source](src/hebrew_acronyms/models/dictabert_cross_encoder/) | `encoding.py`: length-bounded inputs; `model.py`: scoring and checkpoints; `training.py`: optimization; `eval.py`: item records; `workflow.py`: small local setup helpers. |
| [Input contract](src/hebrew_acronyms/models/common/pairs.py) | Exact target spans, IDs, candidate pairs and input identities. |
| [Shared study evaluation](src/hebrew_acronyms/models/common/eval.py) | Strict letter parsing, preliminary selection micro/macro accuracy and item inspection. Historical scoring functions remain separate from the current study. |
| [LLM backends](src/hebrew_acronyms/models/) | Qwen via Ollama, Gemini, GPT-4.1 mini and Claude Haiku 5.5, with bounded requests and response provenance. Explicit model IDs and settings are recorded for each run. |
| [Data preparation](docs/data_processing.md) | Source collection, review application and split-construction functions and commands. |
| [Local checks](docs/pipelines.md) | Environment check and explicit data validation; no combined model runner. |
| [Tests](tests/) | Invented fixtures, tiny learning and reconstruction checks; no Git history needed. |

The [LaTeX manuscript](paper/main.tex), [paper build/export instructions](paper/README.md),
[bibliography](paper/references.bib) and
[course sources](docs/course/README.md) describe the research context. The ID-named PDF
is the submitted proposal. A new manuscript PDF must be built from the current
source and visually checked; the previously tracked `paper/draft.pdf` was removed.
The [data inventory](data/README.md)
and [dataset card](data/mined/DATASET_CARD.md) preserve source and construction details.
[Historical results](results/all_arms_summary.md) and [training records](docs/checkpoints.md)
refer to earlier inputs and include unresolved score discrepancies. Keep these
separate from the saved test CSVs and any new evaluation. Earlier work also exists in `ShakedSchnarch/nlp-hw-team`;
its checkpoints and implementation are not interchangeable with this repository.

<a id="install-and-check"></a>

## Local setup

From this checkout, create an isolated environment and open the main notebook:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e . jupyterlab ipykernel
.venv/bin/python -m pip check
.venv/bin/python -m jupyter lab notebooks/experimental_study.ipynb
```

Select the `.venv` kernel. Project dependencies are pinned in `requirements.txt`
through `pyproject.toml`; the complete transitive environment is not locked. The
checkpoint also records the required model, tokenizer and library versions. For
installation troubleshooting, use a fresh environment rather than import-path overrides.

**Qwen:** provision `qwen2.5:7b` in local Ollama, then check `ollama list` and
`curl http://localhost:11434/api/tags`. The notebook connects to the existing service;
it does not start it or download models.

**Gemini:** the selected model is `gemini-3.8-flash` with
`{"thinkingConfig": {"thinkingLevel": "low"}}`. Low thinking is not disabled thinking.
Copy `.env.example` to `.env` in the checkout root and set `GEMINI_API_KEY` there.
The study loads this file when Gemini is enabled, using the notebook's `root`
setting. Existing environment variables take precedence. Restart the kernel after
changing the key. Keep `.env` local (Git ignores it); never put keys in notebook
cells or settings. See the [official model settings](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash).

The systems can run independently. The basic notebook request has a 120-second timeout
and no automatic retries. The explicit `resume_gemini_study` runner below bounds
transient retries and saves each attempt. Qwen records its server-reported digest;
Gemini records the returned
`modelVersion`. Partial answers are retained and marked incomplete. Checkpoints and
live services still need the manual validation below.

## Configure and run

The [main notebook](notebooks/experimental_study.ipynb) follows inputs → encoder →
LLMs → results. Its centralized settings include:

| Setting | Value or required input |
|---|---|
| `train_path` | Qualified `data/study_v1/encoder_inputs/train.csv`, used to check the checkpoint's training inputs. |
| `input_path` | Qualified `data/study_v1/encoder_inputs/dev.csv`, containing all 62 dev items. |
| `checkpoint` | Exact checkpoint path. Default `checkpoint_format="manifest"` requires the original adjacent JSON; the explicit Colab route below accepts an authorized complete state dictionary. |
| `snapshot_path` | A content-identical local base-model/tokenizer snapshot if relocated; otherwise `None`. |
| `device` | `"cpu"` by default; set a supported `"mps"` or `"cuda"` explicitly if needed. |
| `enable_encoder`, `enable_qwen`, `enable_gemini` | Independent switches, initially `False`. |
| `output_root` | `<project>/artifacts/study-runs`; each run creates a fresh directory outside Git. |
| `saved_run`, `saved_run_id` | Exact result JSON path and run ID for reload. |

For manifest-bound encoder prediction or saved-prediction reuse, the study compares the complete
qualified train/dev rows with the checkpoint's recorded inputs. Validation checks
against the full dev file, even though it predicts only three items. The compact
checkpoint summary shows training settings, selected epoch, train/dev counts and
match status; full metadata stays in the saved result. See [checkpoint details](docs/checkpoints.md).

- **Preview:** leave `mode="preview"` for invented examples with no model calls or research reads.
- **Validation:** set `mode="run"`, `run_kind="validation"` and enable the desired systems. DictaBERT predicts three items; each LLM sends one generation and one selection request, four requests with both enabled.
- **Full dev:** after manual validation succeeds, rerun from fresh settings with `run_kind="full_dev"`. All 62 items are requested per enabled arm: **248 LLM requests** with both providers.
- **Reload:** set `mode="reload"`, `saved_run` and `saved_run_id`. Inspection needs no models, services, credentials or original research files. Older three-arm results remain readable as three-arm results.

The notebook defines the scoring rules and interpretation limits beside the results.
Expand an item to inspect its context, candidate mapping, answers and failure details.
To reuse encoder predictions in a new run, set `enable_encoder=False`, `saved_encoder`
to the source `study.json`, and `saved_encoder_run_id` to its run ID. Keep the current
train/dev paths: reuse must pass the same input comparison as fresh prediction.

## Offline study checks

These checks use invented inputs and mocked responses, without training or live services:

```bash
.venv/bin/python -B -m unittest tests.test_experimental_study tests.test_five_arm_study tests.test_study_evaluation tests.test_qwen_study tests.test_gemini_study tests.test_checkpoint_relocation tests.test_encoder_study_inputs -v
```

Network and research-file guards apply only inside the test processes. The wider
repository suite separately includes tiny-model learning tests.

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
are implementation settings, not a finalized research protocol. The default loader requires a
JSON manifest and matching model/tokenizer, inputs when supplied, and library identities;
see [checkpoint details](docs/checkpoints.md). Unspecified weight-only files are rejected.
The explicitly selected Colab adapter below is a separate, provenance-labelled exception.
An explicit relocated snapshot is accepted only after content identity verification;
the original manifest is not rewritten. GPU training and research performance have
not been verified by the tiny CPU checks.

Qualified dev inputs and preliminary selection scoring are available. Final benchmark
runs and generation judgment rules remain separate work. Natural, substituted and
AI-authored material must remain identifiable; see the data documentation. AI assistance
contributed code, checks and draft prose, not human annotation or scientific validation.
Weights, caches, environments, secrets and raw run outputs stay outside version control.
The explicitly selected paper run may publish small derived PDF figures, TeX tables
and provenance manifests under `paper/generated/`; preview and fixtures cannot do so.

## Separate runs and Colab checkpoints

The main notebook accepts `checkpoint_format="colab_state_dict"`, an explicit
`checkpoint_sha256`, local `snapshot_path`, and the supplied `checkpoint_attestation`.
The [Colab adapter](src/hebrew_acronyms/models/dictabert_cross_encoder/colab.py)
reconstructs CLS pooling, length 256, `[ACR]`/`[/ACR]`, and the original pair encoding
from the inspected Colab notebook. It verifies the exact checkpoint hash and strictly
loads every encoder and head tensor, validates three items, then predicts the remaining
cohort. Original training seed, library versions and exact training-row identity remain
unknown unless separately evidenced. Reconstruction evidence is not an original manifest.

To run or resume Gemini with the exact prompts and candidate orders from a saved Qwen
full-dev source (explicit service authorization is required):

```bash
python -m hebrew_acronyms.resume_gemini_study --root . \
  --source /path/to/qwen/study.json --source-run-id EXACT_QWEN_RUN_ID \
  --output-root /path/outside/repository --run-id NEW_GEMINI_RUN_ID
```

The same command resumes unattempted items, never resends completed responses, and
retains ambiguous interrupted requests for inspection. Each HTTP request is saved
separately; transient failures permit at most three attempts per item with Retry-After
and backoff. Five consecutive failed items stop collection with remaining items marked
unrun. Resume after service recovery; existing terminal failure records are preserved.

The notebook's final comparison cell accepts `(study.json path, original run ID)`
entries for independently saved runs. It checks complete input identity and prompt
agreement, retains original arm run IDs, and displays all 62 items without model or
network calls. Selection includes service/format failures in its denominator; generation
remains semantically unscored. Review gold labels and candidate inventories with Ben,
including institutional uses, before interpreting these diagnostic development scores.
