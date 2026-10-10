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
| [DictaLM LoRA source](src/hebrew_acronyms/models/dictalm_lora/) | `examples.py`: selection prompts and answer letters; `training.py`: LoRA training with development-loss checkpointing; `eval.py`: selection scoring with or without the adapter. [Colab notebook](notebooks/train_dictalm_lora_colab.ipynb). |
| [Input contract](src/hebrew_acronyms/models/common/pairs.py) | Exact target spans, IDs, candidate pairs and input identities. |
| [Shared study evaluation](src/hebrew_acronyms/models/common/eval.py) | Strict letter parsing, preliminary selection micro/macro accuracy and item inspection. Historical scoring functions remain separate from the current study. |
| [LLM backends](src/hebrew_acronyms/models/) | Qwen via local Ollama and Gemini via its API, with bounded requests and response provenance; other retained model code is historical context. |
| [Data preparation](docs/data_processing.md) | Source collection, review application and split-construction functions and commands. |
| [Local checks](docs/pipelines.md) | Environment check and explicit data validation; no combined model runner. |
| [Tests](tests/) | Invented fixtures, tiny learning and reconstruction checks; no Git history needed. |

The [LaTeX manuscript](paper/main.tex), [paper build/export instructions](paper/README.md),
[bibliography](paper/references.bib) and
[course sources](docs/course/README.md) describe the research context. The ID-named PDF
is the submitted proposal; the other PDF is an earlier draft. The [data inventory](data/README.md)
and [dataset card](data/mined/DATASET_CARD.md) preserve source and construction details.
[Historical results](results/all_arms_summary.md) and [training records](docs/checkpoints.md)
refer to earlier inputs and include unresolved score discrepancies. They are not
results for the current study. Earlier work also exists in `ShakedSchnarch/nlp-hw-team`;
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

## DictaLM LoRA selection training

[LoRA](https://arxiv.org/abs/2106.09685) (low-rank adaptation) freezes a pretrained
model and trains small added matrices in its attention layers. The
[DictaLM LoRA code](src/hebrew_acronyms/models/dictalm_lora/) trains a DictaLM causal
language model on candidate selection only. Each training item becomes the exact
selection prompt of the LLM arms, with candidates in a seeded shuffled order, and
the answer is the reference candidate's letter. Only the answer letter and the end
token contribute to the loss. The seed is set before adapter initialization, and the
adapter is saved only when the development loss strictly improves. Evaluation reuses
the shared selection evaluator, so candidate order, letter parsing and scoring match
the other selection arms; without `--adapter` it scores the untrained model through
the same inference code.

```bash
python -m hebrew_acronyms.models.dictalm_lora.training --model-id MODEL \
  --train data/study_v1/encoder_inputs/train.csv \
  --dev data/study_v1/encoder_inputs/dev.csv --output-dir RUN_DIR
python -m hebrew_acronyms.models.dictalm_lora.eval --model-id MODEL \
  --adapter RUN_DIR/adapter --test data/splits/test_items.csv --output details.csv
```

The model ID, revision and the defaults in
[LoraTrainingConfig](src/hebrew_acronyms/models/dictalm_lora/training.py) are
engineering choices for a first run, not an approved protocol. A 7B model needs a
CUDA GPU; `--load-in-4bit` additionally needs `bitsandbytes`, which the
[Colab notebook](notebooks/train_dictalm_lora_colab.ipynb) installs. Real-data training
and test evaluation require explicit authorization. The tiny CPU tests in
`tests/test_dictalm_lora.py` check the code, not model performance.

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
