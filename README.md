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


## Short qualitative review (current protocol)

The current human task is a post-experiment qualitative review with a one-hour
budget including synthesis. It is **not** an independent blind evaluation, a
representative sample, or a basis for corrected benchmark-wide accuracy. The older
six-system calibration workflow below is historical and remains available at
`/legacy`; its queue and annotation meanings have not been changed.

The fixed `qualitative-generation-v1` route has 20 distinct sentences/acronyms:
12 with generation failure and selection success for the same model, four with
both failures for the same model, and four with both generation scores positive.
Items are disjoint. Selection uses saved Boolean scores only, retaining eligible
prior work and preferring diverse acronyms, sources, model witnesses and acronym
length proxies, with a deterministic hash tie-break. The three groups are interleaved
so stopping early does not mean reviewing a contiguous group only. Eligibility pools
may overlap; selected groups do not. The bundle stores all selection reasons and
actual composition. The delivered sample has seven Knesset, six Wikipedia and seven
AI-authored sentences. No answer semantics were inferred to choose them.

Only Qwen and Gemini free-generation answers are required. The main screen uses
neutral codes and immediately shows the sentence and the possibly faulty reference.
Choose “fits context”, “does not fit”, or “unsure”; optional reference/context concern,
article example flag and a note require no explanation. Save-and-next also permits
partial work. Changes autosave to disk, and reload resumes at the first incomplete
sentence. Two marked generation responses complete a **short-route** item only;
this does not change the six-response completion state of the old protocol.
Stop-and-summary works at any point; it never opens another review queue.

Case details are one explicit secondary view containing candidates, saved selection
responses, other systems and source information. Main-screen API payloads exclude
model identities, automatic scores, sampling reasons and derived agreements. Opening
a case records reference/generation exposure before judging; opening details or the
summary separately records the additional exposure. No initial independent attempt
is fabricated. Prior exposure from the old record remains intact.

The supplied `Start Review.command` now loads `review-data-short-v1.json`. The server
keeps the original `annotations.json` and its history, and writes the short protocol
to `annotations.short-v1.json` plus its own append-only history. The original first
item, `manual-0042`, is included: its generation judgments map explicitly from
`wrong` to `not_fits` and `correct` to `fits`; they remain attributed to the earlier
protocol and require no repeat. Original interpretations, all six judgments, notes,
flags, timestamps and exposure history remain unchanged and are copied losslessly
into the short export. Only `correct`, `wrong`, and explicit `undecidable` have direct
mappings. Partial, combined legacy categories and `no_answer` are preserved as
unmapped original judgments; they are never silently assigned a semantic meaning.

The summary distinguishes computed counts/composition/score differences from human
judgments, suspicions and notes. It lists disagreements, possible reference/context
issues, flagged examples, unresolved answers, all reviewed cases and incomplete IDs.
It does not infer a semantic cause from a score difference. Download the summary as
Markdown or the complete annotated bundle/history as JSON. To restore the latter,
stop the server and place a copy at the short annotation path; startup verifies the
protocol, plan and source manifest. Keep a backup before restoring.

To create an explicitly authorized new short plan at a fresh output path:

```bash
python -m hebrew_acronyms.human_review_short --data existing-review-data.json --legacy annotations.json --output fresh-short-data.json
python -m hebrew_acronyms.human_review_server --data fresh-short-data.json --annotations annotations.json
```

Additional offline tests:

```bash
python -m unittest tests.test_human_review_short
node tests/test_human_review_short_ui.js
```

## Local human review of saved test answers

The review-only tool serves existing answers in Hebrew on loopback. It does not
invoke models or edit original research files. Its Python modules use the standard
library; an isolated `python -m pip install --no-deps -e .` installation is sufficient.
For this local delivery use the existing external review runtime and the supplied
`Start Review.command` in the artifact folder. Keep its server running while working.
Closing the browser retains disk annotations; reopening restores the current item.

Build a new bundle at a fresh path, then start with explicit paths:

```bash
python -m hebrew_acronyms.human_review_data --root . --output ../artifacts/human-review-20261009/review-data-v2.json
python -m hebrew_acronyms.human_review_server --data ../artifacts/human-review-20261009/review-data-v2.json --annotations ../artifacts/human-review-20261009/annotations.json
```

The current stage is the fixed 20-item **calibration** queue. Follow-up evaluation
has not been defined; it is not a second copy of calibration. Source-balanced
coverage and acronym-length proxies are deliberately disproportionate, so these
items do not directly estimate benchmark-wide rates. The targeted diagnostic queue
is separate. Decide further review scope after human calibration feedback.

Enter the actual reviewer identity. Before revealing candidates, write an initial
interpretation and select its state, or explicitly select no interpretation or
insufficient context. Revealing candidates permanently captures the text, state,
reviewer and time. Afterward the initial snapshot is read-only; a separate revised
interpretation remains editable. Reference, answers, automatic scores and system
names have separate recorded reveal stages. Pseudonyms reduce name cues but do not
establish full experimental blinding. Highlighted acronym matches are mechanical,
not certified target occurrences. Prior human exposure defaults to unknown.

Autosave and the draft button preserve progress without claiming completion. Use
partial review to stop midway. Each item displays responses marked out of its total.
Full completion requires an item/reference decision and a label for every response;
explicit inability to decide is allowed for both. Revisit is only a reminder and
never supplies missing judgments. Editing a completed item returns it to partial
until explicitly completed again; the earlier completed snapshot remains in history.
A name-only save is a draft, not a completed review. These rules are enforced by the
server as well as the interface.

Judge meaning against the sentence, even when the original reference is suspect:

- Correct: the meaning fits the context.
- Wrong: the meaning does not fit the context.
- Partial: only part of the meaning fits.
- Undecidable: insufficient evidence to judge the meaning.
- No answer: no response was given.

Use the main radio buttons for these judgments. Confidence, format, notes and
attribution explanations are optional and compact. Human proposed expansions and
alternatives are stored separately from the original reference. After saving a
clear judgment, the server derives its disagreement with the original automatic
score; partial and undecidable labels do not determine that comparison. A discrepancy
can be explained as a reference problem, a protocol/scoring problem, or another
reason. No semantic label is generated automatically.

The sidebar summarizes **saved records matching the current filters**. Full and
partial item counts are separate; item completion always concerns all its answers.
The answer denominator respects the selected system. Explicit undecidable responses
are shown separately within marked responses. Answer findings (wrong, partial,
undecidable and disagreement) refer only to the selected system; item/reference
issues remain general. Unsaved changes are not included in these counts.

Export JSON or CSV through the backup panel. Both retain the immutable initial
snapshot, revised interpretation, reviewer identity, schema, source manifest and
sampling plan. CSV is a lossless roundtrip format with validated readable columns;
edit labels in the interface rather than changing CSV summary columns alone.

Annotation schema `human-review-v2` splits the old combined partial/undecidable
label. Old `partial` labels become `legacy_partial`, retaining their original value
and entering an explicit recheck list; they are never guessed into a new category
and cannot satisfy completion. Existing v1 records and exposure history are preserved
with a backup before migration. Initial interpretations are recovered only from a
complete, unambiguous first-candidate-reveal history; otherwise marked unavailable.
An import cannot replace an existing immutable initial snapshot.

`source_identity` (also `dataset_id`) hashes the original repository and sorted
source file paths, hashes and sizes. `sampling_plan_id` independently identifies the
queue plan. A changed queue or sample size can reuse the same annotation path after
source-manifest validation; all items and prior exposures remain saved. For example,
build `--sample-size 30` to a **fresh** bundle path only after an approved review-plan
change, stop the server and restart with that bundle and the same annotation path.
The previous data bundle remains intact. JSON/CSV imports also require the same
source manifest; changed research sources are rejected even if an ID is copied.

Offline checks for the review tool:

```bash
python -m unittest tests.test_human_review_data tests.test_human_review_server tests.test_human_review_v2_server tests.test_human_review_independent tests.test_human_review_sampling_resume
node tests/test_human_review_ui.js
```

The code and QA were AI-assisted. QA sessions use `--qa`, a separate port and separate
annotation path. They must not be included in human research summaries. The original
source files, predictions and manuscript are outside this tool's write scope.
