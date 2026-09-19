# Hebrew Acronym Disambiguation

**Structural organization passed separate AI-agent reviews; this is not human sign-off.**
The scientific protocol has not been approved. Existing results and research claims
below are historical; they are not current findings or authorization to run experiments.

## Start here

Use branch `review-handoff` with its full Git history.
The independently checked handoff snapshot is `a0eab21c30c6ad1b56fb726b03bb03ab71bb30a0`;
subsequent closure documentation does not change its code or data. Superseded local
work branches were removed after verifying that their commits are retained here.
Local `main` remains at the original baseline; research execution still requires authorization.
The remote `main` baseline does not contain these structural changes. A source archive
or shallow clone may also omit the historical commits required by the fixture tests.
Missing reference history fails the equivalence checks with a restoration instruction;
it is not counted as a pass or silently skipped. Use Git and a full clone, not a ZIP.
For a separate check, clone the reviewed branch into a new directory:

```bash
git clone --branch review-handoff https://github.com/BenCarmel123/hebrew-acronym-disambiguation.git hebrew-acronym-check
cd hebrew-acronym-check
git rev-parse HEAD
git status --short
```

Confirm the expected handoff commit and a clean tree before installation. The old
[nlp-hw-team repository](https://github.com/ShakedSchnarch/nlp-hw-team) remains a
read-only historical reference; its history is not merged here.

From the repository root, create a new local environment with Python 3.12 and install
from the single dependency file (do not reuse an unrelated environment):

The file pins selected dependencies, not the entire transitive dependency graph;
the recorded environment is a tested installation, not a complete lockfile.

```bash
python3.12 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
```

Run these safe checks from the repository root, each in a fresh process:

```bash
.venv/bin/python -m pip check
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B -m unittest discover -s tests -v
.venv/bin/python -B -m pipeline.check_environment
.venv/bin/python -B -m tests.run_notebook
```

The fixture suite uses invented inputs. The two smoke checks need an **existing local
DictaBERT cache** and do not download weights. Missing cache means **NOT RUN**, not a
successful model check. These commands do not train on research data or evaluate the
benchmark. The research pipeline is separate: `--skip-llm` does not prevent its test
evaluation. See [the training appendix and check details](#training-appendix-and-structural-checks).

`pipeline.check_environment` reports Python, installed packages, OS and device availability; imports the
required packages and model modules; verifies exact pairs and target marking on two
invented Hebrew sentences; and attempts one short base-encoder forward pass on CPU in
evaluation mode without gradients. It reads no benchmark data and writes no results.
It enforces Hugging Face offline mode and blocks Python socket network operations.
Installation may use the network; the check does not download weights.

The check discovers DictaBERT under the standard Hugging Face cache (respecting
`HF_HOME` / `HF_HUB_CACHE`), using its local `main` reference or a sole snapshot.
To specify a cached snapshot explicitly, append `--snapshot /path/to/local/snapshot`
to the same check command. Several snapshots without a usable reference require this
argument. Exit codes: `0` = all checks passed; `1` = failure; `2` = model not run because
no snapshot was found. A skipped model check is not a pass.

Verified on 2026-09-19 from a fresh local clone of the accepted code baseline
`35488bf036bdbddb87df2a05f2c329d1c68e1e2d`, with documentation-only handoff updates:
Python 3.12.11, macOS 26.6.2 arm64, a new isolated venv installed from
`requirements.txt` (no copied environment or global packages). `pip check` and all
**51 fixture tests passed**. CPU was used; CUDA was unavailable and MPS availability
was detected without running on it.

Both smoke checks passed using existing cached revision
`8884c6db002aba4002ee638fe4070c92e9ffbbf1`, with no weight download. The base check
produced finite output of shape `(2, 23, 768)`; all six notebook code cells ran in a
fresh Python process and produced four pairs, input `(4, 16)` and finite logits `(4,)`.
No research files, training or saved checkpoint were involved. Newly initialized
pooler weights, marker embeddings and scoring-head parameters are expected here;
these checks establish execution, not model quality. The cached revision identifies
this check only and does not change the research model default.

This command checks the base tokenizer/encoder only. The separate S3a checks below
cover the shared pair encoder and cross-encoder; neither validates the scientific
protocol or a historical trained checkpoint. The external GPU environment remains
**not verified**; CPU checks and dependency pins do not reproduce historical training.

Reader map: [data layers](data/README.md), [dataset card](data/mined/DATASET_CARD.md),
[training code appendix](notebooks/train_dictabert.ipynb) (default: offline smoke), and
[agent operating rules](AGENTS.md). Technical explanations assume basic ML knowledge;
NLP-specific terms should be explained when introduced.

## Course documents and provenance

The PDF bytes were checked with SHA-256 against the tracked files at these source commits:

- **Old source:** [ShakedSchnarch/nlp-hw-team](https://github.com/ShakedSchnarch/nlp-hw-team), local commit `3bae9130a6a7086e3861235afc5360e4d08bc7ea` (includes unpublished local history).
- **New baseline:** [BenCarmel123/hebrew-acronym-disambiguation at eb2e7785dab42dc8ae3ca07372d032adafee9dba](https://github.com/BenCarmel123/hebrew-acronym-disambiguation/tree/eb2e7785dab42dc8ae3ca07372d032adafee9dba).

| Local document | Source path and status | SHA-256 |
|---|---|---|
| [ID-named proposal](course/209533108_209233857_315110841_NLP_Project_Proposal.pdf) | Copied unchanged from old source `project/proposal/209533108_209233857_315110841_NLP_Project_Proposal.pdf`. Identified there as submitted. | `1e335e6baddcb699aecadcc290801785a2e62bea010e82c4a5ecb876f7e72c56` |
| [Earlier proposal draft](course/hebrew_acronym_disambiguation_proposal.pdf) | Already present at new baseline `course/hebrew_acronym_disambiguation_proposal.pdf`; identical to old source `project/proposal/hebrew_acronym_disambiguation_proposal.pdf`. Preserved unchanged. | `27c92cb4ff6378418c6d0b162008453cd3885c1a893a1c42e9e5c44606da0b5f` |
| [Course guidelines](course/NLP_course_2025b___project_guidelines.pdf) | Already present at new baseline `course/NLP_course_2025b___project_guidelines.pdf`; identical to old source `project/guidelines/NLP_course_2025b___project_guidelines.pdf`. Preserved unchanged. | `cc67f99fe1efb1a373d564a2332a64509772f3b166afc551769bd3a5676aaaa4` |

The old source commit's `project/proposal/README.md`
identifies the ID-named PDF as the submitted artifact and the other as an earlier draft.
This is documentary identification, not direct verification of submission or acceptance.
The submitted-identified proposal describes open generation versus candidate selection;
its planned systems and metrics remain source material, not approval of the current protocol.

[Instructor feedback](course/mor_feedback.md), inherited at new baseline path
`course/mor_feedback.md`, asks for a clear research question, related literature and
motivation in light of what large language models may already accomplish through prompting.
It is retained as course context, not a final experimental system selection.

## Existing research implementation (reference)

Given a Hebrew sentence containing an acronym and a list of possible expansions, pick the
one the sentence actually means.

Hebrew acronyms are heavily ambiguous. `א״ק` can be `אוויר-קרקע` (air-to-ground),
`אתלטיקה קלה` (athletics), `אדם קדמון`, `אם קריאה`, `אמר קרא` or `אנית קיטור` — and only
the surrounding context distinguishes them.

TAU NLP final project.

## Approach

Fine-tune [DictaBERT](https://huggingface.co/dicta-il/dictabert), a Hebrew encoder, as a
**cross-encoder**: pair the sentence (with the target occurrence marked `[ACR]…[/ACR]`)
with one candidate expansion, score how well they fit, and repeat per candidate. The
argmax is the prediction. The model picks from a fixed candidate list and cannot invent
an expansion outside it.

## Code responsibilities

The current dependency direction is notebook/entry point → callable source functions.
S3a consolidated the cross-encoder training path. S3b separated data command dispatch
from processing, moved shared candidate tables out of Wikipedia, and now reuses the
existing item reader and base-model loader across the affected model entry points.

| Responsibility | Current code |
|---|---|
| Collection and mining | `data_preprocess/wikipedia/`, `wiktionary/`, `knesset/`; `sefaria/` is exploratory and not wired into the CLI. Each owns source access and source-specific extraction. |
| Shared data components | `data_preprocess/common/`: orthography, acronym matching, gematria and mining filters; [candidates.py](data_preprocess/common/candidates.py) owns candidate rows, CSV I/O, resume caches and table summaries; `reporting.py` writes/displays JSON summaries. |
| Data assembly and review | `merge_sources.py`, `dedupe_expansions.py`, `review_duplicates_cli.py`, `build_annotation_table.py`, `apply_dev_review.py`, `build_splits.py` under `data_preprocess/`. These merge sources, apply recorded review and construct existing input files. |
| Data command entry point | [data_preprocess/__main__.py](data_preprocess/__main__.py) parses arguments, configures logging and calls functions with explicit inputs. [candidate_workflows.py](data_preprocess/candidate_workflows.py) coordinates inventory collection/merge/review; [sentence_workflows.py](data_preprocess/sentence_workflows.py) owns type selection, context export, progress and sense-run resumption; [knesset/workflows.py](data_preprocess/knesset/workflows.py) coordinates shard download/local export. Annotation assembly stays in `build_annotation_table.py`. |
| Item reading and pair construction | [model/common/pairs.py](model/common/pairs.py): shared item CSV reader, quote folding, locating/marking the target, candidate pairs and skip summaries. `baselines.load_rows` remains available as an import of this reader; the similarity CLI also uses it. Candidate-signal loading remains separate. |
| Model and input encoding | [dictabert/model.py](model/dictabert/model.py) loads the base tokenizer and encoder for the cross-encoder, similarity CLI and combined runner; callers retain device placement and evaluation mode; [dictabertX/model.py](model/dictabertX/model.py) owns the shared cross-encoder, marker initialization and checkpoint loading; [encoding.py](model/dictabertX/encoding.py) owns pair cropping/padding. |
| Training | [training.py](model/dictabertX/training.py): settings, batch order, BCE loss, AdamW and strict development-loss checkpoint selection. [workflow.py](model/dictabertX/workflow.py): local paths, mode selection, cached-model setup and short smoke. The [notebook](notebooks/train_dictabert.ipynb) explains and calls these functions. |
| Evaluation and decoding | `model/common/eval.py` shares LLM prompts, candidate shuffling, decoding, matching and detail export. `dictabertX/eval.py` retains item selection and calls the shared pair encoder through its compatible wrapper. `dictabert/eval.py` implements the separate similarity method; `baselines.py` implements reference baselines; `qwen/eval.py` and `gemini/eval.py` connect backends. This is not one unified evaluator. |
| Validation and orchestration | `pipeline/validate_data.py` owns schema/overlap checks; `run_all.py` coordinates methods and summary rendering; `run_pipeline.sh` coordinates validation and dev/test evaluation. These research entry points are not authorized during structural organization. `check_environment.py` is the separate safe base-encoder check. |
| Engineering checks | [tests/](tests/): exact-baseline comparisons, temporary data-workflow fixtures, tiny encoder fixtures and the fresh-process notebook runner. [tests/fixtures/](tests/fixtures/) is separate from research inputs. Temporary checkpoint tests use isolated temporary directories; notebook smoke writes no checkpoint. |
| Inputs and outputs | [data/](data/README.md): mining exports, review records and historical split inputs; [results/](results/): historical predictions/summaries; `weights/`: ignored weight files; [course/](course/): source documents. |

## Training appendix and structural checks

The commands in [Start here](#start-here) run the fixture suite and notebook smoke.

The suite isolates definitions and training operations from Git commit
`8ca117d50c3f01d4473b944c99611c7191af05b4`, without running the old notebook's setup,
downloads or research data. Keep that commit in the local Git history for these checks.
It compares encoding tensors, all four pooling modes, state dictionaries, tiny-encoder
optimizer updates, late seeding and strict checkpoint selection. Temporary round trips
use the real checkpoint loader with a tiny fixture encoder, not historical weights.

Data-processing equivalence checks use Python sources from accepted S3a commit
`2b9b84eb20b0be89d728b963cc753924e83b53ea`. They compare command parsing and
temporary fixture outputs with network access blocked; no research inputs are mined
or evaluated. See [data-processing checks](data_preprocess/README.md#structural-checks).

Loading equivalence checks use Git baseline
`f528183164dde019351ad5b00d3f60f354c69989`, temporary CSVs and fake tokenizer/model
factories. They compare BOM handling, row order, read errors and loading calls without
executing the research pipeline or downloading a model. The shared loader passes
`revision=None`, equivalent to the omitted revision in the previous entry points for
the installed dependency versions; it does not pin a new revision. Tokenizer loading
still precedes model loading, then callers move the model to the device and call
`eval()` without adding tokens or setting a seed. These structural checks do not
require rerunning the cached-model notebook smoke.

The notebook runner executes every code cell in a fresh Python process. This checks
Python cell execution, not the Jupyter interface or kernel setup. Jupyter/IPython are
not installed by `requirements.txt` and are not needed for this runner. The first code cell contains imports, paths and mode settings.
`MODE="smoke"` uses invented rows and the cached real DictaBERT model, in CPU evaluation
mode without gradients. Set `DICTABERT_SNAPSHOT` to an existing local snapshot directory
if automatic discovery is unsuitable. A missing cache stops the notebook with `NOT RUN`;
it never downloads or substitutes a different model.

The clean-install smoke results are recorded in [Start here](#start-here). They
verify the notebook as a code appendix, not historical research results.

The explicit `train` mode is retained for later authorized use. It requires local
train/development CSV paths and a new checkpoint output path in an existing directory.
Both modes require a model snapshot prepared locally before starting the process.
Both install a network-blocking audit hook for the rest of that Python process; restart
the kernel/process before unrelated network work. Colab setup and external GPU execution
have not been validated. Training does not download a model automatically.
Settings remain in `TrainingConfig`; the seed is applied after model initialization,
and `best.pt` remains selected by strict improvement in development **pair loss**.
Pair accuracy is not per-item candidate-selection accuracy. Full training, real dev/test
scoring, GPU execution and historical checkpoint validation were not run in S3a.

Known encoding edge behavior remains unchanged: missing markers/empty batches raise
errors; an oversized candidate or target may exceed the length budget. Evaluators
retain their distinct candidate selection, scoring and skip rules; shared loading does
not make them one evaluator. Scientific decisions remain pending; these structural
checks do not approve a protocol or establish model quality.

The old Colab instructions fetched code/data from a moving branch and ran training.
They have been replaced by the local appendix above. No analysis notebook or final
experiment protocol has been implemented in this package.

## Historical results and interpretation

The following tables and interpretations are preserved from the baseline README.
Their scientific claims were not revalidated by the environment check. References
to a next run below are historical plans, not current execution instructions.

> **Stale as of 2026-09-13.** Every number below was measured on an earlier dev set
> (285 items, later corrected by hand to 292 — see the note further down). Since then,
> dev has had a full human review (measured substitution damage rate: 7.3%, see
> `data/mined/DATASET_CARD.md`) plus a small number of `deglossed`/`authored` rows added,
> and now stands at **289 items over 55 acronym types**. `pipeline/run_pipeline.sh` has
> not yet been re-run against it, so every figure below should be treated as
> **not yet verified against the current data** rather than corrected or withdrawn.
> Re-running the pipeline was the earlier proposed next step; it is deferred pending approval.

Dev split (as measured below): 285 items over 55 acronym types, none seen during training.

| Method | Accuracy | |
|---|---|---|
| Random | 0.338 | floor — 1/n_candidates per item |
| Most frequent sense | 0.488 | by Wikipedia hit count |
| Most attested sense | 0.533 | by sentences actually mined — the stronger frequency bar |
| Untrained DictaBERT | 0.692 | `[CLS]` cosine similarity, no training at all |
| **Fine-tuned cross-encoder** | **0.786** | 1 epoch, lr 2e-5, batch 16, seed 42 |
| Oracle | 1.000 | ceiling — gold is always among the candidates |

### Open generation vs. candidate-constrained selection

The comparison the project is actually about: does a general-purpose LLM, given the same
sentence, do better freely generating an expansion or picking one from the candidate
list? Same 285-item dev set, same prompts, two formulations per model.

| Model | Mode | Accuracy | Invalid rate |
|---|---|---|---|
| Qwen2.5:7b (local, via Ollama) | Generate (no candidates shown) | 0.021 | 0.979 |
| Qwen2.5:7b (local, via Ollama) | Select (candidates shown, shuffled) | 0.628–0.635 | ~0.005 |
| Gemini Flash-Lite (hosted) | Generate (no candidates shown) | 0.477–0.502 | 0.425–0.467 |
| Gemini Flash-Lite (hosted) | Select (candidates shown, shuffled) | 0.867 | 0.000 |
| Gemini 3.6 Flash, thinking on (hosted) | Generate (no candidates shown) | 0.723 | 0.242 |
| Gemini 3.6 Flash, thinking off (hosted) | Generate (no candidates shown) | 0.698 | 0.284 |
| Gemini 3.6 Flash, thinking on (hosted) | Select (candidates shown, shuffled) | 0.951 | 0.000 |
| Gemini 3.6 Flash, thinking off (hosted) | Select (candidates shown, shuffled) | **0.954** | 0.000 |

"Invalid rate" is how often the response matches none of the item's candidates — the
concrete cost of unconstrained generation. Select mode shuffles candidate order per item
(scored by decoded letter, not position) specifically because early testing found Qwen
defaults to always answering the first-shown option on the hardest items rather than
guessing from content — see `results/qwen/select_details.csv`'s `shown_order` column.

**Candidate-constrained selection helps every model tried so far**, and helps weak
models most: Qwen2.5:7b goes from 0.021 to ~0.63 just by being handed the candidate
list, Gemini Flash-Lite goes from ~0.49 to 0.867, and Gemini 3.6 Flash goes from 0.72 to
**0.951** — comfortably above the fine-tuned DictaBERT cross-encoder's 0.786–0.825, with
zero task-specific training. That is the headline finding: for this task, prompting a
strong general model with the candidate list already visible outperforms training a
small model specifically for it, by a wide margin once the model is strong enough.

**Thinking mode does not matter for this task, in either formulation.** Gemini 3.6 Flash
scores 0.723 (thinking) vs. 0.698 (minimized) on generate mode, and 0.951 vs. 0.954 on
select mode — both gaps are inside the run-to-run noise band already established for
DictaBERT, and select mode's gap is in the *opposite* direction, confirming it is noise
rather than a real effect. Select mode's ~0.95 ceiling needs a strong base model, not
extended reasoning: thinkingBudget=1 reaches it in roughly 60% of the wall-clock time
(9 vs. 15 minutes for 285 items) at presumably lower cost, with no accuracy cost.

The dev set was 292 items until the model's errors were reviewed by hand: 3 carried a
wrong gold label and were corrected, and 7 had no defensible single answer and were
removed. The untrained-DictaBERT figure predates that and is the one number above still
measured on 292.

**Excluding rabbinic-name acronyms, the same model scores 0.855** on the remaining 248
items. `מהר״ש`, `מהר״י`, `מהרי״א`, `מהרי״ץ` and `יעב״ץ` each offer several different
rabbis sharing an initial, so choosing between them needs to know which one lived in Belz
and which in Lubavitch — biography, not context. They are 13% of dev and 41% of its
errors, at a 68% error rate against 15% for every other type. Both figures are worth
reporting: the first is the task as posed, the second is the task the method is actually
suited to.

Reproduce the first four with `model/baselines.py` and `model/dictabert/eval.py`.

**Fine-tuning is worth 12.3 points over the untrained encoder**, not the 29 points the
frequency baselines alone would suggest. Most of the work is already done by DictaBERT's
Hebrew pretraining: an off-the-shelf encoder with no task supervision reaches 0.692 just
by asking which candidate is most similar to the sentence. That is the honest comparison,
and it is the more interesting one.

**Three runs of that configuration span 4.5 points** (0.770, 0.781, 0.815 on the
uncorrected 292-item dev) on nothing but batch order and dropout. A single run therefore
cannot establish a gain smaller than roughly five points, which is larger than most
configuration changes are worth — so treat any single-run comparison below that margin as
undecided rather than as a result. Pooling from the marked span rather than `[CLS]` was
tried on that basis and made no difference.

**Training saturates after one epoch.** Dev loss rose from 0.42 to 0.53 to 0.54 across
three epochs while train loss kept falling (0.178 → 0.106 → 0.072), so the selected
checkpoint is epoch 1 and the notebook now defaults to it. A model starting from a strong
pretrained position has less left to learn, and 2,966 weakly-labelled examples are
exhausted quickly.

**Where the errors are.** 25 of the 61 errors are the rabbinic-name types described
above, and their margins are frequently under 0.1 — the model is not confidently wrong
there so much as unable to choose. Of the rest, gold ranked second in 40 of 61 cases and
39 were near-misses under a 1.0 margin. Hand review of the non-rabbinic errors from an
earlier run found the model genuinely wrong in 27 of 37, so label noise is not what caps
the score.

### What these numbers are not

Every figure above is measured on **substituted** text, where the mining pipeline rewrote
a sentence that spelled the expansion out. Such a sentence was *written about* that
meaning, so its topic words point at the answer. Real Hebrew, where a writer chose to
abbreviate, is a different and harder distribution — accuracy there should be expected to
be lower, and the gap between the two is the number worth reporting. Scoring against the
held-out natural-usage set is the next step.

## Data and evidence status

Read the [data location map](data/README.md), then the
[dataset card](data/mined/DATASET_CARD.md) for definitions, construction history and
limitations. [The split document](data/splits/README.md) describes the existing files.
Processing layer, text origin/construction, label status and split role are separate
axes. A reviewed file or `gold_expansion` column does not prove approval of the next
experiment; `manual` includes disclosed AI-authored text. Research inputs and results
remain unchanged by structural organization and handoff checks.

<details>
<summary>Historical inventory summary retained from the setup baseline (not recounted during handoff)</summary>

| File | Rows | Types | What |
|---|---|---|---|
| `data/splits/train_items.csv` | 3,115 | 435 | Training split |
| `data/splits/dev_items.csv` | 289 | 55 | Historical development input |
| `data/splits/test_items.csv` | 395 | 60 | Held-out test split, built from Knesset + natural + authored rows |
| `data/splits/all_items.csv` | 4,649 | 549 | Separate historical aggregate export; not guaranteed to equal the current split union |
| `data/splits/by_category/*.csv` | — | — | The same rows split one file per `category` value |
| `data/mined/acronym_items.csv` | 3,386 | — | Wikipedia mining output — the source most of train/dev were drawn from |
| `data/mined/knesset/knesset_reviewed.csv` | 1,041 | 191 | Human-reviewed Knesset Corpus rows — the source test/part of train were drawn from |
| `data/mined/candidate_table.csv` | 2,700+ | 638 | Acronym → expansion inventory. An input to mining, not a result |

The baseline also reported approximately 3,250 substituted Wikipedia rows and 229
`manual` rows in its aggregate description. These historical counts were not reconciled
in this structural package; use the source files and identified revision for analysis.

</details>

## Historical data regeneration instructions

`data_preprocess/` contains historical collection and construction steps; see
[its documentation](data_preprocess/README.md). These scripts do not alone reconstruct
every later reviewed or authored addition to the committed splits. A full sweep is thousands of throttled API calls (Wikipedia)
or a multi-GB download (the Knesset Corpus shards) — hours either way — which is why the
mined output is committed rather than regenerated on demand.

Source text is Hebrew Wikipedia, Wiktionary, and the
[Knesset Proceedings Corpus](https://huggingface.co/datasets/HaifaCLGroup/KnessetCorpus).
Sefaria (rabbinic/Talmudic text) was tried and set aside — see the DATASET_CARD's
"Scope decisions" for why.

## Historical status and next steps (baseline snapshot)

These earlier instructions are deferred. Do not run `pipeline/run_pipeline.sh` or
`python -m pipeline.run_all` during setup: the shell entry point automatically evaluates
test when its file exists, and `--skip-llm` does not disable test. No training,
benchmark evaluation, mining, Ollama, or paid API use is part of the current package.

- `weights/` holds a `dictabertX` checkpoint trained **before** this session's data
  changes (Knesset Corpus, the frozen test split, the thin-sense fixes). Any future
  retraining requires approved inputs and a prepared environment as described in the
  training appendix; these historical notes do not authorize it.
- `pipeline/run_pipeline.sh` has not yet been re-run against the current data — the
  Results section above is stale and says so. Re-run it (with a fresh checkpoint) and
  update Results once training is done.
- `test_items.csv` exists and validates cleanly but has not been evaluated by any arm
  yet — that comparison (substituted dev vs. natural-text test) is the next real result
  to produce.
