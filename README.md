# Hebrew Acronym Disambiguation

**Active stage: workspace organization (S1 and an early local part of S2).**
The scientific protocol has not been approved. Existing results and research claims
below are historical; they are not current findings or authorization to run experiments.

## Start here

Use this repository as the permanent working copy. The initial setup branch is
`setup-workspace`, based on commit `eb2e7785dab42dc8ae3ca07372d032adafee9dba`
of [BenCarmel123/hebrew-acronym-disambiguation](https://github.com/BenCarmel123/hebrew-acronym-disambiguation).
On 2026-09-19, remote `main` and `improve-data` both pointed to that commit.
The old [nlp-hw-team repository](https://github.com/ShakedSchnarch/nlp-hw-team)
is a read-only historical reference; its history is not merged here.

From the repository root, create a new local environment with Python 3.12 and install
from the single dependency file (do not reuse an unrelated environment):

```bash
python3.12 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
```

Run the local check in a fresh process:

```bash
.venv/bin/python -B -m pipeline.check_environment
```

The check reports Python, installed packages, OS and device availability; imports the
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

Verified on 2026-09-19: Python 3.12.11, macOS 26.6.2 arm64, CPU available,
CUDA unavailable, MPS available (availability only; no MPS execution).
With local revision `8884c6db002aba4002ee638fe4070c92e9ffbbf1`, imports, all four
invented pairs, tokenization and the CPU forward passed; output shape was `(2, 23, 768)`
and all values were finite. Loading reported newly initialized BERT pooler weights;
the check inspects `last_hidden_state`, not pooled scores or model quality.
This revision identifies the check only and does not change the model's research default.

This is a base-tokenizer/encoder check, not validation of the notebook's custom encoding,
training loop, trained checkpoints, or scientific protocol. The external GPU environment
was **not verified in this package**. S2 is not complete.

Reader map: [data layers](data/README.md), [dataset card](data/mined/DATASET_CARD.md),
[existing notebook](notebooks/train_dictabert.ipynb) (read-only in this stage), and
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

## Layout

```
data_preprocess/   builds the dataset from Hebrew Wikipedia, Wiktionary, and the Knesset
                    Proceedings Corpus (see per-source subpackages: wikipedia/, wiktionary/,
                    knesset/, sefaria/ — the last exploratory, not wired into any pipeline)
  common/          hebrew_text.py (orthography, gematria), filters.py (shared mining filters)
  build_splits.py  constructs a type-disjoint test split from newly-reviewed source data
model/
  common/          pairs.py (span-marking, pair-building); eval.py (shared LLM-eval loop)
  baselines.py     random / most-frequent / most-mined / oracle — no model, just stats
  dictabert/       untrained DictaBERT baseline (model.py, eval.py)
  dictabertX/      fine-tuned cross-encoder (model.py, eval.py — training is in the notebook)
  qwen/            local open LLM arm, via Ollama (eval.py)
  gemini/          hosted SOTA LLM arm (eval.py)
notebooks/         Colab training notebook (dictabertX only)
weights/       trained weights (gitignored — ~700MB each)
data/              the dataset — see data/README.md for the layer rules
pipeline/          validate_data.py, run_all.py, run_pipeline.sh — one command, one table
results/           eval outputs per arm — per-item CSVs and the combined summary table
```

## Historical training instructions — not the current entry point

Do not run these instructions during the workspace-organization stage.

Open `notebooks/train_dictabert.ipynb` in Colab
([direct link, `improve-data` branch](https://colab.research.google.com/github/BenCarmel123/hebrew-acronym-disambiguation/blob/improve-data/notebooks/train_dictabert.ipynb)),
set `Runtime > Change runtime type > T4 GPU`, and run all cells. It pulls the data and
`model/common/pairs.py` from this repo, so there is nothing to upload. Point it at the
branch that actually has the `train_items.csv` you want to train on; both branches pointed to the same base commit at setup (see Start here).

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

## Data

| File | Rows | Types | What |
|---|---|---|---|
| `data/splits/train_items.csv` | 3,115 | 435 | Training split |
| `data/splits/dev_items.csv` | 289 | 55 | Dev split — frozen; never rewritten once reviewed |
| `data/splits/test_items.csv` | 395 | 60 | Held-out test split, built from Knesset + natural + authored rows |
| `data/splits/all_items.csv` | 4,649 | 549 | Every row across train/dev/test, one file, with a `category` column |
| `data/splits/by_category/*.csv` | — | — | The same rows split one file per `category` value |
| `data/mined/acronym_items.csv` | 3,386 | — | Wikipedia mining output — the source most of train/dev were drawn from |
| `data/mined/knesset/knesset_reviewed.csv` | 1,041 | 191 | Human-reviewed Knesset Corpus rows — the source test/part of train were drawn from |
| `data/mined/candidate_table.csv` | 2,700+ | 638 | Acronym → expansion inventory. An input to mining, not a result |

`data/splits/` holds only what training/eval reads. Everything the mining pipeline
produced, including the full occurrence tables, stays in `data/mined/`.

**Train and dev are disjoint by acronym type**, so dev measures generalization to
acronyms never seen in training rather than recall of a memorized expansion. **Test is
disjoint from both train and dev** — this required actively removing 60 types' existing
rows from train and rebuilding them from the newer source pool (Knesset, natural
Wikipedia usage, and disclosed AI-authored examples); see
`data_preprocess/build_splits.py` and `data/mined/DATASET_CARD.md`'s
"Knesset corpus, natural-text pool, and the frozen test split" section for why that
carve-out was necessary rather than optional.

### The `category` column

Every row in `all_items.csv` (and the split files) carries one of six values:

| category | rows | meaning |
|---|---:|---|
| `wiki_substituted` | ~3,250 | Wikipedia, mechanically rewritten from a spelled-out phrase |
| `knesset` | 1,041 | Knesset Corpus, human-reviewed |
| `wiki_natural` | 112 | Wikipedia, real unedited usage |
| `manual` | 229 | Disclosed AI-authored (`source=claude-sonnet-5`) — written to fill specific sense gaps, never passed off as mined |
| `wiki_deglossed` | 16 | Wikipedia, real usage with an inline gloss removed |
| `wiktionary` | 0 | Placeholder — no such rows exist in the current dataset (header-only file) |

`wiki_substituted` vs. `wiki_deglossed`: both start from Wikipedia prose, but
substitution *invents* the abbreviated form (a sentence that never used the acronym is
rewritten to use it), while deglossing finds a sentence where a Wikipedia author
**already used the acronym** and only removes a nearby parenthetical explanation.

### Label status

| status | meaning |
|---|---|
| `weak` | Correct by construction — the mining pipeline substituted the acronym into a sentence that spelled the expansion out. Usable for training; not held-out evaluation data |
| `verified` | Hand-annotated or human-reviewed (natural Wikipedia usage, Knesset review, or disclosed AI-authored) |
| `unverified` | Skipped during annotation, left unlabeled — dropped wherever it would leave a row with no gold answer |

`data/mined/DATASET_CARD.md` documents the mining process, every filter, and the known
limitations — including a full human review of the dev split's substituted rows (a
measured 7.3% damage rate, 95% CI 4.1–10.6%), the Knesset Corpus addition and its review,
the systematic gematria/privacy-redaction candidate fixes, and a known accepted leakage
condition (page-title overlap between train/dev and train/test — judged low severity;
see the card for why). Read it before quoting any number.

## Historical data regeneration instructions

`data_preprocess/` builds the dataset from scratch against the live sources; see
`data_preprocess/README.md`. A full sweep is thousands of throttled API calls (Wikipedia)
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
  changes (Knesset Corpus, the frozen test split, the thin-sense fixes). Retrain against
  the current `data/splits/train_items.csv` in Colab before trusting that arm's numbers.
- `pipeline/run_pipeline.sh` has not yet been re-run against the current data — the
  Results section above is stale and says so. Re-run it (with a fresh checkpoint) and
  update Results once training is done.
- `test_items.csv` exists and validates cleanly but has not been evaluated by any arm
  yet — that comparison (substituted dev vs. natural-text test) is the next real result
  to produce.
