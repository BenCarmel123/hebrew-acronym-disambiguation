# Hebrew Acronym Disambiguation

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
data_preprocess/   builds the dataset from Hebrew Wikipedia + Wiktionary
model/
  common/          pairs.py (span-marking, pair-building); eval.py (shared LLM-eval loop)
  baselines.py     random / most-frequent / most-mined / oracle — no model, just stats
  dictabert/       untrained DictaBERT baseline (model.py, eval.py)
  dictabertX/      fine-tuned cross-encoder (model.py, eval.py — training is in the notebook)
  qwen/            local open LLM arm, via Ollama (eval.py)
  gemini/          hosted SOTA LLM arm (eval.py)
notebooks/         Colab training notebook (dictabertX only)
checkpoints/       trained weights (gitignored — ~700MB each)
data/              the dataset — see data/README.md for the layer rules
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Training

Open `notebooks/train_dictabert.ipynb` in Colab
([direct link](https://colab.research.google.com/github/BenCarmel123/hebrew-acronym-disambiguation/blob/main/notebooks/train_dictabert.ipynb)),
set `Runtime > Change runtime type > T4 GPU`, and run all cells. It pulls the data and
`model/common/pairs.py` from this repo, so there is nothing to upload.

## Results

Dev split: 285 items over 55 acronym types, none seen during training.

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
guessing from content — see `data/mined/llm_select_details.csv`'s `shown_order` column.

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

| File | Rows | What |
|---|---|---|
| `data/splits/train_items.csv` | 2,966 | Training split, 491 acronym types |
| `data/splits/dev_items.csv` | 292 | Dev split, 55 acronym types |
| `data/mined/acronym_items.csv` | 3,386 | Every mined occurrence — the source the splits were drawn from |
| `data/mined/candidate_table.csv` | 2,002 | Acronym → expansion inventory, 638 types. An input to mining, not a result |

`data/splits/` holds only what training reads. Everything the mining pipeline
produced, including the full occurrence table, stays in `data/mined/`.

Train and dev are **disjoint by acronym type**, so dev measures generalization to
acronyms never seen in training rather than recall of a memorized expansion.

### Label status

| status | rows | meaning |
|---|---|---|
| `weak` | 3,258 | Correct by construction — the mining pipeline substituted the acronym into a sentence that spelled the expansion out. Usable for training; not evaluation data |
| `verified` | 115 | Hand-annotated natural-usage sentences |
| `unverified` | 13 | Skipped during annotation, left unlabeled |

`data/splits/DATASET_CARD.md` documents the mining process, every filter, and the
known limitations — including that the data is 96% substituted rather than natural, and
that a spot-check suggested roughly a third of substituted rows may read awkwardly. Read
it before quoting any number.

## Regenerating the data

`data_preprocess/` builds the dataset from scratch against the live wikis; see
`data_preprocess/README.md`. A full sweep is thousands of throttled API calls over
hours, which is why the mined output is committed rather than regenerated on demand.

Source text is Hebrew Wikipedia and Wiktionary.
