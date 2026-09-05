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
model/             encoder.py loads DictaBERT; pairs.py formats model inputs
                   baselines.py and zero_shot.py score the comparisons
notebooks/         Colab training notebook
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
`model/pairs.py` from this repo, so there is nothing to upload.

## Results

Dev split: 285 items over 55 acronym types, none seen during training.

| Method | Accuracy | |
|---|---|---|
| Random | 0.338 | floor — 1/n_candidates per item |
| Most frequent sense | 0.488 | by Wikipedia hit count |
| Most attested sense | 0.533 | by sentences actually mined — the stronger frequency bar |
| Untrained DictaBERT | 0.692 | `[CLS]` cosine similarity, no training at all |
| **Fine-tuned cross-encoder** | **~0.82** | 1 epoch, lr 2e-5, batch 16 |
| Oracle | 1.000 | ceiling — gold is always among the candidates |

The dev set was 292 items until the model's errors were reviewed by hand: 3 carried a
wrong gold label and were corrected, and 7 had no defensible single answer and were
removed. The untrained-DictaBERT figure predates that and is the one number above still
measured on 292.

Reproduce the first four with `model/baselines.py` and `model/zero_shot.py`.

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

**Where the errors are.** Of 61 errors on the uncorrected dev set, roughly 24 were
rabbinic-name acronyms — `מהר״ש`, `מהר״י`, `מהרי״א` — whose candidates are different
rabbis sharing an initial. Telling `רבי שלום רוקח` from `רבי שלמה מלובלין` requires
knowing which one lived in Belz, which is biographical knowledge rather than contextual
reasoning, and the margins there are near zero. Hand review of the other 37 found the
model genuinely wrong in 27 of them, so label noise is not what caps the score.

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
