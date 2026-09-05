# `checkpoints/`

Trained model weights. **Not committed** — each file is ~700MB, over GitHub's 100MB
limit, and they are reproducible by re-running the notebook.

Naming: `dictabert-crossenc-<YYYYMMDD-HHMMSS>.pt`, timestamped at the run that
produced it.

## Runs

| Checkpoint | Pooling | Config | Dev item accuracy |
|---|---|---|---|
| `dictabert-crossenc-20260905-181754.pt` | cls | 3 epochs, epoch 1 selected; lr 2e-5, batch 16, unseeded | **0.815** |
| (not kept) | cls | 1 epoch, lr 2e-5, batch 16, unseeded | 0.781 |
| (not kept) | cls | 1 epoch, lr 2e-5, batch 16, seed 42 | 0.770 |
| (not kept) | span_mean | 1 epoch, lr 2e-5, batch 16, seed 42 | <0.80 |
| current | cls | 1 epoch, lr 2e-5, batch 16, seed 42, **corrected dev** | **0.786** |

The last row is the first number measured on the corrected 285-item dev set; everything
above it was scored against the earlier 292-item version, which carried three wrong gold
labels and seven items with no defensible answer. The two are not comparable, and 0.786
is the figure to compare future runs against.

Excluding the rabbinic-name types (`מהר״ש`, `מהר״י`, `מהרי״א`, `מהרי״ץ`, `יעב״ץ`) it
scores **0.855** on the remaining 248 items. Those five types run a 68% error rate
against 15% everywhere else — they are 13% of dev and 41% of its errors.

Epochs 2 and 3 of the first run only overfit — dev loss rose from 0.42 to 0.53 to 0.54
while train loss kept falling — so its saved checkpoint is epoch 1 and the notebook now
defaults to `EPOCHS = 1`.

**All three rows above are the same configuration** — one effective epoch of `cls`
pooling at lr 2e-5 — and they span 0.770 to 0.815, a spread of 4.5 points on nothing but
batch order and dropout. Treat that as the noise floor: **a single run cannot establish a
gain smaller than roughly 5 points.** Comparing two configurations honestly means either
running each several times and comparing the spread, or accepting that only a large gap
is evidence.

The notebook pins `SEED = 42`, which makes any given configuration repeatable, but that
does not shrink the variance — it only fixes which draw you get. Two *different*
configurations under the same seed still differ by an unknown amount of luck.

## Pooling variants

`POOLING` in the model cell selects which vector the scoring head reads:

| | what it reads |
|---|---|
| `cls` | the `[CLS]` summary of the whole pair — the conventional default |
| `marker` | the `[ACR]` token's own vector, sitting on the target span |
| `span_mean` | mean of the acronym's own subword tokens, between the markers |
| `concat` | `[CLS]` and `span_mean` together, `2 x hidden` into the head |

`[CLS]` summarises the sentence; the question is about one span within it. The other
three read that span directly, which is standard for span-targeted tasks.

**It made no difference.** `span_mean` landed in the same range as `cls`, inside the
noise floor. A plausible reason: the acronym's own tokens carry no meaning — `מ״מ`
tokenises to `מ`, `״`, `מ`, the same two characters for all sixteen of its senses — so
after twelve layers of attention their vectors encode the same surrounding context
`[CLS]` already summarises. Where you pool from does not matter when the span itself is
semantically empty and only its context disambiguates it.

`marker` and `concat` were left untried on that reasoning.

## Loading one

The tokenizer gains two tokens (`[ACR]`, `[/ACR]`) and the embedding matrix is resized
to match before training. Any code loading these weights must repeat that step first,
or `load_state_dict` fails on a shape mismatch. See `model/encoder.py`.
