"""Untrained DictaBERT on the ranking task — what the encoder knows before fine-tuning.

Isolates what fine-tuning actually contributed. The cross-encoder head is randomly
initialised, so scoring pairs through an untrained head measures nothing but noise;
instead this uses the pretrained encoder the way it can be used without training —
EMBEDDING SIMILARITY.

Each candidate is scored by the cosine similarity between the marked context's [CLS]
vector and the candidate's own [CLS] vector, and the argmax is the prediction. That is a
real zero-shot method (it asks "which expansion is most semantically similar to this
sentence?"), and it needs no labels, no head and no training.

WHAT A LOW NUMBER HERE MEANS. Not that DictaBERT is weak. Sentence similarity via [CLS]
is known to be a poor sentence representation without fine-tuning — BERT's [CLS] is
trained for next-sentence prediction, not similarity, which is the finding that motivated
Sentence-BERT. Read this as the floor a pretrained encoder gives you off the shelf, and
the distance fine-tuning had to travel.

    python -m model.dictabert.eval --items data/splits/dev_items.csv
"""
from __future__ import annotations

import argparse
import csv

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from model.common.pairs import find_span, mark_span

MODEL_ID = "dicta-il/dictabert"
MAX_LEN = 256


@torch.no_grad()
def embed(tok, model, texts: list[str], device: str) -> torch.Tensor:
    """-> (n, hidden) [CLS] vectors."""
    enc = tok(texts, padding=True, truncation=True, max_length=MAX_LEN,
              return_tensors="pt").to(device)
    return model(**enc).last_hidden_state[:, 0]


@torch.no_grad()
def evaluate(rows: list[dict], tok, model, device: str) -> dict:
    correct = total = 0
    for r in rows:
        cands = [c.strip() for c in r["candidates"].split("|") if c.strip()]
        gold = r["gold_expansion"].strip()
        span = find_span(r["sentence"], r["acronym"])
        if len(cands) < 2 or not gold or span is None:
            continue
        marked = mark_span(r["sentence"], span)

        vecs = embed(tok, model, [marked] + cands, device)
        sims = F.cosine_similarity(vecs[0:1], vecs[1:])
        correct += int(cands[int(sims.argmax())] == gold)
        total += 1

    return {"n_items": total, "accuracy": round(correct / total, 4) if total else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", default="data/splits/dev_items.csv")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device)
    # No dropout, and the markers are NOT added to the vocabulary here: an untrained
    # embedding row would be pure noise, so the markers stay as ordinary subword text
    # the pretrained model can at least read.
    model.eval()

    with open(a.items, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    res = evaluate(rows, tok, model, device)
    print(f"{a.items}\n")
    print(f"  items scored          {res['n_items']}")
    print(f"  zero-shot accuracy    {res['accuracy']:.3f}   untrained DictaBERT, "
          f"[CLS] cosine similarity")


if __name__ == "__main__":
    main()
