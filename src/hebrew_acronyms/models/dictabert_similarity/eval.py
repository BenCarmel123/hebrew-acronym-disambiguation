"""Rank candidates by cosine similarity using pretrained DictaBERT.

The marked context and each candidate are encoded separately. Their [CLS] vectors
are compared by cosine similarity, and the highest-scoring candidate is selected.
This method uses no task-specific training or learned scoring head.

It is a separate baseline from the trained cross-encoder, which jointly encodes
each context/candidate pair and applies a learned head. Their comparison changes
both the scoring method and training, so it does not isolate the effect of fine-tuning.

    python -m hebrew_acronyms.models.dictabert_similarity.eval --items data/splits/dev_items.csv
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from tqdm import tqdm

from hebrew_acronyms.models.common.pairs import find_span, load_rows, mark_span
from hebrew_acronyms.models.dictabert_similarity.model import MODEL_ID, build_model

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
    for r in tqdm(rows, desc="dictabert (untrained) eval", unit="item"):
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
    tok, model = build_model(MODEL_ID)
    model = model.to(device)
    # No dropout, and the markers are NOT added to the vocabulary here: an untrained
    # embedding row would be pure noise, so the markers stay as ordinary subword text
    # the pretrained model can at least read.
    model.eval()

    rows = load_rows(a.items)

    res = evaluate(rows, tok, model, device)
    print(f"{a.items}\n")
    print(f"  items scored          {res['n_items']}")
    print(f"  zero-shot accuracy    {res['accuracy']:.3f}   untrained DictaBERT, "
          f"[CLS] cosine similarity")


if __name__ == "__main__":
    main()
