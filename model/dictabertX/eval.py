"""Fine-tuned DictaBERT cross-encoder on the ranking task — the candidate-constrained
selection arm this project trained, evaluated locally against a checkpoint instead of
inside the training notebook.

Requires a trained checkpoint (see weights/README.md — not committed, ~700MB).
Notebook smoke writes no checkpoint; historical checkpoint reproduction is unverified.

    python -m model.dictabertX.eval --checkpoint weights/dictabert-crossenc-<ts>.pt
"""
from __future__ import annotations

import argparse

import torch
from tqdm import tqdm

from model.common.pairs import MIN_CANDIDATES, find_span, load_rows, mark_span
from model.dictabertX.encoding import encode_pairs
from model.dictabertX.model import load_finetuned

MAX_LEN = 256


def encode_batch(tok, device, acr_open_id, acr_close_id,
                  context: str, candidates: list[str]) -> dict:
    """One context and its candidates, using the shared target-centred encoding."""
    if not candidates:
        # The previous evaluator checked markers before rejecting an empty batch.
        ctx_ids = tok.encode(context, add_special_tokens=False)
        ctx_ids.index(acr_open_id)
        ctx_ids.index(acr_close_id)
    return encode_pairs(tok, [(context, candidate) for candidate in candidates], device,
                        max_len=MAX_LEN, acr_open_id=acr_open_id, acr_close_id=acr_close_id)


@torch.no_grad()
def evaluate(rows: list[dict], tok, model, acr_open_id, acr_close_id, device: str) -> dict:
    correct = total = 0
    for r in tqdm(rows, desc="dictabertX (fine-tuned) eval", unit="item"):
        cands = [c.strip() for c in r["candidates"].split("|") if c.strip()]
        gold = r["gold_expansion"].strip()
        span = find_span(r["sentence"], r["acronym"])
        if len(cands) < MIN_CANDIDATES or not gold or span is None:
            continue
        marked = mark_span(r["sentence"], span)

        enc = encode_batch(tok, device, acr_open_id, acr_close_id, marked, cands)
        logits = model(**enc, acr_open_id=acr_open_id, acr_close_id=acr_close_id)
        pred = cands[int(torch.argmax(logits).item())]

        correct += int(pred == gold)
        total += 1

    return {"n_items": total, "accuracy": round(correct / total, 4) if total else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", default="data/splits/dev_items.csv")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--pooling", default="cls",
                    choices=["cls", "marker", "span_mean", "concat"])
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok, model, acr_open_id, acr_close_id = load_finetuned(
        a.checkpoint, pooling=a.pooling, device=device)

    res = evaluate(load_rows(a.items), tok, model, acr_open_id, acr_close_id, device)
    print(f"{a.items}  (checkpoint: {a.checkpoint}, pooling: {a.pooling})\n")
    print(f"  items scored      {res['n_items']}")
    print(f"  accuracy          {res['accuracy']:.3f}")


if __name__ == "__main__":
    main()
