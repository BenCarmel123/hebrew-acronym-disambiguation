"""Fine-tuned DictaBERT cross-encoder on the ranking task — the candidate-constrained
selection arm this project trained, evaluated locally against a checkpoint instead of
inside the Colab notebook.

Requires a trained checkpoint (see checkpoints/README.md — not committed, ~700MB;
reproduce it by running notebooks/train_dictabert.ipynb).

    python -m model.dictabertX.eval --checkpoint checkpoints/dictabert-crossenc-<ts>.pt
"""
from __future__ import annotations

import argparse

import torch

from model.common.pairs import MIN_CANDIDATES, find_span, load_rows, mark_span
from model.dictabertX.model import load_finetuned

MAX_LEN = 256


def encode_batch(tok, device, acr_open_id, acr_close_id,
                  context: str, candidates: list[str]) -> dict:
    """One context, several candidates -> a batch of (context, candidate) pairs.

    Mirrors the notebook's encode_batch: truncates the context around the marked span
    when a pair would exceed MAX_LEN, keeping [ACR]...[/ACR] intact.
    """
    cls_id, sep_id = tok.cls_token_id, tok.sep_token_id
    rows = []
    ctx_ids = tok.encode(context, add_special_tokens=False)
    o, c = ctx_ids.index(acr_open_id), ctx_ids.index(acr_close_id)
    for candidate in candidates:
        cand_ids = tok.encode(candidate, add_special_tokens=False)
        budget = MAX_LEN - 3 - len(cand_ids)
        lo, hi = 0, len(ctx_ids)
        while (hi - lo) > budget:
            left_room, right_room = o - lo, hi - (c + 1)
            if left_room >= right_room and left_room > 0:
                lo += 1
            elif right_room > 0:
                hi -= 1
            else:
                break
        trimmed = ctx_ids[lo:hi]
        ids = [cls_id] + trimmed + [sep_id] + cand_ids + [sep_id]
        types = [0] * (len(trimmed) + 2) + [1] * (len(cand_ids) + 1)
        rows.append((ids, types))

    width = max(len(ids) for ids, _ in rows)
    pad = tok.pad_token_id or 0
    input_ids, attention_mask, token_type_ids = [], [], []
    for ids, types in rows:
        n = width - len(ids)
        input_ids.append(ids + [pad] * n)
        attention_mask.append([1] * len(ids) + [0] * n)
        token_type_ids.append(types + [0] * n)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long, device=device),
    }


@torch.no_grad()
def evaluate(rows: list[dict], tok, model, acr_open_id, acr_close_id, device: str) -> dict:
    correct = total = 0
    for r in rows:
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
