"""Identified cross-encoder predictions; no research metric or denominator policy."""
from __future__ import annotations

import argparse
import json

import torch

from hebrew_acronyms.models.common.pairs import candidates_for, explicit_span, load_rows, mark_span, validate_ids
from hebrew_acronyms.models.dictabert_cross_encoder.encoding import encode_pairs
from hebrew_acronyms.models.dictabert_cross_encoder.model import load_finetuned

MAX_LEN = 256


def encode_batch(tok, device, acr_open_id, acr_close_id,
                 context: str, candidates: list[str], max_len=None) -> dict:
    return encode_pairs(tok, [(context, candidate) for candidate in candidates], device,
                        max_len=MAX_LEN if max_len is None else max_len,
                        acr_open_id=acr_open_id, acr_close_id=acr_close_id)


@torch.no_grad()
def evaluate(rows: list[dict], tok, model, acr_open_id, acr_close_id, device: str,
             max_len: int | None = None) -> list[dict]:
    """One ordered record per identified item, including singleton and failed inputs.

    Missing/duplicate IDs reject the whole request: results would not be identifiable.
    Invalid rows, encoding errors and forward runtime failures retain distinct
    statuses. Setup/identity errors reject the request. Scores are raw logits.
    """
    validate_ids(rows)
    saved_config = getattr(model, "training_config", None)
    if saved_config is not None:
        if max_len is not None and max_len != saved_config.max_len:
            raise ValueError("max_len override disagrees with model training configuration")
        max_len = saved_config.max_len
    elif max_len is None:
        max_len = MAX_LEN
    model.eval()
    records = []
    for row in rows:
        record = {"item_id": row["item_id"], "status": "invalid_input",
                  "selected_candidate": None, "selected_index": None,
                  "candidate_scores": [], "error": None}
        records.append(record)
        try:
            context = mark_span(row["sentence"], explicit_span(row))
            candidates = candidates_for(row)
            if len(set(candidates)) != len(candidates):
                raise ValueError("Duplicate candidate entries are not supported")
        except (KeyError, ValueError) as error:
            record["error"] = str(error)
            continue
        try:
            encoded = encode_batch(tok, device, acr_open_id, acr_close_id,
                                   context, candidates, max_len=max_len)
        except ValueError as error:
            record.update(status="encoding_error", error=str(error))
            continue
        try:
            logits = model(**encoded, acr_open_id=acr_open_id, acr_close_id=acr_close_id)
        except RuntimeError as error:
            record.update(status="model_error", error=str(error))
            continue
        if tuple(logits.shape) != (len(candidates),) or not torch.isfinite(logits).all().item():
            record.update(status="invalid_scores", error="Expected one finite logit per candidate")
            continue
        selected = int(torch.argmax(logits).item())
        record.update(status="ok", selected_candidate=candidates[selected], selected_index=selected,
                      candidate_scores=[{"candidate": candidate, "score": float(score)}
                                        for candidate, score in zip(candidates, logits.cpu().tolist())])
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", required=True, help="Explicit input CSV with item IDs and raw spans")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--pooling", default=None, choices=["cls", "marker", "span_mean", "concat"])
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok, model, opened, closed = load_finetuned(args.checkpoint, pooling=args.pooling, device=device)
    print(json.dumps(evaluate(load_rows(args.items), tok, model, opened, closed, device),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
