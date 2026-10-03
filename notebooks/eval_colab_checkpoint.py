"""Local item-level accuracy eval for a checkpoint trained by train_dictabert_colab.ipynb.

That notebook saves a PLAIN torch.save(state_dict) with no metadata sidecar (same
format the old model/dictabertX/eval.py used) — the new, stricter
src/hebrew_acronyms/models/dictabert_cross_encoder eval.py refuses files like that
(it requires a .json sidecar with hashes/provenance). This script reconstructs the
same small CrossEncoder architecture directly and loads the plain state_dict, so you
don't need a sidecar to get a number out.

Usage (from the repo root, with the project installed — see README's "Install and
check" — or just plain `python3 -m pip install torch transformers` in any venv):

    python3 notebooks/eval_colab_checkpoint.py \
        --checkpoint ~/Downloads/best.pt \
        --items data/study_v1/encoder_inputs/dev.csv \
        --pooling cls

Prints item-level accuracy: for each item, scores every candidate and takes the
argmax; correct if it matches gold_expansion. This is NOT the same metric as the
training loop's printed dev_pair_acc (that's pairwise, not per-item) — see the old
weights/README.md's explanation of why those two numbers disagree.
"""
from __future__ import annotations

import argparse
import csv

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "dicta-il/dictabert"
MAX_LEN = 256
ACR_OPEN, ACR_CLOSE = "[ACR]", "[/ACR]"

_EQUIV = {"״": '"', "“": '"', "”": '"',
          "׳": "'", "‘": "'", "’": "'"}


def _normalise(s: str) -> str:
    return "".join(_EQUIV.get(ch, ch) for ch in s)


class CrossEncoder(nn.Module):
    """Mirrors train_dictabert_colab.ipynb's model exactly, so a saved state_dict
    loads back with matching parameter names and shapes."""

    def __init__(self, encoder, hidden, pooling="cls", dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.score = nn.Linear(hidden * 2 if pooling == "concat" else hidden, 1)

    def _span_mean(self, hidden_states, input_ids, acr_open_id, acr_close_id):
        opened = (input_ids == acr_open_id).cumsum(dim=1)
        closed = (input_ids == acr_close_id).cumsum(dim=1)
        inside = ((opened == 1) & (closed == 0)
                  & (input_ids != acr_open_id)).float().unsqueeze(-1)
        counts = inside.sum(dim=1)
        pooled = (hidden_states * inside).sum(dim=1) / counts.clamp(min=1)
        empty = (counts.squeeze(-1) == 0)
        if empty.any():
            pooled[empty] = hidden_states[empty, 0]
        return pooled

    def _marker(self, hidden_states, input_ids, acr_open_id):
        is_open = (input_ids == acr_open_id)
        idx = is_open.float().argmax(dim=1)
        pooled = hidden_states[torch.arange(hidden_states.size(0)), idx]
        missing = ~is_open.any(dim=1)
        if missing.any():
            pooled[missing] = hidden_states[missing, 0]
        return pooled

    def forward(self, input_ids, attention_mask, token_type_ids, acr_open_id, acr_close_id):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                         token_type_ids=token_type_ids).last_hidden_state
        cls = h[:, 0]
        if self.pooling == "cls":
            pooled = cls
        elif self.pooling == "marker":
            pooled = self._marker(h, input_ids, acr_open_id)
        elif self.pooling == "span_mean":
            pooled = self._span_mean(h, input_ids, acr_open_id, acr_close_id)
        elif self.pooling == "concat":
            pooled = torch.cat([cls, self._span_mean(h, input_ids, acr_open_id, acr_close_id)], dim=-1)
        else:
            raise ValueError(f"unknown pooling {self.pooling!r}")
        return self.score(self.dropout(pooled)).squeeze(-1)


def load_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def explicit_span(row: dict) -> tuple[int, int] | None:
    """Prefer the row's own span_start/span_end/target_raw (study_v1 schema) when
    present; fall back to a plain first-occurrence search (older data/splits schema)."""
    sentence = row.get("sentence", "")
    if row.get("span_start") not in (None, "") and row.get("span_end") not in (None, ""):
        start, end = int(row["span_start"]), int(row["span_end"])
        if 0 <= start < end <= len(sentence):
            return start, end
    acronym = row.get("acronym") or row.get("target_raw")
    if not acronym:
        return None
    hay, needle = _normalise(sentence), _normalise(acronym)
    i = hay.find(needle)
    return None if i < 0 else (i, i + len(needle))


def mark_span(sentence: str, span: tuple[int, int]) -> str:
    s, e = span
    return f"{sentence[:s]}{ACR_OPEN}{sentence[s:e]}{ACR_CLOSE}{sentence[e:]}"


def encode_batch(tok, device, acr_open_id, acr_close_id, context: str,
                 candidates: list[str], max_len=MAX_LEN) -> dict:
    cls_id, sep_id = tok.cls_token_id, tok.sep_token_id
    rows = []
    ctx_ids = tok.encode(context, add_special_tokens=False)
    o, c = ctx_ids.index(acr_open_id), ctx_ids.index(acr_close_id)
    for candidate in candidates:
        cand_ids = tok.encode(candidate, add_special_tokens=False)
        budget = max_len - 3 - len(cand_ids)
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
    for row in rows:
        candidates_raw = row.get("candidates", "")
        candidates = [c.strip() for c in candidates_raw.split("|") if c.strip()]
        gold = (row.get("gold_expansion") or "").strip()
        span = explicit_span(row)
        if len(candidates) < 2 or not gold or span is None:
            continue
        marked = mark_span(row["sentence"], span)
        enc = encode_batch(tok, device, acr_open_id, acr_close_id, marked, candidates)
        logits = model(**enc, acr_open_id=acr_open_id, acr_close_id=acr_close_id)
        pred = candidates[int(torch.argmax(logits).item())]
        correct += int(pred == gold)
        total += 1
    return {"n_items": total, "accuracy": round(correct / total, 4) if total else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="plain state_dict .pt from train_dictabert_colab.ipynb")
    ap.add_argument("--items", default="data/study_v1/encoder_inputs/dev.csv")
    ap.add_argument("--pooling", default="cls", choices=["cls", "marker", "span_mean", "concat"])
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    encoder = AutoModel.from_pretrained(MODEL_ID)
    n_added = tok.add_special_tokens({"additional_special_tokens": [ACR_OPEN, ACR_CLOSE]})
    if n_added:
        encoder.resize_token_embeddings(len(tok))
    acr_open_id, acr_close_id = tok.convert_tokens_to_ids([ACR_OPEN, ACR_CLOSE])

    model = CrossEncoder(encoder, hidden=encoder.config.hidden_size, pooling=a.pooling).to(device)
    state = torch.load(a.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    rows = load_rows(a.items)
    res = evaluate(rows, tok, model, acr_open_id, acr_close_id, device)
    print(f"{a.items}  (checkpoint: {a.checkpoint}, pooling: {a.pooling})\n")
    print(f"  items scored      {res['n_items']}")
    print(f"  accuracy          {res['accuracy']:.3f}")


if __name__ == "__main__":
    main()
