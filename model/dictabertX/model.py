"""The fine-tuned cross-encoder architecture: DictaBERT + dropout + one linear head.

Mirrors notebooks/train_dictabert.ipynb's CrossEncoder exactly, so a checkpoint trained
there loads here without a shape mismatch. If the notebook's architecture changes, this
must change with it — there is no single shared source today because the notebook is
self-contained (it fetches model/common/pairs.py by URL but defines the model itself).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from model.common.pairs import ACR_CLOSE, ACR_OPEN
from model.dictabert.model import build_model as build_base_model


class CrossEncoder(nn.Module):
    """Emits ONE raw logit per (context, candidate) pair — not a probability."""

    def __init__(self, encoder, hidden: int, pooling: str = "cls", dropout: float = 0.1):
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

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                acr_open_id=None, acr_close_id=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        h = self.encoder(**kwargs).last_hidden_state
        cls = h[:, 0]

        if self.pooling == "cls":
            pooled = cls
        elif self.pooling == "marker":
            pooled = self._marker(h, input_ids, acr_open_id)
        elif self.pooling == "span_mean":
            pooled = self._span_mean(h, input_ids, acr_open_id, acr_close_id)
        elif self.pooling == "concat":
            pooled = torch.cat([cls, self._span_mean(h, input_ids, acr_open_id, acr_close_id)],
                                dim=-1)
        else:
            raise ValueError(f"unknown pooling {self.pooling!r}")

        return self.score(self.dropout(pooled)).squeeze(-1)


def load_finetuned(checkpoint_path: str, pooling: str = "cls", device: str = "cpu"):
    """Rebuild the tokenizer+encoder exactly as training did, then load trained weights.

    Order matters: the special tokens must be added and the embedding matrix resized
    BEFORE `load_state_dict`, or the saved embedding tensor's shape won't match.
    """
    tok, encoder = build_base_model()
    n_added = tok.add_special_tokens({"additional_special_tokens": [ACR_OPEN, ACR_CLOSE]})
    if n_added:
        encoder.resize_token_embeddings(len(tok))

    model = CrossEncoder(encoder, hidden=encoder.config.hidden_size, pooling=pooling)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()

    acr_open_id, acr_close_id = tok.convert_tokens_to_ids([ACR_OPEN, ACR_CLOSE])
    return tok, model, acr_open_id, acr_close_id
