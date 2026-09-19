"""Shared cross-encoder architecture, initialization and checkpoint loading."""
from __future__ import annotations

import torch
import torch.nn as nn

from hebrew_acronyms.models.common.pairs import ACR_CLOSE, ACR_OPEN
from hebrew_acronyms.models.dictabert_similarity.model import MODEL_ID, REVISION, build_model as build_base_model


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


def build_cross_encoder(model_id=MODEL_ID, revision=REVISION, pooling="cls", device="cpu"):
    """Add target markers, resize embeddings, then initialize the scoring head.

    No seed is set here: the original training path seeds after model initialization.
    Pass a local snapshot directory as model_id for an offline check.
    device=None leaves construction on CPU for the original checkpoint-loading order.
    """
    tok, encoder = build_base_model(model_id=model_id, revision=revision)
    n_added = tok.add_special_tokens({"additional_special_tokens": [ACR_OPEN, ACR_CLOSE]})
    if n_added:
        encoder.resize_token_embeddings(len(tok))
    acr_open_id, acr_close_id = tok.convert_tokens_to_ids([ACR_OPEN, ACR_CLOSE])
    model = CrossEncoder(encoder, hidden=encoder.config.hidden_size, pooling=pooling)
    if device is not None:
        model.to(device)
    return tok, model, acr_open_id, acr_close_id


def load_finetuned(checkpoint_path: str, pooling: str = "cls", device: str = "cpu", *,
                   model_id=MODEL_ID, revision=REVISION):
    """Rebuild the training architecture and strictly load its saved state_dict.

    Target tokens and resized embeddings precede weight loading. Defaults retain the
    existing base-model selection; a local snapshot can be supplied explicitly.
    """
    tok, model, acr_open_id, acr_close_id = build_cross_encoder(
        model_id=model_id, revision=revision, pooling=pooling, device=None)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=True)
    model.to(device).eval()
    return tok, model, acr_open_id, acr_close_id
