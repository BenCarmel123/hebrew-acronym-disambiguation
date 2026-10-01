"""Invented tokenizer and trainable encoder; no research data or model weights."""

import re
from types import SimpleNamespace

import torch
from torch import nn

from hebrew_acronyms.models.common.pairs import ACR_CLOSE, ACR_OPEN


class TinyTokenizer:
    """Stable character IDs with two separately added target markers."""

    cls_token_id = 1
    sep_token_id = 2
    pad_token_id = 0

    def __init__(self):
        self.markers = {}

    def __len__(self):
        return 64 + len(self.markers)

    def get_vocab(self):
        return {**{f"char-{i}": i for i in range(64)}, **self.markers}

    def add_special_tokens(self, values):
        added = 0
        for token in values["additional_special_tokens"]:
            if token not in self.markers:
                self.markers[token] = len(self)
                added += 1
        return added

    def convert_tokens_to_ids(self, tokens):
        return [self.markers[token] for token in tokens]

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        parts = re.findall(r"\[/?ACR\]|\S", text)
        return [self.markers[part] if part in self.markers else 3 + ord(part) % 61
                for part in parts]


class TinyEncoder(nn.Module):
    """Embedding and contextual mean keep the fixture small and trainable."""

    def __init__(self, vocab_size=64, hidden_size=8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.projection = nn.Linear(hidden_size, hidden_size)

    def resize_token_embeddings(self, size):
        old = self.embeddings
        replacement = nn.Embedding(size, self.config.hidden_size)
        with torch.no_grad():
            replacement.weight[:len(old.weight)].copy_(old.weight)
        self.embeddings = replacement
        return replacement

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        values = self.embeddings(input_ids)
        mask = attention_mask.unsqueeze(-1)
        context = (values * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True).clamp(min=1)
        hidden = self.projection(values + context).tanh()
        return SimpleNamespace(last_hidden_state=hidden)


def tiny_base_model(model_id=None, revision=None):
    """Match the real base-loader signature without downloads or external files."""
    return TinyTokenizer(), TinyEncoder()


def marked_tokenizer():
    tokenizer = TinyTokenizer()
    tokenizer.add_special_tokens({"additional_special_tokens": [ACR_OPEN, ACR_CLOSE]})
    return tokenizer


# Explicit invented items for training-loop equivalence (not research examples).
TRAINING_ROWS = [
    dict(item_id="tiny-1", sentence='היום ב״ד בכיתה.', target_raw='ב״ד',
         span_start=5, span_end=8, candidates='בדיקת דוגמה|בניית דגם', gold_expansion='בדיקת דוגמה'),
    dict(item_id="tiny-2", sentence='כעת מ״ד מוכן.', target_raw='מ״ד',
         span_start=4, span_end=7, candidates='משחק דוגמה|מספר דגמים', gold_expansion='משחק דוגמה'),
]
from hebrew_acronyms.models.common.pairs import build_pairs
TRAINING_PAIRS = build_pairs(TRAINING_ROWS)
