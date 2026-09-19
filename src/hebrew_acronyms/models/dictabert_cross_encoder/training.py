"""Pair-level training extracted from the accepted training notebook.

The seed is deliberately applied after model construction. Checkpoints contain only
state_dict and are selected by strict improvement in development pair loss.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import time

import torch
from torch import nn

from hebrew_acronyms.models.common.pairs import ACR_CLOSE, ACR_OPEN
from hebrew_acronyms.models.dictabert_cross_encoder.encoding import encode_pairs


@dataclass(frozen=True)
class TrainingConfig:
    """Existing notebook settings; no new experimental defaults."""

    max_len: int = 256
    batch_size: int = 16
    eval_batch_size: int = 32
    epochs: int = 1
    lr: float = 2e-5
    seed: int = 42
    pooling: str = "cls"


def batches(pairs, batch_size: int, shuffle: bool):
    """Yield context/candidate pairs and labels in the original batching order."""
    idx = list(range(len(pairs)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        chunk = [pairs[j] for j in idx[i:i + batch_size]]
        yield [(c, cand) for c, cand, _ in chunk], [label for _, _, label in chunk]


@torch.no_grad()
def evaluate_pairs(model, tok, pairs, device: str,
                   config: TrainingConfig = TrainingConfig()) -> tuple[float, float]:
    """Mean BCE and pair accuracy; this is not item-level candidate accuracy."""
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    open_id, close_id = tok.convert_tokens_to_ids([ACR_OPEN, ACR_CLOSE])
    total_loss, total_correct, n = 0.0, 0, 0
    for ctx_batch, labels in batches(pairs, config.eval_batch_size, shuffle=False):
        enc = encode_pairs(tok, ctx_batch, device, max_len=config.max_len)
        y = torch.tensor(labels, dtype=torch.float, device=device)
        logits = model(**enc, acr_open_id=open_id, acr_close_id=close_id)
        loss = loss_fn(logits, y)
        total_loss += loss.item() * len(labels)
        total_correct += ((torch.sigmoid(logits) > 0.5).float() == y).sum().item()
        n += len(labels)
    return total_loss / n, total_correct / n


def train(model, tok, train_pairs, dev_pairs, checkpoint_path: str | Path,
          device: str = "cpu", config: TrainingConfig = TrainingConfig()) -> list[dict]:
    """Train an already initialized model using the original notebook operation order.

    The caller supplies inputs and an output path. This function does not fetch data,
    construct a model, choose a device or change any research split.
    """
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    open_id, close_id = tok.convert_tokens_to_ids([ACR_OPEN, ACR_CLOSE])
    history = []
    best_dev_loss = float("inf")
    for epoch in range(1, config.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss, n_seen = 0.0, 0
        for ctx_batch, labels in batches(train_pairs, config.batch_size, shuffle=True):
            enc = encode_pairs(tok, ctx_batch, device, max_len=config.max_len)
            y = torch.tensor(labels, dtype=torch.float, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(**enc, acr_open_id=open_id, acr_close_id=close_id)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(labels)
            n_seen += len(labels)

        train_loss = running_loss / n_seen
        dev_loss, dev_pair_acc = evaluate_pairs(model, tok, dev_pairs, device, config)
        print(f"epoch {epoch}/{config.epochs}  train_loss={train_loss:.4f}  "
              f"dev_loss={dev_loss:.4f}  dev_pair_acc={dev_pair_acc:.3f}  "
              f"({time.time()-t0:.0f}s)")
        saved = dev_loss < best_dev_loss
        if saved:
            best_dev_loss = dev_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> saved {checkpoint_path}")
        history.append({"epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss,
                        "dev_pair_accuracy": dev_pair_acc, "checkpoint_saved": saved})
    return history
