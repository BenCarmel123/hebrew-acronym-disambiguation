"""Pair BCE training; strict development-loss checkpoint selection is unchanged."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import time

import torch
from torch import nn

from hebrew_acronyms.models.common.pairs import ACR_CLOSE, ACR_OPEN, PreparedPairs
from hebrew_acronyms.models.dictabert_cross_encoder.model import save_checkpoint, set_seed
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

    def __post_init__(self):
        for name in ("max_len", "batch_size", "eval_batch_size", "epochs"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 < self.lr < float("inf"):
            raise ValueError("lr must be positive and finite")
        if type(self.seed) is not int or self.pooling not in {"cls", "marker", "span_mean", "concat"}:
            raise ValueError("Invalid seed or pooling")


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
    if not pairs:
        raise ValueError("Evaluation pairs must be nonempty")
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
    """Train validated item pairs without changing loss, budgets or selection rule."""
    for name, values in (("train", train_pairs), ("dev", dev_pairs)):
        if not isinstance(values, PreparedPairs) or not values:
            raise ValueError(f"{name} requires nonempty build_pairs output with explicit item identities")
        values.verify()
    if config.seed != model.initialization["seed"] or config.pooling != model.pooling:
        raise ValueError("Training config disagrees with model initialization")
    # Fail impossible budgets before any optimizer update, including development.
    for values in (train_pairs, dev_pairs):
        for contexts, _ in batches(values, config.eval_batch_size, shuffle=False):
            encode_pairs(tok, contexts, device, max_len=config.max_len)
    model.training_config = config
    set_seed(config.seed)
    inputs = {"train": train_pairs.identity, "dev": dev_pairs.identity}

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
            save_checkpoint(model, tok, checkpoint_path, config, inputs, epoch, dev_loss)
            print(f"  -> saved {checkpoint_path}")
        history.append({"epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss,
                        "dev_pair_accuracy": dev_pair_acc, "checkpoint_saved": saved})
    return history


def sanity_overfit(model, tok, rows, checkpoint_path, *, config, device="cpu") -> dict:
    """Explicit engineering-only repetition of a tiny invented set, also used as dev.

    Caller chooses a separate sanity config. No research defaults are changed and
    using the same rows twice is deliberate: this tests memorization, not generalization.
    """
    from hebrew_acronyms.models.common.pairs import build_pairs
    from hebrew_acronyms.models.dictabert_cross_encoder.eval import evaluate
    if not rows or len(rows) > 8:
        raise ValueError("Sanity requires between one and eight explicitly supplied invented items")
    pairs = build_pairs(rows)
    initial_loss, _ = evaluate_pairs(model, tok, pairs, device, config)
    history = train(model, tok, pairs, pairs, checkpoint_path, device, config)
    final_loss, _ = evaluate_pairs(model, tok, pairs, device, config)
    opened, closed = tok.convert_tokens_to_ids([ACR_OPEN, ACR_CLOSE])
    predictions = evaluate(rows, tok, model, opened, closed, device)
    learned = all(p["status"] == "ok" and p["selected_candidate"] == r["gold_expansion"].strip()
                  for p, r in zip(predictions, rows))
    if not final_loss < initial_loss or not learned:
        raise AssertionError("Engineering overfit failed: loss must decrease and every choice must be learned")
    return {"mode": "sanity", "status": "PASS", "initial_loss": initial_loss,
            "final_loss": final_loss, "history": history, "predictions": predictions,
            "engineering_only": True}
