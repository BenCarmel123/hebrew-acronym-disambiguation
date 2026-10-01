"""Local input and execution steps used by the training notebook.

Imports are inert. Smoke uses invented fixtures and a cached real model; the explicit
training mode uses explicit item inputs. Sanity is a tiny-model engineering check.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys


def block_network(event: str, args: tuple) -> None:
    """Reject Python socket network operations during this check."""
    if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
        raise RuntimeError(f"Network access is disabled during this check: {event}")


def find_snapshot(explicit: Path | None) -> Path | None:
    """Use an explicit directory or the locally cached DictaBERT main reference."""
    if explicit is not None:
        snapshot = explicit.expanduser().resolve()
        if not snapshot.is_dir():
            raise FileNotFoundError(f"Snapshot directory does not exist: {snapshot}")
        return snapshot

    from huggingface_hub.constants import HF_HUB_CACHE

    model_cache = Path(HF_HUB_CACHE) / "models--dicta-il--dictabert"
    reference = model_cache / "refs" / "main"
    if reference.is_file():
        revision = reference.read_text().strip()
        snapshot = model_cache / "snapshots" / revision
        if len(revision) == 40 and all(c in "0123456789abcdef" for c in revision):
            if snapshot.is_dir():
                return snapshot.resolve()
    snapshots = sorted(p for p in (model_cache / "snapshots").glob("*") if p.is_dir())
    if len(snapshots) == 1:
        return snapshots[0].resolve()
    if len(snapshots) > 1:
        raise ValueError("Multiple cached snapshots without a usable main ref; pass --snapshot.")
    return None


def enable_offline() -> None:
    """Set offline mode before loading Hugging Face and block Python socket traffic."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.dont_write_bytecode = True
    sys.addaudithook(block_network)


def configure_run(root: Path, mode: str = "smoke", snapshot: Path | None = None,
                  train_path: Path | None = None, dev_path: Path | None = None,
                  checkpoint_path: Path | None = None) -> dict:
    """Resolve explicit local inputs; no files are read from research data in smoke."""
    if mode not in {"smoke", "train", "sanity"}:
        raise ValueError("mode must be smoke, train or sanity")
    enable_offline()
    root = Path(root).resolve()
    if not (root / "src" / "hebrew_acronyms" / "models" / "dictabert_cross_encoder").is_dir():
        raise ValueError(f"Not a repository root: {root}")
    local = None if mode == "sanity" else find_snapshot(Path(snapshot) if snapshot is not None else None)
    if local is None and mode != "sanity":
        raise FileNotFoundError(
            "Model NOT RUN: no local DictaBERT snapshot; set DICTABERT_SNAPSHOT "
            "for the notebook or pass snapshot= to configure_run")
    if mode == "train":
        if train_path is None or dev_path is None or checkpoint_path is None:
            raise ValueError("Training requires explicit train_path, dev_path and checkpoint_path")
        for path in (train_path, dev_path):
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        if Path(checkpoint_path).exists():
            raise FileExistsError(f"Choose a new checkpoint path: {checkpoint_path}")
        if not Path(checkpoint_path).parent.is_dir():
            raise FileNotFoundError(f"Create the output directory first: {Path(checkpoint_path).parent}")
    if mode == "sanity":
        if train_path is not None or dev_path is not None or snapshot is not None:
            raise ValueError("Sanity uses invented fixtures and a tiny local model only")
        if checkpoint_path is None or Path(checkpoint_path).exists() or Path(str(checkpoint_path) + ".json").exists():
            raise ValueError("Sanity requires a new checkpoint path")
    if checkpoint_path is not None and Path(str(checkpoint_path) + ".json").exists():
        raise FileExistsError("Checkpoint metadata already exists; choose a new path")
    import torch
    device = "cuda" if mode == "train" and torch.cuda.is_available() else "cpu"
    return {"root": root, "mode": mode, "model_id": "tiny-fixture" if mode == "sanity" else str(local), "device": device,
            "train_path": train_path, "dev_path": dev_path, "checkpoint_path": checkpoint_path}


def load_inputs(run: dict) -> tuple[list[dict], list[dict]]:
    """Read invented smoke rows or the explicitly selected training/development CSVs."""
    from hebrew_acronyms.models.common.pairs import load_rows
    if run["mode"] in {"smoke", "sanity"}:
        path = run["root"] / "tests" / "fixtures" / "training_rows.json"
        return json.loads(path.read_text(encoding="utf-8")), []
    return load_rows(run["train_path"]), load_rows(run["dev_path"])


def prepare_pairs(train_rows: list[dict], dev_rows: list[dict]) -> tuple[list, list]:
    """Validate explicit item inputs and report engineering pair counts."""
    from hebrew_acronyms.models.common.pairs import build_pairs, describe_skips
    print("Input rows:", describe_skips(train_rows))
    print("Development rows:", describe_skips(dev_rows))
    train_pairs, dev_pairs = build_pairs(train_rows), build_pairs(dev_rows)
    print(f"Pairs: {len(train_pairs)} input, {len(dev_pairs)} development")
    if not train_pairs:
        raise ValueError("No usable input pairs")
    print("Example pair:", train_pairs[0])
    return train_pairs, dev_pairs


def load_model(run: dict, config):
    """Seed before encoder, marker and head initialization."""
    from hebrew_acronyms.models.dictabert_cross_encoder.model import build_cross_encoder
    if run["mode"] == "sanity":
        # This explicit test mode depends on source-checkout fixtures, never research inputs.
        from unittest.mock import patch
        from tests.fixtures.tiny import tiny_base_model
        with patch("hebrew_acronyms.models.dictabert_cross_encoder.model.build_base_model", tiny_base_model):
            return build_cross_encoder(model_id="tiny-fixture", revision="fixture-v1",
                                       pooling=config.pooling, device="cpu", seed=config.seed)
    return build_cross_encoder(model_id=run["model_id"], pooling=config.pooling,
                               device=run["device"], seed=config.seed)


def run_action(run: dict, model, tok, train_pairs, dev_pairs, config) -> dict:
    """Run the selected explicit smoke, tiny sanity or training action."""
    import torch
    from hebrew_acronyms.models.common.pairs import ACR_CLOSE, ACR_OPEN
    from hebrew_acronyms.models.dictabert_cross_encoder.encoding import encode_pairs
    from hebrew_acronyms.models.dictabert_cross_encoder.training import train, sanity_overfit
    from hebrew_acronyms.models.dictabert_cross_encoder.eval import evaluate
    if run["mode"] == "sanity":
        return sanity_overfit(model, tok, train_pairs.rows, run["checkpoint_path"],
                              config=config, device="cpu")
    if run["mode"] == "train":
        if not dev_pairs:
            raise ValueError("Training requires usable development pairs")
        history = train(model, tok, train_pairs, dev_pairs, run["checkpoint_path"],
                        device=run["device"], config=config)
        return {"mode": "train", "history": history,
                "checkpoint": str(run["checkpoint_path"])}

    torch.set_num_threads(2)
    model.to("cpu").eval()
    contexts = [(context, candidate) for context, candidate, _ in train_pairs]
    batch = encode_pairs(tok, contexts, "cpu", max_len=config.max_len)
    if batch["input_ids"].shape[1] > 64 or len(contexts) > 4:
        raise ValueError("Smoke is limited to four invented pairs of at most 64 tokens")
    open_id, close_id = tok.convert_tokens_to_ids([ACR_OPEN, ACR_CLOSE])
    for marker_id in (open_id, close_id):
        if not ((batch["input_ids"] == marker_id).sum(dim=1) == 1).all().item():
            raise ValueError("Smoke marker encoding was not preserved")
    with torch.no_grad():
        logits = model(**batch, acr_open_id=open_id, acr_close_id=close_id)
    if tuple(logits.shape) != (len(contexts),) or not torch.isfinite(logits).all().item():
        raise ValueError("Invalid smoke logits shape or nonfinite values")
    if logits.requires_grad or logits.device.type != "cpu":
        raise ValueError("Expected CPU logits without gradients")
    return {"mode": "smoke", "status": "PASS", "pairs": len(contexts),
            "input_shape": tuple(batch["input_ids"].shape), "logits_shape": tuple(logits.shape),
            "snapshot": Path(run["model_id"]).name, "device": "cpu", "all_finite": True,
            "training_performed": False, "checkpoint_written": False,
            "predictions": evaluate(train_pairs.rows, tok, model, open_id, close_id, "cpu",
                                    max_len=config.max_len)}


def summarize_run(result: dict) -> None:
    """Display a small engineering summary, without interpreting logits as accuracy."""
    for key, value in result.items():
        print(f"{key}: {value}")
    if result["mode"] == "smoke":
        print("Engineering smoke only; no training, benchmark score or historical checkpoint validation.")
