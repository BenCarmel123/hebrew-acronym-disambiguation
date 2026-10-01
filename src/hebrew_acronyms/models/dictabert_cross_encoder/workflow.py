"""Small local setup helpers; model operations are called explicitly by the notebook."""
from __future__ import annotations

import os
from pathlib import Path


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
    """Request local Hugging Face files; no socket hooks or process-wide I/O guards.

    Call before importing Transformers. Notebook model inputs must additionally be
    existing local snapshot directories, including when libraries were imported earlier.
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def validate_training_paths(train_path, dev_path, checkpoint_path) -> None:
    """Require explicit input files and a fresh output before reading or training."""
    if train_path is None or dev_path is None or checkpoint_path is None:
        raise ValueError("Training requires explicit train, dev and checkpoint paths")
    for path in (train_path, dev_path):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    output = Path(checkpoint_path)
    if output.exists() or Path(str(output) + ".json").exists():
        raise FileExistsError("Choose a new checkpoint path; weights or metadata already exist")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"Create the output directory first: {output.parent}")
