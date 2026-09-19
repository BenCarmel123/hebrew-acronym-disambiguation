"""Check local imports, invented pairs, and an optional offline CPU forward pass."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
from pathlib import Path
import platform
import sys


def block_network(event: str, args: tuple) -> None:
    """Reject Python socket network operations during this check."""
    if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
        raise RuntimeError(f"Network access is disabled during this check: {event}")


def check_pairs() -> list[tuple[str, str, int]]:
    """Check two invented sentences, including Hebrew/ASCII quote equivalence."""
    from model.common.pairs import build_pairs

    rows = [
        {"sentence": 'היום נערך מפגש של ב״ד בכיתה.', "acronym": 'ב״ד',
         "candidates": 'בדיקת דוגמה | בניית דגם', "gold_expansion": 'בדיקת דוגמה'},
        {"sentence": 'מחר נציג ב"ד קטן.', "acronym": 'ב״ד',
         "candidates": 'בדיקת דוגמה | בניית דגם', "gold_expansion": 'בניית דגם'},
    ]
    expected = [
        ('היום נערך מפגש של [ACR]ב״ד[/ACR] בכיתה.', 'בדיקת דוגמה', 1),
        ('היום נערך מפגש של [ACR]ב״ד[/ACR] בכיתה.', 'בניית דגם', 0),
        ('מחר נציג [ACR]ב"ד[/ACR] קטן.', 'בדיקת דוגמה', 0),
        ('מחר נציג [ACR]ב"ד[/ACR] קטן.', 'בניית דגם', 1),
    ]
    pairs = build_pairs(rows)
    if pairs != expected:
        raise ValueError(f"Unexpected pairs or target marking: {pairs!r}")
    print("Pairs: PASS (2 invented rows, 2 candidates each; exact text and labels)")
    return pairs


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


def check_model(snapshot: Path, pairs: list[tuple[str, str, int]]) -> None:
    """Load through the existing base-model interface; run one short CPU batch."""
    import torch
    from model.dictabert.model import build_model

    print(f"Snapshot: {snapshot}")
    print(f"Snapshot directory / revision: {snapshot.name} (check only)")
    if not (snapshot / "config.json").is_file():
        raise FileNotFoundError(f"Missing config.json in {snapshot}")
    tokenizer, encoder = build_model(model_id=str(snapshot), revision=None)
    encoder.to("cpu").eval()
    # This uses the base tokenizer on short pairs, not the notebook's custom encoding.
    contexts, candidates, _ = zip(*pairs[:2])
    batch = tokenizer(list(contexts), list(candidates), padding=True,
                      truncation=False, return_tensors="pt")
    if batch["input_ids"].shape[1] > 64:
        raise ValueError("Invented input unexpectedly exceeds the 64-token smoke limit")
    torch.set_num_threads(2)
    with torch.no_grad():
        hidden = encoder(**batch).last_hidden_state
    expected_shape = (*batch["input_ids"].shape, encoder.config.hidden_size)
    if tuple(hidden.shape) != expected_shape or not torch.isfinite(hidden).all().item():
        raise ValueError(f"Invalid encoder output: shape={tuple(hidden.shape)}")
    if hidden.device.type != "cpu" or hidden.requires_grad:
        raise ValueError("Expected CPU output without gradients")
    print(f"Model: PASS (one CPU forward; shape={tuple(hidden.shape)}; all finite)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, help="Existing local DictaBERT snapshot directory")
    args = parser.parse_args()
    # Set before importing Hugging Face. Never fall back to downloading weights.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.dont_write_bytecode = True
    sys.addaudithook(block_network)
    print(f"Python: {platform.python_version()}")
    print(f"OS: {platform.platform()} ({platform.machine()})")
    print("Network: Hugging Face offline; Python socket network operations blocked")
    print("Installed packages: " + ", ".join(sorted(
        f"{dist.metadata['Name']}=={dist.version}" for dist in importlib.metadata.distributions()
    )))

    try:
        for name in ("requests", "torch", "transformers", "dotenv", "tqdm",
                     "model.common.pairs", "model.dictabert.model", "model.dictabertX.model"):
            importlib.import_module(name)
            print(f"Import: PASS ({name})")
    except Exception as exc:
        print(f"Imports: FAIL ({type(exc).__name__}: {exc})")
        print("Pairs and model: NOT RUN (required import failed)")
        return 1

    import torch
    print(f"Devices: CPU=available; CUDA={torch.cuda.is_available()}; "
          f"MPS={torch.backends.mps.is_available()}")
    print("External GPU environment: NOT VERIFIED")
    try:
        pairs = check_pairs()
    except Exception as exc:
        print(f"Pairs: FAIL ({type(exc).__name__}: {exc})")
        print("Model: NOT RUN (pair check failed)")
        return 1
    try:
        snapshot = find_snapshot(args.snapshot)
        if snapshot is None:
            print("Model: NOT RUN (no local DictaBERT snapshot; supply --snapshot)")
            return 2
        check_model(snapshot, pairs)
    except Exception as exc:
        print(f"Model: FAIL ({type(exc).__name__}: {exc})")
        return 1
    print("Environment check: PASS; no research accuracy or checkpoint compatibility tested")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
