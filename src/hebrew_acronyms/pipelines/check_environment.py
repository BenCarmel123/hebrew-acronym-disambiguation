"""Check local imports, invented pairs, and an optional offline CPU forward pass."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
from pathlib import Path
import platform

from hebrew_acronyms.models.dictabert_cross_encoder.workflow import enable_offline, find_snapshot


def check_pairs() -> list[tuple[str, str, int]]:
    """Check two invented sentences, including Hebrew/ASCII quote equivalence."""
    from hebrew_acronyms.models.common.pairs import build_pairs

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


def check_model(snapshot: Path, pairs: list[tuple[str, str, int]]) -> None:
    """Load through the existing base-model interface; run one short CPU batch."""
    import torch
    from hebrew_acronyms.models.dictabert_similarity.model import build_model

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
    enable_offline()
    print(f"Python: {platform.python_version()}")
    print(f"OS: {platform.platform()} ({platform.machine()})")
    print("Network: Hugging Face offline; Python socket network operations blocked")
    print("Installed packages: " + ", ".join(sorted(
        f"{dist.metadata['Name']}=={dist.version}" for dist in importlib.metadata.distributions()
    )))

    try:
        for name in ("requests", "torch", "transformers", "dotenv", "tqdm",
                     "hebrew_acronyms.models.common.pairs", "hebrew_acronyms.models.dictabert_similarity.model", "hebrew_acronyms.models.dictabert_cross_encoder.model"):
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
