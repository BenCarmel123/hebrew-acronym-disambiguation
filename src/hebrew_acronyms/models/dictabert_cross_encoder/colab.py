"""Explicit inference adapter for Ben's complete Colab state_dict checkpoints.

This is separate from the manifest-bound loader. It does not claim an original
manifest, training-input identity, seed, or original library/tokenizer identity.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
from types import SimpleNamespace
import re

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from hebrew_acronyms.models.common.pairs import ACR_OPEN, ACR_CLOSE
from hebrew_acronyms.models.dictabert_cross_encoder.model import (
    CrossEncoder, library_versions, tokenizer_identity,
)

COLAB_MODEL_ID = "dicta-il/dictabert"
COLAB_MAX_LEN = 256
COLAB_NOTEBOOK_REVISION = "09f815bb8c7658ef6dfa1c0efca19aa6a032bdf0"
COLAB_NOTEBOOK_SHA256 = "a57e668d20ba5c5cd8f6bcacd5b52fe8d67288cf1ea9a09cdb67e7108f0b010c"


def _readable_digest(path):
    """Reject dataless/partial reads even when metadata reports a positive size."""
    expected_bytes = path.stat().st_size
    digest = hashlib.sha256()
    actual_bytes = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            actual_bytes += len(chunk)
            digest.update(chunk)
    if not actual_bytes or actual_bytes != expected_bytes:
        raise ValueError(f"File not fully readable: {path}; read {actual_bytes}/{expected_bytes} bytes")
    return digest.hexdigest()


def load_colab_finetuned(checkpoint_path, *, snapshot_path, expected_sha256,
                         device="cpu"):
    """Load all Colab weights strictly, using an explicitly supplied local snapshot.

    The caller must supply the human-authorized checkpoint SHA-256. No download,
    random-head fallback, manifest inference, or training occurs. The meta-device
    construction avoids allocating a second full model before loading the fp16
    state; inference uses float32 as instructed by the Colab notebook.
    """
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("An explicit authorized checkpoint SHA-256 is required")
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    actual_hash = _readable_digest(checkpoint)
    if actual_hash != expected_sha256:
        raise ValueError("Colab checkpoint differs from the authorized SHA-256")
    snapshot = Path(snapshot_path).expanduser().resolve()
    required = ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt")
    hashes = {}
    for name in required:
        path = snapshot / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Nonempty local tokenizer/config file required: {path}")
        hashes[name] = _readable_digest(path)
    config = AutoConfig.from_pretrained(str(snapshot), local_files_only=True)
    expected_config = {"model_type": "bert", "hidden_size": 768,
                       "num_hidden_layers": 12, "num_attention_heads": 12,
                       "intermediate_size": 3072, "vocab_size": 128000,
                       "type_vocab_size": 2, "max_position_embeddings": 512}
    for key, value in expected_config.items():
        if getattr(config, key, None) != value:
            raise ValueError(f"Local configuration differs from DictaBERT: {key}")
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    added = tokenizer.add_special_tokens({"additional_special_tokens": [ACR_OPEN, ACR_CLOSE]})
    opened, closed = tokenizer.convert_tokens_to_ids([ACR_OPEN, ACR_CLOSE])
    if added != 2 or len(tokenizer) != 128002 or [opened, closed] != [128000, 128001]:
        raise ValueError("Local tokenizer does not reproduce the Colab marker additions")
    # Same AutoModel architecture and resized embeddings as the notebook, without
    # reading the unused pretrained base weights or allocating random parameters.
    config.vocab_size = len(tokenizer)
    with torch.device("meta"):
        encoder = AutoModel.from_config(config)
        model = CrossEncoder(encoder, hidden=config.hidden_size, pooling="cls", dropout=0.1)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict) or not state or any(not isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Colab artifact must be a nonempty tensor state_dict")
    if any(v.dtype != torch.float16 for v in state.values()):
        raise ValueError("Expected the notebook's all-fp16 saved state_dict")
    if any(not torch.isfinite(v).all().item() for v in state.values()):
        raise ValueError("Colab checkpoint contains nonfinite values")
    # strict=True rejects missing encoder/head keys, extra keys, and wrong shapes.
    loaded = model.load_state_dict(state, strict=True, assign=True)
    tensor_count = len(state)
    del state
    # BERT has deterministic nonpersistent position/type buffers that are not
    # checkpoint keys; recreate only those after meta construction.
    embeddings = model.encoder.embeddings
    embeddings.position_ids = torch.arange(config.max_position_embeddings).expand((1, -1))
    embeddings.token_type_ids = torch.zeros((1, config.max_position_embeddings), dtype=torch.long)
    model.float().to(device).eval()
    if any(parameter.is_meta for parameter in model.parameters()):
        raise ValueError("Full checkpoint load left unmaterialized parameters")
    model.training_config = SimpleNamespace(max_len=COLAB_MAX_LEN)
    model.colab_reconstruction = {
        "checkpoint_format": "colab_state_dict",
        "weights_sha256": actual_hash,
        "original_manifest": None,
        "reference_notebook": {"path": "notebooks/train_dictabert_colab.ipynb",
                               "revision": COLAB_NOTEBOOK_REVISION,
                               "sha256": COLAB_NOTEBOOK_SHA256},
        "architecture": {"model_id": COLAB_MODEL_ID, "pooling": "cls", "dropout": 0.1,
                         "max_len": COLAB_MAX_LEN, "head": "Linear(768, 1)",
                         "pair_layout": "[CLS] marked_context [SEP] candidate [SEP]",
                         "context_truncation": "outside-in, retaining target and candidate",
                         "config": expected_config},
        "tokenizer": tokenizer_identity(tokenizer),
        "local_snapshot": str(snapshot),
        "local_snapshot_revision": snapshot.name,
        "tokenizer_config_files_sha256": hashes,
        "libraries": library_versions(),
        "strict_load": {"missing_keys": list(loaded.missing_keys),
                        "unexpected_keys": list(loaded.unexpected_keys),
                        "tensor_count": tensor_count, "saved_dtype": "float16",
                        "inference_dtype": "float32", "all_finite": True,
                        "parameter_count": sum(p.numel() for p in model.parameters())},
        "unknowns": ["original training library versions",
                     "original resolved base-model/tokenizer revision and tokenizer hash",
                     "exact training rows identity and selected epoch/loss",
                     "actual training seed (filename seed43; reference notebook seed42)"],
    }
    return tokenizer, model, opened, closed


def predict_colab_study(artifact, *, checkpoint, snapshot_path, expected_sha256,
                         device="cpu"):
    """Validate three items, predict the identified cohort, and save the study.

    Artifact settings select ``checkpoint_format='colab_state_dict'`` explicitly.
    Training provenance remains an attestation/unknown, never a forged manifest.
    """
    import gc
    from copy import deepcopy
    from hebrew_acronyms.experimental_study import (
        attach_encoder, save_study, validate_artifact,
    )
    from hebrew_acronyms.models.dictabert_cross_encoder.eval import evaluate
    from hebrew_acronyms.models.dictabert_cross_encoder.model import json_digest

    validate_artifact(artifact, expected_run_id=artifact["run_id"])
    settings = artifact["settings"]
    if settings.get("saved_encoder") is not None:
        raise ValueError("Colab saved predictions must use reload or saved-run comparison, not fresh inference")
    if settings.get("checkpoint_format") != "colab_state_dict":
        raise ValueError("Explicit checkpoint_format='colab_state_dict' is required")
    for key, value in {"checkpoint": checkpoint, "snapshot_path": snapshot_path,
                       "device": device}.items():
        value = str(value) if isinstance(value, Path) else value
        if key in settings and settings[key] != value:
            raise ValueError(f"Colab setting {key} differs from the recorded study")
    if any(r["status"] != "not_run" for r in artifact["records"]
           if r["condition"] == "dictabert"):
        raise ValueError("Encoder predictions already exist; start a new run")
    rows = artifact["items"]
    tokenizer, model, opened, closed = load_colab_finetuned(
        checkpoint, snapshot_path=snapshot_path, expected_sha256=expected_sha256,
        device=device)
    try:
        validation_count = min(3, len(rows))
        predictions = evaluate(rows[:validation_count], tokenizer, model, opened, closed, device)
        if any(p["status"] != "ok" for p in predictions):
            raise ValueError(f"Colab validation failed: {predictions}")
        predictions += evaluate(rows[validation_count:], tokenizer, model, opened, closed, device)
        evidence = deepcopy(model.colab_reconstruction)
        evidence["input_identity"] = deepcopy(artifact["input_identity"])
        evidence["validation"] = {"items": validation_count, "status": "passed"}
        evidence["user_attestation"] = deepcopy(settings.get("checkpoint_attestation"))
        evidence["provenance_basis"] = (
            "Explicitly selected checkpoint SHA-256 and inspected reference notebook; "
            "exact historical train rows/library/tokenizer identities are not embedded in weights.")
        origin = {"checkpoint_format": "colab_state_dict", "checkpoint": str(Path(checkpoint).resolve()),
                  "weights_sha256": evidence["weights_sha256"], "reconstruction": evidence,
                  "reconstruction_sha256": json_digest(evidence), "device": device,
                  "input_match": "prediction cohort identified; original training inputs unverified"}
    finally:
        del model, tokenizer
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()
    attach_encoder(artifact, predictions, origin=origin)
    save_study(artifact)
    return predictions
