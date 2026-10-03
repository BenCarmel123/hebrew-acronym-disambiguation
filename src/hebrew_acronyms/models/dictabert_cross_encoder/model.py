"""Shared cross-encoder architecture, initialization and checkpoint loading."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random

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


def set_seed(seed: int) -> None:
    """Seed before any encoder, marker embeddings or scoring head are created."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_digest(path) -> str:
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def library_versions() -> dict:
    return {name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "tokenizers")}


def tokenizer_identity(tok) -> dict:
    backend = getattr(tok, "backend_tokenizer", None)
    return {"class": type(tok).__name__, "size": len(tok),
            "vocab_sha256": json_digest(tok.get_vocab()),
            "backend_sha256": json_digest(backend.to_str()) if backend is not None else None,
            "special_tokens": getattr(tok, "special_tokens_map", {}),
            "marker_ids": tok.convert_tokens_to_ids([ACR_OPEN, ACR_CLOSE]),
            "cls_id": tok.cls_token_id, "sep_id": tok.sep_token_id, "pad_id": tok.pad_token_id}


def _local_source_files(model_id) -> dict:
    path = Path(model_id)
    files = {}
    if path.is_dir():
        # Bind actual local model/tokenizer contents, including mutable directories.
        for file in sorted(path.iterdir()):
            if file.is_file() and file.suffix in {".json", ".txt", ".safetensors", ".bin"}:
                files[file.name] = file_digest(file)
    return files


def source_identity(model_id, revision, encoder) -> dict:
    return {"model_id": str(model_id),
            "revision": getattr(encoder.config, "_commit_hash", None) or revision,
            "local_files_sha256": _local_source_files(model_id)}


def build_cross_encoder(model_id=MODEL_ID, revision=REVISION, pooling="cls", device="cpu", seed=42):
    """Seed, load the encoder, add markers, then initialize the scoring head."""
    set_seed(seed)
    tok, encoder = build_base_model(model_id=model_id, revision=revision)
    n_added = tok.add_special_tokens({"additional_special_tokens": [ACR_OPEN, ACR_CLOSE]})
    if n_added:
        encoder.resize_token_embeddings(len(tok))
    acr_open_id, acr_close_id = tok.convert_tokens_to_ids([ACR_OPEN, ACR_CLOSE])
    model = CrossEncoder(encoder, hidden=encoder.config.hidden_size, pooling=pooling)
    model.initialization = {"seed": seed, "pooling": pooling, "dropout": model.dropout.p,
                            "source": source_identity(model_id, revision, encoder),
                            "tokenizer": tokenizer_identity(tok), "libraries": library_versions()}
    if device is not None:
        model.to(device)
    return tok, model, acr_open_id, acr_close_id


def metadata_path(checkpoint_path) -> Path:
    return Path(str(checkpoint_path) + ".json")


def save_checkpoint(model, tok, checkpoint_path, config, inputs, epoch, dev_loss):
    """Save state_dict plus a versioned, weight-bound reconstruction manifest."""
    from dataclasses import asdict
    initialization = model.initialization
    if config.seed != initialization["seed"] or config.pooling != model.pooling:
        raise ValueError("Training config disagrees with model initialization")
    if tokenizer_identity(tok) != initialization["tokenizer"]:
        raise ValueError("Tokenizer changed since model initialization")
    source = initialization["source"]
    if not source["local_files_sha256"] and not source["revision"]:
        raise ValueError("Checkpoint requires a resolved model revision or local snapshot identity")
    torch.save(model.state_dict(), checkpoint_path)
    metadata = {"format_version": 1, "initialization": initialization,
                "training_config": asdict(config), "inputs": inputs,
                "selection": {"rule": "strict_dev_pair_loss_improvement", "epoch": epoch,
                              "dev_pair_loss": dev_loss},
                "weights_sha256": file_digest(checkpoint_path)}
    metadata_path(checkpoint_path).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def inspect_checkpoint(checkpoint_path, *, snapshot_path=None) -> dict:
    """Inspect original artifacts locally, without constructing a model or tokenizer.

    An explicit relocated snapshot is accepted only when its complete recorded
    file inventory has identical hashes. Hashless Hub identities cannot establish
    equivalence to an arbitrary local directory. Tokenizer reconstruction is still
    checked by load_finetuned; this inspection does not claim inference success.
    """
    if checkpoint_path is None or not str(checkpoint_path).strip():
        raise ValueError("Set checkpoint to Ben's weights file; the adjacent original JSON is required.")
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    path = metadata_path(checkpoint_path)
    if not path.is_file():
        raise ValueError(f"Checkpoint metadata is required: {path}. Obtain Ben's original JSON.")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format_version")
    weights_hash = file_digest(checkpoint_path)
    if weights_hash != metadata["weights_sha256"]:
        raise ValueError("Checkpoint weights do not match metadata")
    saved = metadata["initialization"]
    actual_libraries = library_versions()
    if saved["libraries"] != actual_libraries:
        raise ValueError(f"Checkpoint library identity mismatch: required {saved['libraries']}; "
                         f"installed {actual_libraries}. Use a matching isolated environment.")
    source = saved["source"]
    load_model_id = source["model_id"]
    if snapshot_path is not None:
        if not source["local_files_sha256"]:
            raise ValueError("Snapshot relocation requires original local file hashes in the manifest")
        load_model_id = str(Path(snapshot_path).expanduser().resolve())
    if source["local_files_sha256"]:
        if not Path(load_model_id).is_dir():
            raise FileNotFoundError(f"Recorded snapshot is missing: {load_model_id}. "
                                    "Supply an identical local copy with snapshot_path.")
        if _local_source_files(load_model_id) != source["local_files_sha256"]:
            raise ValueError("Snapshot content identity mismatch with the original model/tokenizer files")
    return {"metadata": metadata, "weights_sha256": weights_hash,
            "manifest_sha256": file_digest(path), "load_model_id": load_model_id,
            "relocated": load_model_id != source["model_id"],
            "libraries": actual_libraries, "tokenizer_reconstruction": "not yet checked"}


def load_finetuned(checkpoint_path: str, pooling: str | None = None, device: str = "cpu", *,
                   model_id=None, revision=None, expected_inputs=None, config=None,
                   snapshot_path=None):
    """Reconstruct from saved settings; fail on incompatible overrides or artifacts.

    Legacy weights without metadata are intentionally not guessed. This restores
    inference, not optimizer state for resuming an interrupted training run.
    snapshot_path permits only a content-identical local snapshot relocation;
    the original manifest, tokenizer, revision and library checks remain intact.
    """
    from hebrew_acronyms.models.dictabert_cross_encoder.training import TrainingConfig
    from dataclasses import asdict
    inspection = inspect_checkpoint(checkpoint_path, snapshot_path=snapshot_path)
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    metadata = inspection["metadata"]
    settings = TrainingConfig(**metadata["training_config"])
    saved = metadata["initialization"]
    source = saved["source"]
    if pooling is not None and pooling != settings.pooling:
        raise ValueError("Pooling override disagrees with checkpoint")
    if model_id is not None and str(model_id) != source["model_id"]:
        raise ValueError("Model override disagrees with checkpoint")
    if revision is not None and revision != source["revision"]:
        raise ValueError("Revision override disagrees with checkpoint")
    if config is not None and asdict(config) != metadata["training_config"]:
        raise ValueError("Training config override disagrees with checkpoint")
    if expected_inputs is not None and expected_inputs != metadata["inputs"]:
        raise ValueError("Input identities disagree with checkpoint")
    tok, model, acr_open_id, acr_close_id = build_cross_encoder(
        model_id=inspection["load_model_id"], revision=source["revision"],
        pooling=settings.pooling, device=None, seed=settings.seed)
    reconstructed = deepcopy(model.initialization)
    if inspection["relocated"]:
        # Only the location may differ; content hashes were checked before loading
        # and are compared again here alongside tokenizer and library identities.
        reconstructed["source"]["model_id"] = source["model_id"]
    if reconstructed != saved:
        raise ValueError("Model, tokenizer, initialization or library identity mismatch")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True), strict=True)
    model.training_config = settings
    model.checkpoint_metadata = metadata
    model.checkpoint_load_identity = {"load_model_id": inspection["load_model_id"],
                                      "relocated": inspection["relocated"]}
    model.to(device).eval()
    return tok, model, acr_open_id, acr_close_id
