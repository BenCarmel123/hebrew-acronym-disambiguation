"""LoRA training of a DictaLM causal LM on candidate selection, with dev-loss checkpointing.

The settings below are engineering defaults for a first run, not approved
experimental choices. Every run saves its settings beside the adapter.

    python -m hebrew_acronyms.models.dictalm_lora.training --model-id MODEL \
        --train data/study_v1/encoder_inputs/train.csv \
        --dev data/study_v1/encoder_inputs/dev.csv --output-dir RUN_DIR
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time

import torch

from hebrew_acronyms.data_processing.common.csv_io import load_rows
from hebrew_acronyms.models.common.pairs import input_identity
from hebrew_acronyms.models.dictalm_lora.examples import (
    IGNORE_INDEX, build_selection_examples, collate, encode_example,
)

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


@dataclass(frozen=True)
class LoraTrainingConfig:
    """First-run settings; the effective batch is batch_size * grad_accum items."""

    epochs: int = 1
    batch_size: int = 4
    grad_accum: int = 4
    eval_batch_size: int = 8
    lr: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    max_len: int = 1024
    seed: int = 42

    def __post_init__(self):
        for name in ("epochs", "batch_size", "grad_accum", "eval_batch_size",
                     "lora_r", "lora_alpha", "max_len"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 < self.lr < float("inf") or not 0 <= self.lora_dropout < 1:
            raise ValueError("lr must be positive and lora_dropout in [0, 1)")
        if type(self.seed) is not int or not self.target_modules:
            raise ValueError("Invalid seed or empty target_modules")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_base_model(model_id: str, revision: str | None = None, *, dtype: str = "bfloat16",
                    load_in_4bit: bool = False):
    """Load tokenizer and causal LM wholly on one device; 4-bit loading needs bitsandbytes and CUDA.

    The model is never split or offloaded: offloaded layers cannot be trained, so a
    model that does not fit raises an out-of-memory error instead.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}")
    device_map = {"": 0} if torch.cuda.is_available() else None
    kwargs = {"revision": revision, "dtype": DTYPES[dtype], "device_map": device_map}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=DTYPES[dtype])
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tok.pad_token_id is None:
        # Padding is masked from attention and loss, so reusing EOS changes no target.
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    offloaded = {name: str(p.device) for name, p in model.named_parameters() if p.device.type == "meta"}
    if offloaded:
        raise RuntimeError(f"{len(offloaded)} parameters are not on the GPU; free GPU memory "
                           "(restart the runtime) or load in 4-bit")
    return tok, model


def add_lora(model, config: LoraTrainingConfig, *, quantized: bool = False,
             gradient_checkpointing: bool = True):
    """Freeze the base model and add LoRA adapters; the seed is set before their initialization."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if quantized:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    elif gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    set_seed(config.seed)
    lora = LoraConfig(r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
                      target_modules=list(config.target_modules), task_type="CAUSAL_LM")
    return get_peft_model(model, lora)


def input_device(model):
    return model.get_input_embeddings().weight.device


def _batch_loss(model, tok, encoded, device):
    """Mean answer-token loss for the batch and the number of answer tokens."""
    batch = {key: torch.tensor(value, device=device)
             for key, value in collate(encoded, tok.pad_token_id).items()}
    n_tokens = int((batch["labels"][:, 1:] != IGNORE_INDEX).sum())
    return model(**batch).loss, n_tokens


@torch.no_grad()
def evaluate_loss(model, tok, encoded, config: LoraTrainingConfig) -> float:
    """Answer-token loss over all examples, weighted by answer tokens."""
    if not encoded:
        raise ValueError("Evaluation examples must be nonempty")
    model.eval()
    device = input_device(model)
    total, n = 0.0, 0
    for i in range(0, len(encoded), config.eval_batch_size):
        loss, n_tokens = _batch_loss(model, tok, encoded[i:i + config.eval_batch_size], device)
        total += loss.item() * n_tokens
        n += n_tokens
    return total / n


def save_adapter(model, output_dir: Path, metadata: dict) -> None:
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    (output_dir / "training_run.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def train(model, tok, train_rows, dev_rows, output_dir: str | Path,
          config: LoraTrainingConfig = LoraTrainingConfig(), run_info: dict | None = None) -> list[dict]:
    """Train the adapter; save it only when the development loss strictly improves."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {output_dir}")
    # Build and check every example before any update, so a long item fails early.
    train_examples = build_selection_examples(train_rows, config.seed)
    dev_examples = build_selection_examples(dev_rows, config.seed)
    train_encoded = [encode_example(tok, example, config.max_len) for example in train_examples]
    dev_encoded = [encode_example(tok, example, config.max_len) for example in dev_examples]
    output_dir.mkdir(parents=True)

    set_seed(config.seed)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("The model has no trainable parameters; call add_lora first")
    optimizer = torch.optim.AdamW(params, lr=config.lr)
    device = input_device(model)
    metadata = {"config": asdict(config), "run_info": dict(run_info or {}),
                "inputs": {"train": input_identity(train_rows), "dev": input_identity(dev_rows)}}
    history, best_dev_loss = [], float("inf")
    for epoch in range(1, config.epochs + 1):
        model.train()
        t0 = time.time()
        order = list(range(len(train_encoded)))
        random.shuffle(order)
        batches = [order[i:i + config.batch_size] for i in range(0, len(order), config.batch_size)]
        running, n_seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, indices in enumerate(batches, start=1):
            loss, n_tokens = _batch_loss(model, tok, [train_encoded[i] for i in indices], device)
            (loss / config.grad_accum).backward()
            running += loss.item() * n_tokens
            n_seen += n_tokens
            if step % config.grad_accum == 0 or step == len(batches):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 50 == 0:
                print(f"  step {step}/{len(batches)}  train_loss={running / n_seen:.4f}")
        train_loss = running / n_seen
        dev_loss = evaluate_loss(model, tok, dev_encoded, config)
        print(f"epoch {epoch}/{config.epochs}  train_loss={train_loss:.4f}  "
              f"dev_loss={dev_loss:.4f}  ({time.time() - t0:.0f}s)")
        saved = dev_loss < best_dev_loss
        if saved:
            best_dev_loss = dev_loss
            save_adapter(model, output_dir, {**metadata, "epoch": epoch, "dev_loss": dev_loss})
            print(f"  -> saved {output_dir / 'adapter'}")
        history.append({"epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss,
                        "checkpoint_saved": saved})
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-id", required=True, help="Hugging Face ID of the DictaLM model")
    ap.add_argument("--revision", default=None, help="Model revision (commit SHA preferred)")
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--dev", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--epochs", type=int, default=LoraTrainingConfig.epochs)
    ap.add_argument("--lr", type=float, default=LoraTrainingConfig.lr)
    args = ap.parse_args(argv)
    config = LoraTrainingConfig(epochs=args.epochs, lr=args.lr)
    tok, model = load_base_model(args.model_id, args.revision, dtype=args.dtype,
                                 load_in_4bit=args.load_in_4bit)
    model = add_lora(model, config, quantized=args.load_in_4bit)
    run_info = {"model_id": args.model_id, "revision": args.revision,
                "resolved_commit": getattr(model.config, "_commit_hash", None),
                "dtype": args.dtype, "load_in_4bit": args.load_in_4bit}
    train(model, tok, load_rows(args.train), load_rows(args.dev), args.output_dir, config, run_info)


if __name__ == "__main__":
    main()
