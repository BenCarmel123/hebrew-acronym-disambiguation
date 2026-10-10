"""Candidate selection with a DictaLM model, with or without a trained LoRA adapter.

Scoring, candidate order and answer parsing come from the shared LLM evaluator,
so results are comparable with the other selection arms. Omit --adapter to score
the untrained model through the same inference code.

    python -m hebrew_acronyms.models.dictalm_lora.eval --model-id MODEL \
        --adapter RUN_DIR/adapter --test data/splits/test_items.csv --output details.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from hebrew_acronyms.data_processing.common.csv_io import load_rows
from hebrew_acronyms.models.common.eval import evaluate, write_details_csv
from hebrew_acronyms.models.dictalm_lora.examples import prompt_token_ids
from hebrew_acronyms.models.dictalm_lora.training import DTYPES, input_device, load_base_model

#: A trained answer is one letter and EOS; a few extra tokens keep untrained output visible.
MAX_NEW_TOKENS = 8


def load_adapter(model, adapter_dir: str | Path):
    from peft import PeftModel
    return PeftModel.from_pretrained(model, str(adapter_dir))


def make_generate_fn(model, tok, max_new_tokens: int = MAX_NEW_TOKENS):
    """Greedy `generate_fn(prompt) -> str` in the same chat format used for training."""
    model.eval()
    device = input_device(model)

    @torch.no_grad()
    def generate_fn(prompt: str) -> str:
        ids = torch.tensor([prompt_token_ids(tok, prompt)], device=device)
        output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                max_new_tokens=max_new_tokens, do_sample=False,
                                pad_token_id=tok.pad_token_id)
        return tok.decode(output[0, ids.shape[1]:], skip_special_tokens=True)

    return generate_fn


def evaluate_selection(rows: list[dict], model, tok) -> dict:
    """Shared selection scoring: seeded candidate order and one-letter answers."""
    return evaluate(rows, make_generate_fn(model, tok), mode="select")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--adapter", type=Path, default=None, help="Omit to score the untrained model")
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True, help="Per-item details CSV")
    ap.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Choose a new output path: {args.output}")
    tok, model = load_base_model(args.model_id, args.revision, dtype=args.dtype,
                                 load_in_4bit=args.load_in_4bit)
    if args.adapter is not None:
        model = load_adapter(model, args.adapter)
    result = evaluate_selection(load_rows(args.test), model, tok)
    write_details_csv(str(args.output), result["details"])
    print(f"n={result['n_items']}  accuracy={result['accuracy']}  invalid_rate={result['invalid_rate']}")


if __name__ == "__main__":
    main()
