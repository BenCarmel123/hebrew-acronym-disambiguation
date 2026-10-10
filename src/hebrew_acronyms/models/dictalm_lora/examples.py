"""Selection training examples in the exact prompt and answer format of the LLM arms.

Each item becomes one prompt from `build_select_prompt` with the candidates in a
seeded shuffled order, and one answer: the letter of the reference candidate.
Only the answer tokens are trained; the prompt tokens are masked from the loss.
"""
from __future__ import annotations

import random
import string

from hebrew_acronyms.models.common.eval import build_select_prompt
from hebrew_acronyms.models.common.pairs import MIN_CANDIDATES, candidates_for, validate_ids

#: Label value ignored by the causal language-model loss.
IGNORE_INDEX = -100

#: Letters available to the selection prompt and its parser.
MAX_CANDIDATES = len(string.ascii_uppercase)


def build_selection_examples(rows: list[dict], seed: int) -> list[dict]:
    """One (prompt, answer letter) example per item; reject rows instead of skipping them."""
    validate_ids(rows)
    rng = random.Random(seed)
    examples = []
    for row in rows:
        try:
            candidates = candidates_for(row)
            gold = row.get("gold_expansion")
            if not isinstance(gold, str) or not gold.strip():
                raise ValueError("A nonempty gold_expansion is required for training")
            gold = gold.strip()
            if sum(candidate == gold for candidate in candidates) != 1:
                raise ValueError("Training item requires exactly one positive candidate")
            if not MIN_CANDIDATES <= len(candidates) <= MAX_CANDIDATES:
                raise ValueError(f"Training requires between {MIN_CANDIDATES} and {MAX_CANDIDATES} candidates")
            if len(set(candidates)) != len(candidates):
                raise ValueError("Duplicate candidate entries are not supported")
            shuffled = candidates[:]
            rng.shuffle(shuffled)
            prompt = build_select_prompt(row["acronym"], row["sentence"], shuffled)
            answer = string.ascii_uppercase[shuffled.index(gold)]
        except (KeyError, ValueError) as error:
            raise ValueError(f"Item {row.get('item_id')!r}: {error}") from error
        examples.append({"item_id": row["item_id"], "prompt": prompt, "answer": answer,
                         "shown_order": shuffled})
    return examples


def format_prompt(tok, prompt: str) -> str:
    """Wrap the prompt in the model's chat template, or leave it unchanged without one."""
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
    return prompt


def prompt_token_ids(tok, prompt: str) -> list[int]:
    """Token IDs for a formatted prompt; the chat template already supplies BOS."""
    has_template = bool(getattr(tok, "chat_template", None))
    return tok(format_prompt(tok, prompt), add_special_tokens=not has_template)["input_ids"]


def encode_example(tok, example: dict, max_len: int) -> dict:
    """Prompt and answer token IDs, with the loss restricted to the answer and EOS."""
    if tok.eos_token_id is None:
        raise ValueError("The tokenizer requires an EOS token to end the answer")
    prompt_ids = prompt_token_ids(tok, example["prompt"])
    answer_ids = tok(example["answer"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    input_ids = prompt_ids + answer_ids
    if len(input_ids) > max_len:
        raise ValueError(f"Item {example['item_id']!r} needs {len(input_ids)} tokens; max_len is {max_len}")
    return {"input_ids": input_ids,
            "labels": [IGNORE_INDEX] * len(prompt_ids) + answer_ids}


def collate(encoded: list[dict], pad_token_id: int) -> dict:
    """Right-pad a batch; padded positions are masked from attention and loss."""
    width = max(len(item["input_ids"]) for item in encoded)
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for item in encoded:
        pad = width - len(item["input_ids"])
        batch["input_ids"].append(item["input_ids"] + [pad_token_id] * pad)
        batch["attention_mask"].append([1] * len(item["input_ids"]) + [0] * pad)
        batch["labels"].append(item["labels"] + [IGNORE_INDEX] * pad)
    return batch
