"""Shared open-generation / candidate-select evaluation loop, usable against any LLM
backend. A backend module (model/qwen/eval.py, model/gemini/eval.py, ...) only
needs to supply a `generate_fn(prompt: str) -> str` and call `evaluate()`.
"""
from __future__ import annotations

import random
import string

from tqdm import tqdm

from model.common.pairs import _normalise, load_rows

#: Fixed per-item ordering, not per-run: an item's candidate order must not depend on
#: what ran before it, or re-running with --out only won't reproduce the same prompts.
SHUFFLE_SEED = 42


def build_generate_prompt(acronym: str, sentence: str) -> str:
    return (
        f"בהתחשב במשפט הבא בעברית, מהו הפירוש של ראשי התיבות \"{acronym}\"?\n"
        f"ענה במילים ספורות בלבד, ללא הסבר.\n\n"
        f"משפט: {sentence}\n"
        f"פירוש:"
    )


def build_select_prompt(acronym: str, sentence: str, shuffled_candidates: list[str]) -> str:
    """`shuffled_candidates` must already be in the order to display — shuffling is the
    caller's job (see evaluate), so this function alone can't accidentally reintroduce
    positional bias by re-deriving its own order.

    Asks for a letter, not the candidate text: a small model reliably outputs a single
    letter but does not reliably copy a multi-word Hebrew string verbatim.
    """
    letters = string.ascii_uppercase[:len(shuffled_candidates)]
    options = "\n".join(f"{l}. {c}" for l, c in zip(letters, shuffled_candidates))
    return (
        f"בהתחשב במשפט הבא בעברית, מהו הפירוש של ראשי התיבות \"{acronym}\"?\n"
        f"ענה באות אחת בלבד (למשל: A), בלי שום טקסט נוסף.\n\n"
        f"משפט: {sentence}\n\n"
        f"{options}\n\n"
        f"תשובה:"
    )


def parse_letter_choice(response: str, n_candidates: int) -> int | None:
    """First A/B/C... in the response -> its 0-based index, or None if none found."""
    valid_letters = string.ascii_uppercase[:n_candidates]
    for ch in response.strip().upper():
        if ch in valid_letters:
            return valid_letters.index(ch)
    return None


def is_correct(response: str, gold: str) -> bool:
    """Loose match: gold's text appears in the model's response, quote-folded."""
    return _normalise(gold) in _normalise(response)


def is_valid(response: str, candidates: list[str]) -> bool:
    """Whether the response matches ANY candidate, not just the gold one."""
    return any(_normalise(c) in _normalise(response) for c in candidates)


def evaluate(rows: list[dict], generate_fn, mode: str = "generate") -> dict:
    """`generate_fn(prompt: str) -> str` — the one thing that differs per LLM backend."""
    n = 0
    correct = 0
    invalid = 0
    # One Random instance, seeded once: shuffles are independent draws across items,
    # not the same permutation repeated, while still reproducing identically run to run.
    rng = random.Random(SHUFFLE_SEED)
    details = []  # per-item record, so errors can be sliced later (by acronym, type, etc.)
    for r in tqdm(rows, desc=f"{mode} eval", unit="item"):
        # Same skip rule as baselines.py/pairs.py: a single-candidate item has no real
        # choice to make, and a missing gold can't be scored either way.
        cands = [c.strip() for c in r["candidates"].split("|") if c.strip()]
        gold = r["gold_expansion"].strip()
        if len(cands) < 2 or not gold:
            continue
        n += 1

        # "generate": no candidate list shown, the model produces the expansion from
        # scratch. "select": the candidate list is shown (order shuffled per item, so
        # a model with a positional bias like "always pick A" scores at chance instead
        # of being flattered by gold sitting first) and the model answers with a letter.
        shown_order = None  # only meaningful in select mode; logged so the CSV shows
                            # what the model actually saw, not just the row's raw order
        if mode == "generate":
            prompt = build_generate_prompt(r["acronym"], r["sentence"])
            response = generate_fn(prompt)
            correct_item = is_correct(response, gold)
            valid_item = is_valid(response, cands)
        elif mode == "select":
            shuffled = cands[:]
            rng.shuffle(shuffled)
            shown_order = shuffled
            prompt = build_select_prompt(r["acronym"], r["sentence"], shuffled)
            response = generate_fn(prompt)
            choice = parse_letter_choice(response, len(shuffled))
            valid_item = choice is not None
            correct_item = valid_item and shuffled[choice] == gold
        else:
            raise ValueError(f"unknown mode {mode!r}")

        if correct_item:
            correct += 1
        if not valid_item:
            invalid += 1

        details.append({
            "item_id": r.get("item_id", ""),
            "acronym": r["acronym"],
            "gold": gold,
            "candidates": cands,
            "shown_order": shown_order,
            "response": response,
            "correct": correct_item,
            "valid": valid_item,
            "multi_sense_type": r.get("multi_sense_type", ""),
        })

    return {
        "n_items": n,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "invalid_rate": round(invalid / n, 4) if n else 0.0,
        "details": details,
    }


def write_details_csv(path: str, details: list[dict]) -> None:
    import csv
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "item_id", "acronym", "gold", "candidates", "shown_order", "response",
            "correct", "valid", "multi_sense_type",
        ])
        writer.writeheader()
        for d in details:
            row = dict(d)
            row["candidates"] = " | ".join(row["candidates"])
            row["shown_order"] = " | ".join(row["shown_order"]) if row["shown_order"] else ""
            writer.writerow(row)
