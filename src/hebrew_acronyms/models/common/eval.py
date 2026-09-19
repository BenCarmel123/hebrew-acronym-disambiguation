"""Shared open-generation / candidate-select evaluation loop, usable against any LLM
backend. A backend module (src/hebrew_acronyms/models/qwen/eval.py, src/hebrew_acronyms/models/gemini/eval.py, ...) only
needs to supply a `generate_fn(prompt: str) -> str` and call `evaluate()`.
"""
from __future__ import annotations

import random
import string

from tqdm import tqdm

from hebrew_acronyms.models.common.pairs import _normalise

#: Seed for one random generator per evaluation call. Reproducing candidate order
#: requires the same ordered inputs and candidate lists; it is not keyed by item ID.
SHUFFLE_SEED = 42


def build_generate_prompt(acronym: str, sentence: str) -> str:
    return (
        f"בהתחשב במשפט הבא בעברית, מהו הפירוש של ראשי התיבות \"{acronym}\"?\n"
        f"ענה במילים ספורות בלבד, ללא הסבר.\n\n"
        f"משפט: {sentence}\n"
        f"פירוש:"
    )


def build_select_prompt(acronym: str, sentence: str, shuffled_candidates: list[str]) -> str:
    """Request a letter choice using the candidate order supplied by the caller."""
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
    # In select mode, each scored row advances this generator. Earlier rows and their
    # candidate counts therefore affect the order shown for later rows.
    rng = random.Random(SHUFFLE_SEED)
    details = []  # per-item record, so errors can be sliced later (by acronym, type, etc.)
    for r in tqdm(rows, desc=f"{mode} eval", unit="item"):
        # This evaluator skips rows with fewer than two candidates or an empty gold.
        # Unlike the encoder evaluators, it does not check the target's text span.
        cands = [c.strip() for c in r["candidates"].split("|") if c.strip()]
        gold = r["gold_expansion"].strip()
        if len(cands) < 2 or not gold:
            continue
        n += 1

        # Generate mode shows no candidates. Select mode shuffles their order and
        # requests a letter; the displayed order is saved with the response.
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
