from __future__ import annotations

import argparse
import csv

import requests

from model.common.pairs import _normalise, load_rows

OLLAMA_URL = "http://localhost:11434/api/generate"


def ollama_generate(prompt: str, model: str = "qwen2.5:7b") -> str:
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
    )
    response.raise_for_status()
    return response.json()["response"]


def build_generate_prompt(acronym: str, sentence: str) -> str:
    return (
        f"בהתחשב במשפט הבא בעברית, מהו הפירוש של ראשי התיבות \"{acronym}\"?\n"
        f"ענה במילים ספורות בלבד, ללא הסבר.\n\n"
        f"משפט: {sentence}\n"
        f"פירוש:"
    )


def is_correct(response: str, gold: str) -> bool:
    """Loose match: gold's text appears in the model's response, quote-folded."""
    return _normalise(gold) in _normalise(response)


def is_valid(response: str, candidates: list[str]) -> bool:
    """Whether the response matches ANY candidate, not just the gold one."""
    return any(_normalise(c) in _normalise(response) for c in candidates)


def evaluate(rows: list[dict], model: str) -> dict:
    n = 0
    correct = 0
    invalid = 0
    details = []  # per-item record, so errors can be sliced later (by acronym, type, etc.)
    for r in rows:
        # Same skip rule as baselines.py/pairs.py: a single-candidate item has no real
        # choice to make, and a missing gold can't be scored either way.
        cands = [c.strip() for c in r["candidates"].split("|") if c.strip()]
        gold = r["gold_expansion"].strip()
        if len(cands) < 2 or not gold:
            continue
        n += 1

        # No candidate list goes into the prompt here — this is the open-generation
        # arm, so the model must produce the expansion from scratch.
        prompt = build_generate_prompt(r["acronym"], r["sentence"])
        response = ollama_generate(prompt, model=model)

        correct_item = is_correct(response, gold)
        # "valid" tracks whether the free-text response lands on ANY real candidate,
        # not just the gold one — this is what the invalid-expansion-rate metric needs.
        valid_item = is_valid(response, cands)
        if correct_item:
            correct += 1
        if not valid_item:
            invalid += 1

        details.append({
            "item_id": r.get("item_id", ""),
            "acronym": r["acronym"],
            "gold": gold,
            "candidates": cands,
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", default="data/splits/dev_items.csv")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--out", default="data/mined/llm_generate_details.csv")
    a = ap.parse_args()

    res = evaluate(load_rows(a.items), model=a.model)
    print(f"{a.items}  (model: {a.model})\n")
    print(f"  items scored      {res['n_items']}")
    print(f"  accuracy          {res['accuracy']:.3f}")
    print(f"  invalid rate      {res['invalid_rate']:.3f}   response matched no candidate")

    with open(a.out, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "item_id", "acronym", "gold", "candidates", "response",
            "correct", "valid", "multi_sense_type",
        ])
        writer.writeheader()
        for d in res["details"]:
            row = dict(d)
            row["candidates"] = " | ".join(row["candidates"])
            writer.writerow(row)
    print(f"\n  per-item details written to {a.out}")


if __name__ == "__main__":
    main()
