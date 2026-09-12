"""Sanity-check a train/dev pair before running the eval pipeline on it.

Catches a bad dataset swap early (missing columns, empty gold, no real choice to make,
type leakage between train and dev) instead of failing deep inside some arm's eval.py
with a confusing error, or — worse — running to completion on quietly broken data.

    python -m pipeline.validate_data --train data/splits/train_items.csv --dev data/splits/dev_items.csv
"""
from __future__ import annotations

import argparse
import sys

from model.common.pairs import load_rows

REQUIRED_COLUMNS = {
    "item_id", "acronym", "sentence", "gold_expansion", "candidates",
}


def validate(train_path: str, dev_path: str) -> list[str]:
    """-> list of problem descriptions. Empty means the pair is usable."""
    problems = []

    train_rows = load_rows(train_path)
    dev_rows = load_rows(dev_path)

    if not train_rows:
        problems.append(f"{train_path} is empty")
    if not dev_rows:
        problems.append(f"{dev_path} is empty")
    if not train_rows or not dev_rows:
        return problems  # nothing further can be checked meaningfully

    for name, rows in [("train", train_rows), ("dev", dev_rows)]:
        missing = REQUIRED_COLUMNS - set(rows[0].keys())
        if missing:
            problems.append(f"{name}: missing required column(s) {sorted(missing)}")

    for name, rows in [("train", train_rows), ("dev", dev_rows)]:
        no_gold = sum(1 for r in rows if not r.get("gold_expansion", "").strip())
        if no_gold:
            problems.append(f"{name}: {no_gold}/{len(rows)} rows have an empty gold_expansion")

        too_few = sum(
            1 for r in rows
            if len([c for c in r.get("candidates", "").split("|") if c.strip()]) < 2
        )
        if too_few:
            problems.append(
                f"{name}: {too_few}/{len(rows)} rows have fewer than 2 candidates "
                "(no real choice to make)"
            )

        gold_not_in_candidates = sum(
            1 for r in rows
            if r.get("gold_expansion", "").strip()
            and r.get("gold_expansion", "").strip()
            not in [c.strip() for c in r.get("candidates", "").split("|")]
        )
        if gold_not_in_candidates:
            problems.append(
                f"{name}: {gold_not_in_candidates}/{len(rows)} rows have a gold_expansion "
                "not present in their own candidates list"
            )

    train_types = {r["acronym"] for r in train_rows}
    dev_types = {r["acronym"] for r in dev_rows}
    overlap = train_types & dev_types
    if overlap:
        problems.append(
            f"train/dev share {len(overlap)} acronym type(s) — dev should be "
            f"type-disjoint from train, or it measures memorization rather than "
            f"generalization (see data/splits/README.md): {sorted(overlap)[:10]}"
            + (" ..." if len(overlap) > 10 else "")
        )

    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="data/splits/train_items.csv")
    ap.add_argument("--dev", default="data/splits/dev_items.csv")
    a = ap.parse_args()

    problems = validate(a.train, a.dev)
    if not problems:
        print(f"OK: {a.train} and {a.dev} pass all checks.")
        sys.exit(0)

    print(f"Found {len(problems)} issue(s) with {a.train} / {a.dev}:\n")
    for p in problems:
        print(f"  - {p}")
    sys.exit(1)


if __name__ == "__main__":
    main()
