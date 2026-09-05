"""Interactive terminal reviewer for `duplicate_review.csv`.

Shows one flagged pair at a time and asks for a decision. Saves after every
answer, so quitting partway through loses nothing already decided.

    python -m data_preprocess.review_duplicates_cli data/mined/duplicate_review.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from .dedupe_expansions import REVIEW_FIELDS

KEYS = {"m": "merge", "k": "keep", "s": "", "q": None}


def load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def save(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)  # atomic: a Ctrl-C mid-write never corrupts the file


def iso(text: str) -> str:
    """Make a Hebrew phrase read correctly in a terminal with broken bidi.

    VS Code's integrated terminal, placed after LTR text like "(hits=619)",
    reverses both the order of words AND the letters within each word.
    Reversing the whole string once counteracts both at the same time. This is
    a display-only hack scoped to this interactive tool \u2014 anywhere else it
    would make the phrase read backwards.
    """
    return text[::-1]


def prompt(row: dict, index: int, total: int, pending: int) -> str | None:
    print(f"\n[{index + 1}/{total}]  ({pending} left to decide)  acronym: {iso(row['acronym'])}   similarity={row['similarity']}")
    print(f"  a: (hits={row['hits_a']})  {iso(row['expansion_a'])}")
    print(f"  b: (hits={row['hits_b']})  {iso(row['expansion_b'])}")
    while True:
        answer = input("  merge / keep / skip / quit  [m/k/s/q]: ").strip().lower()
        if answer in KEYS:
            return KEYS[answer]
        print("  please enter m, k, s, or q")


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    path = Path(argv[0] if argv else "data/mined/duplicate_review.csv")
    rows = load(path)
    pending = [r for r in rows if not r.get("decision", "").strip()]
    print(f"{len(rows)} pairs total, {len(pending)} undecided.")

    for i, row in enumerate(rows):
        if row.get("decision", "").strip():
            continue
        decision = prompt(row, i, len(rows), sum(1 for r in rows if not r.get("decision", "").strip()))
        if decision is None:
            print("Stopping. Progress saved.")
            break
        row["decision"] = decision
        save(path, rows)
    else:
        print("All pairs decided.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
