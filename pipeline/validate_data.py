"""Sanity-check a train/dev(/test) set before running the eval pipeline on it.

Catches a bad dataset swap early (missing columns, empty gold, no real choice to make,
type/sentence/page leakage between splits) instead of failing deep inside some arm's
eval.py with a confusing error, or — worse — running to completion on quietly broken
data.

    python -m pipeline.validate_data --train data/splits/train_items.csv --dev data/splits/dev_items.csv
    python -m pipeline.validate_data --train data/splits/train_items.csv --dev data/splits/dev_items.csv \\
        --test data/splits/test_items.csv
"""
from __future__ import annotations

import argparse
import sys

from model.common.pairs import load_rows

REQUIRED_COLUMNS = {
    "item_id", "acronym", "sentence", "gold_expansion", "candidates",
}


def _cross_split_overlaps(
    named_rows: list[tuple[str, list[dict]]]
) -> tuple[list[str], list[str]]:
    """Pairwise type/sentence/page overlap checks across all given splits.

    -> (errors, warnings). Type and sentence overlap are errors: either means
    an eval split can't measure what it's supposed to. Page-title overlap is
    a warning, not an error — it is a real but much softer leakage channel
    (the same source document can supply both a train and an eval row with
    different acronym types and no literal duplicate), and for a source like
    the Knesset Corpus, where one long multi-topic protocol transcript
    routinely mentions several unrelated acronym types, some overlap here is
    close to unavoidable without shrinking an already-thin stratified split
    further. See data/mined/DATASET_CARD.md's "known, accepted leakage
    channel" note.
    """
    errors: list[str] = []
    warnings: list[str] = []
    for i in range(len(named_rows)):
        for j in range(i + 1, len(named_rows)):
            name_a, rows_a = named_rows[i]
            name_b, rows_b = named_rows[j]

            types_a = {r["acronym"] for r in rows_a}
            types_b = {r["acronym"] for r in rows_b}
            overlap = types_a & types_b
            if overlap:
                errors.append(
                    f"{name_a}/{name_b} share {len(overlap)} acronym type(s) — splits "
                    f"should be type-disjoint, or eval measures memorization rather "
                    f"than generalization (see data/splits/README.md): "
                    f"{sorted(overlap)[:10]}" + (" ..." if len(overlap) > 10 else "")
                )

            sentences_a = {r["sentence"].strip() for r in rows_a if r.get("sentence", "").strip()}
            sentences_b = {r["sentence"].strip() for r in rows_b if r.get("sentence", "").strip()}
            sent_overlap = sentences_a & sentences_b
            if sent_overlap:
                errors.append(
                    f"{name_a}/{name_b} share {len(sent_overlap)} identical sentence(s)"
                )

            pages_a = {r["page_title"].strip() for r in rows_a if r.get("page_title", "").strip()}
            pages_b = {r["page_title"].strip() for r in rows_b if r.get("page_title", "").strip()}
            page_overlap = pages_a & pages_b
            if page_overlap:
                warnings.append(
                    f"{name_a}/{name_b} share {len(page_overlap)} source page_title(s) — "
                    f"same document can supply near-identical prose to both sides even "
                    f"when acronym types are disjoint: {sorted(page_overlap)[:10]}"
                    + (" ..." if len(page_overlap) > 10 else "")
                )
    return errors, warnings


def validate(
    train_path: str, dev_path: str, test_path: str | None = None
) -> tuple[list[str], list[str]]:
    """-> (errors, warnings). Empty errors means the split set is usable —
    a non-empty warnings list is worth reading but not a reason to stop."""
    problems = []

    named_paths = [("train", train_path), ("dev", dev_path)]
    if test_path:
        named_paths.append(("test", test_path))

    named_rows: list[tuple[str, list[dict]]] = []
    for name, path in named_paths:
        rows = load_rows(path)
        if not rows:
            problems.append(f"{path} is empty")
        named_rows.append((name, rows))

    if any(not rows for _, rows in named_rows):
        return problems, []  # nothing further can be checked meaningfully

    for name, rows in named_rows:
        missing = REQUIRED_COLUMNS - set(rows[0].keys())
        if missing:
            problems.append(f"{name}: missing required column(s) {sorted(missing)}")

    for name, rows in named_rows:
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

    errors, warnings = _cross_split_overlaps(named_rows)
    problems.extend(errors)

    return problems, warnings


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="data/splits/train_items.csv")
    ap.add_argument("--dev", default="data/splits/dev_items.csv")
    ap.add_argument("--test", default=None, help="optional held-out test split")
    a = ap.parse_args()

    problems, warnings = validate(a.train, a.dev, a.test)
    label = f"{a.train}, {a.dev}" + (f", {a.test}" if a.test else "")

    if warnings:
        print(f"{len(warnings)} warning(s) (not fatal) with {label}:\n")
        for w in warnings:
            print(f"  - {w}")
        print()

    if not problems:
        print(f"OK: {label} pass all checks.")
        sys.exit(0)

    print(f"Found {len(problems)} issue(s) with {label}:\n")
    for p in problems:
        print(f"  - {p}")
    sys.exit(1)


if __name__ == "__main__":
    main()
