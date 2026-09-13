"""Build train/dev/test from the full labeled pool: existing train+dev rows,
plus human-reviewed Knesset rows (data/mined/knesset/knesset_reviewed.csv).

Policy (decided with the user, 2026-09-13):

- **dev is frozen.** It is never rewritten by this script — it already went
  through its own substitution-damage review and stays the fixed iteration
  set it has been. Its acronym types are excluded from every allocation
  decision below, so nothing here can violate its type-disjointness from
  train.
- **test is Knesset-only**, built entirely from human-reviewed natural text
  (`knesset_reviewed.csv`). The 115 `label_status=verified` natural rows
  already in `acronym_items.csv` are deliberately left out of this pass —
  a different review history/schema, folded in later as a separate addition
  rather than complicating this algorithm now.
- **Knesset rows whose acronym type is already in dev are dropped entirely**
  (not train, not test) — dev already "owns" that type for evaluation
  purposes, and mixing another source's examples elsewhere would blur what
  each split measures.
- **Test-type selection is stratified and algorithmic**, not manual: ~20% of
  the remaining eligible Knesset types, chosen to mix ambiguity levels
  (n_candidates) and row-count levels (well-attested vs. thin), rather than
  cherry-picked or taken as a prefix of the type list.
- **Every reviewed Knesset type is already in the existing 549-type train+dev
  inventory — checked directly, zero exceptions.** So a genuinely type-disjoint
  ("unseen acronym") test set cannot be built by layering Knesset rows on top
  of an unchanged train: whatever types are chosen for test have to be
  actively removed from train, not merely not-added-to. This script does
  that removal — every selected test type's existing (Wikipedia/substituted)
  rows are dropped from train's output, and only that type's Knesset rows
  populate test. Train shrinks by the removed types' original row count.
- **Every other reviewed Knesset row (types not selected for test) goes to
  train**, capped at 8 rows/type so a handful of easy, high-yield types
  (e.g. "דוקטור", "שקל חדש") don't dominate train's natural-data share.
- `multi_sense_type` is left blank for new Knesset rows — checked against
  existing dev data and confirmed NOT mechanically re-derivable from
  n_candidates/gold_expansion alone (40/289 mismatches on that hypothesis),
  so guessing it would silently fabricate an annotation that was actually a
  human call when train/dev were first built.

Run after the Knesset review is complete:

    python -m data_preprocess.build_splits \\
        --train data/splits/train_items.csv \\
        --dev data/splits/dev_items.csv \\
        --knesset-reviewed data/mined/knesset/knesset_reviewed.csv \\
        --test-out data/splits/test_items.csv \\
        --train-out data/splits/train_items.csv \\
        --test-fraction 0.2 --train-cap-per-type 8

Prints a gap report (thin test types, skewed sense distributions) so the
user can spot exactly where a manually-authored example would help most —
this script never fabricates or pads data on its own.
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

TRAIN_DEV_FIELDS = [
    "item_id", "acronym", "sentence", "provisional_expansion", "sense_id",
    "gold_expansion", "label_status", "n_candidates", "candidates",
    "provenance", "multi_sense_type", "source", "page_title",
    "review_verdict", "review_note", "label_origin",
]

# Stratification buckets. n_candidates buckets separate "easy" (2-3 options)
# from "hard" (4+) items; row-count buckets separate well-attested types
# (mining found many clean sentences) from thin ones (found only 1-2) — a
# test set built purely from the easiest, best-attested types would overstate
# how well a model does on the task overall.
def _n_candidates_bucket(n: int) -> str:
    return "easy" if n <= 3 else "hard"


def _row_count_bucket(n: int) -> str:
    return "well_attested" if n >= 5 else "thin"


def load_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def knesset_row_to_split_schema(row: dict, item_id: str) -> dict:
    """Map a knesset_reviewed.csv row onto the train/dev column set.

    Fields with no Knesset equivalent are left blank rather than guessed —
    see the module docstring for `multi_sense_type` specifically.
    """
    return {
        "item_id": item_id,
        "acronym": row["acronym"],
        "sentence": row["sentence"],
        "provisional_expansion": row["gold_expansion"],
        "sense_id": "",
        "gold_expansion": row["gold_expansion"],
        "label_status": "verified",  # every row here passed human review
        "n_candidates": row["n_candidates"],
        "candidates": row["candidates"],
        "provenance": "knesset",
        "multi_sense_type": "",
        "source": row["source"],
        "page_title": row["page_title"],
        "review_verdict": row["review_verdict"],
        "review_note": "",
        "label_origin": "human_review",
    }


def select_test_types(
    knesset_rows: list[dict], *, fraction: float, seed: int = 42
) -> set[str]:
    """Pick a stratified sample of acronym types for the new test set.

    Types are grouped into 4 strata (easy/hard x well_attested/thin); each
    stratum contributes the same fraction, so test is not accidentally all
    easy-and-common or all hard-and-rare types. Selection is seeded for
    reproducibility — re-running this script on the same input reproduces
    the same test set.
    """
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in knesset_rows:
        by_type[r["acronym"]].append(r)

    strata: dict[str, list[str]] = defaultdict(list)
    for acronym, rows in by_type.items():
        n_cand = int(rows[0]["n_candidates"])
        stratum = f"{_n_candidates_bucket(n_cand)}_{_row_count_bucket(len(rows))}"
        strata[stratum].append(acronym)

    rng = random.Random(seed)
    selected: set[str] = set()
    for stratum, types in strata.items():
        types = sorted(types)  # deterministic order before shuffling
        rng.shuffle(types)
        k = max(1, round(len(types) * fraction)) if types else 0
        selected.update(types[:k])
    return selected


def build(
    train_path: str, dev_path: str, knesset_reviewed_path: str,
    *, test_fraction: float, train_cap_per_type: int, seed: int = 42,
) -> tuple[list[dict], list[dict], dict]:
    """-> (new_train_rows, test_rows, report). Does not touch dev."""
    train_rows = load_rows(train_path)
    dev_rows = load_rows(dev_path)
    knesset_rows = load_rows(knesset_reviewed_path)

    dev_types = {r["acronym"] for r in dev_rows}
    train_types = {r["acronym"] for r in train_rows}

    dropped_dev_overlap = [r for r in knesset_rows if r["acronym"] in dev_types]
    eligible = [r for r in knesset_rows if r["acronym"] not in dev_types]

    test_types = select_test_types(eligible, fraction=test_fraction, seed=seed)

    # Every reviewed Knesset type is already somewhere in train (checked:
    # zero types are free of the existing 549-type inventory), so a test
    # type must be actively carved OUT of train's existing rows — adding
    # Knesset rows for it elsewhere while leaving its Wikipedia rows in train
    # would not make it unseen at all.
    train_rows = [r for r in train_rows if r["acronym"] not in test_types]
    removed_from_train = [r for r in load_rows(train_path) if r["acronym"] in test_types]

    test_rows_raw = [r for r in eligible if r["acronym"] in test_types]
    train_candidates_raw = [r for r in eligible if r["acronym"] not in test_types]

    # Cap train's contribution per type, taking a stable (seeded) sample
    # rather than always the first N — avoids always favouring whichever
    # sentence happened to be mined first.
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in train_candidates_raw:
        by_type[r["acronym"]].append(r)
    train_from_knesset: list[dict] = []
    capped_types = []
    for acronym, rows in by_type.items():
        if len(rows) > train_cap_per_type:
            capped_types.append((acronym, len(rows)))
            rows = rows[:]
            rng.shuffle(rows)
            rows = rows[:train_cap_per_type]
        train_from_knesset.extend(rows)

    # Assign item_ids that won't collide with existing train/dev ids.
    def next_ids(prefix: str, n: int) -> list[str]:
        return [f"{prefix}-{i:04d}" for i in range(n)]

    test_ids = next_ids("test-kn", len(test_rows_raw))
    train_ids = next_ids("train-kn", len(train_from_knesset))

    test_rows = [
        knesset_row_to_split_schema(r, tid) for r, tid in zip(test_rows_raw, test_ids)
    ]
    new_train_from_knesset = [
        knesset_row_to_split_schema(r, tid) for r, tid in zip(train_from_knesset, train_ids)
    ]
    new_train_rows = train_rows + new_train_from_knesset

    # --- gap report -----------------------------------------------------
    test_by_type: dict[str, list[dict]] = defaultdict(list)
    for r in test_rows_raw:
        test_by_type[r["acronym"]].append(r)
    thin_test_types = sorted(
        (a, len(rows)) for a, rows in test_by_type.items() if len(rows) < 3
    )
    # A type whose reviewed rows are all the SAME gold sense gives no real
    # disambiguation signal in test even with several rows — flag it.
    single_sense_test_types = sorted(
        a for a, rows in test_by_type.items()
        if len({r["gold_expansion"] for r in rows}) == 1
    )

    report = {
        "train_types_before": len(train_types),
        "removed_from_train_rows": len(removed_from_train),
        "removed_from_train_types": sorted(test_types),
        "dev_types": len(dev_types),
        "knesset_reviewed_total": len(knesset_rows),
        "dropped_dev_overlap_rows": len(dropped_dev_overlap),
        "dropped_dev_overlap_types": sorted({r["acronym"] for r in dropped_dev_overlap}),
        "eligible_rows": len(eligible),
        "eligible_types": len(by_type) + len(test_types),
        "test_types": len(test_types),
        "test_rows": len(test_rows),
        "train_from_knesset_rows": len(new_train_from_knesset),
        "train_from_knesset_types": len(by_type),
        "capped_types": capped_types,
        "thin_test_types": thin_test_types,
        "single_sense_test_types": single_sense_test_types,
    }
    return new_train_rows, test_rows, report


def print_report(report: dict) -> None:
    print(f"train types before: {report['train_types_before']}")
    print(
        f"removed from train to make test genuinely unseen: "
        f"{report['removed_from_train_rows']} rows, {len(report['removed_from_train_types'])} types"
    )
    print(f"  {report['removed_from_train_types']}")
    print(f"dev types (frozen, excluded from allocation): {report['dev_types']}")
    print(f"knesset reviewed rows: {report['knesset_reviewed_total']}")
    print(
        f"dropped (type already in dev): {report['dropped_dev_overlap_rows']} rows, "
        f"{len(report['dropped_dev_overlap_types'])} types"
    )
    if report["dropped_dev_overlap_types"]:
        print(f"  {report['dropped_dev_overlap_types']}")
    print(f"eligible for train/test: {report['eligible_rows']} rows, {report['eligible_types']} types")
    print(f"-> test: {report['test_rows']} rows, {report['test_types']} types")
    print(
        f"-> train (+knesset): {report['train_from_knesset_rows']} rows, "
        f"{report['train_from_knesset_types']} types"
    )
    if report["capped_types"]:
        print(f"\ncapped in train (had more than the per-type cap):")
        for acronym, n in sorted(report["capped_types"], key=lambda x: -x[1]):
            print(f"  {acronym}: {n} reviewed rows, capped")
    if report["thin_test_types"]:
        print(f"\nTHIN test types (<3 rows) — consider authoring more examples by hand:")
        for acronym, n in report["thin_test_types"]:
            print(f"  {acronym}: {n} row(s)")
    if report["single_sense_test_types"]:
        print(
            f"\nSINGLE-SENSE test types (all reviewed rows share one gold answer, "
            f"no disambiguation signal even with several rows):"
        )
        for acronym in report["single_sense_test_types"]:
            print(f"  {acronym}")


def write_csv(path: str, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=TRAIN_DEV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="data/splits/train_items.csv")
    ap.add_argument("--dev", default="data/splits/dev_items.csv")
    ap.add_argument("--knesset-reviewed", default="data/mined/knesset/knesset_reviewed.csv")
    ap.add_argument("--test-out", default="data/splits/test_items.csv")
    ap.add_argument("--train-out", default="data/splits/train_items.csv")
    ap.add_argument("--test-fraction", type=float, default=0.2)
    ap.add_argument("--train-cap-per-type", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="print the report only, write nothing")
    a = ap.parse_args()

    new_train_rows, test_rows, report = build(
        a.train, a.dev, a.knesset_reviewed,
        test_fraction=a.test_fraction, train_cap_per_type=a.train_cap_per_type, seed=a.seed,
    )
    print_report(report)

    if a.dry_run:
        print("\n(dry run — nothing written)")
        return

    write_csv(a.test_out, test_rows)
    write_csv(a.train_out, new_train_rows)
    print(f"\nwrote {len(test_rows)} rows -> {a.test_out}")
    print(f"wrote {len(new_train_rows)} rows -> {a.train_out}")


if __name__ == "__main__":
    main()
