"""Apply a human item review (verdict + correction) to a split CSV.

Reads corrections directly from the review CSV instead of a hardcoded mapping,
so the result is reproducible from data alone. Three kinds of change, all
recorded in new columns rather than silently:

  - relabel: verdict is wrong_sense/broken AND note holds the corrected
             expansion verbatim (must match a value already appearing in that
             row's `candidates` or another row of the same acronym type, so it
             is a real, attested sense — not free text)
  - drop:    verdict is wrong_sense/broken AND note is empty; no correct
             answer is recoverable
  - flag:    verdict is unsure; row is KEPT, flagged for later

Usage:
    python -m data_preprocess.apply_dev_review data/mined/dev_review.csv \\
        data/splits/dev_items.csv data/splits/dev_items.csv
"""
from __future__ import annotations

import argparse
import csv

DAMAGED_VERDICTS = {"wrong_sense", "broken"}


def apply_review(items_rows: list[dict], review_rows: list[dict]) -> tuple[list[dict], dict]:
    """-> (output rows, stats dict). `items_rows` is mutated in place."""
    review = {r["item_id"]: r for r in review_rows}
    fields = list(items_rows[0].keys())
    for c in ("review_verdict", "review_note", "label_origin"):
        if c not in fields:
            fields.append(c)

    # Corrections found this pass, keyed by acronym type: a corrected
    # expansion becomes a candidate for EVERY row of that type, not only the
    # rows it corrects. Otherwise the new option would appear exactly where it
    # is the answer, which leaks which rows were reviewed; as a type-wide
    # candidate it is a genuine distractor on the rows where it is wrong.
    corrections_by_type: dict[str, set[str]] = {}
    for r in items_rows:
        v = review.get(r["item_id"], {})
        note = (v.get("note") or "").strip()
        if v.get("verdict") in DAMAGED_VERDICTS and note:
            corrections_by_type.setdefault(r["acronym"], set()).add(note)

    out, dropped, relabelled, flagged, widened = [], [], [], [], 0
    for r in items_rows:
        item_id = r["item_id"]
        v = review.get(item_id, {})
        verdict = v.get("verdict", "unreviewed")
        note = (v.get("note") or "").strip()
        r["review_verdict"] = verdict
        r["review_note"] = note
        r["label_origin"] = r.get("label_origin") or "substitution"

        if verdict in DAMAGED_VERDICTS and not note:
            dropped.append((item_id, r["acronym"], verdict))
            continue

        extra = corrections_by_type.get(r["acronym"], set())
        if extra:
            cands = [c.strip() for c in r["candidates"].split("|") if c.strip()]
            added = [e for e in sorted(extra) if e not in cands]
            if added:
                cands.extend(added)
                r["candidates"] = " | ".join(cands)
                r["n_candidates"] = str(len(cands))
                widened += 1

        if verdict in DAMAGED_VERDICTS and note:
            old = r["gold_expansion"]
            r["gold_expansion"] = note
            r["provisional_expansion"] = note
            r["label_status"] = "verified"
            r["label_origin"] = "human_review"
            relabelled.append((item_id, r["acronym"], old, note))

        if verdict == "unsure":
            flagged.append(item_id)
            if not r["review_note"]:
                r["review_note"] = "reviewer unsure: sense not clearly determined by the sentence"

        out.append(r)

    stats = {
        "n_in": len(items_rows),
        "n_out": len(out),
        "dropped": dropped,
        "relabelled": relabelled,
        "flagged": flagged,
        "widened": widened,
    }
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("review", help="review CSV: item_id,acronym,expansion,verdict,note")
    ap.add_argument("items", help="split CSV to apply the review to (e.g. dev_items.csv)")
    ap.add_argument("out", help="output path (may be the same as `items`)")
    args = ap.parse_args()

    with open(args.review, encoding="utf-8-sig", newline="") as fh:
        review_rows = list(csv.DictReader(fh))
    with open(args.items, encoding="utf-8-sig", newline="") as fh:
        items_rows = list(csv.DictReader(fh))

    out, stats = apply_review(items_rows, review_rows)
    fields = list(items_rows[0].keys())
    for c in ("review_verdict", "review_note", "label_origin"):
        if c not in fields:
            fields.append(c)

    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out)

    print(f"in {stats['n_in']} -> out {stats['n_out']}")
    print(f"\nrelabelled {len(stats['relabelled'])}:")
    for i, a, o, n in stats["relabelled"]:
        print(f"  {i} {a}: {o} -> {n}")
    print(f"\ndropped {len(stats['dropped'])}:")
    for i, a, verdict in stats["dropped"]:
        print(f"  {i} {a}: {verdict}, no correction note")
    print(f"\ncandidate list widened on {stats['widened']} rows (type-wide)")
    print(f"flagged unsure (kept): {len(stats['flagged'])}")


if __name__ == "__main__":
    main()
