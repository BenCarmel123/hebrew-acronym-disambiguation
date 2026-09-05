"""Join the candidate table with mined contexts into an annotation-ready CSV.

`candidate_table.csv` (acronym -> candidate expansions) and `mined_contexts.csv`
(acronym -> real usage sentences) are the two things collected so far; neither
is training data on its own. This join produces one row per mined sentence,
carrying that acronym's full candidate list alongside it, plus an empty
`gold_expansion` column for a human annotator to fill in — the last step
before this becomes `data/splits/` material (see `data/README.md`).
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterable

LOG = logging.getLogger(__name__)

FIELDS = [
    "acronym",
    "context",
    "candidates",
    "n_candidates",
    "gold_expansion",
    "context_source",
    "page_title",
]


def load_candidates(path: str | Path) -> dict[str, list[str]]:
    """acronym -> its candidate expansions, in the table's existing rank order."""
    by_acronym: dict[str, list[str]] = {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for rec in csv.DictReader(fh):
            by_acronym.setdefault(rec["acronym"], []).append(rec["expansion"])
    return by_acronym


def build_rows(
    candidates: dict[str, list[str]], contexts: Iterable[dict]
) -> list[dict]:
    """One row per mined context, annotated with its acronym's candidate list.

    A mined context whose acronym is absent from the candidate table (out of
    scope, or dropped after the table was built) is skipped rather than
    emitted with an empty candidate list.
    """
    rows: list[dict] = []
    skipped = 0
    for ctx in contexts:
        acronym = ctx["acronym"]
        expansions = candidates.get(acronym)
        if not expansions:
            skipped += 1
            continue
        rows.append(
            {
                "acronym": acronym,
                "context": ctx["context"],
                "candidates": "|".join(expansions),
                "n_candidates": len(expansions),
                "gold_expansion": "",
                "context_source": ctx["source"],
                "page_title": ctx["page_title"],
            }
        )
    if skipped:
        LOG.warning("skipped %d contexts whose acronym is not in the candidate table", skipped)
    return rows


def write_annotation_table(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    LOG.info("wrote %d rows -> %s", len(rows), path)
