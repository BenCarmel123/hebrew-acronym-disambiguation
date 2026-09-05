"""Merge the per-source candidate tables into one view.

Wikipedia and Wiktionary overlap only partially — 50 shared types out of 107 and
687 — so the union is strictly larger than either. Rows are kept at their
original grain (one row per source x expansion); a `sources` column records
where each expansion was attested, which is the cross-source agreement signal
the source probe measured.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import replace
from typing import Iterable, Sequence

from . import hebrew_text
from .wikipedia_source import BulletRow

LOG = logging.getLogger(__name__)


def _key(row: BulletRow) -> tuple[str, str]:
    """Identity of a candidate: normalised acronym + normalised expansion."""
    return (
        hebrew_text.normalize_acronym(row.acronym),
        hebrew_text.strip_niqqud(row.expansion).strip(),
    )


def merge_rows(*tables: Sequence[BulletRow]) -> list[BulletRow]:
    """Union the tables, deduplicating identical (acronym, expansion) pairs.

    When both sources carry the same expansion the row is kept once, its
    `source` becomes "wikipedia+wiktionary", and the richer metadata wins: a
    Wiktionary domain label is preserved, and the larger hit count is taken
    (they should agree, but a failed lookup is recorded as -1).
    """
    merged: dict[tuple[str, str], BulletRow] = {}
    seen_sources: dict[tuple[str, str], set[str]] = defaultdict(set)

    for table in tables:
        for row in table:
            key = _key(row)
            seen_sources[key].add(row.source)
            current = merged.get(key)
            if current is None:
                merged[key] = row
                continue
            merged[key] = replace(
                current,
                hits=max(current.hits, row.hits),
                domain=current.domain or row.domain,
                # Prefer a Wiktionary raw line: it is the literal sense text.
                raw_line=row.raw_line if row.source == "wiktionary" else current.raw_line,
            )

    out = []
    for key, row in merged.items():
        out.append(replace(row, source="+".join(sorted(seen_sources[key]))))
    out.sort(key=lambda r: (r.acronym, -r.hits, r.expansion))
    LOG.info("merged %d unique acronym/expansion pairs", len(out))
    return out


def merge_summary(rows: Sequence[BulletRow], thresholds: tuple[int, ...] = (10, 50, 100, 500)) -> dict:
    by_acronym: dict[str, list[BulletRow]] = defaultdict(list)
    for row in rows:
        by_acronym[row.acronym].append(row)

    both = [r for r in rows if "+" in r.source]
    summary = {
        "n_rows": len(rows),
        "n_acronyms": len(by_acronym),
        "n_attested_by_both_sources": len(both),
        "rows_by_source": {
            src: sum(1 for r in rows if r.source == src)
            for src in sorted({r.source for r in rows})
        },
        "thresholds": {},
    }
    for t in thresholds:
        kept = {a: [r for r in g if r.hits >= t] for a, g in by_acronym.items()}
        hebrew_types = {
            a for a, g in kept.items()
            if len(g) >= 2 and any(r.script == "hebrew" for r in g)
        }
        summary["thresholds"][str(t)] = {
            "rows_at_or_above": sum(len(v) for v in kept.values()),
            "acronyms_with_2plus_senses": sum(1 for v in kept.values() if len(v) >= 2),
            "hebrew_acronyms_with_2plus_senses": len(hebrew_types),
        }
    return summary
