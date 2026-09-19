"""Shared candidate rows, CSV persistence and threshold summaries."""
from __future__ import annotations

import csv
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from . import hebrew_text

# Preserve the logger used by the original table helpers.
LOG = logging.getLogger("data_preprocess.wikipedia.source")


@dataclass
class BulletRow:
    """One candidate expansion from one source.

    Shared by the Wikipedia and Wiktionary counters and by the merged table, so
    all three CSVs have identical columns.
    """

    acronym: str
    page_title: str
    expansion: str
    hits: int
    script: str  # "hebrew" | "latin" | "other"
    initials_match: bool | None  # None when the acronym is not Hebrew
    looks_like_person: bool
    raw_line: str
    source: str = "wikipedia"  # "wikipedia" | "wiktionary"
    domain: str = ""  # Wiktionary register label; empty for Wikipedia


def acronym_script(acronym: str) -> str:
    """Which alphabet the acronym itself is written in.

    The category is sorted alphabetically and opens with Latin-script entries
    (AI, ATM, BPM). Those have Hebrew expansions but a Latin acronym, so the
    Hebrew initials check does not apply to them.
    """
    if any(c in hebrew_text.HEBREW_LETTERS for c in acronym):
        return "hebrew"
    if any("A" <= c.upper() <= "Z" for c in acronym):
        return "latin"
    return "other"


FIELDS = [
    "acronym", "page_title", "expansion", "hits", "script",
    "initials_match", "looks_like_person", "source", "domain", "raw_line",
]


def read_existing(path: str | Path) -> tuple[list[BulletRow], set[str], dict[str, int]]:
    """Load a partial CSV so a killed run can resume from where it stopped."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return [], set(), {}
    rows: list[BulletRow] = []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for rec in csv.DictReader(fh):
            match = rec.get("initials_match") or ""
            rows.append(
                BulletRow(
                    acronym=rec["acronym"],
                    page_title=rec["page_title"],
                    expansion=rec["expansion"],
                    hits=int(rec["hits"]),
                    script=rec["script"],
                    initials_match={"True": True, "False": False}.get(match),
                    looks_like_person=rec["looks_like_person"] == "True",
                    raw_line=rec["raw_line"],
                    source=rec.get("source") or "wikipedia",
                    domain=rec.get("domain") or "",
                )
            )
    done = {r.page_title for r in rows}
    cache = {r.expansion: r.hits for r in rows if r.hits >= 0}
    LOG.info("resuming: %d bullets across %d pages already counted", len(rows), len(done))
    return rows, done, cache


def write_csv(
    path: str | Path, rows: Iterable[BulletRow], *, existing: list[BulletRow] | None = None
) -> list[BulletRow]:
    """Stream rows to CSV, flushing each one.

    Flushing per row matters: the run makes hundreds of throttled API calls and
    can be interrupted (rate limiting, a killed shell). Whatever was counted
    before the interruption stays on disk and is not repeated.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    collected: list[BulletRow] = list(existing or [])
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in collected:
            writer.writerow(asdict(row))
        fh.flush()
        for row in rows:
            writer.writerow(asdict(row))
            fh.flush()
            collected.append(row)
    LOG.info("wrote %d bullets -> %s", len(collected), path)
    return collected


def summarise(rows: list[BulletRow], thresholds: tuple[int, ...] = (10, 50, 100, 500)) -> dict:
    """How many acronyms would survive at each candidate threshold."""
    by_acronym: dict[str, list[BulletRow]] = {}
    for row in rows:
        by_acronym.setdefault(row.acronym, []).append(row)
    summary = {
        "n_pages": len(by_acronym),
        "n_bullets": len(rows),
        "n_person_flagged": sum(1 for r in rows if r.looks_like_person),
        "n_hebrew_acronyms": len({r.acronym for r in rows if r.script == "hebrew"}),
        "n_initials_mismatch": sum(1 for r in rows if r.initials_match is False),
        "thresholds": {},
    }
    for t in thresholds:
        survivors = {
            acr: [r.expansion for r in group if r.hits >= t] for acr, group in by_acronym.items()
        }
        summary["thresholds"][str(t)] = {
            "bullets_at_or_above": sum(len(v) for v in survivors.values()),
            "acronyms_with_2plus_senses": sum(1 for v in survivors.values() if len(v) >= 2),
        }
    return summary
