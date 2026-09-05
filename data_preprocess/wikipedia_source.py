"""Count how often every disambiguation bullet occurs in Hebrew Wikipedia.

This is the first exploration task, not dataset construction: for each page in
`קטגוריה:פירושון ראשי תיבות`, list every bullet (its candidate expansion) and
attach the number of articles that contain that exact phrase. Nothing is
filtered out here — person senses and initials mismatches are kept and merely
flagged, so the team can see the whole distribution before choosing thresholds.

Outputs a CSV (one row per bullet) and a JSON summary.
"""
from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

from . import hebrew_text
from .wikipedia_parser import (
    ACRONYM_DISAMBIG_CATEGORY,
    candidate_from_line,
    clean_line,
    iter_sense_bullets,
    looks_like_person,
    page_acronym,
)
from .wiki_client import WikiAPI

LOG = logging.getLogger(__name__)


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


def bullets(wikitext: str, acronym: str) -> Iterator[tuple[str, str]]:
    """Yield `(expansion, cleaned_line)` for every bullet on the page."""
    seen: set[str] = set()
    for line in iter_sense_bullets(wikitext):
        candidate = candidate_from_line(line, acronym)
        if not candidate:
            continue
        key = hebrew_text.strip_niqqud(candidate)
        if key in seen:
            continue
        seen.add(key)
        yield candidate, clean_line(line)


def count_bullets(
    api: WikiAPI,
    category: str = ACRONYM_DISAMBIG_CATEGORY,
    *,
    limit: int | None = None,
    cache: dict[str, int] | None = None,
    skip_titles: set[str] | None = None,
) -> Iterator[BulletRow]:
    """Yield one `BulletRow` per bullet across the whole category.

    `skip_titles` lets an interrupted run resume: pages already present in the
    output CSV are not fetched or counted again. Wikipedia rate-limits this
    workload, so not repaying that cost after an interruption matters.
    """
    cache = {} if cache is None else cache
    skip_titles = skip_titles or set()
    pages = 0
    for page in api.category_members(category):
        if limit is not None and pages >= limit:
            return
        pages += 1
        title = page["title"]
        if title in skip_titles:
            LOG.debug("resume: skipping %s", title)
            continue
        acronym = page_acronym(title)
        try:
            text = api.wikitext(title)
        except RuntimeError as exc:
            LOG.warning("skipping %s: %s", title, exc)
            continue
        script = acronym_script(acronym)
        for expansion, cleaned in bullets(text, acronym):
            if expansion not in cache:
                try:
                    cache[expansion] = api.search_hits(expansion)
                except RuntimeError as exc:
                    LOG.warning("hit count failed for %r: %s", expansion, exc)
                    cache[expansion] = -1
            yield BulletRow(
                acronym=acronym,
                page_title=title,
                expansion=expansion,
                hits=cache[expansion],
                script=script,
                initials_match=(
                    hebrew_text.matches_acronym(expansion, acronym) if script == "hebrew" else None
                ),
                looks_like_person=looks_like_person(cleaned),
                raw_line=cleaned,
            )
        if pages % 25 == 0:
            LOG.info("processed %d pages", pages)


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
