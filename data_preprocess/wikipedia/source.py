"""Count how often every disambiguation bullet occurs in Hebrew Wikipedia.

This is the first exploration task, not dataset construction: for each page in
`קטגוריה:פירושון ראשי תיבות`, list every bullet (its candidate expansion) and
attach the number of articles that contain that exact phrase. Nothing is
filtered out here — person senses and initials mismatches are kept and merely
flagged, so the team can see the whole distribution before choosing thresholds.

Outputs a CSV (one row per bullet) and a JSON summary.
"""
from __future__ import annotations

import logging
from typing import Iterator

from ..common import hebrew_text
from ..common.candidates import BulletRow, acronym_script
from .parser import (
    ACRONYM_DISAMBIG_CATEGORY,
    candidate_from_line,
    clean_line,
    iter_sense_bullets,
    looks_like_person,
    page_acronym,
)
from .client import WikiAPI

LOG = logging.getLogger(__name__)


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
