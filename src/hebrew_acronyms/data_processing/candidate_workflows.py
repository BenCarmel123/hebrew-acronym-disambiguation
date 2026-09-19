"""Collect, merge and review candidate tables using explicit input/output paths."""
from __future__ import annotations

import logging

from .common.candidates import read_existing, summarise, write_csv
from .common.reporting import write_summary
from .wikipedia.source import count_bullets
from .wiktionary.source import count_senses, fetch_entries
from .wikipedia.client import API_URL, WikiAPI
from .wiktionary.parser import WIKTIONARY_API
from .merge_sources import merge_rows, merge_summary
from .dedupe_expansions import (
    apply_review_decisions, dedupe_expansions, drop_single_letter_acronyms,
    find_near_duplicates, write_review_csv,
)

LOG = logging.getLogger("hebrew_acronyms.data_processing")


def collect_wikipedia(
    *,
    out: str,
    category: str,
    limit: int | None,
    delay: float,
    restart: bool,
    summary: str | None,
) -> None:
    """Count Wikipedia candidates, preserving existing rows unless restarted."""
    api = WikiAPI(delay=delay)
    existing, done, cache = ([], set(), {}) if restart else read_existing(out)
    rows = write_csv(
        out,
        count_bullets(api, category, limit=limit, cache=cache, skip_titles=done),
        existing=existing,
    )
    write_summary(out, summarise(rows), summary)


def collect_wiktionary(
    *,
    out: str,
    category: str,
    limit: int | None,
    min_senses: int,
    delay: float,
    restart: bool,
    summary: str | None,
) -> None:
    """Count Wiktionary senses with the existing Wikipedia hit cache."""
    wikt = WikiAPI(api_url=WIKTIONARY_API, delay=delay)
    wiki = WikiAPI(api_url=API_URL, delay=delay)
    existing, done, cache = ([], set(), {}) if restart else read_existing(out)
    texts = fetch_entries(wikt, category, limit=limit)
    rows = write_csv(
        out,
        count_senses(
            wiki, texts, min_senses=min_senses, cache=cache, skip_titles=done
        ),
        existing=existing,
    )
    write_summary(out, summarise(rows), summary)


def merge_tables(
    *,
    wikipedia_path: str,
    wiktionary_path: str,
    out: str,
    summary: str | None,
) -> None:
    """Merge inventories, apply existing deduplication and write their summary."""
    wikipedia, _, _ = read_existing(wikipedia_path)
    wiktionary, _, _ = read_existing(wiktionary_path)
    if not wikipedia and not wiktionary:
        LOG.error("both input tables are empty; nothing to merge")
        return
    LOG.info("merging %d Wikipedia + %d Wiktionary rows", len(wikipedia), len(wiktionary))
    rows = merge_rows(wikipedia, wiktionary)
    rows = dedupe_expansions(rows)
    rows = drop_single_letter_acronyms(rows)
    write_csv(out, iter(()), existing=rows)
    write_summary(out, merge_summary(rows), summary)


def flag_duplicates(
    *,
    inp: str,
    out: str,
    min_ratio: float,
) -> None:
    """Write the existing near-duplicate review table."""
    rows, _, _ = read_existing(inp)
    flagged = find_near_duplicates(rows, min_ratio=min_ratio)
    write_review_csv(out, flagged)


def apply_review(
    *,
    inp: str,
    review: str,
    out: str,
) -> None:
    """Apply recorded review decisions and write the remaining candidates."""
    rows, _, _ = read_existing(inp)
    cleaned = apply_review_decisions(rows, review)
    write_csv(out, iter(()), existing=cleaned)
