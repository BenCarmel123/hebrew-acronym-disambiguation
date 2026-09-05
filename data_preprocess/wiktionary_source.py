"""Count Wikipedia hits for every Hebrew Wiktionary acronym sense.

Same output shape as `count_bullets`, different source. Wiktionary states
*literal expansions* where Wikipedia disambiguation pages mostly list referents
(`research/source_evidence/REPORT.md`), so for a project whose prediction target
is the literal expansion string this is the higher-value inventory: 687 Hebrew
polysemous types against Wikipedia's ~80.

Note the two wikis play different roles here: senses come from Wiktionary, but
`hits` is always measured against Hebrew *Wikipedia*, because that is the corpus
contexts will be drawn from.
"""
from __future__ import annotations

import logging
from typing import Iterable, Iterator

from . import hebrew_text
from .wikipedia_source import BulletRow, acronym_script
from .wikipedia_parser import looks_like_person
from .wiki_client import WikiAPI
from .wiktionary_parser import ACRONYM_CATEGORY, WIKTIONARY_API, parse_entry

LOG = logging.getLogger(__name__)


def fetch_entries(
    wikt: WikiAPI,
    category: str = ACRONYM_CATEGORY,
    *,
    limit: int | None = None,
    batch_size: int = 50,
) -> dict[str, str]:
    """Wikitext for every page in the Wiktionary acronym category."""
    titles = [p["title"] for p in wikt.category_members(category)]
    if limit is not None:
        titles = titles[:limit]
    LOG.info("wiktionary category: %d pages", len(titles))
    texts: dict[str, str] = {}
    for i in range(0, len(titles), batch_size):
        texts.update(wikt.wikitext_batch(titles[i : i + batch_size], batch_size))
        if (i // batch_size) % 10 == 0:
            LOG.info("fetched %d/%d entries", len(texts), len(titles))
    return texts


def count_senses(
    api: WikiAPI,
    texts: dict[str, str],
    *,
    min_senses: int = 1,
    cache: dict[str, int] | None = None,
    skip_titles: set[str] | None = None,
) -> Iterator[BulletRow]:
    """Yield a `BulletRow` per Wiktionary sense, with its Wikipedia hit count.

    `api` must point at Hebrew Wikipedia — hits are counted in the corpus, not
    in Wiktionary.
    """
    cache = {} if cache is None else cache
    skip_titles = skip_titles or set()
    done = 0
    for title, wikitext in texts.items():
        if title in skip_titles:
            continue
        acronym = hebrew_text.normalize_acronym(title)
        senses = parse_entry(wikitext, title)
        if len(senses) < min_senses:
            continue
        script = acronym_script(acronym)
        for sense in senses:
            phrase = sense["expansion"]
            if phrase not in cache:
                try:
                    cache[phrase] = api.search_hits(phrase)
                except RuntimeError as exc:
                    LOG.warning("hit count failed for %r: %s", phrase, exc)
                    cache[phrase] = -1
            yield BulletRow(
                acronym=acronym,
                page_title=title,
                expansion=phrase,
                hits=cache[phrase],
                script=script,
                initials_match=sense["initials_match"] if script == "hebrew" else None,
                looks_like_person=looks_like_person(sense["raw_line"]),
                raw_line=sense["raw_line"],
                source="wiktionary",
                domain=sense["domain"],
            )
        done += 1
        if done % 100 == 0:
            LOG.info("counted %d entries", done)
