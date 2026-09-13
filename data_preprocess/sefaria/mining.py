"""Mine natural acronym usage from Sefaria's rabbinic/Talmudic text corpus.

Unlike Wikipedia prose, rabbinic/Talmudic text routinely disambiguates a
name-acronym (מהר"ש, רמב"ם) through the surrounding halachic discussion
itself — exactly the context Wikipedia's `unsure` rabbinic-name rows lack (see
`data/splits/README.md`).

Known limitation (see conversation history / commit notes): real yield here
has tested low — many search hits do not resolve to a clean single sentence
once the full ref text is fetched, because Sefaria's `he` segments are not
reliably sentence-granular the way a Wikipedia plaintext extract is. Treat
this source as exploratory pending further tuning, not a solved pipeline like
`wikipedia.mining` or `knesset_source`.
"""
from __future__ import annotations

import logging
import re
from typing import Iterable, Iterator

from ..common.filters import ContextRow, is_clean_sentence_multi_acronym, page_sentences
from .client import SefariaAPI

LOG = logging.getLogger(__name__)

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def mine_sefaria(
    api: SefariaAPI,
    acronyms: Iterable[str],
    *,
    hits_per_acronym: int = 40,
    max_per_acronym: int = 15,
) -> Iterator[ContextRow]:
    """Yield clean prose sentences mentioning each acronym, from Sefaria.

    Flat, sense-blind mining like `wikipedia.mining.mine_sentences`: returns
    whichever sentences mention the acronym, with no sense attached — a human
    labels the sense afterward, from the candidate list.

    Search hits are ref-level (a book section), and Sefaria's search snippets
    are clause-fragmented — the same reason `wikipedia.mining`'s docstring
    gives for not mining Wikipedia search snippets directly. So each hit
    ref's full text is fetched and split into sentences, same as a Wikipedia
    page. The relaxed `is_clean_sentence_multi_acronym` check is used instead
    of the Wikipedia-tuned default, since rabbinic prose is acronym-dense
    (a sentence naming מהר"ש will often also carry ז"ל, רשב"י, מהרח"ו).
    """
    for acronym in acronyms:
        found = 0
        try:
            hits = api.search(acronym, size=hits_per_acronym)
        except RuntimeError as exc:
            LOG.warning("Sefaria search failed for %r: %s", acronym, exc)
            continue
        seen_refs: set[str] = set()
        for hit in hits:
            if found >= max_per_acronym:
                break
            ref = hit["ref"]
            if ref in seen_refs:
                continue
            seen_refs.add(ref)
            try:
                segments = api.text(ref)
            except RuntimeError as exc:
                LOG.warning("Sefaria text fetch failed for %r: %s", ref, exc)
                continue
            for segment in segments:
                plain = _HTML_TAG_RE.sub("", segment)
                for sentence in page_sentences(
                    plain, acronym, is_clean=is_clean_sentence_multi_acronym
                ):
                    yield ContextRow(acronym, sentence, "sefaria", ref)
                    found += 1
                    if found >= max_per_acronym:
                        break
                if found >= max_per_acronym:
                    break
        LOG.info("%s: %d sentences from %d Sefaria refs", acronym, found, len(seen_refs))
