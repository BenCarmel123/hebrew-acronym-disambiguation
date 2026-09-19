"""Mine clean prose sentences for acronyms, from Wikipedia article plain text.

An earlier approach mined `list=search` snippets, which are fixed-width
excerpts cut from anywhere on a page — reference lists, tables, image captions,
infoboxes. No amount of post-cleaning made those reliably read as prose (~32%
usable after ten filters), so that module was removed and this one takes a
different route: use search only to *find* pages, then fetch each
page's rendered plain text (`prop=extracts&explaintext`), split it into
sentences, and keep only sentences that read like natural usage
(`common.filters.is_clean_sentence`).

Four mining strategies live here, in increasing order of how much they
intervene in the text:

`mine_sentences`
    Searches the acronym surface form and keeps clean sentences. Produces
    fully natural text, but returns whichever sense dominates the corpus —
    every `מ"מ` hit comes back as מילימטר (rainfall), and rarer senses never
    appear. It also records no sense at all, so a human labels from scratch.

`mine_by_expansion`
    Searches for pages carrying the acronym *and* a given expansion, so each
    sense gets its own pool of candidate pages. Still fully natural text, and
    it attaches a provisional sense. Yield is low: for `מ"מ`, 2 of 17
    expansions produced anything, because most senses are simply not written
    with the abbreviation in this corpus.

`mine_substituted`
    Finds sentences with the expansion *spelled out* and rewrites it to the
    acronym. Yield is far higher — 14 of 17 expansions for `מ"מ` — because a
    full form is ordinary Hebrew with none of the acronym's search problems,
    and the sense is known by construction.

    The cost is naturalness. A writer who spelled a phrase out might not have
    abbreviated it in that position, so substituted items can read stiffly or
    carry thinner context than real abbreviated usage. Two failure modes recur
    and are worth knowing when reading output:

    - Expansions that are *common word sequences* rather than fixed terms
      (מכל מקום, מראה מקום) match text where the words were not functioning as
      that unit — `מכל מקום בו הרכב נמצא` is compositional "from any place
      where", and rewriting it produces nonsense.
    - Expansions that are *fragments of longer terms* (מרחב מכפלה, part of
      מרחב מכפלה פנימית) rewrite the fragment and leave the remainder dangling.

    Fixed terminology (מפקד מחלקה, מכונאי מוטס, ממלא מקום) substitutes cleanly.
    Rows are stamped `provenance="substituted"` so they stay separable from
    mined text through annotation and analysis — keep that distinction, since
    it is what allows checking later whether models behave differently on
    synthetic items than on natural ones.

`mine_deglossed`
    Finds sentences that use an acronym and gloss it inline
    (`בא"ח (בסיס אימונים חטיבתי)`), then strips the gloss. `is_clean_sentence`
    drops these upstream because the answer would sit in the input, but a
    glossed sentence is the one place the corpus proves a sense is real *and*
    shows a writer actually abbreviating it — strictly more natural than
    substitution, which invents the abbreviation. See its docstring below for
    the residual proper-name defect this catches.

Every strategy applies the same output bar: `is_clean_sentence` runs on the
final text, so a substituted or deglossed sentence that ends up with two
acronyms or an inline gloss is dropped exactly as a mined one would be.
"""
from __future__ import annotations

import logging
import re
from typing import Iterable, Iterator

from ..common import hebrew_text
from ..common.filters import (
    HEADING_RE,
    HEBREW_NUMERAL_REF_RE,
    MAX_LEN,
    MIN_LEN,
    NON_PROSE_RE,
    SENTENCE_SPLIT_RE,
    ContextRow,
    SenseRow,
    is_clean_sentence,
    mentions_acronym,
    mentions_expansion,
    page_sentences,
)
from .client import WikiAPI

LOG = logging.getLogger(__name__)


def mine_sentences(
    api: WikiAPI,
    acronyms: Iterable[str],
    *,
    pages_per_acronym: int = 30,
    max_per_acronym: int = 12,
) -> Iterator[ContextRow]:
    """Yield clean prose sentences for each acronym.

    Search finds candidate pages; each page is then fetched as plain text,
    because the search snippet itself is not prose. `max_per_acronym` stops
    fetching further pages for an acronym once enough sentences are found,
    which keeps the request count proportional to what is actually needed.
    """
    for acronym in acronyms:
        found = 0
        titles: list[str] = []
        for surface in hebrew_text.variants(acronym) or [acronym]:
            try:
                titles.extend(h["title"] for h in api.search_snippets(surface, limit=pages_per_acronym))
            except RuntimeError as exc:
                LOG.warning("search failed for %r: %s", surface, exc)
        seen_titles: set[str] = set()
        for title in titles:
            if found >= max_per_acronym:
                break
            if title in seen_titles:
                continue
            seen_titles.add(title)
            try:
                data = api.get(action="query", prop="extracts", explaintext=1, titles=title)
            except RuntimeError as exc:
                LOG.warning("extract failed for %r: %s", title, exc)
                continue
            for page in data.get("query", {}).get("pages", []):
                for sentence in page_sentences(page.get("extract") or "", acronym):
                    yield ContextRow(acronym, sentence, "wikipedia", title)
                    found += 1
                    if found >= max_per_acronym:
                        break
        LOG.info("%s: %d sentences from %d pages", acronym, found, len(seen_titles))


def mine_by_expansion(
    api: WikiAPI,
    acronym: str,
    expansions: Iterable[str],
    *,
    pages_per_expansion: int = 20,
    max_per_expansion: int = 5,
) -> Iterator[SenseRow]:
    """Mine sentences for each expansion separately, from its own page pool.

    Searching the acronym alone returns whichever sense dominates the corpus.
    Searching for pages that carry the acronym *and* a given expansion gives
    each sense its own candidate pages, so a rare sense is not crowded out by
    a common one. Sentences that spell the expansion out are dropped — those
    give the answer away exactly as an inline gloss would.
    """
    for expansion in expansions:
        found = 0
        try:
            titles = api.search_cooccurrence(acronym, expansion, limit=pages_per_expansion)
        except RuntimeError as exc:
            LOG.warning("co-occurrence search failed for %r/%r: %s", acronym, expansion, exc)
            continue
        for title in titles:
            if found >= max_per_expansion:
                break
            try:
                data = api.get(action="query", prop="extracts", explaintext=1, titles=title)
            except RuntimeError as exc:
                LOG.warning("extract failed for %r: %s", title, exc)
                continue
            for page in data.get("query", {}).get("pages", []):
                for sentence in page_sentences(page.get("extract") or "", acronym):
                    if mentions_expansion(sentence, expansion):
                        continue
                    yield SenseRow(acronym, sentence, expansion, "wikipedia", title)
                    found += 1
                    if found >= max_per_expansion:
                        break
        LOG.info("%s / %s: %d sentences from %d pages", acronym, expansion, found, len(titles))


# A sentence that introduces the abbreviation — "ממלא מקום (בראשי תיבות: מ\"מ)",
# "מפקד מחלקה, או בקיצור מ\"מ" — is defining the term, not using it. Rewriting
# the full form to the acronym turns it into `מ"מ (בראשי תיבות: מ"מ)`, so these
# must be dropped before substitution rather than filtered after.
DEFINITIONAL_RE = re.compile(r"ראשי\s+ה?תיבות|בקיצור|קיצור\s+של|נוטריקון|הוא\s+כינוי")


def substitutable(sentence: str, acronym: str, expansion: str) -> bool:
    """True when the full form can be rewritten to the acronym cleanly.

    The sentence must spell the expansion out, must not already carry the
    acronym (or the rewrite duplicates it), and must not be a definition of
    the abbreviation itself.
    """
    if not mentions_expansion(sentence, expansion):
        return False
    if mentions_acronym(sentence, acronym):
        return False
    if DEFINITIONAL_RE.search(sentence):
        return False
    return True


def substitute(sentence: str, acronym: str, expansion: str) -> str:
    """Rewrite the spelled-out expansion to the acronym, keeping clitics.

    `לממלא מקום ראש הממשלה` becomes `למ"מ ראש הממשלה`: the proclitic that
    attached to the full form has to survive onto the acronym, or the result
    is ungrammatical.
    """
    pattern = r"(?<![א-ת])([" + hebrew_text.CLITIC_LETTERS + r"]?)" + re.escape(expansion) + r"(?![א-ת])"
    return re.sub(pattern, lambda m: m.group(1) + acronym, sentence)


def mine_substituted(
    api: WikiAPI,
    acronym: str,
    expansions: Iterable[str],
    *,
    pages_per_expansion: int = 12,
    max_per_expansion: int = 5,
) -> Iterator[SenseRow]:
    """Build items by rewriting spelled-out expansions into the acronym.

    Natural acronym usage is scarce for rare senses, but the *full form* is
    ordinary Hebrew and easy to find. Rewriting it yields an item whose sense
    is known by construction — at the cost of naturalness, since a writer who
    spelled the phrase out might not have abbreviated it there. Rows carry
    `provenance="substituted"` so they stay separable from mined text.
    """
    for expansion in expansions:
        found = 0
        try:
            titles = [h["title"] for h in api.search_snippets(expansion, limit=pages_per_expansion)]
        except RuntimeError as exc:
            LOG.warning("search failed for %r: %s", expansion, exc)
            continue
        for title in titles:
            if found >= max_per_expansion:
                break
            try:
                data = api.get(action="query", prop="extracts", explaintext=1, titles=title)
            except RuntimeError as exc:
                LOG.warning("extract failed for %r: %s", title, exc)
                continue
            for page in data.get("query", {}).get("pages", []):
                for raw in SENTENCE_SPLIT_RE.split(page.get("extract") or ""):
                    sentence = " ".join(raw.split())
                    if not (MIN_LEN <= len(sentence) <= MAX_LEN):
                        continue
                    if HEADING_RE.search(sentence) or NON_PROSE_RE.search(sentence):
                        continue
                    if HEBREW_NUMERAL_REF_RE.search(sentence):
                        continue
                    if not substitutable(sentence, acronym, expansion):
                        continue
                    rewritten = substitute(sentence, acronym, expansion)
                    # The rewrite must leave a sentence that passes the same
                    # bar as mined text — one acronym, no gloss, prose only.
                    if not is_clean_sentence(rewritten, acronym):
                        continue
                    yield SenseRow(acronym, rewritten, expansion, "wikipedia", title, "substituted")
                    found += 1
                    if found >= max_per_expansion:
                        break
        LOG.info("%s / %s: %d substituted from %d pages", acronym, expansion, found, len(titles))


# --- deglossed mining -------------------------------------------------------
#
# `is_clean_sentence` drops any sentence that glosses the acronym inline,
# because the answer would sit in the input. But a glossed sentence is the one
# place the corpus proves a sense is real *and* shows a writer actually
# abbreviating it: `בא"ח (בסיס אימונים חטיבתי) גולני שהצטרף ללחימה…`. Removing
# the parenthetical leaves genuine abbreviated usage with a known sense —
# strictly more natural than substitution, which invents the abbreviation.
#
# Senses attested only inside glosses are otherwise unminable. `בא"ח` =
# `בסיס אימונים חטיבתי` occurs on four Hebrew Wikipedia pages and every one
# glosses it, so both the natural and the substituted strategies return zero.

# The gloss forms this removes, all anchored on the acronym:
#   ACR (EXPANSION)        בא"ח (בסיס אימונים חטיבתי)
#   ACR - EXPANSION        בא"ח - בסיס אימונים חטיבתי      (spaced dash)
#   ACR – EXPANSION        בא"ח העורף – בסיס אימונים חטיבתי
# The reverse order (EXPANSION (ACR)) is deliberately NOT handled: removing the
# expansion there leaves the acronym in a slot the expansion's syntax governed,
# which is the same dangling-remainder damage substitution already suffers from.
#
# Known residual defect: a gloss also appears where the acronym is a proper name
# *derived* from the expansion — `כפר אז"ר, על שם אז"ר` names a village after the
# person. Deglossing keeps the sentence, but the acronym there is a name rather
# than a reading, so such rows need the same human check as any other output of
# this module. NAMED_AFTER_RE below catches the commonest phrasing.
def _gloss_patterns(variant: str, expansion: str) -> list[re.Pattern]:
    v, e = re.escape(variant), re.escape(expansion)
    return [
        # ACR (EXPANSION) — including an intervening word: בא"ח העורף (…)
        re.compile(v + r"(?P<mid>(?:\s+[א-ת\"״'׳-]+){0,2})\s*\(\s*" + e + r"\s*\)"),
        # ACR – EXPANSION, spaced dash only (a tight hyphen joins a compound)
        re.compile(v + r"(?P<mid>(?:\s+[א-ת\"״'׳-]+){0,2})\s+[-–—]\s+" + e + r"(?![א-ת])"),
    ]


# "על שם X" / "הקרוי על שמו" mark the acronym as a name given in someone's
# honour rather than an abbreviation being used — `כפר אז"ר, על שם אז"ר`.
NAMED_AFTER_RE = re.compile(r"על\s+שמו?\b|הקרוי|קרוי\s+על|נקרא\s+על")


def deglossable(sentence: str, acronym: str, expansion: str) -> bool:
    """True when `sentence` glosses `acronym` with `expansion` removably."""
    if NAMED_AFTER_RE.search(sentence):
        return False
    for variant in hebrew_text.variants(acronym) or [acronym]:
        for pat in _gloss_patterns(variant, expansion):
            if pat.search(sentence):
                return True
    return False


def degloss(sentence: str, acronym: str, expansion: str) -> str:
    """Strip the inline gloss, keeping the acronym and any words between.

    `בא"ח העורף – בסיס אימונים חטיבתי של חטיבת החילוץ` becomes
    `בא"ח העורף של חטיבת החילוץ`: the acronym and the material that belongs to
    the surrounding clause stay, only the parenthetical definition goes.
    """
    out = sentence
    for variant in hebrew_text.variants(acronym) or [acronym]:
        for pat in _gloss_patterns(variant, expansion):
            out = pat.sub(lambda m: variant + m.group("mid"), out)
    return " ".join(out.split()).replace(" ,", ",").replace(" .", ".")


def mine_deglossed(
    api: WikiAPI,
    acronym: str,
    expansions: Iterable[str],
    *,
    pages_per_expansion: int = 12,
    max_per_expansion: int = 5,
) -> Iterator[SenseRow]:
    """Mine sentences that gloss the acronym, then remove the gloss.

    The text around the acronym is real abbreviated usage, and the gloss makes
    the sense certain, so the result is a natural item with a known label. Rows
    carry `provenance="deglossed"` to stay separable from both `natural` and
    `substituted` — the sentence is genuine but was edited.
    """
    for expansion in expansions:
        found = 0
        try:
            titles = [h["title"] for h in api.search_snippets(expansion, limit=pages_per_expansion)]
        except RuntimeError as exc:
            LOG.warning("search failed for %r: %s", expansion, exc)
            continue
        for title in titles:
            if found >= max_per_expansion:
                break
            try:
                data = api.get(action="query", prop="extracts", explaintext=1, titles=title)
            except RuntimeError as exc:
                LOG.warning("extract failed for %r: %s", title, exc)
                continue
            for page in data.get("query", {}).get("pages", []):
                for raw in SENTENCE_SPLIT_RE.split(page.get("extract") or ""):
                    sentence = " ".join(raw.split())
                    # Length is checked after deglossing: the gloss inflates
                    # the source sentence past MAX_LEN in most cases.
                    if len(sentence) > MAX_LEN + len(expansion) + 8:
                        continue
                    if HEADING_RE.search(sentence) or NON_PROSE_RE.search(sentence):
                        continue
                    if not deglossable(sentence, acronym, expansion):
                        continue
                    stripped = degloss(sentence, acronym, expansion)
                    # The expansion must be gone — a second, unhandled mention
                    # would leave the answer in the input.
                    if mentions_expansion(stripped, expansion):
                        continue
                    # And the result must clear the same bar as mined text.
                    if not is_clean_sentence(stripped, acronym):
                        continue
                    yield SenseRow(acronym, stripped, expansion, "wikipedia", title, "deglossed")
                    found += 1
                    if found >= max_per_expansion:
                        break
        LOG.info("%s / %s: %d deglossed from %d pages", acronym, expansion, found, len(titles))
