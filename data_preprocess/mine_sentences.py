"""Mine clean prose sentences for acronyms, from Wikipedia article plain text.

An earlier approach mined `list=search` snippets, which are fixed-width
excerpts cut from anywhere on a page — reference lists, tables, image captions,
infoboxes. No amount of post-cleaning made those reliably read as prose (~32%
usable after ten filters), so that module was removed and this one takes a
different route: use search only to *find* pages, then fetch each
page's rendered plain text (`prop=extracts&explaintext`), split it into
sentences, and keep only sentences that read like natural usage.

A sentence is kept when it:

- contains the target acronym (in either quote variant),
- contains no *other* acronym-shaped token, so the item is unambiguous,
- does not gloss the acronym inline (`אח"ם (אגף חקירות ומודיעין)`), which
  would give the answer away in the input,
- carries no section-header, list, citation or Hebrew-numeral-reference
  residue, and
- is a plausible sentence length.

`ContextRow` is the row shape the rest of the pipeline
(`build_annotation_table`) consumes; `SenseRow` extends it with the expansion
and provenance that sense-aware mining records.

Three mining strategies live here, in increasing order of how much they
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

Every strategy applies the same output bar: `is_clean_sentence` runs on the
final text, so a substituted sentence that ends up with two acronyms or an
inline gloss is dropped exactly as a mined one would be.
"""
from __future__ import annotations

import logging
import re
from typing import Iterable, Iterator

from dataclasses import dataclass

from . import hebrew_text
from .wiki_client import WikiAPI
from .wiktionary_parser import extract_examples


@dataclass(frozen=True)
class ContextRow:
    """One mined sentence, with the acronym it illustrates."""

    acronym: str
    context: str
    source: str  # "wikipedia" | "wiktionary"
    page_title: str

LOG = logging.getLogger(__name__)

# A Hebrew acronym marks its final letter: the quote sits immediately before
# the last letter of a short token (צה"ל, אג"ם, רמב"ם). Requiring that shape —
# rather than "letters, quote, letters" — is what separates an acronym from a
# prefixed quotation such as ב"אליאנס or ו"שרוכים, where the same character
# opens a quoted phrase instead.
ACRONYM_TOKEN_RE = re.compile(r"(?<![א-ת])[א-ת]{1,6}[\"״][א-ת](?![א-ת])")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# Hebrew-numeral references — "משניות י' - י\"ג", "פרק ב'" — are chapter/verse
# citations, not prose, and their geresh reads as an acronym marker.
#
# The same geresh marks the honorific "ר' וידאל" (Rabbi), which is ordinary
# prose and common in rabbinic articles. What separates them is the letter
# itself: a citation uses a letter with a numeric value in its numeral sense
# (א-ט units, י-צ tens, ק-ת hundreds), while the honorifics that matter here
# are a small closed set (ר' rabbi, ד' the Name, ע' page). Listing the
# honorific letters as exceptions keeps them and still drops the citations.
HONORIFIC_GERESH = "רדע"
HEBREW_NUMERAL_REF_RE = re.compile(r"(?<![א-ת])[א-ת][׳']")
# A sentence that says "ראשי תיבות של…" / "קיצור של…" spells the expansion out
# in words rather than leaving it to be inferred from context.
EXPLICIT_GLOSS_RE = re.compile(r"ראשי\s+ה?תיבות|קיצור\s+של|נוטריקון")
# Plain-text extracts keep section headings as "== Title ==" lines.
HEADING_RE = re.compile(r"==+[^=]*==+")
# Residue that marks a line as a list entry, table row or citation rather
# than a sentence.
NON_PROSE_RE = re.compile(
    r"https?://|\.(?:jpg|png|svg|gif)\b|\|[a-z]+=|rowspan|colspan|ממוזער|\d+px|\{\{|\}\}|\[\[|\]\]",
    re.IGNORECASE,
)

MIN_LEN = 40
MAX_LEN = 200



def is_numeral_reference(sentence: str) -> bool:
    """True when a geresh in `sentence` marks a chapter/verse citation.

    A geresh after a single Hebrew letter is usually a numeral reference
    ("פרק ב'", "משניות י' - י\"ג"), which is citation apparatus rather than
    prose. The exception is a handful of honorific abbreviations — ר' for
    rabbi above all — that are ordinary prose and frequent in exactly the
    rabbinic articles the rarer acronyms live in.
    """
    for m in HEBREW_NUMERAL_REF_RE.finditer(sentence):
        if m.group()[0] not in HONORIFIC_GERESH:
            return True
    return False


def mentions_acronym(sentence: str, acronym: str) -> bool:
    """True when `acronym` occurs in `sentence` as a whole token.

    Substring matching is not enough: `ב"ש` occurs inside `ב"שירות` — a
    prefix letter followed by a quoted word — which is not the acronym at
    all. The occurrence must not be followed by further Hebrew letters, and
    may only be preceded by Hebrew proclitics (ה, ו, ב, כ, ל, מ, ש).
    """
    for variant in hebrew_text.variants(acronym) or [acronym]:
        pattern = r"(?<![א-ת])[" + hebrew_text.CLITIC_LETTERS + r"]?" + re.escape(variant) + r"(?![א-ת])"
        if re.search(pattern, sentence):
            return True
    return False


def is_clean_sentence(sentence: str, acronym: str) -> bool:
    """True when `sentence` is usable as a disambiguation item for `acronym`."""
    if not (MIN_LEN <= len(sentence) <= MAX_LEN):
        return False
    variants = hebrew_text.variants(acronym) or [acronym]
    if not mentions_acronym(sentence, acronym):
        return False
    if HEADING_RE.search(sentence) or NON_PROSE_RE.search(sentence):
        return False
    if is_numeral_reference(sentence):
        return False
    # An inline gloss states the expansion outright, in either order:
    # `ח"ש (חודר שריון)` or, more commonly, `חודר שריון (ח"ש)`. Both make the
    # item unanswerable-by-context — the answer is already in the input.
    if any(
        re.search(re.escape(v) + r"\s*\(", sentence)
        or re.search(r"\(\s*" + re.escape(v) + r"\s*\)", sentence)
        # A *spaced* dash introduces an expansion ("ד\"ש – התנועה הדמוקרטית"),
        # while a tight hyphen joins a compound name ("ש\"ס-העבודה") and is
        # ordinary prose — so the spacing, not the dash, is the signal.
        or re.search(re.escape(v) + r"\s+[-–—]\s+[א-ת]", sentence)
        for v in variants
    ):
        return False
    # An explicit "ראשי תיבות" gloss names the expansion in words.
    if EXPLICIT_GLOSS_RE.search(sentence):
        return False
    # Exactly one acronym in the sentence: the target, in whichever variant.
    canonical = {v.replace("״", '"') for v in variants}
    present = {t.replace("״", '"') for t in ACRONYM_TOKEN_RE.findall(sentence)}
    return not (present - canonical)


def page_sentences(text: str, acronym: str) -> Iterator[str]:
    """Clean sentences mentioning `acronym`, from one article's plain text."""
    for raw in SENTENCE_SPLIT_RE.split(text):
        sentence = " ".join(raw.split())
        if is_clean_sentence(sentence, acronym):
            yield sentence


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


def mine_wiktionary_sentences(entries: dict[str, str]) -> Iterator[ContextRow]:
    """Usage-example sentences carried in Wiktionary entries themselves.

    `entries` is `{page_title: wikitext}`. Unlike a search snippet, a `#:`/`#*`
    example line is written by a lexicographer to illustrate the word, so it
    needs no prose filtering — only the same gloss and single-acronym checks
    every other item is held to.
    """
    for title, wikitext in entries.items():
        acronym = hebrew_text.normalize_acronym(title)
        for example in extract_examples(wikitext):
            if is_clean_sentence(example, acronym):
                yield ContextRow(acronym, example, "wiktionary", title)


@dataclass(frozen=True)
class SenseRow:
    """One mined sentence, with the sense its source page suggests.

    `expansion` is *provisional*: it records which expansion's page pool the
    sentence came from, not a verified reading. A human confirms or flips it
    during annotation, so it is a starting point that makes labelling fast,
    never a gold label.
    """

    acronym: str
    context: str
    expansion: str
    source: str
    page_title: str
    provenance: str = "natural"


def mentions_expansion(sentence: str, expansion: str) -> bool:
    """True when the sentence spells the expansion out, with clitics allowed."""
    pattern = r"(?<![א-ת])[" + hebrew_text.CLITIC_LETTERS + r"]?" + re.escape(expansion) + r"(?![א-ת])"
    return re.search(pattern, sentence) is not None


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
