"""Source-agnostic sentence shape/cleanliness checks shared by every miner.

A sentence is kept when it:

- contains the target acronym (in either quote variant),
- contains no *other* acronym-shaped token, so the item is unambiguous
  (relaxed for acronym-dense genres — see `is_clean_sentence_multi_acronym`),
- does not gloss the acronym inline (`אח"ם (אגף חקירות ומודיעין)`), which
  would give the answer away in the input,
- carries no section-header, list, citation or Hebrew-numeral-reference
  residue, and
- is a plausible sentence length.

`ContextRow` is the row shape every miner emits; `SenseRow` extends it with
the expansion and provenance that sense-aware mining records.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

from . import hebrew_text


@dataclass(frozen=True)
class ContextRow:
    """One mined sentence, with the acronym it illustrates."""

    acronym: str
    context: str
    source: str  # "wikipedia" | "wiktionary" | "sefaria" | "knesset"
    page_title: str


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


def mentions_expansion(sentence: str, expansion: str) -> bool:
    """True when the sentence spells the expansion out, with clitics allowed."""
    pattern = r"(?<![א-ת])[" + hebrew_text.CLITIC_LETTERS + r"]?" + re.escape(expansion) + r"(?![א-ת])"
    return re.search(pattern, sentence) is not None


def _passes_common_filters(sentence: str, acronym: str, variants: list[str]) -> bool:
    """Length, prose shape, and gloss checks shared by every mining source."""
    if not (MIN_LEN <= len(sentence) <= MAX_LEN):
        return False
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
    return True


def is_clean_sentence(sentence: str, acronym: str) -> bool:
    """True when `sentence` is usable as a disambiguation item for `acronym`."""
    variants = hebrew_text.variants(acronym) or [acronym]
    if not _passes_common_filters(sentence, acronym, variants):
        return False
    # Exactly one acronym in the sentence: the target, in whichever variant.
    canonical = {v.replace("״", '"') for v in variants}
    present = {t.replace("״", '"') for t in ACRONYM_TOKEN_RE.findall(sentence)}
    return not (present - canonical)


def is_clean_sentence_multi_acronym(sentence: str, acronym: str) -> bool:
    """`is_clean_sentence`, but tolerant of *other*, different acronyms.

    Rabbinic/Talmudic prose is acronym-dense — a sentence naming מהר"ש will
    routinely also carry ז"ל, רשב"י, מהרח"ו. Rejecting every sentence with more
    than one acronym-shaped token (as `is_clean_sentence` does for Wikipedia)
    would reject almost this entire genre. What must still hold is that the
    *target* acronym itself is unambiguous — it names one type, once, not
    several distinct forms of the same letters mixed together — since a human
    reviewer reading the sentence still knows which acronym they are being
    asked about even when others appear nearby.
    """
    variants = hebrew_text.variants(acronym) or [acronym]
    if not _passes_common_filters(sentence, acronym, variants):
        return False
    canonical = {v.replace("״", '"') for v in variants}
    present = [t.replace("״", '"') for t in ACRONYM_TOKEN_RE.findall(sentence)]
    target_hits = [t for t in present if t in canonical]
    return len(target_hits) == 1


def page_sentences(text: str, acronym: str, *, is_clean=is_clean_sentence) -> Iterator[str]:
    """Clean sentences mentioning `acronym`, from one article's/segment's plain text.

    `is_clean` is pluggable so a different-genre source (Sefaria's acronym-
    dense rabbinic prose) can supply `is_clean_sentence_multi_acronym` instead
    of the Wikipedia-tuned default.
    """
    for raw in SENTENCE_SPLIT_RE.split(text):
        sentence = " ".join(raw.split())
        if is_clean(sentence, acronym):
            yield sentence
