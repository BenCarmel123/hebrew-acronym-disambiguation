"""Hebrew orthography helpers shared by the acronym pipeline.

All functions here are pure and dependency-free so they can be unit tested
without any network access.
"""
from __future__ import annotations

import re
import unicodedata

# Final (sofit) forms mapped to their medial counterparts. Acronym initials are
# always written in medial form, so normalising here lets us verify that an
# expansion really abbreviates to a given acronym.
FINAL_TO_MEDIAL = {"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"}

HEBREW_LETTERS = "אבגדהוזחטיכךלמםנןסעפףצץקרשת"

# Gershayim (U+05F4) is the correct character; the ASCII double quote is what
# most corpora actually contain. We accept both everywhere and emit both.
GERSHAYIM = "״"
GERESH = "׳"
QUOTE_CHARS = (GERSHAYIM, '"', "“", "”")

NIQQUD = re.compile(r"[֑-ׇ]")

# Single-letter proclitics that attach to the following word. Hebrew allows
# short stacks of them ("וכשב..."), so we match one to four in sequence.
CLITIC_LETTERS = "ובכלמשהד"
CLITIC_PREFIX = f"[{CLITIC_LETTERS}]{{0,4}}"

# Words that carry no initial in a conventional acronym.
SKIPPABLE_WORDS = {"של", "את", "ה", "ו"}


def strip_niqqud(text: str) -> str:
    """Remove vowel points and cantillation marks."""
    return NIQQUD.sub("", unicodedata.normalize("NFC", text))


def normalize_quotes(text: str) -> str:
    """Fold every quote-like character onto the gershayim."""
    out = text
    for ch in QUOTE_CHARS:
        out = out.replace(ch, GERSHAYIM)
    return out


def normalize_acronym(acronym: str) -> str:
    """Canonical form of an acronym surface: no niqqud, gershayim, no spaces."""
    return normalize_quotes(strip_niqqud(acronym)).replace(" ", "").strip()


def acronym_letters(acronym: str) -> str:
    """The bare Hebrew letters of an acronym, in medial form."""
    letters = [c for c in normalize_acronym(acronym) if c in HEBREW_LETTERS]
    return "".join(FINAL_TO_MEDIAL.get(c, c) for c in letters)


def expansion_initials(phrase: str) -> str:
    """Initial letters of a phrase, in medial form, skipping function words."""
    words = [w for w in strip_niqqud(phrase).split() if w]
    initials = []
    for word in words:
        word = word.strip("־-()[],.׳״\"'")
        if not word or word in SKIPPABLE_WORDS:
            continue
        first = word[0]
        if first not in HEBREW_LETTERS:
            continue
        initials.append(FINAL_TO_MEDIAL.get(first, first))
    return "".join(initials)


def matches_acronym(phrase: str, acronym: str) -> bool:
    """True when `phrase` plausibly abbreviates to `acronym`.

    Two conventions are accepted. Strict initials (מפקד מחלקה -> מ״מ) is the
    common case. The looser case covers acronyms that lift extra letters from
    inside a word (בית ספר -> ביה״ס, מילימטר -> מ״מ): every acronym letter must
    then occur in the phrase in order, and the phrase must start with the first
    acronym letter.
    """
    target = acronym_letters(acronym)
    if not target:
        return False
    initials = expansion_initials(phrase)
    if initials == target:
        return True
    letters = [FINAL_TO_MEDIAL.get(c, c) for c in strip_niqqud(phrase) if c in HEBREW_LETTERS]
    if not letters or letters[0] != target[0]:
        return False
    it = iter(letters)
    return all(c in it for c in target)


def variants(acronym: str) -> list[str]:
    """Surface variants of an acronym as they appear in real text."""
    canon = normalize_acronym(acronym)
    seen, out = set(), []
    for quote in (GERSHAYIM, '"'):
        v = canon.replace(GERSHAYIM, quote)
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out
