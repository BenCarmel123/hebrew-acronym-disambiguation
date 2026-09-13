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


# Standard gematria values, sofit (final) forms folded onto their medial
# letter — a citation or serial number is written with whichever form falls
# at a word boundary (ך and כ both stand for 20), and the acronym's own
# gershayim placement doesn't change the value.
GEMATRIA_VALUES = {
    "א": 1, "ב": 2, "ג": 3, "ד": 4, "ה": 5, "ו": 6, "ז": 7, "ח": 8, "ט": 9,
    "י": 10, "כ": 20, "ל": 30, "מ": 40, "נ": 50, "ס": 60, "ע": 70, "פ": 80,
    "צ": 90, "ק": 100, "ר": 200, "ש": 300, "ת": 400,
}

_HUNDREDS_HE = {100: "מאה", 200: "מאתיים", 300: "שלוש מאות", 400: "ארבע מאות",
                500: "חמש מאות", 600: "שש מאות", 700: "שבע מאות", 800: "שמונה מאות",
                900: "תשע מאות"}
_TENS_HE = {10: "עשר", 20: "עשרים", 30: "שלושים", 40: "ארבעים", 50: "חמישים",
            60: "שישים", 70: "שבעים", 80: "שמונים", 90: "תשעים"}
_ONES_HE = {1: "אחת", 2: "שתיים", 3: "שלוש", 4: "ארבע", 5: "חמש", 6: "שש",
            7: "שבע", 8: "שמונה", 9: "תשע"}
# 11-19 are irregular compounds (ones-then-"עשרה"), not ones+"עשר" or the
# reverse — "שמונה עשרה" (18), never "עשר ושמונה".
_TEENS_HE = {
    11: "אחת עשרה", 12: "שתים עשרה", 13: "שלוש עשרה", 14: "ארבע עשרה",
    15: "חמש עשרה", 16: "שש עשרה", 17: "שבע עשרה", 18: "שמונה עשרה",
    19: "תשע עשרה",
}


def gematria_value(acronym: str) -> int | None:
    """The standard (mispar hechrachi) numeric value of an acronym's letters,
    or None when any letter falls outside 1-400 (the single-letter values —
    there is no compounding beyond simple digit sum, which is what every
    citation/serial-number use of a two-letter acronym relies on).

    Every two-Hebrew-letter acronym is *also* a valid number in this system —
    ת"ק is 500 (400+100), א"ח is 9 (1+8) — so this is deliberately permissive:
    it says nothing about whether a number reading is the *likely* sense in
    a given sentence, only whether one exists at all as a distractor/gold
    candidate.
    """
    letters = acronym_letters(acronym)
    if not letters:
        return None
    total = 0
    for c in letters:
        v = GEMATRIA_VALUES.get(c)
        if v is None:
            return None
        total += v
    return total


def gematria_value_hebrew(value: int) -> str:
    """Spell out a gematria value in words, Hebrew feminine cardinal form
    (as used for a bare count — "ארבע מאות ושתיים", not "ה-402") — matching
    how these numbers are spoken/written out in Knesset bill-number prose.
    Only 1-999 is supported, since no acronym's letter-sum exceeds 400+400.
    """
    if not (1 <= value <= 999):
        raise ValueError(f"gematria_value_hebrew only supports 1-999, got {value}")
    hundreds, rem = divmod(value, 100)
    parts = []
    if hundreds:
        parts.append(_HUNDREDS_HE[hundreds * 100])
    if 11 <= rem <= 19:
        parts.append(_TEENS_HE[rem])
    else:
        tens, ones = divmod(rem, 10)
        if tens:
            parts.append(_TENS_HE[tens * 10])
        if ones:
            parts.append(_ONES_HE[ones])
    if not parts:
        return "אפס"
    return " ו".join(parts) if len(parts) > 1 else parts[0]


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
