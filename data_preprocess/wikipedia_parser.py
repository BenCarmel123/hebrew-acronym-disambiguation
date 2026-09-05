"""Parsing of Hebrew Wikipedia acronym disambiguation pages.

Source of truth is the Hebrew Wikipedia category `קטגוריה:פירושון ראשי תיבות`
(acronym disambiguation pages). Each member page is an acronym type; the bullet
list on the page enumerates its senses.
"""
from __future__ import annotations

import logging
import re
from . import hebrew_text

LOG = logging.getLogger(__name__)

ACRONYM_DISAMBIG_CATEGORY = "קטגוריה:פירושון ראשי תיבות"

# --- wikitext cleaning ------------------------------------------------------

TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")
REF_RE = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", re.DOTALL)
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")
PIPED_LINK_RE = re.compile(r"\[\[([^\[\]|]+)\|([^\[\]]+)\]\]")
LINK_RE = re.compile(r"\[\[([^\[\]]+)\]\]")
EXT_LINK_RE = re.compile(r"\[(?:https?|//)\S+\s+([^\]]*)\]")
BOLD_ITALIC_RE = re.compile(r"'{2,5}")
PARENTHETICAL_RE = re.compile(r"\s*\((?:פירושונים|פירושים נוספים|עמוד פירושונים)\)\s*")
# Only "*" and "#" introduce a sense. ";" is a definition-list header
# ("; בצבא") and ":" is an indented continuation — both are page furniture.
BULLET_RE = re.compile(r"^[*#]+\s*")
DEFLIST_RE = re.compile(r"^[;:]\s*")
SECTION_RE = re.compile(r"^=+\s*(?P<title>.*?)\s*=+\s*$")

# Sections that list related pages rather than senses of this acronym. They can
# appear inside the {{פירושונים}} template, so tracking the current section is
# the only way to exclude their bullets.
NON_SENSE_SECTIONS = {"ראו גם", "ראו עוד", "קישורים חיצוניים", "לקריאה נוספת", "הערות שוליים"}

# Sense lines commonly read "מ״מ – מפקד מחלקה, קצין ..." — everything after one
# of these separators is the gloss, not the expansion itself.
SEPARATORS = ("–", "—", "־", " - ", ":", "&ndash;")

# Person-name markers. Disambiguation pages for rabbinic acronyms list people;
# those are named entities, not general expansions, and the project scopes them
# out (see planning/DECISIONS.md §1, "literal expansion" types).
PERSON_MARKERS = (
    "רבי ", "הרב ", "רב ", "האדמו", "אדמו\"ר", "ר' ", "בן ", "בר ",
    "כינויו", "כינוי ל", "שם עט", "מקובל", "פוסק", "תנא", "אמורא",
)
BIRTH_YEAR_RE = re.compile(r"\(\s*\d{3,4}\s*[-–]\s*\d{0,4}\s*\)")


LATIN_RE = re.compile(r"[A-Za-z]")


def _split_outside_parens(text: str) -> str:
    """Truncate at the first comma that is not inside parentheses."""
    depth = 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            return text[:i].strip()
    return text.strip()


# A trailing parenthesis is a gloss (drop it) when it holds a year range, Latin
# script — a translation such as "(Adobe Illustrator)" — or a quoted alias such
# as ("ירי דו\"צ"). A bare Hebrew parenthetical is a page disambiguator: keep it.
TRAILING_GLOSS_RE = re.compile(
    r"\s*\((?:[^()]*\d{3,4}[^()]*|[^()]*[A-Za-z][^()]*|\s*[\"\u05f4\u201c][^()]*)\)\s*$"
)

# Boilerplate that introduces the expansion instead of being it.
LEAD_IN_RE = re.compile(r"^(?:ראשי תיבות של|קיצור של|כינוי ל|ראו)\s+")


def clean_line(line: str) -> str:
    """Strip wiki markup from a single disambiguation bullet."""
    out = COMMENT_RE.sub("", line)
    out = REF_RE.sub("", out)
    for _ in range(3):  # nested templates
        new = TEMPLATE_RE.sub("", out)
        if new == out:
            break
        out = new
    out = PIPED_LINK_RE.sub(r"\2", out)
    out = LINK_RE.sub(r"\1", out)
    out = EXT_LINK_RE.sub(r"\1", out)
    out = TAG_RE.sub("", out)
    out = BOLD_ITALIC_RE.sub("", out)
    out = BULLET_RE.sub("", out)
    out = PARENTHETICAL_RE.sub(" ", out)
    out = out.replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", out).strip(" \t,.;·-–—")


def looks_like_person(text: str) -> bool:
    """Heuristic filter for biographical senses."""
    if BIRTH_YEAR_RE.search(text):
        return True
    return any(marker in text for marker in PERSON_MARKERS)


def candidate_from_line(line: str, acronym: str) -> str | None:
    """Extract the expansion phrase from one cleaned bullet, or None."""
    text = clean_line(line)
    if not text:
        return None
    # Drop the acronym itself when the line restates it before the separator.
    for sep in SEPARATORS:
        if sep in text:
            head, _, tail = text.partition(sep)
            head_clean = hebrew_text.normalize_acronym(head)
            text = tail.strip() if head_clean == hebrew_text.normalize_acronym(acronym) else head.strip()
            break
    # Split on a comma only outside parentheses, so a life-span like
    # "(1811–1899)" is not cut in half.
    text = _split_outside_parens(text)
    # Strip a trailing gloss in parentheses, but keep a Wikipedia disambiguator
    # such as "שמואל בק (צייר)" — that is part of the page name.
    text = LEAD_IN_RE.sub("", text)
    for _ in range(2):  # a line may carry both a quoted alias and a Latin gloss
        stripped = TRAILING_GLOSS_RE.sub("", text).strip()
        if stripped == text:
            break
        text = stripped
    if not text or len(text) > 60:
        return None
    if not any(c in hebrew_text.HEBREW_LETTERS for c in text):
        return None
    return text


def page_acronym(title: str) -> str:
    """The acronym surface implied by a disambiguation page title."""
    return hebrew_text.normalize_acronym(PARENTHETICAL_RE.sub("", title))


def iter_sense_bullets(wikitext: str) -> "list[str]":
    """Bullet lines that plausibly introduce a sense.

    Skips definition-list headers and any bullet under a "see also"-style
    section, including such sections nested inside {{פירושונים}}.
    """
    out: list[str] = []
    in_non_sense = False
    for raw in wikitext.splitlines():
        line = raw.strip()
        if not line:
            continue
        section = SECTION_RE.match(line)
        if section:
            in_non_sense = section.group("title").strip() in NON_SENSE_SECTIONS
            continue
        if in_non_sense or DEFLIST_RE.match(line) or not BULLET_RE.match(line):
            continue
        out.append(line)
    return out
