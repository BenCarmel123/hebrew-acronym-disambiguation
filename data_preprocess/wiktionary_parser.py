"""Parse Hebrew Wiktionary acronym entries.

Wiktionary is the complementary source to Wikipedia: its entries state *literal
expansions* where Wikipedia disambiguation pages mostly list referents (see
`research/source_evidence/REPORT.md`). Since the ratified prediction target is
the literal expansion string, this is the higher-value inventory of the two.

Entry structure (from `מ"מ`, `ש"ס`):

    == מ"מ ==
    # {{משלב/ר"ת|יחידות מידה}} [[מילימטר|'''מ'''ילי '''מ'''טר]] (אלפית ה[[מטר]])
    # '''מ'''כל '''מ'''קום
    #:* הרב הזה בקיא בכל ה'''ש"ס'''.     <- usage example, not a sense

So: `#` lines are senses, `#:` / `#*` lines are examples, `{{משלב/ר"ת|X}}` gives a
free domain label, and bold markup marks the acronym letters *inside* a word,
which means it must be stripped without splitting the word.
"""
from __future__ import annotations

import logging
import re

from . import hebrew_text
from .wikipedia_parser import (
    COMMENT_RE,
    EXT_LINK_RE,
    LINK_RE,
    PIPED_LINK_RE,
    REF_RE,
    TAG_RE,
    TEMPLATE_RE,
)

LOG = logging.getLogger(__name__)

WIKTIONARY_API = "https://he.wiktionary.org/w/api.php"
ACRONYM_CATEGORY = "קטגוריה:ראשי תיבות"

# A sense is "#" possibly repeated; "#:" and "#*" introduce examples/quotations.
SENSE_RE = re.compile(r"^#+(?![:*])\s*")
EXAMPLE_RE = re.compile(r"^#+[:*]")

# {{משלב/ר"ת|יחידות מידה}} — register template carrying a domain label.
REGISTER_RE = re.compile(r"\{\{משלב(?:/[^|{}]*)?\|([^|{}]+)\}\}")

# Trailing gloss in parentheses: "(אלפית המטר)".
TRAILING_PAREN_RE = re.compile(r"\s*\([^()]*\)\s*$")

# Some senses append a free-text definition after the expansion, separated by a
# full stop: "שומרי תורה ספרדיים. שם מפלגה בכנסת ישראל". Keep the first clause.
DEFINITION_TAIL_RE = re.compile(r"(?<=[א-ת])\.\s+\S.*$", re.DOTALL)

# Bold marks the acronym letters inside a word — strip the quotes
# only, never the letters, and never introduce a space.
BOLD_RE = re.compile(r"'{2,5}")

# Line breaks inside a sense separate an English gloss from the Hebrew one
# ("# Artificial Intelligence</br>[[בינה מלאכותית]]"). Split, do not concatenate.
BREAK_RE = re.compile(r"<\s*/?\s*br\s*/?\s*>", re.IGNORECASE)


def clean_sense(line: str) -> str:
    """Reduce one `#` sense line to its bare expansion phrase."""
    out = SENSE_RE.sub("", line)
    out = COMMENT_RE.sub("", out)
    out = REF_RE.sub("", out)
    out = REGISTER_RE.sub("", out)
    for _ in range(3):
        new = TEMPLATE_RE.sub("", out)
        if new == out:
            break
        out = new
    out = PIPED_LINK_RE.sub(r"\2", out)
    out = LINK_RE.sub(r"\1", out)
    out = EXT_LINK_RE.sub(r"\1", out)
    # Split on <br> first: TAG_RE would delete it and fuse the two glosses.
    out = BREAK_RE.split(out)[-1] if BREAK_RE.search(out) else out
    out = TAG_RE.sub("", out)
    out = BOLD_RE.sub("", out)
    out = out.replace("&nbsp;", " ")
    out = TRAILING_PAREN_RE.sub("", out)
    out = re.sub(r"\s+", " ", out)
    out = DEFINITION_TAIL_RE.sub("", out)
    return out.strip(" \t,.;:·-–—")


def sense_domain(line: str) -> str:
    """The register/domain label on a sense line, or '' when absent."""
    match = REGISTER_RE.search(line)
    return match.group(1).strip() if match else ""


def clean_example(line: str) -> str:
    """Reduce one `#:`/`#*` usage-example line to plain sentence text.

    Same cleanup as `clean_sense` minus the sense-line-specific steps (register
    template, trailing-parenthesis gloss, definition tail) that do not apply to
    a quoted sentence.
    """
    out = EXAMPLE_RE.sub("", line)
    out = out.lstrip("#:* \t")
    out = COMMENT_RE.sub("", out)
    out = REF_RE.sub("", out)
    for _ in range(3):
        new = TEMPLATE_RE.sub("", out)
        if new == out:
            break
        out = new
    out = PIPED_LINK_RE.sub(r"\2", out)
    out = LINK_RE.sub(r"\1", out)
    out = EXT_LINK_RE.sub(r"\1", out)
    out = BREAK_RE.split(out)[-1] if BREAK_RE.search(out) else out
    out = TAG_RE.sub("", out)
    out = BOLD_RE.sub("", out)
    out = out.replace("&nbsp;", " ")
    out = re.sub(r"\s+", " ", out)
    return out.strip(" \t,.;:·-–—")


def extract_examples(wikitext: str) -> list[str]:
    """Usage-example sentences (`#:`/`#*` lines) from a Wiktionary entry.

    These quote the acronym in a real sentence, e.g. `הרב הזה בקיא בכל
    ה'''ש"ס'''` — unlike the `#` sense lines, which state a bare expansion
    phrase. `parse_entry` discards these lines entirely; this is the
    complementary extractor for context mining.
    """
    out: list[str] = []
    for raw in wikitext.splitlines():
        line = raw.strip()
        if not line or not EXAMPLE_RE.match(line):
            continue
        cleaned = clean_example(line)
        if cleaned and any(c in hebrew_text.HEBREW_LETTERS for c in cleaned):
            out.append(cleaned)
    return out


def parse_entry(wikitext: str, acronym: str) -> list[dict]:
    """Return `[{expansion, domain, initials_match, raw_line}, ...]` for a page.

    Senses are taken in page order; duplicates are dropped. Nothing is filtered
    on plausibility — `initials_match` is recorded as a flag, matching the
    Wikipedia counter's convention.
    """
    out: list[dict] = []
    seen: set[str] = set()
    for raw in wikitext.splitlines():
        line = raw.strip()
        if not line or EXAMPLE_RE.match(line) or not SENSE_RE.match(line):
            continue
        expansion = clean_sense(line)
        if not expansion or len(expansion) > 60:
            continue
        if not any(c in hebrew_text.HEBREW_LETTERS for c in expansion):
            continue
        key = hebrew_text.strip_niqqud(expansion)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "expansion": expansion,
                "domain": sense_domain(line),
                "initials_match": hebrew_text.matches_acronym(expansion, acronym),
                "raw_line": line,
            }
        )
    return out
