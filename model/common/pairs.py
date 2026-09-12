"""Preprocessing for the DictaBERT cross-encoder: rows -> (marked context, candidate, label) pairs.

Self-contained; standard library only. Used by both local code and the Colab training
notebook (fetched by raw URL), so this is the ONE place span-marking and pair-building
happen — if the notebook built its own copy, training and any later local scoring could
silently drift apart in how a span is marked or a pair is assembled.
"""
from __future__ import annotations

import csv

#: Marker tokens wrapping the target acronym occurrence, added to the tokenizer as real
#: tokens by whoever builds the model (see model.py). Defined here because pair-building
#: is what actually inserts them into text.
ACR_OPEN, ACR_CLOSE = "[ACR]", "[/ACR]"

#: A well-formed ranking task needs a real choice. With one candidate the argmax is
#: correct by construction and any accuracy computed over it is inflated.
MIN_CANDIDATES = 2

#: Hebrew punctuation the sources use interchangeably with ASCII quotes.
_EQUIV = {"״": '"', "“": '"', "”": '"',      # gershayim, curly doubles
          "׳": "'", "‘": "'", "’": "'"}      # geresh, curly singles


def _normalise(s: str) -> str:
    """Fold quote variants. LENGTH-PRESERVING, so a span found in the normalised string
    is a valid span into the original."""
    return "".join(_EQUIV.get(ch, ch) for ch in s)


def find_span(sentence: str, acronym: str) -> tuple[int, int] | None:
    """First occurrence of `acronym` in `sentence`, quote-variant insensitive.

    First occurrence only: this snapshot carries no annotated char_span, so which
    occurrence is the real target is genuinely unknown when an acronym appears twice.
    None when the acronym cannot be located at all.
    """
    hay, needle = _normalise(sentence), _normalise(acronym)
    i = hay.find(needle)
    return None if i < 0 else (i, i + len(needle))


def mark_span(sentence: str, span: tuple[int, int]) -> str:
    """Wrap the given span in the marker tokens."""
    s, e = span
    return f"{sentence[:s]}{ACR_OPEN}{sentence[s:e]}{ACR_CLOSE}{sentence[e:]}"


def load_rows(path: str) -> list[dict]:
    """Read one of the acronym_items-shaped CSVs (utf-8-sig: these carry a BOM)."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def build_pairs(rows: list[dict]) -> list[tuple[str, str, int]]:
    """rows -> list of (marked_context, candidate_string, label).

    One pair per candidate of every row. label=1 for the gold expansion, 0 for every
    other candidate of that same acronym. Rows with fewer than MIN_CANDIDATES candidates,
    or whose acronym cannot be located in its sentence, are silently skipped — call
    `describe_skips` first if you want to know how many and why.
    """
    pairs = []
    for r in rows:
        cands = [c.strip() for c in r["candidates"].split("|") if c.strip()]
        if len(cands) < MIN_CANDIDATES:
            continue
        span = find_span(r["sentence"], r["acronym"])
        if span is None:
            continue
        marked = mark_span(r["sentence"], span)
        gold = r["gold_expansion"].strip()
        for c in cands:
            pairs.append((marked, c, 1 if c == gold else 0))
    return pairs


def describe_skips(rows: list[dict]) -> dict:
    """Counts of why rows were dropped by build_pairs, so a surprise shows up here and
    not as a silently smaller training set."""
    too_few, unlocatable, ok = 0, 0, 0
    for r in rows:
        cands = [c.strip() for c in r["candidates"].split("|") if c.strip()]
        if len(cands) < MIN_CANDIDATES:
            too_few += 1
        elif find_span(r["sentence"], r["acronym"]) is None:
            unlocatable += 1
        else:
            ok += 1
    return {"total": len(rows), "usable": ok,
            "skipped_too_few_candidates": too_few, "skipped_unlocatable": unlocatable}
