"""Merge near-duplicate expansions that differ only in punctuation/spacing.

"מילימטר" vs "מילי מטר", "אלוף-משנה" vs "אלוף משנה" are the same sense written
two ways, not two candidates. Left unmerged they inflate the sense count for an
acronym and corrupt any frequency-balance calculation (a real 2-way choice can
look like 3, or a genuine skew can look artificially balanced).
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import replace
from typing import Sequence

from .wikipedia_source import BulletRow

LOG = logging.getLogger(__name__)

# Punctuation/whitespace that does not change meaning: hyphens, spaces, dots,
# apostrophes. Niqqud/gershayim differences are handled separately by
# hebrew_text.normalize_acronym on the acronym field; this is about the expansion.
_STRIP_RE = re.compile(r"[-\s.'׳״]")


def dedupe_key(expansion: str) -> str:
    """Identity of an expansion once cosmetic punctuation is ignored."""
    return _STRIP_RE.sub("", expansion)


def drop_single_letter_acronyms(rows: Sequence[BulletRow]) -> list[BulletRow]:
    """Remove rows whose "acronym" is a single Latin letter (A, C, D, ...).

    Wiktionary has an entry for each Latin letter on its own (the letter "a",
    "the third letter of the Latin alphabet", unit symbols like "C" for
    Celsius/Coulomb/Carbon). Those are dictionary entries for the letter, not
    ambiguous Hebrew acronyms, and mining their bullet lists pulls in unrelated
    words that merely happen to appear on the page (e.g. "C" -> "מספיק", which
    is not an expansion of anything). Out of scope regardless of hit count.
    """
    out = [r for r in rows if len(r.acronym) != 1 or not r.acronym.isascii()]
    LOG.info("dropped %d single-letter-acronym rows", len(rows) - len(out))
    return out


def dedupe_expansions(rows: Sequence[BulletRow]) -> list[BulletRow]:
    """Collapse same-acronym rows whose expansions differ only cosmetically.

    Keeps the row with the higher hit count (the more standard spelling is
    usually also the more frequent one); ties keep the first row seen.
    """
    best: dict[tuple[str, str], BulletRow] = {}
    for row in rows:
        key = (row.acronym, dedupe_key(row.expansion))
        current = best.get(key)
        if current is None or row.hits > current.hits:
            best[key] = row
    out = list(best.values())
    LOG.info("dedupe: %d rows -> %d rows (%d merged)", len(rows), len(out), len(rows) - len(out))
    return out


# --- fuzzy near-duplicate review (human-in-the-loop) ------------------------
#
# Exact-punctuation dedup (above) only catches "מילימטר" vs "מילי מטר". It
# misses cases like "מספן המודיעין" vs "מספן מודיעין" (a dropped definite ה) or
# "חכמת הנסתר" vs "חכמת נסתר" — real duplicates with a small textual edit — and
# it must never touch "רבי אלעזר" vs "רבי אליעזר" (two different rabbis) or
# "על ידי" vs "על יד" (different meanings). That distinction requires reading
# the Hebrew, so the script's job stops at flagging candidates; a person makes
# the merge/keep call. See `review_near_duplicates` and `apply_review_decisions`
# below, and `docs/near_duplicate_review.md` for the workflow.

import csv
import difflib
from pathlib import Path


def find_near_duplicates(rows: Sequence[BulletRow], min_ratio: float = 0.75) -> list[dict]:
    """Pairs of same-acronym expansions whose text is suspiciously similar.

    Returns dicts with an empty `decision` column for a human to fill in with
    "merge" or "keep". Sorted by similarity, most suspicious first.
    """
    by_acronym: dict[str, list[BulletRow]] = defaultdict(list)
    for row in rows:
        by_acronym[row.acronym].append(row)

    flagged = []
    for acronym, group in by_acronym.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                ratio = difflib.SequenceMatcher(None, a.expansion, b.expansion).ratio()
                if ratio >= min_ratio:
                    flagged.append(
                        {
                            "acronym": acronym,
                            "expansion_a": a.expansion,
                            "hits_a": a.hits,
                            "expansion_b": b.expansion,
                            "hits_b": b.hits,
                            "similarity": round(ratio, 3),
                            "decision": "",  # human fills in: "merge" or "keep"
                        }
                    )
    flagged.sort(key=lambda d: -d["similarity"])
    LOG.info("flagged %d near-duplicate pairs for review", len(flagged))
    return flagged


REVIEW_FIELDS = ["acronym", "expansion_a", "hits_a", "expansion_b", "hits_b", "similarity", "decision"]


def write_review_csv(path: str | Path, flagged: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for row in flagged:
            writer.writerow(row)
    LOG.info("wrote %d rows for review -> %s", len(flagged), path)


def apply_review_decisions(rows: Sequence[BulletRow], review_path: str | Path) -> list[BulletRow]:
    """Apply a filled-in review CSV: drop the lower-hit row of each "merge" pair.

    Rows marked "keep" or left blank are untouched. A pair marked "merge" but
    already removed by an earlier pair (transitive duplicates) is skipped
    silently — nothing to drop twice.
    """
    to_drop: set[tuple[str, str]] = set()
    with open(review_path, encoding="utf-8-sig", newline="") as fh:
        for rec in csv.DictReader(fh):
            if rec.get("decision", "").strip().lower() != "merge":
                continue
            hits_a, hits_b = int(rec["hits_a"]), int(rec["hits_b"])
            loser = rec["expansion_b"] if hits_a >= hits_b else rec["expansion_a"]
            to_drop.add((rec["acronym"], loser))

    out = [r for r in rows if (r.acronym, r.expansion) not in to_drop]
    LOG.info("applied review: dropped %d rows", len(rows) - len(out))
    return out
