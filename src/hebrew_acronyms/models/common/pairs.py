"""Build target-marked context/candidate pairs for training and evaluation."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re

from hebrew_acronyms.data_processing.common.csv_io import load_rows

#: Marker tokens wrapping the target acronym occurrence, added to the tokenizer as real
#: tokens by whoever builds the model (see model.py). Defined here because pair-building
#: is what actually inserts them into text.
ACR_OPEN, ACR_CLOSE = "[ACR]", "[/ACR]"

#: Preserve the existing training minimum; inference accepts singleton inventories.
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


def validate_ids(rows: list[dict]) -> None:
    """Reject missing/duplicate IDs before training or prediction starts."""
    seen = set()
    for row in rows:
        item_id = row.get("item_id")
        if not isinstance(item_id, str) or not item_id.strip():
            raise ValueError("Each item requires a nonempty string item_id")
        if item_id in seen:
            raise ValueError(f"Duplicate item_id: {item_id!r}")
        seen.add(item_id)


def explicit_span(row: dict) -> tuple[int, int]:
    """Half-open Python character offsets into the original, unnormalised sentence.

    Decimal strings are accepted for CSV offsets. target_raw includes any attached
    prefix. No acronym search, quote folding or alias inference occurs here.
    """
    sentence, target = row.get("sentence"), row.get("target_raw")
    if not isinstance(sentence, str) or not isinstance(target, str) or not target:
        raise ValueError("sentence and nonempty target_raw are required")
    if ACR_OPEN in sentence or ACR_CLOSE in sentence:
        raise ValueError("The original sentence must not contain reserved target markers")
    offsets = []
    for field in ("span_start", "span_end"):
        value = row.get(field)
        if isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
            value = int(value)
        if type(value) is not int:
            raise ValueError(f"{field} must be an explicit integer character offset")
        offsets.append(value)
    start, end = offsets
    if not 0 <= start < end <= len(sentence):
        raise ValueError("Target span is outside the original sentence")
    if sentence[start:end] != target:
        raise ValueError("Target span does not match target_raw exactly (including prefix)")
    return start, end


def candidates_for(row: dict) -> list[str]:
    value = row.get("candidates")
    if not isinstance(value, str):
        raise ValueError("candidates must be a pipe-separated string")
    candidates = [candidate.strip() for candidate in value.split("|")]
    if not candidates or any(not candidate for candidate in candidates):
        raise ValueError("Candidate inventory contains an empty candidate")
    if any(ACR_OPEN in c or ACR_CLOSE in c for c in candidates):
        raise ValueError("Candidates must not contain reserved target markers")
    return candidates


def input_identity(rows: list[dict]) -> dict:
    """Bind ordered IDs and the complete supplied rows, without saving their text."""
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {"item_ids": [row["item_id"] for row in rows],
            "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest()}


class PreparedPairs(list):
    """Validated pairs with the original item identities for checkpoint provenance."""

    def __init__(self, values, rows):
        super().__init__(values)
        self.rows = deepcopy(rows)
        self.identity = input_identity(rows)

    def verify(self):
        rebuilt = build_pairs(self.rows)
        if rebuilt != self or rebuilt.identity != self.identity:
            raise ValueError("Prepared pairs or input identity changed after validation")


def build_pairs(rows: list[dict]) -> PreparedPairs:
    """Strict training input: explicit target and exactly one positive per item.

    Singleton training remains unsupported, with an explicit error instead of a
    silent skip. Prediction has a separate record-preserving path.
    """
    validate_ids(rows)
    pairs = []
    for row in rows:
        try:
            span = explicit_span(row)
            candidates = candidates_for(row)
            gold = row.get("gold_expansion")
            if not isinstance(gold, str) or not gold.strip():
                raise ValueError("A nonempty gold_expansion is required for training")
            gold = gold.strip()
            if sum(candidate == gold for candidate in candidates) != 1:
                raise ValueError("Training item requires exactly one positive candidate")
            if len(candidates) < MIN_CANDIDATES:
                raise ValueError("Training requires at least two candidates; singleton policy is pending")
            if len(set(candidates)) != len(candidates):
                raise ValueError("Duplicate candidate entries are not supported")
            marked = mark_span(row["sentence"], span)
            pairs.extend((marked, candidate, int(candidate == gold)) for candidate in candidates)
        except ValueError as error:
            raise ValueError(f"Item {row['item_id']!r}: {error}") from error
    return PreparedPairs(pairs, rows)


def describe_skips(rows: list[dict]) -> dict:
    """Validate the full input without silently dropping training rows."""
    build_pairs(rows)
    return {"total": len(rows), "usable": len(rows), "skipped": 0}
