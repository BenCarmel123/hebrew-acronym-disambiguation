"""The scored test cohort: 381 items, without 14 that share a Knesset document with training.

The 14 items come from four Knesset protocols that also supply training items.
`data/splits/test_items.csv` holds the 381 scored items, and new runs collect only
those. Runs saved before the removal cover all 395 items and keep their own copy of
the rows; their records for these 14 items are kept as evidence but never scored.
Candidate order is seeded per item ID, so the scored items have identical prompts
in both cohorts.
"""
from __future__ import annotations

DOCUMENT_OVERLAP_IDS = frozenset({
    "kn-0116", "kn-0127", "kn3-0017", "kn3-0081",
    "kn4-0279", "kn4-0280", "kn4-0281", "kn4-0282", "kn4-0283",
    "kn4-0284", "kn4-0285", "kn4-0286", "kn4-0287", "kn4-0288",
})
COLLECTED_ITEMS = 395
SCORED_ITEMS = COLLECTED_ITEMS - len(DOCUMENT_OVERLAP_IDS)


def is_scored(item_id: str) -> bool:
    return item_id not in DOCUMENT_OVERLAP_IDS


def check_scored_cohort(item_ids) -> None:
    """Require exactly the 381 scored items, none of them excluded."""
    item_ids = list(item_ids)
    if (len(item_ids) != SCORED_ITEMS or len(set(item_ids)) != SCORED_ITEMS
            or DOCUMENT_OVERLAP_IDS.intersection(item_ids)):
        raise ValueError(f"The test cohort must be the {SCORED_ITEMS} items without document overlap")


def scored_ids(item_ids) -> list[str]:
    """Scored item IDs, in order, from a saved 395-item cohort or a new 381-item cohort."""
    item_ids = list(item_ids)
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("Duplicate test item IDs")
    if len(item_ids) == COLLECTED_ITEMS and DOCUMENT_OVERLAP_IDS <= set(item_ids):
        kept = [item_id for item_id in item_ids if is_scored(item_id)]
    else:
        kept = item_ids
    check_scored_cohort(kept)
    return kept
