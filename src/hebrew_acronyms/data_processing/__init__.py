"""Hebrew acronym disambiguation — bullet frequency counting.

For every page in `קטגוריה:פירושון ראשי תיבות`, list each disambiguation bullet
and how many Hebrew Wikipedia articles contain that phrase. Nothing is filtered;
person senses and initials mismatches are flagged, not removed.
"""
from .common.candidates import BulletRow, summarise, write_csv
from .wikipedia.source import bullets, count_bullets
from .wikipedia.client import WikiAPI

__all__ = ["BulletRow", "bullets", "count_bullets", "summarise", "write_csv", "WikiAPI"]
