"""Read CSV rows without changing their columns, values or order."""
import csv
from pathlib import Path


def load_rows(path: str | Path) -> list[dict]:
    """Read UTF-8 CSV, accepting an optional byte-order mark."""
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))
