"""Bind the historical untrained similarity predictions to the scored cohort.

This reads the existing CSV; it neither runs DictaBERT nor reconstructs missing
collection metadata. The CSV has no sentence snapshot or identified run manifest.
"""
import csv
import hashlib
import io
from pathlib import Path

from .test_cohort import check_scored_cohort


SOURCE_PATH = "results/dictabert/test_details.csv"
SOURCE_COMMIT = "4a3d0051f4ae299f492ab7263cd57bfacd1e8793"
SOURCE_SHA256 = "f0b7ba4f81d79fb734199117d09445b4d3516a8d3b102d7ee7a60d1eb9589873"


def _validate_records(records, cohort):
    check_scored_cohort(row["item_id"] for row in records)
    check_scored_cohort(row["item_id"] for row in cohort)
    by_id = {row["item_id"]: row for row in cohort}
    if {row["item_id"] for row in records} != set(by_id):
        raise ValueError("Similarity predictions differ from the scored cohort IDs")
    for record in records:
        row = by_id[record["item_id"]]
        if any(record[old] != row[current] for old, current in (
                ("acronym", "acronym"), ("gold", "gold_expansion"),
                ("candidates", "candidates"))):
            raise ValueError("Similarity gold, acronym or candidates differ from the scored cohort")
        candidates = [candidate.strip() for candidate in row["candidates"].split("|")]
        if (record["valid"] != "True" or record["response"] not in candidates
                or record["shown_order"] != record["candidates"]):
            raise ValueError("Similarity saved candidate prediction or order is invalid")
        if record["correct"] != str(record["response"] == record["gold"]):
            raise ValueError("Similarity saved score differs from exact candidate equality")


def load_saved_similarity(repo_root, cohort_path=None):
    """Validate the byte-identified CSV and return its separate historical metric.

    Source commit identifies the preserved 381-item CSV, not a model run. Completion
    here means a saved candidate prediction; this source cannot establish technical
    attempt/failure counts, model revision, collection settings or elapsed time.
    """
    root = Path(repo_root)
    source = (root / SOURCE_PATH).read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("Similarity predictions differ from the identified saved CSV")
    records = list(csv.DictReader(io.StringIO(source.decode("utf-8-sig"))))
    cohort_path = Path(cohort_path) if cohort_path is not None else root / "data/splits/test_items.csv"
    with cohort_path.open(encoding="utf-8-sig", newline="") as handle:
        cohort = list(csv.DictReader(handle))
    _validate_records(records, cohort)
    correct = sum(row["correct"] == "True" for row in records)
    metric = {
        "system": "dictabert_similarity", "baseline": "dictabert_similarity",
        "task": "select", "run_id": None,
        "n_items": len(records), "n_completed": len(records), "n_valid": len(records),
        "n_correct": correct, "accuracy": correct / len(records),
        "score": "exact_candidate_match",
        "status_counts": {"saved_candidate_prediction": len(records)},
        "technical_failures": None, "source_path": SOURCE_PATH,
        "source_commit": SOURCE_COMMIT, "source_sha256": SOURCE_SHA256,
    }
    provenance = {
        "path": SOURCE_PATH, "commit": SOURCE_COMMIT, "sha256": SOURCE_SHA256,
        "bytes": len(source), "source_kind": "historical_prediction_csv",
        "limitation": "No identified run manifest, sentence snapshot, model revision, "
                      "collection settings, timing or technical attempt log is preserved in this CSV.",
    }
    return {"metric": metric, "records": records, "provenance": provenance}
