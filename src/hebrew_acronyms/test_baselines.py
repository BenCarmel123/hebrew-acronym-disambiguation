"""Save identified deterministic reference baselines without model execution."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from hebrew_acronyms.models.baselines import evaluate, load_signals
from hebrew_acronyms.models.common.pairs import load_rows, validate_ids


def run_test_baselines(input_path, candidates_path, output_dir, *, code_revision):
    """Preserve every item and reject incompatible reuse; never overwrite history."""
    if not isinstance(code_revision, str) or len(code_revision) != 40:
        raise ValueError("An exact 40-character code revision is required")
    paths = [Path(input_path), Path(candidates_path)]
    rows = load_rows(str(paths[0]))
    validate_ids(rows)
    ranks, mined = load_signals(str(paths[1]))
    details = []
    for row in rows:
        candidates = [c.strip() for c in row["candidates"].split("|")]
        gold = row["gold_expansion"].strip()
        if len(candidates) < 2 or any(not c for c in candidates) or not gold:
            raise ValueError("Baseline input would be skipped by historical evaluator")
        acronym = row["acronym"]
        frequent = min(candidates, key=lambda c: ranks.get((acronym, c), 1 << 30))
        most_mined = min(candidates, key=lambda c: -mined.get((acronym, c), 0))
        details.append({"item_id": row["item_id"], "source": row.get("source"),
                        "n_candidates": len(candidates), "gold": gold,
                        "random_expected_correct": 1 / len(candidates),
                        "most_frequent_prediction": frequent,
                        "most_frequent_correct": frequent == gold,
                        "most_mined_prediction": most_mined,
                        "most_mined_correct": most_mined == gold,
                        "oracle_correct": gold in candidates})
    result = {"schema_version": 1, "kind": "deterministic_baselines",
              "code_revision": code_revision,
              "inputs": [{"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                         for p in paths],
              "n_items": len(rows), "summary": evaluate(rows, ranks, mined),
              "details": details,
              "note": "Random is the analytical expectation, not a sampled prediction. "
                      "No encoder or LLM ran. Historical tie breaking is retained."}
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "baselines.json"
    if destination.exists():
        if json.loads(destination.read_text()) != result:
            raise ValueError("Existing baseline artifact differs; choose a new output directory")
        return result
    temporary = directory / ".baselines.tmp"
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return result
