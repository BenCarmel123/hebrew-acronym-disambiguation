"""Post-hoc uniform reading of saved LLM labels, preserving historical scores.

This analysis does not change the collector or its strict parser. Only a completed answer accepted by
the historical identity checks can receive credit; all retained items stay in denominators.
"""
from collections import Counter
import json
from pathlib import Path
import re

from . import test_evaluation as evaluation
from .dictalm_saved_audit import audit_saved_dictalm
from .test_cohort import is_scored

RULE = "posthoc_leading_candidate_label_v1"
LLMS = ("qwen", "qwen14", "dictalm", "gemini", "openai", "anthropic", "xai")


def leading_label_result(record, gold):
    """Read the initial uppercase label token, permitting subsequent text.

    Labels are A..Z, AA.. and must be in range. The boundary rejects prefixes of
    longer words/digits. A technical failure or unverified identity earns no credit,
    even if its partial response begins with the correct label.
    """
    if record["task"] != "select":
        raise ValueError("Leading-label parsing applies only to selection")
    match = re.match(r"^\s*([A-Z]+)\b", record["response"])
    eligible = (record["status"] == "response_received"
                and record.get("response_metadata", {}).get("identity_status") != "unverified")
    index = (evaluation.scoring.parse_letter_choice(match[1], len(record["shown_order"]))
             if match and eligible else None)
    candidate = record["shown_order"][index] if index is not None else None
    return {"leading_label_candidate": candidate, "leading_label_valid": index is not None,
            "leading_label_correct": index is not None and candidate == gold.strip()}


def derive_saved_comparison(root, saved=None):
    """Load verified journals, apply one rule to seven LLMs, retain strict counts.

    Returned rows are new objects; source responses, original correct/valid fields,
    annotations and the historical six-model exports are never overwritten.
    """
    from .saved_analysis import load_saved_analysis
    root = Path(root)
    saved = load_saved_analysis(root) if saved is None else saved
    audit = audit_saved_dictalm(root / "saved-results", saved["original_manifest"])
    records = []
    for model in saved["comparison"]["models"]:
        path = Path(model["manifest_path"])
        manifest = json.loads(path.read_text(encoding="utf-8"))
        gold = {r["item_id"]: r["gold_expansion"] for r in manifest["identity"]["rows"]}
        for record in evaluation.summarize_evaluation(path.parent)["records"]:
            if is_scored(record["item_id"]):
                records.append(record | {"gold": gold[record["item_id"]], "run_id": manifest["run_id"]})
    records.extend(audit["records"])
    selection = []
    reference_ids = None
    metrics = []
    historical = saved["metrics"] + audit["metrics"]
    for system in LLMS:
        subset = [r for r in records if r["system"] == system and r["task"] == "select"]
        ids = {r["item_id"] for r in subset}
        if len(subset) != 381 or len(ids) != 381 or (reference_ids is not None and ids != reference_ids):
            raise ValueError("Uniform comparison requires the same 381 unique items per LLM")
        reference_ids = ids
        derived = [r | leading_label_result(r, r["gold"]) for r in subset]
        selection.extend(derived)
        strict = next(m for m in historical if m["system"] == system and m["task"] == "select")
        n_correct = sum(r["leading_label_correct"] for r in derived)
        metrics.append({"system": system, "task": "select", "n_items": len(derived),
                        "n_completed": sum(r["status"] == "response_received" for r in derived),
                        "status_counts": dict(Counter(r["status"] for r in derived)),
                        "accuracy": n_correct / len(derived), "score": RULE,
                        "n_valid": sum(r["leading_label_valid"] for r in derived),
                        "n_correct": n_correct, "run_id": strict["run_id"],
                        "strict_n_correct": strict["n_correct"], "strict_n_valid": strict["n_valid"],
                        "n_changed_correct": sum(r["correct"] != r["leading_label_correct"] for r in derived)})
        metrics.append(dict(next(m for m in historical if m["system"] == system and m["task"] == "generate")))
    metrics.append(dict(saved["encoder"]["metric"]))
    return {"metrics": metrics, "selection_records": selection, "dictalm_audit": audit,
            "run_ids": saved["run_ids"] | {"dictalm": audit["run_id"]},
            "rule": RULE, "analysis_status": "post-hoc saved-response reanalysis"}
