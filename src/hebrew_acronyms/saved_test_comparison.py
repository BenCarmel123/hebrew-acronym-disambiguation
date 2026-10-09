"""Read-only comparison of identified staged test runs and their expenditure."""
from collections import Counter
from html import escape
import hashlib
import json
import math
from pathlib import Path

from hebrew_acronyms import test_evaluation as evaluation


def _finish_reason(record):
    metadata = record["response_metadata"]
    return metadata.get("finish_reason") or (metadata.get("response_metadata") or {}).get("done_reason", "not reported")


def _test_signature(identity):
    rows = identity["rows"]
    item_ids = [row["item_id"] for row in rows]
    if identity["cohort"] != "full_test" or len(set(item_ids)) != 395 or len(rows) != 395:
        raise ValueError("Comparison requires the complete 395-item test cohort")
    requests = identity["requests"]
    keys = [(r["item_id"], r["task"]) for r in requests]
    expected = {(item, task) for item in item_ids for task in ("generate", "select")}
    if len(keys) != 790 or set(keys) != expected:
        raise ValueError("Test requests must cover each item and task exactly once")
    for request in requests:
        if hashlib.sha256(request["prompt"].encode()).hexdigest() != request["prompt_sha256"]:
            raise ValueError("Prompt hash mismatch")
    return {
        key: identity[key] for key in
        ("protocol", "source_sha256", "rows", "seed", "target_policy", "target_occurrences")
    } | {"requests": [{k: r[k] for k in
                      ("item_id", "task", "prompt", "prompt_sha256", "shown_order")}
                     for r in requests]}


def compare_saved_tests(session_sources, expected_test_runs):
    """Read chronological (directory, session hash) pairs; never call models or write.

    expected_test_runs maps each selected system to its original full-test run ID.
    Session prior spending must carry the preceding total exactly once. The first
    session's prior allocation remains explicit rather than being reconstructed.
    """
    if not session_sources or not expected_test_runs:
        raise ValueError("Provide identified sessions and expected test run IDs")
    roots = [Path(path).resolve() for path, _ in session_sources]
    if len(set(roots)) != len(roots):
        raise ValueError("Duplicate session directory")
    sessions, models, metrics, failures, pilots, costs = [], [], [], [], [], []
    seen_sessions, seen_runs, seen_attempts = set(), set(), set()
    found_tests, signature, reference_identity = {}, None, None
    total, initial_prior, measured, uncertain = None, None, 0.0, 0.0
    baseline_paths = []
    for root, (_, expected_session) in zip(roots, session_sources):
        session = json.loads((root / "session.json").read_text())
        identity = session["identity"]
        if session["identity_sha256"] != expected_session or evaluation._hash(identity) != expected_session:
            raise ValueError("Session identity mismatch")
        if expected_session in seen_sessions:
            raise ValueError("Duplicate session identity")
        seen_sessions.add(expected_session)
        if total is None:
            total = initial_prior = identity["prior_spend_ils"]
        elif not math.isclose(identity["prior_spend_ils"], total, rel_tol=0, abs_tol=1e-9):
            raise ValueError("Prior expenditure does not match the preceding cumulative total")
        if reference_identity is None:
            reference_identity = identity
        elif any(identity[k] != reference_identity[k] for k in
                 ("code_revision", "code_sha256", "rates", "reserves", "ils_per_usd", "seed", "max_attempts")):
            raise ValueError("Collection protocol or accounting settings differ")
        # Reuse the collection scorer only when its implementation is unchanged.
        for name in ("test_evaluation.py", "models/common/eval.py", "models/common/pairs.py"):
            current = Path(evaluation.__file__).parent / name
            if hashlib.sha256(current.read_bytes()).hexdigest() != identity["code_sha256"][name]:
                raise ValueError("The collection scoring implementation has changed")
        session_measured, session_uncertain = 0.0, 0.0
        manifests = sorted(root.glob("*/*/manifest.json"))
        if set(root.rglob("manifest.json")) != set(manifests):
            raise ValueError("Unrecognized run directory")
        if any(p.parent / "manifest.json" not in manifests for p in root.rglob("attempts*.jsonl")):
            raise ValueError("Orphan attempt journal")
        for path in manifests:
            name, cohort = path.parent.parent.name, path.parent.name.replace("-", "_")
            manifest = evaluation._load_manifest(path.parent)
            run = manifest["identity"]
            if (name not in identity["systems"] or cohort not in ("dev_pilot", "full_test")
                    or run["cohort"] != cohort or len(run["systems"]) != 1
                    or run["systems"][0]["name"] != name
                    or run["metadata"]["session_identity"] != expected_session
                    or run["code_revision"] != identity["code_revision"]
                    or run["code_sha256"] != identity["code_sha256"]
                    or run["source_sha256"] != identity["sources"][cohort]["sha256"]
                    or run["rates_usd_per_million"] != {name: identity["rates"][name]}
                    or run["reserve_per_call_usd"] != {name: identity["reserves"][name]}):
                raise ValueError("Run does not match its collection session")
            if manifest["run_id"] in seen_runs:
                raise ValueError("Duplicate run ID")
            seen_runs.add(manifest["run_id"])
            starts, _, tails = evaluation._read_events(path.parent, manifest)
            if seen_attempts.intersection(starts):
                raise ValueError("Duplicate attempt ID across runs")
            seen_attempts.update(starts)
            if tails:
                raise ValueError("Inspect truncated journals before comparing saved results")
            summary = evaluation.summarize_evaluation(path.parent)
            reserve = summary["reserved_usd"]
            session_measured += summary["charged_or_reserved_usd"] - reserve
            session_uncertain += reserve
            system = run["systems"][0]
            common = {"system": name, "model": system["model"], "run_id": manifest["run_id"],
                      "session": root.name if root.name != "extracted" else root.parent.name,
                      "collection_revision": run["code_revision"], "identity_sha256": manifest["identity_sha256"]}
            if cohort == "dev_pilot":
                pilots.append(common | {"completed": summary["n_completed"], "responses": summary["n_records"],
                    "calls": summary["n_calls"], "ambiguous": summary["n_ambiguous"],
                    "unverified": summary["n_identity_unverified"], "seconds": summary["timings_seconds"][name],
                    "token_cost_usd": summary["charged_or_reserved_usd"] - reserve,
                    "uncertain_usd": reserve})
                continue
            if expected_test_runs.get(name) != manifest["run_id"] or name in found_tests:
                raise ValueError("Unexpected or duplicate test system/run ID")
            current_signature = _test_signature(run)
            if signature is not None and signature != current_signature:
                raise ValueError("Test items, protocol, prompts or candidate orders differ")
            signature = current_signature
            found_tests[name] = manifest["run_id"]
            returned = sorted({r["response_metadata"].get("model_version") or
                               r["response_metadata"].get("model") or "not reported"
                               for r in summary["records"] if r["status"] == "response_received"})
            models.append(common | {"settings": system["settings"], "model_identity": run["metadata"]["model_identity"],
                "returned_models": returned, "calls": summary["n_calls"], "usage": summary["usage"][name],
                "seconds": summary["timings_seconds"][name], "unverified": summary["n_identity_unverified"],
                "finish_reasons": dict(Counter(_finish_reason(r) for r in summary["records"])),
                "manifest_path": str(path), "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            for group in summary["by_system_task"]:
                records = [r for r in summary["records"] if r["task"] == group["task"]]
                metrics.append(group | {"run_id": manifest["run_id"], "n_correct": sum(r["correct"] for r in records)})
            for record in summary["records"]:
                if record["status"] != "response_received":
                    failures.append({k: record[k] for k in ("system", "task", "item_id", "status", "attempts", "response")}
                                    | {"run_id": manifest["run_id"],
                                       "finish_reason": _finish_reason(record)})
        conversion = identity["ils_per_usd"]
        measured += session_measured * conversion
        uncertain += session_uncertain * conversion
        total += (session_measured + session_uncertain) * conversion
        costs.append({"session": common["session"] if manifests else root.name,
                      "prior_ils": identity["prior_spend_ils"], "new_token_cost_ils": session_measured * conversion,
                      "new_uncertain_ils": session_uncertain * conversion, "cumulative_ils": total})
        sessions.append({"path": str(root), "identity_sha256": expected_session,
                         "collection_revision": identity["code_revision"]})
        if (root / "baselines/baselines.json").exists():
            baseline_paths.append(root / "baselines/baselines.json")
    if found_tests != expected_test_runs:
        raise ValueError("Missing expected full-test run")
    if not baseline_paths:
        raise ValueError("Saved test baselines are required")
    baselines = json.loads(baseline_paths[0].read_text())
    rows = {r["item_id"]: r for r in signature["rows"]}
    details = baselines["details"]
    if (baselines["n_items"] != 395 or len(details) != 395
            or {r["item_id"] for r in details} != set(rows)
            or baselines["code_revision"] != reference_identity["code_revision"]
            or next(x["sha256"] for x in baselines["inputs"] if x["name"] == "test_items.csv") != signature["source_sha256"]
            or any(r["gold"] != rows[r["item_id"]]["gold_expansion"].strip() for r in details)):
        raise ValueError("Baseline test cohort or provenance differs")
    for path in baseline_paths[1:]:
        if json.loads(path.read_text()) != baselines:
            raise ValueError("Saved baseline results differ")
    return {"sessions": sessions, "models": models, "metrics": metrics, "failures": failures,
            "pilots": pilots, "cost_sessions": costs, "baselines": baselines,
            "baseline_path": str(baseline_paths[0]), "test_source_sha256": signature["source_sha256"],
            "cost": {"initial_prior_allocation_ils": initial_prior, "token_cost_ils": measured,
                     "uncertain_attempt_allowances_ils": uncertain, "total_accounted_ils": total},
            "unique_attempts": len(seen_attempts), "test_runs": found_tests}


def result_table(rows, columns):
    """Small escaped HTML table; columns maps field names to reader-facing labels."""
    def text(value):
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, float):
            value = f"{value:.9g}"
        return escape(str(value))
    head = "".join(f"<th>{text(label)}</th>" for label in columns.values())
    body = "".join("<tr>" + "".join(f'<td style="overflow-wrap:anywhere">{text(row.get(key, "—"))}</td>'
                                   for key in columns) + "</tr>" for row in rows)
    return f'<table style="width:100%;text-align:left"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'
