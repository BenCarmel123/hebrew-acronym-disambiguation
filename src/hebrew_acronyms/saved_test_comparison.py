"""Read-only comparison of identified staged test runs and their expenditure."""
from collections import Counter
from html import escape
import hashlib
import json
import math
from pathlib import Path

from hebrew_acronyms import test_evaluation as evaluation
from hebrew_acronyms.test_cohort import SCORED_ITEMS, is_scored, scored_ids


# Audited collector implementations: a319f9d and the xAI/Qwen14 extension.
# The extension adds provider dispatch/usage only; existing scoring is unchanged.
# The third requires the 381-item scored cohort for new full tests and uses it for
# cost projection; prompts, parsing and scoring are unchanged.
COMPATIBLE_COLLECTOR_SHA256 = {
    "1434bedc4dac4a79e681d3178d8c2eb2f408ba12d3931255c09f90ca3f8abbff",
    "480eaf990037d51f3fedb153a6558367f5ec94ddbdb1d7a207e116424e6cb3de",
    "3205040d9db8f7a3e7a2a5eea2ca1d4c54d74b7e35545fe3bbc8da56a34eae1c",
}


def _finish_reason(record):
    metadata = record["response_metadata"]
    return metadata.get("finish_reason") or (metadata.get("response_metadata") or {}).get("done_reason", "not reported")


def _test_signature(identity):
    """Scored rows and prompts of a saved 395-item or a new 381-item full test.

    Both cohorts must agree on these. Source hashes differ between them, so they
    are checked separately against the session, not as part of the signature.
    """
    rows = identity["rows"]
    item_ids = [row["item_id"] for row in rows]
    if identity["cohort"] != "full_test" or len(set(item_ids)) != len(rows):
        raise ValueError("Comparison requires a complete full-test cohort")
    scored = set(scored_ids(item_ids))
    requests = identity["requests"]
    keys = [(r["item_id"], r["task"]) for r in requests]
    expected = {(item, task) for item in item_ids for task in ("generate", "select")}
    if len(keys) != 2 * len(rows) or set(keys) != expected:
        raise ValueError("Test requests must cover each item and task exactly once")
    for request in requests:
        if hashlib.sha256(request["prompt"].encode()).hexdigest() != request["prompt_sha256"]:
            raise ValueError("Prompt hash mismatch")
    return {key: identity[key] for key in ("protocol", "seed", "target_policy")} | {
        "rows": [row for row in rows if row["item_id"] in scored],
        "target_occurrences": {k: v for k, v in identity["target_occurrences"].items() if k in scored},
        "requests": [{k: r[k] for k in ("item_id", "task", "prompt", "prompt_sha256", "shown_order")}
                     for r in requests if r["item_id"] in scored]}


def _scored_metric(group, records):
    """Recompute one system/task group over the scored items only."""
    subset = [r for r in records if r["task"] == group["task"] and is_scored(r["item_id"])]
    return {"system": group["system"], "task": group["task"], "n_items": len(subset),
            "n_completed": sum(r["status"] == "response_received" for r in subset),
            "status_counts": dict(Counter(r["status"] for r in subset)),
            "accuracy": sum(r["correct"] for r in subset) / len(subset), "score": group["score"],
            "n_valid": sum(r["valid"] for r in subset), "n_correct": sum(r["correct"] for r in subset)}


def scored_baselines(baselines):
    """Saved deterministic baselines restricted to the scored items, rounded as saved."""
    details = [d for d in baselines["details"] if is_scored(d["item_id"])]
    n = len(details)
    summary = {"n_items": n, "mean_candidates": round(sum(d["n_candidates"] for d in details) / n, 2),
               "random": round(sum(d["random_expected_correct"] for d in details) / n, 4)}
    for name in ("most_frequent", "most_mined", "oracle"):
        summary[name] = round(sum(d[f"{name}_correct"] for d in details) / n, 4)
    return baselines | {"n_items": n, "summary": summary, "details": details,
                        "collected_n_items": baselines["n_items"]}


def compare_saved_tests(session_sources, expected_test_runs, *, diagnostic_sources=(), continuation_sources=()):
    """Read chronological (directory, session hash) pairs; never call models or write.

    expected_test_runs maps each selected system to its original full-test run ID.
    Session prior spending must carry the preceding total exactly once. The first
    session's prior allocation remains explicit rather than being reconstructed.
    Explicit continuations may add new runs to an earlier session only when all
    previously counted manifests and attempt journals are byte-identical.
    """
    if not session_sources or not expected_test_runs:
        raise ValueError("Provide identified sessions and expected test run IDs")
    # One separately journalled xAI HTTP diagnostic was made between sessions.
    diagnostics = []
    seen_diagnostics = set()
    for directory, start_hash, finish_hash in diagnostic_sources:
        directory = Path(directory)
        start_bytes = (directory / "diagnostic.jsonl").read_bytes()
        finish_bytes = (directory / "recovered-finish.json").read_bytes()
        if (hashlib.sha256(start_bytes).hexdigest() != start_hash
                or hashlib.sha256(finish_bytes).hexdigest() != finish_hash):
            raise ValueError("Diagnostic evidence hash mismatch")
        if (start_hash, finish_hash) in seen_diagnostics:
            raise ValueError("Duplicate diagnostic")
        seen_diagnostics.add((start_hash, finish_hash))
        start, finish = json.loads(start_bytes), json.loads(finish_bytes)
        result = finish["result"]
        reserve = start["reserved_ils"]
        if (start["event"] != "start" or start["max_requests"] != 1
                or finish["event"] != "finish_recovered_from_kernel"
                or result["attempts"] != 1 or result["http_status"] != 400
                or result["usage_metadata"] is not None
                or not isinstance(reserve, (int, float)) or isinstance(reserve, bool)
                or not math.isfinite(reserve) or reserve <= 0
                or not math.isclose(finish["accounted_ils_including_reserve"],
                                    start["prior_accounted_ils"] + reserve, abs_tol=1e-9)):
            raise ValueError("Unsupported diagnostic accounting evidence")
        diagnostics.append((directory, start))
    all_sources = [*session_sources, *continuation_sources]
    roots = [Path(path).resolve() for path, _ in all_sources]
    if len(set(roots)) != len(roots):
        raise ValueError("Duplicate session directory")
    sessions, models, metrics, failures, pilots, costs = [], [], [], [], [], []
    seen_sessions, seen_runs, seen_attempts = set(), set(), set()
    found_tests, signature, reference_identity, test_sources = {}, None, None, set()
    total, initial_prior, measured, uncertain = None, None, 0.0, 0.0
    baseline_paths, original_roots = [], {}
    for index, (root, (_, expected_session)) in enumerate(zip(roots, all_sources)):
        continuation = index >= len(session_sources)
        old_run_ids = {}
        session = json.loads((root / "session.json").read_text())
        identity = session["identity"]
        if session["identity_sha256"] != expected_session or evaluation._hash(identity) != expected_session:
            raise ValueError("Session identity mismatch")
        if continuation:
            original = original_roots.get(expected_session)
            if original is None:
                raise ValueError("Continuation requires an earlier identified snapshot")
            if (root / "session.json").read_bytes() != (original / "session.json").read_bytes():
                raise ValueError("Continuation changed its original session")
            for old in [*original.glob("*/*/manifest.json"), *original.rglob("attempts*.jsonl")]:
                new = root / old.relative_to(original)
                if not new.is_file() or new.read_bytes() != old.read_bytes():
                    raise ValueError("Continuation changed previously counted evidence")
                if old.name == "manifest.json":
                    old_run_ids[json.loads(old.read_text())["run_id"]] = old.relative_to(original)
                    if ({q.name for q in old.parent.glob("attempts*.jsonl")}
                            != {q.name for q in new.parent.glob("attempts*.jsonl")}):
                        raise ValueError("Continuation changed previously counted journals")
            new_manifests = [p for p in root.glob("*/*/manifest.json")
                             if json.loads(p.read_text())["run_id"] not in old_run_ids]
            if not new_manifests:
                raise ValueError("Continuation contains no new run")
        elif expected_session in seen_sessions:
            raise ValueError("Duplicate session identity")
        else:
            original_roots[expected_session] = root
        seen_sessions.add(expected_session)
        if total is None:
            total = initial_prior = identity["prior_spend_ils"]
        elif not continuation and not math.isclose(identity["prior_spend_ils"], total, rel_tol=0, abs_tol=1e-9):
            raise ValueError("Prior expenditure does not match the preceding cumulative total")
        if reference_identity is None:
            reference_identity = identity
        elif any(identity[k] != reference_identity[k] for k in
                 ("ils_per_usd", "seed", "max_attempts")):
            raise ValueError("Collection protocol or accounting settings differ")
        for name in set(identity["systems"]) & set(reference_identity["systems"]):
            if any(identity[field][name] != reference_identity[field][name]
                   for field in ("rates", "reserves")):
                raise ValueError("Shared system accounting settings differ")
        current_collector = hashlib.sha256(Path(evaluation.__file__).read_bytes()).hexdigest()
        if (current_collector not in COMPATIBLE_COLLECTOR_SHA256
                or identity["code_sha256"]["test_evaluation.py"] not in COMPATIBLE_COLLECTOR_SHA256):
            raise ValueError("Unreviewed collection implementation")
        # Prompts, parsing, pair construction and scoring must remain byte-identical.
        for name in ("models/common/eval.py", "models/common/pairs.py"):
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
            if continuation and manifest["run_id"] in old_run_ids:
                if path.relative_to(root) != old_run_ids[manifest["run_id"]]:
                    raise ValueError("Duplicate run ID in continuation")
                continue
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
            test_sources.add(run["source_sha256"])
            found_tests[name] = manifest["run_id"]
            returned = sorted({r["response_metadata"].get("model_version") or
                               r["response_metadata"].get("model") or "not reported"
                               for r in summary["records"] if r["status"] == "response_received"})
            models.append(common | {"settings": system["settings"], "model_identity": run["metadata"]["model_identity"],
                "returned_models": returned, "calls": summary["n_calls"], "usage": summary["usage"][name],
                "seconds": summary["timings_seconds"][name], "unverified": summary["n_identity_unverified"],
                "finish_reasons": dict(Counter(_finish_reason(r) for r in summary["records"])),
                "manifest_path": str(path), "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            # Calls, usage and cost above cover every collected item; scores cover the scored items.
            for group in summary["by_system_task"]:
                metrics.append(_scored_metric(group, summary["records"]) | {"run_id": manifest["run_id"]})
            for record in summary["records"]:
                if record["status"] != "response_received" and is_scored(record["item_id"]):
                    failures.append({k: record[k] for k in ("system", "task", "item_id", "status", "attempts", "response")}
                                    | {"run_id": manifest["run_id"],
                                       "finish_reason": _finish_reason(record)})
        conversion = identity["ils_per_usd"]
        measured += session_measured * conversion
        uncertain += session_uncertain * conversion
        total += (session_measured + session_uncertain) * conversion
        costs.append({"session": common["session"] if manifests else root.name,
                      "prior_ils": identity["prior_spend_ils"], "new_token_cost_ils": session_measured * conversion,
                      "new_uncertain_ils": session_uncertain * conversion, "cumulative_ils": total, "continuation": continuation})
        sessions.append({"path": str(root), "identity_sha256": expected_session,
                         "collection_revision": identity["code_revision"]})
        if (root / "baselines/baselines.json").exists():
            baseline_paths.append(root / "baselines/baselines.json")
        session_name = root.name if root.name != "extracted" else root.parent.name
        for directory, start in list(diagnostics):
            if start["source_session"] != session_name:
                continue
            if (start["source_revision"] != identity["code_revision"]
                    or not math.isclose(start["prior_accounted_ils"], total, rel_tol=0, abs_tol=1e-9)):
                raise ValueError("Diagnostic prior or collection revision mismatch")
            reserve = start["reserved_ils"]
            costs.append({"session": session_name + "/xai-http-diagnostic", "prior_ils": total,
                          "new_token_cost_ils": 0.0, "new_uncertain_ils": reserve,
                          "cumulative_ils": total + reserve})
            total += reserve
            uncertain += reserve
            diagnostics.remove((directory, start))
    if diagnostics:
        raise ValueError("Diagnostic source session not found")
    if found_tests != expected_test_runs:
        raise ValueError("Missing expected full-test run")
    if not baseline_paths:
        raise ValueError("Saved test baselines are required")
    baselines = json.loads(baseline_paths[0].read_text())
    rows = {r["item_id"]: r for r in signature["rows"]}
    details = baselines["details"]
    if (baselines["n_items"] != len(details)
            or scored_ids([r["item_id"] for r in details]) != list(rows)
            or baselines["code_revision"] != reference_identity["code_revision"]
            or next(x["sha256"] for x in baselines["inputs"] if x["name"] == "test_items.csv") not in test_sources
            or any(r["gold"] != rows[r["item_id"]]["gold_expansion"].strip() for r in details if r["item_id"] in rows)):
        raise ValueError("Baseline test cohort or provenance differs")
    for path in baseline_paths[1:]:
        if json.loads(path.read_text()) != baselines:
            raise ValueError("Saved baseline results differ")
    baselines = scored_baselines(baselines)
    return {"sessions": sessions, "models": models, "metrics": metrics, "failures": failures,
            "pilots": pilots, "cost_sessions": costs, "baselines": baselines,
            "baseline_path": str(baseline_paths[0]), "test_source_sha256": sorted(test_sources),
            "n_scored_items": SCORED_ITEMS,
            "cost": {"initial_prior_allocation_ils": initial_prior, "token_cost_ils": measured,
                     "uncertain_attempt_allowances_ils": uncertain, "total_accounted_ils": total},
            "unique_attempts": len(seen_attempts), "diagnostic_requests": len(seen_diagnostics), "test_runs": found_tests}


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
