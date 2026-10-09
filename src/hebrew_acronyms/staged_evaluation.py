"""Stage existing evaluation journals per system under one frozen spending cap.

There is no second cost ledger: every cap is recomputed from all saved attempts,
including deselected systems and ambiguous requests. Use one writer per session.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

from hebrew_acronyms import test_evaluation as evaluation

SYSTEM_NAMES = ("qwen", "gemini", "openai", "anthropic", "qwen14", "xai")


def prepare_session(output_dir, dev_path, test_path, *, code_revision,
                    prior_spend_ils, rates, reserves, ils_per_usd=4.0):
    """Freeze shared inputs/prices and prior spend; selection is not an identity."""
    if not re.fullmatch(r"[0-9a-f]{40}", code_revision or ""):
        raise ValueError("An exact code commit is required")
    if type(prior_spend_ils) not in {int, float} or not math.isfinite(prior_spend_ils) or not 0 <= prior_spend_ils < 100:
        raise ValueError("Record all earlier expenditure, including failed attempts, below 100 ILS")
    evaluation._positive(ils_per_usd, "ils_per_usd")
    if set(rates) != set(SYSTEM_NAMES) or set(reserves) != set(SYSTEM_NAMES):
        raise ValueError("Rates and reserves must cover all approved systems, including unselected ones")
    for name in SYSTEM_NAMES:
        if set(rates[name]) != {"input", "output"}:
            raise ValueError("Prices need input and output rates")
        for value in [reserves[name], *rates[name].values()]:
            if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
                raise ValueError("Prices and reserves must be finite and nonnegative")
    identity = {"code_revision": code_revision, "code_sha256": evaluation._code_hashes(),
                "budget_ils": 100.0, "prior_spend_ils": prior_spend_ils,
                "ils_per_usd": ils_per_usd, "rates": rates, "reserves": reserves,
                "systems": list(SYSTEM_NAMES), "seed": 42, "max_attempts": 2,
                "sources": {cohort: {"path": str(Path(path).resolve()),
                    "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                    for cohort, path in (("dev_pilot", dev_path), ("full_test", test_path))}}
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with evaluation._lock(root):
        path = root / "session.json"
        if path.exists():
            existing = _session(root)
            if existing["identity"] != identity:
                raise ValueError("Session identity mismatch; selection cannot reset inputs or prior spend")
            return existing
        if any(root.glob("**/manifest.json")) or any(root.glob("**/attempts*.jsonl")):
            raise ValueError("Existing evaluation outputs require their original session; use a new name and carry prior spend")
        session = {"identity": identity, "identity_sha256": evaluation._hash(identity)}
        evaluation._atomic_json(path, session)
        return session


def _session(root):
    session = json.loads((root / "session.json").read_text())
    identity = session["identity"]
    if session["identity_sha256"] != evaluation._hash(identity):
        raise ValueError("Session identity hash mismatch")
    if identity["code_sha256"] != evaluation._code_hashes():
        raise ValueError("Installed package source differs from this session")
    for source in identity["sources"].values():
        if hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("Session input identity mismatch")
    return session


def _directory(root, name, cohort):
    if name not in SYSTEM_NAMES or cohort not in {"dev_pilot", "full_test"}:
        raise ValueError("Unknown system or cohort")
    return root / name / cohort.replace("_", "-")


def _prepare(root, session, system, cohort, metadata):
    identity = session["identity"]
    name = system["name"]
    if name not in SYSTEM_NAMES or system["provider"] != ("qwen" if name == "qwen14" else name):
        raise ValueError("Preserve the approved system/provider names")
    return evaluation.prepare_evaluation(
        identity["sources"][cohort]["path"], _directory(root, name, cohort),
        cohort=cohort, systems=[system], code_revision=identity["code_revision"],
        seed=identity["seed"], max_attempts=identity["max_attempts"],
        max_calls=40 if cohort == "dev_pilot" else 1580,
        metadata={"session_identity": session["identity_sha256"], "model_identity": metadata},
        reserve_per_call_usd={name: identity["reserves"][name]},
        budget_usd=(100 - identity["prior_spend_ils"]) / identity["ils_per_usd"],
        rates_usd_per_million={name: identity["rates"][name]})


def session_summary(output_dir):
    """Report all approved systems, and charge every journal regardless of selection."""
    root = Path(output_dir).resolve()
    session = _session(root)
    identity = session["identity"]
    result = {"systems": {}, "charged_or_reserved_usd": 0.0}
    expected_paths = set()
    for name in SYSTEM_NAMES:
        result["systems"][name] = {}
        for cohort in ("dev_pilot", "full_test"):
            directory = _directory(root, name, cohort)
            expected_paths.add(directory / "manifest.json")
            if not (directory / "manifest.json").exists():
                result["systems"][name][cohort] = {"status": "not_run", "n_calls": 0,
                    "reason": "No saved pilot" if cohort == "dev_pilot" else "No eligible inspected pilot/full run"}
                continue
            manifest = evaluation._load_manifest(directory)
            saved = manifest["identity"]
            if (saved["metadata"].get("session_identity") != session["identity_sha256"]
                    or saved["code_sha256"] != identity["code_sha256"]
                    or saved["cohort"] != cohort or [s["name"] for s in saved["systems"]] != [name]):
                raise ValueError("Saved run does not belong to this session")
            summary = evaluation.summarize_evaluation(directory)
            if any(tail["unfinished_attempt_id"] is None for tail in summary["truncated_journal_tails"]):
                raise ValueError("Truncated journal has unknown spending; inspect before any system continues")
            result["charged_or_reserved_usd"] += summary["charged_or_reserved_usd"]
            result["systems"][name][cohort] = {
                **{key: summary[key] for key in ("n_calls", "n_completed", "n_records", "n_ambiguous", "charged_or_reserved_usd")},
                "provider_states": summary.get("provider_states", {}),
                "status": "complete" if summary["n_completed"] == summary["n_records"] else "incomplete"}
    if set(root.glob("**/manifest.json")) - expected_paths:
        raise ValueError("Unrecognized run directory; refusing to omit it from cumulative spending")
    if any(path.parent / "manifest.json" not in expected_paths or not (path.parent / "manifest.json").exists()
           for path in root.glob("**/attempts*.jsonl")):
        raise ValueError("Orphan attempt journal; spending cannot be reconstructed safely")
    result["total_accounted_ils"] = identity["prior_spend_ils"] + result["charged_or_reserved_usd"] * identity["ils_per_usd"]
    result["remaining_usd"] = max(0, (100 - result["total_accounted_ils"]) / identity["ils_per_usd"])
    return result


def _run(root, session, name, cohort, **run_options):
    # The caller holds the session lock across the snapshot and all requests.
    total = session_summary(root)
    directory = _directory(root, name, cohort)
    own_cost = evaluation.summarize_evaluation(directory)["charged_or_reserved_usd"]
    try:
        return evaluation.run_evaluation(directory, code_revision=session["identity"]["code_revision"],
            max_total_cost_usd=own_cost + total["remaining_usd"], **run_options)
    finally:
        evaluation._atomic_json(root / "session-summary.json", session_summary(root))


def run_system_pilot(output_dir, system, *, model_identity=None, max_new_calls=None):
    root = Path(output_dir).resolve()
    with evaluation._lock(root):
        session = _session(root)
        manifest = _prepare(root, session, system, "dev_pilot", model_identity)
        summary = _run(root, session, system["name"], "dev_pilot", max_new_calls=max_new_calls)
        return {"manifest": manifest, "summary": summary}


def review_system_pilot(output_dir, system, *, model_identity=None):
    """Recheck current settings and saved evidence before projecting this system."""
    root = Path(output_dir).resolve()
    session = _session(root)
    directory = _directory(root, system["name"], "dev_pilot")
    if not (directory / "manifest.json").exists():
        raise ValueError("Complete this system's pilot first")
    manifest = _prepare(root, session, system, "dev_pilot", model_identity)
    summary = evaluation.summarize_evaluation(directory)
    if (summary["n_items"] != 10 or summary["n_records"] != 20 or summary["n_completed"] != 20
            or summary["n_ambiguous"] or summary["n_identity_unverified"]):
        raise ValueError("This system's pilot is incomplete or its identity is unverified")
    cost = evaluation.estimate_cost(summary, session["identity"]["rates"], full_items=395)
    total = session_summary(root)
    full_dir = _directory(root, system["name"], "full_test")
    full_cost = (evaluation.summarize_evaluation(full_dir)["charged_or_reserved_usd"]
                 if (full_dir / "manifest.json").exists() else 0)
    remaining_projection = max(0, cost["projected_full_with_reserve_usd"] - full_cost)
    return {"pilot_identity": manifest["identity_sha256"], "cost": cost,
            "projected_total_ils": total["total_accounted_ils"] + remaining_projection * session["identity"]["ils_per_usd"],
            "summary": summary}


def run_system_full(output_dir, system, *, inspected_pilot_identity, model_identity=None, max_new_calls=None):
    root = Path(output_dir).resolve()
    with evaluation._lock(root):
        session = _session(root)
        review = review_system_pilot(root, system, model_identity=model_identity)
        if inspected_pilot_identity != review["pilot_identity"]:
            raise ValueError("Inspect this exact pilot before full test")
        if review["projected_total_ils"] > 100:
            raise ValueError("Fresh projection exceeds the combined 100 ILS budget")
        _prepare(root, session, system, "full_test", model_identity)
        return _run(root, session, system["name"], "full_test", max_new_calls=max_new_calls)
