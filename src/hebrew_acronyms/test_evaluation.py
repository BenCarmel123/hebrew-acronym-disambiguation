"""Identity-bound, append-only evaluation of the fixed dev pilot and test cohort.

Each request has a durable start and finish event. A start without a finish is
ambiguous and is never automatically resent; it may already have been billed.
A partial final journal line is retained as evidence, and recovery writes a new
segment. Atomic writes/fsync improve local durability; mounted remote filesystems
must actually persist writes. Run only one process against a run directory.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import time
import uuid

from hebrew_acronyms.models.common import eval as scoring
from hebrew_acronyms.models.common.pairs import _normalise, candidates_for, load_rows, validate_ids

PROTOCOL = "fixed-cohort-item-shuffle-labels-v1"
TASKS = ("generate", "select")
STATUSES = {"response_received", "incomplete_response", "missing_response", "blocked", "service_error"}


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(_json(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def _lock(directory):
    # Advisory POSIX locking prevents overlapping local notebook kernels, without
    # a stale lock after disconnect. Cross-host Drive locks are not guaranteed.
    import fcntl
    with (directory / ".run.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("This run is already active in another local process") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _code_hashes():
    root = Path(__file__).parent
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*.py"))}


def _positive(value, name, integer=False):
    if type(value) not in ({int} if integer else {int, float}) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive {'integer' if integer else 'number'}")


def _validate_system(system, fixture):
    if set(system) != {"name", "provider", "model", "settings"}:
        raise ValueError("Each system needs exactly name, provider, model, settings")
    if not all(isinstance(system[key], str) and system[key] for key in ("name", "provider", "model")):
        raise ValueError("System names, providers and models must be nonempty strings")
    settings = system["settings"]
    if not isinstance(settings, dict):
        raise ValueError("System settings must be a dictionary")
    if fixture and system["provider"] == "fixture":
        return
    allowed = {
        "gemini": {"timeout", "generation_config"},
        "qwen": {"timeout", "options", "expected_digest", "base_url"},
        "openai": {"timeout", "max_output_tokens", "temperature"},
        "anthropic": {"timeout", "max_output_tokens", "effort"},
    }
    if system["provider"] not in allowed or not set(settings) <= allowed[system["provider"]]:
        raise ValueError("Unknown provider or unsupported settings; never store credentials in the manifest")
    _positive(settings.get("timeout"), "timeout")
    if system["provider"] == "gemini":
        from hebrew_acronyms.models.gemini.eval import validate_gemini_settings
        config = validate_gemini_settings(system["model"], settings["timeout"], settings.get("generation_config"))
        _positive(config.get("maxOutputTokens"), "maxOutputTokens", True)
    elif system["provider"] == "qwen":
        _positive(settings.get("options", {}).get("num_predict"), "num_predict", True)
        options = settings.get("options", {})
        if type(options.get("seed")) is not int or type(options.get("temperature")) not in {int, float} or not math.isfinite(options["temperature"]) or options["temperature"] < 0:
            raise ValueError("Qwen requires explicit finite temperature and integer sampling seed")
        if not settings.get("expected_digest"):
            raise ValueError("Qwen requires an inspected expected_digest")
    else:
        _positive(settings.get("max_output_tokens"), "max_output_tokens", True)
        if system["provider"] == "openai":
            from hebrew_acronyms.models.openai.eval import validate_openai_settings
            validate_openai_settings(system["model"], **settings)
        else:
            from hebrew_acronyms.models.anthropic.eval import validate_anthropic_settings
            validate_anthropic_settings(system["model"], **settings)


def prepare_evaluation(input_path, output_dir, *, cohort, systems, code_revision,
                       seed=42, max_attempts=2, max_calls=None, metadata=None,
                       reserve_per_call_usd=None, budget_usd=None, rates_usd_per_million=None):
    """Freeze full source bytes, selected rows, settings, code and all prompts.

    Existing runs are reused only if this complete identity matches. ``dev_pilot``
    uses the first ten CSV rows; ``full_test`` requires exactly 395 rows. Fixture
    mode is solely for offline tests. Price reserve is an upper-bound allocation
    per started call, including calls with unknown billing; it is not an invoice.
    """
    if cohort not in {"dev_pilot", "full_test", "fixture"}:
        raise ValueError("cohort must be dev_pilot, full_test or fixture")
    if cohort != "fixture" and not re.fullmatch(r"[0-9a-f]{40}", code_revision or ""):
        raise ValueError("Provide the exact 40-character code commit")
    _positive(max_attempts, "max_attempts", True)
    if type(seed) is not int:
        raise ValueError("Candidate shuffle seed must be an integer")
    source = Path(input_path).resolve()
    rows = load_rows(source)
    validate_ids(rows)
    if cohort == "dev_pilot":
        if len(rows) < 10:
            raise ValueError("dev_pilot requires at least ten rows")
        rows = rows[:10]
    elif cohort == "full_test" and len(rows) != 395:
        raise ValueError("full_test requires all 395 items")
    if not rows or not systems:
        raise ValueError("At least one item and one system are required")
    if len({system["name"] for system in systems}) != len(systems):
        raise ValueError("System names must be unique")
    for system in systems:
        _validate_system(system, cohort == "fixture")
    max_calls = len(rows) * len(systems) * 2 * max_attempts if max_calls is None else max_calls
    _positive(max_calls, "max_calls", True)
    if (reserve_per_call_usd is None) != (budget_usd is None):
        raise ValueError("Provide both budget_usd and reserve_per_call_usd, or neither")
    if budget_usd is not None:
        _positive(budget_usd, "budget_usd")
        if isinstance(reserve_per_call_usd, dict):
            if set(reserve_per_call_usd) != {system["name"] for system in systems}:
                raise ValueError("Per-system reserves must cover every selected system")
            values = reserve_per_call_usd.values()
        else:
            values = [reserve_per_call_usd]
        if any(type(value) not in {int, float} or not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("Call reserves must be finite and nonnegative")
    if rates_usd_per_million is not None:
        if set(rates_usd_per_million) != {system["name"] for system in systems}:
            raise ValueError("Price rates must cover every selected system")
        for rates in rates_usd_per_million.values():
            if set(rates) != {"input", "output"} or any(type(rate) not in {int, float} or not math.isfinite(rate) or rate < 0 for rate in rates.values()):
                raise ValueError("Prices require finite nonnegative input and output token rates")
    requests = []
    for row in rows:
        for field in ("sentence", "acronym", "gold_expansion"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"{row['item_id']}: missing {field}")
        candidates = candidates_for(row)
        if len(candidates) != len(set(candidates)) or row["gold_expansion"].strip() not in candidates:
            raise ValueError(f"{row['item_id']}: duplicate candidates or gold outside inventory")
        shown = candidates[:]
        random.Random(int(_hash({"seed": seed, "item_id": row["item_id"]}), 16)).shuffle(shown)
        for system in systems:
            for task in TASKS:
                prompt = (scoring.build_generate_prompt(row["acronym"], row["sentence"]) if task == "generate"
                          else scoring.build_select_prompt(row["acronym"], row["sentence"], shown))
                requests.append({"key": _hash([system["name"], row["item_id"], task]),
                                 "system": system["name"], "item_id": row["item_id"], "task": task,
                                 "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                                 "shown_order": shown if task == "select" else None})
    identity = {"protocol": PROTOCOL, "cohort": cohort, "source_path": str(source),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "rows": rows,
                "systems": systems, "code_revision": code_revision, "code_sha256": _code_hashes(),
                "seed": seed, "max_attempts": max_attempts, "max_calls": max_calls,
                "reserve_per_call_usd": reserve_per_call_usd, "budget_usd": budget_usd, "rates_usd_per_million": rates_usd_per_million,
                "requests": requests, "metadata": metadata or {},
                "target_occurrences": {row["item_id"]: _normalise(row["sentence"]).count(_normalise(row["acronym"])) for row in rows},
                "target_policy": "Whole original sentence; occurrence counts are observations, never gold-derived span choices"}
    identity = json.loads(_json(identity))
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with _lock(directory):
        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            existing = _load_manifest(directory)
            if existing["identity_sha256"] != _hash(identity):
                raise ValueError("Resume identity mismatch: inputs, code, prompts or settings changed")
            return existing
        if any(path.name != ".run.lock" for path in directory.iterdir()):
            raise ValueError("Refusing to initialize a nonempty output directory")
        versions = {}
        for package in ("requests", "transformers", "torch", "numpy"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        manifest = {"run_id": uuid.uuid4().hex, "created_utc": _now(), "identity": identity,
                    "identity_sha256": _hash(identity),
                    "environment": {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}}
        _atomic_json(manifest_path, manifest)
    return manifest


def _load_manifest(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["identity_sha256"] != _hash(manifest["identity"]):
        raise ValueError("Manifest content does not match its identity hash")
    return manifest


def _read_events(directory, manifest):
    """Only an unterminated final line may be ignored, never a corrupt middle line."""
    starts, finishes, tails = {}, {}, []
    allowed = {request["key"] for request in manifest["identity"]["requests"]}
    paths = sorted(directory.glob("attempts-*.jsonl"))
    if (directory / "attempts.jsonl").exists():
        paths.insert(0, directory / "attempts.jsonl")
    for path in paths:
        data = path.read_bytes()
        lines = data.splitlines(keepends=True)
        last_event = None
        for number, line in enumerate(lines, 1):
            if not line.endswith(b"\n"):
                tails.append({"path": path.name, "line": number, "bytes": len(line),
                              "sha256": hashlib.sha256(line).hexdigest(),
                              "unfinished_attempt_id": last_event["attempt_id"] if last_event and last_event["event"] == "started" else None})
                continue
            try:
                event = json.loads(line)
                if event["run_id"] != manifest["run_id"] or event["identity_sha256"] != manifest["identity_sha256"]:
                    raise ValueError("Journal identity mismatch")
                attempt_id = event["attempt_id"]
                if event["event"] == "started":
                    if attempt_id in starts or event["key"] not in allowed:
                        raise ValueError("Duplicate or unknown attempt")
                    starts[attempt_id] = event
                elif event["event"] == "finished":
                    if attempt_id not in starts or attempt_id in finishes or event["key"] != starts[attempt_id]["key"]:
                        raise ValueError("Orphan or duplicate completion")
                    finishes[attempt_id] = event
                else:
                    raise ValueError("Unknown journal event")
                last_event = event
            except (KeyError, TypeError, json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
                raise ValueError(f"Invalid journal event at {path.name}:{number}") from error
    return starts, finishes, tails


def _append(stream, event):
    stream.write(_json(event) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def _call(system, prompt):
    provider = system["provider"]
    if provider == "gemini":
        from hebrew_acronyms.models.gemini.eval import gemini_response as request
    elif provider == "qwen":
        from hebrew_acronyms.models.qwen.eval import ollama_response as request
    elif provider == "openai":
        from hebrew_acronyms.models.openai.eval import openai_response as request
    elif provider == "anthropic":
        from hebrew_acronyms.models.anthropic.eval import anthropic_response as request
    else:
        raise ValueError("Fixture execution requires injected responders")
    return request(prompt, model=system["model"], **system["settings"])


def run_evaluation(output_dir, *, code_revision, max_new_calls=None, responders=None, sleep=time.sleep):
    """Execute or resume, retrying only explicit transient failures within saved caps.

    Injected responders are restricted to fixture runs. Return a derived summary;
    raw response dictionaries (including usage) remain in the durable journal.
    KeyboardInterrupt/SystemExit leave an ambiguous start and propagate.
    """
    directory = Path(output_dir).resolve()
    if max_new_calls is not None:
        _positive(max_new_calls, "max_new_calls", True)
    with _lock(directory):
        manifest = _load_manifest(directory)
        identity = manifest["identity"]
        if code_revision != identity["code_revision"] or _code_hashes() != identity["code_sha256"]:
            raise ValueError("Resume code identity mismatch")
        if hashlib.sha256(Path(identity["source_path"]).read_bytes()).hexdigest() != identity["source_sha256"]:
            raise ValueError("Resume input identity mismatch")
        if responders is not None and identity["cohort"] != "fixture":
            raise ValueError("Injected responders are allowed only for offline fixture runs")
        starts, finishes, tails = _read_events(directory, manifest)
        if any(tail["unfinished_attempt_id"] is None for tail in tails):
            raise ValueError("Truncated journal has no identifiable unfinished request; inspect before any further calls")
        by_key = defaultdict(list)
        for attempt_id, start in starts.items():
            by_key[start["key"]].append((start, finishes.get(attempt_id)))
        systems = {system["name"]: system for system in identity["systems"]}
        call_count, new_calls = len(starts), 0
        charged_or_reserved = sum(_attempt_cost(identity, systems[start["system"]],
                                   finishes[attempt_id]["response"] if attempt_id in finishes else None)
                                  for attempt_id, start in starts.items())
        paths = list(directory.glob("attempts*.jsonl"))
        log_path = directory / ("attempts.jsonl" if not paths else f"attempts-{len(paths):06d}.jsonl")
        stop_reason = None
        with log_path.open("x", encoding="utf-8") as stream:
            for request in identity["requests"]:
                previous = by_key[request["key"]]
                while len(previous) < identity["max_attempts"]:
                    if previous:
                        finished = previous[-1][1]
                        if finished is None:
                            break
                        response = finished["response"]
                        if response.get("status") != "service_error" or not response.get("retryable"):
                            break
                    if call_count >= identity["max_calls"] or (max_new_calls is not None and new_calls >= max_new_calls):
                        stop_reason = "call_limit"
                        break
                    reserve = _reserve(identity, request["system"])
                    if identity["budget_usd"] is not None and charged_or_reserved + reserve > identity["budget_usd"] + 1e-12:
                        stop_reason = "budget_reserve_limit"
                        break
                    if previous:
                        retry_after = previous[-1][1]["response"].get("retry_after")
                        try:
                            delay = float(retry_after)
                            if not math.isfinite(delay):
                                raise ValueError
                        except (TypeError, ValueError):
                            delay = min(60, 2 ** len(previous))
                        if delay > 60:
                            stop_reason = "retry_after_exceeds_wait_limit"
                            break
                        sleep(max(0, delay))
                    attempt_id = f"{manifest['run_id']}:{request['key']}:{len(previous) + 1}"
                    common = {"run_id": manifest["run_id"], "identity_sha256": manifest["identity_sha256"],
                              "key": request["key"], "attempt_id": attempt_id}
                    started = dict(common, event="started", at_utc=_now(), ordinal=len(previous) + 1,
                                   system=request["system"], item_id=request["item_id"], task=request["task"],
                                   prompt=request["prompt"], settings=systems[request["system"]])
                    _append(stream, started)
                    call_count += 1
                    new_calls += 1
                    began = time.monotonic()
                    try:
                        result = (responders[request["system"]](request["prompt"]) if responders is not None
                                  else _call(systems[request["system"]], request["prompt"]))
                        if not isinstance(result, dict) or result.get("status") not in STATUSES or not isinstance(result.get("response"), str):
                            raise ValueError("Adapter did not return a supported response record")
                        _json(result)
                    except Exception as error:
                        # Exception strings can contain request headers/credentials.
                        result = {"status": "service_error", "response": "", "retryable": False,
                                  "error": f"Adapter raised {type(error).__name__}; details omitted", "usage_metadata": None}
                    finished = dict(common, event="finished", at_utc=_now(), elapsed_seconds=time.monotonic() - began, response=result)
                    _append(stream, finished)
                    previous.append((started, finished))
                    charged_or_reserved += _attempt_cost(identity, systems[request["system"]], result)
                if stop_reason:
                    break
        summary = summarize_evaluation(directory)
        summary["stop_reason"] = stop_reason
        _atomic_json(directory / "summary.json", summary)
        return summary


def _usage(response, provider):
    raw = response.get("usage_metadata")
    if provider == "qwen":
        raw = response.get("response_metadata")
        keys = ("prompt_eval_count", "eval_count", None, None)
    elif provider == "gemini":
        keys = ("promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount", "cachedContentTokenCount")
    elif provider == "openai":
        keys = ("prompt_tokens", "completion_tokens", None, None)
    else:
        keys = ("input_tokens", "output_tokens", None, "cache_read_input_tokens")
    if not isinstance(raw, dict) or any(key not in raw for key in keys[:2]):
        return None
    result = {name: raw.get(key, 0) if key else 0 for name, key in
              zip(("input_tokens", "output_tokens", "thinking_tokens", "cached_input_tokens"), keys)}
    if provider == "openai":
        result["cached_input_tokens"] = (raw.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    if provider == "anthropic":
        # Claude input_tokens excludes cached input; charge all cache at ordinary
        # input rate conservatively for this simple estimate.
        result["input_tokens"] += raw.get("cache_creation_input_tokens", 0) + raw.get("cache_read_input_tokens", 0)
    if any(type(value) is not int or value < 0 for value in result.values()):
        return None
    return result


def _reserve(identity, system):
    reserve = identity["reserve_per_call_usd"]
    return (reserve.get(system, 0) if isinstance(reserve, dict) else reserve) or 0


def _attempt_cost(identity, system, response):
    rates = (identity.get("rates_usd_per_million") or {}).get(system["name"])
    usage = _usage(response, system["provider"]) if response else None
    if rates is None or usage is None:
        return _reserve(identity, system["name"])
    return (usage["input_tokens"] * rates["input"] +
            (usage["output_tokens"] + usage["thinking_tokens"]) * rates["output"]) / 1_000_000


def summarize_evaluation(output_dir):
    """Rebuild full-cohort coverage and explicitly labelled historical scoring."""
    directory = Path(output_dir).resolve()
    manifest = _load_manifest(directory)
    identity = manifest["identity"]
    starts, finishes, tails = _read_events(directory, manifest)
    by_key = defaultdict(list)
    for attempt_id, start in starts.items():
        by_key[start["key"]].append((start, finishes.get(attempt_id)))
    systems = {system["name"]: system for system in identity["systems"]}
    rows = {row["item_id"]: row for row in identity["rows"]}
    usage = {name: {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0, "cached_input_tokens": 0,
                    "attempts_with_usage": 0, "attempts_without_usage": 0} for name in systems}
    for attempt_id, started in starts.items():
        system = started["system"]
        normalized = _usage(finishes[attempt_id]["response"], systems[system]["provider"]) if attempt_id in finishes else None
        usage[system]["attempts_with_usage" if normalized is not None else "attempts_without_usage"] += 1
        if normalized:
            for key, value in normalized.items():
                usage[system][key] += value
    records = []
    for request in identity["requests"]:
        attempts = by_key[request["key"]]
        last = attempts[-1][1] if attempts else None
        response = last["response"] if last else {}
        status = response.get("status", "ambiguous" if attempts else "not_run")
        row = rows[request["item_id"]]
        raw = response.get("response", "")
        correct, valid, selected = False, False, None
        if status == "response_received" and response.get("identity_status") != "unverified":
            if request["task"] == "select":
                choice = scoring.parse_letter_choice(raw, len(request["shown_order"]))
                valid = choice is not None
                selected = request["shown_order"][choice] if valid else None
                correct = valid and selected == row["gold_expansion"].strip()
            else:
                correct = scoring.is_correct(raw, row["gold_expansion"].strip())
                valid = scoring.is_valid(raw, candidates_for(row))
        retry_pending = bool(last and status == "service_error" and response.get("retryable") and len(attempts) < identity["max_attempts"])
        records.append(dict(request, status=status, response=raw, selected_candidate=selected,
                            correct=correct, valid=valid, attempts=len(attempts), retry_pending=retry_pending,
                            error=response.get("error"), response_metadata=response))
    groups = []
    for system in systems:
        for task in TASKS:
            subset = [record for record in records if record["system"] == system and record["task"] == task]
            groups.append({"system": system, "task": task, "n_items": len(subset),
                           "n_completed": sum(record["status"] == "response_received" for record in subset),
                           "status_counts": dict(Counter(record["status"] for record in subset)),
                           "accuracy": sum(record["correct"] for record in subset) / len(subset),
                           "score": "strict_candidate_label_v1" if task == "select" else "historical_quote_normalized_gold_substring",
                           "n_valid": sum(record["valid"] for record in subset)})
    return {"run_id": manifest["run_id"], "identity_sha256": manifest["identity_sha256"],
            "cohort": identity["cohort"], "n_items": len(rows), "n_records": len(records), "n_calls": len(starts),
            "n_completed": sum(record["status"] == "response_received" for record in records),
            "n_pending": sum(record["status"] == "not_run" or record["retry_pending"] for record in records),
            "n_ambiguous": sum(record["status"] == "ambiguous" for record in records),
            "n_incomplete": sum(record["status"] not in {"response_received", "not_run"} for record in records),
            "n_identity_unverified": sum(record["response_metadata"].get("identity_status") == "unverified" for record in records),
            "truncated_journal_tails": tails, "usage": usage, "by_system_task": groups, "records": records,
            "unknown_usage_reserve_usd": {
                name: sum(_reserve(identity, name) for attempt_id, start in starts.items()
                          if start["system"] == name and (attempt_id not in finishes or
                              _usage(finishes[attempt_id]["response"], systems[name]["provider"]) is None))
                for name in systems},
            "timings_seconds": {name: sum(event["elapsed_seconds"] for attempt_id, event in finishes.items()
                                           if starts[attempt_id]["system"] == name) for name in systems},
            "reserved_usd": sum(_reserve(identity, start["system"]) for attempt_id, start in starts.items()
                                if attempt_id not in finishes or _usage(finishes[attempt_id]["response"], systems[start["system"]]["provider"]) is None),
            "charged_or_reserved_usd": sum(_attempt_cost(identity, systems[start["system"]],
                                      finishes[attempt_id]["response"] if attempt_id in finishes else None)
                                      for attempt_id, start in starts.items())}


def estimate_cost(summary, rates_usd_per_million, *, full_items=395, reserve_fraction=.25):
    """Scale measured pilot costs plus recorded reserves for uncertain attempts.

    All logical answers must still be complete and identity-verified. Unknown
    usage is covered by the saved per-call allowance, never treated as free.
    These tariff estimates and reserves are not a settled provider invoice.
    """
    if summary["n_completed"] != summary["n_records"] or summary.get("n_identity_unverified"):
        raise ValueError("Complete all pilot responses with verified identity before cost extrapolation")
    _positive(full_items, "full_items", True)
    if not isinstance(reserve_fraction, (int, float)) or not math.isfinite(reserve_fraction) or reserve_fraction < 0:
        raise ValueError("reserve_fraction must be finite and nonnegative")
    by_system = {}
    for name, usage in summary["usage"].items():
        rates = rates_usd_per_million[name]
        for key in ("input", "output"):
            if type(rates[key]) not in {int, float} or not math.isfinite(rates[key]) or rates[key] < 0:
                raise ValueError("Token rates must be finite and nonnegative")
        cost = (usage["input_tokens"] * rates["input"] +
                (usage["output_tokens"] + usage["thinking_tokens"]) * rates["output"]) / 1_000_000
        unknown_attempts = usage["attempts_without_usage"]
        reserve = summary.get("unknown_usage_reserve_usd", {}).get(name, 0)
        if type(reserve) not in {int, float} or not math.isfinite(reserve) or reserve < 0:
            raise ValueError(f"{name}: invalid saved unknown-usage reserve")
        if unknown_attempts and not reserve and any(rates.values()):
            raise ValueError(f"{name}: unknown billed usage has no saved conservative reserve")
        accounted = cost + reserve
        by_system[name] = {"pilot_usd": accounted, "measured_pilot_usd": cost,
                           "unknown_usage_reserve_usd": reserve,
                           "unknown_usage_attempts": unknown_attempts,
                           "projected_full_usd": accounted * full_items / summary["n_items"]}
    pilot = sum(value["pilot_usd"] for value in by_system.values())
    full = sum(value["projected_full_usd"] for value in by_system.values())
    return {"pilot_usd": pilot,
            "measured_pilot_usd": sum(value["measured_pilot_usd"] for value in by_system.values()),
            "unknown_usage_reserve_usd": sum(value["unknown_usage_reserve_usd"] for value in by_system.values()),
            "projected_full_usd": full,
            "projected_full_with_reserve_usd": full * (1 + reserve_fraction), "by_system": by_system,
            "note": "Pilot total combines measured token costs and saved allowances for unknown usage; full projection scales both and adds the stated margin. Not a settled bill or a worst-case guarantee; thinking included and cache charged at full input rate."}
