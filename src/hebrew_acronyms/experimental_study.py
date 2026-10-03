"""Notebook glue for identified predictions, portable raw artifacts and dev inspection.

No models, services, filesystem operations or network hooks run at import time.
The shared prompt builders and encoder evaluator retain ownership of model behavior.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import gc
import inspect
import json
import os
from pathlib import Path
import platform
import random
import subprocess
from importlib.metadata import version

from hebrew_acronyms.models.common import eval as prompts
from hebrew_acronyms.models.common.pairs import candidates_for, explicit_span, input_identity, validate_ids, mark_span, load_rows

CONDITIONS = ("dictabert", "generate", "select")  # Historical format 1.
ARMS = {"dictabert": ("dictabert", "select"),
        "qwen_generate": ("qwen", "generate"), "qwen_select": ("qwen", "select"),
        "gemini_generate": ("gemini", "generate"), "gemini_select": ("gemini", "select")}


def arm_specs(artifact):
    """Read format 1 explicitly, without migrating files or inventing Gemini records."""
    if artifact.get("format_version") == 1:
        return {"dictabert": ("dictabert", "select"), "generate": ("qwen", "generate"),
                "select": ("qwen", "select")}
    if artifact.get("format_version") == 2:
        return ARMS
    raise ValueError("Unsupported study artifact format")


def llm_runtime(artifact, system):
    if artifact["format_version"] == 1:
        return artifact.get("llm_runtime", {}) if system == "qwen" else {}
    return artifact.get("llm_runtimes", {}).get(system, {})


def _set_runtime(artifact, system, runtime):
    if artifact["format_version"] == 1:
        if system != "qwen":
            raise ValueError("Historical runs cannot gain a new system; start a new run")
        artifact["llm_runtime"] = runtime
    else:
        artifact.setdefault("llm_runtimes", {})[system] = runtime


def _without_secret(value):
    """Defense in depth: backend errors and returned metadata must never retain the key."""
    key = os.environ.get("GEMINI_API_KEY")
    if isinstance(value, str):
        return value.replace(key, "[REDACTED]") if key else value
    if isinstance(value, dict):
        return {_without_secret(k): _without_secret(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_without_secret(v) for v in value]
    return value



def fixture_rows():
    """Invented examples only; deliberately different acronym and raw target."""
    return [
        {"item_id": "invented-1", "type_id": "invented-type-one", "sentence": "היא פנתה לב״מ היום.", "acronym": "ב״מ",
         "target_raw": "לב״מ", "span_start": 9, "span_end": 13,
         "candidates": "בית מלאכה|בית מדרש", "gold_expansion": "בית מלאכה",
         "source_kind": "invented_fixture"},
        {"item_id": "invented-2", "type_id": "invented-type-two", "sentence": "זהו א״ב.", "acronym": "א״ב",
         "target_raw": "א״ב", "span_start": 4, "span_end": 7,
         "candidates": "אור בוקר", "gold_expansion": "אור בוקר",
         "source_kind": "invented_fixture"},
    ]


def check_readiness(root, input_path, checkpoint, output_dir, *, run, encoder, llm,
                    model, saved_encoder=None):
    """Check paths and installed interfaces locally; never read research inputs or call a service."""
    from hebrew_acronyms.models.dictabert_cross_encoder.model import load_finetuned
    from hebrew_acronyms.models.dictabert_cross_encoder.eval import evaluate
    from hebrew_acronyms.models.qwen.eval import ollama_generate
    required = [(load_finetuned, {"checkpoint_path", "device"}),
                (evaluate, {"rows", "tok", "model", "acr_open_id", "acr_close_id", "device"}),
                (ollama_generate, {"prompt", "model"}),
                (prompts.build_generate_prompt, {"acronym", "sentence"}),
                (prompts.build_select_prompt, {"acronym", "sentence", "shuffled_candidates"})]
    for function, names in required:
        if not names <= set(inspect.signature(function).parameters):
            raise RuntimeError(f"Installed interface changed: {function.__module__}.{function.__name__}")
    if not run:
        return {"mode": "preview", "models_called": False, "research_inputs_read": False}
    if not Path(input_path).is_file():
        raise FileNotFoundError(f"Qualified dev input missing: {input_path}")
    if encoder and saved_encoder is not None:
        raise ValueError("Choose checkpoint inference OR saved encoder predictions")
    if encoder:
        if checkpoint is None or not Path(checkpoint).is_file():
            raise FileNotFoundError("Set CHECKPOINT to Ben's weights and provide the adjacent .json")
        if not Path(str(checkpoint) + ".json").is_file():
            raise FileNotFoundError(f"Matching checkpoint JSON is required: {checkpoint}.json")
    if saved_encoder is not None and not Path(saved_encoder).is_file():
        raise FileNotFoundError(saved_encoder)
    if llm and (not isinstance(model, str) or not model.strip()):
        raise ValueError("Set the exact Qwen Ollama model tag; its local digest is resolved at connection")
    destination = Path(output_dir).expanduser().resolve()
    if (destination.is_relative_to(Path(root).resolve()) or
            any((parent / ".git").exists() for parent in (destination, *destination.parents))):
        raise ValueError("Run outputs must be outside the repository")
    if destination.exists():
        raise FileExistsError(f"Choose a fresh run directory: {destination}")
    return {"mode": "run", "local_paths_checked": True,
            "checkpoint_compatibility": "enforced by load_finetuned, not by readiness",
            "qwen_service": "not contacted; user-managed Ollama must already be available"}


def source_provenance(root, source_revision=None):
    """Record local source identity without importing or running the backends."""
    root = Path(root).resolve()
    paths = [Path(__file__), Path(inspect.getfile(prompts))]
    paths += [Path(__file__).parent / "models" / name / "eval.py"
              for name in ("qwen", "gemini", "dictabert_cross_encoder")]
    paths += [Path(__file__).parent / "models/dictabert_cross_encoder/model.py"]
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=False).stdout.strip() or source_revision
    except FileNotFoundError:
        head = source_revision
    return {"git_head": head, "revision_supplied_by_operator": source_revision, "python": platform.python_version(),
            "package_version": version("hebrew-acronym-disambiguation"),
            "source_sha256": {str(p.relative_to(Path(__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in paths}}


def new_artifact(rows, run_id, settings, provenance):
    validate_ids(rows)
    if not rows or not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("A nonempty item set and explicit run ID are required")
    identity = input_identity(rows)
    records = [{"run_id": run_id, "input_sha256": identity["sha256"],
                "item_id": row["item_id"], "condition": condition,
                "system": ARMS[condition][0], "task": ARMS[condition][1], "status": "not_run",
                "score_status": "unscored", "raw_response": None, "error": None}
               for condition in ARMS for row in rows]
    return {"format_version": 2, "run_id": run_id, "input_identity": identity,
            "items": deepcopy(rows), "settings": deepcopy(settings),
            "provenance": deepcopy(provenance), "records": records}


def attach_encoder(artifact, predictions, *, origin):
    """Join identified source-evaluator records; reject dropped, duplicate or foreign IDs."""
    if any(record["status"] != "not_run" for record in artifact["records"] if record["condition"] == "dictabert"):
        raise ValueError("Encoder predictions already exist; start a new run instead of overwriting")
    validate_ids(predictions)
    expected = artifact["input_identity"]["item_ids"]
    by_id = {prediction["item_id"]: prediction for prediction in predictions}
    if set(by_id) != set(expected):
        raise ValueError("Encoder predictions must cover exactly the current item IDs")
    for record in artifact["records"]:
        if record["condition"] == "dictabert":
            prediction = by_id[record["item_id"]]
            # Do not permit source dictionaries to replace run/input/condition identity.
            record.update({key: deepcopy(prediction.get(key)) for key in
                           ("status", "selected_candidate", "selected_index", "candidate_scores", "error")})
            record["encoder_origin"] = deepcopy(origin)
    validate_artifact(artifact, expected_run_id=artifact["run_id"])


def _record_response(artifact, record, system, response):
    """Attach provider evidence; Gemini versions are reported, never digest-verified."""
    if isinstance(response, str) and artifact["format_version"] == 1:
        record.update(status="response_received", raw_response=response)
        return
    if not isinstance(response, dict) or not isinstance(response.get("response"), str):
        raise TypeError("Backend response must contain text and metadata")
    response = _without_secret(response)
    record["backend_metadata"] = {key: deepcopy(value) for key, value in response.items() if key != "response"}
    record.update(status=response.get("status", "response_received"), raw_response=response["response"],
                  error=response.get("error"))
    if system == "qwen" and response.get("completion_status") == "incomplete":
        record["status"] = "incomplete_response"
    if system == "gemini":
        record["model_revision"] = response.get("model_version")
    if record["status"] != "response_received":
        return
    runtime = llm_runtime(artifact, system)
    if system == "qwen":
        if response.get("identity_status") != "verified":
            record.update(status="model_identity_error", error=response.get("error") or "Ollama identity was not verified")
    else:
        version = response.get("model_version")
        record["model_revision"] = version
        if (not isinstance(version, str) or not version.strip()
                or response.get("requested_model") != record["model"]):
            record.update(status="model_identity_error", error="Gemini did not report the requested model and a modelVersion")
        elif runtime.get("model_version") not in (None, version):
            record.update(status="model_identity_error", error="Gemini modelVersion changed within this run")
        else:
            runtime["model_version"] = version
            runtime["version_evidence"] = "reported_by_generateContent; no digest verification"


def collect_responses(artifact, mode, generate_fn, *, system="qwen", on_record=None, limit=None):
    """Collect one identified arm; preserve raw failures and every unattempted item."""
    if system not in {"qwen", "gemini"} or mode not in {"generate", "select"}:
        raise ValueError("Only Qwen/Gemini generate/select with the sentence are supported")
    condition = next((name for name, pair in arm_specs(artifact).items() if pair == (system, mode)), None)
    if condition is None:
        raise ValueError("System absent from this historical run; start a new run")
    runtime = llm_runtime(artifact, system)
    model = artifact["settings"].get(system + "_model")
    revision = runtime.get("digest") if system == "qwen" else runtime.get("model_version")
    if artifact["format_version"] == 1:
        revision = revision or artifact["settings"].get("qwen_revision")
        if not revision:
            raise ValueError("Explicit Qwen model and resolved version/digest are required")
    if not model or (artifact["format_version"] == 2 and not runtime):
        raise ValueError("Configure and connect the requested system before collection")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("Response limit must be a positive integer or None")
    validate_artifact(artifact, expected_run_id=artifact["run_id"])
    pending = [record for record in artifact["records"] if record["condition"] == condition]
    if any(record["status"] != "not_run" for record in pending):
        raise ValueError("Responses already exist; start a new run instead of overwriting")
    rng = random.Random(prompts.SHUFFLE_SEED)
    items = {row["item_id"]: row for row in artifact["items"]}
    for record in pending[:limit]:
        row = items[record["item_id"]]
        record.update(model=model, model_revision=revision, sentence=row.get("sentence"),
                      target_raw=row.get("target_raw"), prompt=None, shown_order=None)
        try:
            marked_sentence = mark_span(row["sentence"], explicit_span(row))
            record["prompt_sentence"] = marked_sentence
            if mode == "generate":
                prompt = prompts.build_generate_prompt(row["target_raw"], marked_sentence)
            else:
                candidates = candidates_for(row)
                if len(set(candidates)) != len(candidates) or len(candidates) > 26:
                    raise ValueError("Selection requires 1–26 distinct candidates for the shared letter prompt")
                rng.shuffle(candidates)
                record["shown_order"] = candidates
                prompt = prompts.build_select_prompt(row["target_raw"], marked_sentence, candidates)
            record["prompt"] = prompt
            record["prompt_sha256"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        except (KeyError, ValueError) as error:
            record.update(status="invalid_input", error=str(error))
        else:
            record["status"] = "interrupted"
            try:
                _record_response(artifact, record, system, generate_fn(prompt))
            except Exception as error:
                # Gemini exceptions may embed credential-bearing requests. Never stringify them.
                message = "Gemini request failed; inspect configuration/service availability" if system == "gemini" else f"{type(error).__name__}: {error}"
                record.update(status="service_error", error=_without_secret(message))
            finally:
                if on_record is not None:
                    on_record(artifact)
            continue
        if on_record is not None:
            on_record(artifact)
    return artifact


def validate_artifact(artifact, *, expected_run_id, rows=None):
    if not isinstance(artifact, dict) or artifact.get("format_version") not in {1, 2}:
        raise ValueError("Expected an identity-bound study artifact, not a legacy prediction list/CSV")
    if not expected_run_id or artifact.get("run_id") != expected_run_id:
        raise ValueError("Saved run ID differs from the requested run")
    validate_ids(artifact["items"])
    identity = input_identity(artifact["items"])
    if identity != artifact["input_identity"] or (rows is not None and identity != input_identity(rows)):
        raise ValueError("Saved input identity differs (including text, candidate order and metadata)")
    cohort = artifact.get("cohort")
    if cohort is not None:
        if (cohort.get("requested_items") != len(artifact["items"])
                or type(cohort.get("available_items")) is not int
                or cohort["available_items"] < len(artifact["items"])
                or cohort.get("full_dev") is not (cohort.get("kind") == "full_dev")):
            raise ValueError("Cohort counts/scope disagree with the requested items")
        full_identity = cohort.get("full_input_identity")
        if full_identity is not None and (len(full_identity.get("item_ids", [])) != cohort["available_items"]
                or full_identity["item_ids"][:len(artifact["items"])] != identity["item_ids"]):
            raise ValueError("Cohort differs from the complete input identity")
        if cohort["full_dev"] and (cohort["available_items"] != len(artifact["items"]) or full_identity != identity):
            raise ValueError("A full-dev label requires the complete original input identity")
        if cohort.get("kind") == "validation" and cohort["available_items"] == len(artifact["items"]):
            raise ValueError("Validation must remain a subset of the qualified dev input")
    specs = arm_specs(artifact)
    for system in {system for system, task in specs.values()} - {"dictabert"}:
        runtime = llm_runtime(artifact, system)
        if runtime:
            model_key = "model" if system == "qwen" else "requested_model"
            if runtime.get(model_key) != artifact["settings"].get(system + "_model"):
                raise ValueError("Runtime LLM identity disagrees with run settings")
            if system == "gemini":
                settings = artifact["settings"]
                expected_request = {"generationConfig": settings.get("gemini_generation_config"),
                                    "timeout_seconds": settings.get("request_timeout"), "max_attempts": 1}
                if runtime.get("request_settings") != expected_request:
                    raise ValueError("Gemini runtime settings disagree with the recorded run")
            if system == "qwen" and not runtime.get("digest") and not runtime.get("connection_error"):
                raise ValueError("Qwen runtime requires its resolved digest or explicit connection failure")
            if system == "qwen" and artifact["format_version"] == 2 and not runtime.get("connection_error"):
                for setting, evidence in (("qwen_options", "options"), ("ollama_url", "base_url"),
                                          ("request_timeout", "timeout_seconds")):
                    if setting in artifact["settings"]:
                        expected_value = artifact["settings"][setting]
                        if setting == "ollama_url":
                            expected_value = expected_value.rstrip("/")
                        if runtime.get(evidence) != expected_value:
                            raise ValueError("Qwen runtime settings disagree with the recorded run")
    expected = {(condition, item_id) for condition in specs for item_id in identity["item_ids"]}
    actual = [(record["condition"], record["item_id"]) for record in artifact["records"]]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Every condition must retain exactly one record per item")
    items = {row["item_id"]: row for row in artifact["items"]}
    for record in artifact["records"]:
        system, task = specs[record["condition"]]
        if artifact["format_version"] == 2 and (record.get("system"), record.get("task")) != (system, task):
            raise ValueError("Mixed system/task identity")
        if record["run_id"] != expected_run_id or record["input_sha256"] != identity["sha256"]:
            raise ValueError("Mixed run/input record identity")
        if record.get("score_status") != "unscored" or not record.get("status"):
            raise ValueError("Records require a status and must remain unscored")
        if record["condition"] == "dictabert" and record["status"] != "not_run":
            if record["status"] not in {"ok", "invalid_input", "encoding_error", "model_error", "invalid_scores"}:
                raise ValueError("Unknown encoder status")
            origin = record.get("encoder_origin")
            if not isinstance(origin, dict) or any(
                    not isinstance(origin.get(key), str) or len(origin[key]) != 64
                    or any(ch not in "0123456789abcdef" for ch in origin[key])
                    for key in ("weights_sha256", "manifest_sha256")):
                raise ValueError("Encoder predictions require original weights and manifest SHA-256 provenance")
            if record["status"] == "ok":
                candidates = candidates_for(items[record["item_id"]])
                index = record.get("selected_index")
                scores = record.get("candidate_scores")
                if (type(index) is not int or not 0 <= index < len(candidates)
                        or record.get("selected_candidate") != candidates[index]
                        or not isinstance(scores, list)
                        or [score.get("candidate") for score in scores] != candidates):
                    raise ValueError("Encoder prediction disagrees with the original candidate inventory")
        if system in {"qwen", "gemini"} and record["status"] != "not_run":
            if record["status"] not in {"invalid_input", "interrupted", "service_error", "response_received", "model_identity_error", "blocked", "missing_response", "incomplete_response"}:
                raise ValueError("Unknown LLM status")
            if record["status"] in {"response_received", "model_identity_error"} and not isinstance(record.get("raw_response"), str):
                raise ValueError("Received LLM response must retain raw text")
            runtime = llm_runtime(artifact, system)
            if record.get("model") != artifact["settings"].get(system + "_model"):
                raise ValueError("Mixed LLM model/version settings")
            evidence = record.get("backend_metadata")
            if system == "qwen":
                revision = runtime.get("digest") or artifact["settings"].get("qwen_revision")
                if record.get("model_revision") != revision:
                    raise ValueError("Mixed LLM model/version settings")
                if evidence is not None and "completion_status" in evidence and record["status"] == "response_received":
                    completion = evidence.get("response_metadata", {})
                    if (evidence["completion_status"] != "complete" or completion.get("done") is not True
                            or completion.get("done_reason") != "stop"):
                        raise ValueError("Qwen response is not a confirmed complete answer")
                if record["status"] == "response_received" and evidence is not None:
                    request = evidence.get("request_settings", {})
                    if (evidence.get("identity_status") != "verified" or not revision
                            or any(evidence.get(key) != revision for key in ("expected_digest", "digest_before", "digest_after"))
                            or evidence.get("model") != record["model"] or request.get("model") != record["model"]
                            or (runtime and request.get("options") != runtime.get("options"))):
                        raise ValueError("Backend response evidence differs from the bound model/digest/settings")
            elif evidence is not None:
                if evidence.get("requested_model") != record["model"] or evidence.get("request_settings") != runtime.get("request_settings"):
                    raise ValueError("Gemini response settings differ from the bound request")
                if record["status"] == "response_received" and (not record.get("model_revision")
                        or record["model_revision"] != runtime.get("model_version")
                        or record["model_revision"] != evidence.get("model_version")
                        or evidence.get("finish_reason") != "STOP"):
                    raise ValueError("Gemini response version/completion differs from its evidence")
            if artifact["format_version"] == 2 and record["status"] == "response_received" and not evidence:
                raise ValueError("New LLM responses require provider evidence")
            row = items[record["item_id"]]
            if record.get("sentence") != row.get("sentence") or record.get("target_raw") != row.get("target_raw"):
                raise ValueError("LLM input differs from the shared item")
            if task == "generate" and record.get("shown_order") is not None:
                raise ValueError("Generation must not receive candidates")
            shown = record.get("shown_order")
            if task == "select" and record["status"] != "invalid_input":
                if not isinstance(shown, list) or not 1 <= len(shown) <= 26 or len(set(shown)) != len(shown):
                    raise ValueError("Attempted selection requires its complete displayed letter mapping")
            if shown is not None and Counter(shown) != Counter(candidates_for(row)):
                raise ValueError("Displayed candidates differ from the shared inventory")
            if record["status"] != "invalid_input" and not isinstance(record.get("prompt"), str):
                raise ValueError("Saved LLM attempt must retain its prompt")
            # Old artifacts predate these optional fields. Do not reconstruct their
            # prompt with today's builder: historical wording remains authoritative.
            if "prompt_sha256" in record and record["prompt_sha256"] != hashlib.sha256(record["prompt"].encode("utf-8")).hexdigest():
                raise ValueError("Stored prompt hash differs from the saved prompt")
            if "prompt_sentence" in record and record["prompt_sentence"] != mark_span(row["sentence"], explicit_span(row)):
                raise ValueError("Stored marked sentence differs from the identified occurrence")
    if artifact["format_version"] == 2:
        for row in artifact["items"]:
            for task in ("generate", "select"):
                attempts = [record for record in artifact["records"] if record["item_id"] == row["item_id"]
                            and record["system"] != "dictabert" and record["task"] == task and record.get("prompt") is not None]
                if len(attempts) == 2 and any(attempts[0].get(key) != attempts[1].get(key)
                                              for key in ("prompt", "prompt_sentence", "shown_order")):
                    raise ValueError("Providers must share the same prompt and displayed candidate order")
        key = os.environ.get("GEMINI_API_KEY")
        if key and key in json.dumps(artifact, ensure_ascii=False):
            raise ValueError("A secret was found in the artifact; keep the API key only in the environment")
    origins = [record["encoder_origin"] for record in artifact["records"]
               if record["condition"] == "dictabert" and record["status"] != "not_run"]
    if origins and any(origin != origins[0] for origin in origins):
        raise ValueError("Mixed encoder checkpoint/run provenance")
    return artifact


def save_artifact(artifact, output_dir, *, create=False):
    """Create a fresh run directory, then atomically update only that same run's JSON."""
    validate_artifact(artifact, expected_run_id=artifact["run_id"])
    folder = Path(output_dir).expanduser().resolve()
    if any((parent / ".git").exists() for parent in (folder, *folder.parents)):
        raise ValueError("Run outputs must be outside Git working trees")
    if create:
        folder.mkdir(parents=True, exist_ok=False)
    path = folder / "study.json"
    if not create:
        existing = load_artifact(path, expected_run_id=artifact["run_id"], rows=artifact["items"])
        if existing["settings"] != artifact["settings"]:
            raise ValueError("Cannot overwrite a run with changed settings")
    temporary = folder / "study.json.tmp"
    temporary.write_text(json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def load_artifact(path, *, expected_run_id, rows=None):
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_artifact(artifact, expected_run_id=expected_run_id, rows=rows)


def saved_encoder_predictions(path, *, expected_run_id, rows):
    artifact = load_artifact(path, expected_run_id=expected_run_id, rows=rows)
    records = [record for record in artifact["records"] if record["condition"] == "dictabert"]
    if any(record["status"] == "not_run" for record in records):
        raise ValueError("Saved encoder predictions include items that were never run")
    origin = deepcopy(records[0]["encoder_origin"])
    origin["reused_from"] = {"source_run_id": artifact["run_id"],
                           "artifact_path": str(Path(path).resolve()),
                           "source_provenance": artifact["provenance"]}
    return records, origin


def inspect_results(artifact):
    """Derive current shared dev metrics without rewriting historical raw output."""
    validate_artifact(artifact, expected_run_id=artifact["run_id"])
    inspection = prompts.inspect_predictions(artifact["items"], artifact["records"])
    inspection["cohort"] = artifact.get("cohort", {
        "kind": "legacy_or_unspecified", "requested_items": len(artifact["items"]),
        "available_items": None, "full_dev": False,
        "note": "Historical artifact: full qualified-dev coverage was not independently bound"})
    return inspection


def display_tables(artifact, *, max_items=10, start=0):
    """Bounded, stacked item panels: inspect another page with start/max_items."""
    from html import escape
    if type(start) is not int or start < 0 or type(max_items) is not int or not 1 <= max_items <= 62:
        raise ValueError("Use start >= 0 and max_items between 1 and 62")
    result = inspect_results(artifact)
    def text(value):
        return escape("—" if value is None or value == "" else str(value))
    scope = result["cohort"]
    html = '<section style="max-width:100%;overflow-wrap:anywhere;font-size:14px">'
    html += f"<h3>{text(scope['kind'])}: {scope['requested_items']} requested / {text(scope.get('available_items'))} available</h3>"
    if not scope.get("full_dev"):
        html += "<p><strong>This is not a full-dev result.</strong></p>"
    html += "<table><tr><th>System · selection</th><th>State</th><th>Micro</th><th>Macro</th><th>Items</th><th>Attempted</th><th>Failures</th></tr>"
    for metric in result["metrics"]:
        values = [metric["system"], metric["execution_status"],
                  *[f"{metric[k]:.3f}" if metric[k] is not None else "not measured" if metric["execution_status"] == "not_run" else "unavailable" for k in ("micro_accuracy", "macro_accuracy")],
                  metric["n_items"], metric["n_attempted"], metric["n_failures"]]
        html += "<tr>" + "".join(f"<td style='padding:6px'>{text(value)}</td>" for value in values) + "</tr>"
    html += "</table><p>Preliminary dev selection accuracy. Failures and unrun items remain in the denominator of an attempted system. An entirely unrun system is not measured. Generation requires manual review.</p>"
    for metric in result["metrics"]:
        html += "".join(f"<p>{text(warning)}</p>" for warning in metric["warnings"])
    stop = min(start + max_items, len(result["items"]))
    html += f"<p>Showing items {min(start + 1, stop)}–{stop} of {len(result['items'])}. Use start/max_items for another page; all items remain in results.</p>"
    for item in result["items"][start:stop]:
        html += f"<details style='border:1px solid #ddd;padding:10px;margin:8px 0'><summary>{text(item['item_id'])} · {text(item['target_raw'])}" + (" · disagreement" if item["disagreement"] else "") + "</summary>"
        for label, value in (("Type", item["type_id"]), ("Original sentence", item["sentence"]),
                             ("Marked target", item["marked_sentence"]), ("Span", item["span"]),
                             ("Gold", item["gold"]), ("Candidates", item["candidates"])):
            html += f"<p><strong>{label}:</strong> <span dir='auto'>{text(value)}</span></p>"
        for system, data in item["systems"].items():
            html += f"<h4>{text(system)}</h4>"
            labels = [("Decoded selection", "selection_decoded"), ("Selection state", "selection_status")]
            if system != "dictabert":
                mapping = " | ".join(f"{letter}: {candidate}" for letter, candidate in data["letter_mapping"].items())
                html += f"<p><strong>Displayed letters:</strong> <span dir='auto'>{text(mapping)}</span></p>"
                labels += [("Selection raw", "selection_raw"), ("Generation raw · manual review", "generation_raw"),
                           ("Generation state", "generation_status")]
            labels += [("Selection failure", "selection_error"), ("Generation failure", "generation_error")]
            for label, key in labels:
                if key.endswith("error") and not data.get(key):
                    continue
                value = data.get(key)
                if isinstance(value, str) and len(value) > 1500:
                    html += f"<p><strong>{label}:</strong> {text(value[:1500])}…</p><details><summary>Complete raw text</summary><pre style='white-space:pre-wrap'>{text(value)}</pre></details>"
                else:
                    html += f"<p><strong>{label}:</strong> <span dir='auto' style='white-space:pre-wrap'>{text(value)}</span></p>"
        html += "</details>"
    html += f"<h3>Disagreements between valid selections: {len(result['disagreements'])}</h3>"
    html += "<p>" + (", ".join(text(item["item_id"]) for item in result["disagreements"]) or "None among valid selections; failures remain in each item.") + "</p></section>"
    return html, result


def read_study_inputs(settings):
    """Choose invented preview, saved snapshot, or explicitly requested qualified dev."""
    mode = settings["mode"]
    if mode == "reload":
        if not settings.get("saved_run") or not settings.get("saved_run_id"):
            raise ValueError("Reload requires an explicit saved_run path and saved_run_id")
        artifact = load_artifact(settings["saved_run"], expected_run_id=settings["saved_run_id"])
        return artifact["items"], artifact.get("cohort")
    if mode == "preview":
        rows = fixture_rows()
        return rows, {"kind": "invented_fixture", "requested_items": len(rows), "available_items": len(rows), "full_dev": False}
    if mode != "run" or settings["run_kind"] not in {"validation", "full_dev"}:
        raise ValueError("Choose mode preview/run/reload and run_kind validation/full_dev")
    rows = load_rows(settings["input_path"])
    expected = settings.get("expected_dev_items", 62)
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} qualified dev items; verify the selected input")
    identity = input_identity(rows)
    if settings["run_kind"] == "validation":
        count = settings.get("validation_items", 3)
        if type(count) is not int or not 1 <= count < len(rows):
            raise ValueError("Validation requires an explicit positive subset smaller than the full dev input")
        rows = rows[:count]
    cohort = {"kind": settings["run_kind"], "available_items": expected, "requested_items": len(rows),
              "full_dev": settings["run_kind"] == "full_dev", "full_input_identity": identity}
    return rows, cohort


def prepare_study(rows, cohort, settings):
    """Initialize an inspectable run and initial artifact; no model/service calls."""
    if settings["mode"] == "reload":
        return load_artifact(settings["saved_run"], expected_run_id=settings["saved_run_id"], rows=rows)
    output_dir = Path(settings["output_root"]) / settings["run_id"]
    check_readiness(settings["root"], settings["input_path"], settings.get("checkpoint"), output_dir,
                    run=settings["mode"] == "run", encoder=settings["enable_encoder"], llm=settings["enable_qwen"],
                    model=settings.get("qwen_model"), saved_encoder=settings.get("saved_encoder"))
    stored = {key: str(value) if isinstance(value, Path) else deepcopy(value) for key, value in settings.items()}
    stored.update(selection_shuffle_seed=prompts.SHUFFLE_SEED, prompt_scope="exact_marked_occurrence_with_sentence",
                  scoring="derived preliminary dev selection accuracy; generation manual review")
    artifact = new_artifact(rows, settings["run_id"], stored, source_provenance(settings["root"]))
    artifact["cohort"] = deepcopy(cohort)
    if settings["mode"] == "run":
        save_artifact(artifact, output_dir, create=True)
    return artifact


def save_study(artifact):
    """Save a run to its recorded output directory; notebook reload never calls this."""
    settings = artifact["settings"]
    return save_artifact(artifact, Path(settings["output_root"]) / artifact["run_id"])


def current_encoder_inputs(artifact, rows):
    """Read qualified train/full dev only for an explicit encoder run."""
    settings = artifact["settings"]
    if settings.get("mode") != "run":
        raise ValueError("Encoder input verification requires mode='run'")
    identities = {}
    for split, setting in (("train", "train_path"), ("dev", "input_path")):
        if not settings.get(setting):
            raise ValueError(f"Set {setting} to the current qualified {split} CSV")
        source_rows = load_rows(Path(settings[setting]).expanduser())
        validate_ids(source_rows)
        identities[split] = input_identity(source_rows)
        if split == "dev":
            if input_identity(source_rows[:len(rows)]) != input_identity(rows):
                raise ValueError("Prediction rows differ from the current qualified dev input")
            if artifact.get("cohort", {}).get("full_input_identity") != identities["dev"]:
                raise ValueError("Qualified dev changed since this run was prepared; start a new run")
    return identities


def check_checkpoint_inputs(metadata, expected_inputs):
    """Compare independently computed identities without changing the sidecar."""
    stored = metadata.get("inputs") if isinstance(metadata, dict) else None
    if not isinstance(stored, dict):
        raise ValueError("Checkpoint lacks original train/dev identities; obtain the original JSON")
    for split in ("train", "dev"):
        if split not in stored:
            raise ValueError(f"Checkpoint lacks the original {split} identity")
        if stored[split] != expected_inputs[split]:
            raise ValueError(f"Checkpoint {split} input differs from the current qualified {split} data")
    if stored != expected_inputs:
        raise ValueError("Checkpoint input metadata differs from the train/dev contract")


def checkpoint_report(artifact):
    """Return full checkpoint metadata and a compact display summary."""
    records = [record for record in artifact["records"] if record["condition"] == "dictabert"]
    origin = next((record.get("encoder_origin") for record in records if record.get("encoder_origin")), None)
    if not origin:
        return None, {"state": "not run"}
    metadata = origin.get("checkpoint_metadata", {})
    inputs = metadata.get("inputs", {})
    return metadata, {"training_config": metadata.get("training_config"),
            "selected_epoch": metadata.get("selection", {}).get("epoch"),
            "train_items": len(inputs["train"]["item_ids"]) if inputs.get("train", {}).get("item_ids") is not None else None,
            "dev_items": len(inputs["dev"]["item_ids"]) if inputs.get("dev", {}).get("item_ids") is not None else None,
            "input_match": origin.get("input_match", "not checked for this historical run")}


def predict_encoder(artifact, rows, *, checkpoint, snapshot_path=None, device="cpu", saved_path=None, saved_run_id=None):
    """Load/evaluate/release the encoder using original model APIs; no training."""
    from hebrew_acronyms.models.dictabert_cross_encoder.model import load_finetuned, file_digest, metadata_path
    from hebrew_acronyms.models.dictabert_cross_encoder.eval import evaluate
    from hebrew_acronyms.models.dictabert_cross_encoder.workflow import enable_offline
    validate_artifact(artifact, expected_run_id=artifact["run_id"], rows=rows)
    actual_settings = {"checkpoint": checkpoint, "snapshot_path": snapshot_path, "device": device,
                       "saved_encoder": saved_path, "saved_encoder_run_id": saved_run_id}
    for key, value in actual_settings.items():
        value = str(value) if isinstance(value, Path) else value
        if key in artifact["settings"] and value != artifact["settings"][key]:
            raise ValueError(f"Encoder setting {key} differs from the recorded run; start a new run")
    if any(record["status"] != "not_run" for record in artifact["records"] if record["condition"] == "dictabert"):
        raise ValueError("Encoder predictions already exist; start a new run")
    expected_inputs = current_encoder_inputs(artifact, rows)
    if saved_path is not None:
        predictions, origin = saved_encoder_predictions(saved_path, expected_run_id=saved_run_id, rows=rows)
        check_checkpoint_inputs(origin.get("checkpoint_metadata"), expected_inputs)
    else:
        checkpoint = Path(checkpoint).expanduser().resolve()
        sidecar = metadata_path(checkpoint)
        if not sidecar.is_file():
            raise ValueError("Checkpoint metadata is required; obtain Ben's original JSON")
        check_checkpoint_inputs(json.loads(sidecar.read_text(encoding="utf-8")), expected_inputs)
        enable_offline()
        tokenizer, model, opened, closed = load_finetuned(
            str(checkpoint), device=device, snapshot_path=snapshot_path, expected_inputs=expected_inputs)
        try:
            predictions = evaluate(rows, tokenizer, model, opened, closed, device)
            origin = {"checkpoint": str(Path(checkpoint).resolve()), "weights_sha256": file_digest(checkpoint),
                      "manifest_sha256": file_digest(str(checkpoint) + ".json"),
                      "checkpoint_metadata": model.checkpoint_metadata,
                      "snapshot_path": str(snapshot_path) if snapshot_path is not None else None, "device": device,
                      "load_identity": getattr(model, "checkpoint_load_identity", None)}
        finally:
            del model, tokenizer
            gc.collect()
            if device.startswith("cuda"):
                import torch
                torch.cuda.empty_cache()
            elif device == "mps":
                import torch
                torch.mps.empty_cache()
    origin["input_match"] = "matched current qualified train and full dev"
    origin["expected_inputs"] = expected_inputs
    attach_encoder(artifact, predictions, origin=origin)
    save_study(artifact)
    return predictions


def _check_unstarted(artifact, system):
    specs = arm_specs(artifact)
    if any(record["status"] != "not_run" for record in artifact["records"]
           if specs[record["condition"]][0] == system):
        raise ValueError("Responses already exist; start a new run")


def connect_qwen(artifact):
    """Resolve Qwen once; a connection failure becomes records without blocking Gemini."""
    from functools import partial
    from hebrew_acronyms.models.qwen.eval import inspect_ollama, ollama_response
    _check_unstarted(artifact, "qwen")
    settings = artifact["settings"]
    try:
        runtime = inspect_ollama(settings["qwen_model"], base_url=settings["ollama_url"],
                                 timeout=settings["request_timeout"], options=settings["qwen_options"])
    except Exception as error:
        message = _without_secret(f"Ollama connection failed ({type(error).__name__}); check service, installed tag and timeout")
        runtime = {"model": settings["qwen_model"], "digest": None, "connection_error": message}
        _set_runtime(artifact, "qwen", runtime)
        save_study(artifact)
        return lambda prompt: {"response": "", "status": "service_error", "error": message}
    _set_runtime(artifact, "qwen", runtime)
    save_study(artifact)
    return partial(ollama_response, model=runtime["model"], expected_digest=runtime["digest"],
                   base_url=runtime["base_url"], timeout=runtime["timeout_seconds"], options=deepcopy(runtime["options"]))


def connect_gemini(artifact):
    """Bind explicit Gemini settings. No network request or API key enters the artifact."""
    from functools import partial
    from dotenv import load_dotenv
    from hebrew_acronyms.models.gemini.eval import gemini_response, validate_gemini_settings
    _check_unstarted(artifact, "gemini")
    settings = artifact["settings"]
    load_dotenv(Path(settings["root"]) / ".env", override=False, interpolate=False)
    if not settings.get("gemini_model"):
        raise ValueError("Set an explicit gemini_model before enabling Gemini")
    config = validate_gemini_settings(settings["gemini_model"], settings["request_timeout"], settings["gemini_generation_config"])
    runtime = {"requested_model": settings["gemini_model"], "model_version": None,
               "version_evidence": "not yet reported", "request_settings": {
                   "generationConfig": config, "timeout_seconds": settings["request_timeout"], "max_attempts": 1}}
    _set_runtime(artifact, "gemini", runtime)
    save_study(artifact)
    return partial(gemini_response, model=runtime["requested_model"], timeout=settings["request_timeout"],
                   generation_config=deepcopy(settings["gemini_generation_config"]))
