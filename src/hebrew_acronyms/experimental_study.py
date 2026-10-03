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
from pathlib import Path
import platform
import random
import subprocess
from importlib.metadata import version

from hebrew_acronyms.models.common import eval as prompts
from hebrew_acronyms.models.common.pairs import candidates_for, explicit_span, input_identity, validate_ids, mark_span, load_rows

CONDITIONS = ("dictabert", "generate", "select")


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
                    model, revision, saved_encoder=None):
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
              for name in ("qwen", "dictabert_cross_encoder")]
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
                "item_id": row["item_id"], "condition": condition, "status": "not_run",
                "score_status": "unscored", "raw_response": None, "error": None}
               for condition in CONDITIONS for row in rows]
    return {"format_version": 1, "run_id": run_id, "input_identity": identity,
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


def collect_responses(artifact, mode, generate_fn, *, on_record=None, limit=None):
    """Collect raw responses with existing prompts. No parsing, scoring or row skipping.

    on_record may save after every attempt. Provider exceptions remain records;
    KeyboardInterrupt propagates, with the interrupted record retained by finally.
    """
    if mode not in {"generate", "select"}:
        raise ValueError("Only generate and select with the sentence are supported")
    revision = artifact.get("llm_runtime", {}).get("digest") or artifact["settings"].get("qwen_revision")
    if not artifact["settings"].get("qwen_model") or not revision:
        raise ValueError("Explicit Qwen model and resolved version/digest are required")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("Response limit must be a positive integer or None")
    validate_artifact(artifact, expected_run_id=artifact["run_id"])
    if any(record["status"] != "not_run" for record in artifact["records"] if record["condition"] == mode):
        raise ValueError("Responses already exist; start a new run instead of overwriting")
    rng = random.Random(prompts.SHUFFLE_SEED)
    items = {row["item_id"]: row for row in artifact["items"]}
    pending = [record for record in artifact["records"] if record["condition"] == mode]
    for record in pending[:limit]:
        if record["status"] != "not_run":
            raise ValueError("Responses already exist; start a new run instead of overwriting")
        row = items[record["item_id"]]
        record.update(model=artifact["settings"]["qwen_model"],
                      model_revision=revision,
                      sentence=row.get("sentence"), target_raw=row.get("target_raw"),
                      prompt=None, shown_order=None)
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
                response = generate_fn(prompt)
                if isinstance(response, dict):
                    record["backend_metadata"] = {key: deepcopy(value) for key, value in response.items() if key != "response"}
                    raw = response.get("response")
                    if not isinstance(raw, str):
                        raise TypeError("Backend response must contain text")
                    record.update(status="response_received", raw_response=raw)
                    if response.get("identity_status") != "verified":
                        record.update(status="model_identity_error", error=response.get("error"))
                elif isinstance(response, str):
                    record.update(status="response_received", raw_response=response)
                else:
                    raise TypeError("Backend response must be text or a metadata-bearing response")
            except Exception as error:
                record.update(status="service_error", error=f"{type(error).__name__}: {error}")
            finally:
                if on_record is not None:
                    on_record(artifact)
            continue
        if on_record is not None:
            on_record(artifact)
    return artifact


def validate_artifact(artifact, *, expected_run_id, rows=None):
    if not isinstance(artifact, dict) or artifact.get("format_version") != 1:
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
    runtime = artifact.get("llm_runtime")
    if runtime is not None and (runtime.get("model") != artifact["settings"].get("qwen_model") or not runtime.get("digest")):
        raise ValueError("Runtime LLM identity disagrees with run settings")
    expected = {(condition, item_id) for condition in CONDITIONS for item_id in identity["item_ids"]}
    actual = [(record["condition"], record["item_id"]) for record in artifact["records"]]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Every condition must retain exactly one record per item")
    items = {row["item_id"]: row for row in artifact["items"]}
    for record in artifact["records"]:
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
        if record["condition"] in {"generate", "select"} and record["status"] != "not_run":
            if record["status"] not in {"invalid_input", "interrupted", "service_error", "response_received", "model_identity_error"}:
                raise ValueError("Unknown LLM status")
            if record["status"] in {"response_received", "model_identity_error"} and not isinstance(record.get("raw_response"), str):
                raise ValueError("Received LLM response must retain raw text")
            if (record.get("model") != artifact["settings"].get("qwen_model") or
                    record.get("model_revision") != (artifact.get("llm_runtime", {}).get("digest") or artifact["settings"].get("qwen_revision"))):
                raise ValueError("Mixed LLM model/version settings")
            evidence = record.get("backend_metadata")
            if record["status"] == "response_received" and evidence is not None:
                digest = record["model_revision"]
                request = evidence.get("request_settings", {})
                if (evidence.get("identity_status") != "verified"
                        or any(evidence.get(key) != digest for key in ("expected_digest", "digest_before", "digest_after"))
                        or evidence.get("model") != record["model"] or request.get("model") != record["model"]
                        or (runtime is not None and request.get("options") != runtime.get("options"))):
                    raise ValueError("Backend response evidence differs from the bound model/digest/settings")
            row = items[record["item_id"]]
            if record.get("sentence") != row.get("sentence") or record.get("target_raw") != row.get("target_raw"):
                raise ValueError("LLM input differs from the shared item")
            if record["condition"] == "generate" and record.get("shown_order") is not None:
                raise ValueError("Generation must not receive candidates")
            shown = record.get("shown_order")
            if record["condition"] == "select" and record["status"] != "invalid_input":
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


def display_tables(artifact):
    """Readable per-item table plus derived metrics; generation stays for manual review."""
    from html import escape
    inspection = inspect_results(artifact)
    columns = [("item_id", "Item"), ("type_id", "Type"), ("sentence", "Original sentence"),
               ("target_raw", "Target"), ("span", "Span"), ("gold", "Gold"),
               ("encoder_prediction", "DictaBERT"), ("encoder_status", "Encoder status"),
               ("generation_raw", "Generation · manual review"), ("generation_status", "Generation status"),
               ("selection_raw", "Selection raw"), ("letter_mapping", "Displayed letters"),
               ("selection_decoded", "Decoded choice"), ("selection_status", "Choice status"), ("failures", "Failures")]
    table = '<div style="overflow-x:auto"><table style="border-collapse:collapse;font-size:13px"><thead><tr>'
    table += "".join(f"<th style='padding:8px;border:1px solid #ddd'>{label}</th>" for _, label in columns)
    table += "</tr></thead><tbody>"
    for item in inspection["items"]:
        table += "<tr>"
        for key, _ in columns:
            value = item[key]
            if isinstance(value, dict):
                value = "\n".join(f"{label}: {text}" for label, text in value.items())
            text = "—" if value is None or value == "" else str(value)
            table += f"<td style='padding:8px;border:1px solid #ddd;min-width:90px;white-space:pre-wrap'>{escape(text)}</td>"
        table += "</tr>"
    table += "</tbody></table></div>"
    scope = inspection["cohort"]
    heading = f"<h3>{escape(scope['kind'])}: {scope['requested_items']} requested / {scope.get('available_items', 'unknown')} available items</h3>"
    if not scope.get("full_dev"):
        heading += "<p><strong>This is not a full-dev result.</strong></p>"
    metrics_table = "<table><tr><th>System</th><th>Micro</th><th>Macro by type</th><th>Attempted / requested</th><th>Valid predictions</th><th>Unfinished</th></tr>"
    for metric in inspection["metrics"]:
        micro, macro = metric["micro_accuracy"], metric["macro_accuracy"]
        metrics_table += (f"<tr><td>{metric['condition']}</td><td>{f'{micro:.3f}' if micro is not None else 'unavailable'}</td>"
                          f"<td>{f'{macro:.3f}' if macro is not None else 'unavailable'}</td>"
                          f"<td>{metric['n_attempted']} / {metric['n_items']}</td><td>{metric['n_valid_predictions']}</td>"
                          f"<td>{metric['partial']}</td></tr>")
    metrics_table += "</table><p>Preliminary dev selection accuracy; failures and unrun items stay in the denominator. Generation is unscored manual review.</p>"
    warnings = "".join(f"<p>{escape(warning)}</p>" for metric in inspection["metrics"] for warning in metric["warnings"])
    disagreements = "<h3>Disagreements between valid selections</h3><ul>"
    for item in inspection["disagreements"]:
        disagreements += f"<li>{escape(item['item_id'])}: DictaBERT = {escape(item['encoder_prediction'])}; Qwen = {escape(item['selection_decoded'])}</li>"
    disagreements += "</ul>" if inspection["disagreements"] else "<li>None among items with two valid selections; failures remain in the item table.</li></ul>"
    return heading + metrics_table + warnings + table + disagreements, inspection


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
                    run=settings["mode"] == "run", encoder=settings["enable_encoder"], llm=settings["enable_llm"],
                    model=settings.get("qwen_model"), revision=None, saved_encoder=settings.get("saved_encoder"))
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


def predict_encoder(artifact, rows, *, checkpoint, snapshot_path=None, device="cpu", saved_path=None, saved_run_id=None):
    """Load/evaluate/release the encoder using original model APIs; no training."""
    from hebrew_acronyms.models.dictabert_cross_encoder.model import load_finetuned, file_digest
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
    if saved_path is not None:
        predictions, origin = saved_encoder_predictions(saved_path, expected_run_id=saved_run_id, rows=rows)
    else:
        checkpoint = Path(checkpoint).expanduser().resolve()
        enable_offline()
        tokenizer, model, opened, closed = load_finetuned(str(checkpoint), device=device, snapshot_path=snapshot_path)
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
    attach_encoder(artifact, predictions, origin=origin)
    save_study(artifact)
    return predictions


def connect_qwen(artifact):
    """Resolve local Qwen identity/settings explicitly and bind one callable for both modes."""
    from functools import partial
    from hebrew_acronyms.models.qwen.eval import inspect_ollama, ollama_response
    settings = artifact["settings"]
    if any(record["status"] != "not_run" for record in artifact["records"] if record["condition"] in {"generate", "select"}):
        raise ValueError("Qwen responses already exist; start a new run")
    runtime = inspect_ollama(settings["qwen_model"], base_url=settings["ollama_url"],
                             timeout=settings["request_timeout"], options=settings["qwen_options"])
    artifact["llm_runtime"] = runtime
    save_study(artifact)
    return partial(ollama_response, model=runtime["model"], expected_digest=runtime["digest"],
                   base_url=runtime["base_url"], timeout=runtime["timeout_seconds"], options=deepcopy(runtime["options"]))
