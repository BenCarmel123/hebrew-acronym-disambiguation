"""Notebook glue for identified, unscored predictions and self-contained artifacts.

No models, services, filesystem operations or network hooks run at import time.
The shared prompt builders and encoder evaluator retain ownership of model behavior.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import platform
import random
import subprocess
from importlib.metadata import version

from hebrew_acronyms.models.common import eval as prompts
from hebrew_acronyms.models.common.pairs import candidates_for, explicit_span, input_identity, validate_ids

CONDITIONS = ("dictabert", "generate", "select")


def fixture_rows():
    """Invented examples only; deliberately different acronym and raw target."""
    return [
        {"item_id": "invented-1", "sentence": "היא פנתה לב״מ היום.", "acronym": "ב״מ",
         "target_raw": "לב״מ", "span_start": 9, "span_end": 13,
         "candidates": "בית מלאכה|בית מדרש", "gold_expansion": "בית מלאכה",
         "source_kind": "invented_fixture"},
        {"item_id": "invented-2", "sentence": "זהו א״ב.", "acronym": "א״ב",
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
    if llm and (not isinstance(model, str) or not model.strip()
                or not isinstance(revision, str) or not revision.strip()):
        raise ValueError("Set the exact Qwen Ollama model tag AND its resolved version/digest")
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


def collect_responses(artifact, mode, generate_fn, *, on_record=None):
    """Collect raw responses with existing prompts. No parsing, scoring or row skipping.

    on_record may save after every attempt. Provider exceptions remain records;
    KeyboardInterrupt propagates, with the interrupted record retained by finally.
    """
    if mode not in {"generate", "select"}:
        raise ValueError("Only generate and select with the sentence are supported")
    if not artifact["settings"].get("qwen_model") or not artifact["settings"].get("qwen_revision"):
        raise ValueError("Explicit Qwen model and resolved version/digest are required")
    validate_artifact(artifact, expected_run_id=artifact["run_id"])
    if any(record["status"] != "not_run" for record in artifact["records"] if record["condition"] == mode):
        raise ValueError("Responses already exist; start a new run instead of overwriting")
    rng = random.Random(prompts.SHUFFLE_SEED)
    items = {row["item_id"]: row for row in artifact["items"]}
    for record in artifact["records"]:
        if record["condition"] != mode:
            continue
        if record["status"] != "not_run":
            raise ValueError("Responses already exist; start a new run instead of overwriting")
        row = items[record["item_id"]]
        record.update(model=artifact["settings"]["qwen_model"],
                      model_revision=artifact["settings"]["qwen_revision"],
                      sentence=row.get("sentence"), target_raw=row.get("target_raw"),
                      prompt=None, shown_order=None)
        try:
            explicit_span(row)
            if mode == "generate":
                prompt = prompts.build_generate_prompt(row["target_raw"], row["sentence"])
            else:
                candidates = candidates_for(row)
                if len(set(candidates)) != len(candidates) or len(candidates) > 26:
                    raise ValueError("Selection requires 1–26 distinct candidates for the shared letter prompt")
                rng.shuffle(candidates)
                record["shown_order"] = candidates
                prompt = prompts.build_select_prompt(row["target_raw"], row["sentence"], candidates)
            record["prompt"] = prompt
        except (KeyError, ValueError) as error:
            record.update(status="invalid_input", error=str(error))
        else:
            record["status"] = "interrupted"
            try:
                response = generate_fn(prompt)
                if not isinstance(response, str):
                    raise TypeError("Backend response must be text")
                record.update(status="response_received", raw_response=response)
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
            if record["status"] not in {"invalid_input", "interrupted", "service_error", "response_received"}:
                raise ValueError("Unknown LLM status")
            if record["status"] == "response_received" and not isinstance(record.get("raw_response"), str):
                raise ValueError("Received LLM response must retain raw text")
            if (record.get("model") != artifact["settings"].get("qwen_model") or
                    record.get("model_revision") != artifact["settings"].get("qwen_revision")):
                raise ValueError("Mixed LLM model/version settings")
            row = items[record["item_id"]]
            if record.get("sentence") != row.get("sentence") or record.get("target_raw") != row.get("target_raw"):
                raise ValueError("LLM input differs from the shared item")
            if record["condition"] == "generate" and record.get("shown_order") is not None:
                raise ValueError("Generation must not receive candidates")
            shown = record.get("shown_order")
            if shown is not None and Counter(shown) != Counter(candidates_for(row)):
                raise ValueError("Displayed candidates differ from the shared inventory")
            if record["status"] != "invalid_input":
                expected_prompt = (prompts.build_generate_prompt(row["target_raw"], row["sentence"])
                                   if record["condition"] == "generate" else
                                   prompts.build_select_prompt(row["target_raw"], row["sentence"], shown))
                if record.get("prompt") != expected_prompt:
                    raise ValueError("Saved prompt differs from its identified input/order")
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


def display_tables(artifact):
    """An escaped item table and operational status counts, not research metrics."""
    from html import escape
    validate_artifact(artifact, expected_run_id=artifact["run_id"])
    fields = ("item_id", "condition", "status", "selected_candidate", "raw_response", "score_status", "error")
    table = "<table><thead><tr>" + "".join(f"<th>{field}</th>" for field in fields) + "</tr></thead><tbody>"
    for record in artifact["records"]:
        table += "<tr>" + "".join(f"<td>{escape(str(record.get(field) or '—'))}</td>" for field in fields) + "</tr>"
    table += "</tbody></table>"
    counts = Counter((record["condition"], record["status"]) for record in artifact["records"])
    summary = [{"condition": condition, "status": status, "items": count}
               for (condition, status), count in sorted(counts.items())]
    return table, summary
