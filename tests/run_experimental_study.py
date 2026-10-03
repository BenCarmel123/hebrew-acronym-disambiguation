"""Execute unchanged study cells on invented inputs with guarded, mocked interfaces."""
import ast
import csv
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def guard(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto", "socket.__new__"}:
        raise RuntimeError("Network forbidden in study fixture test process")
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(args[0])).resolve()
        for parent in (ROOT, ROOT.parent / "hebrew-acronym-disambiguation"):
            if any(path.is_relative_to(parent / name) for name in ("data", "results", "weights", "sources")):
                raise RuntimeError(f"Research I/O forbidden in study fixture: {path}")


def predictions(rows):
    return [{"item_id": row["item_id"], "status": "ok", "selected_candidate": row["candidates"].split("|")[0],
             "selected_index": 0, "candidate_scores": [{"candidate": candidate, "score": float(-i)}
               for i, candidate in enumerate(row["candidates"].split("|"))], "error": None}
            for row in rows]


def runtime(model, *, base_url, timeout, options):
    return {"model": model, "digest": "fixture-digest", "base_url": base_url,
            "timeout_seconds": timeout, "options": options, "identity_verification": "mock_server",
            "server_version": "invented", "model_parameters": "fixture defaults"}


def response(prompt, *, model, expected_digest, base_url, timeout, options):
    return {"response": "אור בוקר" if "פירוש:" in prompt else "A", "model": model,
            "expected_digest": expected_digest, "identity_status": "verified",
            "status": "response_received", "completion_status": "complete",
            "response_metadata": {"done": True, "done_reason": "stop"},
            "digest_before": expected_digest, "digest_after": expected_digest,
            "request_settings": {"model": model, "options": options, "stream": False}}


def legacy_artifact(rows, run_id, settings, provenance):
    """Frozen format-1 fixture shape from 1a8708f; never invent Gemini records."""
    from hebrew_acronyms.models.common.pairs import input_identity
    identity = input_identity(rows)
    records = [{"run_id": run_id, "input_sha256": identity["sha256"],
                "item_id": row["item_id"], "condition": condition, "status": "not_run",
                "score_status": "unscored", "raw_response": None, "error": None}
               for condition in ("dictabert", "generate", "select") for row in rows]
    return {"format_version": 1, "run_id": run_id, "input_identity": identity,
            "items": deepcopy(rows), "settings": deepcopy(settings),
            "provenance": deepcopy(provenance), "records": records}


def execute(controls=None, *, qwen_failure=False, gemini_failure=False):
    from hebrew_acronyms.models.dictabert_cross_encoder import model, eval as encoder_eval
    from hebrew_acronyms.models.qwen import eval as qwen_eval
    from hebrew_acronyms.models.gemini import eval as gemini_eval
    namespace = {"__name__": "__main__"}
    notebook = json.loads((ROOT / "notebooks/experimental_study.ipynb").read_text())
    display_module = SimpleNamespace(HTML=lambda value: value, display=lambda value: None)
    def fake_load(checkpoint, **kwargs):
        metadata = json.loads(Path(str(checkpoint) + ".json").read_text())
        assert kwargs["expected_inputs"] == metadata["inputs"]
        return object(), SimpleNamespace(checkpoint_metadata=metadata), 1, 2

    def gemini_http(*args, **kwargs):
        if gemini_failure:
            raise gemini_eval.requests.Timeout("invented secret must not leak")
        prompt = kwargs["json"]["contents"][0]["parts"][0]["text"]
        payload = {"modelVersion": "gemini-3.8-flash-fixture", "usageMetadata": {"totalTokenCount": 10},
                   "candidates": [{"finishReason": "STOP", "content": {"parts": [
                       {"text": "not a final answer", "thought": True},
                       {"text": "אור בוקר" if "פירוש:" in prompt else "A"}]}}]}
        return SimpleNamespace(status_code=200, json=lambda: payload)
    with (patch.dict(sys.modules, {"IPython": SimpleNamespace(), "IPython.display": display_module}),
          patch.dict(os.environ),
          patch.object(model, "inspect_checkpoint", autospec=True, return_value={"fixture": True}) as inspect,
          patch.object(model, "load_finetuned", autospec=True,
                       side_effect=fake_load) as load,
          patch.object(encoder_eval, "evaluate", autospec=True, side_effect=lambda rows, *a: predictions(rows)) as evaluate,
          patch.object(qwen_eval, "inspect_ollama", autospec=True,
                       side_effect=RuntimeError("fixture unavailable") if qwen_failure else runtime) as inspect_llm,
          patch.object(qwen_eval, "ollama_response", autospec=True, side_effect=response) as generate,
          patch.object(gemini_eval.requests, "post", side_effect=gemini_http) as gemini_post):
        os.environ.pop("GEMINI_API_KEY", None)
        if controls and controls.get("enable_gemini") and controls.get("mode") == "run":
            os.environ["GEMINI_API_KEY"] = "invented-key-for-fixture-only"
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            if index == 1:
                for statement in ast.parse(source).body:
                    exec(compile(ast.Module(body=[statement], type_ignores=[]), "study-settings-cell", "exec"), namespace)
                    if isinstance(statement, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "settings" for target in statement.targets):
                        namespace["settings"].update(controls or {})
            else:
                exec(compile(source, f"experimental_study.ipynb:{index}", "exec"), namespace)
        settings = namespace["settings"]
        mode = settings["mode"]
        count = 2 if settings["run_kind"] == "validation" else 2 * len(namespace["rows"])
        if mode == "run" and settings["enable_qwen"]:
            inspect_llm.assert_called_once()
            assert generate.call_count == (0 if qwen_failure else count)
        else:
            inspect_llm.assert_not_called();generate.assert_not_called()
        if mode == "run" and settings["enable_gemini"]:
            assert gemini_post.call_count == count
            assert all('key=' not in call.args[0] for call in gemini_post.call_args_list)
        else:
            gemini_post.assert_not_called()
        if mode == "run" and settings["enable_encoder"]:
            load.assert_called_once();evaluate.assert_called_once()
        else:
            load.assert_not_called();evaluate.assert_not_called()
        inspect.assert_not_called()
    return namespace


def main():
    sys.addaudithook(guard)
    from hebrew_acronyms import experimental_study as study
    default = execute()
    assert all(record["status"] == "not_run" for record in default["artifact"]["records"])
    assert default["artifact_path"] is None
    with tempfile.TemporaryDirectory(prefix="study-fixtures-") as temp:
        folder = Path(temp)
        rows = study.fixture_rows()
        rows += [dict(row, item_id=row["item_id"] + "-copy") for row in rows]
        input_path = folder / "invented.csv"
        with input_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
        checkpoint = folder / "invented.pt"
        checkpoint.write_text("not weights; mocked load")
        train_path = folder / "invented-train.csv"
        with train_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0]);writer.writeheader()
            writer.writerows([dict(row, item_id="train-" + row["item_id"]) for row in rows])
        metadata = {"inputs": {"train": study.input_identity(study.load_rows(train_path)),
                               "dev": study.input_identity(study.load_rows(input_path))},
                    "training_config": {"epochs": 2}, "selection": {"epoch": 1}}
        Path(str(checkpoint) + ".json").write_text(json.dumps(metadata))
        controls = dict(mode="run", train_path=train_path, input_path=input_path, expected_dev_items=4, validation_items=3,
                        enable_encoder=True, enable_qwen=True, enable_gemini=True, checkpoint=checkpoint, device="cpu",
                        output_root=folder, run_id="validation-one", run_kind="validation")
        validation = execute(controls)
        assert len(validation["rows"]) == 3
        assert validation["results"]["cohort"]["full_dev"] is False
        assert validation["results"]["metrics"][1]["n_attempted"] == 1
        assert validation["results"]["metrics"][1]["partial"] is True
        encoder_only = execute(dict(controls, run_id="encoder-only", enable_qwen=False, enable_gemini=False, qwen_model=None))
        assert encoder_only["results"]["metrics"][0]["n_valid_predictions"] == 3
        assert encoder_only["results"]["metrics"][1]["n_attempted"] == 0
        qwen_only = execute(dict(controls, run_id="qwen-only", enable_encoder=False, checkpoint=None, enable_gemini=False, gemini_model=None))
        assert qwen_only["results"]["metrics"][2]["execution_status"] == "not_run"
        gemini_only = execute(dict(controls, run_id="gemini-only", enable_encoder=False, checkpoint=None, enable_qwen=False, qwen_model=None))
        assert gemini_only["results"]["metrics"][1]["micro_accuracy"] is None
        qwen_failed = execute(dict(controls, run_id="qwen-failed"), qwen_failure=True)
        assert qwen_failed["results"]["metrics"][2]["n_valid_predictions"] == 1
        gemini_failed = execute(dict(controls, run_id="gemini-failed"), gemini_failure=True)
        assert gemini_failed["results"]["metrics"][1]["n_valid_predictions"] == 1
        assert gemini_failed["results"]["metrics"][2]["n_failures"] == 1
        full = execute(dict(controls, run_id="full-one", run_kind="full_dev"))
        assert len(full["rows"]) == 4
        assert all(not metric["partial"] for metric in full["results"]["metrics"])
        reloaded = execute(dict(mode="reload", saved_run=full["artifact_path"], saved_run_id="full-one",
                               enable_qwen=True, enable_gemini=True, enable_encoder=True, qwen_model=None))
        assert reloaded["artifact"] == full["artifact"]
        old = legacy_artifact(rows, "old-run", {"qwen_model": "old-model", "qwen_revision": "old-version"}, {})
        study.attach_encoder(old, predictions(rows), origin={"weights_sha256": "a"*64, "manifest_sha256": "b"*64})
        study.collect_responses(old, "generate", lambda prompt: "historic final")
        study.collect_responses(old, "select", lambda prompt: "A")
        old_path = study.save_artifact(old, folder / "old", create=True)
        original_bytes = old_path.read_bytes()
        old_reload = execute(dict(mode="reload", saved_run=old_path, saved_run_id="old-run", enable_gemini=True))
        assert old_reload["artifact"] == old and old_path.read_bytes() == original_bytes
        assert len(old_reload["results"]["metrics"]) == 2
        assert all("gemini" not in item["systems"] for item in old_reload["results"]["items"])
        reused = execute(dict(controls, run_id="saved-encoder", run_kind="full_dev", enable_encoder=False,
                              enable_qwen=False, enable_gemini=False, qwen_model=None, checkpoint=None,
                              saved_encoder=full["artifact_path"], saved_encoder_run_id="full-one"))
        assert reused["results"]["metrics"][0]["n_valid_predictions"] == 4
        assert reused["results"]["metrics"][1]["n_attempted"] == 0
    print("Study notebook: PASS (11 routes including five arms, isolated providers, failures and legacy reload; no real models/services/data)")


if __name__ == "__main__":
    main()
