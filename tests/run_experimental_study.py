"""Execute unchanged study cells on invented inputs with guarded, mocked interfaces."""
import ast
import csv
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
            "digest_before": expected_digest, "digest_after": expected_digest,
            "request_settings": {"model": model, "options": options, "stream": False}}


def execute(controls=None):
    from hebrew_acronyms.models.dictabert_cross_encoder import model, eval as encoder_eval
    from hebrew_acronyms.models.qwen import eval as qwen_eval
    namespace = {"__name__": "__main__"}
    notebook = json.loads((ROOT / "notebooks/experimental_study.ipynb").read_text())
    display_module = SimpleNamespace(HTML=lambda value: value, display=lambda value: None)
    with (patch.dict(sys.modules, {"IPython": SimpleNamespace(), "IPython.display": display_module}),
          patch.object(model, "inspect_checkpoint", autospec=True, return_value={"fixture": True}) as inspect,
          patch.object(model, "load_finetuned", autospec=True,
                       return_value=(object(), SimpleNamespace(checkpoint_metadata={"fixture": True}), 1, 2)) as load,
          patch.object(encoder_eval, "evaluate", autospec=True, side_effect=lambda rows, *a: predictions(rows)) as evaluate,
          patch.object(qwen_eval, "inspect_ollama", autospec=True, side_effect=runtime) as inspect_llm,
          patch.object(qwen_eval, "ollama_response", autospec=True, side_effect=response) as generate):
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            print(f"Study cell {index}", flush=True)
            source = "".join(cell["source"])
            if index == 1:
                # Execute the unchanged first-cell statements, overriding only the
                # centralized settings immediately after assignment, before its guard.
                for statement in ast.parse(source).body:
                    exec(compile(ast.Module(body=[statement], type_ignores=[]), "study-settings-cell", "exec"), namespace)
                    if isinstance(statement, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "settings" for target in statement.targets):
                        namespace["settings"].update(controls or {})
            else:
                exec(compile(source, f"experimental_study.ipynb:{index}", "exec"), namespace)
        mode = namespace["settings"]["mode"]
        if mode != "run":
            inspect.assert_not_called();load.assert_not_called();evaluate.assert_not_called()
            inspect_llm.assert_not_called();generate.assert_not_called()
        elif namespace["settings"]["enable_llm"]:
            count = 2 if namespace["settings"]["run_kind"] == "validation" else 2 * len(namespace["rows"])
            assert generate.call_count == count
            assert all(call.kwargs["model"] == "qwen2.5:7b" for call in generate.call_args_list)
        else:
            inspect_llm.assert_not_called();generate.assert_not_called()
        if mode == "run" and namespace["settings"]["enable_encoder"]:
            inspect.assert_called_once();load.assert_called_once();evaluate.assert_called_once()
        else:
            inspect.assert_not_called();load.assert_not_called();evaluate.assert_not_called()
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
        Path(str(checkpoint) + ".json").write_text("{}")
        controls = dict(mode="run", input_path=input_path, expected_dev_items=4, validation_items=3,
                        enable_encoder=True, enable_llm=True, checkpoint=checkpoint, device="cpu",
                        output_root=folder, run_id="validation-one", run_kind="validation")
        validation = execute(controls)
        assert len(validation["rows"]) == 3
        assert validation["results"]["cohort"]["full_dev"] is False
        assert validation["results"]["metrics"][1]["n_attempted"] == 1
        assert validation["results"]["metrics"][1]["partial"] is True
        encoder_only = execute(dict(controls, run_id="encoder-only", enable_llm=False, qwen_model=None))
        assert encoder_only["results"]["metrics"][0]["n_valid_predictions"] == 3
        assert encoder_only["results"]["metrics"][1]["n_attempted"] == 0
        full = execute(dict(controls, run_id="full-one", run_kind="full_dev"))
        assert len(full["rows"]) == 4
        assert all(not metric["partial"] for metric in full["results"]["metrics"])
        reloaded = execute(dict(mode="reload", saved_run=full["artifact_path"], saved_run_id="full-one",
                               enable_llm=True, enable_encoder=True, qwen_model=None))
        assert reloaded["artifact"] == full["artifact"]
        reused = execute(dict(controls, run_id="saved-encoder", run_kind="full_dev", enable_encoder=False,
                              enable_llm=False, qwen_model=None, checkpoint=None,
                              saved_encoder=full["artifact_path"], saved_encoder_run_id="full-one"))
        assert reused["results"]["metrics"][0]["n_valid_predictions"] == 4
        assert reused["results"]["metrics"][1]["n_attempted"] == 0
    print("Study notebook: PASS (preview, validation, full fixture, reload, saved encoder; no real models/services/data)")


if __name__ == "__main__":
    main()
