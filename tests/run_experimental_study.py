"""Execute study cells with invented inputs and mocked interfaces in a guarded process."""
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
        # All repository research folders, including sibling main checkout, are blocked.
        for parent in (ROOT, ROOT.parent / "hebrew-acronym-disambiguation"):
            if any(path.is_relative_to(parent / name) for name in ("data", "results", "weights", "sources")):
                raise RuntimeError(f"Research I/O forbidden in study fixture: {path}")


def predictions(rows):
    return [{"item_id": row["item_id"], "status": "ok", "selected_candidate": row["candidates"].split("|")[0],
             "selected_index": 0, "candidate_scores": [{"candidate": candidate, "score": float(-i)}
               for i, candidate in enumerate(row["candidates"].split("|"))], "error": None}
            for row in rows]


def execute(controls=None):
    from hebrew_acronyms.models.dictabert_cross_encoder import model, eval as encoder_eval
    from hebrew_acronyms.models.qwen import eval as qwen_eval
    namespace = {"__name__": "__main__"}
    notebook = json.loads((ROOT / "notebooks/experimental_study.ipynb").read_text())
    calls = []
    def response(prompt, model):
        calls.append((prompt, model))
        if len(calls) == 2:
            raise RuntimeError("invented service failure")
        return "invented raw answer" if "פירוש:" in prompt else "A or B; ambiguous"
    display_module = SimpleNamespace(HTML=lambda value: value, display=lambda value: None)
    with (patch.dict(sys.modules, {"IPython": SimpleNamespace(), "IPython.display": display_module}),
          patch.object(model, "load_finetuned", autospec=True, return_value=(object(), SimpleNamespace(checkpoint_metadata={"fixture": True}), 1, 2)) as load,
          patch.object(encoder_eval, "evaluate", autospec=True, side_effect=lambda rows, *a: predictions(rows)) as evaluate,
          patch.object(qwen_eval, "ollama_generate", autospec=True, side_effect=response) as generate):
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            print(f"Study cell {index}", flush=True)
            exec(compile("".join(cell["source"]), f"experimental_study.ipynb:{index}", "exec"), namespace)
            if index == 1 and controls:
                namespace.update(controls)
        if not controls or controls.get("MODE") in {"preview", "reload"}:
            load.assert_not_called(); evaluate.assert_not_called(); generate.assert_not_called()
        else:
            assert all(model == "fixture-qwen-tag" for _, model in calls)
    return namespace


def main():
    sys.addaudithook(guard)
    from hebrew_acronyms import experimental_study as study
    default = execute()
    assert all(record["status"] == "not_run" for record in default["artifact"]["records"])
    assert not default.get("artifact_path")
    with tempfile.TemporaryDirectory(prefix="study-fixtures-") as temp:
        folder = Path(temp)
        rows = study.fixture_rows()
        input_path = folder / "invented.csv"
        with input_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
        checkpoint = folder / "invented.pt"
        checkpoint.write_text("not weights; mocked load")
        Path(str(checkpoint) + ".json").write_text("{}")
        controls = {"MODE": "run", "INPUT_PATH": input_path, "EXPECTED_DEV_ITEMS": 2,
                    "ENABLE_ENCODER": True, "ENABLE_LLM": True, "CHECKPOINT": checkpoint,
                    "QWEN_MODEL": "fixture-qwen-tag", "QWEN_REVISION": "fixture-immutable-digest",
                    "DEVICE": "cpu", "OUTPUT_DIR": folder / "run-one", "RUN_ID": "fixture-run-one"}
        first = execute(controls)
        records = first["artifact"]["records"]
        assert len(records) == 6
        assert [r["status"] for r in records] == ["ok", "ok", "response_received", "service_error", "response_received", "response_received"]
        reloaded = execute({"MODE": "reload", "SAVED_RUN": first["artifact_path"], "SAVED_RUN_ID": "fixture-run-one",
                            "ENABLE_LLM": True, "ENABLE_ENCODER": True})
        assert reloaded["artifact"] == first["artifact"]
        reuse_controls = dict(controls, ENABLE_ENCODER=False, ENABLE_LLM=False, CHECKPOINT=None,
                              SAVED_ENCODER=first["artifact_path"], SAVED_ENCODER_RUN_ID="fixture-run-one",
                              RUN_ID="fixture-run-two", OUTPUT_DIR=folder / "run-two")
        reused = execute(reuse_controls)
        assert [r["status"] for r in reused["artifact"]["records"]][:2] == ["ok", "ok"]
    print("Study notebook: PASS (preview, fake inference, reload, saved encoder; no real models/services/data)")


if __name__ == "__main__":
    main()
