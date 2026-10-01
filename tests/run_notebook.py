"""Execute unchanged notebook cells with a tiny model, offline and without research I/O.

This is an isolated test process, not a research runner. No Jupyter dependency is needed.
"""
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
from hebrew_acronyms.models.dictabert_cross_encoder.workflow import enable_offline


def execute_notebook(root, snapshot):
    from tests.fixtures.tiny import tiny_base_model
    notebook = root / "notebooks/train_dictabert.ipynb"
    namespace = {"__name__": "__main__"}
    content = json.loads(notebook.read_text(encoding="utf-8"))
    with (patch("hebrew_acronyms.models.dictabert_cross_encoder.model.build_base_model", tiny_base_model),
          patch("hebrew_acronyms.models.dictabert_cross_encoder.workflow.find_snapshot", return_value=snapshot)):
        for index, cell in enumerate(content["cells"]):
            if cell["cell_type"] == "code":
                print(f"Running notebook cell {index}", flush=True)
                exec(compile("".join(cell["source"]), f"{notebook.name}:cell{index}", "exec"), namespace)
    predictions = namespace["predictions"]
    assert len(predictions) == 2
    assert [p["item_id"] for p in predictions] == [r["item_id"] for r in namespace["train_rows"]]
    assert all(p["status"] == "ok" and len(p["candidate_scores"]) == 2 for p in predictions)
    assert namespace["history"] == []
    assert namespace["TRAIN"] is False
    return predictions


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    enable_offline()
    def guard(event, args):
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
            raise RuntimeError("Network forbidden in notebook fixture")
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if any(path.is_relative_to(root / name) for name in ("data", "results", "weights")):
                raise RuntimeError(f"Research I/O forbidden in notebook fixture: {path}")
    sys.addaudithook(guard)
    import torch
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="notebook-fixture-") as directory:
        execute_notebook(root, Path(directory))
    print("Notebook: PASS (all cells, tiny inference, offline, invented inputs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
