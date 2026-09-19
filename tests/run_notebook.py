"""Execute every training-notebook code cell in a fresh offline Python process.

Run from the repository root: python -B -m tests.run_notebook
The notebook uses plain Python, so this does not require an IPython installation.
"""
from pathlib import Path
import json
import sys

from hebrew_acronyms.models.dictabert_cross_encoder.workflow import enable_offline


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = root / "notebooks" / "train_dictabert.ipynb"
    enable_offline()

    def guard_research_reads(event, args):
        if event == "open" and isinstance(args[0], (str, bytes)):
            path = Path(args[0]).resolve()
            if path.is_relative_to(root / "data") or path.is_relative_to(root / "results"):
                raise RuntimeError(f"Notebook smoke must not access research files: {path}")

    sys.addaudithook(guard_research_reads)
    namespace = {"__name__": "__main__"}
    content = json.loads(notebook.read_text(encoding="utf-8"))
    for index, cell in enumerate(content["cells"]):
        if cell["cell_type"] == "code":
            print(f"Running notebook cell {index}", flush=True)
            exec(compile("".join(cell["source"]), f"{notebook.name}:cell{index}", "exec"), namespace)
    result = namespace["result"]
    if result.get("mode") != "smoke" or result.get("status") != "PASS":
        raise RuntimeError(f"Notebook did not complete smoke: {result}")
    print("Notebook: PASS (all code cells, fresh process, offline, no research data reads)")


if __name__ == "__main__":
    main()
