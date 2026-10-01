"""Execute the thin notebook offline; default is cached inference only.

--tiny-sanity explicitly selects invented-input learning with temporary checkpoints.
No Jupyter installation is required. Execute this module in a fresh process.
"""
import argparse
import ast
import json
from pathlib import Path
import sys
import tempfile

from hebrew_acronyms.models.dictabert_cross_encoder.workflow import enable_offline


def execute_notebook(root, tiny_sanity=False, checkpoint=None):
    notebook = root / "notebooks" / "train_dictabert.ipynb"
    namespace = {"__name__": "__main__"}
    content = json.loads(notebook.read_text(encoding="utf-8"))
    first = True
    for index, cell in enumerate(content["cells"]):
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        if first and tiny_sanity:
            # Override only the explicit settings, then execute every notebook cell.
            replacements = {"MODE": repr("sanity"), "SNAPSHOT": "None",
                            "CHECKPOINT_PATH": repr(str(checkpoint)),
                            "CONFIG": "TrainingConfig(epochs=180, lr=0.03, batch_size=4, max_len=64)"}
            for node in tree.body:
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                    name = node.targets[0].id
                    if name in replacements:
                        node.value = ast.parse(replacements.pop(name), mode="eval").body
            if replacements:
                raise AssertionError(f"Notebook settings not found: {replacements}")
            ast.fix_missing_locations(tree)
        first = False
        print(f"Running notebook cell {index}", flush=True)
        exec(compile(tree, f"{notebook.name}:cell{index}", "exec"), namespace)
    result = namespace["result"]
    expected = "sanity" if tiny_sanity else "smoke"
    if result.get("mode") != expected or result.get("status") != "PASS":
        raise RuntimeError(f"Notebook did not complete {expected}: {result}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiny-sanity", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    enable_offline()

    def guard_research_reads(event, args):
        if event == "open" and isinstance(args[0], (str, bytes)):
            path = Path(args[0]).resolve()
            if path.is_relative_to(root / "data") or path.is_relative_to(root / "results"):
                raise RuntimeError(f"Notebook smoke must not access research files: {path}")

    sys.addaudithook(guard_research_reads)
    import torch
    torch.set_num_threads(1)
    try:
        with tempfile.TemporaryDirectory(prefix="encoder-notebook-") as directory:
            result = execute_notebook(root, args.tiny_sanity, Path(directory) / "tiny.pt")
    except FileNotFoundError as error:
        if "Model NOT RUN" in str(error):
            print(f"Notebook: NOT RUN ({error})")
            return 2
        raise
    print(f"Notebook: PASS ({result['mode']}, all cells, offline, invented inputs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
