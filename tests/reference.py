"""Read fixed Git references for automated structural comparisons."""

import ast
import json
from pathlib import Path
import subprocess

BASE = "8ca117d50c3f01d4473b944c99611c7191af05b4"
ROOT = Path(__file__).resolve().parents[1]


def reference_git(commit, *args, text=True):
    """Fail explicitly when a required reference cannot be read; never skip checks."""
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=text, stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        raise RuntimeError(
            f"Cannot read Git reference {commit}. Equivalence checks require Git "
            "and the full project history; a ZIP or shallow clone is insufficient. "
            "Use a full clone of review-handoff (see the root README). "
            "If this is already a full clone, verify the requested reference/path. "
            "This is a failed check, not a skip."
        ) from error


def baseline_source(path):
    """Read an exact tracked file from the fixed commit, never the working tree."""
    return reference_git(BASE, "show", f"{BASE}:{path}")


def baseline_notebook_cell(index):
    """Return a baseline cell's source for explicit, reviewed AST selection."""
    notebook = json.loads(baseline_source("notebooks/train_dictabert.ipynb"))
    return "".join(notebook["cells"][index]["source"])


def _definitions(sources, names, namespace):
    scope = {} if namespace is None else namespace.copy()
    selected = []
    for source in sources:
        # The old installation cell is IPython syntax, not Python definitions.
        if source.lstrip().startswith("!"):
            continue
        for node in ast.parse(source).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
                selected.append(node)
    found = {node.name for node in selected}
    if found != set(names):
        raise ValueError(f"Missing baseline definitions: {set(names) - found}")
    tree = ast.Module(body=selected, type_ignores=[])
    exec(compile(tree, f"git:{BASE}", "exec"), scope)
    return scope


def baseline_notebook_namespace(names, namespace=None):
    """Load only named definitions; their imports and globals must be supplied."""
    notebook = json.loads(baseline_source("notebooks/train_dictabert.ipynb"))
    sources = ["".join(cell["source"]) for cell in notebook["cells"]
               if cell["cell_type"] == "code"]
    return _definitions(sources, names, namespace)


def baseline_module_namespace(path, names, namespace=None):
    """Load only named definitions from a baseline source module."""
    return _definitions([baseline_source(path)], names, namespace)


def current_source_path(historical_path):
    """Locate a moved current file while keeping Git lookups at their original paths."""
    moves = {
        "model/dictabertX/": "src/hebrew_acronyms/models/dictabert_cross_encoder/",
        "model/dictabert/": "src/hebrew_acronyms/models/dictabert_similarity/",
        "model/": "src/hebrew_acronyms/models/",
        "data_preprocess/": "src/hebrew_acronyms/data_processing/",
        "pipeline/": "src/hebrew_acronyms/pipelines/",
    }
    for old, new in moves.items():
        if historical_path.startswith(old):
            return ROOT / (new + historical_path[len(old):])
    raise ValueError(f"No source move recorded for {historical_path}")


def migrated_imports(node):
    """Adapt only the moved imports inside the historical combined evaluator.

    All non-import AST nodes, aliases, constants and call arguments remain exact.
    This does not execute or replace any historical evaluation logic.
    """
    moves = {
        "model.dictabertX.eval": "hebrew_acronyms.models.dictabert_cross_encoder.eval",
        "model.dictabertX.model": "hebrew_acronyms.models.dictabert_cross_encoder.model",
        "model.qwen.eval": "hebrew_acronyms.models.qwen.eval",
        "model.gemini.eval": "hebrew_acronyms.models.gemini.eval",
    }
    for child in ast.walk(node):
        if isinstance(child, ast.ImportFrom) and child.module in moves:
            child.module = moves[child.module]
    return node
