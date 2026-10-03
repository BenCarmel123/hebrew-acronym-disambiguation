"""Publish one explicit saved study as a small, immutable paper-result bundle.

This module only renders the existing study scorer's output. It neither reads
research inputs nor invokes models. Importing it has no plotting side effects.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from hebrew_acronyms import experimental_study as study
from hebrew_acronyms.models.common import eval as scoring


LABELS = {"dictabert": "DictaBERT selection", "qwen_select": "Qwen selection",
          "gemini_select": "Gemini selection", "qwen_generate": "Qwen generation",
          "gemini_generate": "Gemini generation"}
OUTPUT_FILES = {"table.tex", "macros.tex", "selection_accuracy.pdf"}


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _tex(value):
    escapes = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
               "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
               "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(escapes.get(char, char) for char in str(value).replace("\n", " ").replace("\r", " "))


def _number(value):
    if value is not None and not 0 <= value <= 1:
        raise ValueError("Accuracy must be between zero and one")
    return "--" if value is None else f"{100 * value:.1f}"


def _check_scope(artifact, output, fixture_root):
    cohort = artifact.get("cohort", {})
    settings = artifact["settings"]
    fixture = (artifact.get("provenance", {}).get("fixture") is True
               or any(row.get("source_kind") == "invented_fixture" for row in artifact["items"]))
    if fixture_root is not None:
        root = Path(fixture_root).resolve(strict=True)
        temporary = Path(tempfile.gettempdir()).resolve()
        if root == temporary or not root.is_relative_to(temporary) or not output.is_relative_to(root):
            raise ValueError("Fixture permission requires a dedicated temporary root containing output_dir")
        if output == root or not fixture:
            raise ValueError("Fixture permission is only for invented fixtures in a temporary subdirectory")
    elif fixture:
        raise ValueError("Invented fixtures cannot publish paper results")
    if settings.get("mode") == "preview" or settings.get("run_kind") == "validation":
        raise ValueError("Preview and validation runs cannot publish paper results")
    if (artifact.get("format_version") != 2 or cohort.get("kind") != "full_dev"
            or not cohort.get("full_dev") or settings.get("run_kind") != "full_dev"):
        raise ValueError("Paper export requires an identified five-arm full_dev cohort")


def _summaries(artifact):
    inspection = study.inspect_results(artifact)
    # Selection numbers, failures and execution labels belong to the shared scorer.
    summaries = {metric["condition"]: {key: deepcopy(value) for key, value in metric.items()
                 if key not in {"details", "by_type"}} for metric in inspection["metrics"]}
    for condition, (system, task) in study.arm_specs(artifact).items():
        if task != "generate":
            continue
        counts = Counter(r["status"] for r in artifact["records"] if r["condition"] == condition)
        attempted = len(artifact["items"]) - counts["not_run"]
        summaries[condition] = {"condition": condition, "system": system,
            "n_items": len(artifact["items"]), "n_attempted": attempted,
            "execution_status": "not_run" if not attempted else "partial"
                if counts["not_run"] or counts["interrupted"] else "completed",
            "status_counts": dict(counts), "n_complete_responses": counts["response_received"],
            "micro_accuracy": None, "macro_accuracy": None,
            "score_status": inspection["generation_score_status"]}
    return [summaries[name] for name in study.ARMS]


def _model_identities(artifact):
    records = artifact["records"]
    return {"requested": {key: artifact["settings"].get(key)
                          for key in ("qwen_model", "gemini_model")},
            "returned_revisions": {system: sorted({r["model_revision"] for r in records
                if r.get("system") == system and r.get("model_revision")})
                for system in ("qwen", "gemini")},
            "encoder": next(({key: r["encoder_origin"][key]
                              for key in ("weights_sha256", "manifest_sha256")}
                             for r in records if r.get("encoder_origin")), None)}


def _render_table(folder, manifest):
    rows = [r"\begin{tabular}{lrrrrl}", r"\hline",
            r"Arm & Run/$N$ & Valid/resp. & Micro & Macro & State \\", r"\hline"]
    state = {"not_run": "not run", "partial": "partial", "completed": "complete"}
    for metric in manifest["summaries"]:
        rows.append(" & ".join([_tex(LABELS[metric["condition"]]),
            f'{metric["n_attempted"]}/{metric["n_items"]}',
            str(metric.get("n_valid_predictions", metric.get("n_complete_responses"))),
            _number(metric["micro_accuracy"]), _number(metric["macro_accuracy"]),
            state[metric["execution_status"]]]) + r" \\")
    rows += [r"\hline", r"\end{tabular}", r"\par\smallskip",
             r"{\footnotesize Accuracy in percent; -- means unavailable or unjudged. "
             r"Failures and unrun items remain in the selection denominator. "
             r"Valid/resp. counts valid selections or complete generation responses, not correct answers. "
             r"Complete means all requests were attempted without interruption, not all succeeded. "
             r"Generation is unjudged. Run: \PaperRunID; cohort: \PaperCohort; "
             r"execution: \PaperRunStatus.}"]
    (folder / "table.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    values = {"PaperRunID": manifest["run_id"], "PaperCohort": "full dev (preliminary)",
              "PaperItemCount": manifest["cohort"]["requested_items"],
              "PaperInputHash": manifest["input_identity"]["sha256"],
              "PaperRunStatus": manifest["execution_status"],
              "PaperGenerationStatus": "unjudged"}
    for metric in manifest["summaries"]:
        name = "".join(word.title() for word in metric["condition"].split("_"))
        values["Paper" + name + "Micro"] = _number(metric["micro_accuracy"])
        values["Paper" + name + "Macro"] = _number(metric["macro_accuracy"])
    (folder / "macros.tex").write_text("".join(
        rf"\newcommand{{\{key}}}{{{_tex(value)}}}" + "\n" for key, value in values.items()), encoding="utf-8")


def _render_figure(folder, manifest):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_pdf import FigureCanvasPdf
    figure = Figure(figsize=(6.4, 3.5), layout="constrained")
    FigureCanvasPdf(figure)
    axis = figure.subplots()
    metrics = [m for m in manifest["summaries"] if not m["condition"].endswith("generate")]
    for position, metric in enumerate(metrics):
        for offset, key, color in ((-.19, "micro_accuracy", "#276B91"), (.19, "macro_accuracy", "#D98B39")):
            value = metric[key]
            if value is None:
                axis.text(position + offset, 4, "N/A", ha="center", va="bottom", fontsize=8, rotation=90)
            else:
                axis.bar(position + offset, 100 * value, .35, color=color,
                         label=key.split("_")[0].title() if position == 0 else None)
                axis.text(position + offset, 100 * value + 1, _number(value), ha="center", fontsize=8)
    # Explicit legend handles also work when the first system is entirely unrun.
    from matplotlib.patches import Patch
    figure.legend(handles=[Patch(color="#276B91", label="Micro"), Patch(color="#D98B39", label="Macro")],
                  loc="outside upper right", ncol=2, frameon=False)
    axis.set_xticks(range(len(metrics)), [LABELS[m["condition"]].replace(" selection", "") + "\n" +
                   m["execution_status"].replace("_", " ") for m in metrics])
    axis.set_xlim(-.6, len(metrics) - .4)
    axis.set_ylim(0, 112)  # Space above 100% keeps value labels inside the plot.
    axis.set_yticks([0, 25, 50, 75, 100])
    axis.set_ylabel("Preliminary dev selection accuracy (%)")
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_title("Full dev; " + manifest["execution_status"] + "; N=" + str(manifest["cohort"]["requested_items"]), fontsize=10)
    figure.savefig(folder / "selection_accuracy.pdf", metadata={"CreationDate": None, "ModDate": None,
                   "Title": "Selection accuracy: " + manifest["run_id"], "Subject": manifest["input_identity"]["sha256"]})


def _pointer(bundle_id):
    prefix = "generated/bundles/" + bundle_id
    return (rf"\input{{{prefix}/macros.tex}}" + "\n" +
            rf"\newcommand{{\PaperResultsTable}}{{\input{{{prefix}/table.tex}}}}" + "\n" +
            rf"\newcommand{{\PaperSelectionFigure}}{{\includegraphics[width=\linewidth]{{{prefix}/selection_accuracy.pdf}}}}" + "\n")


def verify_paper_bundle(output_dir):
    """Verify the atomic include and every bound file before compiling the paper."""
    output = Path(output_dir).resolve()
    pointer = (output / "current.tex").read_text(encoding="utf-8")
    match = re.match(r"\\input\{generated/bundles/([0-9a-f]{64})/macros\.tex\}\n", pointer)
    if not match or pointer != _pointer(match[1]):
        raise ValueError("Invalid paper bundle pointer")
    folder = output / "bundles" / match[1]
    payload = (folder / "manifest.json").read_bytes()
    if _sha(payload) != match[1]:
        raise ValueError("Paper manifest hash mismatch")
    manifest = json.loads(payload)
    if set(manifest["files"]) != OUTPUT_FILES:
        raise ValueError("Unexpected paper bundle file set")
    for name, expected in manifest["files"].items():
        if _sha((folder / name).read_bytes()) != expected:
            raise ValueError("Paper bundle hash mismatch: " + name)
    return manifest


def export_paper(artifact_or_saved_path, *, expected_run_id, output_dir,
                 paper_source_run_id, fixture_root=None):
    """Render only the explicitly designated full-dev run, without rerunning it.

    ``fixture_root`` grants fixture-only permission inside a dedicated temporary
    directory; it is never a switch permitting fixtures in paper/generated.
    A stable current.tex atomically selects an immutable bundle. TeX paths are
    relative to paper/, so compile the manuscript with paper/ as the cwd.
    """
    if not paper_source_run_id or paper_source_run_id != expected_run_id:
        raise ValueError("An exact, explicit paper-source run ID is required")
    source_file_sha256 = None
    if isinstance(artifact_or_saved_path, dict):
        artifact = deepcopy(artifact_or_saved_path)
    else:
        # Read once so the validated snapshot and its file hash cannot diverge.
        payload = Path(artifact_or_saved_path).read_bytes()
        source_file_sha256 = _sha(payload)
        artifact = json.loads(payload)
    study.validate_artifact(artifact, expected_run_id=expected_run_id)
    output = Path(output_dir).resolve()
    _check_scope(artifact, output, fixture_root)
    summaries = _summaries(artifact)
    manifest = {"format_version": 1, "run_id": expected_run_id,
        "paper_source_run_id": paper_source_run_id, "fixture": fixture_root is not None,
        "source_artifact_sha256": _sha(_json_bytes(artifact)),
        "source_file_sha256": source_file_sha256,
        "input_identity": artifact["input_identity"], "cohort": artifact["cohort"],
        "model_identities": _model_identities(artifact), "summaries": summaries,
        "execution_status": "completed" if all(m["execution_status"] == "completed" for m in summaries) else "partial",
        "generation_score_status": "manual_review_unscored",
        "source_revision": artifact.get("provenance", {}).get("git_head"),
        "code_sha256": {"exporter": _sha(Path(__file__).read_bytes()),
                        "shared_scorer": _sha(Path(scoring.__file__).read_bytes())}}
    bundles = output / "bundles"
    bundles.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=bundles))
    pointer_temp = None
    try:
        _render_table(staging, manifest)
        _render_figure(staging, manifest)
        manifest["files"] = {name: _sha((staging / name).read_bytes()) for name in sorted(OUTPUT_FILES)}
        payload = _json_bytes(manifest)
        (staging / "manifest.json").write_bytes(payload)
        bundle_id = _sha(payload)
        destination = bundles / bundle_id
        if destination.exists():
            if any((destination / path.name).read_bytes() != path.read_bytes() for path in staging.iterdir()):
                raise ValueError("Existing immutable paper bundle was modified")
        else:
            staging.rename(destination)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output,
                                         prefix=".current-", delete=False) as stream:
            pointer_temp = Path(stream.name)
            stream.write(_pointer(bundle_id))
        os.replace(pointer_temp, output / "current.tex")
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if pointer_temp is not None and pointer_temp.exists():
            pointer_temp.unlink()
