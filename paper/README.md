# Paper: source, build and result updates

`main.tex` and `sections/*.tex` are the single maintained manuscript. The previous
Markdown version is preserved at `9a14416:paper/manuscript.md`; its current file is
only a pointer. This draft contains no verified model results and is not ready for
submission. The active deadline is **14 October 2026**, as supplied by Shaked; the
course PDF's 30 September date is superseded.

## Build the readable draft

Use a TeX installation with XeLaTeX, BibTeX, latexmk and Polyglossia, plus Poppler's
`pdfinfo`. The manuscript uses Times New Roman (including Hebrew), Arial and Menlo.
The unmodified course ACL style is vendored with source commit and SHA-256 records
in [acl_template/PROVENANCE.md](acl_template/PROVENANCE.md). No source files from the
old paper or old build system are incorporated.

From the repository root:

```bash
make -C paper PYTHON=../.venv/bin/python
make -C paper check PYTHON=../.venv/bin/python
```

The PDF is `paper/build/main.pdf`. The check verifies ACL file hashes, missing
references/glyphs, overfull boxes and the last page of main text or a labeled main-content table/figure. Inspect
the rendered PDF as well, especially Hebrew and any new figure/table. The body must
stay within eight pages, excluding references and appendix. Do not reduce fonts or
margins to make room. Results and discussion should replace the short planned-results
structure, using the remaining space.

Build outputs are ignored. With no `generated/current.tex`, the paper builds its
explicit pending-results section. Building never loads a model, calls a service,
opens a research split or regenerates data. The existing multi-file ACL project is
built with its own TeX toolchain; the source remains editable in a normal editor.

## Select and export a run

Install the project in an isolated environment using the root README. The small
plot dependency is included in `requirements.txt`. See
[generated/README.md](generated/README.md) for the exported files and safeguards.
The notebook's optional paper-export cell is disabled until an exact
`paper_source_run_id` is provided. It accepts an explicitly selected full-dev run,
including a run reloaded using the notebook's exact saved path and run ID. Dev is
always labeled preliminary; this exporter does not claim to support a final test
protocol that has not yet been adopted.

To export an already saved run without running the notebook or models:

```python
from hebrew_acronyms.paper_export import export_paper

export_paper(
    "/path/to/chosen-run/study.json",
    expected_run_id="chosen-run-id",
    paper_source_run_id="chosen-run-id",
    output_dir="paper/generated",
)
```

Choose the source explicitly for its role in the study, never by best score or last
modification time. All figures, tables and numeric macros come from this one saved
artifact and the existing scorer. The exporter stages a complete immutable bundle
and atomically replaces `generated/current.tex`, the stable path used by the paper.
No manual copy is needed. Rebuild with the environment containing this package:

```bash
make -C paper check PYTHON=../.venv/bin/python
```

The build verifies the selected bundle's hashes before compilation. Missing or
altered files cause failure; an old graph is not silently paired with new numbers.
A concurrent export may select another complete bundle, but does not mutate the
files already named by a prior include. Preview, invented fixtures and short
validation runs cannot publish paper results. Fixture-only permission is limited
to a dedicated temporary directory. The exported table and manifest preserve partial
execution, failures and unavailable scores. Free generation remains unjudged; this
exporter deliberately introduces no second scoring implementation or human rubric.

Small derived PDF figures, TeX tables/macros and manifests under `paper/generated/`
are explicitly allowed in version control for this task. Weights, secrets, source
run JSON and raw responses remain outside it. Retained immutable bundles provide
provenance and are never picked automatically. Do not edit generated files by hand.

**After changing the selected run, reread the abstract, Results, Discussion and all
numerical claims.** Updating numbers does not update conclusions. The current
pending-results prose must be revised only once research execution and interpretation
have actually been checked. Human confirmation is still needed for the final
model/checkpoint settings, generation judgments, evaluation population, baseline
and uncertainty choices, author details and personal AI reflection.

## Offline verification

```bash
.venv/bin/python -B -m unittest tests.test_paper_export -v
```

These invented fixture tests exercise replacement, shared metrics, missing scores,
service failures, preview/validation rejection, exact saved-run loading, interrupted
publication and content verification. They provide evidence about the export path,
not performance. No fixture output belongs in `paper/generated/`.
