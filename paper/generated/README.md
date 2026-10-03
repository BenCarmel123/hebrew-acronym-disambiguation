# Selected paper results

No research results are exported in this draft. The paper must compile with its
explicit pending-results text when `current.tex` is absent.

In `notebooks/experimental_study.ipynb`, set `paper_source_run_id` to the exact
chosen full-dev run ID. The final export cell renders the in-memory artifact;
`mode="reload"` uses the same path after loading the explicit `saved_run` and
`saved_run_id`. Do not select a source run according to its best score. Current
exports describe preliminary development results, never held-out test results.

`export_paper` reuses the study's shared selection scorer. Missing scores are
`null` in the manifest and `--` in TeX, generation remains unjudged, all requested
items remain in the scorer's denominator, and partial execution is labelled.
An execution marked complete can still contain service or parsing failures.
The table shows attempted and valid-selection/complete-response counts; generation
response counts do not imply judged correctness. Use a full-width `table*` and
`figure*` in ACL so the six-column table and 6.4-inch PDF remain legible.
The manifest preserves scorer status counts
and warnings. Raw predictions and source text remain outside this directory.

Publication first writes an immutable `bundles/<manifest-sha256>/` containing
`selection_accuracy.pdf`, `table.tex`, `macros.tex` and `manifest.json`. Only after
all outputs succeed does it atomically replace `current.tex`. That stable include
binds `\PaperSelectionFigure`, `\PaperResultsTable` and scalar macros to the same
bundle. Compile from `paper/`. The manuscript can display `\PaperRunID`,
`\PaperCohort`, `\PaperItemCount`, `\PaperRunStatus` and `\PaperGenerationStatus`.
Accuracy macros are percentages, for example `\PaperDictabertMicro` and
`\PaperQwenSelectMacro`. Old bundles are retained for provenance, not selected
implicitly. Export does not rewrite scientific conclusions.

Before compiling, verify the selected bundle from the repository root:

```bash
.venv/bin/python -c 'from hebrew_acronyms.paper_export import verify_paper_bundle; verify_paper_bundle("paper/generated")'
```

Run that check only when `current.tex` exists. It rejects modified or mixed files.
The manifest binds source artifact and input hashes, requested model IDs,
returned revisions, encoder checkpoint hashes, exporter and scorer source hashes,
and hashes of every generated file. A saved-file export also records its exact
byte hash. An in-memory export records the canonical artifact hash instead.

Preview and validation runs cannot export. Invented fixture tests require an
explicit `fixture_root` confined to a dedicated temporary directory; this cannot
permit fixture outputs here. Reproduce those offline checks with:

```bash
.venv/bin/python -B -m unittest tests.test_paper_export -v
```
