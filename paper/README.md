# Paper: source, build and result updates

The reviewed, compiled snapshot is [draft.pdf](draft.pdf), checked into Git so it
can be opened without a local TeX build. It is a working draft, not a submission.
The LaTeX files below remain the authoritative source.

`main.tex` and `sections/*.tex` are the single maintained manuscript. The previous
Markdown version is preserved at `9a14416:paper/manuscript.md`; its current file is
only a pointer. This draft describes verified development execution and its
limitations, but contains no final-test results and is not ready for submission.
The active deadline is **14 October 2026**, as supplied by Shaked; the course
PDF's 30 September date is superseded.

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
margins to make room. Results and discussion should replace the short draft note, using the remaining
space. There is no planned-display table in the manuscript.

Build outputs under `build/` are ignored; `draft.pdf` is the explicitly published
snapshot. After a successful build and visual review, refresh it with
`cp paper/build/main.pdf paper/draft.pdf` from the repository root and commit it
together with the source changes. With no `generated/current.tex`, the paper builds its
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
numerical claims.** Updating numbers does not update conclusions. The final-results
placeholders must be revised only after the identified test execution and
interpretation have been checked. Saved dev runs remain diagnostic. Human
confirmation is still needed for the final model/checkpoint settings, generation
judgments, evaluation population, baseline and variability-assessment choices,
author details and personal AI reflection.

## Input counts and evidence for the manuscript

The data table is a read-only count of the train/dev CSVs at `04ec317`, not a
regenerated qualification manifest or proof of the original training inputs:

| File | SHA-256 |
|---|---|
| `data/study_v1/encoder_inputs/train.csv` | `9e88f87b45d131f2f405393f8a73cb4a5e3b36c2d596da4ac3eb01b0cfd9dedd` |
| `data/study_v1/encoder_inputs/dev.csv` | `155c771ad7a37f59ef2dcbc710ead5d9fe73a316a85792e50a721c6c50f8ce6e` |

Counts use CSV records, distinct `acronym`, distinct nonempty `doc_id`, and the
sum of pipe-separated candidate counts. Construction totals group by
`construction` and `source`; review evidence groups by `label_evidence`, not the
coarser `label_status`. There are 136 authored rows without document keys. The
164 added rows comprise 136 authored and 28 natural rows; the documented 126-row
release group excludes the 10 authored boundary-correction cases.

Exact train/dev acronym and sentence intersections are empty. The nonempty
`doc_id` intersection has three keys, involving three train and five dev rows:
`doc-e7c170bc620ba9e456f8b550`, `doc-154690507021202f9730f042`, and
`doc-c2c93d64c922428639a059dc`. This contradicts complete document separation for
the updated files; the manuscript reports it without modifying data. Reserved
evaluation files were not opened. The old manifest is historical evidence only.

The [saved development summary](../results/dev-2026-10-03/summary.json) identifies
three original run IDs and artifact hashes. The manuscript's loading claims come
from the DictaBERT artifact's saved reconstruction and validation record, including
strict loading of 201 tensors, equality of encoding tensors for 62 items and
forward-logit checks for three items. They were not rerun during writing.
Qwen's recorded digest is
`845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`;
Gemini's returned version is `gemini-3.8-flash`. These are saved-run identities,
not claims about current service availability. Full original artifacts remain
outside Git as described in the results package.

Code links in the manuscript pin the inspected integrated revision `04ec317`,
including the Colab notebook and adapter. Their targets were checked locally;
this writing package does not publish commits or assert remote availability.

## Reconstruction and interpretation details

The current train file follows the 3 October release from 2,665 to 2,829 rows.
The 164 additions include the 126 authored rows released from missing-document
holds, 10 authored target-boundary corrections, and 28 natural repeated-target
sentences shortened to retain one occurrence. The qualification manifest still
identifies the earlier derivative; it is not a manifest for the expanded file.
The three document-key overlaps listed above are absent when the train file at
`e520a22` is compared with the current dev file. No exact sentence strings overlap.
Neither that history nor the overlap count establishes memorization, quantifies
an effect on accuracy, or establishes anything about test overlap.

Exact acronym strings are disjoint, but morphological families are not. Training
item `enc-85bf2051e1018c131afe7ba3f8d90feb` stores `מד״ר`, includes the phrase
`אם אני אבקש מד"ר שגב`, and has gold `ד"ר (מ- קליטי)`. Development includes `בד״ר`.
The earlier hold of 11 unprefixed `ד״ר` training items is an item-bounded policy;
it does not establish complete separation of the doctor-abbreviation family.
Each dev form has one **stored gold label**, not a verified single semantic sense.

The source of the supplied weights is Shaked's explicit attribution to Ben's
current [Colab notebook](../notebooks/train_dictabert_colab.ipynb) run. This is
author testimony, not metadata embedded in the checkpoint. The weight SHA-256 is
`23bbff0324a7ce4d9d4a24a9ebfb87a4f6ac296bf1f1ce1bd25126c09260267f`.
The inspected notebook is identified at
`09f815bb8c7658ef6dfa1c0efca19aa6a032bdf0`, SHA-256
`a57e668d20ba5c5cd8f6bcacd5b52fe8d67288cf1ea9a09cdb67e7108f0b010c`.
The filename suggests seed 43, while the notebook specifies 42 after model
initialization. The executed seed, original input snapshot, resolved original
base-model/tokenizer revision, library versions, training hardware, selected epoch
and loss are not recovered from the weights.

The [Colab adapter](../src/hebrew_acronyms/models/dictabert_cross_encoder/colab.py)
strictly loads the full 201-tensor state dictionary, including the scoring head
and expanded embedding matrix. The saved record has no missing or unexpected
keys, reports fp16 storage and fp32 CPU inference, and records the local tokenizer
and configuration identities. It contains checks of encoding tensors for all
62 dev items and forward logits for three validation items against the reference
notebook. These support inference reconstruction, not reproduction of training.
The Colab encoder can exceed its nominal 256-token limit if required tokens alone
do not fit, whereas the package predictor rejects that case; the saved dev checks
show no encoding discrepancy on the 62 inspected items.
The [package training appendix](../notebooks/train_dictabert.ipynb) is a separate
route, with seed-before-initialization and a manifest-bound loader. It is not the
reported source of these weights.

The saved comparison preserves separate arm run IDs. DictaBERT has 62 completed
selections; Qwen has 62 responses in each task. Gemini selection has 13 responses,
20 service-error items and 29 unattempted items; generation has two service-error
items and 60 unattempted items. Generation is semantically unjudged. The full-item
selection denominator is unchanged: on partial runs the score combines service
coverage with reference matching, and cannot rank linguistic correctness against
complete runs. The exporter still accepts one designated `full_dev` artifact;
it does not support test or combine the separate runs into a synthetic run ID.

The [course guidelines](../docs/course/NLP_course_2025b___project_guidelines.pdf),
Section 3, Methodology (page 4), require relevant baseline comparisons and an
account of randomness, including seeds, prompts and decoding. The choices of
baseline and variability assessment remain unresolved. The guidelines do not
prescribe a particular significance test, confidence interval or seed count.
No additional experiment is represented as completed in this draft.

The current design uses a smaller natural development collection, includes Knesset
text, and examines separation of stored acronym forms. These differ from the
submitted proposal's document-first split and larger target population; they are
not claims of lecturer approval or completed original commitments.

Annotation provenance retains historical review attributed to Ben and Shaked's
AI-assisted decisions (42/20 dev labels; 316/6 train labels), while 2,507 train
labels lack independent verification in preparation. Codex assisted code, checks,
literature checking, the exporter and manuscript, with assistant reviews during
earlier drafting. Software checks and PDF inspection are distinct from subsequent
real model runs. Author contributions and personal reflections must be written
or confirmed by the authors. This revision uses saved evidence and train/dev only;
it does not inspect test or run models.

## Offline verification

```bash
.venv/bin/python -B -m unittest tests.test_paper_export -v
```

These invented fixture tests exercise replacement, shared metrics, missing scores,
service failures, preview/validation rejection, exact saved-run loading, interrupted
publication and content verification. They provide evidence about the export path,
not performance. No fixture output belongs in `paper/generated/`.
