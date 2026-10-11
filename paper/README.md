# Paper build and evidence

[main.tex](main.tex) and `sections/*.tex` are the maintained manuscript source.
For Overleaf, import the multi-file project, select `main.tex` and XeLaTeX.
Ben's original manuscript remains in Git at
`4d2dc738914fa67970f18ee897b3deb7ff19f4f2:paper/overleaf_main.tex`.

The [main analysis notebook](../notebooks/experimental_study.ipynb) and
[export index](../results/submission-20261010/README.md) connect the paper to saved
responses, measurements and model settings. Ben's separate generation labels are
in the [consolidated review table](../saved-results/human-review/review.csv);
[provenance](../saved-results/README.md#ben-dictalm-annotations) records their
historical source identity and missing annotation evidence. Automatic scores,
human labels and diagnostic selection remain distinct.

Build with XeLaTeX, BibTeX, latexmk, Polyglossia and Poppler. Fonts are Times New
Roman (including Hebrew), Arial and Menlo. The course ACL files and checksums are
in [acl_template/PROVENANCE.md](acl_template/PROVENANCE.md).

```bash
make -C paper check
pdftoppm -r 130 -png paper/build/main.pdf paper/build/review-page
```

The check builds `paper/build/main.pdf` and checks ACL hashes, references, glyphs,
overfull boxes and the eight-page body limit. Inspect all rendered pages,
including references. Build products are ignored by Git. Building does not call
models, alter research data or regenerate analysis exports.

To regenerate the paper's results figure from the saved automatic, baseline and
similarity CSVs, use the installed project environment:

```bash
python paper/make_result_figures.py
make -C paper check
```

The figure generator checks denominators and count/accuracy agreement. It does not
rescore responses. Local builds and evidence validation do not publish the paper
or repository.
