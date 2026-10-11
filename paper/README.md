# Paper

[main.tex](main.tex) and `sections/*.tex` are the maintained manuscript. The
[analysis notebook](../notebooks/experimental_study.ipynb) and
[export index](../results/submission-20261010/README.md) connect its tables to saved
evidence. Build with XeLaTeX, BibTeX, latexmk, Polyglossia and Poppler, using Times
New Roman (including Hebrew), Arial and Menlo. The preserved ACL template is
documented in [its provenance](acl_template/PROVENANCE.md).

```bash
make -C paper check
```

This creates `paper/build/main.pdf` and checks template hashes, references, glyphs,
layout warnings and the eight-page body limit. Inspect all PDF pages visually
after building. Build products are ignored by Git.

[English PDF](https://github.com/BenCarmel123/hebrew-acronym-disambiguation/releases/download/submission-2026-10-12/hebrew-acronym-expansion.pdf).
