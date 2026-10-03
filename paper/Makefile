# Existing multi-file ACL project: XeLaTeX supplies native Hebrew and BibTeX
# supplies author-year references. No model or service is invoked by this build.
PYTHON ?= python3
export TEXINPUTS := .:./acl_template:$(TEXINPUTS)
export BSTINPUTS := .:./acl_template:$(BSTINPUTS)

.PHONY: all check verify-generated
all: verify-generated
	latexmk -xelatex -halt-on-error -interaction=nonstopmode -outdir=build main.tex

verify-generated:
	@if [ -f generated/current.tex ]; then $(PYTHON) -c 'from hebrew_acronyms.paper_export import verify_paper_bundle; verify_paper_bundle("generated")'; fi

check: all
	@cd acl_template && shasum -a 256 -c SHA256SUMS
	@$(PYTHON) check_build.py
