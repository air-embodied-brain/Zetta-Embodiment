# Minimal makefile for the Zetta Sphinx documentation.

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
LANG          ?= en
SOURCEDIR     = source-$(LANG)
BUILDDIR      = build

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help clean Makefile

clean:
	@rm -rf "$(BUILDDIR)"

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
