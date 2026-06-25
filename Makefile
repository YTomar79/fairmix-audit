.PHONY: setup smoke full tables test

PYTHON ?= python

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

# Fast end-to-end check on a small ACS slice.
smoke:
	$(PYTHON) -m fairmix_audit.cli --config configs/smoke.yml

# Full reference experiment (heavy; downloads all configured ACS slices).
full:
	$(PYTHON) -m fairmix_audit.cli --config configs/default.yml

# Regenerate tables/plots/model cards from an existing run, e.g.
#   make tables RUN_DIR=results/<run-dir>
tables:
	@test -n "$(RUN_DIR)" || (echo "Usage: make tables RUN_DIR=results/<run-dir>" && exit 1)
	$(PYTHON) scripts/make_tables.py $(RUN_DIR)

test:
	$(PYTHON) -m pytest tests
