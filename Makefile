.PHONY: setup smoke tables test

PYTHON ?= python

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

smoke:
	$(PYTHON) -m fairmix_audit.cli --config configs/smoke.yml

tables:
	@test -n "$(RUN_DIR)" || (echo "Usage: make tables RUN_DIR=results/<run-dir>" && exit 1)
	$(PYTHON) scripts/make_tables.py $(RUN_DIR)

test:
	$(PYTHON) -m pytest tests
