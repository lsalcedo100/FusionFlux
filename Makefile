# FusionFlux developer workflow.
# `make setup` creates a local .venv and installs pinned runtime + dev deps.
# All other targets run against that .venv, so no manual activation is needed.

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(PYTHON) -m pip

.DEFAULT_GOAL := help
.PHONY: help setup lint typecheck fmt test train check clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "} {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Create the venv and install runtime + dev dependencies
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -c constraints.txt -r requirements.txt -r requirements-dev.txt

lint: ## Run ruff lint checks
	$(VENV)/bin/ruff check .

fmt: ## Auto-fix lint issues where safe
	$(VENV)/bin/ruff check . --fix

typecheck: ## Run mypy on the source modules
	$(VENV)/bin/mypy

train: ## Train once on synthetic data (produces the artifact the suite needs)
	$(PYTHON) train_model.py train --allow-synthetic

test: train ## Train, then run the test suite with coverage
	$(PYTHON) -m pytest --cov --cov-report=term-missing

check: lint typecheck test ## Run the full CI gate locally (lint + types + tests)

clean: ## Remove caches and generated training artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	rm -rf data/processed
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
