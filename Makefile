# Developer shortcuts. Run `make check` to reproduce the CI quality gate locally.
.PHONY: install check lint type test train

# Install the package plus dev tooling against the pinned, tested environment.
install:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev]" -c constraints.txt

# Lint, type-check, and test in the same order CI runs them.
check: lint type test

lint:
	ruff check .

type:
	mypy .

test:
	python -m pytest -q

# Train on the synthetic demo dataset (regenerates the default saved artifact).
train:
	python train_model.py train --allow-synthetic
