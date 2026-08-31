# Developer shortcuts. Run `make check` to reproduce the CI quality gate locally.
.PHONY: install check lint type test train results reproduce page

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

# Regenerate every reported number, figure and CSV under results/, then rebuild
# the page from them. Needs the HDB5 dataset; `hdb5.py download` is a no-op if
# it is already present and verifies its SHA-256 either way.
#
# Order matters: the later analyses read the earlier ones' artifacts.
results:
	python3 hdb5.py download
	python3 analysis_scaling_law.py
	python3 analysis_extrapolation.py
	python3 analysis_flexibility_sweep.py
	python3 analysis_size_extrapolation.py
	python3 analysis_hybrid.py
	python3 analysis_conformal.py
	python3 site/build_page.py

# Rebuild only the page, for when results/ is current but the template changed.
page:
	python3 site/build_page.py

# The full reproducibility check the `reproduce` CI job runs: regenerate
# everything from the raw data and fail if any reported value moved.
#
# The comparison is numeric, not a git diff. float64 values serialize at full
# precision and their last digits jitter between runs because threaded BLAS
# reductions do not fix summation order, so a byte comparison fails every time
# and the gate gets ignored. tools/compare_results.py compares to a relative
# tolerance of 1e-6 instead: looser than the jitter, far tighter than the four
# significant figures anything is reported at.
reproduce:
	@rm -rf .reproduce_baseline && mkdir -p .reproduce_baseline
	@cp results/*.json results/*.csv .reproduce_baseline/
	@git rev-parse HEAD > .reproduce_baseline/.head 2>/dev/null || echo no-git > .reproduce_baseline/.head
	@$(MAKE) results
	python3 tools/compare_results.py .reproduce_baseline results
	python -m pytest -q tests/test_reported_numbers.py
	@# This target verifies; `make results` is the one that updates. On success
	@# put the committed bytes back, so a check does not leave the tree dirty
	@# with rewrites that differ only in the float64 tails. On failure the
	@# regenerated files stay in place to be inspected.
	@#
	@# Restoring is only safe if results/ has not moved underneath the run. The
	@# analyses take about fifteen minutes, which is long enough for a commit to
	@# land in the middle; blindly copying the snapshot back would then revert it
	@# in the working tree, silently and after a passing check. So the snapshot
	@# is only trusted while HEAD is where it was when the snapshot was taken.
	@# The glob skips .head, since a leading dot is not matched by `*`.
	@if [ "$$(git rev-parse HEAD 2>/dev/null || echo no-git)" = "$$(cat .reproduce_baseline/.head)" ]; then \
	  cp .reproduce_baseline/* results/; \
	  rm -rf .reproduce_baseline; \
	  echo "results/ reproduces from the raw data and the prose matches it"; \
	else \
	  rm -rf .reproduce_baseline; \
	  echo "results/ reproduces from the raw data and the prose matches it."; \
	  echo "NOTE: HEAD moved during this run, so the regenerated files were left in"; \
	  echo "place rather than restored over whatever landed. Run 'git checkout --"; \
	  echo "results/' if you want the committed bytes back."; \
	fi
