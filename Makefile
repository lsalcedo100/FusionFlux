# Developer shortcuts. Run `make check` to reproduce the CI quality gate locally.
.PHONY: install check lint type test train results reproduce page arxiv dist paper-fresh

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
# the page from them. Needs both datasets; the two download steps are no-ops if
# the files are already present and verify their SHA-256 either way. DB5.2.3 is
# the full database revision Result 11 replicates on, and it is 11 MB against
# STD5's 0.9 MB, so the first run of this target is noticeably slower.
#
# Order matters: the later analyses read the earlier ones' artifacts. Results 8
# to 12 come last because they reference the size cut and the hybrid that
# Results 5 and 6 establish.
#
# analysis_summary_figure.py is last and reads only the artifacts the others
# wrote, never the raw database, so it is the one step that runs without a
# download.
#
# Results 10 and 12 calibrate by holding out each training machine in turn,
# which costs one extra fit per machine per fold. That makes them the slow ones,
# a few minutes each rather than seconds.
results:
	python3 hdb5.py download
	python3 -c "import replication; replication.download_db523()"
	python3 -c "import allometry; allometry.download_allometry()"
	python3 -c "import tree_allometry; tree_allometry.download_baad()"
	python3 analysis_scaling_law.py
	python3 analysis_extrapolation.py
	python3 analysis_boundedness.py
	python3 analysis_robustness.py
	python3 analysis_sensitivity.py
	python3 analysis_mechanism.py
	python3 analysis_tuned.py
	python3 analysis_flexibility_sweep.py
	python3 analysis_size_extrapolation.py
	python3 analysis_hybrid.py
	python3 analysis_conformal.py
	python3 analysis_dimensional.py
	python3 analysis_conformal_shift.py
	python3 analysis_replication.py
	python3 analysis_forecast.py
	python3 analysis_allometry.py
	python3 analysis_tree_allometry.py
	python3 analysis_gp.py
	python3 analysis_summary_figure.py
	python3 -m fusionflux card
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

# Assemble the flat, self-contained source tarball arXiv wants.
#
# arXiv unpacks a submission into one directory and builds there, so a source
# tree that reaches figures through `../results/` builds locally and fails on
# upload. paper.tex names its figures bare and finds them through
# \graphicspath; this target supplies the flat half of that path.
#
# The tarball carries only the figures the paper actually includes, so a
# figure added to the paper without being added here fails the build on arXiv
# rather than silently shipping without it. `tools/check_paper_submission.py`
# guards that: it is the same check the test suite runs.
arxiv: paper/paper.tex
	@python3 tools/check_paper_submission.py
	@rm -rf build/arxiv && mkdir -p build/arxiv
	@cp paper/paper.tex build/arxiv/
	@cp results/extrapolation.png results/size_extrapolation.png results/dimensional.png results/conformal_shift.png results/allometry.png results/tree_allometry.png results/gp.png build/arxiv/
	@cd build/arxiv && tar czf ../arxiv-submission.tar.gz paper.tex *.png
	@echo "wrote build/arxiv-submission.tar.gz"
	@echo "Build it exactly as arXiv will, from the flat directory:"
	@echo "    cd build/arxiv && pdflatex paper.tex && pdflatex paper.tex"

# Build the distribution and prove it works, which are different claims.
#
# `python -m build` alone only shows that a wheel can be produced. Version 0.2.0
# produced one happily: it was missing results/predictor.json, so `fusionflux
# predict` raised FileNotFoundError on every clean install, and it shipped the
# analysis scripts as top-level modules that shadowed `config`, `storage` and
# `validation` in whatever environment it landed in. Neither fault is visible
# from a checkout, because a checkout has the repository root on sys.path and
# results/ already on disk.
#
# So this runs the two suites that build a wheel, install it into a fresh
# virtualenv with no access to this directory, and run the command the README
# leads with. build/ is removed first because setuptools accumulates into
# build/lib and never prunes it, so a stale tree gets packed along with the
# current one.
dist:
	rm -rf build dist
	python3 -m build
	python3 -m pytest -q --no-cov tests/test_packaging.py tests/test_wheel_smoke.py
	@echo
	@echo "wrote dist/. Push a matching vX.Y.Z tag to publish; see docs/releasing.md."

# Is the committed PDF built from the current paper.tex?
#
# Deliberately not folded into `arxiv`, which is the target you run to *get* the
# tarball the PDF is rebuilt from: gating that on the PDF being current would
# block the only supported way of making it current. This is the release gate,
# and `docs/releasing.md` runs it in pre-flight.
paper-fresh:
	python3 tools/check_paper_submission.py --check-pdf-fresh
