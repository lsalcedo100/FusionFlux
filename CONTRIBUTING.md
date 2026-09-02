# Contributing

This repository is a study, so the useful contributions are not quite the usual ones. The most valuable thing anyone can do here is **show that a number is wrong**, and the second is to run the audit on a dataset it has never seen.

## The one rule that matters

**Every number in the prose is bound to a generated artifact, and nothing is typed by hand.**

`README.md`, `results/RESULTS.md`, the paper and the site all quote figures. `tests/test_reported_numbers.py` ties each one to the field in `results/` it came from, so a claim fails in both directions: rerun an analysis and the number moves, the prose stops matching; edit the prose, it stops matching the artifact.

So if you change an analysis:

```bash
python3 analysis_whatever.py       # regenerate its artifacts
make check                         # ruff, mypy, pytest
```

and then update the prose to whatever the new value is. **Do not adjust a tolerance to make a test pass.** If a reported number moved, either the change was wrong or the document is now stale; both need a human decision, and widening the tolerance hides which.

## Getting set up

```bash
python3.12 -m venv .venv && source .venv/bin/activate
python3 -m pip install -e ".[dev]" -c constraints.txt

make check        # ruff, mypy, pytest: the CI gate, about six minutes
```

Use Python 3.10 or newer. Stock macOS still ships 3.9, and a virtualenv built from it fails the type check and four tests rather than refusing to install.

The full study needs two third-party datasets, fetched on demand and pinned by SHA-256:

```bash
python3 hdb5.py download
python3 -c "import replication; replication.download_db523()"
python3 -c "import allometry; allometry.download_allometry()"
python3 -c "import tree_allometry; tree_allometry.download_baad()"
```

Without them 35 tests skip rather than fail. If you would rather not install anything, `docker build -t fusionflux . && docker run --rm fusionflux` does the whole reproduction in a pinned environment.

## What is most welcome

**A number that does not reproduce.** `make reproduce` regenerates everything from raw data and diffs it. If it disagrees with what is committed, that is the most important issue anyone can open, and it should say which artifact and by how much.

**The audit on a dataset this study has never seen.** `scaling_audit.py` is the method with no plasma physics in it, published separately as [`scaling-audit`](scaling-audit/). Results 13 and 15 are two domains it has already been run on. A third, especially one where the reversal does *not* appear, would sharpen the claim more than another tokamak result would.

**A model that breaks a stated conclusion.** Result 14 is exactly this: the limitations section said a Gaussian process with a physically motivated kernel was untested, someone tested it, and the explanation of Results 4 and 5 changed as a result. The limitations sections are a list of open invitations, not a disclaimer.

**Corrections to the physics.** The scaling-law and dimensional-analysis derivations are the parts where a domain expert is most likely to spot something a careful non-expert would not.

## What to expect from a change

- **Lint and types are not optional.** `ruff check .` and `mypy .` are clean and CI enforces both.
- **Coverage is a ratchet.** `fail_under` in `pyproject.toml` sits just under the measured figure. Raise it when coverage genuinely rises; never lower it to make a run pass. If it trips, find out what stopped running.
- **A new result needs its own module, artifacts, tests and limitations section.** See `docs/repository.md` for how the existing ones are laid out. The "What Result N does not show" sections are load-bearing: a result without one reads as overclaiming.
- **Comments explain why, not what.** The surrounding code is unusually heavily commented because most of the decisions in it are non-obvious and several were wrong once. Matching that is more useful than matching a line length.

## Reporting something

Open an issue. The templates ask for the specific artifact and command, because "the numbers look off" cannot be acted on and "`results/gp.json` gives 0.191 where RESULTS.md says 0.19" can.

For anything about the ITPA data itself, note that the dataset is third-party, fetched from OSF and never redistributed here. Questions about the data belong with its maintainers; questions about what this study does with it belong here.

## Licence

MIT, and contributions are taken under it. The datasets are not covered by it: each is fetched from its own source under its own terms and cited in `README.md`.
