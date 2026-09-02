# Releasing: DOI, preprint, and who to tell

Everything in this repository is self-attested. A reader has to take the
repository's word that the numbers follow from the data, and while the test
suite and the `reproduce` workflow make that checkable, checking it is work
nobody does for a stranger's project. The steps below are the ones that replace
self-attestation with something external: a permanent identifier, a timestamped
preprint, and the people whose database this is.

They are ordered because each depends on the one before. Do not start at step 3.

## 0. Pre-flight

Everything here has to be true of the *tagged commit*, not of a working tree.

```bash
make check        # ruff, mypy, pytest
make dist         # build the wheel and install it into a clean venv, then predict
make reproduce    # regenerate results/ from the raw data and diff numerically
make arxiv        # builds build/arxiv-submission.tar.gz and runs the submission gate
```

`make reproduce` needs both datasets and takes roughly fifteen minutes. It is
the one that matters here: a DOI is permanent, so archiving a state whose
`results/` no longer follows from the data is not correctable by an edit.

Then confirm the release metadata describes the work as it now stands:

- `.zenodo.json` description and keywords
- `CITATION.cff` abstract and `version`
- `version` in `pyproject.toml` (kept equal to the CITATION one)
- the author line in `paper/paper.tex`

## 1. Tag and release

`v0.1.0` exists and points at a commit that predates all of the current work, so
it must not be the tag a DOI is minted on.

The tag has to match `version` in `pyproject.toml`, which is `0.2.1`. The
release workflow fails rather than publishing a mismatch.

```bash
git tag -a v0.2.1 -m "All thirteen results, and a distribution that installs and runs"
git push origin v0.2.1
```

Then publish a GitHub release on that tag and **attach `paper/paper.pdf`**, so
the paper is archived alongside the code rather than only being buildable from
source.

## 1b. PyPI

The tag push runs `.github/workflows/release.yml`, which builds the sdist and
wheel and uploads them to PyPI. There is nothing to run by hand and no token to
supply: publishing uses PyPI's Trusted Publishing, so PyPI verifies the OIDC
identity of that workflow in this repository directly.

**This has to be configured once, before the first tag, and it cannot be done
retroactively.** At <https://pypi.org/manage/account/publishing/>, add a pending
publisher for the project name `fusionflux` with owner `lsalcedo100`, repository
`FusionFlux`, workflow `release.yml`, environment `pypi`. Create a GitHub
environment of the same name under Settings, Environments.

Rehearse on TestPyPI first. Run the workflow manually (Actions, Release, Run
workflow) with `test_pypi` left checked; it goes to TestPyPI, which is
throwaway, and a failure there costs nothing. A version number on the real PyPI
is permanent and cannot be reused even after a delete, so the rehearsal is worth
the five minutes.

What the release job checks before it uploads anything, and why each one exists:

- the packaging and clean-install suites (`tests/test_packaging.py`,
  `tests/test_wheel_smoke.py`), which build a wheel, install it into a fresh
  virtualenv, and run `fusionflux predict` on ITER from a directory that is not
  this repository. Version 0.2.0's metadata was self-consistent and produced a
  wheel that raised `FileNotFoundError` on the README's headline command,
  because `results/predictor.json` was not package data. Nothing in a checkout
  can see that, which is why this suite installs rather than inspects.
- that the wheel installs only `fusionflux` at top level. 0.2.0 also shipped
  every analysis script as a top-level module, so installing it put `config`,
  `storage`, `validation` and `tools` into site-packages, where they shadowed
  any other project's module of those names.
- `twine check --strict`, which catches the metadata faults PyPI rejects on
  upload, while rejection is still free.
- that the tag matches `version` in `pyproject.toml`, so the release page and
  the published version cannot disagree.

After it lands, confirm the thing this is all for actually works, from
somewhere that is not a checkout:

```bash
cd $(mktemp -d) && python3 -m venv v && ./v/bin/pip install fusionflux
./v/bin/fusionflux predict --ip-ma 15 --bt-t 5.3 --ne-line-1e19-m3 10 --p-loss-mw 87 \
                   --r-m 6.2 --inverse-aspect-ratio 0.3226 --kappa 1.7 --m-eff-amu 2.5
```

Then add the PyPI badge to the README badge row.

## 1c. scaling-audit, if it is going out too

`scaling-audit/` is a second distribution over the same `scaling_audit.py`, and
it is versioned independently: the method changes rarely, the study's version
tracks its results. A shared tag would force a release of one whenever the other
moved, so it has its own tag prefix and its own workflow.

```bash
git tag -a scaling-audit-v0.1.0 -m "The extrapolation audit as a standalone package"
git push origin scaling-audit-v0.1.0
```

Trusted Publishing again, configured the same way but with project name
`scaling-audit` and environment `pypi-scaling-audit`. Rehearse on TestPyPI
first, the same way.

The build gate is what matters here. The package's whole claim is that the
fusion study, the mammalian replication and the tree-allometry ladder all run
through the published module rather than a copy of it, so
`.github/workflows/release-scaling-audit.yml` asserts that the module inside the
built wheel is byte-identical to `scaling_audit.py` at the repository root, that
nothing from the study leaked into the wheel, and that its README documents the
API that actually exists. That last check exists because the README's examples
were wrong when first written: they named three parameters and a column that do
not exist.

After it lands, update the two places that currently say it is unpublished:
the `scaling_audit.py` bullet in `README.md` and the Provenance section of
`scaling-audit/README.md`.

## 2. Zenodo DOI

Full instructions, including the concept-versus-version DOI distinction and the
licence caveat about the HDB5 data, are in [`../paper/README.md`](../paper/README.md).
The short version: enable the repository in Zenodo's GitHub integration
*before* publishing the release, because Zenodo only archives releases published
after the toggle is on. Then add the badge to the README and the DOI to the
paper's title page, and cut a `v0.2.2` for that edit.

## 3. arXiv

`make arxiv` produces `build/arxiv-submission.tar.gz`, which contains
`paper.tex` and the four figures it includes, with no parent-directory paths, so
it builds in the single flat directory arXiv unpacks into. That is verified:
`tools/check_paper_submission.py` enforces it and runs in the ordinary suite.

Suggested categories: `physics.plasm-ph` primary, cross-listed to `cs.LG` or
`stat.ML`. The paper is a machine-learning result about a plasma-physics
baseline and either community is a reasonable primary; plasma is the one whose
referees can check the physics claims.

**Budget time for endorsement.** arXiv requires an endorsement for a first
submission to most archives, and it is not automatic for someone without an
institutional affiliation or a `.edu` address. The endorser has to be an
established submitter in that archive. This is the practical reason step 4 is
worth doing first rather than last: the people most likely to endorse this are
the people it is most useful to.

## 4. Tell the people whose data this is

This is the highest-variance step and the cheapest. The finding is a negative
result about how confinement models are validated, computed on the ITPA's own
database, and it is directly useful to the group that maintains it.

Worth contacting:

- **Geert Verdoolaege**, lead author of the 2021 *Nucl. Fusion* paper describing
  HDB5 and the maintainer of the dataset this study uses.
- The **ITPA Confinement Database and Modelling Topical Group**, which is the
  body that would actually act on a validation-methodology finding.

Keep it to a few sentences and lead with the result rather than with the
request. The three things worth stating are the reversal (13 of 13 machines),
the constraint result (0.183 at the ITER-matched cut, better than the analytic
law fitted with those machines included), and that the whole thing regenerates
from their published file by SHA-256. A link to the repository and the paper is
enough; do not attach anything.

A reply from someone in that group is worth more than any further work on the
repository, and an endorsement for step 3 is a plausible side effect.

## 5. A workshop, if the timing works

**NeurIPS Machine Learning and the Physical Sciences (ML4PS)** is the natural
venue: it takes short papers, it exists for exactly this kind of ML-meets-a-physical-baseline
result, and negative results with a stated mechanism do well there. Check the
current year's deadline, which is usually in early autumn. The paper is already
close to the right length; the constraint result is the part to lead with,
because it is the positive finding.

Neither the DOI nor an arXiv posting conflicts with a workshop submission.
