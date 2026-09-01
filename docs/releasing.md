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

```bash
git tag -a v0.2.0 -m "All twelve results: the reversal, its mechanisms, and the constraint that repairs it"
git push origin v0.2.0
```

Then publish a GitHub release on that tag and **attach `paper/paper.pdf`**, so
the paper is archived alongside the code rather than only being buildable from
source.

## 2. Zenodo DOI

Full instructions, including the concept-versus-version DOI distinction and the
licence caveat about the HDB5 data, are in [`../paper/README.md`](../paper/README.md).
The short version: enable the repository in Zenodo's GitHub integration
*before* publishing the release, because Zenodo only archives releases published
after the toggle is on. Then add the badge to the README and the DOI to the
paper's title page, and cut a `v0.2.1` for that edit.

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
