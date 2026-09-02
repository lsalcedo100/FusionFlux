# The paper

`paper.tex` is a condensed writeup of the results in
[`../results/RESULTS.md`](../results/RESULTS.md), in the format a reviewer can
evaluate at a glance. Every number in it is taken from the generated artifacts
under `../results/`, so regenerate those first if anything has changed.

**Scope: the paper covers all twelve results.** It builds to the reversal and
its three mechanisms, then to the two things that repair it: the bounded hybrid
correction, and the Connor-Taylor constraint hierarchy that beats it. It closes
on the replication over STD5-disjoint rows and the locked device forecast.
Result 9's prior-shrinkage control is reported as a paragraph rather than a
section, since its finding is a negative one about an alternative to Result 8
rather than a result in its own right.

`tests/test_reported_numbers.py` binds every headline number to the artifact it
came from, and its `LATE_RESULTS` document tuple includes `PAPER` and
`PAPER_PDF`, so the paper's copies are enforced along with the prose. The PDF is
committed, so **rebuild it whenever `paper.tex` changes** or the test will catch
the two disagreeing.

Cross-references use `\label`/`\ref` rather than hardcoded section numbers.
Two hand-written ones were wrong before that change, and adding a section to the
middle of the paper is exactly the edit that produces such an error silently.

## Build

```bash
tectonic paper/paper.tex     # single self-contained binary, downloads packages on demand
# or, with a TeX distribution already installed:
pdflatex paper/paper.tex && pdflatex paper/paper.tex
```

The two figures are named bare and resolved through `\graphicspath`, which
lists `../results/` first and then the current directory. So the paper builds
from `paper/` in a checkout where `results/` exists, and equally from the flat
directory `make arxiv` assembles. `pdflatex` needs two passes to resolve the
table and figure references.

## Before publishing

`tools/check_paper_submission.py` enforces the things that build fine here and
fail, or mislead, once the paper leaves the repository: a `\today` date on a
permanent record, figure paths that escape the flat directory arXiv builds in,
a figure the paper includes that `make arxiv` does not bundle, and a
placeholder author line. It runs in the ordinary test suite
(`tests/test_paper_submission.py`) and again as the first step of `make arxiv`,
which refuses to build the tarball if any of it fails.

## Submitting to arXiv

```bash
make arxiv        # checks the paper, then writes build/arxiv-submission.tar.gz
```

The tarball is flat and self-contained: `paper.tex` plus the two figures it
includes, which is the shape arXiv unpacks and builds. Verify it compiles the
way arXiv will, from that flat directory rather than from `paper/`:

```bash
cd build/arxiv && pdflatex paper.tex && pdflatex paper.tex
```

Two things to settle before uploading:

- **Category.** `physics.plasm-ph` is where this work belongs and where the
  people who would care about it read. `stat.ML` or `cs.LG` are defensible
  cross-lists, since the load-bearing result is about validation protocol and
  distribution shift rather than about plasma physics.
- **Endorsement.** In physics, each subject class is its own endorsement
  domain, so `physics.plasm-ph` needs its own. Since arXiv's January 2026
  policy change, the path that does not require a personal endorsement needs
  *both* an institutional email address and prior authorship in that same
  domain. Without those, submitting means asking an established
  `physics.plasm-ph` author for a personal endorsement, which is a request
  worth pairing with the email that shares the result.

## Minting a Zenodo DOI

`../.zenodo.json` holds the record metadata (title, description, creators,
license, keywords, and the related identifiers linking the record to the HDB5
dataset on OSF and to the two cited papers). Zenodo reads it automatically.

The usual route is the GitHub integration, which is the one worth using because
it makes the DOI point at a specific tagged commit rather than at a moving
branch:

1. Sign in to <https://zenodo.org> with the GitHub account that owns the repo.
2. Under **Settings to GitHub**, toggle the repository on.
3. Push a tag and publish a GitHub release, then Zenodo archives it and mints
   the DOI. **Tag the state you actually want archived.** `v0.1.0` is already
   pushed and points at a commit that predates the current work, so a DOI
   minted on it would permanently archive the older version; cut a new tag
   instead.
4. Zenodo issues two DOIs: a **concept DOI** that always resolves to the latest
   version, and a **version DOI** fixed to that release. Cite the concept DOI in
   the README and the version DOI in anything that quotes specific numbers.
5. Add the badge Zenodo gives you to the top of the main README, and put the DOI
   on the paper's title page.

Attach `paper.pdf` to the GitHub release so it is archived alongside the code
rather than only being buildable from source.

Two things to get right before step 3, because a published DOI is permanent and
a correction means a new version rather than an edit:

- the author name and affiliation in both `paper.tex` and `.zenodo.json`
- the license: `.zenodo.json` currently declares MIT, matching `../LICENSE`.
  The HDB5 dataset is **not** covered by it. The data is third-party, is not
  redistributed in this repository, and its own terms come from the ITPA. The
  paper and `.zenodo.json` both say so, and that should stay true of anything
  attached to the release.
