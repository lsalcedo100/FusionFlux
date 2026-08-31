# The paper

`paper.tex` is a six-page writeup of the results in
[`../results/RESULTS.md`](../results/RESULTS.md), in the format a reviewer can
evaluate at a glance. Every number in it is taken from the generated artifacts
under `../results/`, so regenerate those first if anything has changed.

## Build

```bash
tectonic paper/paper.tex     # single self-contained binary, downloads packages on demand
# or, with a TeX distribution already installed:
pdflatex paper/paper.tex && pdflatex paper/paper.tex
```

The two figures are included from `../results/`, so the PDF has to be built
from a checkout where those exist. `pdflatex` needs two passes to resolve the
table and figure references.

## Before publishing

The author line in `paper.tex` was taken from this repository's git
configuration. Set the full name and an affiliation before submitting anywhere
or minting a DOI.

## Minting a Zenodo DOI

`../.zenodo.json` holds the record metadata (title, description, creators,
license, keywords, and the related identifiers linking the record to the HDB5
dataset on OSF and to the two cited papers). Zenodo reads it automatically.

The usual route is the GitHub integration, which is the one worth using because
it makes the DOI point at a specific tagged commit rather than at a moving
branch:

1. Sign in to <https://zenodo.org> with the GitHub account that owns the repo.
2. Under **Settings to GitHub**, toggle the repository on.
3. Push a tag and publish a GitHub release (`git tag v1.0.0 && git push --tags`,
   then create the release). Zenodo archives that release and mints the DOI.
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
