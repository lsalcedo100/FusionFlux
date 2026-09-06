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

The six figures are named bare, without an extension, and resolved through
`\graphicspath`, which lists `../results/` first and then the current directory.
So the paper builds from `paper/` in a checkout where `results/` exists, and
equally from the flat directory `make arxiv` assembles. `pdflatex` needs two
passes to resolve the table and figure references.

Dropping the extension is what lets `\DeclareGraphicsExtensions{.pdf,.png}`
choose: `figures.py` writes every figure as both, and the paper takes the
vector copy. IOP asks for vector line art in preference to raster, and for these
axes the PDF is the smaller file as well, roughly 195 KB against 1.1 MB for a
raster at matching effective resolution. The PNG stays because the README and
the built page need something that renders inline, and it is the fallback for a
figure that has not been regenerated since.

## Why the figures stack

Every paper figure is authored at `PAPER_WIDTH_IN`, close to the 6.378 in text
width it is printed at, with its panels stacked in one column. That is not a
style choice, and reverting it to side-by-side panels will reintroduce a real
defect.

The figures were once drawn 12.5 to 13.5 in wide and placed across
`\textwidth`, so everything on them reached the page at about half its
specified size: 9 pt tick labels arrived at 4.3 pt, well under the 7 pt or so
that figure lettering needs. Enlarging the type on a canvas that wide does not
fix it. What governs legibility after scaling is type size *relative to the
canvas*, so raising the type crowds the panels instead, and at the sizes needed
the titles and tick labels collide outright. Shrinking the canvas with the type
held fixed gives the identical ratio and fails the same way; both were tried.

The only thing that buys room is giving each panel more of the width, which
means one panel per row. A three-panel figure is then about 8 in tall, and the
paper is a few pages longer than it would be with the old figures. That is the
trade: the type on the page is now the size it says it is.

## Before publishing

`tools/check_paper_submission.py` enforces the things that build fine here and
fail, or mislead, once the paper leaves the repository: a `\today` date on a
permanent record, figure paths that escape the flat directory arXiv builds in,
a figure the paper includes that `make arxiv` does not bundle, and a
placeholder author line. It runs in the ordinary test suite
(`tests/test_paper_submission.py`) and again as the first step of `make arxiv`,
which refuses to build the tarball if any of it fails.

Two more run on request, since each needs something a plain checkout may not
have. `--check-pdf-fresh` needs a LaTeX toolchain and answers whether the
committed PDF was built from the current source. `--check-provenance` needs git
and answers whether the commit the paper pins is still the one `results/` was
last written at. `make paper-fresh` runs both.

The provenance check exists because the pin went stale once already: the paper
named a commit three commits behind `results/`. Nothing broke, since the three
changed only benchmark timings and float64 tails, but the paper was making a
claim about which tree produced its numbers and the claim had quietly stopped
being true.

It also fixes the order of two commits, since a commit cannot contain its own
hash. When `results/` changes, commit it **first**, on its own, then put that
hash in `paper.tex`, rebuild the PDF, and commit the paper. The check then sees
the pin and the last commit that touched `results/` as the same thing. Doing it
the other way round leaves the gate red and no hash that would satisfy it.

## References

`references.bib` holds the same 26 references as the `thebibliography` block in
`paper.tex`. Both exist because their consumers want different things: arXiv
builds a submission with no BibTeX pass and no `.bbl`, so the printed list has
to be in the source, while a journal wants a `.bib` to run through its own style
file. `tests/test_paper_bibliography.py` binds them, in both directions, keys
and DOIs, so the two cannot drift into disagreeing.

Every DOI was resolved against Crossref rather than reconstructed from the
volume and page. That found one error already: the symbolic-regression paper was
printed as *Nucl. Fusion* **55**, 073009, which is a different Murari paper. The
right one is *Plasma Phys. Control. Fusion* **57**, 014008.

## Submitting to a journal

Nuclear Fusion is the home venue: IPB98, HDB5, Connor-Taylor and ITPA20 are all
NF papers, and the readership is the one the result is aimed at.

IOP is format-free at initial submission, so `article` is fine and `iopart.cls`
is only needed if the paper is accepted. Their abstract guidance is 300 words
and this one is 265. Submission goes through ScholarOne, which asks separately
for the things now carried in the source: affiliation, ORCID, funding, competing
interests, author contributions, a data availability statement, and a statement
on the use of AI tools.

Two things to settle that are not in any file here:

- **Talk to the database maintainers before submitting, not after.** This is a
  negative result about how a community validates models on its own database.
  Being able to say in the cover letter that the ITPA group has seen it is worth
  more than any formatting, and it is how a misreading of STD5's selection
  criteria surfaces before a referee finds it.
- **The cover letter should lead with the constraint result**, not the critique.
  The Connor-Taylor fit and the linear-plus-RBF process are what an editor can
  send to referees as a contribution; the inversion is what makes them
  necessary.

A preprint is compatible with IOP policy either way, and journal submission
needs no arXiv endorsement, so the endorsement problem in `docs/releasing.md`
does not gate this.

## Submitting to arXiv

```bash
make arxiv        # checks the paper, then writes build/arxiv-submission.tar.gz
```

The tarball is flat and self-contained: `paper.tex` plus the figures it
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
