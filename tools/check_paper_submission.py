"""Check that `paper/paper.tex` is in a state arXiv and Zenodo can accept.

Three of these are things that build perfectly well in the repository and fail,
or quietly mislead, only once the paper leaves it. That is exactly the class of
defect nothing else here catches: `make check` is green, the PDF looks right,
and the problem appears at upload time or, worse, in a permanent record.

* **A build-time date.** `\\date{\\today}` puts the compilation date on the
  paper. A DOI record and an arXiv posting are permanent, so the archived PDF
  and any later rebuild disagree about when the work is from, and the PDF is not
  reproducible byte-for-byte either.
* **Parent-directory figure paths.** arXiv unpacks a submission into a single
  directory and builds there. `\\includegraphics{../results/fig.png}` resolves
  in the repository and escapes the submission root on arXiv, where it fails.
* **Figure drift against the submission bundle.** `make arxiv` copies a fixed
  list of figures. A figure added to the paper and not to that list would build
  locally and fail on upload, so the two are compared rather than trusted.
* **A placeholder author line**, which is cheap to check and permanent to get
  wrong.

A fifth check is opt-in, because it is the only one that can fail for a reason
a contributor cannot fix without a LaTeX toolchain installed:

* **A stale committed PDF.** `paper/paper.pdf` is committed rather than built on
  demand, so it goes out of date the moment `paper.tex` gains a section and is
  not rebuilt. Every reader who follows the README's link gets that PDF, and a
  DOI would archive it permanently. `--check-pdf-fresh` compares the section
  titles in the source against the text of the PDF and reports the ones missing.
  It is deliberately *not* part of the default rule set: `make check` must stay
  green on a machine with no `pdflatex`, and the release path is where this
  actually matters.

Run standalone (`python3 tools/check_paper_submission.py`) or via `make arxiv`,
which refuses to build the tarball if anything here fails.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "paper.tex"
MAKEFILE = ROOT / "Makefile"
PDF = ROOT / "paper" / "paper.pdf"
FIGURE_DIR = ROOT / "results"

# Characters LaTeX substitutes on the way to the PDF. Comparing section titles
# against extracted text without undoing these reports sections stale that are
# present, which is worse than no check at all: it teaches the reader to ignore
# it. Ligatures first (ff, fi, fl and friends become one glyph), then the
# typographic quotes and dashes that `'`, `` ` `` and `--` are set as.
TYPESET_SUBSTITUTIONS = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\u2019": "'",
    "\u2018": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "--",
}

PLACEHOLDER_AUTHORS = ("Your Name", "TODO", "FIXME", "Author Name", "Anonymous")


def _strip_comments(latex: str) -> str:
    """Drop LaTeX comments so a rule discussed in a comment is not a violation.

    A bare `%` starts a comment; `\\%` is an escaped percent sign and does not.
    """
    return re.sub(r"(?<!\\)%.*", "", latex)


def included_figures(latex: str) -> list[str]:
    return re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", latex)


def bundled_figures(makefile: str) -> set[str]:
    """The figures `make arxiv` copies into the flat submission directory."""
    match = re.search(r"^\tarxiv:.*$|^\t@cp (results/\S+\.png(?: results/\S+\.png)*) build/arxiv/",
                      makefile, re.MULTILINE)
    if match is None or match.group(1) is None:
        return set()
    return {Path(p).name for p in match.group(1).split()}


def check(paper: Path = PAPER, makefile: Path = MAKEFILE) -> list[str]:
    """Return a list of problems; empty means the paper is submission-ready."""
    problems: list[str] = []
    raw = paper.read_text()
    latex = _strip_comments(raw)

    if re.search(r"\\date\{\s*\\today\s*\}", latex):
        problems.append(
            r"\date{\today} puts the build date on a permanent record; use a fixed date"
        )

    if not re.search(r"\\graphicspath\{", latex):
        problems.append(
            r"no \graphicspath: the paper cannot resolve figures from both "
            "the repository and a flat arXiv submission directory"
        )

    figures = included_figures(latex)
    if not figures:
        problems.append("no \\includegraphics found; the figure check cannot mean anything")

    for figure in figures:
        if figure.startswith(("/", "../")) or "/" in figure:
            problems.append(
                f"\\includegraphics{{{figure}}} is not a bare filename; arXiv builds in "
                "one flat directory, so the path must resolve through \\graphicspath"
            )
        if not (FIGURE_DIR / Path(figure).name).exists():
            problems.append(f"\\includegraphics{{{figure}}} has no file under results/")

    bundled = bundled_figures(makefile.read_text())
    missing = {Path(f).name for f in figures} - bundled
    if missing:
        problems.append(
            f"figure(s) {sorted(missing)} are included by the paper but not copied by "
            "`make arxiv`, so the submission would build here and fail on arXiv"
        )

    author = re.search(r"\\author\{(.+?)\}\s*$", latex, re.MULTILINE | re.DOTALL)
    if author is None:
        problems.append(r"no \author line")
    else:
        for placeholder in PLACEHOLDER_AUTHORS:
            if placeholder.lower() in author.group(1).lower():
                problems.append(f"\\author still contains the placeholder {placeholder!r}")

    return problems


def stale_pdf_sections(paper: Path = PAPER, pdf: Path = PDF) -> list[str]:
    """Section titles present in the source but absent from the committed PDF.

    Returns an empty list when the PDF is fresh, and also when it cannot be read
    at all: an unreadable PDF is a different problem from a stale one, and
    reporting it here would turn a missing optional dependency into a paper
    defect. `pypdf` is in the dev extra, so a developer environment has it.
    """
    if not pdf.exists():
        return []
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover - pypdf is in the dev extra
        return []

    try:
        text = " ".join(page.extract_text() or "" for page in PdfReader(str(pdf)).pages)
    except Exception:  # pragma: no cover - a corrupt PDF is not a staleness result
        return []
    # Undo the substitutions LaTeX made on the way to the PDF, so the comparison
    # is against what the source actually says. Without this the check reports
    # sections stale that are present: the ff/fi/fl ligatures and the
    # right single quote in a possessive both caused exactly that.
    for glyph, plain in TYPESET_SUBSTITUTIONS.items():
        text = text.replace(glyph, plain)
    normalized = re.sub(r"\s+", " ", text)

    latex = _strip_comments(paper.read_text())
    missing = []
    for title in re.findall(r"\\section\{([^}]*)\}", latex):
        # Section titles are plain prose here; strip the little LaTeX that does
        # appear so the comparison is against what a reader sees.
        plain = re.sub(r"\\[a-zA-Z]+\s*", "", title).replace("{", "").replace("}", "")
        plain = re.sub(r"\s+", " ", plain).strip()
        if plain and plain not in normalized:
            missing.append(plain)
    return missing


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    problems = check()

    if "--check-pdf-fresh" in arguments:
        missing = stale_pdf_sections()
        if missing:
            problems.append(
                "paper/paper.pdf does not contain "
                + ", ".join(f"{title!r}" for title in missing)
                + ". The committed PDF is stale: rebuild it with "
                "`make arxiv && cd build/arxiv && pdflatex paper.tex && pdflatex paper.tex` "
                "and copy the result over paper/paper.pdf."
            )

    if problems:
        print(f"{PAPER.relative_to(ROOT)} is not ready to submit:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"{PAPER.relative_to(ROOT)}: ready to submit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
