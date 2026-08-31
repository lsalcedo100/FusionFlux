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
FIGURE_DIR = ROOT / "results"

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


def main() -> int:
    problems = check()
    if problems:
        print(f"{PAPER.relative_to(ROOT)} is not ready to submit:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"{PAPER.relative_to(ROOT)}: ready to submit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
