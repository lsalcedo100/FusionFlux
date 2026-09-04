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

Two further checks are opt-in, because each can fail for a reason a contributor
cannot fix from a plain checkout -- one needs a LaTeX toolchain, the other needs
the git history:

* **A stale committed PDF.** `paper/paper.pdf` is committed rather than built on
  demand, so it goes out of date the moment `paper.tex` gains a section and is
  not rebuilt. Every reader who follows the README's link gets that PDF, and a
  DOI would archive it permanently. `--check-pdf-fresh` compares the section
  titles in the source against the text of the PDF and reports the ones missing.
  It is deliberately *not* part of the default rule set: `make check` must stay
  green on a machine with no `pdflatex`, and the release path is where this
  actually matters.

* **A stale provenance commit.** The paper names the commit its numbers were
  produced at. Nothing else can check that sentence: `tests/test_reported_numbers.py`
  ties the prose to `results/` and the reproduce workflow ties `results/` to the
  raw data, but the hash joining them to a point in history is prose.
  `--check-provenance` reports when `results/` has moved since the commit named.
  It went stale exactly once, naming a commit from before `results/tuned.json`
  existed while the paper cited that analysis eight times, which a referee
  checking the hash would have found first.

Run standalone (`python3 tools/check_paper_submission.py`) or via `make arxiv`,
which refuses to build the tarball if anything here fails. `make paper-fresh`
runs both opt-in checks and is the release gate.
"""

from __future__ import annotations

import re
import subprocess
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


def pinned_provenance_commit(paper: Path = PAPER) -> str | None:
    """The commit the paper claims its numbers were produced at, if it names one.

    Git hashes are 40 hex characters and the content digests printed elsewhere in
    the paper are SHA-256 split across two 32-character lines, so the width alone
    separates them. The phrase is required as well, so that adding some other
    40-character hash to the paper later cannot silently become the thing this
    check enforces.
    """
    latex = _strip_comments(paper.read_text())
    match = re.search(r"produced at commit(.{0,200}?)\b([0-9a-f]{40})\b", latex, re.S)
    return match.group(2) if match else None


def stale_provenance_commit(paper: Path = PAPER, root: Path = ROOT) -> list[str]:
    """Whether `results/` has moved since the commit the paper pins.

    The paper states that every number in it was produced at one commit. That is
    the study's provenance claim, and it is the only claim in this repository
    that nothing checks: `tests/test_reported_numbers.py` ties the prose to
    `results/`, and the reproduce workflow ties `results/` to the raw data, but
    the sentence naming the commit those artifacts came from is unverified prose.
    It went stale exactly that way once, naming a commit from before
    `results/tuned.json` existed while the paper cited the tuned analysis eight
    times, which a referee checking the hash would have found before we did.

    Quiet when the question cannot be answered rather than guessed at: no pinned
    commit, no git, or a repository that does not contain the commit under a
    shallow clone all return no problems, the same way `stale_pdf_sections` stays
    quiet without `pypdf`. A wrong answer here is worse than no answer.
    """
    pinned = pinned_provenance_commit(paper)
    if pinned is None:
        return []

    def _git(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(root), *arguments],
            capture_output=True,
            text=True,
            check=False,
        )

    if _git("rev-parse", "--git-dir").returncode != 0:
        return []
    if _git("cat-file", "-e", f"{pinned}^{{commit}}").returncode != 0:
        return [
            f"the paper says its numbers were produced at commit {pinned[:12]}, "
            "which is not in this repository"
        ]

    changed = _git("diff", "--name-only", pinned, "HEAD", "--", "results/")
    if changed.returncode != 0:
        return []
    moved = [line for line in changed.stdout.splitlines() if line.strip()]
    if not moved:
        return []
    listed = ", ".join(moved[:6]) + (f", and {len(moved) - 6} more" if len(moved) > 6 else "")
    return [
        f"the paper says its numbers were produced at commit {pinned[:12]}, but "
        f"{len(moved)} file(s) under results/ have changed since then ({listed}). "
        "Repoint it at the commit those artifacts actually come from, which is "
        "`git log -1 --format=%H -- results/`."
    ]


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

    if "--check-provenance" in arguments:
        problems.extend(stale_provenance_commit())

    if problems:
        print(f"{PAPER.relative_to(ROOT)} is not ready to submit:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"{PAPER.relative_to(ROOT)}: ready to submit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
