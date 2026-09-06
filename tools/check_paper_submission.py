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
  wrong. A journal also wants an affiliation and an ORCID, neither of which is
  knowable from inside the repository, so both are written as `FILL-IN` markers
  and fail here rather than surviving as a note someone meant to act on.

Two further checks are opt-in, each because it can fail for a reason a
contributor cannot fix from a plain checkout:

* **A stale committed PDF.** `paper/paper.pdf` is committed rather than built on
  demand, so it goes out of date the moment `paper.tex` gains a section and is
  not rebuilt. Every reader who follows the README's link gets that PDF, and a
  DOI would archive it permanently. `--check-pdf-fresh` compares the section
  titles in the source against the text of the PDF and reports the ones missing.
  It is deliberately *not* part of the default rule set: `make check` must stay
  green on a machine with no `pdflatex`, and the release path is where this
  actually matters.
* **A stale provenance pin.** The paper names the commit its numbers were
  produced at. Regenerating `results/` and committing leaves that hash behind,
  and the paper goes on naming a tree whose artifacts are no longer the ones it
  reports. `--check-provenance` compares the pin against the last commit that
  touched `results/`, and fails if the directory is dirty besides. It needs git,
  which a source tarball does not have.

Run standalone (`python3 tools/check_paper_submission.py`) or via `make arxiv`,
which refuses to build the tarball if anything here fails.
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

# The order `\DeclareGraphicsExtensions` in paper.tex gives them: vector first.
GRAPHICS_EXTENSIONS = (".pdf", ".png")

PLACEHOLDER_AUTHORS = (
    "Your Name",
    "TODO",
    "FIXME",
    "Author Name",
    "Anonymous",
    # A journal wants an affiliation and an ORCID where arXiv and Zenodo do not.
    # Both are unknowable from inside the repository, so they are written as
    # FILL-IN markers and caught here rather than left as a note someone has to
    # remember at submission time.
    "FILL-IN",
)


def _strip_comments(latex: str) -> str:
    """Drop LaTeX comments so a rule discussed in a comment is not a violation.

    A bare `%` starts a comment; `\\%` is an escaped percent sign and does not.
    """
    return re.sub(r"(?<!\\)%.*", "", latex)


def included_figures(latex: str) -> list[str]:
    return re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", latex)


def resolve_figure(figure: str) -> Path | None:
    """The file LaTeX would actually load for this `\\includegraphics` argument.

    The paper names its figures without an extension so the vector copy wins
    where one exists, which means the name in the source matches no file on disk
    and this check has to resolve it the way `\\DeclareGraphicsExtensions` does:
    in order, first hit wins. Comparing the bare name against `results/` instead
    would report every figure missing.
    """
    name = Path(figure).name
    if Path(figure).suffix:
        found = FIGURE_DIR / name
        return found if found.exists() else None
    for extension in GRAPHICS_EXTENSIONS:
        found = FIGURE_DIR / (name + extension)
        if found.exists():
            return found
    return None


def bundled_figures(makefile: str) -> set[str]:
    """The figures `make arxiv` copies into the flat submission directory."""
    match = re.search(
        r"^\t@cp ((?:results/\S+\.(?:png|pdf) )*results/\S+\.(?:png|pdf)) build/arxiv/",
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

    resolved = {}
    for figure in figures:
        if figure.startswith(("/", "../")) or "/" in figure:
            problems.append(
                f"\\includegraphics{{{figure}}} is not a bare filename; arXiv builds in "
                "one flat directory, so the path must resolve through \\graphicspath"
            )
        found = resolve_figure(figure)
        if found is None:
            problems.append(
                f"\\includegraphics{{{figure}}} resolves to no file under results/ "
                f"(tried {', '.join(figure + e for e in GRAPHICS_EXTENSIONS)})"
            )
        else:
            resolved[figure] = found

    bundled = bundled_figures(makefile.read_text())
    missing = {found.name for found in resolved.values()} - bundled
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
    # A long heading can be broken with a discretionary hyphen, which arrives
    # here as "ex- cludes". Two repairs cover both cases: dropping the hyphen
    # rejoins a word TeX split, and keeping it rejoins a compound that happened
    # to break at its own hyphen ("size- matched"). A title is present if it
    # matches any of the three readings.
    candidates = (
        normalized,
        normalized.replace("- ", ""),
        normalized.replace("- ", "-"),
    )

    latex = _strip_comments(paper.read_text())
    missing = []
    for title in re.findall(r"\\section\{([^}]*)\}", latex):
        # Section titles are plain prose here; strip the little LaTeX that does
        # appear so the comparison is against what a reader sees.
        plain = re.sub(r"\\[a-zA-Z]+\s*", "", title).replace("{", "").replace("}", "")
        # LaTeX dashes reach the PDF as en/em dash glyphs, which the table above
        # has already folded back to "-" and "--". Fold the source the same way
        # so "Connor--Taylor" matches the "Connor-Taylor" a reader sees.
        plain = plain.replace("---", "\u2014").replace("--", "\u2013")
        for glyph, replacement in TYPESET_SUBSTITUTIONS.items():
            plain = plain.replace(glyph, replacement)
        plain = re.sub(r"\s+", " ", plain).strip()
        if plain and not any(plain in candidate for candidate in candidates):
            missing.append(plain)
    return missing


def stale_provenance(paper: Path = PAPER, root: Path = ROOT) -> list[str]:
    """Ways the commit the paper pins can stop describing the numbers in it.

    The paper states that every number and figure was produced at one named
    commit. That claim is the only thing tying the printed values to a tree a
    reader can check out, and it goes wrong silently: regenerating `results/`
    and committing costs nothing and leaves the pinned hash behind, so the paper
    keeps naming a commit whose artifacts are no longer the ones it reports. It
    happened once already, over three commits.

    Two conditions have to hold. The pinned commit must be the last one that
    touched `results/`, and `results/` must have no uncommitted changes, since a
    dirty artifact is by definition at no commit at all.

    Returns an empty list when git cannot answer, which covers a source tarball
    and any environment without git. An unanswerable question is not a defect in
    the paper, and reporting it as one would train the reader to ignore this.
    """
    latex = _strip_comments(paper.read_text())
    pinned = re.search(r"\\texttt\{([0-9a-f]{40})\}", latex)
    if pinned is None:
        return [r"no 40-character commit hash in \texttt{...}; the paper pins nothing"]

    def git(*arguments: str) -> str | None:
        try:
            finished = subprocess.run(
                ("git", *arguments), cwd=root, capture_output=True, text=True, timeout=30
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return finished.stdout.strip() if finished.returncode == 0 else None

    latest = git("log", "-1", "--format=%H", "--", "results/")
    if not latest:
        return []

    problems = []
    if latest != pinned.group(1):
        problems.append(
            f"the paper pins commit {pinned.group(1)[:12]}, but results/ was last changed "
            f"at {latest[:12]}. Either repoint the paper or confirm the numbers are "
            "unchanged and repoint it anyway; the pin is a claim about which tree "
            "produced the printed values."
        )

    dirty = git("status", "--porcelain", "--", "results/")
    if dirty:
        problems.append(
            "results/ has uncommitted changes, so no commit describes the artifacts "
            "the paper was built from:\n      "
            + "\n      ".join(dirty.splitlines()[:10])
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    problems = check()

    if "--check-provenance" in arguments:
        problems.extend(stale_provenance())

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
