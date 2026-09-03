"""`paper/paper.tex` must stay in a state arXiv and Zenodo can accept.

The paper is the artifact that leaves the repository, and the ways it breaks on
the way out are invisible from inside: it compiles here, `make check` is green,
and the failure shows up at upload time or, for a DOI, in a record that cannot
be edited afterwards. `tools/checker.py` states those rules;
this module runs them in the ordinary suite so they hold continuously rather
than only when someone remembers to run `make arxiv`.

The second half of the module checks the checker. A submission gate that
returns "ready" no matter what it is handed is worse than no gate, because it
converts an unchecked paper into an apparently checked one, so each rule is
exercised against a paper that violates it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import check_paper_submission as checker  # noqa: E402


def test_the_committed_paper_is_submission_ready() -> None:
    problems = checker.check()
    assert problems == [], "paper/paper.tex is not ready to submit:\n  - " + "\n  - ".join(problems)


# ---------------------------------------------------------------------------
# The checker's own rules, each against a paper that breaks exactly one of them.
# ---------------------------------------------------------------------------
GOOD_PAPER = r"""
\documentclass{article}
\usepackage{graphicx}
\graphicspath{{../results/}{./}}
\title{A title}
\author{Real Name\thanks{Independent work.}}
\date{31 August 2026}
\begin{document}
\includegraphics[width=\textwidth]{extrapolation.png}
\includegraphics[width=\textwidth]{size_extrapolation.png}
\end{document}
"""

GOOD_MAKEFILE = "arxiv: paper/paper.tex\n\t@cp results/extrapolation.png results/size_extrapolation.png build/arxiv/\n"


def _write(tmp_path: Path, paper: str, makefile: str = GOOD_MAKEFILE) -> tuple[Path, Path]:
    paper_path = tmp_path / "paper.tex"
    paper_path.write_text(paper)
    makefile_path = tmp_path / "Makefile"
    makefile_path.write_text(makefile)
    return paper_path, makefile_path


def test_the_control_paper_passes(tmp_path: Path) -> None:
    """Without this, every case below could be passing for the wrong reason."""
    assert checker.check(*_write(tmp_path, GOOD_PAPER)) == []


@pytest.mark.parametrize(
    ("label", "paper", "expected"),
    [
        ("build-time date", GOOD_PAPER.replace(r"\date{31 August 2026}", r"\date{\today}"), "fixed date"),
        ("no graphicspath", GOOD_PAPER.replace(r"\graphicspath{{../results/}{./}}", ""), "graphicspath"),
        ("parent-directory figure path",
         GOOD_PAPER.replace("{extrapolation.png}", "{../results/extrapolation.png}"), "flat directory"),
        ("figure with no file",
         GOOD_PAPER.replace("{size_extrapolation.png}", "{no_such_figure.png}"), "no file under results/"),
        ("placeholder author", GOOD_PAPER.replace("Real Name", "Your Name"), "placeholder"),
        ("no author line", GOOD_PAPER.replace(r"\author{Real Name\thanks{Independent work.}}", ""), "author"),
        ("no figures at all", "\n".join(
            line for line in GOOD_PAPER.splitlines() if "includegraphics" not in line), "figure check"),
    ],
)
def test_each_rule_rejects_a_paper_that_breaks_it(
    tmp_path: Path, label: str, paper: str, expected: str
) -> None:
    problems = checker.check(*_write(tmp_path, paper))
    assert any(expected in p for p in problems), f"{label}: nothing matched {expected!r} in {problems}"


def test_a_figure_missing_from_the_arxiv_bundle_is_caught(tmp_path: Path) -> None:
    """The case that builds locally and fails only after upload.

    A figure added to the paper but not to the `make arxiv` copy list resolves
    through the repository half of \\graphicspath here, and has nothing to
    resolve to in the flat directory arXiv builds in.
    """
    makefile = "arxiv: paper/paper.tex\n\t@cp results/extrapolation.png build/arxiv/\n"
    problems = checker.check(*_write(tmp_path, GOOD_PAPER, makefile))
    assert any("not copied by" in p and "size_extrapolation.png" in p for p in problems), problems


def test_a_rule_named_only_in_a_latex_comment_is_not_a_violation(tmp_path: Path) -> None:
    """Comments explain these rules, so the checker must not read its own docs.

    The real paper's preamble discusses `\\today` by name in a comment block.
    Reading that as a violation would make the rule unstatable in the file it
    governs.
    """
    commented = GOOD_PAPER.replace(
        r"\documentclass{article}",
        "% Deliberately a fixed date, not \\date{\\today}, because DOIs are permanent.\n"
        r"\documentclass{article}",
    )
    assert checker.check(*_write(tmp_path, commented)) == []


def test_an_escaped_percent_does_not_truncate_the_source(tmp_path: Path) -> None:
    """`\\%` is a percent sign, not a comment; stripping it would hide later rules."""
    with_pct = GOOD_PAPER.replace(r"\title{A title}", r"\title{A 41\% margin}")
    assert checker.check(*_write(tmp_path, with_pct)) == []


# --- the opt-in PDF freshness gate -----------------------------------------
#
# `paper/paper.pdf` is committed, the README links readers to it, and a DOI
# would archive it permanently, so a PDF that predates the current paper.tex is
# a permanent record of the wrong paper. Rebuilding needs a LaTeX toolchain, so
# this is a release gate (`make paper-fresh`) rather than part of `make check`.


def test_the_freshness_check_is_not_in_the_default_rule_set() -> None:
    """`make check` has to stay green on a machine with no pdflatex."""
    problems = checker.check()
    assert not any("stale" in problem for problem in problems)


def test_stale_pdf_sections_reports_a_section_the_pdf_lacks(tmp_path: Path) -> None:
    paper = tmp_path / "paper.tex"
    paper.write_text(
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "\\section{A section no PDF has ever contained}\n"
        "\\end{document}\n"
    )
    missing = checker.stale_pdf_sections(
        paper=paper, pdf=ROOT / "paper" / "paper.pdf"
    )
    assert missing == ["A section no PDF has ever contained"]


def test_stale_pdf_sections_is_quiet_when_there_is_no_pdf(tmp_path: Path) -> None:
    """A missing PDF is a different problem, and must not read as staleness."""
    paper = tmp_path / "paper.tex"
    paper.write_text("\\section{Anything}\n")
    assert checker.stale_pdf_sections(paper=paper, pdf=tmp_path / "absent.pdf") == []


def test_typeset_substitutions_do_not_read_as_stale_sections() -> None:
    """The committed PDF must read as fresh against its own source.

    Two false positives were found this way and neither was hypothetical. LaTeX
    sets `fi` as a single ligature glyph, so "deficient" in the source did not
    match "deﬁcient" in the PDF; and it sets `'` as a right single quote, so
    "reversal's" did not match "reversal's". Both reported a section stale that
    was present all along, which is worse than having no check, because it
    teaches the reader to ignore the one gate standing between them and a
    permanently archived wrong paper.
    """
    pdf = ROOT / "paper" / "paper.pdf"
    if not pdf.exists():
        pytest.skip("no committed PDF to read back")

    missing = checker.stale_pdf_sections(paper=ROOT / "paper" / "paper.tex", pdf=pdf)
    assert missing == [], (
        f"the committed PDF reads as missing {missing}. Either it is genuinely "
        "stale and needs rebuilding, or another typeset substitution needs adding "
        "to TYPESET_SUBSTITUTIONS."
    )


def test_the_freshness_check_still_catches_a_genuinely_absent_section(tmp_path: Path) -> None:
    """Normalising away false positives must not normalise away the signal."""
    paper = tmp_path / "paper.tex"
    paper.write_text("\\section{Ligatures, quotes and a section never written}\n")
    missing = checker.stale_pdf_sections(paper=paper, pdf=ROOT / "paper" / "paper.pdf")
    assert missing == ["Ligatures, quotes and a section never written"]
