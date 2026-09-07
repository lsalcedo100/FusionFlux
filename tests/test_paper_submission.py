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

import re
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
        (
            "parent-directory figure path",
            GOOD_PAPER.replace("{extrapolation.png}", "{../results/extrapolation.png}"),
            "flat directory",
        ),
        (
            "figure with no file",
            GOOD_PAPER.replace("{size_extrapolation.png}", "{no_such_figure.png}"),
            "no file under results/",
        ),
        ("placeholder author", GOOD_PAPER.replace("Real Name", "Your Name"), "placeholder"),
        ("no author line", GOOD_PAPER.replace(r"\author{Real Name\thanks{Independent work.}}", ""), "author"),
        (
            "no figures at all",
            "\n".join(line for line in GOOD_PAPER.splitlines() if "includegraphics" not in line),
            "figure check",
        ),
    ],
)
def test_each_rule_rejects_a_paper_that_breaks_it(tmp_path: Path, label: str, paper: str, expected: str) -> None:
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
        "\\documentclass{article}\n\\begin{document}\n\\section{A section no PDF has ever contained}\n\\end{document}\n"
    )
    missing = checker.stale_pdf_sections(paper=paper, pdf=ROOT / "paper" / "paper.pdf")
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


# --- the abstract fits the journal's limit ---------------------------------
#
# IOP asks for abstracts of no more than 300 words and warns that a manuscript
# may be returned for rewriting above it. That is a hard submission gate, and a
# word count written into prose goes stale the first time the abstract is
# edited, so it is computed here instead. The ceiling is 290 rather than 300 so
# that no disagreement between counting methods, and no later one-sentence
# addition, can push the submitted version over the real limit unnoticed.

ABSTRACT_WORD_CEILING = 290


def abstract_words(paper: Path | None = None) -> list[str]:
    """The abstract as a reader counts it, with LaTeX markup removed.

    Maths is collapsed to a single token rather than dropped, because a reader
    counts "rho = +0.85" as something; commands and braces disappear because
    nobody counts a backslash.
    """
    source = (paper or ROOT / "paper" / "paper.tex").read_text()
    body = source.split(r"\begin{abstract}")[1].split(r"\end{abstract}")[0]
    body = re.sub(r"\\cite\{[^}]*\}", "", body)
    body = re.sub(r"\$[^$]*\$", "X", body)
    body = re.sub(r"\\[a-zA-Z]+", "", body)
    body = re.sub(r"[{}~\\]", " ", body)
    return [w for w in body.split() if re.search(r"[A-Za-z0-9]", w)]


def test_abstract_is_within_the_journal_word_limit() -> None:
    words = abstract_words()
    assert len(words) <= ABSTRACT_WORD_CEILING, (
        f"the abstract is {len(words)} words, above the {ABSTRACT_WORD_CEILING} "
        "this repository holds itself to and close to IOP's hard limit of 300. "
        "Shorten it rather than raising the ceiling."
    )


def test_the_abstract_word_count_is_measuring_something() -> None:
    """A counter that returned nothing would pass the limit check silently."""
    assert len(abstract_words()) > 150


# --- every float is pointed at from the prose --------------------------------
#
# A figure or table no sentence references is a defect LaTeX cannot see: it
# compiles, and a float with no anchor drifts to wherever the placement
# algorithm leaves it. Five accumulated here, four of them when material moved
# between the two documents and took the referring sentence with it, including a
# full-width figure the main text no longer mentioned at all.
#
# References are pooled across both files rather than checked per file, because
# the supplement's floats are sometimes introduced from the main text.

DOCUMENTS = ("paper/paper.tex", "paper/supplementary.tex")


def _all_references() -> set[str]:
    pooled = "".join((ROOT / name).read_text() for name in DOCUMENTS)
    return set(re.findall(r"\\(?:ref|eqref|autoref)\{([^}]+)\}", pooled))


@pytest.mark.parametrize("document", DOCUMENTS)
def test_no_float_is_orphaned(document: str) -> None:
    source = (ROOT / document).read_text()
    floats = re.findall(r"\\label\{((?:fig|tab):[^}]+)\}", source)
    orphaned = sorted(set(floats) - _all_references())
    assert not orphaned, (
        f"{document} defines {orphaned} but no sentence in either document "
        "references them. Add the reference where the float is described, or "
        "remove the float."
    )


@pytest.mark.parametrize("document", DOCUMENTS)
def test_the_orphan_check_is_looking_at_something(document: str) -> None:
    source = (ROOT / document).read_text()
    assert len(re.findall(r"\\label\{(?:fig|tab):[^}]+\}", source)) >= 8


def test_no_reference_points_at_a_label_that_does_not_exist() -> None:
    """Renders as ?? in the PDF, which is easy to miss in a 31-page proof."""
    labels = set()
    for name in DOCUMENTS:
        labels |= set(re.findall(r"\\label\{([^}]+)\}", (ROOT / name).read_text()))
    # Each file is compiled alone, so a reference has to resolve within its own
    # document; pooling here would hide exactly the break that moving a section
    # between the two causes.
    for name in DOCUMENTS:
        source = (ROOT / name).read_text()
        own = set(re.findall(r"\\label\{([^}]+)\}", source))
        used = set(re.findall(r"\\(?:ref|eqref|autoref)\{([^}]+)\}", source))
        assert not (used - own), f"{name} references {sorted(used - own)}, defined elsewhere"


# --- the hardcoded pointers into the supplement ------------------------------
#
# The two documents compile separately, so the main text cannot \ref a section
# of the supplement and points at it by number instead. Reordering the
# supplement silently redirects every one of those, and neither LaTeX nor the
# orphan check above can see it: the reference still resolves, to the wrong
# section. That is how "Sec.~S6" came to name the GP ladder in a sentence about
# the model specification.
#
# So the mapping is written down. Moving a supplement section fails this with
# the section it now points at, and the fix is to correct the number here and in
# paper.tex together.

EXPECTED_POINTERS = {
    1: "Repairing the intervals",
    2: "Robustness on rows the standard analysis set excludes",
    3: "Locked predictions at three device operating points",
    4: "The same audit on a scaling law from another science",
    5: "The reversal's precondition, measured directly",
    6: "The three-kernel Gaussian-process ladder",
    7: "Full model, kernel and split specification",
    8: "Per-label scores and the eligibility-threshold sweep",
}


def _supplement_sections() -> list[str]:
    source = (ROOT / "paper" / "supplementary.tex").read_text()
    titles = re.findall(r"\\section\{((?:[^{}]|\{[^{}]*\})*)\}", source)
    return [re.sub(r"\s+", " ", title).strip() for title in titles]


@pytest.mark.parametrize("number,expected", sorted(EXPECTED_POINTERS.items()))
def test_each_hardcoded_pointer_names_the_right_section(number: int, expected: str) -> None:
    sections = _supplement_sections()
    assert number <= len(sections), f"paper.tex points at S{number}; there are {len(sections)}"
    assert sections[number - 1].startswith(expected), (
        f"S{number} is now {sections[number - 1]!r}, not {expected!r}. "
        "A supplement section moved: fix the number in paper.tex and here."
    )


def test_every_pointer_in_the_paper_is_covered_here() -> None:
    """A new Sec.~SN in the main text has to be added to the mapping above."""
    source = (ROOT / "paper" / "paper.tex").read_text()
    # "Secs.~S4 and~S5" names two sections in one phrase, and an earlier pattern
    # saw neither: it required "Sec." or "Section", so the plural and the
    # second number both slipped past and went unchecked.
    used = {int(n) for n in re.findall(r"(?:Sec(?:tion|s?\.)~|and~)S(\d+)", source)}
    assert used == set(EXPECTED_POINTERS), (
        f"paper.tex points at S{sorted(used)}; EXPECTED_POINTERS covers S{sorted(EXPECTED_POINTERS)}"
    )


# --- both PDFs are gated, not just the main one ------------------------------
#
# Both documents are committed and both are uploaded, so both can go stale, but
# only paper.pdf was ever checked. tests/test_reported_numbers.py would have
# caught a number drifting in the supplement; a prose edit that moved no number,
# such as renaming a section, would have reached ScholarOne unnoticed.


def test_the_supplement_is_checked_for_freshness() -> None:
    source = ROOT / "paper" / "supplementary.tex"
    pdf = ROOT / "paper" / "supplementary.pdf"
    if not pdf.exists():
        pytest.skip("paper/supplementary.pdf not built")
    assert checker.stale_pdf_sections(source, pdf) == []


def test_a_renamed_supplement_section_is_caught(tmp_path: Path) -> None:
    """The exact edit the missing gate would have let through."""
    pdf = ROOT / "paper" / "supplementary.pdf"
    if not pdf.exists():
        pytest.skip("paper/supplementary.pdf not built")
    renamed = tmp_path / "supplementary.tex"
    renamed.write_text(
        (ROOT / "paper" / "supplementary.tex")
        .read_text()
        .replace(
            "\\section{The three-kernel Gaussian-process ladder}",
            "\\section{A title the committed supplement cannot contain}",
        )
    )
    assert checker.stale_pdf_sections(renamed, pdf) == [
        "A title the committed supplement cannot contain"
    ]


def test_the_checker_knows_where_the_supplement_lives() -> None:
    assert checker.SUPPLEMENT.name == "supplementary.tex"
    assert checker.SUPPLEMENT_PDF.name == "supplementary.pdf"
