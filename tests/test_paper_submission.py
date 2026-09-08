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


# --- the archive the paper cites actually contains the commit it pins --------
#
# This guard exists because the two checks either side of it both passed while
# the cited Zenodo release sat seventeen commits behind the pinned commit and
# did not contain it, so the archived code was not the code that produced the
# printed numbers. It had no tests of its own, which for a check written to
# catch a defect that already happened is the wrong way round.
#
# These build a throwaway repository rather than asserting against this one,
# whose tags move.


def _git(repo: Path, *arguments: str) -> str:
    import subprocess

    done = subprocess.run(
        ("git", *arguments), cwd=repo, capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Two commits and a tag on the first, which is the shape of the defect."""
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "a.txt").write_text("one\n")
    _git(tmp_path, "add", "a.txt")
    _git(tmp_path, "commit", "-qm", "first")
    _git(tmp_path, "tag", "v1.0.0")
    (tmp_path / "a.txt").write_text("two\n")
    _git(tmp_path, "add", "a.txt")
    _git(tmp_path, "commit", "-qm", "second")
    return tmp_path


def _paper_citing(repo: Path, commit: str, version: str) -> Path:
    paper = repo / "paper.tex"
    paper.write_text(
        f"(v{version}; the DOI for all versions is X)\n\\texttt{{{commit}}}\n"
    )
    return paper


def test_a_release_that_predates_the_pin_is_reported(repo: Path) -> None:
    head = _git(repo, "rev-parse", "HEAD")
    problems = checker.stale_archive(_paper_citing(repo, head, "1.0.0"), repo)
    assert len(problems) == 1
    assert "does not contain it" in problems[0]
    assert head[:12] in problems[0]


def test_a_release_containing_the_pin_is_accepted(repo: Path) -> None:
    tagged = _git(repo, "rev-parse", "v1.0.0^{commit}")
    assert checker.stale_archive(_paper_citing(repo, tagged, "1.0.0"), repo) == []


def test_a_version_with_no_tag_is_reported(repo: Path) -> None:
    """The state right after a version bump, before the tag is cut."""
    head = _git(repo, "rev-parse", "HEAD")
    problems = checker.stale_archive(_paper_citing(repo, head, "9.9.9"), repo)
    assert len(problems) == 1
    assert "not a tag" in problems[0]


def test_a_repository_with_no_tags_is_not_a_defect(tmp_path: Path) -> None:
    """A shallow clone or an export cannot answer, which is not the paper's fault."""
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "a.txt").write_text("one\n")
    _git(tmp_path, "add", "a.txt")
    _git(tmp_path, "commit", "-qm", "first")
    head = _git(tmp_path, "rev-parse", "HEAD")
    assert checker.stale_archive(_paper_citing(tmp_path, head, "1.0.0"), tmp_path) == []


def test_a_paper_pinning_nothing_is_left_to_the_other_check(tmp_path: Path) -> None:
    """stale_provenance owns the missing-pin message; two would be noise."""
    paper = tmp_path / "paper.tex"
    paper.write_text("(v1.0.0; the DOI for all versions is X)\n")
    assert checker.stale_archive(paper, tmp_path) == []


def test_a_paper_citing_no_version_is_not_checked(repo: Path) -> None:
    head = _git(repo, "rev-parse", "HEAD")
    paper = repo / "paper.tex"
    paper.write_text(f"\\texttt{{{head}}}\n")
    assert checker.stale_archive(paper, repo) == []


def test_a_comment_cannot_satisfy_the_check(repo: Path) -> None:
    """Comments are stripped first, so a pin written in one does not count."""
    head = _git(repo, "rev-parse", "HEAD")
    paper = repo / "paper.tex"
    paper.write_text(
        f"% (v1.0.0; the DOI for all versions is X)\n% \\texttt{{{head}}}\n"
    )
    assert checker.stale_archive(paper, repo) == []


# --- the pin still describes the artifacts ----------------------------------
#
# The companion guard, and the older one. It asks whether the pinned commit is
# still the last one that touched results/, which is the claim the paper makes
# about which tree produced its printed values. It went wrong once over three
# commits before it existed.


def _repo_with_results(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "a.json").write_text("{}\n")
    _git(tmp_path, "add", "results/a.json")
    _git(tmp_path, "commit", "-qm", "generate results")
    return tmp_path


def test_a_pin_matching_the_last_results_commit_passes(tmp_path: Path) -> None:
    root = _repo_with_results(tmp_path)
    head = _git(root, "rev-parse", "HEAD")
    assert checker.stale_provenance(_paper_citing(root, head, "1.0.0"), root) == []


def test_a_pin_left_behind_by_a_regeneration_is_reported(tmp_path: Path) -> None:
    root = _repo_with_results(tmp_path)
    stale = _git(root, "rev-parse", "HEAD")
    (root / "results" / "a.json").write_text('{"changed": true}\n')
    _git(root, "add", "results/a.json")
    _git(root, "commit", "-qm", "regenerate")
    problems = checker.stale_provenance(_paper_citing(root, stale, "1.0.0"), root)
    assert len(problems) == 1
    assert "was last changed" in problems[0]


def test_uncommitted_results_are_reported(tmp_path: Path) -> None:
    """A dirty artifact is at no commit at all, so no pin can describe it."""
    root = _repo_with_results(tmp_path)
    head = _git(root, "rev-parse", "HEAD")
    (root / "results" / "a.json").write_text('{"dirty": true}\n')
    problems = checker.stale_provenance(_paper_citing(root, head, "1.0.0"), root)
    assert any("uncommitted changes" in problem for problem in problems)


def test_a_paper_with_no_pin_is_reported(tmp_path: Path) -> None:
    paper = tmp_path / "paper.tex"
    paper.write_text("no commit hash anywhere in this file\n")
    problems = checker.stale_provenance(paper, tmp_path)
    assert len(problems) == 1
    assert "pins nothing" in problems[0]


def test_a_directory_git_cannot_answer_for_is_not_a_defect(tmp_path: Path) -> None:
    paper = _paper_citing(tmp_path, "0" * 40, "1.0.0")
    assert checker.stale_provenance(paper, tmp_path) == []


# --- the reference list is the last thing in each document -------------------
#
# The supplement printed its bibliography on p.14 with Section S8 after it on
# p.15, which is the most visible sloppiness signal a package can carry, and
# nothing here caught it. The orphan and label tests ask whether floats are
# referenced; neither asks where the bibliography sits.


@pytest.mark.parametrize("document", DOCUMENTS)
def test_the_bibliography_is_the_last_block(document: str) -> None:
    source = (ROOT / document).read_text()
    end = source.index(r"\end{thebibliography}")
    trailing = source[end:]
    assert r"\section" not in trailing, (
        f"{document} has a \\section after its bibliography. The reference list "
        "must be the last block before \\end{document}."
    )


@pytest.mark.parametrize("document", DOCUMENTS)
def test_each_document_has_exactly_one_bibliography(document: str) -> None:
    source = (ROOT / document).read_text()
    assert source.count(r"\begin{thebibliography}") == 1
    assert source.count(r"\end{thebibliography}") == 1


# --- the version DOI belongs to the version the paper cites ------------------
#
# `stale_archive` compares a tag against a commit and `stale_provenance` compares
# a commit against results/. Neither can see the state between tagging a release
# and publishing it, where the title page names v0.4.4 and links v0.4.3's DOI:
# both strings are well formed, they simply describe different releases. The
# ledger `tools/bump_release.py` writes is the only place the two are recorded
# together, so `mismatched_doi` reads that.


def _paper_with_doi(directory: Path, version: str, doi: str) -> Path:
    paper = directory / "paper.tex"
    paper.write_text(
        f"(v{version}; the DOI for all versions is\n"
        "\\href{https://doi.org/10.5281/zenodo.22215142}{10.5281/zenodo.22215142})\n"
        f"\\href{{https://doi.org/{doi}}}{{doi:{doi}}}\n"
    )
    return paper


def _with_ledger(directory: Path, versions: dict) -> Path:
    import json

    (directory / "docs").mkdir(parents=True, exist_ok=True)
    (directory / "docs" / "releases.json").write_text(json.dumps({"versions": versions}))
    return directory


def test_a_doi_matching_the_cited_release_passes(tmp_path: Path) -> None:
    _with_ledger(tmp_path, {"0.4.4": {"doi": "10.5281/zenodo.7"}})
    paper = _paper_with_doi(tmp_path, "0.4.4", "10.5281/zenodo.7")
    assert checker.mismatched_doi(paper, tmp_path) == []


def test_the_previous_releases_doi_on_a_bumped_version_is_reported(tmp_path: Path) -> None:
    """The exact half-done release this check exists for."""
    _with_ledger(
        tmp_path, {"0.4.3": {"doi": "10.5281/zenodo.6"}, "0.4.4": {"doi": None}}
    )
    paper = _paper_with_doi(tmp_path, "0.4.4", "10.5281/zenodo.6")
    problems = checker.mismatched_doi(paper, tmp_path)
    assert len(problems) == 1
    assert "no archive DOI recorded" in problems[0]
    assert "bump_release.py --doi" in problems[0]


def test_a_doi_belonging_to_another_release_is_reported(tmp_path: Path) -> None:
    _with_ledger(
        tmp_path, {"0.4.3": {"doi": "10.5281/zenodo.6"}, "0.4.4": {"doi": "10.5281/zenodo.7"}}
    )
    paper = _paper_with_doi(tmp_path, "0.4.4", "10.5281/zenodo.6")
    problems = checker.mismatched_doi(paper, tmp_path)
    assert len(problems) == 1
    assert "10.5281/zenodo.7" in problems[0]


def test_the_concept_doi_is_not_read_as_the_version_doi(tmp_path: Path) -> None:
    """It appears on the same line and never moves, so matching it would be wrong."""
    _with_ledger(tmp_path, {"0.4.4": {"doi": "10.5281/zenodo.7"}})
    paper = _paper_with_doi(tmp_path, "0.4.4", "10.5281/zenodo.7")
    assert "10.5281/zenodo.22215142" in paper.read_text()
    assert checker.mismatched_doi(paper, tmp_path) == []


def test_no_ledger_is_not_a_defect_in_the_paper(tmp_path: Path) -> None:
    """A source tarball carries no ledger, and an unanswerable question is not a fault."""
    paper = _paper_with_doi(tmp_path, "0.4.4", "10.5281/zenodo.6")
    assert checker.mismatched_doi(paper, tmp_path) == []


def test_an_unreadable_ledger_is_reported_rather_than_ignored(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "docs" / "releases.json").write_text("{ not json")
    paper = _paper_with_doi(tmp_path, "0.4.4", "10.5281/zenodo.6")
    problems = checker.mismatched_doi(paper, tmp_path)
    assert len(problems) == 1
    assert "unreadable" in problems[0]
