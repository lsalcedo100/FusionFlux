"""The anonymised submission is really anonymous, and still declares what it must.

`tools/make_submission.py` verifies its own output by reading the built PDFs
back, but that check only runs when someone builds the bundle. These tests run
the text transform on the real `paper.tex` on every commit, which is where a
rename or a reworded section would break it.

The second half matters as much as the first. The generative-AI declaration
lives as a `\\paragraph` inside Acknowledgments, so the obvious way to strip the
credits also strips a disclosure IOP requires and that names nobody. That
happened once here; these tests are what would have caught it.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "paper.tex"
SUPPLEMENT = ROOT / "paper" / "supplementary.tex"

_spec = importlib.util.spec_from_file_location("make_submission", ROOT / "tools" / "make_submission.py")
assert _spec is not None and _spec.loader is not None
make_submission = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_submission)


@pytest.fixture(scope="module")
def anonymous_paper() -> str:
    return make_submission.anonymise(PAPER.read_text())


@pytest.fixture(scope="module")
def anonymous_supplement() -> str:
    return make_submission.anonymise_supplement(SUPPLEMENT.read_text())


def _visible(latex: str) -> str:
    """Drop the arguments a reader never sees.

    `IDENTIFYING` is a list of strings checked against *rendered* PDF text, and
    applying it to source raises a false alarm on `\\cite{zenodo}`: a key, which
    renders as a bracketed number. Cross-reference arguments are emptied here so
    the source check asks the same question the PDF check does.
    """
    return re.sub(r"\\(cite|bibitem|ref|label|eqref)\{[^}]*\}", r"\\\1{}", latex)


@pytest.mark.parametrize("token", make_submission.IDENTIFYING)
def test_no_identifying_string_survives_in_the_paper(anonymous_paper: str, token: str) -> None:
    assert token.lower() not in _visible(anonymous_paper).lower()


@pytest.mark.parametrize("token", ("Salcedo", "liamsalcedo", "0009-0001-5039-8147"))
def test_no_identifying_string_survives_in_the_supplement(anonymous_supplement: str, token: str) -> None:
    assert token.lower() not in anonymous_supplement.lower()


def test_the_identified_source_does_contain_those_strings() -> None:
    """Otherwise the test above passes because the strings were never there."""
    source = PAPER.read_text()
    for token in ("Salcedo", "liamsalcedo", "0009-0001-5039-8147", "zenodo"):
        assert token.lower() in source.lower()


@pytest.mark.parametrize(
    "declaration",
    ("Use of generative AI", "Funding", "Competing interests", "Data availability"),
)
def test_required_declarations_survive(anonymous_paper: str, declaration: str) -> None:
    assert f"\\section*{{{declaration}}}" in anonymous_paper


def test_the_ai_declaration_keeps_its_body(anonymous_paper: str) -> None:
    """Named systems and versions, which is the part that satisfies the policy."""
    assert "claude-opus-5" in anonymous_paper
    assert "GPT-5.6" in anonymous_paper


@pytest.mark.parametrize("credit", make_submission.ANONYMISE_SECTIONS)
def test_credit_sections_are_gone(anonymous_paper: str, credit: str) -> None:
    assert f"\\section*{{{credit}}}" not in anonymous_paper


def test_the_paper_still_compiles_in_shape(anonymous_paper: str) -> None:
    """A strip that ate a brace produces a file that fails only at build time."""
    assert anonymous_paper.count("\\begin{document}") == 1
    assert anonymous_paper.count("\\end{document}") == 1
    assert anonymous_paper.count("\\begin{thebibliography}") == 1
    assert anonymous_paper.count("\\end{thebibliography}") == 1
    assert anonymous_paper.count("{") == anonymous_paper.count("}")


def test_the_reader_is_told_what_was_withheld(anonymous_paper: str) -> None:
    """A blank where a repository URL belongs reads as no code, not as anonymity."""
    assert anonymous_paper.count("withheld for anonymous review") >= 2


# --- the ScholarOne field sheet ---------------------------------------------
#
# The sheet is generated from paper.tex so that a retyped abstract cannot differ
# from the one in the PDF beside it. That only helps if the extraction is
# faithful, and the first version was not: it dropped every mathematical symbol
# in the abstract, turning "rho = +0.85" into "=+0.85" and "1.82x" into "1.82",
# because a LaTeX command spelled into a regex reads as an escape.


@pytest.fixture(scope="module")
def sheet() -> str:
    return make_submission.scholarone_metadata(PAPER.read_text(), 32, 15)


def test_every_placeholder_is_filled(sheet: str) -> None:
    assert "{{" not in sheet


def test_the_sheet_carries_no_latex(sheet: str) -> None:
    """A backslash or a brace in a form field is markup the field will not render."""
    for residue in ("\\", "{", "}", "$", "~"):
        assert residue not in sheet, f"{residue!r} survived into the field sheet"


def test_the_abstract_keeps_its_symbols(sheet: str) -> None:
    assert "ρ=+0.85" in sheet, "the correlation lost its rho"
    assert "1.82×" in sheet, "the size jump lost its multiplication sign"
    assert "8.3×" in sheet, "the ITER disagreement lost its multiplication sign"


def test_the_abstract_matches_the_paper(sheet: str) -> None:
    """Same text, same length, as the count the journal limit is enforced on."""
    from test_paper_submission import abstract_words

    counted = re.search(r"ABSTRACT \((\d+) words\)", sheet)
    assert counted is not None
    assert int(counted.group(1)) == len(abstract_words())
    assert "Confinement scaling laws set the size a next-step tokamak" in sheet


def test_the_abstract_is_one_line(sheet: str) -> None:
    """ScholarOne's abstract box is a textarea: a wrapped paste keeps its breaks."""
    body = sheet.split("words)\n")[1].split("\n\nKEYWORDS")[0]
    assert body.count("\n") == 0


def test_the_author_line_is_readable(sheet: str) -> None:
    """Deleting \\thanks{ once ran the name into the affiliation."""
    assert "Liam Salcedo. Independent researcher" in sheet
    assert "ORCID: 0009-0001-5039-8147" in sheet
    # The link text, not the URL spelled out beside it.
    assert "https://orcid.org" not in sheet


def test_the_bundle_is_not_written_under_build() -> None:
    """tests/test_packaging.py removes build/, so a bundle there is temporary.

    That is not theoretical: an assembled upload disappeared partway through a
    session because a full test run had swept the directory it was sitting in.
    """
    assert make_submission.OUT.name == "submission"
    assert "build" not in make_submission.OUT.relative_to(ROOT).parts


def test_the_bundle_directory_is_ignored_by_git() -> None:
    """It is a build product, and one of its PDFs is 270 KB."""
    ignored = (ROOT / ".gitignore").read_text().split("\n")
    assert f"{make_submission.OUT.name}/" in ignored


# --- the check that the whole anonymisation rests on ------------------------
#
# `verify_anonymous` reads the rendered text of the built PDFs back against the
# identifying strings, and it is the only check that sees what a referee sees.
# Nothing tested that it actually catches a leak, which is the one failure that
# would matter: a strip that silently stopped working would produce PDFs this
# function waves through.

IDENTIFIED = ROOT / "paper" / "paper.pdf"
ANONYMOUS = ROOT / "submission" / "anonymous" / "manuscript.pdf"


def test_it_rejects_the_identified_manuscript() -> None:
    if not IDENTIFIED.exists():
        pytest.skip("paper/paper.pdf not built")
    with pytest.raises(SystemExit) as refused:
        make_submission.verify_anonymous([IDENTIFIED])
    assert "not anonymous" in str(refused.value)
    assert "Salcedo" in str(refused.value)


def test_it_passes_the_anonymous_one() -> None:
    if not ANONYMOUS.exists():
        pytest.skip("no assembled bundle; run `make submission`")
    make_submission.verify_anonymous([ANONYMOUS])


def test_it_names_every_string_it_found(tmp_path: Path) -> None:
    """One report listing all of them, not the first and a rerun for the rest."""
    if not IDENTIFIED.exists():
        pytest.skip("paper/paper.pdf not built")
    with pytest.raises(SystemExit) as refused:
        make_submission.verify_anonymous([IDENTIFIED])
    reported = {token for token in make_submission.IDENTIFYING if repr(token) in str(refused.value)}
    assert len(reported) > 1


def test_the_page_count_is_read_from_the_pdf() -> None:
    if not IDENTIFIED.exists():
        pytest.skip("paper/paper.pdf not built")
    assert make_submission._page_count(IDENTIFIED) > 20


def test_a_template_missing_a_placeholder_is_refused(tmp_path: Path) -> None:
    """Half a field sheet would go to ScholarOne looking complete."""
    stand_in = tmp_path / "template.txt"
    stand_in.write_text("TITLE\n{{TITLE}}\n")
    original = make_submission.TEMPLATE
    make_submission.TEMPLATE = stand_in
    try:
        with pytest.raises(SystemExit) as refused:
            make_submission.scholarone_metadata(PAPER.read_text(), 31, 16)
        assert "ABSTRACT" in str(refused.value)
    finally:
        make_submission.TEMPLATE = original


def test_an_unfilled_placeholder_is_refused(tmp_path: Path) -> None:
    stand_in = tmp_path / "template.txt"
    stand_in.write_text(
        "\n".join(
            f"{{{{{key}}}}}"
            for key in (
                "TITLE",
                "ABSTRACT",
                "ABSTRACT_WORDS",
                "KEYWORDS",
                "AUTHOR",
                "MANUSCRIPT_PAGES",
                "SUPPLEMENT_PAGES",
            )
        )
        + "\n{{INVENTED}}\n"
    )
    original = make_submission.TEMPLATE
    make_submission.TEMPLATE = stand_in
    try:
        with pytest.raises(SystemExit) as refused:
            make_submission.scholarone_metadata(PAPER.read_text(), 31, 16)
        assert "unfilled placeholder" in str(refused.value)
    finally:
        make_submission.TEMPLATE = original
