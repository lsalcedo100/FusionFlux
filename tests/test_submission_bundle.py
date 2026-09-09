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
    (
        "\\paragraph*{Use of generative AI.}",
        "\\paragraph*{Funding.}",
        "\\paragraph*{Competing interests.}",
        "\\section*{Data availability}",
    ),
)
def test_required_declarations_survive(anonymous_paper: str, declaration: str) -> None:
    """IOP puts the first three inside Acknowledgements, so that is where they sit.

    Anonymisation can therefore no longer drop that section wholesale, and what
    it removes instead is the credit prose that opens it and the contributions
    paragraph. The heading stays, carrying the declarations under it.
    """
    assert declaration in anonymous_paper


def test_the_acknowledgement_heading_survives_without_its_credits(anonymous_paper: str) -> None:
    """The section has to stay, because three required declarations live in it."""
    assert "\\section*{Acknowledgments}" in anonymous_paper
    assert "The author thanks them" not in anonymous_paper
    assert "reviewed or endorsed this analysis" not in anonymous_paper


def test_the_ai_declaration_keeps_its_body(anonymous_paper: str) -> None:
    """Named systems and versions, which is the part that satisfies the policy."""
    assert "claude-opus-5" in anonymous_paper
    assert "GPT-5.6" in anonymous_paper


@pytest.mark.parametrize("credit", make_submission.ANONYMISE_PARAGRAPHS)
def test_credit_paragraphs_are_gone(anonymous_paper: str, credit: str) -> None:
    """The one declaration of the four that names anyone."""
    assert f"\\paragraph*{{{credit}}}" not in anonymous_paper
    assert "Sole author, in CRediT" not in anonymous_paper


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
    """The flattening must render maths, not drop it and not leave the command.

    Written against the symbol classes rather than against three particular
    numbers: the numbers are abstract prose and get edited, and a test that
    names them fails when a sentence is rewritten rather than when the
    flattening breaks. Editing "8.3x" out of the abstract is what retired the
    previous version of this.
    """
    abstract = sheet.split("ABSTRACT")[1].split("KEYWORDS")[0]
    assert "rho" in abstract, "the correlations lost their rho"
    assert "1.82x" in abstract, "a multiplication sign was dropped"
    for command in ("\\rho", "\\times", "\\textbf", "$"):
        assert command not in abstract, f"{command} survived into the plain-text abstract"


def test_the_pasted_sheet_is_ascii(sheet: str) -> None:
    """ScholarOne's fields are not Unicode-safe, and the abstract is pasted whole.

    A rho or a multiplication sign that arrives as a replacement character turns
    the first thing an editor reads into "at ?=+0.85". The whole sheet is
    checked, not only the abstract, because the same box takes the title and
    keywords.
    """
    offending = sorted({character for character in sheet if ord(character) > 127})
    assert not offending, f"non-ASCII characters in the pasted sheet: {offending}"


def test_the_abstract_matches_the_paper(sheet: str) -> None:
    """Same text, same length, as the count the journal limit is enforced on."""
    from test_paper_submission import abstract_words

    counted = re.search(r"ABSTRACT \((\d+) words\)", sheet)
    assert counted is not None
    assert int(counted.group(1)) == len(abstract_words())
    # The opening run of words comes from paper.tex rather than being typed
    # here. A literal sentence in this test goes stale the moment the abstract
    # is edited, which is exactly what happened when it was trimmed.
    opening = " ".join(abstract_words()[:8])
    assert opening in re.sub(r"\s+", " ", sheet), (
        f"the sheet does not open with the paper's abstract ({opening!r}), so the two "
        "have drifted apart."
    )


def test_the_abstract_is_one_line(sheet: str) -> None:
    """ScholarOne's abstract box is a textarea: a wrapped paste keeps its breaks."""
    body = sheet.split("words)\n")[1].split("\n\nKEYWORDS")[0]
    assert body.count("\n") == 0


def test_the_author_line_is_readable(sheet: str) -> None:
    """Deleting \\thanks{ once ran the name into the affiliation.

    Read with runs of whitespace collapsed, because the sheet is hard-wrapped at
    78 columns and where the wrap falls is not what this is checking. It used to
    be read raw, and shortening the affiliation moved the wrap between "ORCID:"
    and the identifier, which failed a test about the name running into the
    affiliation.
    """
    flat = re.sub(r"\s+", " ", sheet)
    assert "Liam Salcedo. Independent researcher" in flat
    assert "ORCID: 0009-0001-5039-8147" in flat
    # The link text, not the URL spelled out beside it.
    assert "https://orcid.org" not in flat


def test_the_commit_hash_does_not_survive_anonymisation(anonymous_paper: str) -> None:
    """GitHub indexes commits, so a 40-character pin is a search away from the author."""
    assert not make_submission.COMMIT_HASH.search(anonymous_paper)
    assert "[commit hash withheld for anonymous review]" in anonymous_paper


def test_the_identified_paper_does_carry_a_commit_hash() -> None:
    """Otherwise the check above passes because there was nothing to strip."""
    assert make_submission.COMMIT_HASH.search(PAPER.read_text())


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


# The same function also refuses a build whose citations did not survive the
# rewrite. That guard exists because a shipped anonymous PDF once carried seven
# literal [?] marks: the bibitem substitution ran to \end{thebibliography} and
# swallowed the five entries after it, and verify_anonymous waved it through
# because it only ever looked for names.


@pytest.mark.parametrize(
    "rendered",
    [
        "Kardaun [?] constructed intervals",
        "the jackknife+ and CV+ family [ ?], would be",
        "run on a machine [ ?, ?]. What is new",
        "see Sec. ?? of the main text",
    ],
)
def test_an_unresolved_reference_is_refused(rendered: str) -> None:
    assert make_submission.unresolved_references(Path("m.pdf"), rendered)


@pytest.mark.parametrize(
    "rendered",
    [
        "Hall et al. [12] have raised",
        "as reported in [12, 15] and [3]",
        "Flexibility, or long-range saturation? That is the question.",
    ],
)
def test_a_resolved_reference_is_not(rendered: str) -> None:
    """A numeric citation and an ordinary question mark must not trip it."""
    assert not make_submission.unresolved_references(Path("m.pdf"), rendered)


def test_the_built_manuscript_resolves_every_reference() -> None:
    if not IDENTIFIED.exists():
        pytest.skip("paper/paper.pdf not built")
    rendered = make_submission._rendered_text(IDENTIFIED)
    assert not make_submission.unresolved_references(IDENTIFIED, rendered)


def test_the_page_count_is_read_from_the_pdf() -> None:
    if not IDENTIFIED.exists():
        pytest.skip("paper/paper.pdf not built")
    assert make_submission._page_count(IDENTIFIED) > 20


def test_a_template_missing_a_placeholder_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Half a field sheet would go to ScholarOne looking complete."""
    stand_in = tmp_path / "template.txt"
    stand_in.write_text("TITLE\n{{TITLE}}\n")
    monkeypatch.setattr(make_submission, "TEMPLATE", stand_in)
    with pytest.raises(SystemExit) as refused:
        make_submission.scholarone_metadata(PAPER.read_text(), 31, 16)
    assert "ABSTRACT" in str(refused.value)


def test_an_unfilled_placeholder_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
                "REF_HDB5",
                "REF_HALL",
                "REF_HALL26",
                "SEC_KARDAUN",
            )
        )
        + "\n{{INVENTED}}\n"
    )
    monkeypatch.setattr(make_submission, "TEMPLATE", stand_in)
    with pytest.raises(SystemExit) as refused:
        make_submission.scholarone_metadata(PAPER.read_text(), 31, 16)
    assert "unfilled placeholder" in str(refused.value)


def test_the_two_documents_agree_on_the_affiliation() -> None:
    """One submission, two files, and they had two different affiliations.

    The town was dropped from the manuscript and left in the supplement, which
    an editor reading both sees before any referee does. Neither file is
    generated from the other, so nothing else would have caught it.
    """
    import re

    def affiliation(path: Path) -> str:
        block = re.search(r"\\author\{.*?\\thanks\{(.*?)\.\s*\n", path.read_text(), re.DOTALL)
        assert block is not None, f"no author block in {path.name}"
        return block.group(1).strip()

    paper = ROOT / "paper" / "paper.tex"
    supplement = ROOT / "paper" / "supplementary.tex"
    assert affiliation(paper) == affiliation(supplement)
