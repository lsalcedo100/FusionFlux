"""Assemble the ScholarOne upload from the built paper, in both variants.

Nuclear Fusion takes the manuscript as a PDF plus its sources, and offers
single- or double-anonymous review. Which one the author picks is not known
until upload, so this writes both: `build/submission/` carries the identified
manuscript, and `build/submission/anonymous/` the same paper with the author
block, the repository and archive links, and the acknowledgments removed.

The anonymous variant is the reason this is a script rather than a sequence of
`cp` commands. Stripping identity from LaTeX is easy to do incompletely: the
first attempt here matched one spelling of the Zenodo link and left the other
one on the title page, in a PDF that looked anonymous. So the strip is followed
by a read-back of the *rendered text* of both PDFs against every identifying
string, which is the only check that sees what a referee sees.

Run it after `make paper-fresh` passes; it refuses to run on a stale PDF, since
a bundle built from one source and one older PDF is the worst of both.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = ROOT / "build" / "submission"

# Every string that identifies the author, checked against the rendered text of
# the anonymous PDFs. Kept as literals rather than derived from the source, so
# that deleting a name from paper.tex cannot also delete the test for it.
IDENTIFYING = (
    "Salcedo",
    "liamsalcedo",
    "lsalcedo100",
    "Montclair",
    "0009-0001-5039-8147",
    "orcid",
    "10.5281/zenodo.22562235",  # version DOI
    "10.5281/zenodo.22215142",  # concept DOI, a different \href spelling
    "zenodo",
    "github.com/lsalcedo100",
)

# Sections that exist to credit people and therefore cannot survive anonymisation.
# `Funding`, `Competing interests` and `Data availability` stay: they carry no
# identity and IOP wants them in the reviewed manuscript.
ANONYMISE_SECTIONS = ("Acknowledgments", "Author contributions")


def _strip_section(text: str, title: str) -> str:
    """Drop `\\section*{title}` and its body, up to the next section or bibliography."""
    pattern = re.compile(
        r"\\section\*\{" + re.escape(title) + r"\}.*?(?=\\section\*?\{|\\begin\{thebibliography\})",
        re.DOTALL,
    )
    stripped, n = pattern.subn("", text)
    if n != 1:
        raise SystemExit(f"expected exactly one '{title}' section, found {n}")
    return stripped


def anonymise(text: str) -> str:
    """Remove the author block, the availability links and the credit sections."""
    # The author block runs from \author to the line before \date, and the
    # \thanks footnote inside it carries the affiliation and the email.
    text, n = re.subn(r"\\author\{.*?\n\\date\{", "\\\\author{}\n\\\\date{", text, count=1, flags=re.DOTALL)
    if n != 1:
        raise SystemExit("could not find the \\author{...}\\date{...} block")

    # The centred availability block on the title page names the repository and
    # both DOIs. It is one \begin{center} immediately after \maketitle.
    text, n = re.subn(
        r"\\begin\{center\}\\small\s*\nCode, results.*?\\end\{center\}\n",
        "",
        text,
        count=1,
        flags=re.DOTALL,
    )
    if n != 1:
        raise SystemExit("could not find the title-page availability block")

    # The generative-AI declaration is a \paragraph inside Acknowledgments, so
    # stripping that section takes a disclosure IOP requires and that names no
    # one. Lift it out first and re-emit it as a section of its own.
    ai_block = re.search(r"\\paragraph\{Use of generative AI\.\}(.*?)(?=\\section\*\{)", text, re.DOTALL)
    if ai_block is None:
        raise SystemExit("could not find the generative-AI declaration")

    for title in ANONYMISE_SECTIONS:
        if f"\\section*{{{title}}}" in text:
            text = _strip_section(text, title)

    # A function replacement, not a string: the lifted block is LaTeX, and every
    # backslash in it would otherwise be read as a regex escape.
    reinstated = "\\section*{Use of generative AI}\n" + ai_block.group(1).strip() + "\n\n\\section*{Funding}"
    text, n = re.subn(r"\\section\*\{Funding\}", lambda _: reinstated, text, count=1)
    if n != 1:
        raise SystemExit("could not reinsert the generative-AI declaration")

    # The code-availability paragraph and the Zenodo bibitem survive in shape but
    # not in content: a referee still needs to know the code is public and pinned.
    text = re.sub(
        r"\\url\{https://github\.com/lsalcedo100/FusionFlux\}",
        "[repository URL withheld for anonymous review]",
        text,
    )
    text = re.sub(
        r"\\bibitem\{zenodo\}.*?(?=\\end\{thebibliography\})",
        "\\\\bibitem{zenodo} [archive DOI withheld for anonymous review]\n",
        text,
        flags=re.DOTALL,
    )
    # Whatever is left is a spelling this function does not know about.
    for token in ("Salcedo", "liamsalcedo", "0009-0001-5039", "zenodo.2"):
        text = re.sub(re.escape(token), "", text, flags=re.IGNORECASE)
    return text


def anonymise_supplement(text: str) -> str:
    text, n = re.subn(r"\\author\{.*?\n\\date\{", "\\\\author{}\n\\\\date{", text, count=1, flags=re.DOTALL)
    if n != 1:
        raise SystemExit("could not find the supplement's \\author{...}\\date{...} block")
    # The standalone supplement names the parent article and its author.
    text = re.sub(r"\s*\(L\. Salcedo\)", "", text)
    for token in ("Salcedo", "liamsalcedo", "0009-0001-5039"):
        text = re.sub(re.escape(token), "", text, flags=re.IGNORECASE)
    return text


# --- the ScholarOne field sheet --------------------------------------------
#
# The upload form asks for the title, abstract and keywords as plain text, in
# fields that hold no markup. Retyping them is how a submitted abstract comes to
# differ from the one in the PDF beside it, so they are extracted here. The
# judgement-carrying entries, referee policy and the like, stay in the template.

TEMPLATE = PAPER / "scholarone_metadata.template.txt"

# The abstract's whole mathematical vocabulary. Unicode rather than a spelled-out
# name, because the field renders it and "rho" beside "=+0.85" reads as a typo.
# The abstract's whole mathematical vocabulary. Unicode rather than a spelled-out
# name, because the field renders it and "rho" beside "=+0.85" reads as a typo.
# The keys are escaped before use: as a regex, "\\rho" is a carriage return
# followed by "ho" and matches nothing, which is how an earlier version of this
# silently dropped every symbol in the abstract.
MATHS = {r"\rho": "\u03c1", r"\times": "\u00d7"}


def _plain(latex: str) -> str:
    """LaTeX to the plain text a form field will hold."""
    for command, glyph in MATHS.items():
        latex = re.sub(re.escape(command) + r"(?![a-zA-Z])", glyph, latex)
    latex = re.sub(r"\\cite\{[^}]*\}", "", latex)
    # A link's text is what a reader sees; its URL is the same thing spelled long.
    latex = re.sub(r"\\href\{[^}]*\}\{([^}]*)\}", r"\1", latex)
    latex = latex.replace("$", "")
    latex = latex.replace(r"\%", "%").replace("~", " ")
    # A footnote is separate text, so deleting the command that opens one runs
    # the name into the affiliation: "Liam SalcedoIndependent researcher".
    latex = latex.replace(r"\thanks{", ". ")
    latex = re.sub(r"\\[a-zA-Z]+", "", latex)
    latex = latex.replace("\\\\", " ")
    # Emptied, not spaced: "\\texttt{addr}." must not become "addr ."
    latex = re.sub(r"[{}]", "", latex)
    latex = latex.replace("``", '"').replace("''", '"')
    latex = latex.replace("---", "\u2014").replace("--", "\u2013")
    return re.sub(r"\s+", " ", latex).strip()


def _between(source: str, start: str, end: str) -> str:
    return source.split(start)[1].split(end)[0]


def _wrap(text: str, indent: str = "  ", width: int = 78) -> str:
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


def scholarone_metadata(paper: str, manuscript_pages: int, supplement_pages: int) -> str:
    title = _plain(_between(paper, "\\title{", "}\n\\author"))
    # \vspace and \textbf survive as their arguments; the title carries both.
    title = title.replace("-1.4cm", "").strip()
    abstract = _plain(_between(paper, "\\begin{abstract}", "\\end{abstract}"))
    keywords = _plain(_between(paper, "Keywords:}", "\\end{center}"))
    author = _plain(_between(paper, "\\author{", "}}\n\\date"))

    filled = TEMPLATE.read_text()
    for key, value in {
        "TITLE": _wrap(title),
        # One paragraph on one line: the field is a textarea, and a hard-wrapped
        # paste arrives with the line breaks in it.
        "ABSTRACT": "  " + abstract,
        "ABSTRACT_WORDS": str(len(abstract.split())),
        "KEYWORDS": _wrap(keywords),
        "AUTHOR": _wrap(author),
        "MANUSCRIPT_PAGES": str(manuscript_pages),
        "SUPPLEMENT_PAGES": str(supplement_pages),
    }.items():
        placeholder = "{{" + key + "}}"
        if placeholder not in filled:
            raise SystemExit(f"template has no {placeholder}")
        filled = filled.replace(placeholder, value)
    if "{{" in filled:
        raise SystemExit(f"unfilled placeholder in {TEMPLATE.name}")
    return filled


def _compile(source: Path, workdir: Path) -> Path:
    subprocess.run(
        ["tectonic", "-X", "compile", source.name, "--outdir", str(workdir)],
        cwd=workdir,
        check=True,
        capture_output=True,
    )
    return workdir / (source.stem + ".pdf")


def _rendered_text(pdf: Path) -> str:
    import pypdf

    return "".join(page.extract_text() or "" for page in pypdf.PdfReader(pdf).pages)


def _page_count(pdf: Path) -> int:
    import pypdf

    return len(pypdf.PdfReader(pdf).pages)


def verify_anonymous(pdfs: list[Path]) -> None:
    failures = []
    for pdf in pdfs:
        text = _rendered_text(pdf).lower()
        for token in IDENTIFYING:
            if token.lower() in text:
                failures.append(f"{pdf.name}: contains {token!r}")
    if failures:
        raise SystemExit("anonymised build is not anonymous:\n  " + "\n  ".join(failures))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-fresh-check",
        action="store_true",
        help="assemble even if the committed PDF is older than the source (for debugging only)",
    )
    args = parser.parse_args()

    if not args.skip_fresh_check:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "check_paper_submission.py"),
                "--check-pdf-fresh",
                "--check-provenance",
            ],
            check=True,
        )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "anonymous").mkdir(exist_ok=True)

    shutil.copy2(PAPER / "paper.pdf", OUT / "manuscript.pdf")
    shutil.copy2(PAPER / "supplementary.pdf", OUT / "supplementary_material.pdf")
    shutil.copy2(PAPER / "paper.tex", OUT / "paper.tex")
    shutil.copy2(PAPER / "supplementary.tex", OUT / "supplementary_material.tex")
    shutil.copy2(PAPER / "references.bib", OUT / "references.bib")

    pages = [_page_count(OUT / "manuscript.pdf"), _page_count(OUT / "supplementary_material.pdf")]
    (OUT / "scholarone_metadata.txt").write_text(scholarone_metadata((PAPER / "paper.tex").read_text(), *pages))

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        # Figures resolve through \graphicspath's flat-directory entry.
        for figure in (ROOT / "results").glob("*.pdf"):
            shutil.copy2(figure, work / figure.name)
        for figure in (ROOT / "results").glob("*.png"):
            shutil.copy2(figure, work / figure.name)

        (work / "paper.tex").write_text(anonymise((PAPER / "paper.tex").read_text()))
        (work / "supplementary.tex").write_text(anonymise_supplement((PAPER / "supplementary.tex").read_text()))
        shutil.copy2(PAPER / "references.bib", work / "references.bib")

        built = [_compile(work / "paper.tex", work), _compile(work / "supplementary.tex", work)]
        verify_anonymous(built)

        shutil.copy2(built[0], OUT / "anonymous" / "manuscript.pdf")
        shutil.copy2(built[1], OUT / "anonymous" / "supplementary_material.pdf")

    print(f"wrote {OUT.relative_to(ROOT)}/ (identified) and anonymous/ (verified clean)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
