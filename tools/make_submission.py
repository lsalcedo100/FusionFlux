"""Assemble the ScholarOne upload from the built paper, in both variants.

Nuclear Fusion takes the manuscript as a PDF plus its sources, and offers
single- or double-anonymous review. Which one the author picks is not known
until upload, so this writes both: `submission/` carries the identified manuscript,
and `submission/anonymous/` the same paper with the author
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
# Not under build/: tests/test_packaging.py and tests/test_wheel_smoke.py
# remove that directory, so an assembled bundle disappears on the next full
# test run, which is a bad surprise between assembling an upload and making it.
OUT = ROOT / "submission"

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
    "10.5281/zenodo.22651178",  # version DOI, v0.4.3
    "10.5281/zenodo.22215142",  # concept DOI, a different \href spelling
    "zenodo",
    "github.com/lsalcedo100",
)

# Every Zenodo DOI the source currently prints. The literals above are kept for
# the reason stated there, and they are the DOIs of releases already made: the
# moment the paper is repointed at a new one, the list guards a string the PDF
# no longer contains and stops guarding the string it does. So the printed DOIs
# are read out of the source as well, and both sets are checked.
ZENODO_DOI = re.compile(r"10\.5281/zenodo\.\d+")


def identifying_tokens(sources: "list[Path] | None" = None) -> tuple[str, ...]:
    """`IDENTIFYING`, plus every Zenodo DOI the LaTeX sources print today."""
    found = set()
    for source in sources or (PAPER / "paper.tex", PAPER / "supplementary.tex"):
        if source.exists():
            found.update(ZENODO_DOI.findall(source.read_text()))
    return tuple(dict.fromkeys((*IDENTIFYING, *sorted(found))))


# IOP requires the generative-AI disclosure, the funding statement, the competing
# interests and the contributions to sit in Acknowledgements, so paper.tex keeps
# all four there as run-in paragraphs. Anonymisation can therefore no longer drop
# that section wholesale: it would take three required declarations with it. What
# comes out instead is the credit prose that opens the section, and the
# contributions paragraph, which is the only one of the four that names anyone.
ACKNOWLEDGMENT_HEAD = "\\section*{Acknowledgments}"
FIRST_DECLARATION = "\\paragraph*{Use of generative AI.}"
ANONYMISE_PARAGRAPHS = ("Author contributions.",)


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


def _strip_paragraph(text: str, title: str) -> str:
    """Drop one `\\paragraph*{title}` run-in and its body.

    Stops at the next run-in, the next section or the bibliography, so a
    paragraph can be removed from the middle of Acknowledgements without taking
    the declarations after it.
    """
    pattern = re.compile(
        r"\\paragraph\*\{"
        + re.escape(title)
        + r"\}.*?(?=\\paragraph\*?\{|\\section\*?\{|\\begin\{thebibliography\})",
        re.DOTALL,
    )
    stripped, n = pattern.subn("", text)
    if n != 1:
        raise SystemExit(f"expected exactly one '{title}' paragraph, found {n}")
    return stripped


def _strip_acknowledgment_credits(text: str) -> str:
    """Keep the Acknowledgements heading and its declarations, drop the thanks."""
    pattern = re.compile(
        re.escape(ACKNOWLEDGMENT_HEAD) + r".*?(?=" + re.escape(FIRST_DECLARATION) + r")",
        re.DOTALL,
    )
    # A callable replacement, because the heading is full of backslashes and a
    # template string would read them as escapes.
    stripped, n = pattern.subn(lambda _match: ACKNOWLEDGMENT_HEAD + "\n\n", text)
    if n != 1:
        raise SystemExit(f"expected one Acknowledgements credit block, found {n}")
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

    # The three required declarations live inside Acknowledgements because IOP
    # puts them there, so what is removed is the credit prose that opens the
    # section and the one paragraph that names the author. This is checked
    # rather than assumed: a required declaration disappearing is the kind of
    # loss that does not announce itself.
    text = _strip_acknowledgment_credits(text)
    for title in ANONYMISE_PARAGRAPHS:
        text = _strip_paragraph(text, title)

    for required in (FIRST_DECLARATION, "\\paragraph*{Funding.}", "\\paragraph*{Competing interests.}"):
        if required not in text:
            raise SystemExit(f"anonymisation removed a required declaration: {required}")

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
    # The provenance pin is a 40-character commit hash of a public repository,
    # which GitHub indexes: pasting it into a search box is a short route back
    # to the author, and it is the one identifying string that does not look
    # like one.
    text = re.sub(r"[0-9a-f]{40}", "[commit hash withheld for anonymous review]", text)

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


# The figures both documents include, resolved to the vector copy. Kept as a
# derived list rather than a literal so a new figure cannot be left out of the
# bundle while the document that needs it ships.
def _included_figures() -> tuple[str, ...]:
    names: list[str] = []
    for document in (PAPER / "paper.tex", PAPER / "supplementary.tex"):
        for name in re.findall(r"\\includegraphics\[[^]]*\]\{([^}]+)\}", document.read_text()):
            if f"{name}.pdf" not in names:
                names.append(f"{name}.pdf")
    return tuple(names)


FIGURES = _included_figures()

TEMPLATE = PAPER / "scholarone_metadata.template.txt"

# The abstract's whole mathematical vocabulary, spelled out in ASCII. It used to
# render as Unicode, on the reasoning that the field displays it and "rho"
# beside "=+0.85" reads as a typo. That was the wrong trade: ScholarOne's
# abstract box is not a Unicode-safe field, and a rho or a multiplication sign
# that arrives as a replacement character turns the first thing an editor reads
# into "at ?=+0.85". A spelled-out symbol is ugly and survives.
#
# The keys are escaped before use: as a regex, "\\rho" is a carriage return
# followed by "ho" and matches nothing, which is how an earlier version of this
# silently dropped every symbol in the abstract.
MATHS = {r"\rho": "rho", r"\tau": "tau", r"\times": "x"}

# Everything else the LaTeX can produce that is not ASCII. En dashes come from
# "--" in names like Connor--Taylor, and the typeset minus is not a hyphen.
NON_ASCII = {
    "\u2013": "-",
    "\u2014": "--",
    "\u2212": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u00a0": " ",
}


def _plain(latex: str) -> str:
    """LaTeX to the plain text a form field will hold."""
    for command, glyph in MATHS.items():
        latex = re.sub(re.escape(command) + r"(?![a-zA-Z])", glyph, latex)
    # Anything still commanded inside math is about to be deleted by the
    # catch-all below, which is how "the identity $\tau=W/P$" reached a live
    # submission as "the identity =W/P". Unmapped is a bug in MATHS, not text
    # to drop: say so rather than emit a field with a hole in it.
    unmapped = sorted({
        command
        for segment in re.findall(r"\$[^$]*\$", latex)
        for command in re.findall(r"\\[a-zA-Z]+", segment)
    })
    if unmapped:
        raise ValueError(
            "no ASCII spelling for " + ", ".join(unmapped) + "; add it to MATHS"
        )
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
    latex = latex.replace("---", "--").replace("--", "-")
    for glyph, plain in NON_ASCII.items():
        latex = latex.replace(glyph, plain)
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

    def reference_number(key: str) -> str:
        """The printed [n] for a citation key.

        Written out rather than typed into the template. These numbers appear in
        the conflict-of-interest declaration, where a stale one points the editor
        at an unrelated paper, and they move whenever a citation is added earlier
        in the text.
        """
        keys = re.findall(r"\\bibitem\{([^}]*)\}", paper)
        if key not in keys:
            raise SystemExit(f"paper.tex has no \\bibitem{{{key}}}")
        return str(keys.index(key) + 1)

    def section_of_first_citation(key: str) -> str:
        """The numbered section in which a key is first cited, for the same reason."""
        hit = re.search(r"\\cite\{[^}]*\b" + re.escape(key) + r"\b[^}]*\}", paper)
        if hit is None:
            raise SystemExit(f"paper.tex never cites {key}")
        return str(sum(1 for m in re.finditer(r"\\section\{", paper) if m.start() < hit.start()))

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
        "REF_HDB5": reference_number("hdb5"),
        "REF_HALL": reference_number("hall"),
        "REF_HALL26": reference_number("hall26"),
        "SEC_KARDAUN": section_of_first_citation("kardaun"),
    }.items():
        placeholder = "{{" + key + "}}"
        if placeholder not in filled:
            raise SystemExit(f"template has no {placeholder}")
        filled = filled.replace(placeholder, value)
    if "{{" in filled:
        raise SystemExit(f"unfilled placeholder in {TEMPLATE.name}")
    return filled


def _compile(source: Path, workdir: Path) -> Path:  # pragma: no cover - needs tectonic
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


# Any 40-character hex string, which is what a git commit looks like. Checked as
# a pattern rather than as a literal in IDENTIFYING: the pin moves whenever
# results/ is regenerated, and a guard naming one particular hash goes stale
# exactly when it is needed. GitHub indexes commits in public repositories, so
# this is a short search away from the author.
COMMIT_HASH = re.compile(r"\b[0-9a-f]{40}\b")


# A LaTeX \cite with no matching \bibitem renders as a literal [?], and a \ref
# with no \label as ??. Both are silent: the build succeeds and the PDF looks
# finished. The anonymous variant is rewritten by this script rather than
# authored, so it is the build where a citation can go missing without anyone
# having edited a bibliography, and it did: an earlier lookahead here ran from
# \bibitem{zenodo} to \end{thebibliography} and swallowed the five entries after
# it, shipping seven [?] marks in a PDF that verify_anonymous called clean.
# Checked on the rendered text, because that is the only place the marks exist.
# The bracket form allows commas and repeats, because a multi-key \cite renders
# as one bracket: \cite{petty,luce} with neither key defined prints [?, ?], and
# a pattern anchored on a single ? inside brackets walks straight past it. That
# was one of the seven marks in the build this guard exists to have caught.
UNRESOLVED = re.compile(r"\[[\s?,]*\?[\s?,]*\]|\?\?")


def unresolved_references(pdf: Path, rendered: str) -> list[str]:
    found = UNRESOLVED.findall(rendered)
    if not found:
        return []
    return [
        f"{pdf.name}: {len(found)} unresolved citation or cross-reference "
        f"({', '.join(sorted(set(m.strip() for m in found)))}); "
        "a \\cite or \\ref did not survive the rewrite"
    ]


def verify_anonymous(pdfs: list[Path]) -> None:
    failures = []
    for pdf in pdfs:
        rendered = _rendered_text(pdf)
        text = rendered.lower()
        for token in identifying_tokens():
            if token.lower() in text:
                failures.append(f"{pdf.name}: contains {token!r}")
        for found in set(COMMIT_HASH.findall(text)):
            failures.append(f"{pdf.name}: contains the commit hash {found}")
        failures.extend(unresolved_references(pdf, rendered))
    if failures:
        # "not anonymous" stays in the header because it is still the failure
        # this function exists for and the one a reader scanning output looks
        # for; the unresolved-reference lines say plainly what they are.
        raise SystemExit("anonymised build is not anonymous or not usable:\n  " + "\n  ".join(failures))


def main() -> int:  # pragma: no cover - builds PDFs and writes the bundle
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
    # Kept, against the advice to drop it as unread: neither document runs it,
    # but a journal's production style file wants a .bib at revision, which is
    # why it exists. The worry about shipping a file nobody built from is that
    # it drifts from the rendered list, and it had, missing the supplement's
    # four keys, because the test that guards it read paper.tex alone. That test
    # now reads both and binds keys and DOIs in both directions, so a reader who
    # diffs the two finds no disagreement on either.
    shutil.copy2(PAPER / "references.bib", OUT / "references.bib")

    # The figures, so the sources build outside this repository. Without them a
    # reviewer or a production editor unpacking submission/ hits a missing-file
    # error on the first \includegraphics, and IOP asks for source plus figures
    # at revision. \graphicspath's flat-directory entry resolves them here.
    for figure in FIGURES:
        shutil.copy2(ROOT / "results" / figure, OUT / figure)

    pages = [_page_count(OUT / "manuscript.pdf"), _page_count(OUT / "supplementary_material.pdf")]
    (OUT / "scholarone_metadata.txt").write_text(scholarone_metadata((PAPER / "paper.tex").read_text(), *pages))

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        # Figures resolve through \graphicspath's flat-directory entry.
        for artwork in (ROOT / "results").glob("*.pdf"):
            shutil.copy2(artwork, work / artwork.name)
        for artwork in (ROOT / "results").glob("*.png"):
            shutil.copy2(artwork, work / artwork.name)

        (work / "paper.tex").write_text(anonymise((PAPER / "paper.tex").read_text()))
        (work / "supplementary.tex").write_text(anonymise_supplement((PAPER / "supplementary.tex").read_text()))
        shutil.copy2(PAPER / "references.bib", work / "references.bib")

        built = [_compile(work / "paper.tex", work), _compile(work / "supplementary.tex", work)]
        verify_anonymous(built)

        shutil.copy2(built[0], OUT / "anonymous" / "manuscript.pdf")
        shutil.copy2(built[1], OUT / "anonymous" / "supplementary_material.pdf")

    print(f"wrote {OUT.relative_to(ROOT)}/ (identified) and anonymous/ (verified clean)")
    # Not fatal: the bundle is worth having while the nominations are still being
    # decided. But this is the one ScholarOne field that changes who reads the
    # paper, and an unfilled one is easy to click past at upload.
    if "FILL-IN" in (OUT / "scholarone_metadata.txt").read_text():
        print("\nNOTE: referee nominations are still FILL-IN in scholarone_metadata.txt.")
        print("Candidates and the one conflict to declare are listed under REFEREES.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
