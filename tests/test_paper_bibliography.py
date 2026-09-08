"""`paper/paper.tex` and `paper/references.bib` must not drift apart.

The paper prints its references from a hand-written `thebibliography`. That is
the right form for arXiv, which builds a submission with no BibTeX pass and no
.bbl, and it is the wrong form for a journal, which wants a .bib to run through
its own style file. So both exist, and two reference lists maintained by hand
are two reference lists that disagree within a month: a reference added to one,
a DOI corrected in the other, and nothing anywhere notices.

These tests are the thing that notices. Keys have to match in both directions,
and where an entry carries a DOI in one file it has to carry the same DOI in the
other. What is deliberately *not* checked is the prose of each entry: the .bib
holds full author lists that the printed list abbreviates to `et al.`, and
requiring those to match would force one of the two into the wrong shape.

The DOIs themselves were resolved against Crossref rather than reconstructed
from volume and page numbers, which is how the miscitation these tests now
protect was found: the symbolic-regression paper printed as Nucl. Fusion 55,
073009 is Plasma Phys. Control. Fusion 57, 014008.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "paper.tex"
SUPPLEMENT = ROOT / "paper" / "supplementary.tex"
BIB = ROOT / "paper" / "references.bib"

# Entries that legitimately have no DOI: a 1975 Soviet journal that predates
# them, two NeurIPS proceedings papers, a laboratory research plan, and an EPS
# conference contribution. Europhysics Conference Abstracts are not registered
# with Crossref, and a search there for the Hall title returns nothing, so the
# volume and paper number are the citable identifier. The two dataset entries
# have no DOI either: the OSF deposit has none registered (checked against the
# OSF API) and its GUID is the persistent identifier, and the BAAD archive is a
# tagged software release whose data paper carries the DOI instead.
NO_DOI = {"kadomtsev", "shiftcp", "ovadia", "hall", "hall26", "osfdb", "jt60saplan"}


def _strip_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", text)


def printed_entries() -> dict[str, str]:
    """Every `\\bibitem` key in both documents, mapped to the text of that entry.

    Both, because the supplement has its own reference list and reading only the
    paper is how this drifted: `jt60saop1`, `jt60saop2`, `jt60saplan` and `wbe`
    are cited in the supplement and were missing from the .bib for as long as
    these tests looked at `paper.tex` alone. Every one of them passed.
    """
    entries: dict[str, str] = {}
    for document in (PAPER, SUPPLEMENT):
        latex = _strip_comments(document.read_text())
        block = latex[
            latex.index(r"\begin{thebibliography}") : latex.index(r"\end{thebibliography}")
        ]
        for chunk in re.split(r"(?=\\bibitem\{)", block):
            if match := re.match(r"\\bibitem\{([^}]+)\}", chunk):
                entries[match.group(1)] = chunk
    return entries


def bib_entries() -> dict[str, str]:
    """Each key in references.bib, mapped to the body of that entry."""
    text = BIB.read_text()
    return {
        match.group(2): match.group(0)
        for match in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, re.DOTALL)
    }


def test_every_printed_reference_is_in_the_bib_file() -> None:
    missing = sorted(set(printed_entries()) - set(bib_entries()))
    assert not missing, (
        f"cited in the manuscript but absent from references.bib: {missing}. "
        "A journal running the .bib through its own style file would drop them."
    )


def test_every_bib_entry_is_actually_cited() -> None:
    unused = sorted(set(bib_entries()) - set(printed_entries()))
    assert not unused, (
        f"in references.bib but cited in neither document: {unused}. "
        "Either cite them or drop them; an uncited entry is a leftover."
    )


def test_dois_agree_between_the_two_files() -> None:
    printed, bib = printed_entries(), bib_entries()
    disagreements = []
    for key, chunk in printed.items():
        in_paper = re.search(r"doi:(10\.\S+?)\}", chunk)
        in_bib = re.search(r"doi\s*=\s*\{([^}]+)\}", bib.get(key, ""))
        paper_doi = in_paper.group(1) if in_paper else None
        bib_doi = in_bib.group(1) if in_bib else None
        if paper_doi != bib_doi:
            disagreements.append(f"{key}: paper.tex has {paper_doi!r}, references.bib has {bib_doi!r}")
    assert not disagreements, "DOIs disagree between the two reference lists:\n  " + "\n  ".join(
        disagreements
    )


def test_every_reference_without_a_doi_is_one_we_expect() -> None:
    """A silently missing DOI reads exactly like one that does not exist."""
    undoi = {key for key, chunk in printed_entries().items() if "doi:" not in chunk}
    assert undoi == NO_DOI, (
        f"references with no DOI are {sorted(undoi)}, expected {sorted(NO_DOI)}. "
        "If an entry gained or lost a DOI, resolve it against Crossref and "
        "update NO_DOI here so the exception stays deliberate."
    )


# --- each document has to stand on its own -----------------------------------
#
# `printed_entries` pools the two reference lists, which is right for the
# checks above: a key belongs in the .bib if either document prints it. It is
# wrong for the question a reader actually hits, because the two documents
# compile separately and a \cite resolves only against its own
# \begin{thebibliography}.
#
# That is how `wang`, `roberts` and `hall26` came to render as [?] three times
# in the shipped supplement and in the anonymous build. All three had entries in
# paper.tex, so the pooled set contained them and every test here passed. LaTeX
# says "There were undefined references" and exits zero, and nothing read the
# log.


def _own_bibitems(document: Path) -> set[str]:
    latex = _strip_comments(document.read_text())
    block = latex[latex.index(r"\begin{thebibliography}") : latex.index(r"\end{thebibliography}")]
    return set(re.findall(r"\\bibitem\{([^}]+)\}", block))


def _own_citations(document: Path) -> set[str]:
    latex = _strip_comments(document.read_text())
    return {
        key.strip()
        for group in re.findall(r"\\cite[a-z]*\{([^}]*)\}", latex)
        for key in group.split(",")
        if key.strip()
    }


def test_every_citation_resolves_within_its_own_document() -> None:
    for document in (PAPER, SUPPLEMENT):
        dangling = sorted(_own_citations(document) - _own_bibitems(document))
        assert not dangling, (
            f"{document.name} cites {dangling} with no \\bibitem of its own, so each "
            "renders as [?]. The two documents compile separately; an entry in the "
            "other one does not help."
        )


def test_no_document_carries_a_bibitem_it_never_cites() -> None:
    """The other direction, which is how a stale entry survives a rewrite."""
    for document in (PAPER, SUPPLEMENT):
        orphaned = sorted(_own_bibitems(document) - _own_citations(document))
        assert not orphaned, (
            f"{document.name} prints {orphaned} in its reference list and cites none of "
            "them. Either cite them or drop them; a printed reference nothing points at "
            "is a leftover a referee will notice before you do."
        )
