"""Every figure has to fit on a page with its caption.

Figure 1 did not, for at least two archived releases. At ``\\linewidth`` it stood
23.1 cm tall in a 24.9 cm text block, its caption needed eight lines more, and
LaTeX set the float anyway, 67 pt too large, with the page number printed across
the caption's sixth line. LaTeX said so in a log nobody reads, and CI has no TeX
toolchain to produce that log at all.

The sum does not need one. A float's height is the image at the width it is set
at, the space above the caption, and the caption's lines. For the old Figure 1
that comes to 775.3 pt against a text block of 708.5 pt, which is the 66.8 pt
LaTeX reported, to the decimal. So this test does the sum for every figure in
both documents.

The one estimate in it is the caption's line count, taken as one line per 90
characters. Counted on the seven rendered captions that is never an
underestimate: they run between 81 and 104 characters to the line. It errs
towards failing, which is the right way round for a check whose job is to stop a
page number landing on a caption.
"""

from __future__ import annotations

import math
import re
import struct
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS = (ROOT / "paper" / "paper.tex", ROOT / "paper" / "supplementary.tex")

PT_PER_CM = 72.27 / 2.54
# \documentclass[12pt,a4paper] with \usepackage[margin=2.4cm]{geometry}, in both documents.
PAGE_CM, MARGIN_CM = (21.0, 29.7), 2.4
LINEWIDTH_PT = (PAGE_CM[0] - 2 * MARGIN_CM) * PT_PER_CM
TEXTHEIGHT_PT = (PAGE_CM[1] - 2 * MARGIN_CM) * PT_PER_CM
# \captionsetup{font=small} under a 12 pt class: 10.95 pt type on a 13.6 pt baseline,
# and the article class's 10 pt \abovecaptionskip.
CAPTION_BASELINE_PT, ABOVE_CAPTION_PT = 13.6, 10.0
CHARACTERS_PER_LINE = 90

FIGURE = re.compile(
    r"\\begin\{figure\}.*?\\includegraphics\[width=\\linewidth\]\{(?P<name>\w+)\}.*?"
    r"\\caption\{(?P<caption>.*?)\}\s*\\label",
    re.S,
)


def _png_size(path: Path) -> tuple[int, int]:
    """Width and height from the IHDR chunk, so this needs no imaging library."""
    header = path.read_bytes()[:24]
    assert header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR", f"{path.name} is not a PNG"
    width, height = struct.unpack(">II", header[16:24])
    return width, height


def _printed_length(caption: str, label: str) -> int:
    text = re.sub(r"\\textbf\{([^}]*)\}", r"\1", caption)
    text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", "", text)
    text = re.sub(r"\s+", " ", re.sub(r"[{}$~]", " ", text)).strip()
    return len(label) + len(text)


def float_height_pt(image_height_over_width: float, n_caption_characters: int) -> float:
    lines = math.ceil(n_caption_characters / CHARACTERS_PER_LINE)
    return LINEWIDTH_PT * image_height_over_width + ABOVE_CAPTION_PT + lines * CAPTION_BASELINE_PT


def _figures() -> list[tuple[str, str, float, int]]:
    found = []
    for document in DOCUMENTS:
        label = "Figure S1: " if document.name == "supplementary.tex" else "Figure 1: "
        for match in FIGURE.finditer(document.read_text(encoding="utf-8")):
            width, height = _png_size(ROOT / "results" / f"{match['name']}.png")
            found.append((document.name, match["name"], height / width, _printed_length(match["caption"], label)))
    return found


def test_the_geometry_is_the_one_the_documents_declare() -> None:
    """The sum below is only as good as these two lines of each preamble."""
    for document in DOCUMENTS:
        latex = document.read_text(encoding="utf-8")
        assert r"\documentclass[12pt,a4paper]{article}" in latex, document.name
        assert r"\usepackage[margin=2.4cm]{geometry}" in latex, document.name
        assert r"\captionsetup{font=small" in latex, document.name


def test_the_sum_reproduces_the_overflow_latex_reported() -> None:
    """The old Figure 1: 9.4 in tall at 6.6 in wide, under an eight-line caption."""
    eight_lines = 8 * CHARACTERS_PER_LINE
    overflow = float_height_pt(9.4 / 6.6, eight_lines) - TEXTHEIGHT_PT
    assert overflow == pytest.approx(66.8, abs=0.3)


def test_every_figure_is_found() -> None:
    names = {name for _, name, _, _ in _figures()}
    assert names == {"extrapolation", "conformal", "dimensional", "allometry", "tree_allometry", "gp", "size_extrapolation"}


@pytest.mark.parametrize("document,name,aspect,characters", _figures(), ids=lambda value: str(value)[:24])
def test_the_figure_and_its_caption_fit_the_text_block(document: str, name: str, aspect: float, characters: int) -> None:
    height = float_height_pt(aspect, characters)
    assert height <= TEXTHEIGHT_PT, (
        f"{name} in {document} needs about {height:.0f} pt with its caption and the text block is "
        f"{TEXTHEIGHT_PT:.0f} pt, so LaTeX will set it {height - TEXTHEIGHT_PT:.0f} pt too large and the page "
        "number will print over the caption. Make the figure shorter at its source, not smaller in the "
        "document: scaling it down takes its labels under the 8 pt floor figures.py holds them at."
    )
