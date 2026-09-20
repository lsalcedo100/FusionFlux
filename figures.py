"""Write each figure twice, because its two consumers want different things.

The README and the built page are read on GitHub and in a browser, where a
raster is what renders inline and a multi-megabyte one is a slow page. A journal
wants line art as vector, where resolution is not a parameter at all: IOP asks
for 600 dpi from raster line art and for vector in preference to it.

The raster the analyses used to write was not far off that bar. At the pixel
widths used then, a figure placed across `\\textwidth` landed at 333 to 381 dpi,
which clears the 300 dpi minimum and misses the 600 dpi preference. Reaching 600 the
raster way costs a lot of bytes for a plot that is lines and markers: at a
matching effective resolution the PNG is about 1.1 MB where the PDF of the same
axes is about 195 KB, since a vector file stores the points rather than every
pixel they land on. So both are written, from one figure, and each consumer gets
the form it wants.

`paper/paper.tex` names its figures with no extension. LaTeX resolves those
through `\\DeclareGraphicsExtensions`, taking the PDF where there is one and
falling back to the PNG otherwise, so a figure that has not been regenerated
since this module landed still builds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# The figures were written at 170 or 180 depending on the script. The number
# only sets the raster's pixel size now that the paper reads the vector copy, so
# it is one value here rather than two spellings of "big enough for the README".
FIGURE_DPI = 180

# IOP asks that vector figures use a standard font family (Times, Helvetica,
# Courier, Symbol) and, in practice, that they not arrive as Type 3. Matplotlib
# defaults to DejaVu Sans embedded as Type 3, which is what a preflight flags:
# a Type 3 font is a bundle of drawing procedures rather than a real font
# program, so it does not hint, does not always extract as text, and is what
# most print pipelines complain about first.
#
# `pdf.fonttype = 42` embeds a subset TrueType program instead, which is the
# fix for the Type 3 part on its own. The family is set to Helvetica with Arial
# and DejaVu Sans behind it, so a machine without Helvetica still renders rather
# than failing. Mathtext keeps its own stix fonts: the figures set rho, lambda
# and similar in math mode, and the Base-14 route (`pdf.use14corefonts`) cannot
# carry those glyphs.
FONT_STACK = ("Helvetica", "Arial", "DejaVu Sans")


def apply_font_policy() -> None:
    """Set the font rcParams every figure in this project is drawn under."""
    import matplotlib

    matplotlib.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": list(FONT_STACK),
            "mathtext.fontset": "stix",
        }
    )


# Applied on import rather than at save time. Matplotlib resolves a font when a
# text artist is created, not when the figure is written, so setting these in
# `save_figure` would leave every label already drawn under the old defaults.
# Every analysis script imports this module for its styling before it plots.
apply_font_policy()

# Paper figures are authored at the width they are printed at.
#
# The figures used to be drawn 12.5 to 13.5 in wide and placed across the
# paper's 6.378 in text width, so every mark on them arrived at the page at
# roughly half its specified size: 9 pt tick labels became 4.3 pt. Enlarging the
# type on a canvas that wide does not fix it, it only crowds the panels until
# titles and tick labels collide, because what matters is type size relative to
# the canvas, not either alone.
#
# So the canvas shrinks to the printed width and the panels stack instead of
# sitting side by side. Each panel then gets the full width rather than a third
# of it, which leaves *more* room per panel than before while the type lands on
# the page at the size it is written here. These sizes sit just under the 10 pt
# body text, which is where figure lettering belongs.
#
# 8.3 pt is the floor, not a preference. IOP asks for 8 to 12 pt at *final*
# figure size, and these are authored at 6.6 in but placed at \linewidth, which
# is 6.38 in on a4paper with 2.4 cm margins: a 0.967 scale. Anything written
# below 8.3 here renders under 8 pt on the page. The previous 7.5 pt ticks and
# 7.0 pt small text landed at 7.25 and 6.77.
PAPER_WIDTH_IN = 6.6

FONT_TITLE = 10.0
FONT_LABEL = 9.0
FONT_TICK = 8.5
FONT_LEGEND = 8.5
FONT_ANNOTATION = 8.5
FONT_SMALL = 8.3

# Marker and line style per model, so no figure carries its meaning in colour
# alone. A reader printing the paper in grey, or one of the roughly one in twelve
# men with a red-green deficiency, has to be able to tell the random forest from
# the power law, and in Fig. 1 that distinction is the entire result. Colour
# stays, because it is the fastest channel for readers who can use it; these
# make it the second channel rather than the only one.
#
# The pairs are chosen to stay distinct at the printed size: a filled circle, a
# square and an open triangle read apart at 4 pt in a way that, say, a circle and
# an octagon do not.
# One colour per model, for every figure that draws them. Four analysis scripts
# used to carry their own copy of this table, and the copies had drifted into a
# palette that did not separate. The forest's orange and the booster's old
# `#c8873a` sat 8.6 apart in OKLab Delta E for a reader with full colour vision,
# under the 15 at which two series start to read as one, and 0.9 apart under
# deuteranopia. Separately, the old IPB98(y,2) green `#3f8f5c` sat 4.6 from that
# orange under protanopia, which is the ordinary red-green confusion.
#
# The ridge and the forest keep the colours they have always had here, because
# that pair is the result. The booster and the published law were re-chosen by
# search over OKLCH, keeping only colours that clear every pair among the four:
# the worst pair is now 23.0 for full colour vision and 8.6 under any of the
# three deficiencies, against a target of 8. The markers and line styles below
# still carry every distinction a second time.
# The green that stays apart from the forest's orange under protanopia. It is
# IPB98(y,2)'s colour, and figures that draw no published law borrow it for
# their own third series, so it has one name and one value.
SEPARABLE_GREEN = "#0da26b"

MODEL_COLORS = {
    "ridge_loglinear": "#2a78d6",
    "random_forest": "#eb6834",
    "hist_gradient_boosting": "#8b3473",
    "ipb98y2_analytic": SEPARABLE_GREEN,
}

# A series that is on a panel for comparison and is not the panel's subject.
# Dark enough to read in print: the pale greys this replaces were 1.9:1 against
# the page, and this is 4.4:1.
CONTEXT_GREY = "#77767a"

# The three splits are an escalation, from a held-out discharge to a held-out
# device to a held-out size range, so they take one hue running light to dark
# and not three unrelated hues. The steps sit 19 apart in OKLab Delta E. It also
# keeps the split colours off the model colours, which matters in the one figure
# that draws both: blue used to mean "cross-validation" in its top panel and
# "the ridge" in its bottom one.
SPLIT_RAMP = {
    "grouped_cv": "#9fadc6",
    "leave_one_tokamak_out": "#5f7396",
    "size_cut": "#2b3c59",
}

MODEL_MARKERS = {
    "ipb98y2_analytic": "s",
    "ridge_loglinear": "o",
    "hist_gradient_boosting": "^",
    "random_forest": "D",
    "mean_baseline": "x",
}

MODEL_LINESTYLES = {
    "ipb98y2_analytic": "-",
    "ridge_loglinear": "-",
    "hist_gradient_boosting": "--",
    "random_forest": ":",
    "mean_baseline": "-.",
}


def model_color(name: str) -> str:
    """Colour for a model. A name without one is an error, not a silent grey."""
    return MODEL_COLORS[name]


def model_style(name: str) -> tuple[str, str]:
    """Marker and line style for a model, falling back to a plain solid circle."""
    return MODEL_MARKERS.get(name, "o"), MODEL_LINESTYLES.get(name, "-")


def save_figure(figure: Any, path: Path, **savefig_kwargs: Any) -> Path:
    """Save `figure` as the PNG at `path` and as a PDF beside it.

    Returns the PNG path, which is the one the analyses print and the one the
    README and the page link to.

    The PDF is written without a ``CreationDate``. Matplotlib stamps the current
    time into that field by default, so every regeneration produced a
    byte-different file even when the plotted numbers were identical, and
    `make reproduce` left the six committed figure PDFs dirty on every run. The
    PNGs carry no such field and were already reproducible.
    """
    figure.savefig(path, dpi=FIGURE_DPI, **savefig_kwargs)
    pdf_kwargs = {"metadata": {"CreationDate": None}, **savefig_kwargs}
    figure.savefig(path.with_suffix(".pdf"), **pdf_kwargs)
    return path
