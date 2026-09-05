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
PAPER_WIDTH_IN = 6.6

FONT_TITLE = 10.0
FONT_LABEL = 9.0
FONT_TICK = 7.5
FONT_LEGEND = 8.5
FONT_ANNOTATION = 8.0
FONT_SMALL = 7.0

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


def model_style(name: str) -> tuple[str, str]:
    """Marker and line style for a model, falling back to a plain solid circle."""
    return MODEL_MARKERS.get(name, "o"), MODEL_LINESTYLES.get(name, "-")


def save_figure(figure: Any, path: Path, **savefig_kwargs: Any) -> Path:
    """Save `figure` as the PNG at `path` and as a PDF beside it.

    Returns the PNG path, which is the one the analyses print and the one the
    README and the page link to.
    """
    figure.savefig(path, dpi=FIGURE_DPI, **savefig_kwargs)
    figure.savefig(path.with_suffix(".pdf"), **savefig_kwargs)
    return path
