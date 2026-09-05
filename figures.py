"""Write each figure twice, because its two consumers want different things.

The README and the built page are read on GitHub and in a browser, where a
raster is what renders inline and a multi-megabyte one is a slow page. A journal
wants line art as vector, where resolution is not a parameter at all: IOP asks
for 600 dpi from raster line art and for vector in preference to it.

The raster the analyses used to write is not far off that bar. At the pixel
widths here a figure placed across `\\textwidth` lands at 333 to 381 dpi, which
clears the 300 dpi minimum and misses the 600 dpi preference. Reaching 600 the
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

def save_figure(figure: Any, path: Path, **savefig_kwargs: Any) -> Path:
    """Save `figure` as the PNG at `path` and as a PDF beside it.

    Returns the PNG path, which is the one the analyses print and the one the
    README and the page link to.
    """
    figure.savefig(path, dpi=FIGURE_DPI, **savefig_kwargs)
    figure.savefig(path.with_suffix(".pdf"), **savefig_kwargs)
    return path
