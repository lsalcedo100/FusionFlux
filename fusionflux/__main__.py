"""``python3 -m fusionflux`` runs the same command as the console script.

Useful in two places. In a checkout there is no installed console script until
someone runs `pip install -e .`, and `make results` should not depend on that.
And in an environment where the wheel is installed but its scripts directory is
not on ``PATH``, this reaches the same entry point through the interpreter that
imported it.

``python3 -m fusionflux.predictor build`` still works and is what the card
builder's own ``main`` is for, but it warns: importing this package imports
``fusionflux.predictor`` eagerly, so runpy finds it in ``sys.modules`` before it
executes it. Going through this module avoids that.
"""

from __future__ import annotations

import sys

from fusionflux.cli import main

if __name__ == "__main__":
    main(sys.argv[1:])
