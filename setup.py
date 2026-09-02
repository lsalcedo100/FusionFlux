"""Build-time hook: ship the predictor card inside the package.

Everything else about this build is declarative and lives in ``pyproject.toml``.
This file exists for one reason, and it is the defect that made the first wheel
unusable: ``fusionflux predict`` reads ``results/predictor.json``, that file is
the output of an analysis rather than source, and it sits outside the package
directory. Setuptools cannot reach outside a package for package data, so a
purely declarative build produced a wheel whose one advertised command raised
``FileNotFoundError`` on a clean install.

Copying it in at build time, rather than committing a second copy under
``fusionflux/``, is what keeps the repository's existing rule intact: a
generated artifact has exactly one committed home, and everything else reads or
regenerates from it. A committed copy would be a file that `make results` does
not update and that nothing would notice going stale. This copy is gitignored
and recreated from ``results/predictor.json`` on every single build, so the
wheel cannot carry a card older than the analysis it was built from.

``tests/test_packaging.py`` builds a wheel and asserts the file arrives.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

SOURCE_CARD = Path(__file__).parent / "results" / "predictor.json"
PACKAGED_CARD = Path(__file__).parent / "fusionflux" / "predictor.json"


class BuildPyWithCard(build_py):
    """Refresh ``fusionflux/predictor.json`` from ``results/`` before collecting files.

    Two build contexts, and they differ in which copy exists.

    From a checkout, ``results/predictor.json`` is present and is the source of
    truth, so it is copied in on every build and the packaged copy can never be
    older than the analysis.

    From an unpacked sdist there is no ``results/`` directory: the sdist carries
    the package and its data, not the analysis outputs. The card it needs is
    already inside the package, put there when the sdist itself was built. So a
    missing source card is only a failure when there is no packaged card either,
    which is the case that would produce an install whose console command cannot
    start.
    """

    def run(self) -> None:
        if SOURCE_CARD.exists():
            shutil.copyfile(SOURCE_CARD, PACKAGED_CARD)
        elif not PACKAGED_CARD.exists():
            # Fail here rather than shipping a distribution that is missing the
            # one file its console command cannot start without. A build is the
            # last point at which this is cheap to notice.
            raise SystemExit(
                f"Cannot build: neither {SOURCE_CARD} nor {PACKAGED_CARD} exists, "
                "so the install would have a `fusionflux predict` that cannot read "
                "its coefficients. Run `python3 -m fusionflux card` (needs the HDB5 "
                "dataset) or `make results` first."
            )
        super().run()


# Guarded so the module can be imported and its hook inspected without invoking
# a build. `setuptools.build_meta` executes this file with `__name__` set to
# `"__main__"`, so the real build path is unaffected.
if __name__ == "__main__":
    setup(cmdclass={"build_py": BuildPyWithCard})
