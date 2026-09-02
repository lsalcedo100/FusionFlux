"""Build-time hook: take the module from the repository root.

``scaling_audit.py`` lives at the root of the FusionFlux repository, where the
study's own analyses import it and where its test suite exercises it against
real pipelines rather than against a copy. That is the point of the module: the
claim is that it is domain-agnostic, and the evidence is that the fusion study,
the mammalian-metabolism replication and the tree-allometry ladder all run
through this exact file.

Vendoring a second copy into this directory would break that. The published
package would be a fork of the tested one from the first edit onward, and
nothing would notice. So the root file is the single source under version
control and this copies it in at build time; the copy is gitignored.

``tests/test_scaling_audit.py`` in the parent repository asserts that a built
wheel carries a module byte-identical to the root one.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

HERE = Path(__file__).parent
SOURCE_MODULE = HERE.parent / "scaling_audit.py"
VENDORED_MODULE = HERE / "scaling_audit.py"


class BuildPyFromRepositoryRoot(build_py):
    """Refresh ``scaling_audit.py`` from the repository root before collecting files."""

    def run(self) -> None:
        if SOURCE_MODULE.exists():
            shutil.copyfile(SOURCE_MODULE, VENDORED_MODULE)
        elif not VENDORED_MODULE.exists():
            # Building from an unpacked sdist is the legitimate case for the
            # source being absent: the sdist carries the module already. With
            # neither, the wheel would install an empty distribution.
            raise SystemExit(
                f"Cannot build: neither {SOURCE_MODULE} nor {VENDORED_MODULE} exists, "
                "so the wheel would contain no module. Build from a checkout of "
                "the FusionFlux repository, or from the published sdist."
            )
        super().run()


if __name__ == "__main__":
    setup(cmdclass={"build_py": BuildPyFromRepositoryRoot})
