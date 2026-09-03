"""What the wheel actually contains, and what installing it does to an environment.

Two defects shipped in the 0.2.0 metadata, and neither was visible from a
checkout, because a checkout has the repository root on ``sys.path`` and every
file already on disk. Both are only observable in a built distribution:

1. ``results/predictor.json`` was not package data, so the installed
   ``fusionflux predict`` (the command the README leads with) raised
   ``FileNotFoundError`` on a clean install.
2. Every analysis script shipped as a top-level module, so installing this
   package put ``config``, ``storage``, ``validation``, ``tools`` and a dozen
   more into site-packages, shadowing any other project's module of those names
   in the same environment.

The tests here build a real wheel and read it back. That is slower than
asserting against ``pyproject.toml``, and it is the point: a metadata assertion
would have passed on 0.2.0, because the metadata was self-consistent and wrong.

``tests/test_wheel_smoke.py`` is the other half, installing the wheel into a
clean interpreter and running the command.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CARD = PROJECT_ROOT / "results" / "predictor.json"

# Names that must never reach site-packages. Generic enough that another project
# in the same environment plausibly defines its own, which is what makes
# shadowing them a real fault rather than an aesthetic one.
RESERVED_TOP_LEVEL_NAMES = frozenset(
    {
        "allometry",
        "cli",
        "config",
        "conformal_shift",
        "dimensional",
        "forecast",
        "hdb5",
        "lawson",
        "predictor",
        "replication",
        "scaling_audit",
        "scaling_law",
        "spectral",
        "storage",
        "neutron_yield",
        "tools",
        "train_model",
        "validation",
    }
)


@pytest.fixture(scope="module")
def wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a real wheel, or skip if this environment cannot build one."""
    if not SOURCE_CARD.exists():
        pytest.skip("No results/predictor.json to package; run `make results`.")
    try:
        import build  # noqa: F401
    except ImportError:
        pytest.skip("`build` is not installed; cannot exercise the distribution.")

    # Build from the state a fresh clone is in, which is the state CI builds in.
    #
    # Two leftovers made these tests pass while the real build was broken.
    # setuptools accumulates into build/lib and never prunes it, so removed
    # modules kept reappearing in the wheel. And `fusionflux/predictor.json` is
    # generated: with a copy left on disk from an earlier build, setuptools
    # packaged it and the suite went green on a tree whose sdist could not
    # actually produce a wheel. A test that only passes on a dirty tree is worse
    # than no test, so both are removed first.
    shutil.rmtree(PROJECT_ROOT / "build", ignore_errors=True)
    (PROJECT_ROOT / "fusionflux" / "predictor.json").unlink(missing_ok=True)

    output = tmp_path_factory.mktemp("wheel")
    completed = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--no-isolation", "-o", str(output)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        pytest.skip(f"wheel build unavailable here:\n{completed.stderr[-2000:]}")

    wheels = list(output.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"
    return wheels[0]


@pytest.fixture(scope="module")
def wheel_names(wheel: Path) -> list[str]:
    with zipfile.ZipFile(wheel) as archive:
        return archive.namelist()


def test_wheel_ships_the_predictor_card(wheel_names: list[str]) -> None:
    """Defect 1: without this file the installed console command cannot start."""
    assert "fusionflux/predictor.json" in wheel_names


def test_packaged_card_is_the_one_under_results(wheel: Path) -> None:
    """The copy is made at build time, so it cannot be an older card than results/."""
    with zipfile.ZipFile(wheel) as archive:
        packaged = json.loads(archive.read("fusionflux/predictor.json"))
    assert packaged == json.loads(SOURCE_CARD.read_text())


def test_wheel_claims_no_generic_top_level_names(wheel_names: list[str]) -> None:
    """Defect 2: installing a fusion study must not redefine `import config`."""
    installed = {name.split("/")[0].removesuffix(".py") for name in wheel_names}
    installed = {name for name in installed if not name.endswith(".dist-info")}
    collisions = installed & RESERVED_TOP_LEVEL_NAMES
    assert not collisions, (
        f"the wheel would shadow {sorted(collisions)} in site-packages; "
        "only `fusionflux` may be installed top level"
    )


def test_wheel_installs_exactly_one_importable_name(wheel_names: list[str]) -> None:
    """`neutron_yield` is excluded too: see the note in pyproject.toml."""
    installed = {name.split("/")[0] for name in wheel_names}
    installed = {name for name in installed if not name.endswith(".dist-info")}
    assert installed == {"fusionflux"}


def test_console_script_points_at_the_moved_cli(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        entry_points = next(
            archive.read(name).decode()
            for name in archive.namelist()
            if name.endswith("entry_points.txt")
        )
    assert "fusionflux = fusionflux.cli:main" in entry_points


def test_build_refuses_when_no_card_exists_anywhere(tmp_path: Path) -> None:
    """A build with no card must fail loudly rather than ship a broken command."""
    import setup as setup_module

    original = (setup_module.SOURCE_CARD, setup_module.PACKAGED_CARD)
    setup_module.SOURCE_CARD = tmp_path / "absent.json"
    setup_module.PACKAGED_CARD = tmp_path / "also-absent.json"
    try:
        command = setup_module.BuildPyWithCard.__new__(setup_module.BuildPyWithCard)
        with pytest.raises(SystemExit, match="cannot read"):
            command.run()
    finally:
        setup_module.SOURCE_CARD, setup_module.PACKAGED_CARD = original


def test_a_plain_build_succeeds_from_a_clean_tree(tmp_path: Path) -> None:
    """`python -m build` with no flags, which is what the release workflow runs.

    This is the exact command that failed in CI while every other test here
    passed. With no flags, `build` produces the sdist and then builds the wheel
    *from that sdist*, in a temporary directory with no `results/` in it. A card
    missing from the sdist therefore fails the wheel with nothing to recover
    from, and hooking only `build_py` left it missing, because `sdist` never
    invokes `build_py`.

    Building only the wheel, as the other tests here do, never exercises that
    path. So this one runs the real command, from a tree with the generated card
    removed.
    """
    if not SOURCE_CARD.exists():
        pytest.skip("No results/predictor.json to package; run `make results`.")
    try:
        import build  # noqa: F401
    except ImportError:
        pytest.skip("`build` is not installed; cannot exercise the distribution.")

    shutil.rmtree(PROJECT_ROOT / "build", ignore_errors=True)
    (PROJECT_ROOT / "fusionflux" / "predictor.json").unlink(missing_ok=True)

    completed = subprocess.run(
        [sys.executable, "-m", "build", "--no-isolation", "-o", str(tmp_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        "`python -m build` failed, so the release workflow would fail:\n"
        + completed.stdout[-3000:]
        + completed.stderr[-3000:]
    )
    (wheel,) = tmp_path.glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        assert "fusionflux/predictor.json" in archive.namelist()


def test_sdist_carries_the_card_so_a_source_build_works(tmp_path: Path) -> None:
    """`pip install --no-binary fusionflux` builds from the sdist.

    The sdist has no `results/` directory, because that holds analysis outputs
    rather than package source. The card it needs must therefore already be
    inside the package when the sdist is written, which is why `sdist` carries
    the same hook `build_py` does.
    """
    if not SOURCE_CARD.exists():
        pytest.skip("No results/predictor.json to package; run `make results`.")
    try:
        import build  # noqa: F401
    except ImportError:
        pytest.skip("`build` is not installed; cannot exercise the distribution.")

    shutil.rmtree(PROJECT_ROOT / "build", ignore_errors=True)
    (PROJECT_ROOT / "fusionflux" / "predictor.json").unlink(missing_ok=True)
    completed = subprocess.run(
        [sys.executable, "-m", "build", "--sdist", "--no-isolation", "-o", str(tmp_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        pytest.skip(f"sdist build unavailable here:\n{completed.stderr[-2000:]}")

    (archive,) = tmp_path.glob("*.tar.gz")
    with tarfile.open(archive) as tar:
        members = tar.getnames()
    assert any(name.endswith("fusionflux/predictor.json") for name in members), (
        "the sdist has no card, so building from source would fail"
    )
