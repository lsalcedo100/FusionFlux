"""The release bump reaches every place the version and DOI are written.

v0.4.1 was tagged with `pyproject.toml` still reading 0.4.0. The release
workflow's version guard caught it and failed the build, but Zenodo minted a DOI
regardless, because it archives when a release is *published* and does not wait
for the build job to pass. That tag had to be superseded rather than fixed.

`tools/bump_release.py` exists so the five sites move together. These tests run
it against copies, since a test that edited the real files would leave the
repository holding whatever version it last asserted.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "bump_release.py"

_spec = importlib.util.spec_from_file_location("bump_release", SCRIPT)
assert _spec is not None and _spec.loader is not None
bump_release = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bump_release)

TOUCHED = ("pyproject.toml", "CITATION.cff", "paper/paper.tex", "paper/references.bib")


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    for relative in TOUCHED:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    (tmp_path / "tools").mkdir()
    shutil.copy2(SCRIPT, tmp_path / "tools" / "bump_release.py")
    return tmp_path


def _run(sandbox: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "tools/bump_release.py", *args],
        cwd=sandbox,
        capture_output=True,
        text=True,
    )


def test_the_version_moves_everywhere_at_once(sandbox: Path) -> None:
    assert _run(sandbox, "--version", "9.9.9").returncode == 0
    assert 'version = "9.9.9"' in (sandbox / "pyproject.toml").read_text()
    assert "version: 9.9.9" in (sandbox / "CITATION.cff").read_text()
    paper = (sandbox / "paper" / "paper.tex").read_text()
    assert "(v9.9.9; the DOI for all versions is" in paper
    assert "version v9.9.9, Zenodo" in paper
    assert "version   = {v9.9.9}," in (sandbox / "paper" / "references.bib").read_text()
    # Nothing anywhere still says the version it was.
    for relative in TOUCHED:
        assert "0.4.2" not in (sandbox / relative).read_text()


def test_the_doi_moves_in_both_files(sandbox: Path) -> None:
    assert _run(sandbox, "--doi", "10.5281/zenodo.99999999").returncode == 0
    paper = (sandbox / "paper" / "paper.tex").read_text()
    # Two links, the title page and the bibitem. Each spells the DOI twice, once
    # in the URL and once in the text, so four occurrences is the right count.
    assert paper.count(r"\href{https://doi.org/10.5281/zenodo.99999999}") == 2
    assert paper.count("10.5281/zenodo.99999999") == 4
    assert "22562235" not in paper
    assert "doi       = {10.5281/zenodo.99999999}," in (sandbox / "paper" / "references.bib").read_text()


def test_the_rewritten_link_is_still_valid_latex(sandbox: Path) -> None:
    r"""The first version emitted \\href, which sets a literal backslash."""
    _run(sandbox, "--doi", "10.5281/zenodo.99999999")
    paper = (sandbox / "paper" / "paper.tex").read_text()
    assert r"\href{https://doi.org/10.5281/zenodo.99999999}" in paper
    assert r"\\href" not in paper


def test_the_concept_doi_is_left_alone(sandbox: Path) -> None:
    """It addresses every version at once, so it is the one DOI that never moves."""
    _run(sandbox, "--doi", "10.5281/zenodo.99999999")
    assert bump_release.CONCEPT_DOI in (sandbox / "paper" / "paper.tex").read_text()
    assert bump_release.CONCEPT_DOI in (sandbox / "CITATION.cff").read_text()


def test_bumping_to_the_concept_doi_is_refused(sandbox: Path) -> None:
    result = _run(sandbox, "--doi", bump_release.CONCEPT_DOI)
    assert result.returncode != 0
    assert "concept DOI" in result.stderr


@pytest.mark.parametrize("bad", ("v0.4.3", "0.4", "0.4.3-rc1"))
def test_a_malformed_version_is_refused(sandbox: Path, bad: str) -> None:
    result = _run(sandbox, "--version", bad)
    assert result.returncode != 0
    assert (sandbox / "pyproject.toml").read_text() == (ROOT / "pyproject.toml").read_text()


def test_running_it_twice_changes_nothing_the_second_time(sandbox: Path) -> None:
    """The patterns anchor on the field, not on the value they are replacing."""
    _run(sandbox, "--version", "9.9.9", "--doi", "10.5281/zenodo.99999999")
    after_once = {r: (sandbox / r).read_text() for r in TOUCHED}
    assert _run(sandbox, "--version", "9.9.9", "--doi", "10.5281/zenodo.99999999").returncode == 0
    assert {r: (sandbox / r).read_text() for r in TOUCHED} == after_once


def test_a_missing_site_is_an_error_not_a_silent_skip(sandbox: Path) -> None:
    """Half a bump is worse than none: the guard would pass and the paper lie."""
    (sandbox / "pyproject.toml").write_text("[project]\nname = 'fusionflux'\n")
    result = _run(sandbox, "--version", "9.9.9")
    assert result.returncode != 0
    assert "no version site matched" in result.stderr
