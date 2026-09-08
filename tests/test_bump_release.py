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

import contextlib
import importlib.util
import io
import re
import shutil
from dataclasses import dataclass
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


@dataclass
class Outcome:
    """What the tests need from a run: the exit status and what it complained about."""

    returncode: int
    stderr: str


def _run(sandbox: Path, *args: str) -> Outcome:
    """Drive the module in process, against the sandbox rather than the repository.

    A subprocess would test the same behaviour and report none of it as covered,
    and the sites this edits are the whole point of the module. argparse exits
    through SystemExit, so a refusal arrives here as an exception with the
    message on stderr.
    """
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
            code = bump_release.main(list(args), root=sandbox)
    except SystemExit as exit_:
        # argparse exits with a status; `raise SystemExit("...")` exits with the
        # message itself, which Python prints to stderr on the way out.
        if isinstance(exit_.code, int):
            code = exit_.code
        else:
            code = 1
            if exit_.code is not None:
                print(exit_.code, file=stderr)
    return Outcome(returncode=code, stderr=stderr.getvalue())


def test_the_version_moves_everywhere_at_once(sandbox: Path) -> None:
    assert _run(sandbox, "--version", "9.9.9").returncode == 0
    assert 'version = "9.9.9"' in (sandbox / "pyproject.toml").read_text()
    assert "version: 9.9.9" in (sandbox / "CITATION.cff").read_text()
    paper = (sandbox / "paper" / "paper.tex").read_text()
    assert "(v9.9.9; the DOI for all versions is" in paper
    assert re.search(r"version v9\.9\.9,\s+Zenodo", paper)
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


def test_a_rewrapped_bibitem_still_bumps(sandbox: Path) -> None:
    """The LaTeX sites sit in prose, and prose gets reflowed.

    An earlier pattern spanned "version vX.Y.Z, Zenodo" as one string. Rewrapping
    the bibitem put a line break in the middle of it, and the bump refused to run
    rather than doing four sites out of five, which is the right failure but an
    avoidable one.
    """
    paper = sandbox / "paper" / "paper.tex"
    paper.write_text(paper.read_text().replace("version v0.4.2,\nZenodo", "version v0.4.2, Zenodo"))
    assert _run(sandbox, "--version", "9.9.9").returncode == 0
    assert re.search(r"version v9\.9\.9,\s+Zenodo", paper.read_text())


# --- the release ledger, which is what lets the checker see a half-done bump --


def _ledger(sandbox: Path) -> dict:
    import json

    return json.loads((sandbox / "docs" / "releases.json").read_text())


def test_a_version_bump_records_the_release_with_no_doi_yet(sandbox: Path) -> None:
    """Tagging a release and publishing it are separate acts, days apart.

    Between them the paper names one release and prints the previous release's
    DOI, and both strings are well formed. The ledger is what makes that state
    visible, so a bump has to write the version with a null DOI rather than
    leaving it unrecorded.
    """
    assert _run(sandbox, "--version", "9.9.9").returncode == 0
    assert _ledger(sandbox)["versions"]["9.9.9"] == {"doi": None}


def test_minting_the_doi_fills_in_the_version_the_tree_is_at(sandbox: Path) -> None:
    """`--doi` is run without `--version`, so the version comes from pyproject."""
    assert _run(sandbox, "--version", "9.9.9").returncode == 0
    assert _run(sandbox, "--doi", "10.5281/zenodo.99999999").returncode == 0
    assert _ledger(sandbox)["versions"]["9.9.9"] == {"doi": "10.5281/zenodo.99999999"}


def test_earlier_releases_stay_in_the_ledger(sandbox: Path) -> None:
    """It is a record of which DOI belongs to which release, not of the latest one."""
    _run(sandbox, "--version", "9.9.8")
    _run(sandbox, "--doi", "10.5281/zenodo.88888888")
    _run(sandbox, "--version", "9.9.9")
    versions = _ledger(sandbox)["versions"]
    assert versions["9.9.8"] == {"doi": "10.5281/zenodo.88888888"}
    assert versions["9.9.9"] == {"doi": None}


def test_both_steps_at_once_records_one_complete_entry(sandbox: Path) -> None:
    assert _run(sandbox, "--version", "9.9.9", "--doi", "10.5281/zenodo.77777777").returncode == 0
    assert _ledger(sandbox)["versions"] == {"9.9.9": {"doi": "10.5281/zenodo.77777777"}}
