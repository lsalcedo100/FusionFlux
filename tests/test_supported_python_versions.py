"""The supported-Python claims in the prose must match what CI actually runs.

``tests/test_reported_numbers.py`` binds every headline *number* to the artifact
it was computed from. This module does the same job for the *support claim*,
which has the same failure mode and no artifact behind it: the README carries a
version badge and a paragraph describing the CI matrix, both typed by hand, and
nothing read either one.

That went stale exactly once. The project moved to ``requires-python = ">=3.10"``
and the CI matrix moved with it, but the badge still advertised 3.9 and the
testing paragraph still described a 3.9 job installing against
``constraints.txt``. CI was green the whole time, because CI does not read the
README. For a repository whose stated discipline is that every claim is bound to
the thing it came from, an unbound and false support claim is the worst kind of
stale: it is in the first screen of the front page, and it is checkable.

So the single source of truth here is ``pyproject.toml``'s ``requires-python``,
and every other statement of the supported range is asserted against it: the CI
matrix, the ruff target, the badge in ``README.md``, and the four places the
testing paragraph in ``docs/testing.md`` names a version. These read only
committed text files, so they need neither the HDB5 download nor a training run.

The badge and the paragraph live in different files on purpose, and the checks
follow them rather than assuming: the badge is a front-page claim and the
paragraph is reference documentation. If either moves again, retarget its
fixture instead of dropping the assertion.

When one fails, change the matrix or ``requires-python`` deliberately, then bring
the prose to it. Do not relax the assertion.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
README = ROOT / "README.md"
TESTING_DOC = ROOT / "docs" / "testing.md"

Version = tuple[int, int]


def _parse(version: str) -> Version:
    major, minor = version.split(".")
    return int(major), int(minor)


def _format(version: Version) -> str:
    return f"{version[0]}.{version[1]}"


@pytest.fixture(scope="module")
def minimum_version() -> Version:
    """The lowest interpreter the project claims to support.

    Read from ``requires-python`` rather than from the matrix, because this is
    the declaration a user installing the package is actually held to.
    """
    text = PYPROJECT.read_text()
    match = re.search(r'^requires-python\s*=\s*"\s*>=\s*(\d+\.\d+)\s*"', text, re.MULTILINE)
    assert match is not None, 'pyproject.toml has no `requires-python = ">=X.Y"` line'
    return _parse(match.group(1))


@pytest.fixture(scope="module")
def ci_versions() -> tuple[Version, ...]:
    """The interpreter versions the CI matrix actually instantiates."""
    text = CI_WORKFLOW.read_text()
    match = re.search(r"^\s*python-version:\s*\[(.+?)\]", text, re.MULTILINE)
    assert match is not None, "ci.yml has no inline `python-version: [...]` matrix"
    versions = tuple(_parse(v) for v in re.findall(r"\d+\.\d+", match.group(1)))
    assert versions, "the CI matrix parsed to an empty version list"
    return versions


def _collapsed(path: Path) -> str:
    """File text with runs of whitespace collapsed, so a claim survives wrapping."""
    return re.sub(r"\s+", " ", path.read_text())


@pytest.fixture(scope="module")
def readme() -> str:
    return _collapsed(README)


@pytest.fixture(scope="module")
def testing_doc() -> str:
    return _collapsed(TESTING_DOC)


def test_ci_matrix_starts_at_the_declared_minimum(
    minimum_version: Version, ci_versions: tuple[Version, ...]
) -> None:
    """A `>=X.Y` that no job runs is an untested claim, however plausible."""
    assert min(ci_versions) == minimum_version


def test_ci_matrix_has_no_gaps(ci_versions: tuple[Version, ...]) -> None:
    """The badge and the paragraph both write the matrix as a range `A - B`.

    That spelling is only honest if every minor version between the endpoints is
    actually built, so assert the matrix is contiguous rather than letting a
    dropped middle version hide inside a range.
    """
    major = min(ci_versions)[0]
    low, high = min(ci_versions)[1], max(ci_versions)[1]
    expected = tuple((major, minor) for minor in range(low, high + 1))
    assert tuple(sorted(ci_versions)) == expected


def test_ruff_target_matches_the_declared_minimum(minimum_version: Version) -> None:
    """ruff's target gates which syntax it accepts, so a stale one lints too loosely.

    This is the check that would have caught the migration at the point it was
    made: `zip(..., strict=True)` is 3.10+, and a `py39` target is what makes it
    a lint error rather than a silent runtime failure on the older interpreter.
    """
    text = PYPROJECT.read_text()
    match = re.search(r'^target-version\s*=\s*"py(\d)(\d+)"', text, re.MULTILINE)
    assert match is not None, "pyproject.toml has no `[tool.ruff] target-version`"
    assert (int(match.group(1)), int(match.group(2))) == minimum_version


def test_ci_pins_constraints_on_the_lowest_version_job(
    minimum_version: Version, ci_versions: tuple[Version, ...]
) -> None:
    """CI branches on a hardcoded version to pick the constraints-pinned job.

    If the matrix moves off that literal the branch never fires, and every job
    silently resolves current releases. The pinned environment is the one every
    number under `results/` was generated in, so losing it loses the thing that
    makes `reproduce` mean anything, without turning CI red.
    """
    text = CI_WORKFLOW.read_text()
    pinned = tuple(_parse(v) for v in re.findall(r'matrix\.python-version\s*}}"\s*=\s*"(\d+\.\d+)"', text))
    assert pinned, "ci.yml no longer branches on a specific matrix.python-version"
    assert set(pinned) == {minimum_version}
    assert minimum_version in ci_versions


def test_readme_badge_states_the_supported_range(readme: str, ci_versions: tuple[Version, ...]) -> None:
    """The badge is the first version claim a reader sees, in both its parts.

    shields.io takes the label from the URL, so the alt text and the URL are two
    independent copies of the range and either can rot alone.
    """
    low, high = _format(min(ci_versions)), _format(max(ci_versions))
    assert f"[![Python {low} - {high}]" in readme
    assert f"/badge/python-{low}%20--%20{high}-blue.svg" in readme


def test_testing_doc_paragraph_states_the_supported_range(
    testing_doc: str, minimum_version: Version, ci_versions: tuple[Version, ...]
) -> None:
    """The testing paragraph names the range, the job count and the pinned job.

    Three separate facts, all typed by hand, all derivable from the matrix. The
    en dash is the spelling the paragraph uses; the badge uses a hyphen.
    """
    low, high = _format(min(ci_versions)), _format(max(ci_versions))
    count = {2: "two", 3: "three", 4: "four", 5: "five"}[len(ci_versions)]

    assert f"CI runs the same gate on Python {low}–{high}" in testing_doc
    assert f"so the {count} interpreters agree" in testing_doc
    assert f"The {_format(minimum_version)} job installs against `constraints.txt`" in testing_doc
    assert f"the `>={_format(minimum_version)}` support claim is actually exercised" in testing_doc


@pytest.mark.parametrize("document", [README, TESTING_DOC], ids=lambda p: p.name)
def test_no_document_advertises_an_unsupported_version(
    document: Path, minimum_version: Version, ci_versions: tuple[Version, ...]
) -> None:
    """Catch a stray version claim the structured checks above do not cover.

    Scoped to `Python X.Y` and `>=X.Y`, so it reads version claims rather than
    every decimal in the document: the results prose is full of numbers like
    "3.9x above anything any tree can output" that have nothing to do with an
    interpreter.
    """
    supported = {_format(v) for v in ci_versions}
    claimed = set(re.findall(r"(?:Python|>=)\s*(\d+\.\d+)", _collapsed(document)))
    unsupported = claimed - supported - {_format(minimum_version)}
    assert not unsupported, f"{document.name} names unsupported Python version(s): {sorted(unsupported)}"
