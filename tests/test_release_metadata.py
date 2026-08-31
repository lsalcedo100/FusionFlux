"""The release metadata has to agree with the repository it describes.

`.zenodo.json` is read by Zenodo when a GitHub release is published, and the
record it mints is permanent: a correction means a new version, not an edit. It
is also metadata nothing else in the repository consumes, which is exactly the
kind of file that drifts unnoticed. A second copy under `paper/` had already
diverged in wording before it was deleted.

So the fields that have a counterpart elsewhere are checked against it here:
the author against CITATION.cff, and the licence against CITATION.cff and
LICENSE. The numbers inside the description are covered separately, by
tests/test_reported_numbers.py, which treats it as one more document.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ZENODO = ROOT / ".zenodo.json"
CITATION = ROOT / "CITATION.cff"
LICENSE = ROOT / "LICENSE"

# Zenodo rejects a record missing any of these, and it does so after the release
# is published, where the failure is easy to miss.
REQUIRED_FIELDS = ("title", "upload_type", "description", "creators", "license", "access_right")


@pytest.fixture(scope="module")
def zenodo() -> dict:
    return json.loads(ZENODO.read_text())


def test_zenodo_record_is_valid_json_with_the_required_fields(zenodo: dict) -> None:
    missing = [f for f in REQUIRED_FIELDS if f not in zenodo]
    assert not missing, f".zenodo.json is missing {missing}, which Zenodo requires"


def test_every_creator_is_named(zenodo: dict) -> None:
    assert zenodo["creators"], ".zenodo.json lists no creators"
    for creator in zenodo["creators"]:
        assert creator.get("name"), f"creator entry has no name: {creator}"


def test_author_matches_citation_cff(zenodo: dict) -> None:
    """CITATION.cff splits the name; .zenodo.json uses 'Family, Given'."""
    cff = CITATION.read_text()
    given = re.search(r"given-names:\s*(.+)", cff)
    family = re.search(r"family-names:\s*(.+)", cff)
    assert given and family, "CITATION.cff has no given-names/family-names"

    expected = f"{family.group(1).strip()}, {given.group(1).strip()}"
    names = [c["name"] for c in zenodo["creators"]]
    assert expected in names, (
        f"CITATION.cff names {expected!r} but .zenodo.json lists {names}. "
        f"A Zenodo record is permanent, so these must agree before a release."
    )


def test_licence_agrees_across_the_three_files(zenodo: dict) -> None:
    declared = zenodo["license"]
    # Zenodo accepts either a bare identifier or {"id": ...}.
    identifier = declared["id"] if isinstance(declared, dict) else declared

    assert "license: MIT" in CITATION.read_text(), "CITATION.cff no longer says MIT"
    assert "MIT License" in LICENSE.read_text(), "LICENSE is no longer the MIT licence"
    assert identifier.upper() == "MIT", (
        f".zenodo.json declares {identifier!r} while CITATION.cff and LICENSE say MIT"
    )


def test_the_dataset_is_not_claimed_under_the_repository_licence(zenodo: dict) -> None:
    """The HDB5 data is third-party and is not redistributed here.

    The record is what a reader sees before the repository, so the description
    has to carry that caveat rather than leaving the MIT licence to imply the
    data comes with it.
    """
    description = zenodo["description"].lower()
    assert "not redistributed" in description or "third-party" in description, (
        ".zenodo.json no longer states that the HDB5 dataset is third-party and "
        "not redistributed, which the MIT licence would otherwise appear to cover"
    )


def test_there_is_exactly_one_zenodo_record() -> None:
    """Zenodo only ever reads the one at the repository root.

    A second copy elsewhere is not wired to anything, cannot be published, and
    drifts from the real one; that is what happened to paper/.zenodo.json.
    """
    copies = [p for p in ROOT.rglob(".zenodo.json") if ".venv" not in p.parts]
    assert copies == [ZENODO], (
        f"expected only {ZENODO}, found {copies}. Zenodo reads the root file "
        f"only, so any other copy is dead metadata that will drift."
    )
