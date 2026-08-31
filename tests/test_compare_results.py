"""The reproducibility gate needs its own tests.

``tools/compare_results.py`` decides whether a regenerated ``results/`` still
agrees with the committed one. A comparator that silently passes everything
would make the ``reproduce`` workflow look green while guarding nothing, which
is worse than not having the workflow, so its two failure modes are pinned here:
it must ignore float64 jitter, and it must catch a change large enough to move a
reported digit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tools.compare_results import REL_TOL, compare_directories


@pytest.fixture
def pair(tmp_path: Path) -> tuple[Path, Path]:
    left, right = tmp_path / "baseline", tmp_path / "candidate"
    left.mkdir()
    right.mkdir()
    return left, right


def _write(directory: Path, name: str, payload: object) -> None:
    (directory / name).write_text(json.dumps(payload))


def test_identical_directories_agree(pair: tuple[Path, Path]) -> None:
    left, right = pair
    for d in (left, right):
        _write(d, "a.json", {"rmsle": 0.1277352778940419, "machines": 13})
    assert compare_directories(left, right) == []


def test_float64_jitter_is_ignored(pair: tuple[Path, Path]) -> None:
    """The last digits move between runs; that must not fail the gate."""
    left, right = pair
    _write(left, "a.json", {"rmsle": 0.4502539255431943})
    _write(right, "a.json", {"rmsle": 0.4502539255431945})
    assert compare_directories(left, right) == []


def test_change_that_moves_a_reported_digit_is_caught(pair: tuple[Path, Path]) -> None:
    """0.1% is far below any published precision here and must still fail."""
    left, right = pair
    _write(left, "a.json", {"rmsle": 0.1277352778940419})
    _write(right, "a.json", {"rmsle": 0.1277352778940419 * 1.001})
    differences = compare_directories(left, right)
    assert len(differences) == 1
    assert "rmsle" in differences[0]


def test_tolerance_is_tighter_than_reported_precision() -> None:
    """Guards the choice of tolerance, not just its application.

    Values are reported to at most four significant figures, so a change of
    5e-5 relative can alter a printed digit. The tolerance has to sit below
    that or the gate would pass changes the prose would show.
    """
    assert REL_TOL < 5e-5


def test_volatile_fields_are_excluded(pair: tuple[Path, Path]) -> None:
    """Absolute paths and timings differ by machine, not by analysis."""
    left, right = pair
    _write(left, "a.json", {"provenance": {"path": "/home/a/x.csv", "sha256": "abc"},
                            "seconds_per_solve": 0.00012})
    _write(right, "a.json", {"provenance": {"path": "/runner/b/x.csv", "sha256": "abc"},
                             "seconds_per_solve": 0.00007})
    assert compare_directories(left, right) == []


def test_a_changed_checksum_is_not_excluded(pair: tuple[Path, Path]) -> None:
    """Excluding the path must not accidentally excuse the digest beside it."""
    left, right = pair
    _write(left, "a.json", {"provenance": {"path": "/home/a/x.csv", "sha256": "abc"}})
    _write(right, "a.json", {"provenance": {"path": "/runner/b/x.csv", "sha256": "def"}})
    assert any("sha256" in d for d in compare_directories(left, right))


def test_booleans_do_not_compare_equal_to_numbers(pair: tuple[Path, Path]) -> None:
    """bool subclasses int, so True would otherwise pass as 1.0."""
    left, right = pair
    _write(left, "a.json", {"reversed": True})
    _write(right, "a.json", {"reversed": 1})
    assert compare_directories(left, right) != []


def test_missing_and_added_files_are_reported(pair: tuple[Path, Path]) -> None:
    left, right = pair
    _write(left, "gone.json", {"x": 1})
    _write(right, "new.json", {"x": 1})
    differences = compare_directories(left, right)
    assert any("gone.json" in d and "no longer generated" in d for d in differences)
    assert any("new.json" in d and "not committed" in d for d in differences)


def test_all_missing_csv_column_agrees_with_itself(pair: tuple[Path, Path]) -> None:
    """An all-NaN column reads back as object dtype, where NaN != NaN."""
    left, right = pair
    for d in (left, right):
        pd.DataFrame({"model": ["a", "b"], "correction": [None, None]}).to_csv(
            d / "f.csv", index=False)
    assert compare_directories(left, right) == []


def test_csv_value_change_is_caught(pair: tuple[Path, Path]) -> None:
    left, right = pair
    pd.DataFrame({"model": ["a"], "rmsle": [0.128]}).to_csv(left / "f.csv", index=False)
    pd.DataFrame({"model": ["a"], "rmsle": [0.129]}).to_csv(right / "f.csv", index=False)
    assert any("rmsle" in d for d in compare_directories(left, right))


def test_csv_row_count_change_is_caught(pair: tuple[Path, Path]) -> None:
    left, right = pair
    pd.DataFrame({"model": ["a", "b"]}).to_csv(left / "f.csv", index=False)
    pd.DataFrame({"model": ["a"]}).to_csv(right / "f.csv", index=False)
    assert any("rows" in d for d in compare_directories(left, right))
