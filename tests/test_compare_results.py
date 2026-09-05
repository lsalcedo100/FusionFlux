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


# ---------------------------------------------------------------------------
# The non-portable paths.
#
# These exist because results/ was generated on macOS and a Linux runner links
# a different LAPACK, so a handful of values move by far more than REL_TOL
# without anything about the study having changed. An exclusion list is a hole
# in the gate by construction, so what is pinned here is its edges: that it
# covers what it claims to, and that it covers nothing else. The second half
# matters more. A list that quietly swallowed a neighbouring field would make
# the gate report success over a real change, which is the failure this whole
# module exists to prevent.
# ---------------------------------------------------------------------------


def test_the_conditioning_sweeps_measurements_are_excluded(pair: tuple[Path, Path]) -> None:
    """Result 2c measures where float64 breaks down, so it moves with the LAPACK."""
    left, right = pair
    _write(left, "analysis.json", {"solver_conditioning": {"curves": [
        {"median_errors": [1e-11, 0.002], "n_failures": [0, 12], "fitted_slope": 1.9190}]}})
    _write(right, "analysis.json", {"solver_conditioning": {"curves": [
        {"median_errors": [3e-11, 0.004], "n_failures": [0, 11], "fitted_slope": 1.9329}]}})
    assert compare_directories(left, right) == []


def test_the_design_behind_the_sweep_is_still_compared(pair: tuple[Path, Path]) -> None:
    """Only the measurements are unportable. The inputs that produced them are not.

    Without this, the entry above would be excusing the whole analysis, and a
    sweep that silently started measuring a different matrix would pass.
    """
    left, right = pair
    _write(left, "analysis.json", {"solver_conditioning": {
        "n_trials": 40, "curves": [{"solver": "cholesky", "condition_numbers": [1e1, 1e4]}]}})
    _write(right, "analysis.json", {"solver_conditioning": {
        "n_trials": 25, "curves": [{"solver": "cholesky", "condition_numbers": [1e1, 1e6]}]}})
    problems = compare_directories(left, right)
    assert any("n_trials" in p for p in problems), problems
    assert any("condition_numbers" in p for p in problems), problems


def test_the_exclusion_is_by_path_and_not_by_name(pair: tuple[Path, Path]) -> None:
    """``median_errors`` is excused under the sweep and nowhere else."""
    left, right = pair
    _write(left, "conformal.json", {"median_errors": [1e-11]})
    _write(right, "conformal.json", {"median_errors": [3e-11]})
    assert any("median_errors" in p for p in compare_directories(left, right))


def test_the_sweeps_csv_columns_are_excluded_but_its_grid_is_not(pair: tuple[Path, Path]) -> None:
    """The forward error and the breakdown flag move; the grid they were measured on does not."""
    left, right = pair
    pd.DataFrame({"solver": ["cholesky"], "condition_number": [1e9],
                  "relative_forward_error": [1.04], "failed": [True]}).to_csv(
        left / "solver_conditioning.csv", index=False)
    pd.DataFrame({"solver": ["cholesky"], "condition_number": [1e9],
                  "relative_forward_error": [1.48], "failed": [False]}).to_csv(
        right / "solver_conditioning.csv", index=False)
    assert compare_directories(left, right) == []

    pd.DataFrame({"solver": ["cholesky"], "condition_number": [1e10],
                  "relative_forward_error": [1.48], "failed": [False]}).to_csv(
        right / "solver_conditioning.csv", index=False)
    assert any("condition_number" in p for p in compare_directories(left, right))


def test_a_key_appearing_under_an_excluded_path_is_also_excluded(pair: tuple[Path, Path]) -> None:
    """Otherwise the walk reports it as added without ever descending into it."""
    left, right = pair
    _write(left, "analysis.json", {"solver_conditioning": {"curves": [{"median_errors": [1.0]}]}})
    _write(right, "analysis.json", {"solver_conditioning": {"curves": [{}]}})
    assert compare_directories(left, right) == []


def test_the_rank_audit_alignments_are_excluded_but_the_rank_is_not(pair: tuple[Path, Path]) -> None:
    """Result 1 reports the rank, which is an integer. The alignments are a basis choice."""
    left, right = pair
    _write(left, "analysis.json", {"rank_audit": {
        "rank": 8, "rank_deficiency": 2,
        "max_alignment_with_a_printed_basis_vector": {"a = eps * R": 0.9958}}})
    _write(right, "analysis.json", {"rank_audit": {
        "rank": 8, "rank_deficiency": 2,
        "max_alignment_with_a_printed_basis_vector": {"a = eps * R": 0.9991}}})
    assert compare_directories(left, right) == []

    _write(right, "analysis.json", {"rank_audit": {
        "rank": 7, "rank_deficiency": 2,
        "max_alignment_with_a_printed_basis_vector": {"a = eps * R": 0.9991}}})
    assert any("rank" in p for p in compare_directories(left, right))


def test_the_forecast_digest_is_excluded_only_in_the_forecast(pair: tuple[Path, Path]) -> None:
    """It hashes float64 at full precision, so it moves on the jitter REL_TOL absorbs."""
    left, right = pair
    _write(left, "forecast.json", {"content_digest_sha256": "960d537e"})
    _write(right, "forecast.json", {"content_digest_sha256": "611d63ab"})
    assert compare_directories(left, right) == []

    _write(left, "predictor.json", {"content_digest_sha256": "960d537e"})
    _write(right, "predictor.json", {"content_digest_sha256": "611d63ab"})
    assert any("predictor" in p for p in compare_directories(left, right))


def test_the_odr_exponents_are_loosened_rather_than_dropped(pair: tuple[Path, Path]) -> None:
    """An iterative fit lands slightly differently; a changed exponent must still fail."""
    left, right = pair
    _write(left, "sensitivity.json", {"errors_in_variables": {
        "odr_exponents": {"m_eff_amu": -0.13380692886953827}}})
    _write(right, "sensitivity.json", {"errors_in_variables": {
        "odr_exponents": {"m_eff_amu": -0.1338074193068061}}})
    assert compare_directories(left, right) == []

    _write(right, "sensitivity.json", {"errors_in_variables": {
        "odr_exponents": {"m_eff_amu": -0.1341}}})
    assert any("m_eff_amu" in p for p in compare_directories(left, right))


def test_the_shift_the_paper_quotes_keeps_the_full_tolerance(pair: tuple[Path, Path]) -> None:
    """max_abs_exponent_shift is derived from the exponents but is reported, so it stays tight."""
    left, right = pair
    _write(left, "sensitivity.json", {"errors_in_variables": {"max_abs_exponent_shift": 5.606}})
    _write(right, "sensitivity.json", {"errors_in_variables": {"max_abs_exponent_shift": 5.6061}})
    assert any("max_abs_exponent_shift" in p for p in compare_directories(left, right))
