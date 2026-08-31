"""Tests for Result 2c: forward error against condition number.

The claim under test is not "the code runs". It is the textbook statement that
forming the normal equations squares the condition number, and the reason to
measure it on *these* solvers rather than cite it is that a passing measurement
is simultaneously evidence the from-scratch implementations in ``scaling_law``
are correct. So the assertions here are about slopes and orderings:

* ``test_cholesky_slope_is_twice_the_orthogonal_solvers`` is the result. If
  ``solve_lstsq_cholesky`` were quietly delegating to something stable, or if
  ``solve_lstsq_qr`` were secretly forming ``X^T X``, the slopes would not
  separate and this would fail.
* ``test_synthetic_design_has_the_condition_number_it_was_asked_for`` pins the
  instrument. Every other assertion is meaningless if the x-axis is wrong.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import analysis_scaling_law as asl
from scaling_law import _clean_fp_state as clean_fp_state
from storage import _json_safe

# --- the instrument ---------------------------------------------------------


@pytest.mark.parametrize("condition_number", [1e1, 1e4, 1e8, 1e12])
def test_synthetic_design_has_the_condition_number_it_was_asked_for(
    condition_number: float,
) -> None:
    design = asl.synthetic_design(condition_number, seed=3)
    # The tolerance has to grow with kappa: recovering the condition number
    # means an SVD, and that SVD is itself accurate only to about kappa * eps.
    # A fixed tight tolerance would be asserting numpy is more accurate than it
    # can be, not that ``synthetic_design`` is correct.
    tolerance = max(1e-9, 10.0 * condition_number * float(np.finfo(float).eps))
    assert np.linalg.cond(design) == pytest.approx(condition_number, rel=tolerance)


def test_synthetic_design_is_reproducible_and_seed_dependent() -> None:
    first = asl.synthetic_design(1e6, seed=11)
    assert np.array_equal(first, asl.synthetic_design(1e6, seed=11))
    assert not np.array_equal(first, asl.synthetic_design(1e6, seed=12))


def test_synthetic_design_rejects_an_impossible_condition_number() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        asl.synthetic_design(0.5)


def test_synthetic_system_is_consistent_so_forward_error_is_pure_arithmetic() -> None:
    """``y = X b`` exactly, so the exact least-squares answer is ``b`` itself.

    If the constructed system had a residual, the three solvers would be
    compared on a problem whose true solution is only known to the accuracy of
    whichever solver computed it, and the whole experiment would be circular.
    """
    design = asl.synthetic_design(1e3, seed=7)
    rng = np.random.default_rng(7)
    beta = rng.standard_normal(design.shape[1])
    with clean_fp_state():
        target = design @ beta
        fitted = np.linalg.lstsq(design, target, rcond=None)[0]
        residual = target - design @ fitted
    assert np.linalg.norm(residual) < 1e-10 * np.linalg.norm(target)


# --- the result -------------------------------------------------------------


@pytest.fixture(scope="module")
def sweep() -> asl.ConditionSweep:
    # Fewer trials than the published run; the slopes are stable well below the
    # reporting configuration and the suite should not pay for 12 of them.
    return asl.condition_number_sweep(n_trials=5)


def test_cholesky_slope_is_twice_the_orthogonal_solvers(sweep: asl.ConditionSweep) -> None:
    """The headline: kappa^2 for the normal equations, kappa for QR and SVD."""
    slopes = {curve.solver: curve.fitted_slope for curve in sweep.curves}

    assert slopes["cholesky"] == pytest.approx(2.0, abs=0.35)
    assert slopes["qr"] == pytest.approx(1.0, abs=0.35)
    assert slopes["svd"] == pytest.approx(1.0, abs=0.35)
    # The separation is the point, and it is far larger than the tolerances
    # above: a wide margin here is what makes the two bands distinguishable
    # rather than two noisy estimates of the same number.
    assert slopes["cholesky"] > slopes["qr"] + 0.5
    assert slopes["cholesky"] > slopes["svd"] + 0.5


def test_every_curve_reports_the_slope_its_theory_predicts(sweep: asl.ConditionSweep) -> None:
    for curve in sweep.curves:
        assert curve.fitted_slope == pytest.approx(curve.expected_slope, abs=0.35), curve.solver
        # A slope fitted through two points is not a measurement.
        assert curve.n_slope_points >= 4, curve.solver


def test_cholesky_is_the_least_accurate_at_every_condition_number(
    sweep: asl.ConditionSweep,
) -> None:
    """Ordering, not just slope: the squared conditioning has to cost something.

    Slopes could match while the curves sat on top of each other, which would
    mean the fit was picking up noise. This asserts the separation is real at
    every point where all three returned an answer.
    """
    by_solver = {curve.solver: curve for curve in sweep.curves}
    kappa = by_solver["cholesky"].condition_numbers
    for index, condition_number in enumerate(kappa):
        cholesky = by_solver["cholesky"].median_errors[index]
        if cholesky is None:  # refused outright; nothing to compare
            continue
        for stable in ("qr", "svd"):
            other = by_solver[stable].median_errors[index]
            assert other is not None
            if condition_number >= 1e3:
                assert cholesky > other, (condition_number, stable)


def test_cholesky_refuses_before_it_returns_a_meaningless_answer(
    sweep: asl.ConditionSweep,
) -> None:
    """The failure mode is an exception, not a confident wrong number.

    ``solve_lstsq_cholesky`` raises once ``X^T X`` stops being numerically
    positive definite, which is the honest behaviour and is why the curve stops.
    QR and SVD never do over this range.
    """
    breakdown = sweep.breakdown_condition
    assert breakdown["cholesky"] is not None
    assert breakdown["cholesky"] <= 1e11
    assert breakdown["qr"] is None
    assert breakdown["svd"] is None

    cholesky = next(curve for curve in sweep.curves if curve.solver == "cholesky")
    assert sum(cholesky.n_failures) > 0
    for stable in ("qr", "svd"):
        curve = next(c for c in sweep.curves if c.solver == stable)
        assert sum(curve.n_failures) == 0, stable


def test_slope_fit_excludes_the_saturated_and_noise_floor_regimes(
    sweep: asl.ConditionSweep,
) -> None:
    """Only the conditioning-limited band may enter the fit.

    Including the flat top (error at O(1), no digits left) would drag every
    slope down and could make Cholesky look like a first-order method.
    """
    for curve in sweep.curves:
        low, high = curve.slope_fit_range
        if not np.isfinite(low):
            continue
        errors = {
            condition: value
            for condition, value in zip(curve.condition_numbers, curve.median_errors, strict=True)
            if value is not None and low <= condition <= high
        }
        assert errors
        assert max(errors.values()) < asl.SLOPE_FIT_ERROR_CEILING
        assert min(errors.values()) > asl.SLOPE_FIT_ERROR_FLOOR


def test_sweep_serializes_without_non_finite_values(sweep: asl.ConditionSweep) -> None:
    """``write_json_strict`` refuses NaN, so a failed cell must serialize as null."""
    payload = json.dumps(_json_safe(sweep.to_json()), allow_nan=False)
    assert "solver_conditioning" not in payload  # the key is added by the caller
    assert json.loads(payload)["n_trials"] == sweep.n_trials
