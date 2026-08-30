"""Tests for the hand-written least-squares solvers and conditioning analysis.

The three solvers are checked against a problem whose answer is known in closed
form, against each other, and at the boundary where they stop agreeing (rank
deficiency), which is the case the physics actually runs into.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scaling_law import (
    IPB98_FEATURE_COLUMNS,
    analyze_conditioning,
    back_substitution,
    bootstrap_exponents,
    build_log_design_matrix,
    fit_scaling_law,
    forward_substitution,
    ridge_shrinkage_factors,
    solve_constrained_lstsq,
    solve_lstsq_cholesky,
    solve_lstsq_qr,
    solve_lstsq_ridge,
    solve_lstsq_svd,
)

TRUE_EXPONENTS = {
    "ip_ma": 0.90,
    "bt_t": 0.20,
    "ne_line_1e19_m3": 0.40,
    "p_loss_mw": -0.65,
    "r_m": 2.00,
    "inverse_aspect_ratio": 0.60,
    "kappa": 0.75,
    "m_eff_amu": 0.20,
}
TRUE_COEFFICIENT = 0.0562


def make_power_law_frame(
    n_shots: int = 100,
    slices_per_shot: int = 4,
    noise: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """A synthetic dataset obeying a power law exactly, or with correlated noise.

    Rows are grouped into "discharges" the way HDB5 is: each shot contributes
    several quasi-stationary time slices whose engineering parameters differ
    only slightly, and which share a per-shot offset in log tau. That structure
    is what makes row-level resampling wrong, so the fixtures have to reproduce
    it or the bootstrap tests prove nothing.
    """
    rng = np.random.default_rng(seed)
    ranges = {
        "ip_ma": (0.5, 5.0),
        "bt_t": (1.0, 5.0),
        "ne_line_1e19_m3": (1.0, 10.0),
        "p_loss_mw": (1.0, 30.0),
        "r_m": (0.8, 3.5),
        "inverse_aspect_ratio": (0.2, 0.4),
        "kappa": (1.0, 2.0),
        "m_eff_amu": (1.0, 2.5),
    }
    n_rows = n_shots * slices_per_shot
    shot_index = np.repeat(np.arange(n_shots), slices_per_shot)
    columns = {}
    for column, (low, high) in ranges.items():
        per_shot = rng.uniform(low, high, n_shots)
        jitter = 1.0 + rng.normal(0.0, 0.01, n_rows)
        columns[column] = np.clip(per_shot[shot_index] * jitter, low * 0.9, high * 1.1)
    frame = pd.DataFrame(columns)

    log_tau = np.log(TRUE_COEFFICIENT) + sum(
        exponent * np.log(frame[column]) for column, exponent in TRUE_EXPONENTS.items()
    )
    if noise:
        # Two thirds of the scatter is a per-shot offset, one third is per-slice.
        shot_effect = rng.normal(0.0, noise, n_shots)[shot_index]
        log_tau = log_tau + shot_effect + rng.normal(0.0, noise / 2.0, n_rows)
    frame["tau_th_s"] = np.exp(log_tau)
    frame["group_id"] = shot_index
    return frame


# --- Triangular substitution ------------------------------------------------


def test_forward_and_back_substitution_match_a_general_solver():
    rng = np.random.default_rng(1)
    lower = np.tril(rng.normal(size=(6, 6))) + 6.0 * np.eye(6)
    rhs = rng.normal(size=6)
    assert np.allclose(forward_substitution(lower, rhs), np.linalg.solve(lower, rhs))
    upper = lower.T
    assert np.allclose(back_substitution(upper, rhs), np.linalg.solve(upper, rhs))


# --- The three solvers agree --------------------------------------------------


def test_three_solvers_agree_on_a_well_conditioned_problem():
    rng = np.random.default_rng(2)
    design = rng.normal(size=(200, 7))
    target = rng.normal(size=200)
    reference, *_ = np.linalg.lstsq(design, target, rcond=None)
    for solver in (solve_lstsq_cholesky, solve_lstsq_qr, solve_lstsq_svd):
        assert np.allclose(solver(design, target), reference, atol=1e-9)


def test_all_three_solvers_recover_known_exponents_from_noiseless_data():
    frame = make_power_law_frame(noise=0.0)
    for solver in ("cholesky", "qr", "svd"):
        fit = fit_scaling_law(frame, "tau_th_s", solver=solver)
        for column, expected in TRUE_EXPONENTS.items():
            assert fit.exponents[column] == pytest.approx(expected, abs=1e-8), solver
        assert fit.coefficient == pytest.approx(TRUE_COEFFICIENT, rel=1e-8)


def test_solvers_agree_under_realistic_noise():
    frame = make_power_law_frame(noise=0.15, seed=3)
    fits = {
        solver: fit_scaling_law(frame, "tau_th_s", solver=solver).coefficients for solver in ("cholesky", "qr", "svd")
    }
    assert np.allclose(fits["cholesky"], fits["qr"], atol=1e-8)
    assert np.allclose(fits["qr"], fits["svd"], atol=1e-8)


# --- Rank deficiency: where they stop agreeing --------------------------------


def rank_deficient_design(n_rows: int = 200, seed: int = 4) -> np.ndarray:
    """Three independent columns plus two exact linear combinations of them."""
    rng = np.random.default_rng(seed)
    a, b, c = rng.normal(size=(3, n_rows))
    return np.column_stack([a, b, c, a + b + c, a + b])


def test_cholesky_fails_loudly_on_a_rank_deficient_design():
    design = rank_deficient_design()
    target = np.random.default_rng(5).normal(size=design.shape[0])
    with pytest.raises(np.linalg.LinAlgError, match="rank deficient"):
        solve_lstsq_cholesky(design, target)


def test_svd_returns_the_minimum_norm_solution_without_complaining():
    """The failure mode that matters: no error, no warning, arbitrary numbers.

    The SVD solution fits as well as any other, so nothing in a metrics table
    reveals the problem. It is only 'the' answer because the pseudoinverse
    silently picks the shortest vector out of a two-dimensional family.
    """
    design = rank_deficient_design()
    target = np.random.default_rng(6).normal(size=design.shape[0])
    beta = solve_lstsq_svd(design, target)
    reference, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)

    assert rank == 3 < design.shape[1]
    assert np.allclose(beta, reference, atol=1e-9)

    null_basis = np.linalg.svd(design, full_matrices=True)[2][3:]
    shifted = beta + 0.37 * null_basis[0]
    assert np.allclose(design @ shifted, design @ beta, atol=1e-9)  # fits identically
    assert np.linalg.norm(shifted) > np.linalg.norm(beta)  # but is not minimum norm


def test_expected_dependencies_are_found_by_projection_not_by_reading_the_basis():
    """The trap: an arbitrary orthonormal basis will not look like your vector.

    With a two-dimensional null space, SVD returns *some* orthonormal basis for
    it. The expected dependency vectors are in the span but are almost never
    among the printed basis vectors, so the projection residual is the check.
    """
    design = rank_deficient_design()
    report = analyze_conditioning(design, [f"x{i}" for i in range(5)], standardize=False)

    assert report.rank == 3
    assert report.rank_deficiency == 2
    assert np.allclose(report.singular_values[3:], 0.0, atol=1e-10)

    triple = np.array([1.0, 1.0, 1.0, -1.0, 0.0])
    pair = np.array([1.0, 1.0, 0.0, 0.0, -1.0])
    assert report.null_space_residual(triple) < 1e-10
    assert report.null_space_residual(pair) < 1e-10
    assert report.lies_in_null_space(triple)
    assert report.lies_in_null_space(pair)

    # The naive check fails even though the vectors are right: no returned basis
    # vector is parallel to either expected dependency.
    for basis_vector in report.null_space:
        for expected in (triple, pair):
            cosine = abs(basis_vector @ expected / (np.linalg.norm(basis_vector) * np.linalg.norm(expected)))
            assert cosine < 0.999

    unrelated = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    assert not report.lies_in_null_space(unrelated)


def test_unstandardized_rank_is_a_unit_artifact_not_a_dependency():
    """A column carrying huge units can fake a rank deficiency.

    numpy's rank tolerance scales with the largest singular value, so one
    column in units of 1e20 pushes perfectly informative directions below it.
    Standardizing first makes the reported rank a statement about collinearity
    rather than about which SI prefix someone chose.
    """
    rng = np.random.default_rng(7)
    design = rng.normal(size=(200, 4))
    design[:, 0] *= 1e20

    assert np.linalg.matrix_rank(design) < 4
    assert analyze_conditioning(design, list("abcd"), standardize=False).is_rank_deficient
    assert analyze_conditioning(design, list("abcd"), standardize=True).rank == 4


# --- Constrained least squares ------------------------------------------------


def test_constraining_to_the_null_space_complement_reproduces_minimum_norm():
    """Minimum-norm least squares *is* a constrained problem: b orthogonal to null(X)."""
    design = rank_deficient_design()
    target = np.random.default_rng(8).normal(size=design.shape[0])
    report = analyze_conditioning(design, [f"x{i}" for i in range(5)], standardize=False)

    constrained = solve_constrained_lstsq(design, target, report.null_space, np.zeros(report.null_space.shape[0]))
    assert np.allclose(constrained, solve_lstsq_svd(design, target), atol=1e-8)


def test_constrained_fit_satisfies_the_constraint_and_costs_residual():
    rng = np.random.default_rng(9)
    design = rng.normal(size=(150, 4))
    target = rng.normal(size=150)
    constraint = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]])
    rhs = np.array([0.5, -0.25])

    constrained = solve_constrained_lstsq(design, target, constraint, rhs)
    assert np.allclose(constraint @ constrained, rhs, atol=1e-10)

    unconstrained = solve_lstsq_qr(design, target)
    assert np.linalg.norm(design @ constrained - target) >= np.linalg.norm(design @ unconstrained - target)


# --- Ridge as directional shrinkage -------------------------------------------


def test_ridge_at_zero_alpha_is_ordinary_least_squares():
    rng = np.random.default_rng(10)
    design = rng.normal(size=(120, 5))
    target = rng.normal(size=120)
    assert np.allclose(solve_lstsq_ridge(design, target, 0.0), solve_lstsq_qr(design, target))


def test_ridge_shrinkage_factors_follow_the_svd_filter_formula():
    singular_values = np.array([10.0, 1.0, 0.01])
    factors = ridge_shrinkage_factors(singular_values, alpha=1.0)
    assert factors[0] == pytest.approx(100.0 / 101.0)
    assert factors[1] == pytest.approx(0.5)
    assert factors[2] == pytest.approx(1e-4 / (1e-4 + 1.0))
    assert np.all(np.diff(factors) < 0)  # small directions are shrunk hardest


def test_ridge_tames_a_rank_deficient_design_that_cholesky_cannot_touch():
    design = rank_deficient_design()
    target = np.random.default_rng(11).normal(size=design.shape[0])
    ridged = solve_lstsq_ridge(design, target, alpha=1.0)
    assert np.all(np.isfinite(ridged))
    assert np.linalg.norm(ridged) < np.linalg.norm(solve_lstsq_svd(design, target)) * 1.5


# --- Design matrix guards -----------------------------------------------------


def test_design_matrix_refuses_non_positive_values():
    frame = make_power_law_frame(n_shots=5)
    frame.loc[0, "ip_ma"] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        build_log_design_matrix(frame, IPB98_FEATURE_COLUMNS)


def test_design_matrix_reports_missing_columns():
    frame = make_power_law_frame(n_shots=5).drop(columns=["kappa"])
    with pytest.raises(ValueError, match="missing required feature columns"):
        build_log_design_matrix(frame, IPB98_FEATURE_COLUMNS)


# --- Bootstrap ------------------------------------------------------------------


def test_grouped_bootstrap_intervals_cover_the_true_exponents():
    frame = make_power_law_frame(n_shots=150, noise=0.05, seed=12)
    table = bootstrap_exponents(
        frame, "tau_th_s", group_column="group_id", n_resamples=300, confidence=0.99, random_state=1
    )
    covered = table.set_index("variable")
    for column, expected in TRUE_EXPONENTS.items():
        row = covered.loc[column]
        assert row["ci_low"] <= expected <= row["ci_high"], column


def test_grouped_bootstrap_is_wider_than_row_level_bootstrap():
    """Slices from one discharge are not independent observations.

    Resampling rows instead of shots understates the uncertainty; this is the
    single easiest way to publish confidence intervals that are too narrow.
    """
    frame = make_power_law_frame(n_shots=150, noise=0.15, seed=13)
    grouped = bootstrap_exponents(frame, "tau_th_s", group_column="group_id", n_resamples=120, random_state=2)
    rowwise = bootstrap_exponents(frame, "tau_th_s", group_column=None, n_resamples=120, random_state=2)
    grouped_width = (grouped["ci_high"] - grouped["ci_low"]).mean()
    rowwise_width = (rowwise["ci_high"] - rowwise["ci_low"]).mean()
    assert grouped_width > rowwise_width


# --- Coordinates ---------------------------------------------------------------


def test_dependency_vectors_must_be_rescaled_into_standardized_coordinates():
    """Standardizing rescales the null space, and forgetting that reads as failure.

    A dependency ``sum_j c_j x_j = 0`` among raw columns becomes ``c_j * s_j``
    once each column is divided by its standard deviation. Checking the raw
    ``c`` against a standardized null space reports a large residual for a
    dependency that is exactly true, which is indistinguishable from being
    wrong unless you know to look for it.
    """
    rng = np.random.default_rng(20)
    a = rng.normal(0.0, 1.0, 300)
    b = rng.normal(0.0, 1000.0, 300)  # deliberately different units
    design = np.column_stack([a, b, a + b])
    report = analyze_conditioning(design, ["a", "b", "sum"], standardize=True)

    dependency = np.array([1.0, 1.0, -1.0])
    assert report.rank_deficiency == 1
    assert report.null_space_residual(dependency, raw_units=True) < 1e-10
    assert report.lies_in_null_space(dependency, raw_units=True)

    # The same vector read in the wrong coordinate system looks plainly wrong.
    assert report.null_space_residual(dependency, raw_units=False) > 0.1
    assert not report.lies_in_null_space(dependency, raw_units=False)

    rescaled = report.to_analysis_coordinates(dependency)
    assert report.null_space_residual(rescaled, raw_units=False) < 1e-10


def test_null_space_residual_rejects_the_zero_vector():
    design = rank_deficient_design()
    report = analyze_conditioning(design, [f"x{i}" for i in range(5)], standardize=False)
    with pytest.raises(ValueError, match="zero vector"):
        report.null_space_residual(np.zeros(5))
