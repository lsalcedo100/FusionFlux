"""Reproduce the three scaling-law results on the real ITPA HDB5 database.

Run ``python3 analysis_scaling_law.py`` to regenerate everything under
``results/``: the rank audit, the refit of IPB98(y,2) from data, the singular
value spectrum, and the comparison against the published scaling law.

    Result 1  The feature matrix the confinement model is trained on is rank
              deficient by exactly two, and both dependencies are identifiable.
    Result 2  Refit the IPB98(y,2) exponents from HDB5 with three independently
              implemented solvers and compare against the published values.
    Result 3  Show that the disagreement with the published law lives almost
              entirely in the directions the database cannot determine.

The narrative built on these numbers is in ``results/RESULTS.md``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

import hdb5
from scaling_law import (
    IPB98_FEATURE_COLUMNS,
    IPB98Y2_COEFFICIENT,
    IPB98Y2_COEFFICIENT_ROUNDED,
    IPB98Y2_EXPONENTS,
    analyze_conditioning,
    bootstrap_exponents,
    build_log_design_matrix,
    fit_scaling_law,
    ridge_shrinkage_factors,
    solve_lstsq_cholesky,
    solve_lstsq_qr,
    solve_lstsq_svd,
)
from scaling_law import _clean_fp_state as clean_fp_state
from storage import write_dataframe_csv_atomic, write_json_strict

RESULTS_DIR = Path(__file__).resolve().parent / "results"

Solver = Callable[[np.ndarray, np.ndarray], np.ndarray]

ASPECT_RATIO_IDENTITY = "a = eps * R"
IPB98_PRIOR_IDENTITY = "log IPB98 prior = sum of exponent-weighted logs"
CONTROL_VECTOR = "control (log_ip_ma alone)"


# --- Result 1: the rank audit -----------------------------------------------


@dataclass(frozen=True)
class RankAudit:
    columns: list[str]
    n_rows: int
    n_columns: int
    rank: int
    rank_deficiency: int
    condition_number: float
    singular_values: np.ndarray = field(repr=False)
    tolerance: float
    projection_residuals: dict[str, float]
    basis_alignments: dict[str, float]
    unstandardized_rank: int

    def to_json(self) -> dict[str, object]:
        return {
            "columns": self.columns,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "rank": self.rank,
            "rank_deficiency": self.rank_deficiency,
            "condition_number": self.condition_number,
            "singular_values": self.singular_values.tolist(),
            "tolerance": self.tolerance,
            "projection_residuals": self.projection_residuals,
            "max_alignment_with_a_printed_basis_vector": self.basis_alignments,
            "unstandardized_rank": self.unstandardized_rank,
        }


def audit_model_feature_matrix(dataset: pd.DataFrame) -> RankAudit:
    """Audit the log-feature matrix the confinement model is actually trained on.

    Two exact dependencies are expected, and they differ in kind:

    1. ``log a = log R + log eps``, because minor radius is *defined* as
       ``a = eps * R`` in the cleaning step. A definitional identity restated as
       a feature.

    2. ``log_ipb98y2_tau_s`` is the log of a power law in the other eight
       features, so it is their fixed linear combination plus a constant. The
       IPB98 prior carries no information a log-linear model did not already
       have. It is genuinely useful to the tree models, which cannot form that
       combination themselves, and exactly zero new information to the linear
       one sitting beside them in the same comparison.

    The second is the interesting failure: adding a published physics scaling as
    a feature feels like adding knowledge, and in log space it provably is not.
    """
    columns = list(hdb5.MODEL_FEATURE_COLUMNS)
    matrix = dataset.loc[:, columns].to_numpy(dtype=float)
    report = analyze_conditioning(matrix, columns, standardize=True)
    index = {name: position for position, name in enumerate(columns)}

    aspect_ratio = np.zeros(len(columns))
    aspect_ratio[index["log_a_m"]] = 1.0
    aspect_ratio[index["log_r_m"]] = -1.0
    aspect_ratio[index["log_inverse_aspect_ratio"]] = -1.0

    ipb98_prior = np.zeros(len(columns))
    ipb98_prior[index["log_ipb98y2_tau_s"]] = 1.0
    for variable, exponent in IPB98Y2_EXPONENTS.items():
        ipb98_prior[index[f"log_{variable}"]] = -exponent

    control = np.zeros(len(columns))
    control[index["log_ip_ma"]] = 1.0

    def max_alignment(vector: np.ndarray) -> float:
        """How parallel the closest *printed* basis vector is to what we expect.

        Near 1 means the naive check would have happened to work here. It is
        luck, not method: with a null space of dimension greater than one the
        returned basis is arbitrary.
        """
        scaled = report.to_analysis_coordinates(vector)
        return max(
            float(abs(float(basis @ scaled) / float(np.linalg.norm(basis) * np.linalg.norm(scaled))))
            for basis in report.null_space
        )

    return RankAudit(
        columns=columns,
        n_rows=int(matrix.shape[0]),
        n_columns=report.n_columns,
        rank=report.rank,
        rank_deficiency=report.rank_deficiency,
        condition_number=report.condition_number,
        singular_values=report.singular_values,
        tolerance=report.tolerance,
        projection_residuals={
            ASPECT_RATIO_IDENTITY: report.null_space_residual(aspect_ratio, raw_units=True),
            IPB98_PRIOR_IDENTITY: report.null_space_residual(ipb98_prior, raw_units=True),
            CONTROL_VECTOR: report.null_space_residual(control, raw_units=True),
        },
        basis_alignments={
            ASPECT_RATIO_IDENTITY: max_alignment(aspect_ratio),
            IPB98_PRIOR_IDENTITY: max_alignment(ipb98_prior),
        },
        unstandardized_rank=int(np.linalg.matrix_rank(matrix)),
    )


# --- Result 2: refit IPB98 ---------------------------------------------------


@dataclass(frozen=True)
class SolverTiming:
    name: str
    seconds_per_solve: float
    max_deviation_from_svd: float


# --- Result 2b: what the bootstrap is allowed to resample ---------------------
#
# The exponent intervals are only as honest as their resampling unit, and the
# unit is a modelling choice rather than a detail. This database is not a sample
# of independent measurements: 6228 rows come from 18 machine labels that are 16
# physical devices, and JET and ASDEX Upgrade together supply 77% of the rows.
# Three nested units are defensible, each answering a different question:
#
#   discharge  slices move with their shot. Answers "another shot on these
#              machines", which is the interpolation question.
#   machine    whole ``TOK`` labels move together. Answers "another tokamak",
#              but counts JET and JET-with-the-ITER-like-wall as two draws.
#   device     wall variants fold back onto one device (``hdb5.with_device_column``).
#              Answers "another tokamak" without counting JET twice.
#
# Row-level resampling is not offered at all: a discharge contributes several
# quasi-stationary slices that are near-copies, so treating rows as independent
# returns intervals several times too narrow, and there is no question it
# correctly answers.
#
# A scaling law exists to make claims about tokamaks in general, so the device
# interval is the one that matches the claim, and it is necessarily the widest:
# resampling devices can drop JET entirely, and 16 units resampled with
# replacement is a far smaller effective sample than 4471 discharges. All three
# are reported. Publishing only the narrowest would be quoting the uncertainty
# of a question nobody asked.
MACHINE_BOOTSTRAP_RESAMPLES = 1000

# Ordered coarsest-last; the first entry is the baseline every widening factor
# is measured against.
BOOTSTRAP_LEVELS: tuple[tuple[str, str], ...] = (
    ("discharge", hdb5.GROUP_COLUMN),
    ("machine", hdb5.TOKAMAK_LABEL_COLUMN),
    ("device", hdb5.DEVICE_COLUMN),
)
BASELINE_LEVEL = BOOTSTRAP_LEVELS[0][0]


@dataclass(frozen=True)
class BootstrapLevel:
    """One resampling unit's exponent intervals."""

    name: str
    group_column: str
    n_units: int
    intervals: pd.DataFrame = field(repr=False)


@dataclass(frozen=True)
class ResolutionWidth:
    """One exponent's interval under each resampling unit."""

    variable: str
    fitted: float
    published_ipb98y2: float
    # level name -> (low, high). Kept as a mapping rather than one field per
    # level so adding a unit does not mean touching this class.
    bounds: dict[str, tuple[float, float]]
    # level name -> width / baseline width. The baseline entry is 1.0.
    widening_factors: dict[str, float]
    published_inside: dict[str, bool]

    def width(self, level: str) -> float:
        low, high = self.bounds[level]
        return high - low

    def to_json(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "variable": self.variable,
            "fitted": self.fitted,
            "published_ipb98y2": self.published_ipb98y2,
        }
        for level, (low, high) in self.bounds.items():
            payload[f"{level}_ci_low"] = low
            payload[f"{level}_ci_high"] = high
            payload[f"{level}_width"] = high - low
            payload[f"{level}_widening_factor"] = self.widening_factors[level]
            payload[f"published_inside_{level}_ci"] = self.published_inside[level]
        return payload


@dataclass(frozen=True)
class BootstrapResolutionComparison:
    """The same fit resampled at every unit, lined up exponent by exponent."""

    levels: list[BootstrapLevel]
    n_resamples: int
    # Share of rows contributed by the two largest devices, the reason the
    # resolutions differ as much as they do.
    largest_two_devices: list[str]
    largest_two_row_share: float
    widths: list[ResolutionWidth]

    @property
    def level_names(self) -> list[str]:
        return [level.name for level in self.levels]

    def units(self, level: str) -> int:
        return next(entry.n_units for entry in self.levels if entry.name == level)

    def median_widening(self, level: str) -> float:
        return float(np.median([width.widening_factors[level] for width in self.widths]))

    def max_widening(self, level: str) -> ResolutionWidth:
        return max(self.widths, key=lambda width: width.widening_factors[level])

    @property
    def comparable_widths(self) -> list[ResolutionWidth]:
        """Exponents that can be compared against a published value at all.

        The intercept is excluded: the ITER Physics Basis quotes a multiplying
        coefficient rather than a log-intercept, so ``published_ipb98y2`` is NaN
        for that row and every containment test on it is false by default.
        Counting it would report "7 of 9" for something that is 7 of 8, which
        understates the agreement by a whole exponent.
        """
        return [width for width in self.widths if np.isfinite(width.published_ipb98y2)]

    def n_published_inside(self, level: str) -> int:
        return int(sum(width.published_inside[level] for width in self.comparable_widths))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([width.to_json() for width in self.widths])

    # ``Any`` rather than ``object``: this is the whole serialized payload, so
    # its values are nested lists and dicts that callers index into. Declaring
    # ``object`` makes ``payload["widths"]`` unusable without a cast at every
    # call site, which is noise rather than safety.
    def to_json(self) -> dict[str, Any]:
        return {
            "n_resamples": self.n_resamples,
            "baseline_level": BASELINE_LEVEL,
            "levels": [
                {
                    "name": level.name,
                    "group_column": level.group_column,
                    "n_units": level.n_units,
                    "median_widening_factor": self.median_widening(level.name),
                    "max_widening_variable": self.max_widening(level.name).variable,
                    "max_widening_factor": self.max_widening(level.name).widening_factors[
                        level.name
                    ],
                    "n_published_inside_ci": self.n_published_inside(level.name),
                    "n_comparable_exponents": len(self.comparable_widths),
                }
                for level in self.levels
            ],
            "largest_two_devices": self.largest_two_devices,
            "largest_two_row_share": self.largest_two_row_share,
            "n_exponents": len(self.widths),
            "widths": [width.to_json() for width in self.widths],
        }


def bootstrap_every_resolution(
    dataset: pd.DataFrame,
    *,
    n_resamples: int = 1000,
    n_coarse_resamples: int = MACHINE_BOOTSTRAP_RESAMPLES,
) -> list[BootstrapLevel]:
    """Run the identical percentile bootstrap once per resampling unit.

    Same estimator, same percentile method, same feature set. The only thing
    that changes between levels is what the bootstrap is allowed to treat as
    exchangeable, which is precisely the assumption under test.
    """
    framed = hdb5.with_device_column(dataset)
    levels: list[BootstrapLevel] = []
    for name, group_column in BOOTSTRAP_LEVELS:
        levels.append(
            BootstrapLevel(
                name=name,
                group_column=group_column,
                n_units=int(framed[group_column].nunique()),
                intervals=bootstrap_exponents(
                    framed,
                    hdb5.TARGET_COLUMN,
                    IPB98_FEATURE_COLUMNS,
                    group_column=group_column,
                    n_resamples=n_resamples if name == BASELINE_LEVEL else n_coarse_resamples,
                ),
            )
        )
    return levels


def compare_bootstrap_units(
    dataset: pd.DataFrame, levels: list[BootstrapLevel]
) -> BootstrapResolutionComparison:
    """Line the resamplings up per exponent and measure how much wider each is."""
    indexed = {level.name: level.intervals.set_index("variable") for level in levels}
    baseline = indexed[BASELINE_LEVEL]

    widths: list[ResolutionWidth] = []
    for variable in baseline.index:
        published = float(baseline.loc[variable, "published_ipb98y2"])
        bounds: dict[str, tuple[float, float]] = {}
        inside: dict[str, bool] = {}
        for name, table in indexed.items():
            low = float(table.loc[variable, "ci_low"])
            high = float(table.loc[variable, "ci_high"])
            bounds[name] = (low, high)
            inside[name] = bool(low <= published <= high)
        baseline_width = bounds[BASELINE_LEVEL][1] - bounds[BASELINE_LEVEL][0]
        widths.append(
            ResolutionWidth(
                variable=str(variable),
                fitted=float(baseline.loc[variable, "fitted"]),
                published_ipb98y2=published,
                bounds=bounds,
                # A zero-width baseline would be a degenerate fit, not a division
                # worth reporting; guard rather than emit an inf.
                widening_factors={
                    name: float((high - low) / baseline_width)
                    if baseline_width > 0
                    else float("nan")
                    for name, (low, high) in bounds.items()
                },
                published_inside=inside,
            )
        )

    counts = hdb5.with_device_column(dataset)[hdb5.DEVICE_COLUMN].value_counts()
    top_two = counts.head(2)
    return BootstrapResolutionComparison(
        levels=levels,
        n_resamples=MACHINE_BOOTSTRAP_RESAMPLES,
        largest_two_devices=[str(name) for name in top_two.index],
        largest_two_row_share=float(top_two.sum() / len(dataset)),
        widths=widths,
    )


@dataclass(frozen=True)
class Refit:
    design_shape: tuple[int, int]
    solvers: list[SolverTiming]
    fitted_coefficient: float
    residual_std_log: float
    condition_number: float
    rmsle_refit: float
    rmsle_published: float
    # Discharge-level intervals: the resampling unit is one shot, so time slices
    # from the same shot travel together. This is the headline table, and it is
    # the *narrowest* of the three; ``resolution`` carries the coarser units.
    intervals: pd.DataFrame = field(repr=False)
    resolution: "BootstrapResolutionComparison | None" = None

    def to_json(self) -> dict[str, object]:
        return {
            "design_matrix_shape": list(self.design_shape),
            "solvers": [
                {
                    "name": solver.name,
                    "seconds_per_solve": solver.seconds_per_solve,
                    "max_deviation_from_svd": solver.max_deviation_from_svd,
                }
                for solver in self.solvers
            ],
            "fitted_coefficient": self.fitted_coefficient,
            "published_coefficient": IPB98Y2_COEFFICIENT,
            "published_coefficient_rounded_variant": IPB98Y2_COEFFICIENT_ROUNDED,
            "residual_std_log": self.residual_std_log,
            "condition_number": self.condition_number,
            "in_sample_rmsle_refit": self.rmsle_refit,
            "in_sample_rmsle_published_ipb98y2": self.rmsle_published,
            "bootstrap_resolution": self.resolution.to_json() if self.resolution else None,
        }


def refit_ipb98(
    dataset: pd.DataFrame,
    *,
    n_resamples: int = 1000,
    n_machine_resamples: int = MACHINE_BOOTSTRAP_RESAMPLES,
) -> Refit:
    design, _ = build_log_design_matrix(dataset, IPB98_FEATURE_COLUMNS)
    target = np.log(dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float))

    solvers: dict[str, Solver] = {
        "cholesky": solve_lstsq_cholesky,
        "qr": solve_lstsq_qr,
        "svd": solve_lstsq_svd,
    }
    solutions: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}
    for name, solver in solvers.items():
        start = time.perf_counter()
        for _ in range(20):
            beta = solver(design, target)
        timings[name] = (time.perf_counter() - start) / 20.0
        solutions[name] = beta

    reference = solutions["svd"]
    solver_rows = [
        SolverTiming(
            name=name,
            seconds_per_solve=timings[name],
            max_deviation_from_svd=float(np.max(np.abs(solutions[name] - reference))),
        )
        for name in solvers
    ]

    fit = fit_scaling_law(dataset, hdb5.TARGET_COLUMN, IPB98_FEATURE_COLUMNS)
    levels = bootstrap_every_resolution(
        dataset, n_resamples=n_resamples, n_coarse_resamples=n_machine_resamples
    )
    resolution = compare_bootstrap_units(dataset, levels)
    intervals = next(
        level.intervals for level in levels if level.name == BASELINE_LEVEL
    )

    actual = dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float)

    def rmsle(prediction: np.ndarray) -> float:
        return float(np.sqrt(np.mean((np.log(prediction) - np.log(actual)) ** 2)))

    return Refit(
        design_shape=(int(design.shape[0]), int(design.shape[1])),
        solvers=solver_rows,
        fitted_coefficient=fit.coefficient,
        residual_std_log=fit.residual_std_log,
        condition_number=fit.conditioning.condition_number,
        rmsle_refit=rmsle(fit.predict(dataset, IPB98_FEATURE_COLUMNS)),
        rmsle_published=rmsle(dataset["ipb98y2_tau_s"].to_numpy(dtype=float)),
        intervals=intervals,
        resolution=resolution,
    )


# --- Result 2c: the price of the normal equations, measured ------------------
#
# The three solvers agree to ~1e-12 on the HDB5 design matrix (Result 2), whose
# condition number is about 500. That agreement is not evidence they are
# interchangeable; it is evidence that this particular matrix is easy. The
# textbook statement is that forming ``X^T X`` squares the condition number, so
# Cholesky's forward error grows like ``kappa^2 * eps`` where QR and SVD grow
# like ``kappa * eps``. Asserting that in a comment is cheap. This measures it.
#
# The experiment needs matrices of *known* conditioning, which real data cannot
# supply, so they are constructed: draw random orthonormal ``U`` and ``V``, place
# singular values geometrically between 1 and 1/kappa, and set ``X = U S V^T``.
# The right-hand side is ``y = X b_true`` exactly, with no noise and no residual,
# so the system is consistent and every solver has the *same* exact answer to
# find. Whatever separates them is arithmetic, not statistics.
#
# What this buys beyond the textbook: the three solvers under test are the ones
# in ``scaling_law.py``, written here rather than called from a library. If the
# measured slopes come out at 2, 1 and 1, that is simultaneously a check that
# the implementations are correct and a demonstration of why the choice matters.
CONDITION_NUMBERS = tuple(10.0**exponent for exponent in range(1, 13))
CONDITION_SWEEP_ROWS = 200
CONDITION_SWEEP_COLUMNS = 12
CONDITION_SWEEP_TRIALS = 12
CONDITION_SWEEP_SEED = 20240617

# Fit the slope only where the error is genuinely conditioning-limited. Below
# the floor it is rounding noise in ``b_true`` itself; above the ceiling the
# solution has lost every significant digit and the curve flattens at O(1),
# which would bias any slope fitted through it downward.
SLOPE_FIT_ERROR_FLOOR = 1e-15
SLOPE_FIT_ERROR_CEILING = 1e-3


def synthetic_design(
    condition_number: float,
    *,
    n_rows: int = CONDITION_SWEEP_ROWS,
    n_columns: int = CONDITION_SWEEP_COLUMNS,
    seed: int = 0,
) -> np.ndarray:
    """A random matrix with an exactly prescribed condition number.

    Built from its own SVD: orthonormal factors from QR of Gaussian matrices,
    singular values geometrically spaced from 1 down to ``1 / condition_number``.
    Geometric rather than linear spacing so the small directions are populated;
    linear spacing puts almost every singular value near the top and the matrix
    behaves better than its nominal conditioning suggests.
    """
    if condition_number < 1.0:
        raise ValueError("condition_number must be at least 1.")
    rng = np.random.default_rng(seed)
    left, _ = np.linalg.qr(rng.standard_normal((n_rows, n_columns)))
    right, _ = np.linalg.qr(rng.standard_normal((n_columns, n_columns)))
    singular_values = np.geomspace(1.0, 1.0 / condition_number, n_columns)
    with clean_fp_state():
        return (left * singular_values) @ right.T


@dataclass(frozen=True)
class SolverErrorCurve:
    """One solver's forward error as a function of condition number."""

    solver: str
    condition_numbers: list[float]
    # Median relative forward error over the trials at each condition number.
    # ``None`` where every trial at that kappa was refused by the solver.
    median_errors: list[float | None]
    # Trials at each condition number the solver refused outright, as Cholesky
    # does once ``X^T X`` stops being numerically positive definite. A refusal is
    # a better outcome than a confident wrong answer and is counted, not hidden.
    n_failures: list[int]
    # Slope of log10(error) against log10(kappa) over the conditioning-limited
    # band. The theoretical values are 2 for the normal equations, 1 otherwise.
    fitted_slope: float
    expected_slope: float
    n_slope_points: int
    slope_fit_range: tuple[float, float]

    def to_json(self) -> dict[str, object]:
        return {
            "solver": self.solver,
            "condition_numbers": self.condition_numbers,
            "median_errors": self.median_errors,
            "n_failures": self.n_failures,
            "fitted_slope": self.fitted_slope,
            "expected_slope": self.expected_slope,
            "n_slope_points": self.n_slope_points,
            "slope_fit_range": list(self.slope_fit_range),
        }


@dataclass(frozen=True)
class ConditionSweep:
    """The full kappa sweep: one error curve per solver, plus the raw trials."""

    n_rows: int
    n_columns: int
    n_trials: int
    curves: list[SolverErrorCurve]
    # kappa at which each solver first loses every significant digit
    # (median relative error >= 1). None if it never does over the sweep.
    breakdown_condition: dict[str, float | None]
    trials: pd.DataFrame = field(repr=False)

    def to_json(self) -> dict[str, object]:
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "n_trials": self.n_trials,
            "machine_epsilon": float(np.finfo(float).eps),
            "slope_fit_error_floor": SLOPE_FIT_ERROR_FLOOR,
            "slope_fit_error_ceiling": SLOPE_FIT_ERROR_CEILING,
            "curves": [curve.to_json() for curve in self.curves],
            "breakdown_condition_number": self.breakdown_condition,
        }


def _fit_log_log_slope(
    condition_numbers: np.ndarray, errors: np.ndarray
) -> tuple[float, int, tuple[float, float]]:
    """Least-squares slope of log10(error) on log10(kappa), conditioning band only."""
    usable = (
        np.isfinite(errors)
        & (errors > SLOPE_FIT_ERROR_FLOOR)
        & (errors < SLOPE_FIT_ERROR_CEILING)
    )
    if usable.sum() < 3:
        return float("nan"), int(usable.sum()), (float("nan"), float("nan"))
    x = np.log10(condition_numbers[usable])
    y = np.log10(errors[usable])
    # Two-column design, fitted with the module's own SVD solver rather than
    # polyfit: this file should not need a black box to fit a straight line.
    design = np.column_stack([np.ones_like(x), x])
    intercept, slope = solve_lstsq_svd(design, y)
    return (
        float(slope),
        int(usable.sum()),
        (float(condition_numbers[usable].min()), float(condition_numbers[usable].max())),
    )


def condition_number_sweep(
    *,
    condition_numbers: tuple[float, ...] = CONDITION_NUMBERS,
    n_rows: int = CONDITION_SWEEP_ROWS,
    n_columns: int = CONDITION_SWEEP_COLUMNS,
    n_trials: int = CONDITION_SWEEP_TRIALS,
    seed: int = CONDITION_SWEEP_SEED,
) -> ConditionSweep:
    """Measure forward error against condition number for all three solvers."""
    solvers: dict[str, tuple[Solver, float]] = {
        # (solver, slope the theory predicts)
        "cholesky": (solve_lstsq_cholesky, 2.0),
        "qr": (solve_lstsq_qr, 1.0),
        "svd": (solve_lstsq_svd, 1.0),
    }

    records: list[dict[str, object]] = []
    for condition_number in condition_numbers:
        for trial in range(n_trials):
            # Distinct seed per (kappa, trial) so no two cells share a matrix,
            # and reproducible without threading an rng through every call.
            trial_seed = seed + trial + 1000 * int(round(np.log10(condition_number)))
            design = synthetic_design(
                condition_number, n_rows=n_rows, n_columns=n_columns, seed=trial_seed
            )
            rng = np.random.default_rng(trial_seed)
            true_beta = rng.standard_normal(n_columns)
            true_beta /= np.linalg.norm(true_beta)
            # Consistent by construction: the exact least-squares solution is
            # true_beta with zero residual, so forward error is pure arithmetic.
            with clean_fp_state():
                target = design @ true_beta

            for name, (solver, _) in solvers.items():
                try:
                    estimate = solver(design, target)
                except np.linalg.LinAlgError:
                    error: float | None = None
                else:
                    error = float(
                        np.linalg.norm(estimate - true_beta) / np.linalg.norm(true_beta)
                    )
                records.append(
                    {
                        "solver": name,
                        "condition_number": float(condition_number),
                        "trial": trial,
                        "relative_forward_error": error,
                        "failed": error is None,
                    }
                )

    trials = pd.DataFrame(records)
    kappa = np.asarray(condition_numbers, dtype=float)

    curves: list[SolverErrorCurve] = []
    breakdown: dict[str, float | None] = {}
    for name, (_, expected_slope) in solvers.items():
        rows = trials[trials["solver"] == name]
        grouped = rows.groupby("condition_number")
        medians = grouped["relative_forward_error"].median().reindex(kappa)
        failures = grouped["failed"].sum().reindex(kappa).fillna(0)
        errors = medians.to_numpy(dtype=float)
        slope, n_points, fit_range = _fit_log_log_slope(kappa, errors)
        curves.append(
            SolverErrorCurve(
                solver=name,
                condition_numbers=[float(value) for value in kappa],
                median_errors=[
                    None if not np.isfinite(value) else float(value) for value in errors
                ],
                n_failures=[int(value) for value in failures.to_numpy()],
                fitted_slope=slope,
                expected_slope=expected_slope,
                n_slope_points=n_points,
                slope_fit_range=fit_range,
            )
        )
        # "Lost every digit" means the relative error has reached order 1: the
        # answer is no longer an approximation of the truth in any useful sense.
        lost = kappa[(~np.isfinite(errors)) | (errors >= 1.0)]
        breakdown[name] = float(lost.min()) if lost.size else None

    return ConditionSweep(
        n_rows=n_rows,
        n_columns=n_columns,
        n_trials=n_trials,
        curves=curves,
        breakdown_condition=breakdown,
        trials=trials,
    )


def plot_condition_sweep(
    sweep: ConditionSweep, *, reference_condition_number: float | None = None
) -> Path | None:
    """One panel: forward error against condition number, with slope references.

    The reference lines are the whole point of the panel. Cholesky tracking the
    ``kappa^2`` guide while QR and SVD track the ``kappa`` guide is the squared
    condition number made visible, and the vertical marker is where the normal
    equations stop returning an answer at all.

    ``reference_condition_number`` marks where the real HDB5 design matrix sits
    on this axis. It belongs on the plot because it is the honest caveat: the
    three solvers agreeing to 1e-12 in Result 2 is a fact about a matrix that
    lands at the far left of this sweep, not evidence that the choice is free.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - plotting is optional
        return None

    ink, muted = "#0b0b0b", "#52514e"
    colors = {"cholesky": "#eb6834", "qr": "#2a78d6", "svd": "#1a9c6d"}
    markers = {"cholesky": "o", "qr": "s", "svd": "^"}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.6, 5.4))
    figure.patch.set_facecolor("#fcfcfb")
    axis.set_facecolor("#fcfcfb")
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)

    epsilon = float(np.finfo(float).eps)
    guide_kappa = np.array([1e1, 1e12])
    # Anchored at eps so the guides read as "eps * kappa" and "eps * kappa^2"
    # rather than as free-floating fits to the measured points.
    axis.plot(guide_kappa, epsilon * guide_kappa**2, "--", color=muted, linewidth=1.0, zorder=1)
    axis.plot(guide_kappa, epsilon * guide_kappa, ":", color=muted, linewidth=1.0, zorder=1)
    axis.annotate(r"$\epsilon\,\kappa^{2}$", xy=(2e5, epsilon * (2e5) ** 2), xytext=(-2, 6),
                  textcoords="offset points", fontsize=9, color=muted)
    axis.annotate(r"$\epsilon\,\kappa$", xy=(3e10, epsilon * 3e10), xytext=(-4, 6),
                  textcoords="offset points", fontsize=9, color=muted)

    for curve in sweep.curves:
        kappa = np.asarray(curve.condition_numbers, dtype=float)
        errors = np.array(
            [np.nan if value is None else value for value in curve.median_errors], dtype=float
        )
        drawn = np.isfinite(errors)
        axis.plot(
            kappa[drawn],
            errors[drawn],
            marker=markers[curve.solver],
            color=colors[curve.solver],
            linewidth=1.8,
            markersize=5.5,
            label=f"{curve.solver}  (slope {curve.fitted_slope:.2f}, theory {curve.expected_slope:.0f})",
            zorder=3,
        )
        breakdown = sweep.breakdown_condition.get(curve.solver)
        if breakdown is not None:
            axis.axvline(breakdown, color=colors[curve.solver], linewidth=1.0, alpha=0.35, zorder=0)
            # Along the line rather than beside it: the lower right of this axis
            # is where the legend has to go.
            axis.annotate(
                f"{curve.solver} returns no answer beyond here",
                xy=(breakdown, 1e-4),
                xytext=(-6, 0),
                textcoords="offset points",
                fontsize=8.5,
                color=colors[curve.solver],
                rotation=90,
                ha="right",
                va="center",
            )

    if reference_condition_number is not None:
        axis.axvline(reference_condition_number, color=ink, linewidth=1.0, alpha=0.4, zorder=0)
        axis.annotate(
            f"HDB5's own design matrix\n"
            rf"($\kappa$ = {reference_condition_number:.1f}); all three"
            "\nsolvers agree here",
            xy=(reference_condition_number, 1e-3),
            xytext=(7, 0),
            textcoords="offset points",
            fontsize=8.5,
            color=ink,
            va="center",
        )

    axis.axhline(1.0, color=ink, linewidth=0.9, alpha=0.5)
    axis.annotate("no significant digits left", xy=(1.2e1, 1.0), xytext=(0, 5),
                  textcoords="offset points", fontsize=8.5, color=ink)

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(3, 3e12)
    axis.set_ylim(1e-16, 20)
    axis.set_xlabel(r"condition number of the design matrix, $\kappa(X)$", fontsize=9.5, color=muted)
    axis.set_ylabel("relative forward error in the coefficients", fontsize=9.5, color=muted)
    axis.set_title(
        "The normal equations work at the square of the condition number.\n"
        f"{sweep.n_trials} synthetic {sweep.n_rows}x{sweep.n_columns} matrices per point, "
        "consistent systems, median error.",
        fontsize=11,
        color=ink,
    )
    axis.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=muted)

    figure.tight_layout()
    path = RESULTS_DIR / "solver_conditioning.png"
    figure.savefig(path, dpi=180, facecolor="#fcfcfb")
    plt.close(figure)
    return path


# --- Result 3: what the data can determine -----------------------------------


@dataclass(frozen=True)
class Direction:
    index: int
    singular_value: float
    share_of_design_variance: float
    share_of_disagreement: float
    dominant_variables: list[str]


@dataclass(frozen=True)
class Spectrum:
    condition_number: float
    singular_values: np.ndarray = field(repr=False)
    directions: list[Direction] = field(default_factory=list)
    shrinkage: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    def to_json(self) -> dict[str, object]:
        return {
            "condition_number": self.condition_number,
            "singular_values": self.singular_values.tolist(),
            "directions": [
                {
                    "index": direction.index,
                    "singular_value": direction.singular_value,
                    "share_of_design_variance": direction.share_of_design_variance,
                    "share_of_disagreement": direction.share_of_disagreement,
                    "dominant_variables": direction.dominant_variables,
                }
                for direction in self.directions
            ],
        }


def conditioning_analysis(dataset: pd.DataFrame) -> Spectrum:
    """Singular spectrum of the physics design matrix, and where we disagree.

    If the difference between our refit exponents and IPB98's lay along the
    well-determined directions it would be a real physical disagreement. If it
    lies along the weak ones, both laws fit the data about equally well and the
    individual exponents were never separately pinned down. Mapping the
    difference into standardized coordinates first (multiply by the column
    standard deviations) is required, because the singular vectors live there
    and the exponents do not.
    """
    design, names = build_log_design_matrix(dataset, IPB98_FEATURE_COLUMNS, intercept=False)
    report = analyze_conditioning(design, names, standardize=True)

    standardized = np.array(design, dtype=float)
    for column in range(standardized.shape[1]):
        values = standardized[:, column]
        standardized[:, column] = (values - values.mean()) / values.std()
    _, singular_values, vt = np.linalg.svd(standardized, full_matrices=False)
    design_variance = singular_values**2 / np.sum(singular_values**2)

    fit = fit_scaling_law(dataset, hdb5.TARGET_COLUMN, IPB98_FEATURE_COLUMNS)
    raw_difference = np.array([fit.exponents[column] - IPB98Y2_EXPONENTS[column] for column in IPB98_FEATURE_COLUMNS])
    difference = raw_difference * report.column_scales
    projections = vt @ difference
    disagreement = projections**2 / np.sum(projections**2)

    directions = []
    for position in range(len(singular_values)):
        loading = vt[position]
        ordered = np.argsort(-np.abs(loading))
        directions.append(
            Direction(
                index=position + 1,
                singular_value=float(singular_values[position]),
                share_of_design_variance=float(design_variance[position]),
                share_of_disagreement=float(disagreement[position]),
                dominant_variables=[f"{names[i].removeprefix('log_')}: {loading[i]:+.3f}" for i in ordered[:4]],
            )
        )

    shrinkage = pd.DataFrame(
        {
            "singular_value": singular_values,
            **{
                f"alpha_{alpha:g}": ridge_shrinkage_factors(singular_values, alpha) for alpha in (0.1, 1.0, 10.0, 100.0)
            },
        }
    )

    return Spectrum(
        condition_number=report.condition_number,
        singular_values=singular_values,
        directions=directions,
        shrinkage=shrinkage,
    )


# --- Figure -------------------------------------------------------------------


def plot_spectrum(spectrum: Spectrum, audit: RankAudit) -> Path | None:
    """Three panels: the physics spectrum, the rank cliff, and where we disagree.

    The right panel is the result. It puts the share of the design matrix's
    variance carried by each singular direction next to the share of our
    disagreement with IPB98 that lives in that direction, on one common
    percentage axis so the two are directly comparable.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - plotting is optional
        return None

    blue, orange = "#2a78d6", "#eb6834"
    ink, muted = "#0b0b0b", "#52514e"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), gridspec_kw={"width_ratios": [1.0, 1.0, 1.5]})
    for axis in axes:
        axis.grid(alpha=0.25, linewidth=0.6)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(muted)
            axis.spines[side].set_linewidth(0.8)
        axis.tick_params(colors=muted, labelsize=9)

    physics = spectrum.singular_values
    axes[0].semilogy(range(1, len(physics) + 1), physics, "o-", color=blue, linewidth=1.8, markersize=6)
    axes[0].set_title("Engineering variables\n(8 columns, full rank)", fontsize=11, color=ink)
    axes[0].set_xlabel("singular value index", fontsize=9, color=muted)
    axes[0].set_ylabel("singular value (standardized)", fontsize=9, color=muted)
    axes[0].annotate(
        f"condition number {spectrum.condition_number:.1f}\nno ill conditioning to report",
        xy=(0.06, 0.06),
        xycoords="axes fraction",
        fontsize=9,
        color=muted,
    )

    model = audit.singular_values
    floor = 1e-16
    axes[1].semilogy(
        range(1, len(model) + 1),
        np.maximum(model, floor),
        "o-",
        color=orange,
        linewidth=1.8,
        markersize=6,
    )
    axes[1].axhline(audit.tolerance, color=muted, linestyle="--", linewidth=1.0)
    axes[1].set_title(
        f"Model feature matrix\n({audit.n_columns} columns, rank {audit.rank})",
        fontsize=11,
        color=ink,
    )
    axes[1].set_xlabel("singular value index", fontsize=9, color=muted)
    axes[1].annotate("numerical zero", xy=(1.2, audit.tolerance * 2.0), fontsize=8, color=muted)
    axes[1].annotate(
        "two exact\ndependencies",
        xy=(len(model) - 0.6, max(float(model[-1]), floor)),
        xytext=(len(model) - 4.4, 1e-9),
        fontsize=9,
        color=ink,
        arrowprops={"arrowstyle": "->", "color": muted, "linewidth": 0.9},
    )

    index = np.arange(len(spectrum.directions))
    design_share = np.array([d.share_of_design_variance for d in spectrum.directions]) * 100
    disagreement_share = np.array([d.share_of_disagreement for d in spectrum.directions]) * 100
    width = 0.38
    axes[2].bar(
        index - width / 2 - 0.01,
        design_share,
        width,
        color=blue,
        label="share of what the data determines",
    )
    axes[2].bar(
        index + width / 2 + 0.01,
        disagreement_share,
        width,
        color=orange,
        label="share of our disagreement with IPB98",
    )
    axes[2].annotate(
        f"{disagreement_share[-1]:.0f}%",
        xy=(index[-1] + width / 2 + 0.01, disagreement_share[-1]),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        fontsize=10,
        color=ink,
    )
    axes[2].set_xticks(index)
    axes[2].set_xticklabels([str(d.index) for d in spectrum.directions])
    axes[2].set_xlabel("singular direction (strongest to weakest)", fontsize=9, color=muted)
    axes[2].set_ylabel("percent", fontsize=9, color=muted)
    axes[2].set_title("The disagreement lives where the data is blind", fontsize=11, color=ink)
    axes[2].legend(frameon=False, fontsize=9, loc="upper center", labelcolor=muted)
    axes[2].set_ylim(0, max(disagreement_share.max(), design_share.max()) * 1.28)

    figure.tight_layout()
    path = RESULTS_DIR / "singular_value_spectrum.png"
    figure.savefig(path, dpi=180, facecolor="#fcfcfb")
    plt.close(figure)
    return path


def main() -> None:
    dataset = hdb5.prepare_dataset()
    print(
        f"HDB5: {len(dataset)} rows, {dataset[hdb5.GROUP_COLUMN].nunique()} discharges, "
        f"{dataset[hdb5.TOKAMAK_LABEL_COLUMN].nunique()} tokamaks"
    )

    audit = audit_model_feature_matrix(dataset)
    refit = refit_ipb98(dataset)
    spectrum = conditioning_analysis(dataset)
    sweep = condition_number_sweep()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_dataframe_csv_atomic(RESULTS_DIR / "ipb98_refit_exponents.csv", refit.intervals)
    write_dataframe_csv_atomic(RESULTS_DIR / "ridge_shrinkage.csv", spectrum.shrinkage)
    if refit.resolution is not None:
        write_dataframe_csv_atomic(
            RESULTS_DIR / "bootstrap_resolution.csv", refit.resolution.to_frame()
        )
    write_dataframe_csv_atomic(RESULTS_DIR / "solver_conditioning.csv", sweep.trials)
    write_json_strict(
        RESULTS_DIR / "analysis.json",
        {
            # The dataset fingerprint leads: every number below it is a
            # statement about these exact bytes and nothing else.
            "dataset": hdb5.dataset_provenance(),
            "rank_audit": audit.to_json(),
            "refit": refit.to_json(),
            "conditioning": spectrum.to_json(),
            "solver_conditioning": sweep.to_json(),
        },
    )
    figure_path = plot_spectrum(spectrum, audit)
    conditioning_figure = plot_condition_sweep(
        sweep, reference_condition_number=refit.condition_number
    )

    print("\n--- Result 1: rank audit of the model feature matrix ---")
    print(f"rank {audit.rank} of {audit.n_columns} (deficiency {audit.rank_deficiency})")
    print(f"numerical-zero tolerance {audit.tolerance:.2e}")
    for label, residual in audit.projection_residuals.items():
        print(f"  projection residual, {label}: {residual:.3e}")
    for label, alignment in audit.basis_alignments.items():
        print(f"  best alignment with a printed basis vector, {label}: {alignment:.3f}")
    print(f"  rank without standardizing first: {audit.unstandardized_rank} (a unit artifact)")

    print("\n--- Result 2: IPB98 refit from HDB5 ---")
    print(f"design matrix {list(refit.design_shape)}, cond {refit.condition_number:.1f}")
    for solver in refit.solvers:
        print(
            f"  {solver.name:<9} {solver.seconds_per_solve * 1e3:7.3f} ms   "
            f"max deviation from SVD {solver.max_deviation_from_svd:.2e}"
        )
    print(refit.intervals.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
    print(
        f"  fitted coefficient {refit.fitted_coefficient:.4f} vs published "
        f"{IPB98Y2_COEFFICIENT} (also quoted as {IPB98Y2_COEFFICIENT_ROUNDED})"
    )
    print(f"  in-sample RMSLE: refit {refit.rmsle_refit:.4f}, published IPB98(y,2) {refit.rmsle_published:.4f}")

    resolution = refit.resolution
    if resolution is not None:
        print("\n--- Result 2b: the same intervals, resampling coarser units ---")
        units = ", ".join(
            f"{resolution.units(name)} {name}s" for name in resolution.level_names
        )
        print(
            f"{units}; "
            f"{' and '.join(resolution.largest_two_devices)} alone are "
            f"{resolution.largest_two_row_share * 100:.0f}% of the rows"
        )
        header = f"  {'exponent':<20}" + "".join(
            f"{name + ' 95%':>24}" for name in resolution.level_names
        )
        print(header)
        for width in resolution.widths:
            cells = "".join(
                f"{f'[{width.bounds[name][0]:+.3f}, {width.bounds[name][1]:+.3f}]':>24}"
                for name in resolution.level_names
            )
            print(f"  {width.variable:<20}{cells}")
        for name in resolution.level_names:
            if name == BASELINE_LEVEL:
                continue
            worst = resolution.max_widening(name)
            print(
                f"  vs {BASELINE_LEVEL}: {name} intervals are "
                f"{resolution.median_widening(name):.1f}x wider at the median, "
                f"{worst.widening_factors[name]:.1f}x on {worst.variable}"
            )
        n_comparable = len(resolution.comparable_widths)
        inside = ", ".join(
            f"{resolution.n_published_inside(name)}/{n_comparable} by {name}"
            for name in resolution.level_names
        )
        print(f"  published IPB98 exponent inside the interval: {inside}")

    print("\n--- Result 2c: forward error against condition number ---")
    print(
        f"{sweep.n_trials} synthetic {sweep.n_rows}x{sweep.n_columns} consistent systems per "
        f"condition number, kappa from 1e1 to 1e12"
    )
    print(f"  {'solver':<10}{'fitted slope':>14}{'theory':>8}{'points':>8}{'breaks down at':>18}")
    for curve in sweep.curves:
        breakdown = sweep.breakdown_condition.get(curve.solver)
        breakdown_text = f"{breakdown:.0e}" if breakdown is not None else "not in sweep"
        print(
            f"  {curve.solver:<10}{curve.fitted_slope:>14.2f}{curve.expected_slope:>8.0f}"
            f"{curve.n_slope_points:>8}{breakdown_text:>18}"
        )

    print("\n--- Result 3: what the data determines ---")
    print(f"condition number {spectrum.condition_number:.1f}")
    print("  direction  sigma    share of design variance    share of disagreement")
    for direction in spectrum.directions:
        print(
            f"  {direction.index:>9}  {direction.singular_value:7.2f}"
            f"  {direction.share_of_design_variance * 100:22.2f}%"
            f"  {direction.share_of_disagreement * 100:19.2f}%"
        )
    weakest = spectrum.directions[-1]
    print(f"  weakest direction: {', '.join(weakest.dominant_variables)}")
    for path in (figure_path, conditioning_figure):
        if path:
            print(f"\nfigure: {path}")


if __name__ == "__main__":
    main()
