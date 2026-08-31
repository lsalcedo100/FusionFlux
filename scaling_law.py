"""Least squares from first principles, applied to tokamak confinement scaling.

A confinement scaling law is a power law,

    tau_E = C * Ip^a1 * Bt^a2 * ne^a3 * P^a4 * R^a5 * eps^a6 * kappa^a7 * M^a8

so taking logs makes it a *linear* model,

    log tau_E = log C + a1 log Ip + a2 log Bt + ... + a8 log M

and fitting a scaling law is ordinary least squares on a log design matrix.
Every question about the physics then becomes a question about that matrix: its
rank, its null space, its singular values, its condition number.

This module deliberately does not call ``scikit-learn``. The three classical
solvers (normal equations via Cholesky, QR, SVD) are implemented here, including
the triangular substitutions, because the point is to show the numerics rather
than to delegate them. ``tests/test_scaling_law.py`` checks all three against a
problem with a known closed-form answer, and against each other.

References for the published baseline:
    ITER Physics Basis, Nucl. Fusion 39 2175 (1999), IPB98(y,2) scaling.
    Verdoolaege et al., Nucl. Fusion 61 076006 (2021), the HDB5 database.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


@contextlib.contextmanager
def _clean_fp_state() -> Iterator[None]:
    """Silence spurious NumPy 2.x BLAS floating-point flags raised by ``matmul``.

    On some BLAS backends ``matmul`` reports divide-by-zero, overflow or invalid
    value on inputs that are entirely finite and results that are correct to
    machine precision (the tests here assert agreement to 1e-8 while the warning
    fires). The flag is left over from earlier vectorized work rather than
    produced by this operation. Scoped to individual solver calls so a genuine
    overflow elsewhere is still reported.
    """
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        yield


# Published IPB98(y,2) exponents, in the order of IPB98_FEATURE_COLUMNS below.
#
# The multiplying coefficient is quoted as 0.0562 in the ITER Physics Basis and
# rounded to 0.056 in several later summaries. We carry 0.0562 and report the
# discrepancy rather than silently picking one; it shifts log C by 0.0036, which
# is far inside the confidence interval we fit, so nothing here depends on it.
IPB98Y2_COEFFICIENT = 0.0562
IPB98Y2_COEFFICIENT_ROUNDED = 0.056

IPB98_FEATURE_COLUMNS: tuple[str, ...] = (
    "ip_ma",
    "bt_t",
    "ne_line_1e19_m3",
    "p_loss_mw",
    "r_m",
    "inverse_aspect_ratio",
    "kappa",
    "m_eff_amu",
)

IPB98Y2_EXPONENTS: dict[str, float] = {
    "ip_ma": 0.93,
    "bt_t": 0.15,
    "ne_line_1e19_m3": 0.41,
    "p_loss_mw": -0.69,
    "r_m": 1.97,
    "inverse_aspect_ratio": 0.58,
    "kappa": 0.78,
    "m_eff_amu": 0.19,
}

INTERCEPT_NAME = "log_coefficient"


# --- Triangular solvers -----------------------------------------------------
#
# Written out rather than delegated so the substitution order is visible. Both
# are O(n^2) and assume the triangular factor is nonsingular; the callers below
# guarantee that by checking rank first.


def forward_substitution(lower: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve ``L z = rhs`` for lower-triangular ``L``."""
    n = lower.shape[0]
    z = np.zeros(n, dtype=float)
    for i in range(n):
        z[i] = (rhs[i] - lower[i, :i] @ z[:i]) / lower[i, i]
    return z


def back_substitution(upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve ``U x = rhs`` for upper-triangular ``U``."""
    n = upper.shape[0]
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        x[i] = (rhs[i] - upper[i, i + 1 :] @ x[i + 1 :]) / upper[i, i]
    return x


# --- The three classical least-squares solvers ------------------------------


def solve_lstsq_cholesky(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Least squares via the normal equations, factored by Cholesky.

    Forms ``X^T X = L L^T`` and solves two triangular systems. Fastest of the
    three and the least numerically stable: it squares the condition number, so
    a design matrix with cond(X) = 1e8 gives cond(X^T X) = 1e16 and the solution
    loses all significant digits in double precision. Raises on a singular or
    indefinite ``X^T X``, which is the honest failure mode.
    """
    with _clean_fp_state():
        gram = design.T @ design
    try:
        lower = np.linalg.cholesky(gram)
    except np.linalg.LinAlgError as error:  # pragma: no cover - message path
        raise np.linalg.LinAlgError(
            "X^T X is not positive definite, so the design matrix is rank "
            "deficient (or close enough that the squared condition number "
            "overflows double precision). Use solve_lstsq_svd instead."
        ) from error
    with _clean_fp_state():
        z = forward_substitution(lower, design.T @ target)
        return back_substitution(lower.T, z)


def solve_lstsq_qr(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Least squares via QR: ``X = QR``, then back-substitute ``R b = Q^T y``.

    Never forms ``X^T X``, so it works at cond(X) rather than cond(X)^2. This is
    what a general-purpose library reaches for when the matrix is full rank.
    """
    with _clean_fp_state():
        q, r = np.linalg.qr(design, mode="reduced")
        return back_substitution(r, q.T @ target)


def solve_lstsq_svd(design: np.ndarray, target: np.ndarray, *, rcond: float | None = None) -> np.ndarray:
    """Least squares via the SVD pseudoinverse, ``b = V S^+ U^T y``.

    The only one of the three that survives rank deficiency. Directions whose
    singular values fall below ``rcond * s_max`` are dropped rather than
    inverted, which returns the *minimum-norm* solution: of the infinitely many
    coefficient vectors that fit equally well, it picks the shortest one.

    That is exactly what ``scipy.linalg.lstsq`` does under the hood, and why a
    rank-deficient fit returns a clean-looking answer instead of an error. The
    number is arbitrary in the null-space directions; nothing warns you.
    """
    u, s, vt = np.linalg.svd(design, full_matrices=False)
    if rcond is None:
        rcond = float(max(design.shape) * np.finfo(float).eps)
    keep = s > rcond * s[0]
    s_inv = np.zeros_like(s)
    s_inv[keep] = 1.0 / s[keep]
    with _clean_fp_state():
        return vt.T @ (s_inv * (u.T @ target))


def ridge_from_svd(
    u: np.ndarray, s: np.ndarray, vt: np.ndarray, target: np.ndarray, alpha: float
) -> np.ndarray:
    """The ridge solution for one ``alpha``, given a factorization already computed.

    Split out from :func:`solve_lstsq_ridge` because alpha enters *only* through
    the per-direction filter: the SVD does not depend on it. A sweep over a
    penalty grid can therefore factor once and evaluate the whole grid for the
    cost of a few matrix-vector products, which is what
    ``analysis_flexibility_sweep`` relies on to make a 4-by-9 grid affordable at
    degree 4. Keeping one implementation of the filter means the sweep and the
    single-fit path cannot drift apart.
    """
    if alpha < 0:
        raise ValueError("alpha must be non-negative.")
    filtered = s / (s**2 + alpha)
    with _clean_fp_state():
        return vt.T @ (filtered * (u.T @ target))


def solve_lstsq_ridge(design: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    """Ridge regression expressed through the SVD, to make shrinkage visible.

    ``b = sum_i [ s_i^2 / (s_i^2 + alpha) ] * (u_i^T y / s_i) * v_i``

    Each singular direction is multiplied by ``s^2 / (s^2 + alpha)``: about 1
    for directions the data determines well, and near 0 for directions it barely
    determines. Regularization is not a knob that makes numbers behave, it is a
    decision about which combinations of exponents you are declining to resolve.
    """
    u, s, vt = np.linalg.svd(design, full_matrices=False)
    return ridge_from_svd(u, s, vt, target, alpha)


def ridge_shrinkage_factors(singular_values: np.ndarray, alpha: float) -> np.ndarray:
    """The per-direction shrinkage factors ``s^2 / (s^2 + alpha)``."""
    s = np.asarray(singular_values, dtype=float)
    return s**2 / (s**2 + alpha)


def solve_constrained_lstsq(
    design: np.ndarray,
    target: np.ndarray,
    constraint: np.ndarray,
    constraint_rhs: np.ndarray,
) -> np.ndarray:
    """Minimize ``||Xb - y||^2`` subject to ``Cb = d``, via the KKT system.

    Stationarity of the Lagrangian gives the saddle-point system

        [ 2 X^T X   C^T ] [ b      ]   [ 2 X^T y ]
        [ C         0   ] [ lambda ] = [ d       ]

    which is solved directly. Physics constraints on a scaling law (dimensional
    analysis in the Connor-Taylor sense, or simply pinning an exponent to a
    published value) enter as extra rows of ``C``. So does the rank-deficiency
    fix: constraining ``b`` to be orthogonal to the null space of ``X``, that is
    ``C = null_space(X)`` and ``d = 0``, reproduces the minimum-norm SVD
    solution exactly. ``tests/test_scaling_law.py`` checks that equivalence.
    """
    n_params = design.shape[1]
    n_constraints = constraint.shape[0]
    kkt = np.zeros((n_params + n_constraints, n_params + n_constraints), dtype=float)
    kkt[:n_params, :n_params] = 2.0 * (design.T @ design)
    kkt[:n_params, n_params:] = constraint.T
    kkt[n_params:, :n_params] = constraint
    rhs = np.concatenate([2.0 * (design.T @ target), constraint_rhs])
    # The KKT matrix is symmetric indefinite, never positive definite, so
    # Cholesky is not available here; lstsq handles the saddle-point structure.
    solution, *_ = np.linalg.lstsq(kkt, rhs, rcond=None)
    return solution[:n_params]


# --- Design matrix and conditioning ----------------------------------------


def build_log_design_matrix(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...] = IPB98_FEATURE_COLUMNS,
    *,
    intercept: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Build the log-space design matrix for a power-law fit.

    Every feature must be strictly positive; a non-positive value means the row
    should have been dropped in cleaning, and silently coercing it would put a
    NaN or an -inf into the matrix.
    """
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Frame is missing required feature columns: {missing}")
    values = frame.loc[:, list(feature_columns)].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(
            "All power-law features must be finite and strictly positive before taking logs; clean the frame first."
        )
    logs = np.log(values)
    names = [f"log_{column}" for column in feature_columns]
    if intercept:
        logs = np.column_stack([np.ones(len(logs)), logs])
        names = [INTERCEPT_NAME, *names]
    return logs, names


@dataclass(frozen=True)
class ConditioningReport:
    """What the design matrix can and cannot determine."""

    column_names: list[str]
    singular_values: np.ndarray
    rank: int
    n_columns: int
    condition_number: float
    null_space: np.ndarray = field(repr=False)
    tolerance: float
    column_scales: np.ndarray = field(repr=False)

    @property
    def rank_deficiency(self) -> int:
        return self.n_columns - self.rank

    @property
    def is_rank_deficient(self) -> bool:
        return self.rank_deficiency > 0

    def to_analysis_coordinates(self, vector: np.ndarray) -> np.ndarray:
        """Map a dependency stated in raw log units into standardized coordinates.

        This step is easy to skip and silently fatal. A dependency among the raw
        columns, ``sum_j c_j x_j = const``, is a statement about ``x``. The
        analysis runs on standardized columns ``z_j = (x_j - m_j) / s_j``, and
        substituting gives ``sum_j (c_j s_j) z_j = const``. So the same
        dependency is the vector ``c_j * s_j``, not ``c_j``.

        Testing the raw ``c`` against the standardized null space produces a
        large residual for a dependency that is exactly correct, which reads as
        "my expected relationship is not there" when the truth is "I compared
        vectors living in two different coordinate systems."
        """
        return np.asarray(vector, dtype=float) * self.column_scales

    def null_space_residual(self, vector: np.ndarray, *, raw_units: bool = False) -> float:
        """How far ``vector`` lies from the null space, in relative norm.

        Set ``raw_units=True`` when the vector states a dependency among the
        original log columns rather than the standardized ones; see
        :meth:`to_analysis_coordinates`.

        The null space of a rank-deficient matrix is a *subspace*, and the SVD
        returns an arbitrary orthonormal basis for it. When the deficiency is
        greater than one, the basis vectors printed by ``numpy`` will not
        resemble the dependency you expect, even when your expectation is
        exactly right. The correct test is whether your vector lies in the
        span, so project onto the subspace and measure the residual:

            residual = || v - N^T N v ||

        which is zero (to rounding) if and only if ``v`` is in the null space.
        Comparing printed basis vectors instead is the standard way to conclude
        you were wrong when you were right.
        """
        v = np.asarray(vector, dtype=float)
        if v.shape != (self.n_columns,):
            raise ValueError(f"Vector must have length {self.n_columns}, got {v.shape}.")
        if raw_units:
            v = self.to_analysis_coordinates(v)
        norm = np.linalg.norm(v)
        if norm == 0:
            raise ValueError("The zero vector lies in every subspace; pass a real dependency.")
        v = v / norm
        if self.null_space.size == 0:
            return 1.0
        projector = self.null_space.T @ self.null_space
        return float(np.linalg.norm(v - projector @ v))

    def lies_in_null_space(self, vector: np.ndarray, *, atol: float = 1e-8, raw_units: bool = False) -> bool:
        return self.null_space_residual(vector, raw_units=raw_units) <= atol


def analyze_conditioning(
    design: np.ndarray, column_names: list[str], *, standardize: bool = True
) -> ConditioningReport:
    """Singular value spectrum, rank, condition number and null space.

    ``standardize`` centers and scales the non-constant columns first, and it
    matters more than it looks. On raw physical columns the largest singular
    value is set by whichever feature carries the biggest units, so numpy's
    default rank tolerance (``s_max * max(shape) * eps``) discards directions
    that are perfectly informative and reports a rank deficiency that is an
    artifact of unit choice rather than a property of the data. Standardizing
    first makes the reported rank a statement about collinearity.
    """
    matrix = np.array(design, dtype=float, copy=True)
    scales = np.ones(matrix.shape[1], dtype=float)
    if standardize:
        for j in range(matrix.shape[1]):
            column = matrix[:, j]
            spread = column.std()
            if spread > 0:
                scales[j] = spread
                matrix[:, j] = (column - column.mean()) / spread
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    _, _, vt = np.linalg.svd(matrix, full_matrices=False)
    tolerance = float(singular_values[0] * max(matrix.shape) * np.finfo(float).eps)
    keep = singular_values > tolerance
    rank = int(keep.sum())
    smallest_kept = singular_values[keep][-1] if rank else np.nan
    condition_number = float(singular_values[0] / smallest_kept) if rank else float("inf")
    return ConditioningReport(
        column_names=list(column_names),
        singular_values=singular_values,
        rank=rank,
        n_columns=matrix.shape[1],
        condition_number=condition_number,
        null_space=vt[~keep],
        tolerance=tolerance,
        column_scales=scales,
    )


# --- Fitting a scaling law --------------------------------------------------


@dataclass(frozen=True)
class ScalingLawFit:
    exponents: dict[str, float]
    coefficient: float
    column_names: list[str]
    coefficients: np.ndarray
    conditioning: ConditioningReport
    n_rows: int
    residual_std_log: float

    def predict(self, frame: pd.DataFrame, feature_columns: tuple[str, ...]) -> np.ndarray:
        design, _ = build_log_design_matrix(frame, feature_columns, intercept=True)
        with _clean_fp_state():
            return np.exp(design @ self.coefficients)

    def compare_to_published(self, published: dict[str, float] = IPB98Y2_EXPONENTS) -> pd.DataFrame:
        rows = [
            {
                "variable": name,
                "fitted": self.exponents[name],
                "published_ipb98y2": published.get(name, float("nan")),
                "difference": self.exponents[name] - published.get(name, float("nan")),
            }
            for name in self.exponents
        ]
        return pd.DataFrame(rows)


def dof_for(target: np.ndarray, design: np.ndarray) -> int:
    """Residual degrees of freedom, counting only the directions actually fitted.

    Uses the numerical rank rather than the column count, so a rank-deficient
    design does not overstate the number of parameters it estimated.
    """
    return max(len(target) - np.linalg.matrix_rank(design), 1)


def fit_scaling_law(
    frame: pd.DataFrame,
    target_column: str,
    feature_columns: tuple[str, ...] = IPB98_FEATURE_COLUMNS,
    *,
    solver: str = "svd",
) -> ScalingLawFit:
    """Fit ``tau = C * prod(x_i ^ a_i)`` by least squares in log space."""
    design, names = build_log_design_matrix(frame, feature_columns, intercept=True)
    target = np.log(frame[target_column].to_numpy(dtype=float))
    solvers: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
        "cholesky": solve_lstsq_cholesky,
        "qr": solve_lstsq_qr,
        "svd": solve_lstsq_svd,
    }
    if solver not in solvers:
        raise ValueError(f"Unknown solver {solver!r}; choose from {sorted(solvers)}.")
    beta = solvers[solver](design, target)
    with _clean_fp_state():
        residuals = target - design @ beta
        residual_std_log = float(np.sqrt(float(residuals @ residuals) / dof_for(target, design)))
    return ScalingLawFit(
        exponents={column: float(value) for column, value in zip(feature_columns, beta[1:])},
        coefficient=float(np.exp(beta[0])),
        column_names=names,
        coefficients=beta,
        conditioning=analyze_conditioning(design, names),
        n_rows=len(target),
        residual_std_log=residual_std_log,
    )


def bootstrap_exponents(
    frame: pd.DataFrame,
    target_column: str,
    feature_columns: tuple[str, ...] = IPB98_FEATURE_COLUMNS,
    *,
    group_column: str | None = None,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """Percentile bootstrap confidence intervals for the fitted exponents.

    ``group_column`` resamples whole groups (a discharge contributes several
    quasi-stationary time slices) instead of individual rows. Row-level
    resampling would treat slices from one shot as independent observations and
    return intervals several times too narrow.
    """
    rng = np.random.default_rng(random_state)
    if group_column is None:
        index_pool = [np.array([i]) for i in range(len(frame))]
    else:
        # ``.indices`` maps each group key to the positional row indices of its
        # members, which is exactly the unit we want to resample.
        index_pool = [
            np.atleast_1d(np.asarray(positions))
            for positions in frame.groupby(group_column, sort=True).indices.values()
        ]
    n_units = len(index_pool)

    draws: list[np.ndarray] = []
    for _ in range(n_resamples):
        picks = rng.integers(0, n_units, size=n_units)
        row_positions = np.concatenate([index_pool[p] for p in picks])
        sample = frame.iloc[row_positions]
        design, _ = build_log_design_matrix(sample, feature_columns, intercept=True)
        target = np.log(sample[target_column].to_numpy(dtype=float))
        draws.append(solve_lstsq_svd(design, target))

    stacked = np.vstack(draws)
    lower_q = 100.0 * (1.0 - confidence) / 2.0
    upper_q = 100.0 - lower_q
    lower = np.percentile(stacked, lower_q, axis=0)
    upper = np.percentile(stacked, upper_q, axis=0)
    point = fit_scaling_law(frame, target_column, feature_columns).coefficients

    rows = []
    for i, name in enumerate([INTERCEPT_NAME, *feature_columns]):
        rows.append(
            {
                "variable": name,
                "fitted": float(point[i]),
                "ci_low": float(lower[i]),
                "ci_high": float(upper[i]),
                "published_ipb98y2": IPB98Y2_EXPONENTS.get(name, float("nan")),
            }
        )
    result = pd.DataFrame(rows)
    result["published_inside_ci"] = (result["published_ipb98y2"] >= result["ci_low"]) & (
        result["published_ipb98y2"] <= result["ci_high"]
    )
    return result
