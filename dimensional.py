"""Connor-Taylor dimensional constraints on a confinement scaling law.

``scaling_law.py`` fits the eight exponents of

    tau_E = C * Ip^a1 Bt^a2 ne^a3 P^a4 R^a5 eps^a6 kappa^a7 M^a8

with nothing but least squares deciding what they are. Physics says they are not
free. If the plasma is governed by some set of equations, those equations are
invariant under a family of scale transformations, and that invariance forces
the *dimensionless* confinement time to depend only on dimensionless parameters:

    Omega_i tau_E = F(rho*, beta, nu*, q, eps, kappa, M)

Connor and Taylor (Nucl. Fusion 17 1047, 1977) turned that into an arithmetic
statement. Requiring the engineering power law above to be expressible in that
dimensionless form imposes *linear equality constraints on the exponents*, one
per independent scale transformation the physics admits. Which transformations
those are depends on how much physics you are willing to assume, so the result
is a hierarchy of nested models: each extra assumption is one more row of a
constraint matrix, and one fewer free parameter.

That makes this module a natural companion to ``scaling_law.py`` rather than a
departure from it. A physics assumption becomes a matrix ``C`` and a vector
``d``, the fit becomes ``min ||Xb - y||^2 subject to Cb = d``, and
``scaling_law.solve_constrained_lstsq`` already solves exactly that through its
KKT system. Nothing here needs a new solver; the physics enters as data.

The derivation is done numerically rather than by quoting the answer
--------------------------------------------------------------------
Every constraint below is derived in code from the definitions of rho*, beta and
nu*, by finding the null space of the invariance conditions. The alternative was
to hard-code the exponent vectors from a paper, which is exactly the kind of
hand-copied constant this repository avoids everywhere else: a transcription
error in a constraint row would produce a clean-looking fit obeying the wrong
physics, and nothing downstream would notice.

Two independent checks that the derivation is right:

* ``IPB98(y,2)`` satisfies the Kadomtsev constraint to **0.0025** and the
  collisionless constraint to **0.0100**. Its exponents are published to two
  decimal places, so both residuals are inside the rounding of the law's own
  coefficients. A published scaling law landing on a surface derived here from
  scratch is not something a wrong derivation does by accident.
* ``tests/test_dimensional.py`` re-derives the transformations from the group
  definitions and checks the resulting engineering exponents against the
  closed-form values written out in :func:`similarity_transformation`.

The scale transformations
-------------------------
Write a transformation as a power of a single parameter ``lam`` acting on the
four independent physical scales: a length ``L`` (so ``R`` and ``a`` together,
which keeps ``eps`` fixed), the field ``B``, the temperature ``T`` and the
density ``n``. Under ``L -> lam^l``, ``B -> lam^b``, ``T -> lam^t``,
``n -> lam^m`` the three dimensionless groups scale as

    rho* ~ T^(1/2) / (B L)      ->   t/2 - b - l
    beta ~ n T / B^2            ->   m + t - 2b
    nu*  ~ n L / T^2            ->   m + l - 2t

and the dimensionless confinement time ``Omega_i tau_E ~ B tau_E`` is invariant
exactly when ``tau_E -> lam^(-b)``.

Holding a group fixed is one linear condition on ``(l, b, t, m)``. Four unknowns
minus the number of conditions is the dimension of the family of transformations
the physics admits, and each independent direction in that family is one
constraint on the exponents. So *more* physics assumed means *fewer* groups that
have to be held fixed, means a larger family, means more constraints:

    model            groups held fixed        free directions   constraints
    free             (none assumed)                  -                0
    kadomtsev        rho*, beta, nu*                 1                1
    collisionless    rho*, beta                      2                2
    electrostatic    rho*                            3                3

``kadomtsev`` assumes only that a dimensionless description exists at all, which
is why it is the weakest and why the field regards it as close to mandatory.
``collisionless`` additionally assumes tau_E does not depend on collisionality,
``electrostatic`` that it does not depend on beta either.

References:
    J.W. Connor and J.B. Taylor, Nucl. Fusion 17 1047 (1977).
    B.B. Kadomtsev, Sov. J. Plasma Phys. 1 295 (1975).
    ITER Physics Basis, Nucl. Fusion 39 2175 (1999), Chapter 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

import scaling_law as sl

# Same import spelling as ``analysis_scaling_law`` and ``analysis_flexibility_sweep``:
# the guard silences BLAS floating-point flags that ``matmul`` raises on entirely
# finite inputs, and every module building on these solvers needs it.
from scaling_law import _clean_fp_state as clean_fp_state

# The engineering variables the constraints are stated on, in the order used
# throughout. Deliberately ``scaling_law.IPB98_FEATURE_COLUMNS`` rather than
# ``hdb5.BLIND_FEATURE_COLUMNS``: the latter also carries ``log_a_m``, which is
# exactly ``log_r_m + log_inverse_aspect_ratio`` (Result 1), so a design matrix
# including it is rank deficient and a constraint on its exponent would be a
# statement about an arbitrary member of a one-parameter family. Minor radius
# enters here through the length scale ``L``, which is what ``r_m`` and ``eps``
# already encode between them.
CONSTRAINED_FEATURE_COLUMNS: tuple[str, ...] = sl.IPB98_FEATURE_COLUMNS

# The three dimensionless groups, as their exponents on (l, b, t, m).
#
#   rho* ~ T^(1/2) B^-1 L^-1
#   beta ~ n T B^-2
#   nu*  ~ n L T^-2
#
# Ordered (l, b, t, m) throughout.
DIMENSIONLESS_GROUPS: dict[str, np.ndarray] = {
    "rho_star": np.array([-1.0, -1.0, 0.5, 0.0]),
    "beta": np.array([0.0, -2.0, 1.0, 1.0]),
    "nu_star": np.array([1.0, 0.0, -2.0, 1.0]),
}

# Which groups each model requires to be held fixed. Fewer groups means more
# admissible transformations means more constraints; see the module docstring.
MODEL_FIXED_GROUPS: dict[str, tuple[str, ...]] = {
    "kadomtsev": ("rho_star", "beta", "nu_star"),
    "collisionless": ("rho_star", "beta"),
    "electrostatic": ("rho_star",),
}

# Ordered weakest assumption first, which is also fewest constraints first.
CONSTRAINT_MODELS: tuple[str, ...] = ("kadomtsev", "collisionless", "electrostatic")


@dataclass(frozen=True)
class SimilarityTransformation:
    """One admissible scale transformation, and how the observables move under it.

    ``length``, ``field``, ``temperature`` and ``density`` are the exponents
    ``(l, b, t, m)`` of the underlying physical scales. Everything else is
    derived from them, so this object cannot describe an inconsistent
    transformation.
    """

    length: float
    field: float
    temperature: float
    density: float

    @property
    def scales(self) -> np.ndarray:
        return np.array([self.length, self.field, self.temperature, self.density])

    @property
    def tau(self) -> float:
        """How tau_E must scale for ``Omega_i tau_E ~ B tau_E`` to be invariant."""
        return -self.field

    def group_exponents(self) -> dict[str, float]:
        """How each dimensionless group scales. Zero means the group is held fixed."""
        return {
            name: float(vector @ self.scales) for name, vector in DIMENSIONLESS_GROUPS.items()
        }

    def engineering_exponents(self) -> np.ndarray:
        """How each of the eight engineering variables scales, in feature order.

        Two of these are not independent scales but consequences, and both are
        worth stating explicitly because they are where a derivation like this
        usually goes wrong:

        ``Ip``  The safety factor ``q ~ a B kappa / (R Ip)`` is a dimensionless
                group and is held fixed by every transformation here, and so are
                ``kappa`` and ``eps``. With ``a`` and ``R`` both scaling as the
                length, that forces ``Ip ~ L B``.

        ``P``   Loss power is not free either. The stored energy is
                ``W ~ n T V ~ n T L^3`` and the confinement time is defined by
                ``tau_E = W / P``, so ``P ~ n T L^3 / tau_E``. Treating ``P`` as
                an independent scale would drop this relation and produce a
                constraint that no published law satisfies.

        ``eps``, ``kappa`` and ``M`` are dimensionless and held fixed, so their
        exponents are zero: these transformations move a machine along the
        similarity family, not into a different shape or a different fuel.
        """
        length, field = self.length, self.field
        temperature, density = self.temperature, self.density
        return np.array(
            [
                length + field,  # ip_ma, from fixed q
                field,  # bt_t
                density,  # ne_line_1e19_m3
                # p_loss_mw, from tau_E = W / P with W ~ n T L^3
                density + temperature + 3.0 * length - self.tau,
                length,  # r_m
                0.0,  # inverse_aspect_ratio, dimensionless and fixed
                0.0,  # kappa, dimensionless and fixed
                0.0,  # m_eff_amu, dimensionless and fixed
            ]
        )


def admissible_transformations(model: str) -> list[SimilarityTransformation]:
    """An orthonormal basis for the transformations ``model`` admits.

    Holding a dimensionless group fixed is one linear condition on ``(l, b, t,
    m)``; the admissible family is the null space of the conditions stacked
    together. The basis returned is whatever the SVD produces, which is not
    unique, but the *span* is, and only the span enters
    :func:`constraint_matrix`: a different basis of the same subspace gives a
    different-looking constraint matrix defining the identical feasible set.
    """
    if model not in MODEL_FIXED_GROUPS:
        raise ValueError(
            f"Unknown model {model!r}; expected one of {sorted(MODEL_FIXED_GROUPS)}."
        )
    conditions = np.vstack([DIMENSIONLESS_GROUPS[name] for name in MODEL_FIXED_GROUPS[model]])
    _, singular_values, vt = np.linalg.svd(conditions)
    tolerance = float(singular_values[0] * max(conditions.shape) * np.finfo(float).eps)
    n_independent = int((singular_values > tolerance).sum())
    null_basis = vt[n_independent:]
    return [SimilarityTransformation(*row) for row in null_basis]


def constraint_matrix(model: str, *, intercept: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """The constraint ``C b = d`` that ``model`` imposes on the exponents.

    One row per admissible transformation. A transformation scaling each
    engineering variable by ``lam^e_j`` and tau_E by ``lam^e_tau`` is consistent
    with the power law exactly when ``sum_j a_j e_j = e_tau``, so the row is the
    engineering exponent vector and the right-hand side is the tau exponent.

    With ``intercept=True`` a leading zero column is prepended, because the
    design matrix built by ``scaling_law.build_log_design_matrix`` carries the
    intercept first and no constraint here touches the multiplying coefficient:
    these are statements about how tau_E *scales*, which say nothing about ``C``.
    """
    transformations = admissible_transformations(model)
    rows = np.vstack([transform.engineering_exponents() for transform in transformations])
    rhs = np.array([transform.tau for transform in transformations])
    if intercept:
        rows = np.column_stack([np.zeros(len(rows)), rows])
    return rows, rhs


def constraint_residuals(exponents: dict[str, float] | np.ndarray) -> pd.DataFrame:
    """How far a set of exponents sits from each model's constraint surface.

    The residual is ``||C b - d||`` in the least-squares sense of the rows, which
    is zero exactly when the exponents are consistent with that model's physics.
    Applied to ``IPB98Y2_EXPONENTS`` this is the check described in the module
    docstring; applied to a free refit it measures which physics the data
    declines to obey.
    """
    if isinstance(exponents, dict):
        vector = np.array([exponents[name] for name in CONSTRAINED_FEATURE_COLUMNS])
    else:
        vector = np.asarray(exponents, dtype=float)
    if vector.shape != (len(CONSTRAINED_FEATURE_COLUMNS),):
        raise ValueError(
            f"Expected {len(CONSTRAINED_FEATURE_COLUMNS)} exponents, got {vector.shape}."
        )
    records = []
    for model in CONSTRAINT_MODELS:
        rows, rhs = constraint_matrix(model, intercept=False)
        violation = rows @ vector - rhs
        records.append(
            {
                "model": model,
                "n_constraints": int(rows.shape[0]),
                "residual_norm": float(np.linalg.norm(violation)),
                "max_abs_violation": float(np.max(np.abs(violation))),
            }
        )
    return pd.DataFrame(records)


def fit_constrained_power_law(
    frame: pd.DataFrame,
    target_column: str,
    model: str,
    *,
    feature_columns: tuple[str, ...] = CONSTRAINED_FEATURE_COLUMNS,
) -> dict[str, float]:
    """Least squares for the exponents, subject to ``model``'s physics.

    ``model="free"`` runs the ordinary unconstrained SVD fit, so the hierarchy
    has an anchor scored by exactly the same code path as its constrained rungs.
    """
    design, _ = sl.build_log_design_matrix(frame, feature_columns, intercept=True)
    target = np.log(frame[target_column].to_numpy(dtype=float))
    with clean_fp_state():
        if model == "free":
            beta = sl.solve_lstsq_svd(design, target)
        else:
            rows, rhs = constraint_matrix(model, intercept=True)
            beta = sl.solve_constrained_lstsq(design, target, rows, rhs)
    return {
        sl.INTERCEPT_NAME: float(beta[0]),
        **{name: float(value) for name, value in zip(feature_columns, beta[1:], strict=True)},
    }


class ConstrainedPowerLaw(RegressorMixin, BaseEstimator):
    """A log-linear power law fitted under one rung of the Connor-Taylor hierarchy.

    ``RegressorMixin`` leads for the same reason it does in
    ``hdb5.PowerLawResidualHybrid``: scikit-learn resolves estimator tags along
    the MRO, and with ``BaseEstimator`` first the mixin's ``__sklearn_tags__``
    never runs and ``is_regressor`` returns False for a regressor.

    Fits and predicts in log space, matching every other model in the zoo, so
    ``X`` is a frame of ``log_*`` columns and ``y`` is ``log tau``. The
    constraint is stated on the eight engineering exponents, so this selects
    those columns out of whatever it is handed rather than using all of them;
    see :data:`CONSTRAINED_FEATURE_COLUMNS` for why ``log_a_m`` is excluded.

    There is no penalty and no hyperparameter. That is the point of the
    comparison: any difference from ``ridge_loglinear`` out of distribution is
    the physics assumption doing the work, not a tuned amount of shrinkage.
    """

    def __init__(self, model: str = "kadomtsev") -> None:
        self.model = model

    def _log_columns(self) -> list[str]:
        return [f"log_{column}" for column in CONSTRAINED_FEATURE_COLUMNS]

    def _select(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        columns = self._log_columns()
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "ConstrainedPowerLaw needs a DataFrame so it can select the eight "
                "engineering log columns the constraint is stated on."
            )
        missing = [column for column in columns if column not in X.columns]
        if missing:
            raise ValueError(f"Input frame is missing required columns: {missing}")
        values = X.loc[:, columns].to_numpy(dtype=float)
        return np.column_stack([np.ones(len(values)), values])

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray) -> "ConstrainedPowerLaw":
        if self.model != "free" and self.model not in MODEL_FIXED_GROUPS:
            raise ValueError(
                f"Unknown model {self.model!r}; expected 'free' or one of "
                f"{sorted(MODEL_FIXED_GROUPS)}."
            )
        design = self._select(X)
        target = np.asarray(y, dtype=float)
        with clean_fp_state():
            if self.model == "free":
                self.coefficients_ = sl.solve_lstsq_svd(design, target)
            else:
                rows, rhs = constraint_matrix(self.model, intercept=True)
                self.coefficients_ = sl.solve_constrained_lstsq(design, target, rows, rhs)
        self.exponents_ = {
            name: float(value)
            for name, value in zip(CONSTRAINED_FEATURE_COLUMNS, self.coefficients_[1:], strict=True)
        }
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        with clean_fp_state():
            return np.asarray(self._select(X) @ self.coefficients_, dtype=float)


def build_constrained_models(
    models: tuple[str, ...] = ("free", *CONSTRAINT_MODELS),
) -> dict[str, object]:
    """One estimator per rung, wrapped so they drop into the existing zoo.

    Imported lazily to keep ``dimensional`` free of a hard dependency on
    ``hdb5``, which imports this module's sibling ``scaling_law`` already.
    """
    from sklearn.pipeline import Pipeline

    return {
        f"powerlaw_{model}": Pipeline([("model", ConstrainedPowerLaw(model=model))])
        for model in models
    }
