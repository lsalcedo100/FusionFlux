"""Shrinking a scaling law toward the physics prior, along the directions the data cannot resolve.

Result 3 measured something specific and then did nothing with it. Refitting
IPB98(y,2) from this database disagrees with the published exponents, and **77%
of that disagreement lies in the single weakest singular direction of the design
matrix, which carries 0.3% of the matrix's variance**. The disagreement is
concentrated exactly where the data has no opinion: plasma current traded
against machine size, two quantities tokamaks are not built to vary
independently.

That is a diagnosis with an obvious prescription, and ``results/RESULTS.md``
listed the absence of it as a limitation in as many words: the flexibility sweep
varies polynomial degree "under an isotropic L2 penalty throughout", so it
supports "adding unconstrained polynomial freedom costs the tail" and not
anything about *targeted* regularization. A penalty "aimed at the weak direction
from Result 3" is named there as untested. This module tests it.

Two ways to inject a physics prior, one of which is provably vacuous
-------------------------------------------------------------------
Result 1 found that the IPB98 prediction, added to the feature matrix as a
tenth feature, is an exact log-linear combination of the other eight. So for a
log-linear model it contributes *nothing*: the fit spans the same function space
with or without it, and adding it only makes the design matrix rank deficient.
A published physics scaling, handed to the model as data, is not information.

The same law used as a *shrinkage target* is a different object entirely. It
does not enlarge the space of functions the model can express; it picks which
member of that space the fit lands on when the data is indifferent. Where the
data is indifferent is precisely what the singular value spectrum measures. So:

    minimize  ||X b - y||^2  +  alpha * || W V^T (b - b_prior) ||^2

with ``V`` the right singular vectors of the design matrix and ``W`` a diagonal
weight per direction. The estimator is the same information, injected the other
way round, and unlike the feature version it is not a no-op.

Three weightings, from blunt to targeted
----------------------------------------
``isotropic``   ``W = I``. Ordinary ridge, recentred on the prior instead of on
                zero. Every direction is pulled toward IPB98 equally hard,
                whether or not the data has an opinion about it. This is the
                control: it establishes what a *non-targeted* prior buys, so any
                extra from the two below is attributable to the targeting rather
                than to the prior.

``spectral``    ``W = diag(s_1 / s_i)``. The penalty on direction ``i`` scales as
                ``(s_1 / s_i)^2``, so a direction carrying a thousandth of the
                variance is penalised a million times harder than the strongest
                one. Well-determined directions are left almost free and the
                fit is pinned to the prior in the weak ones. This is the literal
                "penalty aimed at the weak direction".

``truncated``   The hard version, with no penalty at all: take the data's answer
                in the ``k`` best-determined directions and the prior's in every
                other. Its parameter is an integer rank rather than a
                continuous alpha, which makes it the interpretable endpoint of
                the same idea. Result 3 predicts something sharp here: if 77% of
                the disagreement really does live in the last direction, then
                dropping only that one (``k = p - 1``) should capture most of
                whatever the prior is worth, and the remaining rungs should add
                little.

All three are computed through the same per-direction filter on a single SVD,
the way ``scaling_law.ridge_from_svd`` handles a penalty grid: the factorization
does not depend on the parameter, so a whole sweep costs one decomposition.

What is shrunk, and what is not
-------------------------------
The exponents are shrunk toward the prior; the multiplying coefficient is always
fitted from the data. That asymmetry is deliberate. The exponents are the
physics, and are what a similarity argument constrains. The coefficient ``C``
absorbs unit conventions and the composition of whatever population was fitted,
and refitting it against a new database while keeping published exponents is
ordinary practice in the field rather than a liberty taken here. It also makes
the ``alpha -> infinity`` endpoint a well-defined model: IPB98(y,2)'s exponents,
renormalised to this database.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

import scaling_law as sl
from scaling_law import _clean_fp_state as clean_fp_state

# The eight engineering exponents, matching ``dimensional`` and for the same
# reason: ``log_a_m`` is an exact combination of ``log_r_m`` and
# ``log_inverse_aspect_ratio``, so including it makes the design rank deficient
# and the "weak direction" this module targets ill-defined.
PRIOR_FEATURE_COLUMNS: tuple[str, ...] = sl.IPB98_FEATURE_COLUMNS

WEIGHTINGS: tuple[str, ...] = ("isotropic", "spectral")

# Nine decades, matching the penalty axis of ``analysis_flexibility_sweep`` so
# the two sweeps can be read against each other.
ALPHA_GRID: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5)


def prior_exponent_vector() -> np.ndarray:
    """IPB98(y,2)'s eight exponents, in :data:`PRIOR_FEATURE_COLUMNS` order."""
    return np.array([sl.IPB98Y2_EXPONENTS[name] for name in PRIOR_FEATURE_COLUMNS])


def direction_filters(
    singular_values: np.ndarray, alpha: float, weighting: str
) -> np.ndarray:
    """Per-direction shrinkage factors toward the prior, one per singular direction.

    A factor of 1 means "take the data's answer in this direction", 0 means
    "take the prior's". Written as an explicit filter rather than folded into a
    matrix inverse because the filter *is* the result: it shows, per direction,
    how much of the fit the prior is deciding.

    For ``isotropic`` this is ``s^2 / (s^2 + alpha)``, exactly
    ``scaling_law.ridge_shrinkage_factors``. For ``spectral`` the penalty in
    direction ``i`` is scaled by ``(s_1 / s_i)^2``, giving
    ``s^4 / (s^4 + alpha * s_1^2)``, which falls away far faster as ``s``
    shrinks.
    """
    s = np.asarray(singular_values, dtype=float)
    if alpha < 0:
        raise ValueError("alpha must be non-negative.")
    if weighting == "isotropic":
        return s**2 / (s**2 + alpha)
    if weighting == "spectral":
        return s**4 / (s**4 + alpha * s[0] ** 2)
    raise ValueError(f"Unknown weighting {weighting!r}; expected one of {WEIGHTINGS}.")


class SpectralPriorRidge(RegressorMixin, BaseEstimator):
    """Log-linear power law shrunk toward IPB98(y,2) along ill-determined directions.

    ``RegressorMixin`` leads for the same scikit-learn tag-resolution reason as
    in ``hdb5.PowerLawResidualHybrid`` and ``dimensional.ConstrainedPowerLaw``.

    Parameters
    ----------
    weighting
        ``"isotropic"``, ``"spectral"`` (see :func:`direction_filters`), or
        ``"truncated"`` for the hard rank cut, which ignores ``alpha`` and uses
        ``n_data_directions``.
    alpha
        Penalty strength. ``0`` reproduces the unconstrained least-squares fit
        in every weighting; large alpha leaves the prior's exponents.
    n_data_directions
        For ``"truncated"``: how many of the best-determined directions to take
        from the data. ``None`` means all of them, which is again the plain fit.

    The standardization is not cosmetic. Singular directions of a raw log design
    matrix are set by whichever column happens to carry the largest units, so
    "the weakest direction" would be a statement about unit choice rather than
    about collinearity. ``scaling_law.analyze_conditioning`` standardizes for
    exactly this reason and Result 3's weak direction is defined in those
    coordinates, so this shrinks in them too and maps back afterwards.
    """

    def __init__(
        self,
        weighting: str = "spectral",
        *,
        alpha: float = 1.0,
        n_data_directions: int | None = None,
    ) -> None:
        self.weighting = weighting
        self.alpha = alpha
        self.n_data_directions = n_data_directions

    def _log_columns(self) -> list[str]:
        return [f"log_{column}" for column in PRIOR_FEATURE_COLUMNS]

    def _select(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "SpectralPriorRidge needs a DataFrame so it can select the eight "
                "engineering log columns the prior is stated on."
            )
        columns = self._log_columns()
        missing = [column for column in columns if column not in X.columns]
        if missing:
            raise ValueError(f"Input frame is missing required columns: {missing}")
        return X.loc[:, columns].to_numpy(dtype=float)

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray) -> "SpectralPriorRidge":
        raw = self._select(X)
        target = np.asarray(y, dtype=float)

        # Standardize the features and centre the target, so the intercept drops
        # out of the penalised problem entirely and is recovered at the end.
        self.feature_mean_ = raw.mean(axis=0)
        spread = raw.std(axis=0)
        # A column with no spread carries no direction to shrink along; leaving
        # its scale at 1 keeps the standardization a no-op there instead of
        # dividing by zero. Cannot happen on the cleaned database, but a caller
        # slicing a single machine can produce it.
        self.feature_scale_ = np.where(spread > 0, spread, 1.0)
        standardized = (raw - self.feature_mean_) / self.feature_scale_

        # The prior's exponents in the same standardized coordinates. This is the
        # ``to_analysis_coordinates`` mapping from ``scaling_law``: a coefficient
        # on a raw log column becomes ``coefficient * scale`` on the standardized
        # one. Skipping it silently compares vectors in two coordinate systems.
        prior_standardized = prior_exponent_vector() * self.feature_scale_

        target_mean = target.mean()
        with clean_fp_state():
            residual = (target - target_mean) - standardized @ prior_standardized
            u, s, vt = np.linalg.svd(standardized, full_matrices=False)
            projected = u.T @ residual

        if self.weighting == "truncated":
            keep = len(s) if self.n_data_directions is None else int(self.n_data_directions)
            if not 0 <= keep <= len(s):
                raise ValueError(f"n_data_directions must be in [0, {len(s)}], got {keep}.")
            filters = np.zeros_like(s)
            filters[:keep] = 1.0
        else:
            filters = direction_filters(s, self.alpha, self.weighting)
        self.direction_filters_ = filters
        self.singular_values_ = s

        # ``filters / s`` is the per-direction filter divided by the singular
        # value, which is the pseudoinverse with each direction damped. Where a
        # filter is exactly zero the direction is dropped rather than inverted,
        # so a zero singular value never divides.
        scaled = np.divide(filters, s, out=np.zeros_like(s), where=filters > 0)
        with clean_fp_state():
            correction = vt.T @ (scaled * projected)
        coefficients_standardized = prior_standardized + correction

        # Back to raw log units, and recover the intercept from the centring.
        self.exponents_ = coefficients_standardized / self.feature_scale_
        self.intercept_ = float(target_mean - self.feature_mean_ @ self.exponents_)
        self.exponent_map_ = {
            name: float(value)
            for name, value in zip(PRIOR_FEATURE_COLUMNS, self.exponents_, strict=True)
        }
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        with clean_fp_state():
            return np.asarray(self._select(X) @ self.exponents_ + self.intercept_, dtype=float)


def build_prior_shrinkage_models(
    *,
    alpha_grid: tuple[float, ...] = ALPHA_GRID,
    weightings: tuple[str, ...] = WEIGHTINGS,
    truncation_ranks: tuple[int, ...] = tuple(range(len(PRIOR_FEATURE_COLUMNS) + 1)),
) -> dict[str, object]:
    """One estimator per rung of the two continuous sweeps and the rank sweep.

    Wrapped in ``Pipeline`` so they drop into the existing zoo, ``clone_pipeline``
    and all three splits with no special-casing, exactly as the hybrids do.
    """
    from sklearn.pipeline import Pipeline

    models: dict[str, object] = {}
    for weighting in weightings:
        for alpha in alpha_grid:
            name = f"prior_{weighting}_a{alpha:g}".replace(".", "p").replace("-", "m")
            models[name] = Pipeline(
                [("model", SpectralPriorRidge(weighting=weighting, alpha=alpha))]
            )
    # The full rank range, endpoints included: ``k = p`` is the unconstrained
    # fit and ``k = 0`` is the prior's exponents with the coefficient refitted.
    # Both are interpretable models already reported elsewhere, which is what
    # makes the sweep between them a frontier rather than a free-floating family.
    for rank in truncation_ranks:
        models[f"prior_truncated_k{rank}"] = Pipeline(
            [("model", SpectralPriorRidge(weighting="truncated", n_data_directions=rank))]
        )
    return models


def prior_model_name(weighting: str, *, alpha: float | None = None, rank: int | None = None) -> str:
    """The zoo key for one rung, so callers never rebuild the naming by hand."""
    if weighting == "truncated":
        if rank is None:
            raise ValueError("The truncated weighting is named by rank, not alpha.")
        return f"prior_truncated_k{rank}"
    if alpha is None:
        raise ValueError(f"The {weighting} weighting is named by alpha.")
    return f"prior_{weighting}_a{alpha:g}".replace(".", "p").replace("-", "m")
