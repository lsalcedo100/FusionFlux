"""Result 14: separating flexibility from boundedness, with Gaussian processes.

``results/RESULTS.md`` Result 4d and 4e establish that adding polynomial freedom
to the log-linear power law costs the tail: degree 1's worst machine of thirteen
is 0.289, degree 2's is 1.083, degree 3's is 4.601. Result 4e sweeps the ridge
penalty across nine decades and finds no setting that rescues a flexible form.

The limitations section says exactly what that does and does not support:

    What the sweep does not vary is the *kind* of flexibility. It is polynomial
    degree under an isotropic L2 penalty throughout, so it supports "adding
    unconstrained polynomial freedom costs the tail" and not the broader
    "flexibility is bad". A differently constrained flexible model, one whose
    penalty targeted the weak direction from Result 3, or a Gaussian process
    with a physically motivated kernel, is untested here.

This module is that test, and it is built to separate two properties that every
model scored so far confounds.

**Flexibility** is how much structure a model can learn beyond a power law.
**Boundedness** is what it does when asked about a point far outside the
training data. A random forest is flexible and bounded: it averages training
targets, so it cannot return a value above the largest one it saw (Result 4c).
Ridge is inflexible and unbounded. Polynomial ridge is flexible and unbounded,
and Result 4d uses exactly that to argue the form matters rather than the
flexibility. But a polynomial's unboundedness is violent: degree 3 diverges, and
that is why its tail is 4.601. So the ladder so far offers no model that is
flexible, unbounded, and *well behaved* far from the data.

A Gaussian process supplies one, and the property is a choice of kernel rather
than a hyperparameter. Three rungs, all fitted by the same optimizer on the same
rows, differing only in what the kernel does at long range:

    gp_rbf          bounded. An RBF kernel decays to zero with distance, so far
                    from the data the posterior reverts to the prior mean, which
                    ``normalize_y`` puts at the training mean of log tau. This is
                    a tree ensemble's failure mode reached by different
                    machinery: not a shortfall but a hard limit, and the
                    prediction is that it fails like one.

    gp_linear       unbounded, inflexible. A dot-product kernel is Bayesian
                    linear regression in the log features, which is a power law.
                    The control: it should land on top of ``ridge_loglinear``,
                    and if it does not, something in this module is wrong rather
                    than interesting.

    gp_linear_rbf   unbounded and flexible, which is the rung that did not exist
                    before. The power law carries the extrapolation; the RBF term
                    learns bounded departures from it that decay to nothing as
                    the query leaves the data. This is the physically motivated
                    kernel the limitation names: dimensional analysis says the
                    law is a power law (Result 8), so the linear part is the
                    physics and the RBF part is the correction.

The design is the point. Only the asymptotic behaviour of the kernel changes
across the three, so a difference between them cannot be attributed to the
optimizer, the feature set, the split or the amount of data.

Why the hyperparameters are learned rather than chosen. An exact GP costs
O(n^3), and marginal-likelihood optimization pays that at every L-BFGS step, so
the obvious shortcut is to fix the kernel by hand. Doing that here produced
``gp_linear_rbf`` at 1.195 on the ITER-matched cut, apparently a failure, purely
because the hand-picked RBF length scale was an order of magnitude longer than
the one the data supports (0.31 in standardized units). That is precisely the
objection Result 4e exists to remove from the polynomial ladder, and it would be
fatal here: a flexible model that fails because its hyperparameters were guessed
tells you nothing about flexibility. So every kernel is fitted by maximising the
log marginal likelihood, and :class:`SubsampledGaussianProcess` makes that
affordable by tuning on a seeded subsample of the *training fold* and then
fitting exactly on all of that fold's rows with the learned kernel held fixed.

Nothing in the tuning ever sees a held-out row. ``tests/test_gp.py`` pins the
subsample independence, the control rung against ridge, and the bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    Kernel,
    WhiteKernel,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import hdb5

RANDOM_STATE = hdb5.RANDOM_STATE

# Rows used to fit the kernel hyperparameters. The exact GP that follows uses
# every training row; this bounds only the O(n^3)-per-step optimization.
#
# 1000 is measured rather than picked: the learned kernel moves by less than 4%
# in every hyperparameter between 750 and 1500 tuning rows on the ITER-matched
# training set, so the fit is not sensitive to this number in the range where it
# is affordable. tests/test_gp.py asserts that stability.
DEFAULT_TUNING_ROWS = 1000

# Names in the order the results report them: bounded, then the control, then
# the rung that is both flexible and unbounded.
KERNEL_NAMES: tuple[str, ...] = ("rbf", "linear", "linear_rbf")

MODEL_LABELS: dict[str, str] = {
    "gp_rbf": "GP, RBF kernel (bounded)",
    "gp_linear": "GP, linear kernel (a power law)",
    "gp_linear_rbf": "GP, linear + RBF (flexible, unbounded)",
}


def build_kernel(name: str) -> Kernel:
    """One rung of the ladder, as a starting point for marginal-likelihood fitting.

    The initial values are deliberately the same across rungs wherever a rung has
    the term at all, so the three fits start from the same place and differ only
    in which terms exist. Bounds are wide enough that the optimizer, not this
    function, decides the answer.
    """
    white = WhiteKernel(noise_level=0.05, noise_level_bounds=(1e-6, 1.0))
    linear = ConstantKernel(1.0, (1e-3, 1e3)) * DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 1e3))
    rbf = ConstantKernel(0.5, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))

    if name == "rbf":
        return rbf + white
    if name == "linear":
        return linear + white
    if name == "linear_rbf":
        return linear + rbf + white
    raise ValueError(f"Unknown kernel {name!r}; expected one of {KERNEL_NAMES}")


class SubsampledGaussianProcess(RegressorMixin, BaseEstimator):
    """Exact GP whose kernel hyperparameters are tuned on a subsample.

    Two stages, and the split between them is what makes this affordable:

    1. Draw ``n_tuning_rows`` rows from the training data with a seeded
       generator and maximise the log marginal likelihood on those alone.
    2. Fit the exact GP on *every* training row with the learned kernel frozen.

    Stage 2 is one Cholesky factorization. Stage 1 is many, which is why it is
    the one that gets a subsample. Both see only the rows handed to ``fit``, so
    when a caller passes a training fold, no held-out row enters either stage.

    ``normalize_y`` centres the target, which matters for interpreting the RBF
    rung rather than for its accuracy: it puts the prior mean at the training
    mean of log tau, so "reverts to the prior far from the data" and "returns
    the average training target" are the same statement, and the parallel with
    a tree ensemble is exact rather than approximate.
    """

    def __init__(
        self,
        kernel_name: str = "linear_rbf",
        *,
        n_tuning_rows: int = DEFAULT_TUNING_ROWS,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.kernel_name = kernel_name
        self.n_tuning_rows = n_tuning_rows
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> SubsampledGaussianProcess:
        features = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float)

        generator = np.random.default_rng(self.random_state)
        n_tuning = min(self.n_tuning_rows, len(features))
        tuning_rows = generator.choice(len(features), n_tuning, replace=False)

        tuned = GaussianProcessRegressor(
            kernel=build_kernel(self.kernel_name),
            normalize_y=True,
            n_restarts_optimizer=0,
            random_state=self.random_state,
        ).fit(features[tuning_rows], target[tuning_rows])

        self.kernel_ = tuned.kernel_
        self.n_tuning_rows_ = n_tuning
        self.log_marginal_likelihood_ = float(tuned.log_marginal_likelihood_value_)
        # optimizer=None freezes the kernel: this is a single factorization on
        # the full fold, not a second search.
        self.gp_ = GaussianProcessRegressor(
            kernel=tuned.kernel_, optimizer=None, normalize_y=True
        ).fit(features, target)
        self.training_target_mean_ = float(target.mean())
        self.training_target_max_ = float(target.max())
        return self

    def predict(self, X: Any, return_std: bool = False) -> Any:
        return self.gp_.predict(np.asarray(X, dtype=float), return_std=return_std)


def build_gp_models(
    *,
    n_tuning_rows: int = DEFAULT_TUNING_ROWS,
    random_state: int = RANDOM_STATE,
) -> dict[str, Pipeline]:
    """The three rungs, as pipelines the existing splits can score unchanged.

    Standardization is part of the pipeline rather than done once outside it,
    so each fold's scaler is fitted on that fold's training rows only. The RBF
    length scale is in those standardized units, which is what makes it
    comparable across folds and across the three rungs.
    """
    return {
        f"gp_{name}": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    SubsampledGaussianProcess(
                        kernel_name=name,
                        n_tuning_rows=n_tuning_rows,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        for name in KERNEL_NAMES
    }


@dataclass(frozen=True)
class ReversionDiagnostic:
    """How far a model's predictions collapse toward the training mean.

    Result 4c measures a tree ensemble's failure as a hard ceiling: no tree can
    output a value above the largest training target. An RBF Gaussian process
    fails in the same direction by a different route, reverting to its prior
    mean, and this quantity is the analogue that applies to both.

    ``reversion`` is 0 when a model's predictions have the same spread as the
    truth and 1 when it has collapsed to a constant. It is measured on the
    held-out rows, in log space, where the errors are.
    """

    model_name: str
    predicted_spread: float
    actual_spread: float
    reversion: float
    predicted_mean_offset: float


def reversion_diagnostic(
    model_name: str,
    log_actual: np.ndarray,
    log_predicted: np.ndarray,
    training_log_mean: float,
) -> ReversionDiagnostic:
    """Spread of the predictions against the spread of the truth."""
    predicted_spread = float(np.std(log_predicted))
    actual_spread = float(np.std(log_actual))
    reversion = 1.0 - predicted_spread / actual_spread if actual_spread > 0 else float("nan")
    return ReversionDiagnostic(
        model_name=model_name,
        predicted_spread=predicted_spread,
        actual_spread=actual_spread,
        reversion=float(np.clip(reversion, -np.inf, 1.0)),
        predicted_mean_offset=float(np.mean(log_predicted) - training_log_mean),
    )
