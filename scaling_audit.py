"""A reusable extrapolation audit, with nothing about tokamaks in it.

Everything else in this repository is a study of one database. This module is
the part of that study that is not about fusion at all: the finding is that
grouped cross-validation can invert the ranking of models in a scaling-law
domain, and that the inversion is visible in advance from two cheap diagnostics
and repairable by a linear constraint. None of that reasoning mentions plasma.

So it is packaged here for someone with a different dataset, in three pieces:

* :func:`audit_groups` runs leave-one-group-out and reports, per group, not only
  the score but *why* the score is what it is: how far the held-out group sits
  from the training data, and how much of its target lies above anything a
  range-bounded model can emit. Reporting the score alone is what lets a
  reversal look like noise.
* :class:`OrderedGroupSplit` is the harder question, as a scikit-learn splitter.
  Leave-one-group-out still surrounds the held-out group with others like it.
  Ordering groups along an axis and predicting the far end does not, and that is
  the split a scaling law actually has to survive.
* :class:`ConstrainedLinearRegression` is equality-constrained least squares as
  an estimator, so a dimensional-analysis constraint is a constructor argument
  rather than a rewrite.

The three are independent; use whichever applies. Nothing here imports ``hdb5``
or any other module of the study, and ``tests/test_scaling_audit.py`` exercises
all of it on a synthetic non-physics problem as well as against the real
pipeline, so the claim that it is domain-agnostic is tested rather than stated.

A worked example, on any frame with a group column::

    from scaling_audit import audit_groups
    report = audit_groups(
        X=frame[features], y=np.log(frame["target"]), groups=frame["machine"],
        estimators={"ridge": Ridge(), "forest": RandomForestRegressor()},
    )
    report.groupby("estimator")["rmse"].mean()

Scores are plain RMSE on whatever ``y`` is handed over. Pass ``log`` targets and
that is RMSLE, which is what the study reports; the transform is the caller's
because only the caller knows whether their target is positive.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone


@contextmanager
def _clean_fp_state() -> Iterator[None]:
    """Silence spurious NumPy 2.x BLAS floating-point flags raised by ``matmul``.

    On some BLAS backends ``matmul`` reports divide-by-zero, overflow or invalid
    value on inputs that are entirely finite and results correct to machine
    precision: the flag is left over from earlier vectorized work rather than
    produced by this operation. Scoped to individual calls so a genuine overflow
    elsewhere is still reported. Mirrors ``scaling_law._clean_fp_state``, kept
    local so this module stands alone.
    """
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        yield

__all__ = [
    "ConstrainedLinearRegression",
    "GroupDiagnostic",
    "OrderedGroupSplit",
    "audit_groups",
    "distance_score_correlation",
    "group_diagnostic",
    "summarize",
]


# ---------------------------------------------------------------------------
# Diagnostics: the two numbers that make an extrapolation failure legible.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GroupDiagnostic:
    """How far one held-out group sits outside the data used to train."""

    group: str
    n_train_rows: int
    n_held_out_rows: int
    mahalanobis: float
    """Distance between held-out and training feature means, in training units.

    Computed through the pseudo-inverse of the training covariance rather than
    its inverse. Real design matrices in these domains are routinely rank
    deficient (one feature derived from two others is enough), and a plain
    inverse turns that into an arbitrary large number instead of an error.
    """
    n_features_outside_train_range: int
    fraction_above_train_max: float
    """Share of held-out targets above the largest value seen in training.

    This is the one that predicts hard failure rather than degradation. Any
    model that predicts by averaging training targets -- every tree ensemble,
    every nearest-neighbour method -- cannot emit a value above its training
    maximum, so this fraction is a floor on the rows it must get wrong however
    well it is tuned.
    """
    fraction_below_train_min: float
    log_target_headroom: float
    """``max(held) - max(train)`` on the target as handed in.

    Positive means the group asks for values the training range does not cover.
    """


def group_diagnostic(
    X: pd.DataFrame | np.ndarray,
    y: Sequence[float] | np.ndarray,
    train_index: np.ndarray,
    test_index: np.ndarray,
    *,
    group: str = "",
) -> GroupDiagnostic:
    """Measure one split without fitting anything.

    Cheap enough to run before choosing a model, which is the point: both
    quantities are known in advance of any target values for the held-out
    group, so they are available for a device or a batch that does not exist
    yet.
    """
    features = np.asarray(X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else X, dtype=float)
    target = np.asarray(y, dtype=float)
    train_features, test_features = features[train_index], features[test_index]
    train_target, test_target = target[train_index], target[test_index]

    difference = test_features.mean(axis=0) - train_features.mean(axis=0)
    covariance = np.atleast_2d(np.cov(train_features, rowvar=False))
    with _clean_fp_state():
        quadratic = float(difference @ np.linalg.pinv(covariance) @ difference)

    minimum, maximum = train_features.min(axis=0), train_features.max(axis=0)
    median = np.median(test_features, axis=0)

    return GroupDiagnostic(
        group=str(group),
        n_train_rows=int(train_index.size),
        n_held_out_rows=int(test_index.size),
        mahalanobis=float(np.sqrt(max(quadratic, 0.0))),
        n_features_outside_train_range=int(np.sum((median < minimum) | (median > maximum))),
        fraction_above_train_max=float(np.mean(test_target > train_target.max())),
        fraction_below_train_min=float(np.mean(test_target < train_target.min())),
        log_target_headroom=float(test_target.max() - train_target.max()),
    )


# ---------------------------------------------------------------------------
# The audit.
# ---------------------------------------------------------------------------
def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))


def audit_groups(
    X: pd.DataFrame,
    y: Sequence[float] | np.ndarray,
    groups: Sequence[Any] | np.ndarray | pd.Series,
    estimators: Mapping[str, Any],
    *,
    min_held_out_rows: int = 1,
    scorer: Callable[[np.ndarray, np.ndarray], float] = _rmse,
) -> pd.DataFrame:
    """Leave-one-group-out, scored per group and per estimator, with diagnostics.

    Returns one row per (group, estimator) carrying the score alongside the
    :class:`GroupDiagnostic` fields for that split, so the score can be read
    against the distance rather than on its own. Groups with fewer than
    ``min_held_out_rows`` rows are skipped: a group of three rows produces a
    score with an enormous standard error and, averaged in unweighted, moves the
    headline more than the machines the claim is about.

    Estimators are cloned before each fit, so one call does not leak state
    between folds or mutate what the caller passed.
    """
    if not estimators:
        raise ValueError("audit_groups needs at least one estimator.")

    target = np.asarray(y, dtype=float)
    labels = np.asarray(pd.Series(groups).to_numpy())
    if not (len(X) == target.size == labels.size):
        raise ValueError(
            f"X, y and groups must agree in length; got {len(X)}, {target.size}, {labels.size}."
        )

    records: list[dict[str, Any]] = []
    for group in pd.unique(labels):
        held = labels == group
        test_index = np.flatnonzero(held)
        train_index = np.flatnonzero(~held)
        if test_index.size < min_held_out_rows or train_index.size == 0:
            continue

        diagnostic = group_diagnostic(X, target, train_index, test_index, group=str(group))
        train_X = X.iloc[train_index] if isinstance(X, pd.DataFrame) else X[train_index]
        test_X = X.iloc[test_index] if isinstance(X, pd.DataFrame) else X[test_index]

        for name, estimator in estimators.items():
            fitted = clone(estimator).fit(train_X, target[train_index])
            predicted = np.asarray(fitted.predict(test_X), dtype=float)
            records.append(
                {
                    "group": diagnostic.group,
                    "estimator": name,
                    "score": scorer(target[test_index], predicted),
                    "n_train_rows": diagnostic.n_train_rows,
                    "n_held_out_rows": diagnostic.n_held_out_rows,
                    "mahalanobis": diagnostic.mahalanobis,
                    "n_features_outside_train_range": diagnostic.n_features_outside_train_range,
                    "fraction_above_train_max": diagnostic.fraction_above_train_max,
                    "fraction_below_train_min": diagnostic.fraction_below_train_min,
                    "log_target_headroom": diagnostic.log_target_headroom,
                    # A prediction that never exceeds the training maximum is the
                    # signature of a range-bounded learner, and it is worth
                    # recording per fold rather than inferred from the model type:
                    # a pipeline can hide the ensemble inside it.
                    "prediction_bounded_by_train_range": bool(
                        predicted.max() <= target[train_index].max() + 1e-12
                    ),
                }
            )

    if not records:
        raise ValueError(
            f"No group had at least {min_held_out_rows} rows with a non-empty training set."
        )
    return pd.DataFrame.from_records(records)


def distance_score_correlation(report: pd.DataFrame) -> pd.Series:
    """Spearman correlation of per-group score against extrapolation distance.

    The single most useful summary of an audit. A model whose error tracks
    distance is failing *because* the group is far away, which is a statement
    about what it will do on the next group; a model whose error does not is
    failing for reasons that do not grow.
    """
    return (
        report.groupby("estimator")[["score", "mahalanobis"]]
        .apply(lambda frame: frame["score"].corr(frame["mahalanobis"], method="spearman"))
        .rename("distance_spearman")
    )


# ---------------------------------------------------------------------------
# The harder split.
# ---------------------------------------------------------------------------
class OrderedGroupSplit:
    """Cumulative splits along an ordering of the groups.

    Cut ``k`` trains on the ``k`` lowest-ranked groups and predicts every
    remaining one, so the held-out set is entirely beyond the training data on
    whatever axis the ordering encodes. Where leave-one-group-out extrapolates
    in group identity while interpolating in everything else, this does not:
    that is the difference between predicting a machine like the ones you have
    and predicting one larger than all of them.

    ``order`` maps each group label to a scalar. Ties are broken by label so the
    sequence of cuts is deterministic.

    Follows the scikit-learn splitter protocol (``split`` and ``get_n_splits``),
    so it drops into ``cross_validate`` and friends.
    """

    def __init__(
        self,
        order: Mapping[Any, float],
        *,
        min_train_groups: int = 2,
        min_test_rows: int = 1,
    ) -> None:
        self.order = dict(order)
        self.min_train_groups = min_train_groups
        self.min_test_rows = min_test_rows

    def _ranked(self, labels: np.ndarray) -> list[Any]:
        present = [g for g in pd.unique(labels) if g in self.order]
        missing = [g for g in pd.unique(labels) if g not in self.order]
        if missing:
            raise ValueError(f"No ordering value for group(s): {sorted(map(str, missing))}")
        return sorted(present, key=lambda g: (self.order[g], str(g)))

    def split(
        self,
        X: Any = None,
        y: Any = None,
        groups: Sequence[Any] | np.ndarray | pd.Series | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if groups is None:
            raise ValueError("OrderedGroupSplit needs `groups`; the split is defined by them.")
        labels = np.asarray(pd.Series(groups).to_numpy())
        ranked = self._ranked(labels)

        for k in range(self.min_train_groups, len(ranked)):
            train_groups = set(ranked[:k])
            in_train = np.isin(labels, list(train_groups))
            train_index = np.flatnonzero(in_train)
            test_index = np.flatnonzero(~in_train)
            if test_index.size < self.min_test_rows or train_index.size == 0:
                continue
            yield train_index, test_index

    def get_n_splits(
        self,
        X: Any = None,
        y: Any = None,
        groups: Sequence[Any] | np.ndarray | pd.Series | None = None,
    ) -> int:
        return sum(1 for _ in self.split(X, y, groups))


# ---------------------------------------------------------------------------
# The repair.
# ---------------------------------------------------------------------------
class ConstrainedLinearRegression(RegressorMixin, BaseEstimator):
    """Least squares subject to ``C b = d``, solved through the KKT system.

    Minimizes ``||Xb - y||^2`` subject to exact linear equalities on the
    coefficients. In a scaling-law setting those equalities are what
    dimensional analysis gives you for free: requiring the fitted law to be
    expressible in dimensionless variables is a set of linear relations among
    the log-space exponents, and imposing it costs nothing to evaluate and adds
    no hyperparameter.

    That is the whole point of the comparison this class exists for. A penalty
    has a strength that must be tuned on a split, and cross-validation cannot
    select for out-of-distribution behaviour. A constraint has nothing to tune,
    so any difference from the unconstrained fit is the assumption doing the
    work rather than a lucky amount of shrinkage.

    ``RegressorMixin`` is listed first deliberately: scikit-learn resolves
    estimator tags along the MRO, and with ``BaseEstimator`` leading, the
    mixin's ``__sklearn_tags__`` never runs and ``is_regressor`` returns False
    for what is plainly a regressor.

    Parameters
    ----------
    constraint, rhs:
        ``C`` and ``d``. ``None`` for both fits ordinary least squares, which
        makes the unconstrained control the same code path as the treatment.
        Columns of ``C`` correspond to coefficients in the order the features
        arrive; when ``fit_intercept`` is true the intercept is *prepended*, so
        ``C`` must carry a leading column for it.
    fit_intercept:
        Whether to prepend an intercept column.
    """

    def __init__(
        self,
        constraint: np.ndarray | None = None,
        rhs: np.ndarray | None = None,
        *,
        fit_intercept: bool = True,
    ) -> None:
        self.constraint = constraint
        self.rhs = rhs
        self.fit_intercept = fit_intercept

    def _design(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        values = np.asarray(
            X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else X, dtype=float
        )
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if self.fit_intercept:
            return np.column_stack([np.ones(len(values)), values])
        return values

    def fit(
        self, X: pd.DataFrame | np.ndarray, y: Sequence[float] | np.ndarray
    ) -> "ConstrainedLinearRegression":
        design = self._design(X)
        target = np.asarray(y, dtype=float)
        if design.shape[0] != target.size:
            raise ValueError(f"X has {design.shape[0]} rows but y has {target.size}.")

        if self.constraint is None:
            with _clean_fp_state():
                coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        else:
            rows = np.atleast_2d(np.asarray(self.constraint, dtype=float))
            values = np.atleast_1d(
                np.zeros(rows.shape[0]) if self.rhs is None else np.asarray(self.rhs, dtype=float)
            )
            if rows.shape[1] != design.shape[1]:
                raise ValueError(
                    f"Constraint matrix has {rows.shape[1]} columns but the design has "
                    f"{design.shape[1]}"
                    + (" (including the intercept column)." if self.fit_intercept else ".")
                )
            if values.size != rows.shape[0]:
                raise ValueError(
                    f"Constraint matrix has {rows.shape[0]} rows but rhs has {values.size}."
                )
            coefficients = _solve_kkt(design, target, rows, values)

        self.coef_ = coefficients[1:] if self.fit_intercept else coefficients
        self.intercept_ = float(coefficients[0]) if self.fit_intercept else 0.0
        self.coefficients_ = coefficients
        self.n_features_in_ = design.shape[1] - (1 if self.fit_intercept else 0)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        with _clean_fp_state():
            return np.asarray(self._design(X) @ self.coefficients_, dtype=float)

    def constraint_violation(self) -> float:
        """``max |Cb - d|`` for the fitted coefficients; 0.0 when unconstrained.

        Worth asserting in a test rather than trusting. A constraint silently
        not applied looks exactly like a constraint that did not help.
        """
        if self.constraint is None:
            return 0.0
        rows = np.atleast_2d(np.asarray(self.constraint, dtype=float))
        values = np.atleast_1d(
            np.zeros(rows.shape[0]) if self.rhs is None else np.asarray(self.rhs, dtype=float)
        )
        return float(np.max(np.abs(rows @ self.coefficients_ - values)))


def _solve_kkt(
    design: np.ndarray, target: np.ndarray, constraint: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    """Stationarity of the Lagrangian, as a saddle-point system.

        [ 2 X^T X   C^T ] [ b      ]   [ 2 X^T y ]
        [ C         0   ] [ lambda ] = [ d       ]

    Duplicated from ``scaling_law.solve_constrained_lstsq`` on purpose: this
    module is meant to be liftable out of the repository on its own, and the
    equivalence of the two is asserted in ``tests/test_scaling_audit.py`` so
    they cannot drift apart unnoticed.
    """
    n_params = design.shape[1]
    n_constraints = constraint.shape[0]
    kkt = np.zeros((n_params + n_constraints, n_params + n_constraints), dtype=float)
    with _clean_fp_state():
        kkt[:n_params, :n_params] = 2.0 * (design.T @ design)
        stacked = np.concatenate([2.0 * (design.T @ target), rhs])
    kkt[:n_params, n_params:] = constraint.T
    kkt[n_params:, :n_params] = constraint
    # Symmetric indefinite, never positive definite, so Cholesky is unavailable.
    solution, *_ = np.linalg.lstsq(kkt, stacked, rcond=None)
    return np.asarray(solution[:n_params], dtype=float)


def summarize(report: pd.DataFrame) -> pd.DataFrame:
    """Collapse an audit to one row per estimator, ranked worst-case first.

    ``worst`` is deliberately alongside ``mean``. The study's whole finding
    about flexibility is that added capacity leaves the typical group roughly
    where it was and ruins the tail, and a table of means hides exactly that.
    For a one-shot prediction on a new group, the tail is the statistic.
    """
    summary = (
        report.groupby("estimator")["score"]
        .agg(mean="mean", median="median", worst="max")
        .join(distance_score_correlation(report))
    )
    return summary.sort_values("worst")
