"""Conformal intervals that survive the shift Result 7 measured. Result 10.

Result 7 is the sharpest negative number in this repository and it stops at the
diagnosis. Split-conformal intervals calibrated to a nominal 90% cover 90% of
rows under grouped cross-validation, 35% on a machine the model has never seen,
and 3% across the ITER-matched size cut. The widths barely move between those
arms: **the intervals do not become vague out of distribution, they stay the
same size and miss**.

That is not a defect in conformal prediction. Split conformal guarantees
coverage under *exchangeability* of the calibration and test scores, and Result 7
breaks exchangeability deliberately, then measures what falls out. Which means
the fix is not "calibrate more carefully". It is to calibrate on a sample that is
exchangeable with the thing actually being predicted.

Two repairs, and a prediction about which one works where
---------------------------------------------------------
``machine_cv``
    Calibrate on **held-out machines** instead of held-out discharges. For each
    machine inside the training set, fit on the others and record the absolute
    log residuals on it; pool those into the conformal quantile. The calibration
    scores are then drawn from "error when predicting a device the model has
    never seen", which is the quantity the test rows are also drawn from. The
    default calibration in Result 7 draws from "error on a new discharge of a
    machine already in the fit", which is a different and far narrower
    distribution.

``machine_cv_distance``
    The same, plus a nonconformity score that is scaled by how far the row sits
    from the training data. Result 4b established that the per-machine error of a
    log-linear law is nearly independent of Mahalanobis distance (rho = -0.06)
    while the trees track it at rho = +0.85. A score of ``|residual| / sigma(d)``
    with ``sigma`` fitted on the calibration residuals therefore produces an
    interval that *widens* with extrapolation distance, which is the property
    Result 7 found missing.

The two repairs make a falsifiable prediction that separates them, and it is
worth stating before the numbers rather than after:

    Leave-one-machine-out should be **fixed** by ``machine_cv`` alone. The
    calibration machines and the test machine are all just machines from this
    database, so once the calibration unit is the machine, exchangeability is
    close to restored and the guarantee close to applicable.

    The ITER-matched size cut should **not** be fixed by ``machine_cv`` alone.
    There the test machines are systematically *larger* than every calibration
    machine, so no amount of recalibrating on training machines makes the two
    exchangeable. Only the distance scaling can help, and only to the extent
    that error really does grow with distance in a way the calibration set can
    see.

If those two land as stated, Result 7's diagnosis is confirmed constructively:
the failure was exchangeability at a particular level, and repairing that level
repairs coverage exactly as far as the level extends and no further.

Cost is reported alongside coverage throughout, for the reason Result 7 gives:
coverage alone is trivial to win, since a wide enough interval covers
everything. A repair that buys coverage purely by inflating the interval has not
repaired anything, so median width travels beside every coverage number here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

import hdb5

# The three calibration schemes compared. ``split`` is Result 7's, carried here
# unchanged so the repairs are measured against it on identical folds rather
# than against the numbers in a different file.
CALIBRATION_METHODS: tuple[str, ...] = ("split", "machine_cv", "machine_cv_distance")

# A machine needs enough rows for its residuals to contribute a meaningful piece
# of the calibration distribution. Matches ``hdb5.MIN_HELD_OUT_ROWS`` so the
# calibration machines are the same population the rest of the repository scores.
MIN_CALIBRATION_MACHINE_ROWS = hdb5.MIN_HELD_OUT_ROWS

# Floor on the fitted distance scale. ``sigma(d)`` multiplies the conformal
# quantile, so a fit that returned zero or a negative scale would collapse the
# interval to a point rather than merely mis-size it.
MIN_DISTANCE_SCALE = 1e-3


def row_mahalanobis(train_features: np.ndarray, query_features: np.ndarray) -> np.ndarray:
    """Per-row Mahalanobis distance from the training distribution.

    ``hdb5._mahalanobis_of_mean`` answers the same question for a whole machine
    at once, which is what Result 4b needs. An interval is a per-row object, so
    this returns one distance per query row.

    The pseudo-inverse is used for the reason given there: the training
    covariance is singular by construction, because ``log a_m`` is exactly
    ``log r_m + log inverse_aspect_ratio``. A plain inverse would amplify a
    direction in which every row sits at the same value anyway.
    """
    covariance = np.atleast_2d(np.cov(train_features, rowvar=False))
    precision = np.linalg.pinv(covariance)
    centred = query_features - train_features.mean(axis=0)
    quadratic = np.einsum("ij,jk,ik->i", centred, precision, centred)
    return np.sqrt(np.maximum(quadratic, 0.0))


def fit_distance_scale(
    distances: np.ndarray, absolute_residuals: np.ndarray
) -> tuple[float, float]:
    """Fit ``sigma(d) = exp(c0 + c1 d)`` to calibration residuals by least squares.

    Exponential rather than linear so the scale cannot go negative however far
    ``d`` is extrapolated, which matters precisely because this is applied at
    distances beyond any in the calibration set. Fitted on ``log|residual|``,
    which makes it an ordinary two-parameter regression rather than an iterative
    fit.

    Residuals that are exactly zero are dropped rather than clipped: a zero
    residual carries no information about scale and ``log 0`` would otherwise
    dominate the fit.
    """
    finite = np.isfinite(distances) & np.isfinite(absolute_residuals) & (absolute_residuals > 0)
    if finite.sum() < 3:
        # Not enough to estimate a slope. Fall back to a constant scale, which
        # reduces this method exactly to the unscaled one rather than failing.
        return 0.0, 0.0
    design = np.column_stack([np.ones(int(finite.sum())), distances[finite]])
    solution, *_ = np.linalg.lstsq(design, np.log(absolute_residuals[finite]), rcond=None)
    return float(solution[0]), float(solution[1])


def distance_scale(distances: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    """Evaluate the fitted scale, floored so it can never collapse an interval."""
    return np.maximum(np.exp(intercept + slope * np.asarray(distances, dtype=float)), MIN_DISTANCE_SCALE)


@dataclass(frozen=True)
class CalibrationSet:
    """Nonconformity scores from held-out machines, and the distances they sit at."""

    absolute_residuals: np.ndarray
    distances: np.ndarray
    n_machines: int

    def scaled_scores(self, intercept: float, slope: float) -> np.ndarray:
        return self.absolute_residuals / distance_scale(self.distances, intercept, slope)


def machine_cv_calibration(
    dataset: pd.DataFrame,
    estimator: Any,
    train_index: np.ndarray,
    *,
    feature_columns: tuple[str, ...] = hdb5.BLIND_FEATURE_COLUMNS,
    min_machine_rows: int = MIN_CALIBRATION_MACHINE_ROWS,
) -> CalibrationSet:
    """Leave-one-machine-out residuals from *inside* the training set.

    This is the whole idea in one function. Nothing here touches the test rows:
    the calibration set is built entirely out of machines the final model will
    also be trained on, by repeatedly holding one of them out. The cost is one
    extra fit per training machine, which is why the estimators this is applied
    to are the linear ones and the two tree models rather than a large sweep.
    """
    columns = list(feature_columns)
    features = dataset[columns]
    log_tau = np.log(dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float))
    labels = dataset[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()

    train_labels = labels[train_index]
    counts = pd.Series(train_labels).value_counts()
    machines = [str(name) for name, count in counts.items() if int(count) >= min_machine_rows]
    if len(machines) < 2:
        raise ValueError(
            f"Only {len(machines)} training machines have {min_machine_rows}+ rows; "
            "machine-level calibration needs at least two."
        )

    residuals: list[np.ndarray] = []
    distances: list[np.ndarray] = []
    for machine in machines:
        held = train_index[train_labels == machine]
        inner = train_index[train_labels != machine]
        model = hdb5.clone_pipeline(estimator)
        with hdb5._suppress_benign_matmul_warnings():
            hdb5.fit_pipeline(model, features.iloc[inner], log_tau[inner])
            predicted = model.predict(features.iloc[held])
        residuals.append(np.abs(log_tau[held] - predicted))
        distances.append(
            row_mahalanobis(
                features.iloc[inner].to_numpy(dtype=float),
                features.iloc[held].to_numpy(dtype=float),
            )
        )
    return CalibrationSet(
        absolute_residuals=np.concatenate(residuals),
        distances=np.concatenate(distances),
        n_machines=len(machines),
    )


def _analytic_predictions(dataset: pd.DataFrame) -> np.ndarray:
    return np.log(dataset["ipb98y2_tau_s"].to_numpy(dtype=float))


def shifted_conformal_arm(
    dataset: pd.DataFrame,
    *,
    train_index: np.ndarray,
    test_index: np.ndarray,
    zoo: dict[str, Any],
    methods: tuple[str, ...] = CALIBRATION_METHODS,
    feature_columns: tuple[str, ...] = hdb5.BLIND_FEATURE_COLUMNS,
    alpha: float = hdb5.DEFAULT_CONFORMAL_ALPHA,
    calibration_fraction: float = hdb5.DEFAULT_CALIBRATION_FRACTION,
    seed: int = hdb5.CONFORMAL_SEED,
    include_ipb98_reference: bool = True,
) -> pd.DataFrame:
    """One train/test pair, every requested scheme, one row per (method, model, test row).

    ``methods`` is a tuple rather than a single value because the two machine-level
    schemes share their entire expensive part. Building the calibration set costs
    one extra fit per training machine, and ``machine_cv`` and
    ``machine_cv_distance`` differ only in how the resulting scores are turned
    into a half-width. Computing them together rather than in two passes halves
    the cost of the whole analysis, and guarantees they are read off identical
    calibration residuals rather than off two runs that could drift.

    ``split`` delegates to ``hdb5._conformal_arm`` rather than reimplementing it,
    so the baseline in every comparison is byte-for-byte the procedure Result 7
    reported.
    """
    unknown = [method for method in methods if method not in CALIBRATION_METHODS]
    if unknown:
        raise ValueError(f"Unknown methods {unknown}; expected from {CALIBRATION_METHODS}.")

    columns = list(feature_columns)
    features = dataset[columns]
    log_tau = np.log(dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float))
    labels = dataset[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()

    # Computed once and attached to every scheme's rows, including ``split``.
    # The split arm does not *use* the distance, but carrying it lets the three
    # schemes be compared on identical footing: "does this scheme's width track
    # distance" is then one question asked the same way of all of them, and the
    # answer for ``split`` is a measured zero rather than a missing value.
    train_features = features.iloc[train_index].to_numpy(dtype=float)
    test_features = features.iloc[test_index].to_numpy(dtype=float)
    test_distances = row_mahalanobis(train_features, test_features)

    frames: list[pd.DataFrame] = []

    if "split" in methods:
        split_rows = hdb5._conformal_arm(
            dataset,
            train_index=train_index,
            test_index=test_index,
            zoo=zoo,
            feature_columns=feature_columns,
            alpha=alpha,
            calibration_fraction=calibration_fraction,
            seed=seed,
            include_ipb98_reference=include_ipb98_reference,
        )
        # That frame stacks one block of test rows per model, so the distances
        # are joined through the ``row`` column rather than assigned positionally.
        # Assigning the array directly happens to work only when the zoo holds
        # exactly one model, which is how a length mismatch here would otherwise
        # survive a small test and fail on the real run.
        distance_by_row = dict(zip(test_index.tolist(), test_distances.tolist(), strict=True))
        frames.append(
            split_rows.assign(
                method="split",
                distance=split_rows["row"].map(distance_by_row),
                distance_scale_slope=0.0,
            )
        )

    machine_methods = tuple(method for method in methods if method != "split")
    if not machine_methods:
        return pd.concat(frames, ignore_index=True)

    def _collect(
        model_name: str, calibration: CalibrationSet, test_log_prediction: np.ndarray, blind: bool
    ) -> None:
        residual = log_tau[test_index] - test_log_prediction
        for method in machine_methods:
            if method == "machine_cv_distance":
                intercept, slope = fit_distance_scale(
                    calibration.distances, calibration.absolute_residuals
                )
                quantile = hdb5.split_conformal_half_width(
                    calibration.scaled_scores(intercept, slope), alpha
                )
                half_width = quantile * distance_scale(test_distances, intercept, slope)
            else:
                slope = 0.0
                half_width = np.full(
                    len(test_index),
                    hdb5.split_conformal_half_width(calibration.absolute_residuals, alpha),
                )
            frames.append(
                pd.DataFrame(
                    {
                        "method": method,
                        "model_name": model_name,
                        "is_blind": blind,
                        "row": test_index,
                        "tokamak": labels[test_index],
                        "covered": np.abs(residual) <= half_width,
                        "half_width_log": half_width,
                        "abs_log_residual": np.abs(residual),
                        "distance": test_distances,
                        # The fitted growth rate of the interval with distance,
                        # recorded rather than recovered downstream. A rank
                        # correlation between width and distance is pinned to
                        # exactly +/-1 within a model, because the half-width is
                        # a monotone function of the distance by construction, so
                        # it would report only this number's sign and none of its
                        # size.
                        "distance_scale_slope": slope,
                    }
                )
            )

    if include_ipb98_reference:
        # The analytic law needs no fitting, so its machine-level calibration is
        # just its residuals on each training machine. Building it through the
        # same loop shape keeps the calibration sample identical in size and
        # composition to the fitted models', which is what makes the coverage
        # numbers comparable across the rows of the output table.
        analytic = _analytic_predictions(dataset)
        train_labels = labels[train_index]
        counts = pd.Series(train_labels).value_counts()
        machines = [
            str(name)
            for name, count in counts.items()
            if int(count) >= MIN_CALIBRATION_MACHINE_ROWS
        ]
        residuals, distances = [], []
        for machine in machines:
            held = train_index[train_labels == machine]
            inner = train_index[train_labels != machine]
            residuals.append(np.abs(log_tau[held] - analytic[held]))
            distances.append(
                row_mahalanobis(
                    features.iloc[inner].to_numpy(dtype=float),
                    features.iloc[held].to_numpy(dtype=float),
                )
            )
        _collect(
            "ipb98y2_analytic",
            CalibrationSet(
                np.concatenate(residuals), np.concatenate(distances), len(machines)
            ),
            analytic[test_index],
            False,
        )

    for name, estimator in zoo.items():
        calibration = machine_cv_calibration(
            dataset, estimator, train_index, feature_columns=feature_columns
        )
        model = hdb5.clone_pipeline(estimator)
        with hdb5._suppress_benign_matmul_warnings():
            hdb5.fit_pipeline(model, features.iloc[train_index], log_tau[train_index])
            predicted = model.predict(features.iloc[test_index])
        _collect(name, calibration, predicted, True)

    return pd.concat(frames, ignore_index=True)


def _summarize_by_method(per_row: pd.DataFrame, *, split: str, alpha: float) -> pd.DataFrame:
    """Coverage per (method, model, scope), reusing hdb5's summariser per method."""
    frames = []
    for method, subset in per_row.groupby("method", sort=False):
        summary = hdb5._summarize_coverage(
            subset.drop(columns=["method"]), split=split, alpha=alpha, by_machine=True
        )
        frames.append(summary.assign(method=method))
    return pd.concat(frames, ignore_index=True)


def coverage_leave_one_tokamak_out(
    dataset: pd.DataFrame,
    zoo: dict[str, Any],
    methods: tuple[str, ...] = CALIBRATION_METHODS,
    *,
    feature_columns: tuple[str, ...] = hdb5.BLIND_FEATURE_COLUMNS,
    alpha: float = hdb5.DEFAULT_CONFORMAL_ALPHA,
    min_rows: int = hdb5.MIN_HELD_OUT_ROWS,
    seed: int = hdb5.CONFORMAL_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Coverage on each unseen machine in turn, under every requested scheme."""
    machines = hdb5.eligible_tokamaks(dataset, min_rows=min_rows)
    labels = dataset[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()
    frames = [
        shifted_conformal_arm(
            dataset,
            train_index=np.flatnonzero(labels != machine),
            test_index=np.flatnonzero(labels == machine),
            zoo=zoo,
            methods=methods,
            feature_columns=feature_columns,
            alpha=alpha,
            seed=seed + index,
        )
        for index, machine in enumerate(machines)
    ]
    per_row = pd.concat(frames, ignore_index=True)
    return per_row, _summarize_by_method(
        per_row, split="leave_one_tokamak_out", alpha=alpha
    )


def coverage_size_split(
    dataset: pd.DataFrame,
    split: hdb5.SizeSplit,
    zoo: dict[str, Any],
    methods: tuple[str, ...] = CALIBRATION_METHODS,
    *,
    feature_columns: tuple[str, ...] = hdb5.BLIND_FEATURE_COLUMNS,
    alpha: float = hdb5.DEFAULT_CONFORMAL_ALPHA,
    seed: int = hdb5.CONFORMAL_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Coverage across a size cut, under every requested scheme.

    This is the arm the two repairs are predicted to behave differently on: the
    calibration machines are all smaller than every test machine, so recovering
    exchangeability by changing the calibration *unit* cannot work here, and only
    the distance scaling has any route to helping.
    """
    labels = dataset[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()
    per_row = shifted_conformal_arm(
        dataset,
        train_index=np.flatnonzero(np.isin(labels, list(split.train_machines))),
        test_index=np.flatnonzero(np.isin(labels, list(split.test_machines))),
        zoo=zoo,
        methods=methods,
        feature_columns=feature_columns,
        alpha=alpha,
        seed=seed,
    )
    return per_row, _summarize_by_method(per_row, split="size_cut", alpha=alpha)
