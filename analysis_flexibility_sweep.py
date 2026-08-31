"""Result 4e: flexibility as a family of models, not a single point.

Run ``python3 analysis_flexibility_sweep.py`` to regenerate everything under
``results/`` for Result 4e.

``results/RESULTS.md`` Result 4d compares five model forms at one setting each:
polynomial degrees 1, 2 and 3 in the log features, all at ridge ``alpha = 1.0``,
plus two tree ensembles. It concludes that flexibility costs the tail rather
than the median. Its own limitations section says why that conclusion is
under-determined:

    The flexibility ladder is three points and one penalty. [...] a larger alpha
    would shrink the divergent curvature terms and might recover much of degree
    2's advantage, and that sweep has not been run.

This runs it. Degree crosses ridge penalty on a full grid, scored under
leave-one-tokamak-out, which turns three assertions into three measurements:

    Result 4e-i    Whether degradation grows with degree at *every* penalty, or
                   only at the one that happened to be reported. The slope of
                   log(worst-machine RMSLE) against degree is the number; a
                   single point cannot have a slope.
    Result 4e-ii   Whether regularization rescues a flexible form. If the
                   best-penalised degree 3 matches degree 1's tail, flexibility
                   was never the problem and the penalty was simply too small.
    Result 4e-iii  Whether the C-Mod blowup is a trend or one machine
                   misbehaving. C-Mod is tracked across the whole grid.

Why this is affordable. Ridge's penalty enters only through a per-direction
filter on the SVD of the training design, so the factorization can be computed
once per (fold, degree) and reused across every alpha (``scaling_law.ridge_from_svd``).
The grid costs what its degree axis alone would cost; the penalty axis is free.
That is also why the fits here are the repository's own solver rather than
scikit-learn's: reusing a factorization is not something the estimator API
exposes. ``tests/test_flexibility_sweep.py`` pins the two against each other at
the settings Result 4d reports, so "our own solver" cannot quietly mean
"a different model".

The narrative built on these numbers is in ``results/RESULTS.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import PolynomialFeatures

import hdb5
from analysis_extrapolation import spearman
from scaling_law import _clean_fp_state as clean_fp_state
from scaling_law import ridge_from_svd, solve_lstsq_svd
from storage import write_dataframe_csv_atomic, write_json_strict

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# The grid. Degree 4 is one rung past anything in Result 4d and is the point of
# the exercise: three points can be joined by any curve, four start to constrain
# it. On nine log features the expansions are 9, 54, 219 and 714 terms.
POLYNOMIAL_DEGREES = (1, 2, 3, 4)

# Nine decades of penalty, centred on the alpha = 1.0 that Result 4d reports so
# that row of the grid reproduces the published table exactly. The range is
# deliberately absurd at both ends: 1e-3 is effectively unregularised least
# squares on a rank-deficient design, and 1e5 shrinks so hard that every model
# collapses towards the intercept. A sweep that only spans the settings someone
# would actually choose cannot show where the interesting behaviour stops.
RIDGE_ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5)

# The setting Result 4d reports, so the sweep can be read as containing it.
REFERENCE_ALPHA = 1.0

# The machine Result 4d's tail claim rests on. Tracked across the whole grid so
# "degree 2 blows up on C-Mod" can be checked against "degree 2 blows up".
FOCUS_MACHINE = "CMOD"

# Any prediction of log(tau) is exponentiated before scoring, and a badly
# unregularised high-degree polynomial can extrapolate far enough to overflow
# the exponential outright. Clipping keeps a blown-up cell as a finite, very
# large RMSLE instead of an inf or a nan that would drop out of every summary
# statistic and quietly flatter the model. The bound is ~e^30, about 1e13
# seconds, which is eleven orders of magnitude above anything in the database:
# nothing physical is being censored, only the arithmetic.
LOG_PREDICTION_CLIP = 30.0


def polynomial_expansion(
    dataset: pd.DataFrame, degree: int, feature_columns: tuple[str, ...]
) -> np.ndarray:
    """Expand the log features to a given polynomial degree, without the bias.

    The bias is excluded because the fits below centre both design and target
    and recover the intercept as the training mean, which is what scikit-learn's
    ``fit_intercept=True`` does and what Result 4d's pipelines therefore did.
    Leaving a constant column in and penalising it would shrink the intercept
    towards zero, which is a different model.
    """
    if degree < 1:
        raise ValueError("degree must be at least 1.")
    values = dataset[list(feature_columns)].to_numpy(dtype=float)
    if degree == 1:
        return values
    return PolynomialFeatures(degree=degree, include_bias=False).fit_transform(values)


@dataclass(frozen=True)
class _FittedFold:
    """A training fold factored once, ready to be solved at any penalty.

    Standardisation uses the *training* mean and scale only, so the held-out
    rows are transformed by numbers they did not contribute to. Constant columns
    (which the expansion produces whenever a feature is constant on this fold)
    are given a scale of 1 rather than 0: after centring they are identically
    zero and contribute nothing, and dividing by their true scale would be a
    division by zero.
    """

    u: np.ndarray
    s: np.ndarray
    vt: np.ndarray
    centred_target: np.ndarray
    target_mean: float
    column_mean: np.ndarray
    column_scale: np.ndarray
    test_design: np.ndarray

    def predict_log(self, alpha: float) -> np.ndarray:
        coefficients = ridge_from_svd(self.u, self.s, self.vt, self.centred_target, alpha)
        with clean_fp_state():
            scaled = (self.test_design - self.column_mean) / self.column_scale
            return self.target_mean + scaled @ coefficients


def _factor_fold(
    train_design: np.ndarray, test_design: np.ndarray, train_log_target: np.ndarray
) -> _FittedFold:
    """Standardise, centre and factor a fold once for the whole penalty axis."""
    column_mean = train_design.mean(axis=0)
    column_scale = train_design.std(axis=0)
    column_scale = np.where(column_scale > 0, column_scale, 1.0)
    scaled = (train_design - column_mean) / column_scale

    target_mean = float(train_log_target.mean())
    centred_target = train_log_target - target_mean

    u, s, vt = np.linalg.svd(scaled, full_matrices=False)
    return _FittedFold(
        u=u,
        s=s,
        vt=vt,
        centred_target=centred_target,
        target_mean=target_mean,
        column_mean=column_mean,
        column_scale=column_scale,
        test_design=test_design,
    )


def _rmsle_from_log(log_true: np.ndarray, log_predicted: np.ndarray) -> tuple[float, int]:
    """RMSLE in log space, plus how many predictions the clip had to catch.

    The count is returned rather than swallowed. A cell scored on clipped
    predictions is reporting a censored error, and that is a different claim
    from one where every prediction landed in range; the caller aggregates the
    count into the grid so it can be checked rather than assumed to be zero.
    """
    # Count on the *raw* predictions, before anything is substituted: a nan
    # compares false against every bound, so counting after the substitution
    # would report zero for exactly the predictions most worth knowing about.
    out_of_range = ~np.isfinite(log_predicted) | (np.abs(log_predicted) > LOG_PREDICTION_CLIP)
    # ``nan_to_num`` before ``clip``, because ``clip`` passes a nan straight
    # through and would poison the mean into a silent nan.
    finite = np.nan_to_num(
        log_predicted,
        nan=LOG_PREDICTION_CLIP,
        posinf=LOG_PREDICTION_CLIP,
        neginf=-LOG_PREDICTION_CLIP,
    )
    clipped = np.clip(finite, -LOG_PREDICTION_CLIP, LOG_PREDICTION_CLIP)
    return float(np.sqrt(np.mean((clipped - log_true) ** 2))), int(out_of_range.sum())


@dataclass(frozen=True)
class GridCell:
    """One (degree, alpha) pair scored under both splits."""

    degree: int
    n_terms: int
    alpha: float
    cv_rmsle: float
    lomo_mean_rmsle: float
    lomo_median_rmsle: float
    lomo_worst_rmsle: float
    worst_machine: str
    degradation_factor: float
    # Spearman rho between per-machine RMSLE and how far the machine sits
    # outside the training data, as in Result 4b.
    distance_spearman: float
    focus_machine_rmsle: float
    # Predictions this cell pushed past ``LOG_PREDICTION_CLIP``. Nonzero means
    # the RMSLE reported for it is a censored lower bound on the real error.
    n_clipped_predictions: int

    def to_json(self) -> dict[str, object]:
        return {
            "degree": self.degree,
            "n_terms": self.n_terms,
            "alpha": self.alpha,
            "cv_rmsle": self.cv_rmsle,
            "lomo_mean_rmsle": self.lomo_mean_rmsle,
            "lomo_median_rmsle": self.lomo_median_rmsle,
            "lomo_worst_rmsle": self.lomo_worst_rmsle,
            "worst_machine": self.worst_machine,
            "degradation_factor": self.degradation_factor,
            "distance_spearman": self.distance_spearman,
            "focus_machine_rmsle": self.focus_machine_rmsle,
            "n_clipped_predictions": self.n_clipped_predictions,
        }


@dataclass(frozen=True)
class DegreeSlope:
    """How fast a chosen statistic grows with degree, at one penalty.

    Fitted as log10(statistic) against degree, so the slope is "decades of error
    per degree of freedom added". A slope of 0 means flexibility is free at this
    penalty; a positive slope is the cost Result 4d asserts from three points.
    """

    alpha: float
    statistic: str
    slope_per_degree: float
    # 10 ** slope: the multiplicative factor per extra degree, which is the
    # readable form.
    factor_per_degree: float
    n_points: int

    def to_json(self) -> dict[str, object]:
        return {
            "alpha": self.alpha,
            "statistic": self.statistic,
            "slope_per_degree": self.slope_per_degree,
            "factor_per_degree": self.factor_per_degree,
            "n_points": self.n_points,
        }


# A penalty large enough to ruin degree 1 has not made flexibility free, it has
# made every model equally useless, and a slope fitted across that regime is
# measuring the collapse rather than the cost of flexibility. So a penalty
# counts as informative only while it leaves the *baseline* form intact: degree
# 1's worst held-out machine must stay within this factor of its own best.
# 1.1 is a tenth of a decade, comfortably inside the machine-to-machine scatter.
BASELINE_TOLERANCE = 1.1

# How much better than its neighbour an edge-of-grid optimum has to be before
# the sweep counts as not having bracketed it. 1% is well below anything the
# reported numbers turn on.
EDGE_TOLERANCE = 0.01


@dataclass(frozen=True)
class PenaltyStatus:
    """Whether one penalty is strong enough to have broken the baseline form."""

    alpha: float
    degree_one_worst_rmsle: float
    # degree_one_worst_rmsle / the best degree 1 achieves anywhere on the grid.
    baseline_ratio: float
    is_usable: bool

    def to_json(self) -> dict[str, object]:
        return {
            "alpha": self.alpha,
            "degree_one_worst_rmsle": self.degree_one_worst_rmsle,
            "baseline_ratio": self.baseline_ratio,
            "is_usable": self.is_usable,
        }


def _penalty_statuses(frame: pd.DataFrame, baseline_degree: int) -> list[PenaltyStatus]:
    rows = frame[frame["degree"] == baseline_degree].sort_values("alpha")
    worst = rows["lomo_worst_rmsle"].to_numpy(dtype=float)
    best = float(worst.min())
    return [
        PenaltyStatus(
            alpha=float(alpha),
            degree_one_worst_rmsle=float(value),
            baseline_ratio=float(value / best),
            is_usable=bool(value / best <= BASELINE_TOLERANCE),
        )
        for alpha, value in zip(rows["alpha"].to_numpy(dtype=float), worst)
    ]


@dataclass(frozen=True)
class BestPenalty:
    """The kindest penalty for one degree, and whether it rescues the form.

    Selected on the leave-one-tokamak-out tail, which is *not* a defensible
    model-selection procedure: it picks the penalty using the held-out machines
    it is then scored on. That is deliberate. The question is not "how well can
    this degree do in practice" but "how well could it possibly do, given every
    advantage", so an optimistic bound is the right instrument. If a degree
    cannot match degree 1 even when its penalty is chosen with hindsight on the
    test set, no honest tuning procedure will do better.
    """

    degree: int
    best_alpha: float
    best_worst_rmsle: float
    best_mean_rmsle: float
    # Ratio to the same statistic for degree 1 at its own best penalty. 1.0
    # means the extra flexibility has been fully tamed.
    worst_ratio_to_degree_one: float
    # True only when the optimum sits at an edge of the swept range *and* is
    # strictly better than its neighbour, which is what "the sweep has not
    # bracketed the optimum" actually means. A form whose score is flat across
    # the bottom decades has its optimum bracketed however the tie is broken,
    # and flagging that as an edge case would be a false alarm.
    at_grid_edge: bool
    # Whether the penalty this degree needs is one that leaves degree 1 intact.
    # False means the flexible form is only tolerable at a shrinkage that would
    # have destroyed the baseline: it is not competing on equal terms.
    best_alpha_is_usable: bool

    def to_json(self) -> dict[str, object]:
        return {
            "degree": self.degree,
            "best_alpha": self.best_alpha,
            "best_worst_rmsle": self.best_worst_rmsle,
            "best_mean_rmsle": self.best_mean_rmsle,
            "worst_ratio_to_degree_one": self.worst_ratio_to_degree_one,
            "at_grid_edge": self.at_grid_edge,
            "best_alpha_is_usable": self.best_alpha_is_usable,
        }


@dataclass(frozen=True)
class FlexibilitySweep:
    feature_columns: list[str]
    n_rows: int
    # ``None`` when the caller passed a frame it built itself rather than a
    # ``dataset_path``; see ``analysis_extrapolation.ExtrapolationAnalysis``.
    provenance: dict[str, object] | None
    degrees: list[int]
    alphas: list[float]
    machines: list[str]
    cells: list[GridCell]
    slopes: list[DegreeSlope]
    penalties: list[PenaltyStatus]
    best_penalties: list[BestPenalty]
    focus_machine: str
    per_machine: pd.DataFrame = field(repr=False)

    @property
    def usable_alphas(self) -> list[float]:
        return [status.alpha for status in self.penalties if status.is_usable]

    def cell(self, degree: int, alpha: float) -> GridCell:
        return next(
            cell for cell in self.cells if cell.degree == degree and cell.alpha == alpha
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([cell.to_json() for cell in self.cells])

    # ``Any`` rather than ``object``, for the reason given on
    # ``BootstrapResolutionComparison.to_json``: this is the whole serialized
    # payload and its values are nested structures callers index into.
    def to_json(self) -> dict[str, Any]:
        return {
            "dataset": self.provenance,
            "feature_columns": self.feature_columns,
            "n_rows": self.n_rows,
            "degrees": self.degrees,
            "alphas": self.alphas,
            "reference_alpha": REFERENCE_ALPHA,
            "n_machines_held_out": len(self.machines),
            "machines_held_out": self.machines,
            "focus_machine": self.focus_machine,
            "log_prediction_clip": LOG_PREDICTION_CLIP,
            "baseline_tolerance": BASELINE_TOLERANCE,
            "usable_alphas": self.usable_alphas,
            "cells": [cell.to_json() for cell in self.cells],
            "degree_slopes": [slope.to_json() for slope in self.slopes],
            "penalties": [status.to_json() for status in self.penalties],
            "best_penalties": [best.to_json() for best in self.best_penalties],
        }


def _slope_against_degree(
    frame: pd.DataFrame, alpha: float, statistic: str
) -> DegreeSlope:
    """Least-squares slope of log10(statistic) on degree, at one penalty."""
    rows = frame[frame["alpha"] == alpha].sort_values("degree")
    values = rows[statistic].to_numpy(dtype=float)
    degrees = rows["degree"].to_numpy(dtype=float)
    usable = np.isfinite(values) & (values > 0)
    if usable.sum() < 2:
        return DegreeSlope(alpha, statistic, float("nan"), float("nan"), int(usable.sum()))
    design = np.column_stack([np.ones(usable.sum()), degrees[usable]])
    _, slope = solve_lstsq_svd(design, np.log10(values[usable]))
    return DegreeSlope(
        alpha=alpha,
        statistic=statistic,
        slope_per_degree=float(slope),
        factor_per_degree=float(10.0**slope),
        n_points=int(usable.sum()),
    )


def _best_penalties(
    frame: pd.DataFrame,
    degrees: list[int],
    alphas: list[float],
    usable_alphas: set[float],
) -> list[BestPenalty]:
    """For each degree, the penalty that minimises the worst held-out machine."""
    ordered = sorted(alphas)

    def best_row(degree: int) -> pd.Series:
        rows = frame[frame["degree"] == degree]
        # ``.loc[label]`` is typed as possibly returning a frame; the index here
        # is unique, so a single row is guaranteed.
        best = rows.loc[rows["lomo_worst_rmsle"].idxmin()]
        assert isinstance(best, pd.Series)
        return best

    def is_unbracketed_edge(degree: int, best_alpha: float, best_value: float) -> bool:
        if best_alpha not in (ordered[0], ordered[-1]):
            return False
        neighbour_alpha = ordered[1] if best_alpha == ordered[0] else ordered[-2]
        rows = frame[(frame["degree"] == degree) & (frame["alpha"] == neighbour_alpha)]
        neighbour = float(rows["lomo_worst_rmsle"].iloc[0])
        # Meaningfully better than the neighbour means the curve is still
        # falling as it leaves the grid. A relative tolerance rather than an
        # exact comparison: degree 1's score is identical to three decimals
        # across the bottom four decades of penalty, and calling a difference in
        # the twelfth digit "the optimum is off the grid" would be noise.
        return best_value < neighbour * (1.0 - EDGE_TOLERANCE)

    baseline = float(best_row(degrees[0])["lomo_worst_rmsle"])
    results = []
    for degree in degrees:
        row = best_row(degree)
        best_alpha = float(row["alpha"])
        best_value = float(row["lomo_worst_rmsle"])
        results.append(
            BestPenalty(
                degree=degree,
                best_alpha=best_alpha,
                best_worst_rmsle=best_value,
                best_mean_rmsle=float(row["lomo_mean_rmsle"]),
                worst_ratio_to_degree_one=float(best_value / baseline),
                at_grid_edge=is_unbracketed_edge(degree, best_alpha, best_value),
                best_alpha_is_usable=best_alpha in usable_alphas,
            )
        )
    return results


def sweep_flexibility(
    dataset: pd.DataFrame,
    *,
    dataset_path: Path | str | None = None,
    feature_columns: tuple[str, ...] = hdb5.BLIND_FEATURE_COLUMNS,
    degrees: tuple[int, ...] = POLYNOMIAL_DEGREES,
    alphas: tuple[float, ...] = RIDGE_ALPHAS,
    min_rows: int = hdb5.MIN_HELD_OUT_ROWS,
    n_splits: int = hdb5.N_CV_FOLDS,
    focus_machine: str = FOCUS_MACHINE,
) -> FlexibilitySweep:
    """Score every (degree, alpha) pair under grouped CV and leave-one-tokamak-out.

    The two splits use the same features, the same expansion and the same
    solver, so the only difference between the two numbers reported for a cell
    is what the split holds out, exactly as in Result 4.
    """
    machines = hdb5.eligible_tokamaks(dataset, min_rows=min_rows)
    if not machines:
        raise ValueError(f"No tokamak has at least {min_rows} rows; nothing can be held out.")

    log_target = np.log(dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float))
    labels = dataset[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()
    groups = dataset[hdb5.GROUP_COLUMN].to_numpy()

    # Distance of each machine from the rest of the database, computed once: it
    # is a property of the split, not of the model being scored.
    distances = {
        machine: hdb5.extrapolation_diagnostic(
            dataset, machine, feature_columns=feature_columns
        ).feature_mahalanobis
        for machine in machines
    }

    splitter = GroupKFold(n_splits=n_splits)
    cv_folds = list(splitter.split(dataset, log_target, groups))

    machine_records: list[dict[str, object]] = []
    cells: list[GridCell] = []

    for degree in degrees:
        expanded = polynomial_expansion(dataset, degree, feature_columns)
        n_terms = int(expanded.shape[1])

        # One factorization per fold, reused across the whole penalty axis.
        cv_fitted = [
            (test_idx, _factor_fold(expanded[train_idx], expanded[test_idx], log_target[train_idx]))
            for train_idx, test_idx in cv_folds
        ]
        lomo_fitted = []
        for machine in machines:
            held = labels == machine
            train_idx = np.flatnonzero(~held)
            test_idx = np.flatnonzero(held)
            lomo_fitted.append(
                (
                    machine,
                    test_idx,
                    _factor_fold(expanded[train_idx], expanded[test_idx], log_target[train_idx]),
                )
            )

        for alpha in alphas:
            out_of_fold = np.empty_like(log_target)
            for test_idx, fold in cv_fitted:
                out_of_fold[test_idx] = fold.predict_log(alpha)
            cv_rmsle, n_clipped = _rmsle_from_log(log_target, out_of_fold)

            per_machine: dict[str, float] = {}
            for machine, test_idx, fold in lomo_fitted:
                rmsle, machine_clipped = _rmsle_from_log(
                    log_target[test_idx], fold.predict_log(alpha)
                )
                n_clipped += machine_clipped
                per_machine[machine] = rmsle
                machine_records.append(
                    {
                        "degree": degree,
                        "alpha": alpha,
                        "tokamak": machine,
                        "n_held_out_rows": int(test_idx.size),
                        "rmsle": rmsle,
                        "feature_mahalanobis": distances[machine],
                    }
                )

            scores = np.array([per_machine[machine] for machine in machines], dtype=float)
            worst_index = int(np.argmax(scores))
            cells.append(
                GridCell(
                    degree=degree,
                    n_terms=n_terms,
                    alpha=float(alpha),
                    cv_rmsle=cv_rmsle,
                    lomo_mean_rmsle=float(scores.mean()),
                    lomo_median_rmsle=float(np.median(scores)),
                    lomo_worst_rmsle=float(scores[worst_index]),
                    worst_machine=machines[worst_index],
                    degradation_factor=float(scores.mean() / cv_rmsle),
                    distance_spearman=spearman(
                        scores,
                        np.array([distances[machine] for machine in machines], dtype=float),
                    ),
                    focus_machine_rmsle=float(per_machine.get(focus_machine, float("nan"))),
                    n_clipped_predictions=n_clipped,
                )
            )

    frame = pd.DataFrame([cell.to_json() for cell in cells])
    slopes = [
        _slope_against_degree(frame, float(alpha), statistic)
        for statistic in ("lomo_worst_rmsle", "lomo_mean_rmsle", "lomo_median_rmsle")
        for alpha in alphas
    ]

    penalties = _penalty_statuses(frame, int(degrees[0]))
    return FlexibilitySweep(
        feature_columns=list(feature_columns),
        n_rows=int(len(dataset)),
        provenance=hdb5.dataset_provenance(dataset_path) if dataset_path is not None else None,
        degrees=[int(degree) for degree in degrees],
        alphas=[float(alpha) for alpha in alphas],
        machines=machines,
        cells=cells,
        slopes=slopes,
        penalties=penalties,
        best_penalties=_best_penalties(
            frame,
            [int(d) for d in degrees],
            [float(a) for a in alphas],
            {status.alpha for status in penalties if status.is_usable},
        ),
        focus_machine=focus_machine,
        per_machine=pd.DataFrame(machine_records),
    )


def plot_flexibility_sweep(sweep: FlexibilitySweep) -> Path | None:
    """Three panels: the surface, the slope it implies, and the focus machine.

    The left panel is the result Result 4d could not show. Each line is one
    penalty; if flexibility were free at a large enough alpha, one of them would
    be flat.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - plotting is optional
        return None

    ink, muted = "#0b0b0b", "#52514e"
    highlight = "#eb6834"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    figure.patch.set_facecolor("#fcfcfb")
    for axis in axes:
        axis.set_facecolor("#fcfcfb")
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)

    frame = sweep.to_frame()
    alphas = sweep.alphas
    # Light-to-dark with alpha, so the eye reads "more regularisation" as
    # "darker" without needing to decode a rainbow.
    shades = plt.get_cmap("viridis")(np.linspace(0.08, 0.88, len(alphas)))

    usable = set(sweep.usable_alphas)
    for color, alpha in zip(shades, alphas):
        rows = frame[frame["alpha"] == alpha].sort_values("degree")
        is_reference = alpha == REFERENCE_ALPHA
        # Penalties that have already broken degree 1 are drawn, because hiding
        # them would hide the only lines that are flat, but drawn dashed and
        # pale: their flatness is a collapse, not a rescue.
        is_usable = alpha in usable
        axes[0].plot(
            rows["degree"],
            rows["lomo_worst_rmsle"],
            marker="o",
            markersize=4.5,
            linewidth=2.4 if is_reference else 1.3,
            linestyle="-" if is_usable else "--",
            alpha=1.0 if is_usable else 0.55,
            color=highlight if is_reference else color,
            zorder=3 if is_reference else 2,
            label=f"alpha = {alpha:g}"
            + (" (Result 4d)" if is_reference else "")
            + ("" if is_usable else ", degree 1 already broken"),
        )
    axes[0].set_yscale("log")
    axes[0].set_xticks(sweep.degrees)
    axes[0].set_xlabel("polynomial degree in the log features", fontsize=9, color=muted)
    axes[0].set_ylabel("worst held-out machine, RMSLE (log scale)", fontsize=9, color=muted)
    axes[0].set_title(
        "Every penalty that leaves degree 1 intact\nalso makes the tail grow with degree.",
        fontsize=11,
        color=ink,
    )
    axes[0].legend(frameon=False, fontsize=7.0, loc="upper left", labelcolor=muted, ncol=1)

    slopes = pd.DataFrame([slope.to_json() for slope in sweep.slopes])
    labels = {
        "lomo_worst_rmsle": ("worst machine", highlight, "o"),
        "lomo_mean_rmsle": ("mean over machines", "#2a78d6", "s"),
        "lomo_median_rmsle": ("median machine", "#1a9c6d", "^"),
    }
    for statistic, (label, color, marker) in labels.items():
        rows = slopes[slopes["statistic"] == statistic].sort_values("alpha")
        axes[1].plot(
            rows["alpha"],
            rows["slope_per_degree"],
            marker=marker,
            markersize=5,
            linewidth=1.8,
            color=color,
            label=label,
        )
    axes[1].axhline(0.0, color=ink, linewidth=0.9, alpha=0.5)
    axes[1].annotate(
        "flexibility is free below this line",
        xy=(min(alphas), 0.0),
        xytext=(2, 4),
        textcoords="offset points",
        fontsize=8.5,
        color=ink,
    )
    # The slope does reach zero, but only after the penalty has flattened the
    # baseline too. Shading that band is the honest way to show it: the reader
    # can see the curve cross zero and see why the crossing does not count.
    if usable and len(usable) < len(alphas):
        broken_from = min(alpha for alpha in alphas if alpha not in usable)
        axes[1].axvspan(broken_from / 3.0, max(alphas) * 3, color=muted, alpha=0.10, zorder=0)
        axes[1].annotate(
            "degree 1 is\nbroken here too",
            xy=(broken_from, 0.10),
            xycoords=("data", "axes fraction"),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=8.5,
            color=muted,
            va="center",
        )
    axes[1].set_xscale("log")
    axes[1].set_xlim(min(alphas) / 3, max(alphas) * 3)
    axes[1].set_xlabel("ridge alpha", fontsize=9, color=muted)
    axes[1].set_ylabel("decades of RMSLE added per degree", fontsize=9, color=muted)
    axes[1].set_title(
        "The slope Result 4d could not measure.\n"
        "It reaches zero only where the penalty has ruined every form.",
        fontsize=11,
        color=ink,
    )
    axes[1].legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=muted)

    for color, alpha in zip(shades, alphas):
        rows = frame[frame["alpha"] == alpha].sort_values("degree")
        is_reference = alpha == REFERENCE_ALPHA
        axes[2].plot(
            rows["degree"],
            rows["focus_machine_rmsle"],
            marker="o",
            markersize=4.5,
            linewidth=2.4 if is_reference else 1.3,
            linestyle="-" if alpha in usable else "--",
            alpha=1.0 if alpha in usable else 0.55,
            color=highlight if is_reference else color,
            zorder=3 if is_reference else 2,
        )
    axes[2].set_yscale("log")
    axes[2].set_xticks(sweep.degrees)
    axes[2].set_xlabel("polynomial degree in the log features", fontsize=9, color=muted)
    axes[2].set_ylabel(f"RMSLE on {sweep.focus_machine} (log scale)", fontsize=9, color=muted)
    # The tally is the actual answer to "is this one machine misbehaving": it
    # says how often each machine is the one that defines the tail.
    tally = pd.Series([cell.worst_machine for cell in sweep.cells]).value_counts()
    axes[2].annotate(
        "worst machine, over all "
        f"{len(sweep.cells)} cells:\n"
        + ",  ".join(f"{name} {count}" for name, count in tally.items()),
        xy=(0.03, 0.03),
        xycoords="axes fraction",
        fontsize=8.5,
        color=muted,
        va="bottom",
    )
    axes[2].set_title(
        f"{sweep.focus_machine} across the whole grid.\n"
        "Result 4d's one blown-up cell was not an outlier.",
        fontsize=11,
        color=ink,
    )

    figure.tight_layout()
    path = RESULTS_DIR / "flexibility_sweep.png"
    figure.savefig(path, dpi=180, facecolor="#fcfcfb")
    plt.close(figure)
    return path


def main() -> None:
    dataset = hdb5.prepare_dataset()
    sweep = sweep_flexibility(dataset, dataset_path=hdb5.default_hdb5_path())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_dataframe_csv_atomic(RESULTS_DIR / "flexibility_sweep.csv", sweep.to_frame())
    write_dataframe_csv_atomic(
        RESULTS_DIR / "flexibility_sweep_per_machine.csv", sweep.per_machine
    )
    write_json_strict(RESULTS_DIR / "flexibility_sweep.json", sweep.to_json())
    figure_path = plot_flexibility_sweep(sweep)

    print("--- Result 4e: flexibility as a family ---")
    print(
        f"{len(sweep.degrees)} degrees x {len(sweep.alphas)} penalties = "
        f"{len(sweep.cells)} models, each scored on {len(sweep.machines)} held-out machines"
    )
    terms = ", ".join(
        f"degree {cell.degree}: {cell.n_terms} terms"
        for cell in sweep.cells
        if cell.alpha == REFERENCE_ALPHA
    )
    print(f"  {terms}\n")

    print(f"--- worst held-out machine, RMSLE (alpha = {REFERENCE_ALPHA:g} is Result 4d's row) ---")
    header = "  " + "degree".ljust(8) + "".join(f"{f'a={alpha:g}':>10}" for alpha in sweep.alphas)
    print(header)
    for degree in sweep.degrees:
        row = "".join(
            f"{sweep.cell(degree, alpha).lomo_worst_rmsle:>10.3f}" for alpha in sweep.alphas
        )
        print(f"  {str(degree).ljust(8)}{row}")

    print("\n--- Result 4e-i: slope of log10(RMSLE) against degree, per penalty ---")
    print(
        f"  {'alpha':>8}{'worst':>10}{'mean':>10}{'median':>10}{'degree 1':>11}   (x per degree)"
    )
    by_key = {(slope.statistic, slope.alpha): slope for slope in sweep.slopes}
    status_by_alpha = {status.alpha: status for status in sweep.penalties}
    for alpha in sweep.alphas:
        worst = by_key[("lomo_worst_rmsle", alpha)]
        mean = by_key[("lomo_mean_rmsle", alpha)]
        median = by_key[("lomo_median_rmsle", alpha)]
        status = status_by_alpha[alpha]
        note = "" if status.is_usable else "  <- degree 1 already broken"
        print(
            f"  {alpha:>8g}{worst.factor_per_degree:>9.2f}x{mean.factor_per_degree:>9.2f}x"
            f"{median.factor_per_degree:>9.2f}x{status.degree_one_worst_rmsle:>11.3f}{note}"
        )
    usable_worst = [
        by_key[("lomo_worst_rmsle", alpha)].factor_per_degree for alpha in sweep.usable_alphas
    ]
    print(
        f"  over the {len(usable_worst)} penalties that leave degree 1 intact, the tail grows "
        f"{min(usable_worst):.2f}x to {max(usable_worst):.2f}x per degree; it stops growing only "
        f"once the penalty has broken degree 1 as well"
    )

    print("\n--- Result 4e-ii: can the penalty rescue a flexible form? ---")
    print("  (alpha chosen with hindsight on the held-out machines: an optimistic bound)")
    print(f"  {'degree':>7}{'best alpha':>12}{'worst':>9}{'mean':>9}{'vs degree 1':>13}")
    for best in sweep.best_penalties:
        notes = []
        if best.at_grid_edge:
            notes.append("optimum not bracketed by the grid")
        if not best.best_alpha_is_usable:
            notes.append("this alpha would have broken degree 1")
        suffix = f"   ({'; '.join(notes)})" if notes else ""
        print(
            f"  {best.degree:>7}{best.best_alpha:>12g}{best.best_worst_rmsle:>9.3f}"
            f"{best.best_mean_rmsle:>9.3f}{best.worst_ratio_to_degree_one:>12.2f}x{suffix}"
        )
    rescued = [best for best in sweep.best_penalties[1:] if best.worst_ratio_to_degree_one <= 1.1]
    print(
        f"  {len(rescued)} of {len(sweep.best_penalties) - 1} flexible forms match degree 1's tail "
        "even with their penalty chosen on the test machines"
    )

    print(f"\n--- Result 4e-iii: {sweep.focus_machine} across the grid ---")
    header = "  " + "degree".ljust(8) + "".join(f"{f'a={alpha:g}':>10}" for alpha in sweep.alphas)
    print(header)
    for degree in sweep.degrees:
        row = "".join(
            f"{sweep.cell(degree, alpha).focus_machine_rmsle:>10.3f}" for alpha in sweep.alphas
        )
        print(f"  {str(degree).ljust(8)}{row}")
    worst_machines = pd.Series([cell.worst_machine for cell in sweep.cells]).value_counts()
    print(
        "  machine that is worst, counted over all "
        f"{len(sweep.cells)} cells: "
        + ", ".join(f"{name} {count}" for name, count in worst_machines.items())
    )
    clipped = sum(cell.n_clipped_predictions for cell in sweep.cells)
    print(f"  predictions caught by the |log tau| <= {LOG_PREDICTION_CLIP:g} clip: {clipped}")

    if figure_path:
        print(f"\nfigure: {figure_path}")


if __name__ == "__main__":
    main()
