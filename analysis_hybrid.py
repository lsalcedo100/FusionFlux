"""Can a model be flexible in range and still extrapolate? Result 6.

Run ``python3 analysis_hybrid.py`` to regenerate everything under ``results/``
for Result 6: the shrinkage sweep, the CV-versus-LOMO frontier, the honest
model-selection outcome, the paired bootstrap against plain ridge, and the
figure.

Results 4 and 5 diagnose a failure and stop. The trees win by 41% under grouped
cross-validation and lose to a log-linear power law on all 13 held-out machines,
and at the ITER-size-matched cut they land closer to a constant predictor than
to the power law. Result 4d attributes that to functional form rather than to
flexibility as such: the power law is the only form on the ladder whose error
stays bounded away from the data.

That diagnosis implies a cure, and the cure is worth building rather than just
describing. Fit the power law, learn a correction on its *log residuals*, and
damp the correction hard. The base term keeps the power law's unbounded,
log-linear behaviour in size, so the model should extrapolate like ridge; the
correction is free to pick up in-range structure the power law misses. Sweeping
the damping factor from 0 to 1 sweeps the model continuously from plain ridge to
an undamped correction, and scoring every rung under both splits traces out
whatever trade-off actually exists.

    Result 6a  The frontier. Whether any rung beats plain ridge under grouped CV
               while keeping its leave-one-machine-out score.
    Result 6b  What a practitioner tuning on CV actually gets, since the damping
               factor has to be chosen without the held-out machines.
    Result 6c  The same rungs at the ITER-size-matched cut of Result 5.

Both outcomes are reportable. A rung that improves CV at no LOMO cost is a point
neither pure model reaches. A sweep where every CV gain is paid for one-for-one
out of distribution is the sharper finding: it would mean the structure the
power law leaves behind is machine-specific and does not transfer at all.

The narrative built on these numbers is in ``results/RESULTS.md``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import hdb5
from analysis_extrapolation import (
    N_BOOTSTRAP_RESAMPLES,
    PairedDifference,
    bootstrap_paired_difference,
    spearman,
)
from storage import write_dataframe_csv_atomic, write_json_strict

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# The two correction families, in the order they appear in the narrative.
CORRECTIONS: tuple[str, ...] = ("ridge", "gbm")

CORRECTION_LABELS = {
    "ridge": "degree-2 log polynomial, ridge alpha 1000",
    "gbm": "depth-2 boosted trees, 200 rounds, l2 10",
}

# The model every rung is measured against. Shrinkage 0 *is* this model, so the
# frontier starts on it exactly rather than near it.
BASELINE_MODEL = "ridge_loglinear"

# Reference points plotted alongside the frontier. These are the models Result 4
# already scored; they are here to show where the swept family sits relative to
# them, not to be re-litigated.
REFERENCE_MODELS = (
    "ipb98y2_analytic",
    "ridge_loglinear",
    "ridge_log_quadratic",
    "hist_gradient_boosting",
    "random_forest",
)

REFERENCE_LABELS = {
    "ipb98y2_analytic": "IPB98(y,2)",
    "ridge_loglinear": "ridge (log-linear)",
    "ridge_log_quadratic": "ridge (log-quadratic)",
    "hist_gradient_boosting": "hist GB",
    "random_forest": "random forest",
}

# A rung counts as keeping ridge's out-of-distribution behaviour if the paired
# bootstrap over machines cannot distinguish its LOMO error from ridge's. That
# is a deliberately permissive test with 13 machines, and it is the right
# direction to be permissive in: it makes it *easier* for a hybrid to qualify,
# so failing it is a strong statement and passing it is a weak one. The mean
# gap is reported alongside so a reader can apply a stricter bar.
PARETO_ALPHA = 0.05


@dataclass(frozen=True)
class FrontierPoint:
    """One model scored under all three splits.

    ``shrinkage`` is NaN for the reference models, which are not rungs of the
    sweep and have no damping factor.
    """

    model_name: str
    correction: str
    shrinkage: float
    is_hybrid: bool
    is_blind: bool
    cv_rmsle: float
    lomo_mean_rmsle: float
    lomo_median_rmsle: float
    lomo_worst_rmsle: float
    size_cut_rmsle: float
    # Spearman rank correlation of per-machine LOMO error with Mahalanobis
    # distance from the training data. Result 4b's statistic: near zero means
    # the error does not grow with distance, which is the property the base
    # power law has and the trees do not.
    distance_spearman: float

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SelectionOutcome:
    """What choosing the damping factor honestly gets you, per correction family.

    The damping factor is a hyperparameter, and the only split a practitioner
    has to tune it on is the one they can compute: grouped CV by discharge. So
    the reported hybrid is the rung CV selects, and its LOMO score is whatever
    that rung happens to get. Selecting on LOMO would be selecting on the test
    set, so the best-on-LOMO rung is reported separately and labelled as the
    oracle it is: not achievable, present only to size what CV selection costs.
    """

    correction: str
    baseline_cv_rmsle: float
    baseline_lomo_rmsle: float

    cv_selected_shrinkage: float
    cv_selected_cv_rmsle: float
    cv_selected_lomo_rmsle: float
    cv_selected_size_cut_rmsle: float

    oracle_shrinkage: float
    oracle_lomo_rmsle: float

    # The best rung, by CV, that also survives the paired-bootstrap test against
    # plain ridge on LOMO. None when no rung does, which is itself the result.
    pareto_shrinkage: float | None
    pareto_cv_rmsle: float | None
    pareto_lomo_rmsle: float | None
    # CV improvement over plain ridge at that rung, as a fraction.
    pareto_cv_gain_fraction: float | None

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CorrectionMechanism:
    """Why the bounded correction helps across the size cut, measured.

    A hybrid scoring better than its own base model on an extrapolation is the
    kind of result that is worth distrusting until the mechanism is visible, so
    this records the three quantities that decide whether it is real:

    1. Is the base power law *biased* on the held-out machines, or just noisy?
       A bias is something a correction can address; scatter is not.
    2. Does the correction stay inside the range it was trained on? If it does,
       it is the Result 4c bound doing the work, and the correction cannot
       diverge however far the extrapolation goes.
    3. Does it move in the right direction on each machine? A bounded
       correction with the wrong sign would be worse than no correction.
    """

    model_name: str
    scope: str
    n_rows: int
    # Mean and sd of log(tau) - log(base prediction). A mean far from zero on
    # the held-out rows against ~0 on the training rows is bias, not scatter.
    base_residual_mean: float
    base_residual_sd: float
    # What the correction actually outputs, in log units.
    correction_mean: float
    correction_sd: float
    # Fraction of the base model's bias the correction supplies. 1.0 would be a
    # complete fix; the bound means it cannot reach that.
    bias_fraction_corrected: float
    # False would mean the correction extrapolated outside everything it saw,
    # which is the failure mode the tree form is supposed to rule out.
    correction_within_training_range: bool

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SizeCutRobustness:
    """The hybrid against its own base model at one size cut.

    Result 6c reports the ITER-size-matched cut, which is one rung of the Result 5
    sweep. A gain that appeared only there would be a property of that rung
    rather than of the method, so every usable cut is scored and the ones where
    the hybrid *loses* are reported alongside the ones where it wins.
    """

    n_train_machines: int
    size_ratio: float
    n_train_rows: int
    n_test_rows: int
    ridge_rmsle: float
    hybrid_rmsle: float
    analytic_rmsle: float
    hybrid_wins: bool
    # Same floor as Result 5d: below this the training set is small enough that
    # a model failing could be failing on sample size instead.
    well_powered: bool

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CorrectionHyperparameter:
    """One correction setting at the ITER-size-matched cut.

    The headline uses depth 2 and 200 rounds. Those were fixed before the
    result was known, but a reader has no way to tell that from the outside, so
    the surrounding grid is scored and reported: if the gain only existed at
    the reported setting, this is where that would show.
    """

    gbm_max_depth: int
    gbm_max_iter: int
    rmsle: float
    beats_ridge: bool

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class HybridAnalysis:
    """Everything Result 6 rests on."""

    n_rows: int
    n_machines_held_out: int
    feature_columns: tuple[str, ...]
    shrinkage_grid: tuple[float, ...]
    correction_labels: dict[str, str]
    size_cut_size_ratio: float
    size_cut_train_rows: int
    frontier: list[FrontierPoint]
    selection: list[SelectionOutcome]
    paired_against_baseline: list[PairedDifference]
    mechanism: list[CorrectionMechanism]
    size_cut_robustness: list[SizeCutRobustness]
    hyperparameter_robustness: list[CorrectionHyperparameter]
    # Does any rung of any family beat plain ridge under CV while keeping its
    # LOMO score? The headline yes/no of Result 6.
    any_pareto_improvement: bool
    per_machine: pd.DataFrame = field(repr=False)

    def to_json(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_machines_held_out": self.n_machines_held_out,
            "feature_columns": list(self.feature_columns),
            "shrinkage_grid": list(self.shrinkage_grid),
            "correction_labels": self.correction_labels,
            "size_cut_size_ratio": self.size_cut_size_ratio,
            "size_cut_train_rows": self.size_cut_train_rows,
            "frontier": [point.to_json() for point in self.frontier],
            "selection": [outcome.to_json() for outcome in self.selection],
            "paired_against_baseline": [
                gap.to_json() for gap in self.paired_against_baseline
            ],
            "mechanism": [row.to_json() for row in self.mechanism],
            "size_cut_robustness": [row.to_json() for row in self.size_cut_robustness],
            "hyperparameter_robustness": [
                row.to_json() for row in self.hyperparameter_robustness
            ],
            "any_pareto_improvement": self.any_pareto_improvement,
            "provenance": hdb5.dataset_provenance(),
        }


def analyze_hybrid(
    dataset: pd.DataFrame,
    *,
    shrinkage_grid: tuple[float, ...] = hdb5.SHRINKAGE_GRID,
    corrections: tuple[str, ...] = CORRECTIONS,
    feature_columns: tuple[str, ...] = hdb5.BLIND_FEATURE_COLUMNS,
    n_resamples: int = N_BOOTSTRAP_RESAMPLES,
) -> HybridAnalysis:
    """Score every rung of the shrinkage sweep under all three splits.

    The blind feature set throughout, for the reason Result 4 gives: the IPB98
    prior was fitted on this database including whichever machine is held out,
    so leaving it in would leak the answer into every fold. Using it here would
    additionally flatter the hybrid specifically, since its base model is a
    power law and the prior is a power law's output.
    """
    hybrids = hdb5.build_hybrid_models(shrinkage_grid, corrections=corrections)

    # One LOMO pass scores the zoo, the Result 4d control and every rung, and
    # attaches the Mahalanobis distance each rung's error is correlated against.
    per_machine = hdb5.extrapolation_report(
        dataset,
        feature_columns=feature_columns,
        include_controls=True,
        extra_models=hybrids,
    )
    cv_scores = hdb5.evaluate_models(
        dataset,
        feature_columns=feature_columns,
        include_controls=True,
        extra_models=hybrids,
    )
    cv_by_model = {score.model_name: score.cv_rmsle for score in cv_scores}

    splits = hdb5.size_ordered_splits(dataset)
    iter_split = hdb5.iter_matched_split(dataset, splits)
    size_cut = hdb5.score_size_split(
        dataset,
        iter_split,
        feature_columns=feature_columns,
        include_controls=True,
        extra_models=hybrids,
    )
    pooled_size = size_cut[size_cut["scope"] == "__pooled__"]
    size_by_model = dict(
        zip(
            pooled_size["model_name"].astype(str),
            pooled_size["rmsle"].astype(float),
            strict=True,
        )
    )

    lomo = per_machine.groupby(["model_name", "is_blind"], as_index=False).agg(
        mean_rmsle=("rmsle", "mean"),
        median_rmsle=("rmsle", "median"),
        worst_rmsle=("rmsle", "max"),
    )

    def _distance_rho(model_name: str) -> float:
        rows = per_machine[per_machine["model_name"] == model_name]
        return spearman(
            rows["feature_mahalanobis"].to_numpy(dtype=float),
            rows["rmsle"].to_numpy(dtype=float),
        )

    frontier: list[FrontierPoint] = []
    for name, is_blind, mean_rmsle, median_rmsle, worst_rmsle in zip(
        lomo["model_name"].astype(str),
        lomo["is_blind"].astype(bool),
        lomo["mean_rmsle"].astype(float),
        lomo["median_rmsle"].astype(float),
        lomo["worst_rmsle"].astype(float),
        strict=True,
                                                                 ):
        if name not in cv_by_model or name not in size_by_model:
            continue
        correction, shrinkage = _parse_hybrid_name(name)
        frontier.append(
            FrontierPoint(
                model_name=name,
                correction=correction,
                shrinkage=shrinkage,
                is_hybrid=correction != "",
                is_blind=bool(is_blind),
                cv_rmsle=float(cv_by_model[name]),
                lomo_mean_rmsle=float(mean_rmsle),
                lomo_median_rmsle=float(median_rmsle),
                lomo_worst_rmsle=float(worst_rmsle),
                size_cut_rmsle=float(size_by_model[name]),
                distance_spearman=_distance_rho(name),
            )
        )
    frontier.sort(key=lambda point: (point.correction, point.shrinkage, point.model_name))

    # Every rung against plain ridge, paired by machine. The marginal intervals
    # over 13 machines are far too wide to separate rungs this close together;
    # the paired difference removes the shared per-machine difficulty, exactly
    # as in Result 4.
    paired = [
        bootstrap_paired_difference(
            per_machine, point.model_name, BASELINE_MODEL, n_resamples=n_resamples
        )
        for point in frontier
        if point.is_hybrid and point.shrinkage > 0.0
    ]
    paired_by_model = {gap.model_a: gap for gap in paired}

    selection = [
        _select(correction, frontier, paired_by_model, cv_by_model)
        for correction in corrections
    ]

    return HybridAnalysis(
        n_rows=int(len(dataset)),
        n_machines_held_out=int(per_machine["tokamak"].nunique()),
        feature_columns=tuple(feature_columns),
        shrinkage_grid=tuple(shrinkage_grid),
        correction_labels={c: CORRECTION_LABELS[c] for c in corrections},
        size_cut_size_ratio=float(iter_split.size_ratio),
        size_cut_train_rows=int(iter_split.n_train_rows),
        frontier=frontier,
        selection=selection,
        paired_against_baseline=paired,
        mechanism=measure_correction_mechanism(
            dataset, iter_split, feature_columns=feature_columns
        ),
        size_cut_robustness=sweep_size_cuts(dataset, feature_columns=feature_columns),
        hyperparameter_robustness=sweep_correction_hyperparameters(
            dataset, iter_split, feature_columns=feature_columns
        ),
        any_pareto_improvement=any(
            outcome.pareto_shrinkage is not None for outcome in selection
        ),
        per_machine=per_machine,
    )


# Below this many training rows a model failing has two possible explanations
# and the split cannot separate them, exactly as in Result 5d.
MIN_WELL_POWERED_TRAIN_ROWS = 1000

# The grid around the reported correction setting.
HYPERPARAMETER_GRID_DEPTHS = (1, 2, 3)
HYPERPARAMETER_GRID_ITERATIONS = (100, 200, 400)


def sweep_size_cuts(
    dataset: pd.DataFrame,
    *,
    correction: str = "gbm",
    shrinkage: float = 1.0,
    feature_columns: tuple[str, ...] = hdb5.BLIND_FEATURE_COLUMNS,
) -> list[SizeCutRobustness]:
    """Score the hybrid against plain ridge at every usable size cut.

    Result 6c's number comes from one rung of the Result 5 sweep. This is the
    check that it is not *only* that rung, and the answer is a qualified one
    worth reporting rather than hiding: the hybrid wins at the larger cuts and
    loses at some smaller ones.
    """
    hybrids = hdb5.build_hybrid_models((shrinkage,), corrections=(correction,))
    name = hdb5.hybrid_model_name(correction, shrinkage)
    sweep, splits = hdb5.size_extrapolation_report(
        dataset, feature_columns=feature_columns, extra_models=hybrids
    )
    by_cut = sweep.pivot_table(
        index="n_train_machines", columns="model_name", values="rmsle"
    )

    rows: list[SizeCutRobustness] = []
    for split in splits:
        scores = by_cut.loc[split.n_train_machines]
        ridge = float(scores["ridge_loglinear"])
        hybrid = float(scores[name])
        rows.append(
            SizeCutRobustness(
                n_train_machines=split.n_train_machines,
                size_ratio=split.size_ratio,
                n_train_rows=split.n_train_rows,
                n_test_rows=split.n_test_rows,
                ridge_rmsle=ridge,
                hybrid_rmsle=hybrid,
                analytic_rmsle=float(scores["ipb98y2_analytic"]),
                hybrid_wins=bool(hybrid < ridge),
                well_powered=bool(split.n_train_rows >= MIN_WELL_POWERED_TRAIN_ROWS),
            )
        )
    return rows


def sweep_correction_hyperparameters(
    dataset: pd.DataFrame,
    split: hdb5.SizeSplit,
    *,
    feature_columns: tuple[str, ...] = hdb5.BLIND_FEATURE_COLUMNS,
    depths: tuple[int, ...] = HYPERPARAMETER_GRID_DEPTHS,
    iterations: tuple[int, ...] = HYPERPARAMETER_GRID_ITERATIONS,
) -> list[CorrectionHyperparameter]:
    """Score the grid around the reported correction setting at one cut."""
    from sklearn.pipeline import Pipeline

    variants = {
        f"grid_d{depth}_n{n_iter}": Pipeline(
            [
                (
                    "model",
                    hdb5.PowerLawResidualHybrid(
                        correction="gbm",
                        shrinkage=1.0,
                        gbm_max_depth=depth,
                        gbm_max_iter=n_iter,
                    ),
                )
            ]
        )
        for depth in depths
        for n_iter in iterations
    }
    scores = hdb5.score_size_split(
        dataset, split, feature_columns=feature_columns, extra_models=variants
    )
    pooled = scores[scores["scope"] == "__pooled__"].set_index("model_name")["rmsle"]
    ridge = float(pooled["ridge_loglinear"])
    return [
        CorrectionHyperparameter(
            gbm_max_depth=depth,
            gbm_max_iter=n_iter,
            rmsle=float(pooled[f"grid_d{depth}_n{n_iter}"]),
            beats_ridge=bool(float(pooled[f"grid_d{depth}_n{n_iter}"]) < ridge),
        )
        for depth in depths
        for n_iter in iterations
    ]


def measure_correction_mechanism(
    dataset: pd.DataFrame,
    split: hdb5.SizeSplit,
    *,
    correction: str = "gbm",
    shrinkage: float = 1.0,
    feature_columns: tuple[str, ...] = hdb5.BLIND_FEATURE_COLUMNS,
) -> list[CorrectionMechanism]:
    """Take one hybrid apart across the size cut and report what each half does.

    Fits the hybrid on the machines below the cut, then reads the base model's
    residual and the correction's output separately, pooled and per held-out
    machine. Only machines with enough rows to score are broken out, matching
    the threshold used everywhere else.
    """
    columns = list(feature_columns)
    labels = dataset[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()
    train_index = np.flatnonzero(np.isin(labels, list(split.train_machines)))
    features = dataset[columns]
    log_tau = np.log(dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float))

    model = hdb5.PowerLawResidualHybrid(correction=correction, shrinkage=shrinkage)
    model.fit(features.iloc[train_index], log_tau[train_index])
    # Taking the hybrid apart means calling its two halves directly, which steps
    # outside the suppression ``PowerLawResidualHybrid.predict`` applies to the
    # same numerics. The flags are the spurious BLAS ones described in
    # ``hdb5._suppress_benign_matmul_warnings``, not a property of these fits.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        train_correction = model.correction_.predict(features.iloc[train_index])
    lower, upper = float(train_correction.min()), float(train_correction.max())

    name = hdb5.hybrid_model_name(correction, shrinkage)

    def _row(scope: str, index: np.ndarray) -> CorrectionMechanism:
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            base_residual = log_tau[index] - model.base_.predict(features.iloc[index])
            correction_output = model.correction_.predict(features.iloc[index])
        bias = float(base_residual.mean())
        return CorrectionMechanism(
            model_name=name,
            scope=scope,
            n_rows=int(index.size),
            base_residual_mean=bias,
            base_residual_sd=float(base_residual.std()),
            correction_mean=float(correction_output.mean()),
            correction_sd=float(correction_output.std()),
            # Guarded: on the training rows the bias is ~0 by construction and
            # the ratio would be a meaningless large number.
            bias_fraction_corrected=(
                float(correction_output.mean() / bias) if abs(bias) > 1e-6 else float("nan")
            ),
            correction_within_training_range=bool(
                correction_output.min() >= lower and correction_output.max() <= upper
            ),
        )

    scopes: list[tuple[str, np.ndarray]] = [
        ("__train__", train_index),
        ("__held_out__", np.flatnonzero(np.isin(labels, list(split.test_machines)))),
    ]
    for machine in split.test_machines:
        machine_index = np.flatnonzero(labels == machine)
        if machine_index.size >= hdb5.MIN_HELD_OUT_ROWS:
            scopes.append((machine, machine_index))
    return [_row(scope, index) for scope, index in scopes]


def _parse_hybrid_name(model_name: str) -> tuple[str, float]:
    """Recover ``(correction, shrinkage)`` from a zoo key, or ``("", nan)``.

    Parsing the name back rather than threading the parameters through every
    frame keeps the hybrids interchangeable with every other model in the zoo,
    which is what lets all three splits score them with no special-casing.
    """
    if not model_name.startswith("hybrid_"):
        return "", float("nan")
    _, correction, shrinkage_token = model_name.split("_", 2)
    return correction, float(shrinkage_token.lstrip("s").replace("p", "."))


def _select(
    correction: str,
    frontier: list[FrontierPoint],
    paired_by_model: dict[str, PairedDifference],
    cv_by_model: dict[str, float],
) -> SelectionOutcome:
    """Resolve one correction family's sweep into what selection actually yields."""
    rungs = [point for point in frontier if point.correction == correction]
    if not rungs:
        raise ValueError(f"No rungs scored for correction {correction!r}.")
    baseline = next(point for point in frontier if point.model_name == BASELINE_MODEL)

    cv_selected = min(rungs, key=lambda point: point.cv_rmsle)
    oracle = min(rungs, key=lambda point: point.lomo_mean_rmsle)

    # Rungs that improve on ridge under CV and whose LOMO gap against ridge the
    # paired bootstrap cannot separate from zero. Among those, the best CV.
    qualifying = [
        point
        for point in rungs
        if point.shrinkage > 0.0
        and point.cv_rmsle < baseline.cv_rmsle
        and not paired_by_model[point.model_name].excludes_zero
    ]
    pareto = min(qualifying, key=lambda point: point.cv_rmsle) if qualifying else None

    return SelectionOutcome(
        correction=correction,
        baseline_cv_rmsle=baseline.cv_rmsle,
        baseline_lomo_rmsle=baseline.lomo_mean_rmsle,
        cv_selected_shrinkage=cv_selected.shrinkage,
        cv_selected_cv_rmsle=cv_selected.cv_rmsle,
        cv_selected_lomo_rmsle=cv_selected.lomo_mean_rmsle,
        cv_selected_size_cut_rmsle=cv_selected.size_cut_rmsle,
        oracle_shrinkage=oracle.shrinkage,
        oracle_lomo_rmsle=oracle.lomo_mean_rmsle,
        pareto_shrinkage=pareto.shrinkage if pareto else None,
        pareto_cv_rmsle=pareto.cv_rmsle if pareto else None,
        pareto_lomo_rmsle=pareto.lomo_mean_rmsle if pareto else None,
        pareto_cv_gain_fraction=(
            float(1.0 - pareto.cv_rmsle / baseline.cv_rmsle) if pareto else None
        ),
    )


# --- Figure -----------------------------------------------------------------

# Two swept families need two hues that survive colour-vision deficiency; this
# pair clears the adjacent-pair separation check with room to spare. The
# reference models are drawn in neutral ink with direct labels rather than
# given hues of their own: they are single points, not series, and adding five
# more colours to the legend would push the palette past what stays separable.
RIDGE_HUE, GBM_HUE = "#2a78d6", "#eb6834"
INK, MUTED = "#0b0b0b", "#52514e"
HUE_BY_CORRECTION = {"ridge": RIDGE_HUE, "gbm": GBM_HUE}
FAMILY_LABELS = {"ridge": "polynomial correction", "gbm": "boosted-tree correction"}


def plot_hybrid(analysis: HybridAnalysis) -> Path | None:
    """Two panels: the trade-off itself, and the knob that controls it.

    Left is the frontier, in the plane the question lives in: cross-validated
    error against held-out-machine error. Down-and-left is better on both, and
    the shape of the swept curve is the answer. Right is the same rungs against
    the damping factor, which is where CV-based selection can be seen picking
    its rung.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - plotting is optional
        return None

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(19.5, 5.6))
    for axis in axes:
        axis.grid(alpha=0.25, linewidth=0.6)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(MUTED)
            axis.spines[side].set_linewidth(0.8)
        axis.tick_params(colors=MUTED, labelsize=9)

    hybrids = [point for point in analysis.frontier if point.is_hybrid]
    references = [
        point for point in analysis.frontier if point.model_name in REFERENCE_MODELS
    ]

    # --- Left: the frontier -------------------------------------------------
    for correction, hue in HUE_BY_CORRECTION.items():
        rungs = sorted(
            (p for p in hybrids if p.correction == correction),
            key=lambda p: p.shrinkage,
        )
        if not rungs:
            continue
        axes[0].plot(
            [p.cv_rmsle for p in rungs],
            [p.lomo_mean_rmsle for p in rungs],
            "o-",
            color=hue,
            linewidth=2.0,
            markersize=7,
            label=FAMILY_LABELS[correction],
            zorder=3,
        )
        # Only the endpoints carry a damping label. A number on every rung
        # would be nine labels per curve and unreadable; the endpoints plus the
        # direction of travel are what the reader needs.
        axes[0].annotate(
            f"$\\lambda$ = {rungs[-1].shrinkage:g}",
            xy=(rungs[-1].cv_rmsle, rungs[-1].lomo_mean_rmsle),
            xytext=(0, 9),
            textcoords="offset points",
            fontsize=8.5,
            color=hue,
            ha="center",
        )

    for point in references:
        if point.model_name == BASELINE_MODEL:
            # Already the shrinkage-0 endpoint of both curves; labelling it
            # twice would imply two models where there is one.
            continue
        axes[0].plot(
            point.cv_rmsle,
            point.lomo_mean_rmsle,
            "D",
            color="none",
            markeredgecolor=MUTED,
            markeredgewidth=1.3,
            markersize=7,
            zorder=2,
        )
        axes[0].annotate(
            REFERENCE_LABELS.get(point.model_name, point.model_name),
            xy=(point.cv_rmsle, point.lomo_mean_rmsle),
            xytext=(7, -3),
            textcoords="offset points",
            fontsize=8.5,
            color=MUTED,
        )
    baseline = next(p for p in analysis.frontier if p.model_name == BASELINE_MODEL)
    # Below-left of the shared endpoint: the two curves leave it going up and to
    # the left, so anything above it collides with their lambda labels.
    axes[0].annotate(
        "ridge, log-linear\n($\\lambda$ = 0, both curves)",
        xy=(baseline.cv_rmsle, baseline.lomo_mean_rmsle),
        xytext=(-8, -26),
        textcoords="offset points",
        fontsize=8.5,
        color=INK,
        ha="right",
    )
    axes[0].axhline(
        baseline.lomo_mean_rmsle, color=MUTED, linewidth=0.9, linestyle=":", zorder=1
    )
    axes[0].axvline(
        baseline.cv_rmsle, color=MUTED, linewidth=0.9, linestyle=":", zorder=1
    )
    axes[0].set_xlabel("cross-validated log-RMSE (held-out discharge)", fontsize=10, color=INK)
    axes[0].set_ylabel(
        "leave-one-tokamak-out log-RMSE (held-out machine)", fontsize=10, color=INK
    )
    axes[0].set_title(
        "Result 6a: what flexibility costs out of distribution",
        fontsize=11.5,
        color=INK,
        loc="left",
    )
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")

    # --- Right: score against the damping factor ---------------------------
    for correction, hue in HUE_BY_CORRECTION.items():
        rungs = sorted(
            (p for p in hybrids if p.correction == correction),
            key=lambda p: p.shrinkage,
        )
        if not rungs:
            continue
        shrinkages = [p.shrinkage for p in rungs]
        axes[1].plot(
            shrinkages,
            [p.cv_rmsle for p in rungs],
            "o-",
            color=hue,
            linewidth=2.0,
            markersize=6,
            label=f"{FAMILY_LABELS[correction]}, CV",
        )
        axes[1].plot(
            shrinkages,
            [p.lomo_mean_rmsle for p in rungs],
            "s--",
            color=hue,
            linewidth=2.0,
            markersize=6,
            alpha=0.75,
            label=f"{FAMILY_LABELS[correction]}, LOMO",
        )

    for outcome in analysis.selection:
        hue = HUE_BY_CORRECTION[outcome.correction]
        axes[1].plot(
            outcome.cv_selected_shrinkage,
            outcome.cv_selected_cv_rmsle,
            "o",
            color="none",
            markeredgecolor=INK,
            markeredgewidth=1.6,
            markersize=13,
            zorder=4,
        )
    axes[1].annotate(
        "circled: the rung cross-validation selects",
        xy=(0.5, 0.03),
        xycoords="axes fraction",
        ha="center",
        fontsize=8.5,
        color=MUTED,
    )
    axes[1].set_xlabel("damping factor $\\lambda$ on the residual correction", fontsize=10, color=INK)
    axes[1].set_ylabel("log-RMSE", fontsize=10, color=INK)
    axes[1].set_title(
        "Result 6b: the two splits disagree about which rung is best",
        fontsize=11.5,
        color=INK,
        loc="left",
    )
    axes[1].legend(frameon=False, fontsize=8.5, loc="center left")

    # --- Right: the ITER direction, which is the one that matters ----------
    #
    # Separate panel rather than a third line on the middle one: it is a
    # different held-out set, so plotting it on the same axes would invite
    # reading a trend across three incomparable splits.
    for correction, hue in HUE_BY_CORRECTION.items():
        rungs = sorted(
            (p for p in hybrids if p.correction == correction),
            key=lambda p: p.shrinkage,
        )
        if not rungs:
            continue
        axes[2].plot(
            [p.shrinkage for p in rungs],
            [p.size_cut_rmsle for p in rungs],
            "o-",
            color=hue,
            linewidth=2.0,
            markersize=6,
            label=FAMILY_LABELS[correction],
        )
    analytic = next(
        (p for p in analysis.frontier if p.model_name == "ipb98y2_analytic"), None
    )
    if analytic is not None:
        axes[2].axhline(
            analytic.size_cut_rmsle, color=MUTED, linewidth=1.0, linestyle="--"
        )
        axes[2].annotate(
            f"IPB98(y,2), {analytic.size_cut_rmsle:.3f}  (not blind)",
            xy=(0.02, analytic.size_cut_rmsle),
            xytext=(0, 4),
            textcoords="offset points",
            fontsize=8.5,
            color=MUTED,
        )
    axes[2].axhline(
        baseline.size_cut_rmsle, color=MUTED, linewidth=0.9, linestyle=":"
    )
    axes[2].annotate(
        f"plain ridge, {baseline.size_cut_rmsle:.3f}",
        xy=(0.02, baseline.size_cut_rmsle),
        xytext=(0, 4),
        textcoords="offset points",
        fontsize=8.5,
        color=MUTED,
    )
    off_scale = [
        p
        for p in analysis.frontier
        if p.model_name in ("random_forest", "hist_gradient_boosting")
    ]
    if off_scale:
        axes[2].annotate(
            "  ".join(
                f"{REFERENCE_LABELS[p.model_name]} {p.size_cut_rmsle:.2f}"
                for p in sorted(off_scale, key=lambda p: p.size_cut_rmsle)
            )
            + "  (off scale)",
            xy=(0.5, 0.94),
            xycoords="axes fraction",
            ha="center",
            fontsize=8.5,
            color=MUTED,
        )
    axes[2].set_xlabel(
        "damping factor $\\lambda$ on the residual correction", fontsize=10, color=INK
    )
    axes[2].set_ylabel("log-RMSE across the ITER-size-matched cut", fontsize=10, color=INK)
    axes[2].set_title(
        f"Result 6c: predicting {analysis.size_cut_size_ratio:.2f}x beyond the training size",
        fontsize=11.5,
        color=INK,
        loc="left",
    )
    axes[2].legend(frameon=False, fontsize=9, loc="center right")

    figure.tight_layout()
    path = RESULTS_DIR / "hybrid.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path


def main() -> None:
    dataset = hdb5.prepare_dataset()
    analysis = analyze_hybrid(dataset)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frontier_frame = pd.DataFrame([point.to_json() for point in analysis.frontier])
    write_dataframe_csv_atomic(RESULTS_DIR / "hybrid_frontier.csv", frontier_frame)
    write_dataframe_csv_atomic(
        RESULTS_DIR / "hybrid_per_machine.csv", analysis.per_machine
    )
    write_json_strict(RESULTS_DIR / "hybrid.json", analysis.to_json())
    figure_path = plot_hybrid(analysis)

    print("--- Result 6: a power law plus a shrunk correction on its residuals ---")
    print(
        f"{analysis.n_rows} rows, {analysis.n_machines_held_out} machines held out one at a "
        f"time, {len(analysis.feature_columns)} blind features"
    )
    print(
        f"shrinkage grid: {', '.join(f'{s:g}' for s in analysis.shrinkage_grid)}"
        "   (0 is exactly ridge_loglinear)\n"
    )

    header = (
        f"  {'model':<26}{'CV':>8}{'LOMO':>8}{'worst':>8}"
        f"{'size cut':>10}{'rho(dist)':>11}"
    )
    for correction in analysis.correction_labels:
        print(f"  {correction} correction: {analysis.correction_labels[correction]}")
        print(header)
        rungs = [p for p in analysis.frontier if p.correction == correction]
        for point in sorted(rungs, key=lambda p: p.shrinkage):
            gap = next(
                (
                    g
                    for g in analysis.paired_against_baseline
                    if g.model_a == point.model_name
                ),
                None,
            )
            flag = ""
            if gap is not None:
                flag = "  LOMO worse than ridge" if gap.excludes_zero else "  ties ridge on LOMO"
            print(
                f"  lambda = {point.shrinkage:<17g}{point.cv_rmsle:>8.3f}"
                f"{point.lomo_mean_rmsle:>8.3f}{point.lomo_worst_rmsle:>8.3f}"
                f"{point.size_cut_rmsle:>10.3f}{point.distance_spearman:>11.2f}{flag}"
            )
        print()

    print("  reference points (Result 4 models, same features, same splits)")
    print(header)
    for point in analysis.frontier:
        if point.model_name not in REFERENCE_MODELS:
            continue
        marker = " " if point.is_blind else "*"
        print(
            f"{marker} {REFERENCE_LABELS[point.model_name]:<25}{point.cv_rmsle:>8.3f}"
            f"{point.lomo_mean_rmsle:>8.3f}{point.lomo_worst_rmsle:>8.3f}"
            f"{point.size_cut_rmsle:>10.3f}{point.distance_spearman:>11.2f}"
        )
    print("  * fitted on this database, held-out machine included; not a blind baseline")

    print("\n--- what selecting the damping factor honestly gets you ---")
    for outcome in analysis.selection:
        print(f"  {outcome.correction} correction:")
        print(
            f"    CV selects lambda = {outcome.cv_selected_shrinkage:g}: "
            f"CV {outcome.cv_selected_cv_rmsle:.3f} (ridge {outcome.baseline_cv_rmsle:.3f}), "
            f"LOMO {outcome.cv_selected_lomo_rmsle:.3f} (ridge {outcome.baseline_lomo_rmsle:.3f}), "
            f"size cut {outcome.cv_selected_size_cut_rmsle:.3f}"
        )
        print(
            f"    best possible LOMO rung is lambda = {outcome.oracle_shrinkage:g} at "
            f"{outcome.oracle_lomo_rmsle:.3f}, which CV cannot see"
        )
        gain = outcome.pareto_cv_gain_fraction
        if outcome.pareto_shrinkage is None or gain is None:
            print(
                "    no rung improves CV while tying ridge on LOMO: "
                "every CV gain is paid for out of distribution"
            )
        else:
            print(
                f"    lambda = {outcome.pareto_shrinkage:g} improves CV by "
                f"{gain * 100:.1f}% "
                f"({outcome.pareto_cv_rmsle:.3f}) and ties ridge on LOMO "
                f"({outcome.pareto_lomo_rmsle:.3f})"
            )

    print("\n--- why the bounded correction helps across the size cut ---")
    print(f"  {'scope':<14}{'rows':>6}{'base resid':>12}{'correction':>12}{'bias fixed':>12}{'bounded':>9}")
    for row in analysis.mechanism:
        fraction = (
            "     n/a" if np.isnan(row.bias_fraction_corrected)
            else f"{row.bias_fraction_corrected * 100:>10.0f}%"
        )
        print(
            f"  {row.scope:<14}{row.n_rows:>6}"
            f"{row.base_residual_mean:>+12.3f}{row.correction_mean:>+12.3f}{fraction:>12}"
            f"{str(row.correction_within_training_range):>9}"
        )
    print(
        "  base resid = mean log(tau) - log(base prediction); a nonzero mean off the\n"
        "  training rows is bias the correction can address rather than scatter it cannot"
    )

    print("\n--- does the size-cut gain survive the correction's own settings? ---")
    grid = analysis.hyperparameter_robustness
    print(f"  {'depth':>6}{'rounds':>8}{'log-RMSE':>9}  beats ridge")
    for setting in grid:
        print(
            f"  {setting.gbm_max_depth:>6}{setting.gbm_max_iter:>8}{setting.rmsle:>9.3f}"
            f"  {'yes' if setting.beats_ridge else 'no'}"
        )
    print(f"  {sum(r.beats_ridge for r in grid)}/{len(grid)} settings beat plain ridge")

    print("\n--- and does it survive at size cuts other than the matched one? ---")
    print(f"  {'machines':>9}{'ratio':>7}{'train':>8}{'ridge':>8}{'hybrid':>8}  powered")
    for cut in analysis.size_cut_robustness:
        print(
            f"  {cut.n_train_machines:>9}{cut.size_ratio:>7.2f}{cut.n_train_rows:>8}"
            f"{cut.ridge_rmsle:>8.3f}{cut.hybrid_rmsle:>8.3f}"
            f"  {'yes' if cut.well_powered else 'no'}"
            f"{'  <- hybrid wins' if cut.hybrid_wins else ''}"
        )
    powered = [c for c in analysis.size_cut_robustness if c.well_powered]
    print(
        f"  hybrid beats ridge at {sum(c.hybrid_wins for c in powered)}/{len(powered)} "
        "well-powered cuts; the gain is not uniform across the sweep"
    )

    verdict = (
        "at least one rung is a genuine improvement on both splits"
        if analysis.any_pareto_improvement
        else "no rung improves cross-validation without paying for it on a new machine"
    )
    print(f"\nverdict: {verdict}")
    if figure_path:
        print(f"figure: {figure_path}")


if __name__ == "__main__":
    main()
