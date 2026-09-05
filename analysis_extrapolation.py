"""Does a learned confinement model transfer to a tokamak it has never seen?

Run ``python3 analysis_extrapolation.py`` to regenerate everything under
``results/`` for Result 4: the interpolation-vs-extrapolation table, the
per-machine breakdown, the distance correlation, and the figure.

``results/RESULTS.md`` Results 1 to 3 treat the confinement scaling law as a
linear algebra problem. This script asks the question those results set up:
grouped cross-validation by discharge scores a model on *machines it has already
seen*, which is interpolation. A scaling law exists to predict a device that
does not exist yet. So hold out an entire tokamak.

    Result 4a  The model ranking under leave-one-tokamak-out is the reverse of
               the ranking under grouped CV. The tree ensembles win the first
               and lose the second.
    Result 4b  The trees' error is explained by how far the held-out machine
               sits outside the training distribution. The power law's is not.
    Result 4c  Two distinct, separately measurable failure modes, one of which
               is a structural bound on what a tree ensemble can output at all.

The narrative built on these numbers is in ``results/RESULTS.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

import hdb5
from figures import save_figure
from storage import write_dataframe_csv_atomic, write_json_strict

if TYPE_CHECKING:  # imported for typing only; the factory imports it lazily
    from sklearn.pipeline import Pipeline

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Models that both fit a functional form to the data and never see the held-out
# machine in any form. The ranking claim is about these three.
#
# ``ipb98y2_analytic`` is excluded because its exponents were fitted on this
# database, held-out machine included, so it is a reference ceiling rather than
# a blind competitor; saying "the published law wins" without that caveat would
# overclaim. ``mean_baseline`` is excluded because it is a floor that fits
# nothing, and it lands last under both splits by construction: leaving it in
# would count a model that cannot participate as evidence that the order was
# preserved. Both still appear in every reported table.
CONTENDER_MODELS = ("ridge_loglinear", "random_forest", "hist_gradient_boosting")

# Controls, scored alongside the contenders but excluded from the ranking claim.
# ``ridge_log_quadratic`` exists to separate two explanations for ridge winning
# on an unseen machine: that the log-linear power-law form is right, or that
# ridge is merely the only model in the zoo that can extrapolate at all. It is
# flexible (curvature and every pairwise log interaction) but still polynomial,
# so it extrapolates. See Result 4d.
# Kept in step with the two factories that supply them, one rung each. They are
# separate constants because they come from different places: hdb5 owns the
# control the ``extrapolate`` CLI can run, this module owns the extra rung that
# is too slow to belong in a default run. ``test_extrapolation`` asserts both
# match their factory, so a new rung cannot be added in one place only.
CONTROL_MODELS = ("ridge_log_quadratic",)
LADDER_MODELS = ("ridge_log_cubic",)

# Reporting order for Result 4d, least to most flexible. Degree 1 is plain
# ridge, already in the zoo; degrees 2 and 3 are the controls; the tree
# ensembles are the nonparametric end of the same axis.
FLEXIBILITY_LADDER = (
    ("ridge_loglinear", "log-linear (degree 1)"),
    ("ridge_log_quadratic", "log-quadratic (degree 2)"),
    ("ridge_log_cubic", "log-cubic (degree 3)"),
    ("hist_gradient_boosting", "gradient-boosted trees"),
    ("random_forest", "random forest"),
)

# Machine-level bootstrap. The sampling unit is the tokamak, not the row: the
# claim is about how a model behaves on an unseen *machine*, and there are only
# 13 of those, so resampling rows would return intervals that are far too narrow.
N_BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20240617


def build_flexibility_ladder() -> dict[str, "Pipeline"]:
    """The rung of the ladder above ``hdb5.build_control_models``.

    ``hdb5`` supplies degree 2. Degree 3 is added here because it is specific to
    this analysis rather than something the ``extrapolate`` CLI should carry: on
    nine log features it expands to 219 terms, which is enough to make the point
    about flexibility and slow enough that it does not belong in a default run.
    """
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    return {
        "ridge_log_cubic": Pipeline(
            [
                ("expand", PolynomialFeatures(degree=3, include_bias=False)),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=1.0, solver="svd")),
            ]
        ),
    }

# A tree ensemble predicts by averaging training targets, so its output is
# bounded by the training target range no matter what its features say. Whether
# that bound *matters* for a given machine is a question about how many of its
# rows sit above the ceiling, not about whether any single row does.
#
# Ranking on the maximum alone would fire whenever one lucky shot happens to be
# the database record holder, which says nothing about the machine. Requiring a
# material share of its rows to be unreachable is the claim actually being made:
# at least one row in twenty cannot be predicted by any tree in the forest.
MIN_TRUNCATED_ROW_FRACTION = 0.05


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, implemented directly rather than pulled from scipy.

    Pearson correlation on midranks. Midranks (rather than ordinal ranks) keep
    the value correct when machines tie on a metric; ``argsort`` twice, the
    usual shortcut, silently breaks ties arbitrarily instead.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.size != b.size:
        raise ValueError("spearman needs two equal-length vectors.")
    if a.size < 2:
        return float("nan")
    rank_a, rank_b = _midranks(a), _midranks(b)
    centered_a, centered_b = rank_a - rank_a.mean(), rank_b - rank_b.mean()
    denominator = float(np.linalg.norm(centered_a) * np.linalg.norm(centered_b))
    if denominator == 0.0:  # a constant vector has no rank correlation
        return float("nan")
    return float(centered_a @ centered_b / denominator)


def _midranks(values: np.ndarray) -> np.ndarray:
    """Ranks from 1, with tied values sharing the mean of the ranks they span."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(1, values.size + 1, dtype=float)
    sorted_values = values[order]
    start = 0
    for stop in range(1, values.size + 1):
        if stop == values.size or sorted_values[stop] != sorted_values[start]:
            if stop - start > 1:
                ranks[order[start:stop]] = ranks[order[start:stop]].mean()
            start = stop
    return ranks


@dataclass(frozen=True)
class MachineBootstrap:
    """A percentile interval for one model's mean log-RMSE over held-out machines."""

    model_name: str
    mean_rmsle: float
    ci_low: float
    ci_high: float

    def to_json(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "mean_rmsle": self.mean_rmsle,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
        }


@dataclass(frozen=True)
class PairedDifference:
    """A percentile interval for the gap between two models, paired by machine.

    Paired matters. The machines differ enormously in how hard they are, so the
    two marginal intervals overlap heavily even where one model beats the other
    on every single machine. Resampling the *difference* on a common machine
    draw removes that shared difficulty and is the honest test of the gap.
    """

    model_a: str
    model_b: str
    mean_difference: float
    ci_low: float
    ci_high: float
    excludes_zero: bool
    n_machines_a_worse: int
    n_machines: int

    def to_json(self) -> dict[str, object]:
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "mean_difference": self.mean_difference,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "excludes_zero": self.excludes_zero,
            "n_machines_a_worse": self.n_machines_a_worse,
            "n_machines": self.n_machines,
        }


def _machine_by_model(per_machine: pd.DataFrame) -> pd.DataFrame:
    """log-RMSE as a machines-by-models table, the unit the bootstrap resamples."""
    return per_machine.pivot_table(index="tokamak", columns="model_name", values="rmsle")


def _resample_indices(n_machines: int, n_resamples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_machines, size=(n_resamples, n_machines))


def bootstrap_over_machines(
    per_machine: pd.DataFrame,
    *,
    n_resamples: int = N_BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> list[MachineBootstrap]:
    """Percentile intervals for each model's mean log-RMSE across held-out machines.

    Every model is resampled on the *same* machine draws, so the intervals are
    directly comparable and :func:`bootstrap_paired_difference` can reuse them.
    """
    table = _machine_by_model(per_machine)
    values = table.to_numpy(dtype=float)
    draws = _resample_indices(len(table), n_resamples, seed)
    means = values[draws].mean(axis=1)
    low, high = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    return [
        MachineBootstrap(
            model_name=str(name),
            mean_rmsle=float(values[:, index].mean()),
            ci_low=float(low[index]),
            ci_high=float(high[index]),
        )
        for index, name in enumerate(table.columns)
    ]


def bootstrap_paired_difference(
    per_machine: pd.DataFrame,
    model_a: str,
    model_b: str,
    *,
    n_resamples: int = N_BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> PairedDifference:
    """Interval for ``mean(log-RMSE of a minus log-RMSE of b)`` over machines, paired by machine."""
    table = _machine_by_model(per_machine)
    for model in (model_a, model_b):
        if model not in table.columns:
            raise ValueError(f"{model!r} was not scored; cannot difference it.")
    difference = (table[model_a] - table[model_b]).to_numpy(dtype=float)
    draws = _resample_indices(len(table), n_resamples, seed)
    means = difference[draws].mean(axis=1)
    low, high = (float(v) for v in np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)]))
    return PairedDifference(
        model_a=model_a,
        model_b=model_b,
        mean_difference=float(difference.mean()),
        ci_low=low,
        ci_high=high,
        excludes_zero=bool(low > 0.0 or high < 0.0),
        n_machines_a_worse=int(np.sum(difference > 0.0)),
        n_machines=int(difference.size),
    )


@dataclass(frozen=True)
class ModelTransfer:
    """One model's score under both splits, and how badly it survives the swap."""

    model_name: str
    is_blind: bool
    cv_rmsle: float
    lomo_mean_rmsle: float
    lomo_median_rmsle: float
    lomo_worst_rmsle: float
    worst_machine: str
    # lomo_mean / cv. 1.0 means the model transfers perfectly; large means its
    # cross-validated score was measuring interpolation and nothing else.
    degradation_factor: float
    cv_rank: int
    lomo_rank: int
    # Spearman rho between the model's per-machine log-RMSE and how far that
    # machine sits outside the training data. High means the model's failures
    # are extrapolation failures specifically.
    distance_spearman: float

    def to_json(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "is_blind": self.is_blind,
            "cv_rmsle": self.cv_rmsle,
            "lomo_mean_rmsle": self.lomo_mean_rmsle,
            "lomo_median_rmsle": self.lomo_median_rmsle,
            "lomo_worst_rmsle": self.lomo_worst_rmsle,
            "worst_machine": self.worst_machine,
            "degradation_factor": self.degradation_factor,
            "cv_rank": self.cv_rank,
            "lomo_rank": self.lomo_rank,
            "distance_spearman": self.distance_spearman,
        }


@dataclass(frozen=True)
class TruncationFinding:
    """A machine whose true confinement times a tree ensemble cannot reach.

    Trees average training targets, so every prediction lies inside
    ``[min(y_train), max(y_train)]``. When a held-out machine's confinement
    times run above that range, the error is not a modelling shortfall that
    more data or better features would fix. It is the functional form.
    """

    tokamak: str
    n_held_out_rows: int
    fraction_above_train_max: float
    log_headroom: float
    # exp(log_headroom): how many times higher the machine's best shot is than
    # anything any tree in the forest is able to output.
    headroom_ratio: float
    tree_rmsle: float
    power_law_rmsle: float

    def to_json(self) -> dict[str, object]:
        return {
            "tokamak": self.tokamak,
            "n_held_out_rows": self.n_held_out_rows,
            "fraction_above_train_max": self.fraction_above_train_max,
            "log_headroom": self.log_headroom,
            "headroom_ratio": self.headroom_ratio,
            "tree_rmsle": self.tree_rmsle,
            "power_law_rmsle": self.power_law_rmsle,
        }


@dataclass(frozen=True)
class ExtrapolationAnalysis:
    feature_columns: list[str]
    n_rows: int
    # Content identity of the dataset file these numbers came from, captured
    # when the analysis runs rather than when it is serialized: fingerprinting
    # at write time would re-read whatever happens to be on disk by then and
    # would ignore an explicit ``dataset_path`` entirely.
    #
    # ``None`` when the caller passed a frame it built itself rather than a
    # ``dataset_path``. There is no file to identify in that case, and
    # naming the default one would stamp these numbers with the provenance
    # of bytes they never read.
    provenance: dict[str, object] | None
    n_machines_held_out: int
    machines_held_out: list[str]
    n_machines_excluded: int
    transfers: list[ModelTransfer]
    truncation: list[TruncationFinding]
    bootstrap: list[MachineBootstrap]
    paired_differences: list[PairedDifference]
    per_machine: pd.DataFrame = field(repr=False)
    # Spearman rho between the CV ranking and the LOMO ranking over
    # CONTENDER_MODELS. -1.0 is an exact inversion.
    ranking_spearman: float = float("nan")
    # Whether the contender order under one split is the exact reverse of the
    # other. With three contenders this is the statistic worth quoting.
    ranking_exactly_reversed: bool = False
    n_contenders: int = 0

    def to_json(self) -> dict[str, object]:
        return {
            # The dataset fingerprint leads: every number below it is a
            # statement about these exact bytes and nothing else.
            "dataset": self.provenance,
            "feature_columns": self.feature_columns,
            "n_rows": self.n_rows,
            "n_machines_held_out": self.n_machines_held_out,
            "machines_held_out": self.machines_held_out,
            "n_machines_excluded": self.n_machines_excluded,
            "ranking_spearman": self.ranking_spearman,
            "ranking_exactly_reversed": self.ranking_exactly_reversed,
            "n_contenders": self.n_contenders,
            "contender_models": list(CONTENDER_MODELS),
            "control_models": list(CONTROL_MODELS),
            "ladder_models": list(LADDER_MODELS),
            "transfers": [transfer.to_json() for transfer in self.transfers],
            "truncation": [finding.to_json() for finding in self.truncation],
            "bootstrap": [interval.to_json() for interval in self.bootstrap],
            "paired_differences": [gap.to_json() for gap in self.paired_differences],
            "flexibility_ladder": [name for name, _ in FLEXIBILITY_LADDER],
        }


def analyze_extrapolation(
    dataset: pd.DataFrame,
    *,
    dataset_path: Path | str | None = None,
    feature_columns: tuple[str, ...] = hdb5.BLIND_FEATURE_COLUMNS,
    min_rows: int = hdb5.MIN_HELD_OUT_ROWS,
    n_splits: int = hdb5.N_CV_FOLDS,
    include_controls: bool = True,
    include_ladder: bool = True,
) -> ExtrapolationAnalysis:
    """Score every model both ways and measure what separates the two rankings.

    Both splits run on the *same* ``feature_columns``. That matters: the default
    model feature set includes the analytic IPB98 prior, whose exponents were
    fitted on this database including whichever machine is held out. Leaving it
    in would leak, and dropping it only for the leave-one-out arm would confound
    the feature set with the split. Holding it fixed means the single difference
    between the two numbers below is what the split holds out.
    """
    extra_models = build_flexibility_ladder() if include_ladder else None
    report = hdb5.extrapolation_report(
        dataset,
        feature_columns=feature_columns,
        min_rows=min_rows,
        include_controls=include_controls,
        extra_models=extra_models,
    )
    cv_scores = hdb5.evaluate_models(
        dataset,
        n_splits=n_splits,
        feature_columns=feature_columns,
        include_controls=include_controls,
        extra_models=extra_models,
    )
    cv_by_name = {score.model_name: score.cv_rmsle for score in cv_scores}

    machines = hdb5.eligible_tokamaks(dataset, min_rows=min_rows)
    all_machines = dataset[hdb5.TOKAMAK_LABEL_COLUMN].nunique()

    grouped = report.groupby("model_name")
    lomo_mean = grouped["rmsle"].mean()
    cv_order = sorted(cv_by_name, key=lambda name: cv_by_name[name])
    lomo_order = list(lomo_mean.sort_values().index)

    transfers: list[ModelTransfer] = []
    for name, rows in grouped:
        model_name = str(name)
        worst_row = rows.loc[rows["rmsle"].idxmax()]
        mean_rmsle = float(rows["rmsle"].mean())
        cv_rmsle = float(cv_by_name[model_name])
        transfers.append(
            ModelTransfer(
                model_name=model_name,
                is_blind=bool(rows["is_blind"].iloc[0]),
                cv_rmsle=cv_rmsle,
                lomo_mean_rmsle=mean_rmsle,
                lomo_median_rmsle=float(rows["rmsle"].median()),
                lomo_worst_rmsle=float(worst_row["rmsle"]),
                worst_machine=str(worst_row["tokamak"]),
                degradation_factor=mean_rmsle / cv_rmsle,
                cv_rank=cv_order.index(model_name) + 1,
                lomo_rank=lomo_order.index(model_name) + 1,
                distance_spearman=spearman(
                    rows["rmsle"].to_numpy(dtype=float),
                    rows["feature_mahalanobis"].to_numpy(dtype=float),
                ),
            )
        )
    transfers.sort(key=lambda transfer: transfer.lomo_mean_rmsle)

    contenders = [t for t in transfers if t.model_name in CONTENDER_MODELS]
    ranking_spearman = spearman(
        np.array([transfer.cv_rank for transfer in contenders], dtype=float),
        np.array([transfer.lomo_rank for transfer in contenders], dtype=float),
    )
    # With three contenders this rho can only take a handful of values, so the
    # exact-reversal flag is the honest headline and rho is the supporting note.
    by_cv = sorted(contenders, key=lambda transfer: transfer.cv_rank)
    by_lomo = sorted(contenders, key=lambda transfer: transfer.lomo_rank)
    ranking_exactly_reversed = [t.model_name for t in by_cv] == [
        t.model_name for t in reversed(by_lomo)
    ]

    scored = set(report["model_name"])
    # The gaps the narrative actually claims, each paired by machine.
    candidate_gaps = (
        ("random_forest", "ridge_loglinear"),
        ("hist_gradient_boosting", "ridge_loglinear"),
        ("ridge_log_quadratic", "ridge_loglinear"),
        ("random_forest", "ridge_log_quadratic"),
    )
    paired_differences = [
        bootstrap_paired_difference(report, a, b)
        for a, b in candidate_gaps
        if a in scored and b in scored
    ]

    return ExtrapolationAnalysis(
        feature_columns=list(feature_columns),
        n_rows=int(len(dataset)),
        provenance=hdb5.dataset_provenance(dataset_path) if dataset_path is not None else None,
        n_machines_held_out=len(machines),
        machines_held_out=machines,
        n_machines_excluded=int(all_machines) - len(machines),
        transfers=transfers,
        truncation=_truncation_findings(report),
        bootstrap=bootstrap_over_machines(report),
        paired_differences=paired_differences,
        per_machine=report,
        ranking_spearman=ranking_spearman,
        ranking_exactly_reversed=ranking_exactly_reversed,
        n_contenders=len(contenders),
    )


def _truncation_findings(report: pd.DataFrame) -> list[TruncationFinding]:
    """Machines whose targets run above what a tree ensemble can output."""
    findings: list[TruncationFinding] = []
    for tokamak, rows in report.groupby("tokamak"):
        headroom = float(rows["log_target_headroom"].iloc[0])
        fraction = float(rows["target_above_train_max_fraction"].iloc[0])
        if headroom <= 0.0 or fraction < MIN_TRUNCATED_ROW_FRACTION:
            continue
        by_model = rows.set_index("model_name")["rmsle"]
        findings.append(
            TruncationFinding(
                tokamak=str(tokamak),
                n_held_out_rows=int(rows["n_held_out_rows"].iloc[0]),
                fraction_above_train_max=fraction,
                log_headroom=headroom,
                headroom_ratio=float(np.exp(headroom)),
                tree_rmsle=float(by_model["random_forest"]),
                power_law_rmsle=float(by_model["ridge_loglinear"]),
            )
        )
    findings.sort(key=lambda finding: finding.log_headroom, reverse=True)
    return findings


def plot_extrapolation(analysis: ExtrapolationAnalysis) -> Path | None:
    """Three panels: the claim, the mechanism, and what the constraint buys.

    Left is Result 4a: each model under both splits on one axis, so the lines
    crossing *is* the result. Middle is Result 4b: per-machine error against
    distance from the training data, rising for the trees and flat for the power
    law. Right is Result 4d: median against worst case along the flexibility
    ladder, which is where the constrained form actually earns its keep.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - plotting is optional
        return None

    blue, orange, green = "#2a78d6", "#eb6834", "#3f8f5c"
    ink, muted = "#0b0b0b", "#52514e"
    style = {
        "random_forest": (orange, "random forest"),
        "hist_gradient_boosting": ("#c8873a", "hist gradient boosting"),
        "ridge_loglinear": (blue, "ridge, log-linear"),
        "ipb98y2_analytic": (green, "IPB98(y,2), analytic"),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        1, 3, figsize=(13.5, 5.4), gridspec_kw={"width_ratios": [1.0, 1.25, 1.05]}
    )
    for axis in axes:
        axis.grid(alpha=0.25, linewidth=0.6)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(muted)
            axis.spines[side].set_linewidth(0.8)
        axis.tick_params(colors=muted, labelsize=9)

    # --- Left: the ranking inversion, with machine-level intervals ---------
    intervals = {interval.model_name: interval for interval in analysis.bootstrap}
    plotted = [t for t in analysis.transfers if t.model_name in style]
    span = max(t.cv_rmsle for t in plotted) - min(t.cv_rmsle for t in plotted)
    cv_offsets: dict[str, float] = {}
    previous = -np.inf
    for transfer in sorted(plotted, key=lambda t: t.cv_rmsle):
        crowded = transfer.cv_rmsle - previous < 0.04 * span
        cv_offsets[transfer.model_name] = 8.0 if crowded else -3.0
        previous = transfer.cv_rmsle

    for transfer in plotted:
        color, label = style[transfer.model_name]
        axes[0].plot(
            [0, 1],
            [transfer.cv_rmsle, transfer.lomo_mean_rmsle],
            "o-",
            color=color,
            linewidth=2.0,
            markersize=7,
            label=label,
        )
        interval = intervals.get(transfer.model_name)
        if interval is not None:
            axes[0].plot(
                [1, 1],
                [interval.ci_low, interval.ci_high],
                "-",
                color=color,
                linewidth=1.2,
                alpha=0.55,
            )
        axes[0].annotate(
            f"{transfer.lomo_mean_rmsle:.3f}",
            xy=(1, transfer.lomo_mean_rmsle),
            xytext=(10, -3),
            textcoords="offset points",
            fontsize=13,
            color=color,
        )
        axes[0].annotate(
            f"{transfer.cv_rmsle:.3f}",
            xy=(0, transfer.cv_rmsle),
            xytext=(-40, cv_offsets[transfer.model_name]),
            textcoords="offset points",
            fontsize=13,
            color=color,
        )
    axes[0].set_xlim(-0.75, 1.95)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(
        ["held out:\ndischarge", "held out:\nmachine"],
        fontsize=14,
        color=ink,
    )
    axes[0].set_ylabel("log-RMSE (lower is better)", fontsize=13, color=muted)
    axes[0].set_title(
        "Interpolation against extrapolation",
        fontsize=15,
        color=ink,
    )
    axes[0].legend(frameon=False, fontsize=13, loc="upper left", labelcolor=muted)

    # --- Middle: error against distance from the training distribution -----
    per_machine = analysis.per_machine
    for model_name in ("random_forest", "ridge_loglinear"):
        rows = per_machine[per_machine["model_name"] == model_name].sort_values(
            "feature_mahalanobis"
        )
        color, label = style[model_name]
        rho = next(
            t.distance_spearman for t in analysis.transfers if t.model_name == model_name
        )
        axes[1].plot(
            rows["feature_mahalanobis"],
            rows["rmsle"],
            "o",
            color=color,
            markersize=8,
            label=f"{label}   (rho = {rho:+.2f})",
        )
    truncated = {finding.tokamak for finding in analysis.truncation}
    forest_rows = per_machine[per_machine["model_name"] == "random_forest"]
    for _, row in forest_rows.iterrows():
        machine = str(row["tokamak"])
        if machine in truncated:
            axes[1].plot(
                row["feature_mahalanobis"],
                row["rmsle"],
                "o",
                markersize=15,
                markerfacecolor="none",
                markeredgecolor=ink,
                markeredgewidth=1.4,
            )
        axes[1].annotate(
            machine,
            xy=(row["feature_mahalanobis"], row["rmsle"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=11,
            color=ink if machine in truncated else muted,
            fontweight="bold" if machine in truncated else "normal",
        )
    lowest = float(per_machine["rmsle"].min())
    highest = float(
        per_machine[per_machine["model_name"].isin(("random_forest", "ridge_loglinear"))][
            "rmsle"
        ].max()
    )
    # Open a band under the data for the caption rather than printing over points.
    axes[1].set_ylim(lowest - 0.10 * (highest - lowest), highest + 0.30 * (highest - lowest))
    axes[1].set_xlabel(
        "Mahalanobis distance from training data",
        fontsize=13,
        color=muted,
    )
    axes[1].set_ylabel("log-RMSE on the held-out machine", fontsize=13, color=muted)
    axes[1].set_title(
        "Error against extrapolation distance",
        fontsize=15,
        color=ink,
    )
    axes[1].legend(frameon=False, fontsize=13, loc="upper left", labelcolor=muted)

    # --- Right: median against worst case along the flexibility ladder -----
    by_name = {transfer.model_name: transfer for transfer in analysis.transfers}
    rungs = [(name, label) for name, label in FLEXIBILITY_LADDER if name in by_name]
    positions = np.arange(len(rungs))
    medians = [by_name[name].lomo_median_rmsle for name, _ in rungs]
    worst = [by_name[name].lomo_worst_rmsle for name, _ in rungs]
    bounded = ["forest" in label or "trees" in label for _, label in rungs]

    # A tree ensemble is not "degree 4", so the two families are drawn as separate
    # runs. Joining them would assert a continuum that does not exist, and the
    # step between them is the panel's point rather than part of a trend.
    split = bounded.index(True) if True in bounded else len(rungs)
    for series, color, label in (
        (medians, blue, "median of 13 machines"),
        (worst, orange, "worst single machine"),
    ):
        axes[2].plot(positions[:split], series[:split], "o-", color=color, linewidth=2.0,
                     markersize=8, label=label)
        axes[2].plot(positions[split:], series[split:], "o-", color=color, linewidth=2.0,
                     markersize=8)
    if 0 < split < len(rungs):
        axes[2].axvline(split - 0.5, color=muted, linestyle=":", linewidth=1.0)
        axes[2].annotate("polynomial\n(unbounded)", xy=(split - 0.62, 0.02),
                         xycoords=("data", "axes fraction"), ha="right", va="bottom",
                         fontsize=11, color=muted)
        axes[2].annotate("trees\n(bounded)", xy=(split - 0.38, 0.02),
                         xycoords=("data", "axes fraction"), ha="left", va="bottom",
                         fontsize=11, color=muted)
    for index, is_bounded in enumerate(bounded):
        if is_bounded:
            axes[2].plot(positions[index], worst[index], "o", markersize=15,
                         markerfacecolor="none", markeredgecolor=ink, markeredgewidth=1.4)
    axes[2].set_yscale("log")
    # Headroom above the worst point so the two family labels sit in clear space.
    axes[2].set_ylim(min(medians) * 0.6, max(worst) * 3.0)
    axes[2].set_xticks(positions)
    axes[2].set_xticklabels([label for _, label in rungs], rotation=28, ha="right",
                            fontsize=13, color=ink)
    axes[2].set_ylabel("log-RMSE (log scale)", fontsize=13, color=muted)
    axes[2].set_title(
        "The flexibility ladder",
        fontsize=15,
        color=ink,
    )
    axes[2].legend(frameon=False, fontsize=13, loc="upper left", labelcolor=muted)

    figure.tight_layout()
    path = RESULTS_DIR / "extrapolation.png"
    save_figure(figure, path, facecolor="#fcfcfb")
    plt.close(figure)
    return path


def main() -> None:
    dataset = hdb5.prepare_dataset()
    analysis = analyze_extrapolation(dataset, dataset_path=hdb5.default_hdb5_path())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_dataframe_csv_atomic(
        RESULTS_DIR / "extrapolation_per_machine.csv", analysis.per_machine
    )
    summary = pd.DataFrame([transfer.to_json() for transfer in analysis.transfers])
    intervals = {interval.model_name: interval for interval in analysis.bootstrap}
    summary["lomo_ci_low"] = summary["model_name"].map(lambda n: intervals[n].ci_low)
    summary["lomo_ci_high"] = summary["model_name"].map(lambda n: intervals[n].ci_high)
    write_dataframe_csv_atomic(RESULTS_DIR / "extrapolation_summary.csv", summary)
    write_json_strict(RESULTS_DIR / "extrapolation.json", analysis.to_json())
    figure_path = plot_extrapolation(analysis)

    print("--- Result 4: interpolation against extrapolation ---")
    print(
        f"{analysis.n_rows} rows, holding out {analysis.n_machines_held_out} machines one at a "
        f"time ({analysis.n_machines_excluded} more too small to score)"
    )
    print(f"both splits on the same {len(analysis.feature_columns)} blind features\n")
    header = f"  {'model':<24}{'CV':>8}{'LOMO':>8}{'ratio':>8}{'CV rank':>9}{'LOMO rank':>11}{'rho(dist)':>11}"
    print(header)
    for transfer in analysis.transfers:
        marker = " " if transfer.is_blind else "*"
        print(
            f"{marker} {transfer.model_name:<24}{transfer.cv_rmsle:>8.3f}"
            f"{transfer.lomo_mean_rmsle:>8.3f}{transfer.degradation_factor:>8.2f}"
            f"{transfer.cv_rank:>9}{transfer.lomo_rank:>11}{transfer.distance_spearman:>11.2f}"
        )
    print("  * fitted on this database, held-out machine included; not a blind baseline")
    non_contenders = set(CONTROL_MODELS) | set(LADDER_MODELS)
    controls = [t for t in analysis.transfers if t.model_name in non_contenders]
    if controls:
        print("  control models are scored above but excluded from the ranking claim:")
        for transfer in controls:
            print(f"    {transfer.model_name} (flexible, but still extrapolates)")
    intervals = {interval.model_name: interval for interval in analysis.bootstrap}
    print("\n--- mean LOMO log-RMSE, 95% percentile interval over the 13 machines ---")
    for transfer in analysis.transfers:
        interval = intervals[transfer.model_name]
        print(
            f"  {transfer.model_name:<24}{interval.mean_rmsle:>7.3f}"
            f"   [{interval.ci_low:.3f}, {interval.ci_high:.3f}]"
        )
    print("  (13 machines is a small sample; these intervals are wide on purpose)")

    if analysis.paired_differences:
        print("\n--- gaps, paired by machine (the marginal intervals above overlap) ---")
        for gap in analysis.paired_differences:
            flag = "excludes 0" if gap.excludes_zero else "includes 0"
            print(
                f"  {gap.model_a} - {gap.model_b}: {gap.mean_difference:+.3f} "
                f"[{gap.ci_low:+.3f}, {gap.ci_high:+.3f}]  {flag}; "
                f"worse on {gap.n_machines_a_worse}/{gap.n_machines} machines"
            )

    ladder = [(name, label) for name, label in FLEXIBILITY_LADDER if name in intervals]
    if len(ladder) > 2:
        print("\n--- Result 4d: degradation against model flexibility ---")
        by_name = {transfer.model_name: transfer for transfer in analysis.transfers}
        print(f"  {'form':<28}{'LOMO':>7}{'ratio':>8}{'rho(dist)':>11}{'worst':>8}")
        for name, label in ladder:
            transfer = by_name[name]
            print(
                f"  {label:<28}{transfer.lomo_mean_rmsle:>7.3f}"
                f"{transfer.degradation_factor:>8.2f}{transfer.distance_spearman:>11.2f}"
                f"{transfer.lomo_worst_rmsle:>8.3f}"
            )

    verdict = "exactly reversed" if analysis.ranking_exactly_reversed else "not exactly reversed"
    print(
        f"\nordering over the {analysis.n_contenders} contender models is {verdict} "
        f"between the two splits (rho = {analysis.ranking_spearman:+.2f})"
    )

    if analysis.truncation:
        print("\n--- machines a tree ensemble structurally cannot reach ---")
        for finding in analysis.truncation:
            print(
                f"  {finding.tokamak}: {finding.fraction_above_train_max * 100:.0f}% of rows above "
                f"max(y_train), best shot {finding.headroom_ratio:.1f}x above anything a tree can "
                f"output (random forest {finding.tree_rmsle:.3f}, power law {finding.power_law_rmsle:.3f})"
            )
    if figure_path:
        print(f"\nfigure: {figure_path}")


if __name__ == "__main__":
    main()
