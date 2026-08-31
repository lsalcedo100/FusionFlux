"""What happens to a calibrated interval on a machine nobody has built. Result 7.

Run ``python3 analysis_conformal.py`` to regenerate everything under
``results/`` for Result 7: coverage under all three splits, the per-machine
collapse against extrapolation distance, the interval widths that go with it,
and the figure.

Every number in Results 1 to 6 is a point error. For a next-step device the
point error is not the deliverable. Nobody sizes a machine off a single
predicted confinement time; the question put to a model is "what range should we
plan for", and the answer is an interval. Result 4 establishes that the model is
wrong on a new machine. The question that decides whether the interval is usable
is whether it is *confidently* wrong.

Split conformal prediction is the right tool because it assumes almost nothing
about the model: fit on part of the data, take the absolute log residuals on a
held-back calibration part, and use their (1-alpha) quantile as a half-width.
Under exchangeability of calibration and test rows the interval covers at least
1-alpha of test rows in finite samples, for any model at all.

That proviso is the whole result:

    Result 7a  Grouped CV by discharge. Calibration and test rows are held-out
               discharges from the same machines, exchangeability holds, and
               coverage should land on the nominal level. This is the control:
               it shows the intervals are built correctly, so a shortfall later
               cannot be blamed on the construction.
    Result 7b  Leave one tokamak out. Calibration rows come from machines the
               model trained on and test rows come from one it has never seen.
               Nothing guarantees anything, and the size of the shortfall is a
               measurement of how far the distribution moved.
    Result 7c  Coverage against Result 4b's Mahalanobis distance, per machine.
    Result 7d  The ITER-matched size cut of Result 5, which is the split a
               next-step device actually faces.

None of this is a defect in conformal prediction and none of it is fixed by
calibrating more carefully. It is an assumption being false, measured. Interval
*width* is reported beside coverage everywhere, because coverage on its own is
trivial to win: an interval wide enough to be useless covers everything.

The narrative built on these numbers is in ``results/RESULTS.md``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import hdb5
from analysis_extrapolation import spearman
from storage import write_dataframe_csv_atomic, write_json_strict

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# The models carried through the narrative, in the order they appear in it.
# The hybrid selected by Result 6 is included so that the model that repaired
# the point error can be checked on the interval as well; repairing one does not
# imply repairing the other, and it is worth knowing which happened.
REPORTED_MODELS = (
    "ipb98y2_analytic",
    "ridge_loglinear",
    "hybrid_gbm_s1",
    "hist_gradient_boosting",
    "random_forest",
)

MODEL_LABELS = {
    "ipb98y2_analytic": "IPB98(y,2), analytic",
    "ridge_loglinear": "ridge, log-linear",
    "hybrid_gbm_s1": "hybrid (Result 6)",
    "hist_gradient_boosting": "hist gradient boosting",
    "random_forest": "random forest",
}

# The hybrid Result 6's cross-validation selects. Named here rather than
# recomputed so the two analyses cannot silently drift apart.
HYBRID_CORRECTION = "gbm"
HYBRID_SHRINKAGE = 1.0

SPLIT_LABELS = {
    "grouped_cv": "grouped CV, by discharge",
    "leave_one_tokamak_out": "leave one tokamak out",
    "size_cut": "ITER-matched size cut",
}


@dataclass(frozen=True)
class CoverageCollapse:
    """One model's coverage under both splits, and the gap between them.

    The pooled CV number is the control and the pooled LOMO number is the
    result. ``coverage_shortfall`` is the drop in percentage points, which is
    the quantity worth quoting: a model whose 90% intervals cover 55% of the
    rows on a new machine is not slightly miscalibrated, it is reporting a
    confidence it does not have.
    """

    model_name: str
    is_blind: bool
    nominal_coverage: float
    cv_coverage: float
    lomo_coverage: float
    size_cut_coverage: float
    coverage_shortfall: float
    # Median conformal half-width in log units under each split, and the
    # multiplicative factor it corresponds to. The widths barely move between
    # splits (they are set by the calibration rows, which are drawn the same way
    # in both), which is exactly why coverage falls instead of width rising.
    cv_interval_factor: float
    lomo_interval_factor: float
    size_cut_interval_factor: float
    # Spearman correlation of per-machine LOMO coverage with Mahalanobis
    # distance. Negative means coverage degrades as the machine gets further
    # from the training data, which is Result 4b's finding restated on
    # intervals rather than on point errors.
    distance_spearman: float

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ConformalAnalysis:
    """Everything Result 7 rests on."""

    n_rows: int
    n_machines: int
    nominal_coverage: float
    calibration_fraction: float
    feature_columns: tuple[str, ...]
    size_cut_size_ratio: float
    collapse: list[CoverageCollapse]
    # Worst per-machine coverage seen under LOMO, over the reported models.
    worst_machine_coverage: float
    worst_machine: str
    worst_machine_model: str
    per_machine: pd.DataFrame = field(repr=False)
    summary: pd.DataFrame = field(repr=False)

    def to_json(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_machines": self.n_machines,
            "nominal_coverage": self.nominal_coverage,
            "calibration_fraction": self.calibration_fraction,
            "feature_columns": list(self.feature_columns),
            "size_cut_size_ratio": self.size_cut_size_ratio,
            "collapse": [row.to_json() for row in self.collapse],
            "worst_machine_coverage": self.worst_machine_coverage,
            "worst_machine": self.worst_machine,
            "worst_machine_model": self.worst_machine_model,
            "provenance": hdb5.dataset_provenance(),
        }


def analyze_conformal(
    dataset: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] = hdb5.BLIND_FEATURE_COLUMNS,
    alpha: float = hdb5.DEFAULT_CONFORMAL_ALPHA,
    calibration_fraction: float = hdb5.DEFAULT_CALIBRATION_FRACTION,
) -> ConformalAnalysis:
    """Score interval coverage for every model under all three splits."""
    hybrid = hdb5.build_hybrid_models(
        (HYBRID_SHRINKAGE,), corrections=(HYBRID_CORRECTION,)
    )
    # Spelled out at each call rather than splatted from one dict: the three
    # arms must agree on the feature set, the level and the calibration split,
    # and a shared **kwargs hides a mismatch instead of preventing one.
    _, cv_summary = hdb5.conformal_coverage_grouped_cv(
        dataset,
        feature_columns=feature_columns,
        alpha=alpha,
        calibration_fraction=calibration_fraction,
        extra_models=hybrid,
    )
    _, lomo_summary = hdb5.conformal_coverage_leave_one_tokamak_out(
        dataset,
        feature_columns=feature_columns,
        alpha=alpha,
        calibration_fraction=calibration_fraction,
        extra_models=hybrid,
    )

    splits = hdb5.size_ordered_splits(dataset)
    iter_split = hdb5.iter_matched_split(dataset, splits)
    _, size_summary = hdb5.conformal_coverage_size_split(
        dataset,
        iter_split,
        feature_columns=feature_columns,
        alpha=alpha,
        calibration_fraction=calibration_fraction,
        extra_models=hybrid,
    )

    summary = pd.concat([cv_summary, lomo_summary, size_summary], ignore_index=True)
    pooled = summary[summary["scope"] == "__pooled__"]
    per_machine = summary[summary["scope"] != "__pooled__"].copy()

    # Attach the same distance Result 4b ordered its table by, so coverage and
    # point error are read against one common x axis rather than two.
    distances = {
        machine: hdb5.extrapolation_diagnostic(
            dataset, machine, feature_columns=feature_columns
        ).feature_mahalanobis
        for machine in hdb5.eligible_tokamaks(dataset)
    }
    per_machine["feature_mahalanobis"] = per_machine["scope"].map(distances)

    def _pooled(model_name: str, split: str, column: str) -> float:
        rows = pooled[
            (pooled["model_name"] == model_name) & (pooled["split"] == split)
        ]
        return float(rows[column].iloc[0]) if len(rows) else float("nan")

    collapse: list[CoverageCollapse] = []
    for model_name in REPORTED_MODELS:
        if not (pooled["model_name"] == model_name).any():
            continue
        cv = _pooled(model_name, "grouped_cv", "empirical_coverage")
        lomo = _pooled(model_name, "leave_one_tokamak_out", "empirical_coverage")
        machine_rows = per_machine[
            (per_machine["model_name"] == model_name)
            & (per_machine["split"] == "leave_one_tokamak_out")
            & per_machine["feature_mahalanobis"].notna()
        ]
        collapse.append(
            CoverageCollapse(
                model_name=model_name,
                is_blind=bool(
                    pooled[pooled["model_name"] == model_name]["is_blind"].iloc[0]
                ),
                nominal_coverage=1.0 - alpha,
                cv_coverage=cv,
                lomo_coverage=lomo,
                size_cut_coverage=_pooled(model_name, "size_cut", "empirical_coverage"),
                coverage_shortfall=cv - lomo,
                cv_interval_factor=_pooled(
                    model_name, "grouped_cv", "median_interval_factor"
                ),
                lomo_interval_factor=_pooled(
                    model_name, "leave_one_tokamak_out", "median_interval_factor"
                ),
                size_cut_interval_factor=_pooled(
                    model_name, "size_cut", "median_interval_factor"
                ),
                distance_spearman=spearman(
                    machine_rows["feature_mahalanobis"].to_numpy(dtype=float),
                    machine_rows["empirical_coverage"].to_numpy(dtype=float),
                ),
            )
        )

    reported_lomo = per_machine[
        (per_machine["split"] == "leave_one_tokamak_out")
        & per_machine["model_name"].isin(REPORTED_MODELS)
    ]
    worst = reported_lomo.loc[reported_lomo["empirical_coverage"].idxmin()]

    return ConformalAnalysis(
        n_rows=int(len(dataset)),
        n_machines=int(per_machine["scope"].nunique()),
        nominal_coverage=1.0 - alpha,
        calibration_fraction=calibration_fraction,
        feature_columns=tuple(feature_columns),
        size_cut_size_ratio=float(iter_split.size_ratio),
        collapse=collapse,
        worst_machine_coverage=float(worst["empirical_coverage"]),
        worst_machine=str(worst["scope"]),
        worst_machine_model=str(worst["model_name"]),
        per_machine=per_machine,
        summary=summary,
    )


# --- Figure -----------------------------------------------------------------

# Three splits, three hues, validated as an adjacent-separable set under
# simulated colour-vision deficiency.
CV_HUE, LOMO_HUE, SIZE_HUE = "#2a78d6", "#eb6834", "#7d5bbe"
INK, MUTED = "#0b0b0b", "#52514e"
SPLIT_HUES = {
    "grouped_cv": CV_HUE,
    "leave_one_tokamak_out": LOMO_HUE,
    "size_cut": SIZE_HUE,
}


def plot_conformal(analysis: ConformalAnalysis) -> Path | None:
    """Two panels: the collapse per model, and the collapse against distance.

    Left is Result 7a and 7b: every model's coverage under each split against
    the nominal line, which is the claim. Right is Result 7c: per-machine
    coverage against how far the machine sits outside the training data, where
    the in-distribution arm stays flat on the nominal line and the held-out arm
    falls away from it.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - plotting is optional
        return None

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        1, 2, figsize=(15.0, 5.6), gridspec_kw={"width_ratios": [1.0, 1.15]}
    )
    for axis in axes:
        axis.grid(alpha=0.25, linewidth=0.6)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(MUTED)
            axis.spines[side].set_linewidth(0.8)
        axis.tick_params(colors=MUTED, labelsize=9)

    nominal = analysis.nominal_coverage

    # --- Left: coverage per model, one group of bars per model -------------
    models = [row for row in analysis.collapse]
    positions = np.arange(len(models), dtype=float)
    width = 0.26
    series = (
        ("grouped_cv", "cv_coverage", -width),
        ("leave_one_tokamak_out", "lomo_coverage", 0.0),
        ("size_cut", "size_cut_coverage", width),
    )
    for split, attribute, offset in series:
        values = [getattr(row, attribute) for row in models]
        axes[0].bar(
            positions + offset,
            values,
            width * 0.92,
            color=SPLIT_HUES[split],
            label=SPLIT_LABELS[split],
            zorder=3,
        )
        # Direct labels: with three bars per group a legend alone would make the
        # reader count positions to recover a number that is the whole point.
        for position, value in zip(positions + offset, values):
            axes[0].annotate(
                f"{value * 100:.0f}",
                xy=(position, value),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                color=MUTED,
            )
    axes[0].axhline(nominal, color=INK, linewidth=1.2, linestyle="--", zorder=4)
    axes[0].annotate(
        f"nominal {nominal * 100:.0f}%",
        xy=(len(models) - 0.5, nominal),
        xytext=(0, 4),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=INK,
    )
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(
        [MODEL_LABELS.get(row.model_name, row.model_name) for row in models],
        rotation=18,
        ha="right",
        fontsize=8.5,
    )
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel(f"empirical coverage of {nominal * 100:.0f}% intervals", fontsize=10, color=INK)
    axes[0].set_title(
        "Result 7a/b: the interval is calibrated only on the split it was calibrated on",
        fontsize=10.5,
        color=INK,
        loc="left",
    )
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower left")

    # --- Right: coverage against extrapolation distance --------------------
    #
    # Three models under the held-out-machine split only. The pooled CV numbers
    # in the left panel are the control for this panel; drawing the per-machine
    # CV curves here as well put two nearly coincident lines on top of the
    # result and made the panel harder to read, not more complete.
    distance_models = (
        ("ridge_loglinear", CV_HUE, "o-"),
        ("hist_gradient_boosting", LOMO_HUE, "^--"),
        ("random_forest", SIZE_HUE, "s--"),
    )
    rho_by_model = {row.model_name: row.distance_spearman for row in analysis.collapse}
    ordered_machines: pd.DataFrame | None = None
    for model_name, hue, marker in distance_models:
        rows = analysis.per_machine[
            (analysis.per_machine["split"] == "leave_one_tokamak_out")
            & (analysis.per_machine["model_name"] == model_name)
            & analysis.per_machine["feature_mahalanobis"].notna()
        ].sort_values("feature_mahalanobis")
        if rows.empty:
            continue
        ordered_machines = rows
        rho = rho_by_model.get(model_name, float("nan"))
        axes[1].plot(
            rows["feature_mahalanobis"].to_numpy(dtype=float),
            rows["empirical_coverage"].to_numpy(dtype=float),
            marker,
            color=hue,
            linewidth=1.8,
            markersize=6.5,
            label=f"{MODEL_LABELS.get(model_name, model_name)}   $\\rho$ = {rho:+.2f}",
            zorder=3,
        )

    # Machine names once, along the bottom, rather than attached to one series:
    # the x position is shared by all three, so hanging them off a single curve
    # would imply they belong to it.
    if ordered_machines is not None:
        # Several machines sit within 0.05 distance units of each other (AUGW
        # and JETILW, MAST and JFT2M), which is closer than a rotated label is
        # wide. Those drop to a second row rather than being written on top of
        # their neighbour.
        span = float(
            ordered_machines["feature_mahalanobis"].max()
            - ordered_machines["feature_mahalanobis"].min()
        )
        crowded_within = 0.03 * span if span > 0 else 0.0
        previous_distance = -np.inf
        previous_row = 1
        for _, row in ordered_machines.iterrows():
            distance = float(row["feature_mahalanobis"])
            if distance - previous_distance < crowded_within and previous_row == 0:
                label_row = 1
            elif distance - previous_distance < crowded_within:
                label_row = 0
            else:
                label_row = 1
            previous_distance, previous_row = distance, label_row
            axes[1].axvline(
                distance, color=MUTED, linewidth=0.5, alpha=0.25, zorder=1
            )
            axes[1].annotate(
                str(row["scope"]),
                xy=(distance, -0.02 if label_row else -0.13),
                rotation=90,
                ha="center",
                va="top",
                fontsize=7,
                color=MUTED,
            )

    axes[1].axhline(nominal, color=INK, linewidth=1.2, linestyle="--", zorder=4)
    axes[1].annotate(
        f"nominal {nominal * 100:.0f}%",
        xy=(0.99, nominal),
        xycoords=("axes fraction", "data"),
        xytext=(0, 5),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=INK,
    )
    # A gutter below zero holds the machine names. Three of the machines sit
    # within 0.25 distance units of each other, so labels placed among the marks
    # collide with them and with each other whatever the rotation.
    axes[1].set_ylim(-0.34, 1.08)
    axes[1].set_yticks(np.arange(0.0, 1.01, 0.2))
    axes[1].axhline(0.0, color=MUTED, linewidth=0.8, zorder=2)
    axes[1].set_xlabel(
        "Mahalanobis distance of the machine from the training data", fontsize=10, color=INK
    )
    axes[1].set_ylabel(
        f"empirical coverage of {nominal * 100:.0f}% intervals", fontsize=10, color=INK
    )
    axes[1].set_title(
        "Result 7c: the trees' coverage tracks distance, the power law's does not",
        fontsize=10.5,
        color=INK,
        loc="left",
    )
    # Centre right is the one region all three curves leave empty.
    axes[1].legend(frameon=False, fontsize=8.5, loc="center right")

    figure.tight_layout()
    path = RESULTS_DIR / "conformal.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path


def main() -> None:
    dataset = hdb5.prepare_dataset()
    analysis = analyze_conformal(dataset)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_dataframe_csv_atomic(
        RESULTS_DIR / "conformal_summary.csv",
        pd.DataFrame([row.to_json() for row in analysis.collapse]),
    )
    write_dataframe_csv_atomic(
        RESULTS_DIR / "conformal_per_machine.csv", analysis.per_machine
    )
    write_json_strict(RESULTS_DIR / "conformal.json", analysis.to_json())
    figure_path = plot_conformal(analysis)

    nominal = analysis.nominal_coverage
    print("--- Result 7: calibrated intervals, and where the calibration goes ---")
    print(
        f"{analysis.n_rows} rows, split conformal on log residuals at nominal "
        f"{nominal * 100:.0f}%"
    )
    print(
        "  coverage is pooled over rows; the per-machine breakdown covers the 13 machines\n"
        "  large enough to hold out, the same 13 as Result 4"
    )
    print(
        f"calibration is {analysis.calibration_fraction:.0%} of *discharges*, held back "
        "from the fit in every arm\n"
    )

    print(
        f"  {'model':<24}{'CV':>7}{'LOMO':>7}{'size cut':>10}"
        f"{'drop':>7}{'width x':>9}{'rho(dist)':>11}"
    )
    for row in analysis.collapse:
        marker = " " if row.is_blind else "*"
        print(
            f"{marker} {MODEL_LABELS.get(row.model_name, row.model_name):<23}"
            f"{row.cv_coverage * 100:>6.0f}%{row.lomo_coverage * 100:>6.0f}%"
            f"{row.size_cut_coverage * 100:>9.0f}%"
            f"{row.coverage_shortfall * 100:>6.0f}%"
            f"{row.lomo_interval_factor:>9.2f}{row.distance_spearman:>11.2f}"
        )
    print("  * fitted on this database, held-out machine included; not a blind baseline")
    print(
        "  'width x' is the multiplicative half-width under LOMO: the interval runs from\n"
        "  prediction / x to prediction * x, so it is a statement about size as well as coverage"
    )

    print(f"\n--- worst single machine, {nominal * 100:.0f}% nominal ---")
    print(
        f"  {MODEL_LABELS.get(analysis.worst_machine_model, analysis.worst_machine_model)} "
        f"on {analysis.worst_machine}: {analysis.worst_machine_coverage * 100:.0f}% covered"
    )

    print("\n--- per machine, ridge, leave one tokamak out ---")
    rows = analysis.per_machine[
        (analysis.per_machine["split"] == "leave_one_tokamak_out")
        & (analysis.per_machine["model_name"] == "ridge_loglinear")
    ].sort_values("feature_mahalanobis")
    print(f"  {'machine':<10}{'distance':>10}{'coverage':>10}{'width x':>9}")
    for _, machine_row in rows.iterrows():
        print(
            f"  {str(machine_row['scope']):<10}{machine_row['feature_mahalanobis']:>10.1f}"
            f"{machine_row['empirical_coverage'] * 100:>9.0f}%"
            f"{machine_row['median_interval_factor']:>9.2f}"
        )

    if figure_path:
        print(f"\nfigure: {figure_path}")


if __name__ == "__main__":
    main()
