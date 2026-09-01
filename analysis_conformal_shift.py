"""Repairing the interval collapse, and finding the limit of the repair. Result 10.

Run ``python3 analysis_conformal_shift.py`` to regenerate everything under
``results/`` for Result 10.

Result 7 ends on a diagnosis: nominal 90% intervals cover 90% under grouped CV,
35% on an unseen machine and 3% across the ITER-matched size cut, at essentially
unchanged width. The reading offered there is that this is not a defect in
conformal prediction but an assumption being false, measured. That reading makes
a prediction, and this script is the test of it.

If the failure really is exchangeability, then calibrating on the right unit
should fix it, and should fix it *exactly as far as that unit extends*:

    On leave-one-machine-out, calibrating on held-out **machines** rather than
    held-out discharges should largely restore coverage. Calibration machines
    and the test machine are all drawn from the same database.

    On the ITER-matched cut it should not, however carefully it is done, because
    every calibration machine is smaller than every test machine and no
    recalibration makes those exchangeable. Only scaling the interval by
    extrapolation distance can help there.

A confirmed prediction of a *limit* is worth more than a repair that works
everywhere, because a repair that worked everywhere would suggest the original
diagnosis was wrong about the cause.

The three schemes are defined in ``conformal_shift.py``. Width travels beside
every coverage number, because a scheme that buys coverage by inflating the
interval has not repaired anything: at the extreme, an infinite interval covers
everything and says nothing.

The narrative built on these numbers is in ``results/RESULTS.md``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import conformal_shift as cshift
import dimensional as dm
import hdb5
from storage import write_dataframe_csv_atomic, write_json_strict

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Deliberately small. Every model here costs one fit per training machine per
# fold to calibrate, so the zoo is the four models the narrative actually
# compares: the two tree families whose intervals collapse, the log-linear law
# whose intervals mostly hold, and the constrained law from Result 8.
MODEL_LABELS: dict[str, str] = {
    "ipb98y2_analytic": "IPB98(y,2), analytic",
    "ridge_loglinear": "ridge, log-linear",
    "powerlaw_collisionless": "power law, collisionless",
    "random_forest": "random forest",
    "hist_gradient_boosting": "hist gradient boosting",
}

METHOD_LABELS: dict[str, str] = {
    "split": "split (Result 7)",
    "machine_cv": "machine-CV",
    "machine_cv_distance": "machine-CV + distance",
}


def build_zoo() -> dict[str, Any]:
    zoo = hdb5.build_model_zoo()
    del zoo["mean_baseline"]
    zoo.update(dm.build_constrained_models(("collisionless",)))
    return zoo


@dataclass(frozen=True)
class CoverageRepair:
    """One model under one scheme: coverage and width on both shifted arms."""

    model_name: str
    method: str
    is_blind: bool
    lomo_coverage: float
    lomo_interval_factor: float
    size_cut_coverage: float
    size_cut_interval_factor: float
    # Coverage minus nominal. Negative is undercoverage, which is the failure
    # mode; a positive number means the interval is wider than it needs to be.
    lomo_shortfall: float
    size_cut_shortfall: float

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ConformalShiftAnalysis:
    n_rows: int
    nominal_coverage: float
    n_calibration_machines: int
    iter_matched_size_ratio: float
    repairs: list[CoverageRepair]
    per_machine: pd.DataFrame = field(repr=False)
    # Fitted growth rate of the interval with extrapolation distance, per
    # (scheme, model). Result 7's complaint was that the width does not move at
    # all, so this is the property being repaired, and its sign is a finding.
    width_distance_growth: dict[str, float] = field(default_factory=dict)
    lomo_best_method: str = ""
    size_cut_best_method: str = ""

    def to_json(self) -> dict[str, object]:
        return {
            "n_rows": self.n_rows,
            "nominal_coverage": self.nominal_coverage,
            "n_calibration_machines": self.n_calibration_machines,
            "iter_matched_size_ratio": self.iter_matched_size_ratio,
            "repairs": [row.to_json() for row in self.repairs],
            "width_distance_growth": self.width_distance_growth,
            "lomo_best_method": self.lomo_best_method,
            "size_cut_best_method": self.size_cut_best_method,
        }


def analyze_conformal_shift(
    dataset: pd.DataFrame,
    *,
    alpha: float = hdb5.DEFAULT_CONFORMAL_ALPHA,
) -> ConformalShiftAnalysis:
    zoo = build_zoo()
    splits = hdb5.size_ordered_splits(dataset)
    cut = hdb5.iter_matched_split(dataset, splits)

    lomo_rows, lomo_summary = cshift.coverage_leave_one_tokamak_out(dataset, zoo, alpha=alpha)
    size_rows, size_summary = cshift.coverage_size_split(dataset, cut, zoo, alpha=alpha)

    lomo_pooled = lomo_summary[lomo_summary["scope"] == "__pooled__"].set_index(
        ["model_name", "method"]
    )
    size_pooled = size_summary[size_summary["scope"] == "__pooled__"].set_index(
        ["model_name", "method"]
    )

    nominal = 1.0 - alpha
    repairs: list[CoverageRepair] = []
    for (model_name, method), row in lomo_pooled.iterrows():
        key = (model_name, method)
        if key not in size_pooled.index:
            continue
        size_row = size_pooled.loc[key]
        repairs.append(
            CoverageRepair(
                model_name=str(model_name),
                method=str(method),
                is_blind=bool(row["is_blind"]),
                lomo_coverage=float(row["empirical_coverage"]),
                lomo_interval_factor=float(row["median_interval_factor"]),
                size_cut_coverage=float(size_row["empirical_coverage"]),
                size_cut_interval_factor=float(size_row["median_interval_factor"]),
                lomo_shortfall=float(row["empirical_coverage"]) - nominal,
                size_cut_shortfall=float(size_row["empirical_coverage"]) - nominal,
            )
        )

    # Result 7's specific complaint: the width does not move with distance.
    #
    # Reported as the fitted growth rate of the interval with distance, per
    # model, rather than as a rank correlation. Within a model the half-width is
    # a monotone function of the distance by construction, so a Spearman
    # correlation there is pinned to exactly +/-1 and carries only this number's
    # sign. The slope is the quantity with content: it is the exponent in
    # ``sigma(d) = exp(c0 + c1 d)``, so a slope of 0.19 means the interval grows
    # by a factor of e^0.19 per unit of Mahalanobis distance, and a negative one
    # means the calibration residuals said that model's error does not grow with
    # distance at all.
    width_growth: dict[str, float] = {}
    for (method, model), subset in size_rows.groupby(["method", "model_name"], sort=False):
        width_growth[f"{method}::{model}"] = float(subset["distance_scale_slope"].iloc[0])

    # Which scheme lands closest to nominal on each arm, over the blind models.
    def _best(getter: str) -> str:
        blind = [row for row in repairs if row.is_blind]
        by_method: dict[str, list[float]] = {}
        for row in blind:
            by_method.setdefault(row.method, []).append(abs(getattr(row, getter)))
        return min(by_method, key=lambda method: float(np.mean(by_method[method])))

    per_machine = pd.concat(
        [
            lomo_summary[lomo_summary["scope"] != "__pooled__"].assign(arm="lomo"),
            size_summary[size_summary["scope"] != "__pooled__"].assign(arm="size_cut"),
        ],
        ignore_index=True,
    )

    return ConformalShiftAnalysis(
        n_rows=int(len(dataset)),
        nominal_coverage=nominal,
        n_calibration_machines=len(hdb5.eligible_tokamaks(dataset)),
        iter_matched_size_ratio=float(cut.size_ratio),
        repairs=repairs,
        per_machine=per_machine,
        width_distance_growth=width_growth,
        lomo_best_method=_best("lomo_shortfall"),
        size_cut_best_method=_best("size_cut_shortfall"),
    )


def plot_conformal_shift(analysis: ConformalShiftAnalysis) -> Path | None:
    """Coverage against nominal, per scheme, on both shifted arms."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - figure is optional
        return None

    frame = pd.DataFrame([row.to_json() for row in analysis.repairs])
    models = [name for name in MODEL_LABELS if name in set(frame["model_name"])]
    methods = list(cshift.CALIBRATION_METHODS)
    colours = {"split": "#888888", "machine_cv": "#2a78d6", "machine_cv_distance": "#1a9850"}

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True)
    positions = np.arange(len(models))
    width = 0.26

    for axis, (column, title) in zip(
        axes,
        (
            ("lomo_coverage", "held-out machine"),
            ("size_cut_coverage", f"ITER-matched cut ({analysis.iter_matched_size_ratio:.2f}x)"),
        ),
        strict=True,
    ):
        for offset, method in enumerate(methods):
            subset = frame[frame["method"] == method].set_index("model_name")
            values = [
                float(subset.loc[model, column]) if model in subset.index else np.nan
                for model in models
            ]
            axis.bar(
                positions + (offset - 1) * width,
                values,
                width,
                color=colours[method],
                label=METHOD_LABELS[method],
            )
        axis.axhline(
            analysis.nominal_coverage, color="#c0392b", linestyle="--", linewidth=1.3
        )
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [MODEL_LABELS[model] for model in models], rotation=28, ha="right", fontsize=9
        )
        axis.set_title(title)
        axis.set_ylim(0.0, 1.02)
        axis.grid(alpha=0.15, axis="y")

    axes[0].set_ylabel(f"empirical coverage (nominal {analysis.nominal_coverage:.0%})")
    axes[0].legend(fontsize=9, frameon=False, loc="lower left")
    figure.suptitle(
        "Calibrating on the right unit repairs the arm it can reach, and not the other",
        fontsize=12,
    )
    figure.tight_layout()
    path = RESULTS_DIR / "conformal_shift.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path


def main() -> None:
    dataset = hdb5.prepare_dataset()
    analysis = analyze_conformal_shift(dataset)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_dataframe_csv_atomic(
        RESULTS_DIR / "conformal_shift_summary.csv",
        pd.DataFrame([row.to_json() for row in analysis.repairs]),
    )
    write_dataframe_csv_atomic(
        RESULTS_DIR / "conformal_shift_per_machine.csv", analysis.per_machine
    )
    write_json_strict(RESULTS_DIR / "conformal_shift.json", analysis.to_json())
    figure_path = plot_conformal_shift(analysis)

    nominal = analysis.nominal_coverage
    print(f"--- Result 10: repairing the interval collapse, nominal {nominal:.0%} ---")
    print(
        f"  calibration draws on {analysis.n_calibration_machines} training machines; "
        "width is the multiplicative half-width, so the\n"
        "  interval runs from prediction / x to prediction * x"
    )

    frame = pd.DataFrame([row.to_json() for row in analysis.repairs])
    for arm, coverage_column, width_column in (
        ("held-out machine", "lomo_coverage", "lomo_interval_factor"),
        ("ITER-matched size cut", "size_cut_coverage", "size_cut_interval_factor"),
    ):
        print(f"\n  {arm}")
        print(f"  {'model':<28}", end="")
        for method in cshift.CALIBRATION_METHODS:
            print(f"{METHOD_LABELS[method]:>24}", end="")
        print()
        for model in MODEL_LABELS:
            subset = frame[frame["model_name"] == model].set_index("method")
            if subset.empty:
                continue
            marker = " " if bool(subset["is_blind"].iloc[0]) else "*"
            print(f"{marker} {MODEL_LABELS[model]:<27}", end="")
            for method in cshift.CALIBRATION_METHODS:
                if method not in subset.index:
                    print(f"{'-':>24}", end="")
                    continue
                coverage = float(subset.loc[method, coverage_column])
                factor = float(subset.loc[method, width_column])
                print(f"{coverage * 100:>17.0f}% x{factor:<5.2f}", end="")
            print()
    print("  * fitted on this database, held-out machines included; not a blind baseline")

    print("\n--- how fast does the interval grow with distance? ---")
    print(
        "  fitted slope of log(half-width) against Mahalanobis distance. 0 is a\n"
        "  constant-width interval, which is what Result 7 found and objected to"
    )
    print(f"  {'model':<28}", end="")
    for method in cshift.CALIBRATION_METHODS:
        print(f"{METHOD_LABELS[method]:>24}", end="")
    print()
    for model in MODEL_LABELS:
        row = [
            analysis.width_distance_growth.get(f"{method}::{model}")
            for method in cshift.CALIBRATION_METHODS
        ]
        if all(value is None for value in row):
            continue
        print(f"  {MODEL_LABELS[model]:<28}", end="")
        for value in row:
            print(f"{'-':>24}" if value is None else f"{value:>+24.3f}", end="")
        print()
    print(
        "  positive means this model's calibration residuals grow with distance, so its\n"
        "  interval is told to widen out of distribution; negative means they do not, and\n"
        "  that split is itself Result 4b's finding recovered from a different quantity"
    )

    print(
        f"\n  closest to nominal on a held-out machine: "
        f"{METHOD_LABELS.get(analysis.lomo_best_method, analysis.lomo_best_method)}"
    )
    print(
        f"  closest to nominal at the ITER-matched cut: "
        f"{METHOD_LABELS.get(analysis.size_cut_best_method, analysis.size_cut_best_method)}"
    )

    if figure_path:
        print(f"\nfigure: {figure_path}")


if __name__ == "__main__":
    main()
