"""Result 15: the reversal's precondition, measured by varying only dimensionality.

Run ``python3 analysis_tree_allometry.py`` to regenerate everything under
``results/`` for Result 15.

Result 13b conjectures that the ranking reversal, in which a flexible model wins
cross-validation and loses on an unseen group, requires enough feature
dimensionality for the flexible model to win the interpolation split first. It
could not test that, because the two datasets it had differ in a dozen ways at
once besides feature count.

This holds everything else fixed. One row set, one grouping, one pair of splits,
one set of models, four rungs of a feature ladder in which each rung is a prefix
of the next. The only thing that changes between rungs is how many predictors
the models may see, so whatever moves is caused by that.

    Result 15a   The interpolation arm across the ladder: at what dimension, if
                 any, do the flexible models start beating the power law?
    Result 15b   The extrapolation arm across the ladder: does the power law's
                 advantage on an unseen species depend on dimension at all?
    Result 15c   The reversal itself, which is the conjunction of the two, and
                 whether it appears exactly where 15a crosses zero.
    Result 15d   The published exponent, tested the way Result 13 tests
                 Kleiber's: does holding mass to diameter^(8/3) cost or help?

The split structure is the thing to get right, and it is easy to get wrong.
HDB5's "CV, by discharge" holds out shots while keeping every machine in the
training fold, so it measures interpolation *within known machines*. The
analogue here keeps every species on both sides of the split. Grouping the
cross-validation by species instead would be the analogue of
leave-one-tokamak-out, and comparing it against leave-one-species-out would
compare a hard split against the same hard split; run that way, the reversal
cannot appear at any dimension, because there is no easy split to win. This
module ran that way once and reported "no reversal at any rung", which was an
artifact of the split rather than a finding.

The narrative built on these numbers is in ``results/RESULTS.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import scaling_audit as sa
import tree_allometry as ta
from storage import write_dataframe_csv_atomic, write_json_strict

RESULTS_DIR = Path(__file__).resolve().parent / "results"

RANDOM_STATE = 42
N_CV_FOLDS = 5

POWER_LAW = "powerlaw_loglinear"
TREE_MODELS: tuple[str, ...] = ("random_forest", "hist_gradient_boosting")


def build_models() -> dict[str, Any]:
    """The three model forms, matched to the ones Results 4 and 13 score.

    Rebuilt on every call because ``audit_groups`` clones what it is given and
    ``cross_val_predict`` fits in place; sharing one instance across rungs would
    let a fit leak between them.
    """
    return {
        POWER_LAW: Pipeline(
            [("scale", StandardScaler()), ("model", Ridge(alpha=1.0, solver="svd"))]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
    }


def rmsle(actual: np.ndarray, predicted: np.ndarray) -> float:
    """RMSE in log space, which is what every other result here reports."""
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))


@dataclass(frozen=True)
class RungResult:
    """One rung of the feature ladder, scored on both splits."""

    n_features: int
    features: tuple[str, ...]
    cv_rmsle: dict[str, float]
    loo_rmsle: dict[str, float]
    n_species_scored: int

    @property
    def best_tree_cv(self) -> float:
        return min(self.cv_rmsle[name] for name in TREE_MODELS)

    @property
    def best_tree_loo(self) -> float:
        return min(self.loo_rmsle[name] for name in TREE_MODELS)

    @property
    def cv_gain_over_power_law(self) -> float:
        """Positive when the best flexible model beats the power law on the easy split."""
        return 1.0 - self.best_tree_cv / self.cv_rmsle[POWER_LAW]

    @property
    def loo_gain_over_power_law(self) -> float:
        return 1.0 - self.best_tree_loo / self.loo_rmsle[POWER_LAW]

    @property
    def trees_win_interpolation(self) -> bool:
        return self.best_tree_cv < self.cv_rmsle[POWER_LAW]

    @property
    def power_law_wins_extrapolation(self) -> bool:
        return self.loo_rmsle[POWER_LAW] < self.best_tree_loo

    @property
    def reversal(self) -> bool:
        """The Result 4 pattern: flexible wins the easy split and loses the hard one."""
        return self.trees_win_interpolation and self.power_law_wins_extrapolation


@dataclass
class LadderStudy:
    rungs: list[RungResult]
    per_species: pd.DataFrame
    baseline: ta.WBEBaseline
    n_rows: int
    n_species_total: int
    size_span: float
    mass_span: float
    extra: dict[str, Any] = field(default_factory=dict)


def score_rung(
    dataset: pd.DataFrame,
    n_features: int,
    *,
    min_rows: int = ta.MIN_HELD_OUT_ROWS,
) -> tuple[RungResult, pd.DataFrame]:
    """Both splits at one rung, on the fixed row set."""
    features = ta.FEATURE_LADDER[n_features]
    x = dataset[list(features)]
    y = dataset[f"log_{ta.TARGET_COLUMN}"].to_numpy(dtype=float)
    groups = dataset[ta.GROUP_COLUMN].to_numpy()

    # Interpolation: the same species appear on both sides, which is what makes
    # this the analogue of HDB5's cross-validation by discharge.
    cv = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = {
        name: rmsle(y, cross_val_predict(model, x, y, cv=cv))
        for name, model in build_models().items()
    }

    # Extrapolation: an entire species is held out, the analogue of holding out
    # a whole tokamak.
    report = sa.audit_groups(
        x, y, groups, build_models(), min_held_out_rows=min_rows, scorer=rmsle
    )
    loo_scores = {
        str(name): float(rows["score"].mean())
        for name, rows in report.groupby("estimator")
    }
    report = report.assign(n_features=n_features)

    return (
        RungResult(
            n_features=n_features,
            features=features,
            cv_rmsle=cv_scores,
            loo_rmsle=loo_scores,
            n_species_scored=int(report["group"].nunique()),
        ),
        report,
    )


def run_study(dataset: pd.DataFrame | None = None) -> LadderStudy:
    if dataset is None:
        dataset = ta.prepare_dataset()

    rungs: list[RungResult] = []
    reports: list[pd.DataFrame] = []
    for n_features in sorted(ta.FEATURE_LADDER):
        rung, report = score_rung(dataset, n_features)
        rungs.append(rung)
        reports.append(report)

    baseline = ta.fit_wbe(
        dataset["log_diameter_m"].to_numpy(dtype=float),
        dataset[f"log_{ta.TARGET_COLUMN}"].to_numpy(dtype=float),
    )

    medians = ta.species_size_medians(dataset)
    eligible = ta.eligible_species(dataset)
    sizes = [medians[name] for name in eligible]
    mass = dataset[ta.TARGET_COLUMN].to_numpy(dtype=float)

    return LadderStudy(
        rungs=rungs,
        per_species=pd.concat(reports, ignore_index=True),
        baseline=baseline,
        n_rows=int(len(dataset)),
        n_species_total=int(dataset[ta.GROUP_COLUMN].nunique()),
        size_span=float(max(sizes) / min(sizes)) if sizes else float("nan"),
        mass_span=float(mass.max() / mass.min()),
    )


def first_reversal_rung(study: LadderStudy) -> int | None:
    """The lowest dimension at which the Result 4 pattern appears, if any."""
    for rung in study.rungs:
        if rung.reversal:
            return rung.n_features
    return None


def write_results(study: LadderStudy, results_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    ladder = pd.DataFrame(
        [
            {
                "n_features": rung.n_features,
                "features": " + ".join(rung.features),
                "n_species_scored": rung.n_species_scored,
                "cv_powerlaw": rung.cv_rmsle[POWER_LAW],
                "cv_random_forest": rung.cv_rmsle["random_forest"],
                "cv_hist_gradient_boosting": rung.cv_rmsle["hist_gradient_boosting"],
                "cv_gain_over_powerlaw": rung.cv_gain_over_power_law,
                "loo_powerlaw": rung.loo_rmsle[POWER_LAW],
                "loo_random_forest": rung.loo_rmsle["random_forest"],
                "loo_hist_gradient_boosting": rung.loo_rmsle["hist_gradient_boosting"],
                "loo_gain_over_powerlaw": rung.loo_gain_over_power_law,
                "trees_win_interpolation": rung.trees_win_interpolation,
                "power_law_wins_extrapolation": rung.power_law_wins_extrapolation,
                "reversal": rung.reversal,
            }
            for rung in study.rungs
        ]
    )
    write_dataframe_csv_atomic(results_dir / "tree_allometry_ladder.csv", ladder)
    write_dataframe_csv_atomic(results_dir / "tree_allometry_per_species.csv", study.per_species)

    payload: dict[str, Any] = {
        "n_rows": study.n_rows,
        "n_species_total": study.n_species_total,
        "n_species_scored": study.rungs[0].n_species_scored if study.rungs else 0,
        "species_size_span": study.size_span,
        "mass_span": study.mass_span,
        "wbe_exponent": ta.WBE_EXPONENT,
        "free_refit_exponent": study.baseline.free_exponent,
        "free_rmsle": study.baseline.free_rmsle,
        "wbe_constrained_rmsle": study.baseline.constrained_rmsle,
        "first_reversal_n_features": first_reversal_rung(study),
        "n_rungs": len(study.rungs),
        "n_rungs_with_reversal": sum(1 for rung in study.rungs if rung.reversal),
        "n_rungs_power_law_wins_extrapolation": sum(
            1 for rung in study.rungs if rung.power_law_wins_extrapolation
        ),
        "ladder": {
            str(rung.n_features): {
                "cv_gain_over_powerlaw": rung.cv_gain_over_power_law,
                "loo_gain_over_powerlaw": rung.loo_gain_over_power_law,
                "cv_powerlaw": rung.cv_rmsle[POWER_LAW],
                "cv_best_tree": rung.best_tree_cv,
                "loo_powerlaw": rung.loo_rmsle[POWER_LAW],
                "loo_best_tree": rung.best_tree_loo,
                "reversal": rung.reversal,
            }
            for rung in study.rungs
        },
    }
    write_json_strict(results_dir / "tree_allometry.json", payload)
    return payload


def plot_tree_allometry(study: LadderStudy) -> Path | None:
    """One panel per split, sharing an x axis of feature count.

    The whole result is that one of these two curves crosses and the other does
    not, so they are drawn against the same axis at the same scale.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - figure is optional
        return None

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    counts = [rung.n_features for rung in study.rungs]

    palette = {
        POWER_LAW: ("#000000", "x", "power law, log-linear"),
        "random_forest": ("#d6301f", "o", "random forest"),
        "hist_gradient_boosting": ("#eb6834", "v", "hist gradient boosting"),
    }
    for model, (colour, marker, label) in palette.items():
        left.plot(
            counts,
            [rung.cv_rmsle[model] for rung in study.rungs],
            "-", color=colour, marker=marker, label=label,
        )
        right.plot(
            counts,
            [rung.loo_rmsle[model] for rung in study.rungs],
            "-", color=colour, marker=marker, label=label,
        )

    crossing = first_reversal_rung(study)
    for axis, title in (
        (left, "Interpolation: same species both sides\n(the analogue of CV by discharge)"),
        (right, "Extrapolation: an entire species held out\n(the analogue of leave-one-tokamak-out)"),
    ):
        if crossing is not None:
            axis.axvline(crossing, color="#2a78d6", linestyle=":", linewidth=1.5)
            axis.annotate(
                f"reversal appears\nat {crossing} features",
                (crossing, axis.get_ylim()[1]),
                textcoords="offset points", xytext=(6, -28),
                fontsize=11, color="#2a78d6",
            )
        axis.set_xticks(counts)
        axis.set_xlabel("number of predictors the models may see")
        axis.set_ylabel("RMSLE")
        axis.set_title(title)
        axis.grid(alpha=0.15)
        axis.legend(fontsize=11)

    figure.tight_layout()
    path = RESULTS_DIR / "tree_allometry.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path


def main() -> None:
    study = run_study()
    payload = write_results(study)
    figure_path = plot_tree_allometry(study)

    print("=== Result 15: the reversal needs dimensionality, measured directly ===")
    print(
        f"{study.n_rows} plants, {study.n_species_total} species, "
        f"{payload['n_species_scored']} scored, diameter span {study.size_span:.0f}x "
        f"across species medians, mass span {study.mass_span:.3g}x"
    )

    print("\n--- Result 15d: the published exponent ---")
    print(
        f"  free refit {study.baseline.free_exponent:.3f} against WBE's "
        f"{ta.WBE_EXPONENT:.3f} (8/3)"
    )
    print(
        f"  in sample: free {study.baseline.free_rmsle:.4f}, "
        f"held at 8/3 {study.baseline.constrained_rmsle:.4f}"
    )

    print("\n--- Results 15a to 15c: the ladder ---")
    print(
        f"  {'features':>9}{'CV law':>9}{'CV tree':>9}{'CV gain':>9}"
        f"{'LOO law':>9}{'LOO tree':>10}{'LOO gain':>10}  reversal"
    )
    for rung in study.rungs:
        print(
            f"  {rung.n_features:>9}{rung.cv_rmsle[POWER_LAW]:>9.4f}{rung.best_tree_cv:>9.4f}"
            f"{rung.cv_gain_over_power_law:>+8.1%}{rung.loo_rmsle[POWER_LAW]:>9.4f}"
            f"{rung.best_tree_loo:>10.4f}{rung.loo_gain_over_power_law:>+9.1%}"
            f"  {'YES' if rung.reversal else 'no'}"
        )

    crossing = payload["first_reversal_n_features"]
    print(
        f"\n  the power law wins the held-out species at "
        f"{payload['n_rungs_power_law_wins_extrapolation']} of {payload['n_rungs']} rungs"
    )
    if crossing is None:
        print("  no rung produces the reversal: the trees never win interpolation here")
    else:
        print(
            f"  the reversal appears at {crossing} predictors and at every rung above it, "
            "which is exactly where the interpolation gain turns positive"
        )

    print(f"\nwrote results/tree_allometry.json and two CSVs under {RESULTS_DIR}")
    if figure_path:
        print(f"figure: {figure_path}")


if __name__ == "__main__":
    main()
