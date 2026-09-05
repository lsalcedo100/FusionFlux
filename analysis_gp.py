"""Result 14: is it flexibility that breaks extrapolation, or boundedness?

Run ``python3 analysis_gp.py`` to regenerate everything under ``results/`` for
Result 14.

Result 4d and 4e establish that adding polynomial freedom to the log-linear
power law costs the tail, and that no ridge penalty rescues it. The limitations
section says what that does not settle: the sweep varies polynomial degree under
an isotropic L2 penalty and nothing else, so it cannot separate *flexibility*
from the particular way a polynomial misbehaves far from its data. It names the
missing experiment, a Gaussian process with a physically motivated kernel.

``gp.py`` builds it as a three-rung ladder in which only the kernel's long-range
behaviour changes: ``gp_rbf`` is bounded, ``gp_linear`` is an unbounded power
law, ``gp_linear_rbf`` is both flexible and unbounded. This scores all three
under the same three splits every other model in this repository is scored on,
by handing them to the existing ``extra_models`` hook rather than building a
second pipeline, so the numbers land in the same table as Results 4, 5 and 8.

The four measurements:

    Result 14a   The control. ``gp_linear`` against ``ridge_loglinear`` at every
                 split. A dot-product kernel is Bayesian linear regression in
                 the log features, so these are the same model reached two ways
                 and they should agree. If they do not, this module is wrong.
    Result 14b   The bounded rung at the ITER-size-matched cut, against the tree
                 ensembles it is predicted to resemble and the mean baseline it
                 is predicted to approach.
    Result 14c   The rung that is flexible *and* unbounded, which is the one the
                 limitation asks about and the one nothing here anticipated.
    Result 14d   Whether the GP's own posterior interval collapses out of
                 distribution the way the conformal intervals of Result 7 do.
                 A GP supplies uncertainty natively rather than by calibration,
                 so this asks whether that changes the answer.

The narrative built on these numbers is in ``results/RESULTS.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import gp
import hdb5
from analysis_extrapolation import spearman
from figures import (
    FONT_ANNOTATION,
    FONT_SMALL,
    PAPER_WIDTH_IN,
    save_figure,
)
from storage import write_dataframe_csv_atomic, write_json_strict

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# The models Result 14 is read against. Every one is already scored elsewhere in
# this repository under the same splits, so the comparison is to published rows
# rather than to a fresh fit.
REFERENCE_MODELS: tuple[str, ...] = (
    "ipb98y2_analytic",
    "ridge_loglinear",
    "random_forest",
    "hist_gradient_boosting",
    "mean_baseline",
)

GP_MODELS: tuple[str, ...] = ("gp_rbf", "gp_linear", "gp_linear_rbf")

# Result 14a's control tolerance. A dot-product-kernel GP and ridge on the same
# log features are the same model, but not the same arithmetic: ridge penalises
# with a fixed alpha while the GP learns its noise level, so they agree closely
# rather than exactly. 0.02 log-RMSE is far tighter than any gap this result
# discusses and far looser than the numerical noise between two solvers.
CONTROL_TOLERANCE_LOG_RMSE = 0.02

# Nominal coverage for Result 14d, matching Result 7 so the two are comparable.
NOMINAL_COVERAGE = 0.90


@dataclass(frozen=True)
class SplitScores:
    """One split, every model, one row each."""

    split_name: str
    scores: pd.DataFrame


@dataclass
class GaussianProcessStudy:
    """Everything Result 14 reports."""

    cv: pd.DataFrame
    per_machine: pd.DataFrame
    leave_one_out: pd.DataFrame
    iter_cut: pd.DataFrame
    size_sweep: pd.DataFrame
    coverage: pd.DataFrame
    reversion: pd.DataFrame
    distance_correlation: pd.Series
    iter_split_machines: tuple[str, ...] = ()
    iter_size_ratio: float = float("nan")
    learned_kernels: dict[str, str] = field(default_factory=dict)


def _rmsle_by_model(frame: pd.DataFrame) -> pd.Series:
    """log-RMSE indexed by model, from whichever shape the split returned.

    ``evaluate_models`` and ``score_size_split`` report one log-RMSE per model in a
    column named ``rmsle``; ``summarize_leave_one_tokamak_out`` averages over
    machines and names it ``mean_rmsle``. Reading both here keeps the three
    splits in one table without a per-split special case at every call site.
    """
    column = "rmsle" if "rmsle" in frame.columns else "mean_rmsle"
    return frame.set_index("model_name")[column]


def run_study(
    dataset: pd.DataFrame | None = None,
    *,
    n_tuning_rows: int = gp.DEFAULT_TUNING_ROWS,
) -> GaussianProcessStudy:
    """Score the kernel ladder under grouped CV, leave-one-machine-out and size."""
    if dataset is None:
        dataset = hdb5.prepare_dataset()

    models = gp.build_gp_models(n_tuning_rows=n_tuning_rows)

    # 1. Grouped CV by discharge: interpolation inside machines already seen.
    cv_scores = hdb5.evaluate_models(
        dataset,
        feature_columns=hdb5.BLIND_FEATURE_COLUMNS,
        extra_models=models,
    )
    cv = pd.DataFrame(
        [
            {"model_name": score.model_name, "rmsle": score.cv_rmsle, "r2_log": score.cv_r2_log}
            for score in cv_scores
        ]
    )

    # 2. Leave one entire machine out.
    per_machine = hdb5.extrapolation_report(dataset, extra_models=models)
    leave_one_out = hdb5.summarize_leave_one_tokamak_out(per_machine)

    # 3. The size-ordered cut matched to ITER's jump, and the whole sweep.
    splits = hdb5.size_ordered_splits(dataset)
    matched = hdb5.iter_matched_split(dataset, splits)
    iter_cut = hdb5.score_size_split(dataset, matched, extra_models=models)
    size_sweep, _ = hdb5.size_extrapolation_report(dataset, extra_models=models)

    # 4. Result 4b's diagnostic, recomputed for the ladder: does error track
    #    distance from the training data, as it does for the trees and does not
    #    for the power law?
    distance_correlation = _distance_correlations(per_machine)

    # 5 and 6. Reversion and native interval coverage need the fitted models and
    #    their predictions rather than a score, so they refit at the matched cut.
    coverage, reversion, learned = _posterior_diagnostics(
        dataset, matched, n_tuning_rows=n_tuning_rows
    )

    return GaussianProcessStudy(
        cv=cv,
        per_machine=per_machine,
        leave_one_out=leave_one_out,
        iter_cut=iter_cut,
        size_sweep=size_sweep,
        coverage=coverage,
        reversion=reversion,
        distance_correlation=distance_correlation,
        iter_split_machines=tuple(matched.test_machines),
        iter_size_ratio=float(matched.size_ratio),
        learned_kernels=learned,
    )


def _distance_correlations(per_machine: pd.DataFrame) -> pd.Series:
    """Rank correlation of per-machine error against extrapolation distance."""
    correlations: dict[str, float] = {}
    for name, rows in per_machine.groupby("model_name"):
        if len(rows) < 2:
            continue
        correlations[str(name)] = spearman(
            rows["feature_mahalanobis"].to_numpy(dtype=float),
            rows["rmsle"].to_numpy(dtype=float),
        )
    return pd.Series(correlations).sort_values(ascending=False)


def _posterior_diagnostics(
    dataset: pd.DataFrame,
    split: hdb5.SizeSplit,
    *,
    n_tuning_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Fit each rung once at the matched cut and interrogate its posterior.

    Two things a score cannot show. Whether the bounded rung has collapsed
    toward the training mean, which is Result 4c's ceiling in the form that
    applies to a GP; and whether the GP's own posterior interval covers, which
    Result 7 asks of conformal intervals and answers with 3% for a random
    forest. A GP does not need calibrating to produce an interval, so this is
    the same question put to a model that supplies uncertainty by construction.
    """
    from scipy.stats import norm

    label = dataset[hdb5.TOKAMAK_LABEL_COLUMN]
    train = dataset[label.isin(split.train_machines)]
    test = dataset[label.isin(split.test_machines)]

    columns = list(hdb5.BLIND_FEATURE_COLUMNS)
    x_train = train[columns]
    x_test = test[columns]
    log_train = np.log(train[hdb5.TARGET_COLUMN].to_numpy(dtype=float))
    log_test = np.log(test[hdb5.TARGET_COLUMN].to_numpy(dtype=float))

    z = float(norm.ppf(0.5 + NOMINAL_COVERAGE / 2.0))

    coverage_rows: list[dict[str, Any]] = []
    reversion_rows: list[dict[str, Any]] = []
    learned: dict[str, str] = {}

    for name, pipeline in gp.build_gp_models(n_tuning_rows=n_tuning_rows).items():
        pipeline.fit(x_train, log_train)
        estimator = pipeline.named_steps["model"]
        scaled = pipeline.named_steps["scale"].transform(x_test)
        mean, std = estimator.predict(scaled, return_std=True)

        learned[name] = str(estimator.kernel_)

        inside = np.abs(log_test - mean) <= z * std
        coverage_rows.append(
            {
                "model_name": name,
                "nominal": NOMINAL_COVERAGE,
                "empirical": float(inside.mean()),
                "median_half_width_log": float(np.median(z * std)),
                "n_rows": int(len(log_test)),
            }
        )

        diagnostic = gp.reversion_diagnostic(
            name, log_test, mean, float(estimator.training_target_mean_)
        )
        reversion_rows.append(
            {
                "model_name": diagnostic.model_name,
                "predicted_spread": diagnostic.predicted_spread,
                "actual_spread": diagnostic.actual_spread,
                "reversion": diagnostic.reversion,
                "predicted_mean_offset": diagnostic.predicted_mean_offset,
            }
        )

    return pd.DataFrame(coverage_rows), pd.DataFrame(reversion_rows), learned


def control_gap(study: GaussianProcessStudy) -> dict[str, float]:
    """Result 14a: how far the linear-kernel GP sits from ridge at each split."""
    gaps: dict[str, float] = {}
    for split_name, frame in (
        ("cv", study.cv),
        ("leave_one_tokamak_out", study.leave_one_out),
        ("iter_matched_cut", study.iter_cut),
    ):
        scores = _rmsle_by_model(frame)
        if "gp_linear" in scores.index and "ridge_loglinear" in scores.index:
            gaps[split_name] = float(abs(scores["gp_linear"] - scores["ridge_loglinear"]))
    return gaps


def write_results(study: GaussianProcessStudy, results_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    """Write every artifact Result 14 reports, and return the JSON payload."""
    write_dataframe_csv_atomic(results_dir / "gp_per_machine.csv", study.per_machine)
    write_dataframe_csv_atomic(results_dir / "gp_size_sweep.csv", study.size_sweep)
    write_dataframe_csv_atomic(results_dir / "gp_coverage.csv", study.coverage)

    cv = _rmsle_by_model(study.cv)
    loo = _rmsle_by_model(study.leave_one_out)
    cut = _rmsle_by_model(study.iter_cut)
    reversion = study.reversion.set_index("model_name")["reversion"]
    coverage = study.coverage.set_index("model_name")["empirical"]

    reported = [name for name in (*GP_MODELS, *REFERENCE_MODELS) if name in cut.index]
    payload: dict[str, Any] = {
        "iter_cut_machines": list(study.iter_split_machines),
        "iter_size_ratio": study.iter_size_ratio,
        "nominal_coverage": NOMINAL_COVERAGE,
        "learned_kernels": study.learned_kernels,
        "control_gap_rmsle": control_gap(study),
        "control_tolerance_rmsle": CONTROL_TOLERANCE_LOG_RMSE,
        "scores": {
            name: {
                "cv": float(cv[name]) if name in cv.index else None,
                "leave_one_tokamak_out": float(loo[name]) if name in loo.index else None,
                "iter_matched_cut": float(cut[name]),
                "distance_correlation": (
                    float(study.distance_correlation[name])
                    if name in study.distance_correlation.index
                    else None
                ),
                "reversion_at_iter_cut": (
                    float(reversion[name]) if name in reversion.index else None
                ),
                "coverage_at_iter_cut": (
                    float(coverage[name]) if name in coverage.index else None
                ),
            }
            for name in reported
        },
    }
    write_json_strict(results_dir / "gp.json", payload)
    return payload


def plot_gp(study: GaussianProcessStudy) -> Path | None:
    """Two panels: the ladder across splits, and what boundedness does.

    Left carries the argument. Three rungs of one model family, scored on the
    two splits that matter, with the tree ensembles and the power law behind
    them for scale: the bounded rung tracks the trees, the unbounded ones track
    the power law, and the split between them is a property of the kernel rather
    than of the amount of flexibility.

    Right is the mechanism. Predicted spread against actual spread on the
    held-out rows: a model that has reverted to its prior mean sits near zero
    however good its in-range fit was. Matplotlib is imported inside the
    function so the analysis can run headless without it installed.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - figure is optional
        return None

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(PAPER_WIDTH_IN, 6.2))

    loo = _rmsle_by_model(study.leave_one_out)
    cut = _rmsle_by_model(study.iter_cut)

    palette = {
        "gp_rbf": ("#d6301f", "o", "GP, RBF (bounded)"),
        "gp_linear": ("#2a78d6", "s", "GP, linear (a power law)"),
        "gp_linear_rbf": ("#1a9850", "D", "GP, linear + RBF"),
        "random_forest": ("#bbbbbb", "^", "random forest"),
        "hist_gradient_boosting": ("#999999", "v", "hist gradient boosting"),
        "ridge_loglinear": ("#000000", "x", "ridge, log-linear"),
    }
    positions = [0, 1]
    for model, (colour, marker, label) in palette.items():
        if model not in loo.index or model not in cut.index:
            continue
        values = [float(loo[model]), float(cut[model])]
        style = "--" if model in ("random_forest", "hist_gradient_boosting") else "-"
        top.plot(
            positions,
            values,
            style,
            color=colour,
            marker=marker,
            label=label,
            linewidth=2.0 if model.startswith("gp_") else 1.2,
            alpha=1.0 if model.startswith("gp_") else 0.65,
        )
        top.annotate(
            f"{values[1]:.3f}",
            (positions[1], values[1]),
            textcoords="offset points",
            xytext=(7, -3),
            fontsize=FONT_ANNOTATION,
            color=colour,
        )
    top.set_xticks(positions)
    top.set_xticklabels(["held-out machine", "ITER-size-matched cut"])
    top.set_yscale("log")
    top.set_ylabel("log-RMSE (log scale)")
    top.set_title("One model family, three kernels")
    top.grid(alpha=0.15)
    top.legend(fontsize=FONT_SMALL, loc="upper left")

    spread = study.reversion.set_index("model_name")
    # Two of the three kernels land almost on top of each other, so labels
    # alternate above and below their marker by rank in predicted spread.
    ranked = [
        model
        for model in spread.sort_values("predicted_spread", ascending=False).index
        if model in palette
    ]
    spread_order = {model: i for i, model in enumerate(ranked)}
    for model, (colour, marker, label) in palette.items():
        if model not in spread.index:
            continue
        row = spread.loc[model]
        bottom.scatter(
            float(row["actual_spread"]),
            float(row["predicted_spread"]),
            color=colour,
            marker=marker,
            s=90,
            label=label,
            zorder=3,
        )
        bottom.annotate(
            label,
            (float(row["actual_spread"]), float(row["predicted_spread"])),
            textcoords="offset points",
            xytext=(-10, 8 if spread_order.get(model, 0) % 2 == 0 else -15),
            ha="right",
            fontsize=FONT_SMALL,
            color=colour,
        )
    if not spread.empty:
        limit = float(spread["actual_spread"].max()) * 1.15
        bottom.plot([0, limit], [0, limit], color="#444444", linewidth=1.0, linestyle=":")
        bottom.annotate(
            "predictions as spread as the truth",
            (limit * 0.30, limit * 0.33),
            fontsize=FONT_SMALL,
            color="#444444",
            rotation=38,
        )
        bottom.set_xlim(0, limit)
        bottom.set_ylim(0, limit)
    bottom.set_xlabel("spread of the truth, held-out rows (std of log tau)")
    bottom.set_ylabel("spread of the predictions")
    bottom.set_title("Spread of predictions against spread of truth")
    bottom.grid(alpha=0.15)

    figure.tight_layout()
    path = RESULTS_DIR / "gp.png"
    save_figure(figure, path)
    plt.close(figure)
    return path


def main() -> None:
    study = run_study()
    payload = write_results(study)
    figure_path = plot_gp(study)

    cv = _rmsle_by_model(study.cv)
    loo = _rmsle_by_model(study.leave_one_out)
    cut = _rmsle_by_model(study.iter_cut)

    print("=== Result 14: flexibility against boundedness, on one model family ===")
    print(
        f"ITER-size-matched cut holds out {', '.join(study.iter_split_machines)} "
        f"at a size ratio of {study.iter_size_ratio:.3f}"
    )

    print("\n--- the ladder, and the models it is read against ---")
    print(f"  {'model':<26}{'CV':>9}{'held-out':>11}{'ITER cut':>11}{'rho(dist)':>11}")
    for name in (*GP_MODELS, *REFERENCE_MODELS):
        if name not in cut.index:
            continue
        cv_value = f"{cv[name]:.3f}" if name in cv.index else "-"
        loo_value = f"{loo[name]:.3f}" if name in loo.index else "-"
        rho = study.distance_correlation.get(name, float("nan"))
        rho_value = f"{rho:+.2f}" if np.isfinite(rho) else "-"
        print(f"  {name:<26}{cv_value:>9}{loo_value:>11}{cut[name]:>11.3f}{rho_value:>11}")

    print("\n--- Result 14a: the control ---")
    for split_name, gap in payload["control_gap_rmsle"].items():
        verdict = "agrees" if gap <= CONTROL_TOLERANCE_LOG_RMSE else "DISAGREES"
        print(f"  {split_name:<24}gap {gap:.4f}  {verdict}")
    print("  (a dot-product kernel is Bayesian linear regression in the log features)")

    print("\n--- Result 14b: what the bounded kernel does at the cut ---")
    spread = study.reversion.set_index("model_name")
    for name in GP_MODELS:
        if name not in spread.index:
            continue
        row = spread.loc[name]
        print(
            f"  {name:<26}reversion {float(row['reversion']):+.2f}  "
            f"predicted spread {float(row['predicted_spread']):.3f} "
            f"against {float(row['actual_spread']):.3f} actual"
        )
    if "gp_rbf" in cut.index and "mean_baseline" in cut.index:
        print(
            f"  gp_rbf scores {cut['gp_rbf']:.3f} against a mean baseline's "
            f"{cut['mean_baseline']:.3f}"
        )

    print("\n--- Result 14c: flexible and unbounded ---")
    if {"gp_linear_rbf", "ridge_loglinear"} <= set(cut.index):
        gain = 1.0 - cut["gp_linear_rbf"] / cut["ridge_loglinear"]
        print(
            f"  gp_linear_rbf {cut['gp_linear_rbf']:.3f} against ridge "
            f"{cut['ridge_loglinear']:.3f}: {gain:+.1%}"
        )
    for name, kernel in study.learned_kernels.items():
        print(f"  {name:<26}{kernel}")

    print(f"\n--- Result 14d: the GP's own {NOMINAL_COVERAGE:.0%} interval at the cut ---")
    for row in study.coverage.itertuples():
        print(
            f"  {row.model_name:<26}covers {row.empirical:>6.1%} of {row.n_rows} rows, "
            f"median half-width {row.median_half_width_log:.3f} in log space"
        )

    print(f"\nwrote results/gp.json and three CSVs under {RESULTS_DIR}")
    if figure_path:
        print(f"figure: {figure_path}")


if __name__ == "__main__":
    main()
