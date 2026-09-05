"""Result 13: run the audit on Kleiber's law and see whether the reversal holds.

Regenerates ``results/allometry.json``, its three CSVs and ``results/allometry.png``.

The three splits mirror Results 4, 5 and 8 exactly, on a dataset with no plasma
in it:

* **Grouped CV by species.** The within-group unit, as a discharge is within a
  machine. Every order in the held-out fold is also in the training fold.
* **Leave-one-order-out.** The unit the claim is about, as a tokamak is.
* **Mass-ordered cuts.** Train on the lightest orders and predict the heaviest,
  which is the size extrapolation ITER asks for, in a different currency.

Everything is computed through ``scaling_audit``, not through a copy of
``hdb5``. The reusable module is therefore exercised on real external data by
the same procedure a reader would apply to their own dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold

import allometry as al
import scaling_audit as sa
from figures import save_figure
from storage import write_dataframe_csv_atomic, write_json_strict

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Pinned so a rerun reproduces: these are reported numbers, not a demo.
RANDOM_STATE = 0
N_CV_SPLITS = 5

LABELS = {
    "kleiber": "Kleiber, exponent 3/4",
    "ols_loglinear": "power law, free exponent",
    "random_forest": "random forest",
    "hist_gradient_boosting": "hist gradient boosting",
    "mean_baseline": "mean baseline",
}

# The Kleiber constraint, in the form scaling_audit takes it: with an intercept
# column prepended, the design is [1, log_mass], so pinning the mass exponent to
# 3/4 is one row.
KLEIBER_CONSTRAINT = np.array([[0.0, 1.0]])
KLEIBER_RHS = np.array([al.KLEIBER_EXPONENT])


def build_models() -> dict[str, Any]:
    """The zoo, deliberately small.

    With one predictor there is nothing for a model to do but choose a shape, so
    two shapes and two flexible learners is the whole space worth scoring.
    """
    return {
        "kleiber": sa.ConstrainedLinearRegression(KLEIBER_CONSTRAINT, KLEIBER_RHS),
        "ols_loglinear": sa.ConstrainedLinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=400, random_state=RANDOM_STATE, n_jobs=1
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        "mean_baseline": DummyRegressor(strategy="mean"),
    }


def _rmsle(actual: np.ndarray, predicted: np.ndarray) -> float:
    """RMSE in log space, which is RMSLE, matching every other result here."""
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))


def grouped_cv_by_species(dataset: pd.DataFrame) -> dict[str, float]:
    """The easy split: hold out species, never a whole order.

    The analogue of grouped CV by discharge. Every order in the test fold is
    also in the training fold, so the model is interpolating within groups it
    has already seen -- which is exactly the condition the reversal hides under.
    """
    features = dataset[list(al.LOG_FEATURES)]
    target = dataset["log_bmr"].to_numpy(dtype=float)
    groups = dataset["species"].to_numpy()
    splitter = GroupKFold(n_splits=N_CV_SPLITS)

    scores: dict[str, list[float]] = {name: [] for name in LABELS}
    for train_index, test_index in splitter.split(features, target, groups):
        for name, estimator in build_models().items():
            fitted = estimator.fit(features.iloc[train_index], target[train_index])
            predicted = fitted.predict(features.iloc[test_index])
            scores[name].append(_rmsle(target[test_index], predicted))
    return {name: float(np.mean(values)) for name, values in scores.items()}


def mass_ordered_sweep(dataset: pd.DataFrame, orders: list[str]) -> pd.DataFrame:
    """Train on the lightest orders, predict every heavier one.

    ``scaling_audit.OrderedGroupSplit`` supplies the cuts; the ordering value is
    each order's median body mass, so the axis is the one the extrapolation is
    about rather than an arbitrary label ordering.
    """
    frame = dataset[dataset[al.GROUP_COLUMN].isin(orders)]
    features = frame[list(al.LOG_FEATURES)]
    target = frame["log_bmr"].to_numpy(dtype=float)
    labels = frame[al.GROUP_COLUMN].to_numpy()
    medians = {k: v for k, v in al.order_mass_medians(frame).items() if k in orders}

    records: list[dict[str, Any]] = []
    splitter = sa.OrderedGroupSplit(medians, min_train_groups=3, min_test_rows=al.MIN_HELD_OUT_ROWS)
    for train_index, test_index in splitter.split(groups=labels):
        train_max = float(np.max(target[train_index]))
        test_orders = sorted(set(labels[test_index]))
        mass_ratio = float(
            np.median(frame[al.MASS_COLUMN].to_numpy()[test_index])
            / np.median(frame[al.MASS_COLUMN].to_numpy()[train_index])
        )
        for name, estimator in build_models().items():
            fitted = estimator.fit(features.iloc[train_index], target[train_index])
            predicted = np.asarray(fitted.predict(features.iloc[test_index]), dtype=float)
            records.append(
                {
                    "model_name": name,
                    "n_train_orders": len(set(labels[train_index])),
                    "n_train_rows": int(train_index.size),
                    "n_test_rows": int(test_index.size),
                    "held_out_orders": ", ".join(test_orders),
                    "mass_ratio": mass_ratio,
                    "rmsle": _rmsle(target[test_index], predicted),
                    "fraction_above_train_max": float(np.mean(target[test_index] > train_max)),
                    "prediction_bounded": bool(predicted.max() <= train_max + 1e-12),
                }
            )
    return pd.DataFrame.from_records(records)


def _skill(score: float, baseline: float, reference: float) -> float:
    """Where a model sits between the mean predictor (0) and Kleiber (1)."""
    if np.isclose(baseline, reference):
        return float("nan")
    return float((baseline - score) / (baseline - reference))


def analyze() -> dict[str, Any]:
    dataset = al.prepare_dataset()
    orders = al.eligible_orders(dataset)
    scored = dataset[dataset[al.GROUP_COLUMN].isin(orders)].reset_index(drop=True)

    cv = grouped_cv_by_species(scored)
    report = sa.audit_groups(
        scored[list(al.LOG_FEATURES)],
        scored["log_bmr"].to_numpy(dtype=float),
        scored[al.GROUP_COLUMN],
        build_models(),
        min_held_out_rows=al.MIN_HELD_OUT_ROWS,
        scorer=_rmsle,
    )
    sweep = mass_ordered_sweep(scored, orders)

    loo = report.groupby("estimator")["score"].agg(["mean", "median", "max"])
    correlation = sa.distance_score_correlation(report)

    # The single cut that most resembles the ITER question: the largest jump in
    # median body mass between the training and held-out halves.
    pooled = sweep.groupby("n_train_orders")["mass_ratio"].first()
    widest = int(pooled.idxmax())
    widest_cut = sweep[sweep["n_train_orders"] == widest].set_index("model_name")

    cv_blind = {k: v for k, v in cv.items() if k != "mean_baseline"}
    loo_blind = loo["mean"].drop(index="mean_baseline", errors="ignore")
    best_cv = min(cv_blind, key=lambda k: cv_blind[k])
    best_loo = str(loo_blind.idxmin())

    per_order = report.pivot(index="group", columns="estimator", values="score")
    forest_loses = int((per_order["random_forest"] > per_order["kleiber"]).sum())

    # Counted across the whole sweep rather than read off the widest cut. The
    # constraint's record here is genuinely mixed and a single cut would hide
    # that in either direction.
    by_cut = sweep.pivot(index="n_train_orders", columns="model_name", values="rmsle")
    n_cuts = int(len(by_cut))
    wins = {
        "kleiber_beats_free_power_law": int((by_cut["kleiber"] < by_cut["ols_loglinear"]).sum()),
        "kleiber_beats_random_forest": int((by_cut["kleiber"] < by_cut["random_forest"]).sum()),
        "power_laws_beat_trees": int(
            (
                by_cut[["kleiber", "ols_loglinear"]].max(axis=1)
                < by_cut[["random_forest", "hist_gradient_boosting"]].min(axis=1)
            ).sum()
        ),
        "n_cuts": n_cuts,
    }

    # The premise of the Results 4 and 11 reversal is that the flexible model
    # wins the easy split first. Whether it holds here is the finding, so it is
    # recorded rather than inferred from the ranking.
    trees_win_cv = bool(
        min(cv["random_forest"], cv["hist_gradient_boosting"])
        < min(cv["kleiber"], cv["ols_loglinear"])
    )

    payload: dict[str, Any] = {
        "source": "Figshare 3549807, Supplement 1 (mammalian BMR and FMR)",
        "dataset_sha256": al.ALLOMETRY_SHA256,
        "published_law": "Kleiber (1932): BMR proportional to mass^(3/4)",
        "n_rows": int(len(scored)),
        "n_species": int(scored["species"].nunique()),
        "n_orders_scored": len(orders),
        "orders_scored": orders,
        "order_mass_ratio": float(
            max(al.order_mass_medians(scored).values())
            / min(al.order_mass_medians(scored).values())
        ),
        "free_refit_exponent": float(
            np.polyfit(scored["log_mass_g"], scored["log_bmr"], 1)[0]
        ),
        "kleiber_exponent": al.KLEIBER_EXPONENT,
        "scores": {
            name: {
                "cv_rmsle": cv[name],
                "loo_mean_rmsle": float(loo.loc[name, "mean"]),
                "loo_median_rmsle": float(loo.loc[name, "median"]),
                "loo_worst_rmsle": float(loo.loc[name, "max"]),
                "degradation_factor": float(loo.loc[name, "mean"] / cv[name]),
                "distance_spearman": float(correlation.get(name, float("nan"))),
                "mass_cut_rmsle": float(widest_cut.loc[name, "rmsle"]),
            }
            for name in LABELS
        },
        "best_cv_model": best_cv,
        "best_loo_model": best_loo,
        "ranking_reversed": best_cv != best_loo,
        "trees_win_cv": trees_win_cv,
        "sweep_wins": wins,
        "n_orders_forest_loses_to_kleiber": forest_loses,
        "n_orders": int(len(per_order)),
        "widest_cut": {
            "n_train_orders": widest,
            "mass_ratio": float(pooled.max()),
            "held_out_orders": str(widest_cut["held_out_orders"].iloc[0]),
            "n_test_rows": int(widest_cut["n_test_rows"].iloc[0]),
        },
    }
    for name in LABELS:
        payload["scores"][name]["mass_cut_skill"] = _skill(
            float(widest_cut.loc[name, "rmsle"]),
            float(widest_cut.loc["mean_baseline", "rmsle"]),
            float(widest_cut.loc["kleiber", "rmsle"]),
        )
    return {"payload": payload, "report": report, "sweep": sweep, "per_order": per_order}


def plot(payload: dict[str, Any], report: pd.DataFrame, sweep: pd.DataFrame) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - plotting is optional
        return None

    ink, muted = "#0b0b0b", "#52514e"
    style = {
        "kleiber": ("#3f8f5c", "Kleiber, 3/4"),
        "ols_loglinear": ("#2a78d6", "power law, free"),
        "random_forest": ("#eb6834", "random forest"),
        "hist_gradient_boosting": ("#c8873a", "hist grad boosting"),
    }

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 5.2))
    for axis in axes:
        axis.set_facecolor("#fcfcfb")
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        axis.tick_params(colors=muted, labelsize=9)

    # --- Left: the same slope chart as Result 4, on mammals -----------------
    for name, (colour, label) in style.items():
        cv = payload["scores"][name]["cv_rmsle"]
        loo = payload["scores"][name]["loo_mean_rmsle"]
        axes[0].plot([0, 1], [cv, loo], "o-", color=colour, linewidth=2.2, markersize=8, label=label)
        axes[0].annotate(f"{cv:.3f}", xy=(0, cv), xytext=(-42, -3), textcoords="offset points",
                         fontsize=13, color=colour)
        axes[0].annotate(f"{loo:.3f}", xy=(1, loo), xytext=(10, -3), textcoords="offset points",
                         fontsize=13, color=colour)
    axes[0].set_xlim(-0.45, 1.5)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["held out:\nspecies", "held out:\norder"],
                            fontsize=14, color=ink)
    axes[0].set_ylabel("RMSLE (lower is better)", fontsize=13, color=muted)
    # Deliberately not "the same inversion". There is none here, and that is the
    # finding: the trees lose under *both* splits, so no ranking can invert.
    axes[0].set_title("Kleiber's law under both splits",
                      fontsize=15, color=ink)
    axes[0].legend(frameon=False, fontsize=13, loc="upper left", labelcolor=muted)

    # --- Middle: error against distance, as in Result 4b --------------------
    for name, (colour, label) in style.items():
        rows = report[report["estimator"] == name].sort_values("mahalanobis")
        rho = payload["scores"][name]["distance_spearman"]
        axes[1].plot(rows["mahalanobis"], rows["score"], "o", color=colour, markersize=8,
                     label=f"{label}  (rho = {rho:+.2f})")
    axes[1].set_xlabel("distance of the held-out order from the rest", fontsize=13, color=muted)
    axes[1].set_ylabel("RMSLE on that order", fontsize=13, color=muted)
    axes[1].set_title("Error against extrapolation distance",
                      fontsize=15, color=ink)
    axes[1].legend(frameon=False, fontsize=13, loc="upper left", labelcolor=muted)

    # --- Right: the mass-ordered sweep, as in Result 5 ----------------------
    for name, (colour, label) in style.items():
        rows = sweep[sweep["model_name"] == name].sort_values("n_train_orders")
        axes[2].plot(rows["n_train_orders"], rows["rmsle"], "o-", color=colour,
                     linewidth=2.0, markersize=7, label=label)
    bounded = sweep[(sweep["model_name"] == "random_forest") & sweep["prediction_bounded"]]
    for _, row in bounded.iterrows():
        axes[2].plot(row["n_train_orders"], row["rmsle"], "o", markersize=15,
                     markerfacecolor="none", markeredgecolor=ink, markeredgewidth=1.3)
    axes[2].set_xlabel("orders in the training half", fontsize=13, color=muted)
    axes[2].set_ylabel("RMSLE on every heavier order", fontsize=13, color=muted)
    axes[2].set_yscale("log")
    lowest = float(sweep[sweep["model_name"] != "mean_baseline"]["rmsle"].min())
    highest = float(sweep[sweep["model_name"] != "mean_baseline"]["rmsle"].max())
    axes[2].set_ylim(lowest * 0.9, highest * 2.1)
    axes[2].set_title("The mass-ordered sweep",
                      fontsize=15, color=ink)
    axes[2].legend(frameon=False, fontsize=13, loc="upper right", labelcolor=muted, ncol=1)

    figure.tight_layout()
    path = RESULTS_DIR / "allometry.png"
    save_figure(figure, path, facecolor="#fcfcfb")
    plt.close(figure)
    return path


def main() -> None:
    al.download_allometry()
    analysis = analyze()
    payload = analysis["payload"]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json_strict(RESULTS_DIR / "allometry.json", payload)
    write_dataframe_csv_atomic(RESULTS_DIR / "allometry_per_order.csv", analysis["report"])
    write_dataframe_csv_atomic(RESULTS_DIR / "allometry_mass_sweep.csv", analysis["sweep"])
    plot(payload, analysis["report"], analysis["sweep"])

    print("--- Result 13: the audit on a scaling law from another science ---")
    print(f"  {payload['n_rows']} species records, {payload['n_orders_scored']} orders scored")
    print(f"  free refit exponent {payload['free_refit_exponent']:.3f} against Kleiber's 0.75")
    print()
    print(f"  {'model':26} {'CV':>8} {'leave-one-order-out':>21} {'mass cut':>10}")
    for name, scores in payload["scores"].items():
        print(
            f"  {LABELS[name]:26} {scores['cv_rmsle']:8.3f} "
            f"{scores['loo_mean_rmsle']:21.3f} {scores['mass_cut_rmsle']:10.3f}"
        )
    print()
    print(f"  best under CV                : {LABELS[payload['best_cv_model']]}")
    print(f"  best on an unseen order      : {LABELS[payload['best_loo_model']]}")
    print(f"  ranking reversed             : {payload['ranking_reversed']}")
    print(
        f"  forest loses to Kleiber on   : "
        f"{payload['n_orders_forest_loses_to_kleiber']} of {payload['n_orders']} orders"
    )
    wins = payload["sweep_wins"]
    print(f"  trees win the easy split     : {payload['trees_win_cv']}")
    print(
        f"  across {wins['n_cuts']} mass cuts             : "
        f"power laws beat both trees at {wins['power_laws_beat_trees']}, "
        f"Kleiber beats the free fit at {wins['kleiber_beats_free_power_law']}"
    )


if __name__ == "__main__":
    main()
