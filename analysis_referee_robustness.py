"""Two controls on choices the headline results leave implicit. Result 4f.

Run ``python3 analysis_referee_robustness.py`` to regenerate
``results/referee_robustness.json`` and its two CSVs.

The reversal is reported at one row threshold and under one deterministic
cross-validation partition. Neither choice was made to produce the result, but
neither is forced by anything either, so both deserve a measurement rather than
an assurance.

    threshold    Leave-one-machine-out scores only the labels with at least 30
                 rows. Five labels fall below that and stay in every training
                 fold. The win count is "13 of 13" at 30; this sweeps the
                 threshold over 10, 20, 30 and 50 and reports the count at each,
                 so the headline cannot be an artifact of where the line sits.

    partition    Grouped CV uses ``GroupKFold`` with ``shuffle=False``, which is
                 one deterministic assignment of discharges to five folds. The
                 paper is about validation protocol, so "the 29% margin is
                 peculiar to that one partition" is the obvious objection. This
                 repeats the comparison across shuffled assignments and reports
                 the spread of the forest-versus-power-law margin.

Neither control is expected to overturn anything. They are here so a reader does
not have to take that on trust.
"""

from __future__ import annotations

from math import comb
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import hdb5
from storage import write_json_strict

RESULTS_DIR = Path(__file__).resolve().parent / "results"

CONTENDERS = ("ridge_loglinear", "random_forest", "hist_gradient_boosting")
REFERENCE = "ipb98y2_analytic"

# The thresholds swept. 30 is the reported one and sits in the middle rather
# than at an end, which is the point of showing the others.
ROW_THRESHOLDS: tuple[int, ...] = (10, 20, 30, 50)

# Shuffled partitions of discharges into folds. The first entry reproduces
# nothing special; every one is an independent assignment.
CV_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)


def _rmsle(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.log(predicted) - np.log(actual)) ** 2)))


def _exact_two_sided_p(n_wins: int, n_trials: int) -> float:
    """Exact sign-test p under an even split, the statistic the win counts support."""
    if n_trials == 0:
        return float("nan")
    extreme = max(n_wins, n_trials - n_wins)
    tail = sum(comb(n_trials, k) for k in range(extreme, n_trials + 1))
    return float(min(1.0, 2.0 * tail / 2**n_trials))


def threshold_sweep(dataset: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-machine-out at each row threshold, with the paired win count.

    Only which labels are *scored* changes across rows of this table. Every
    label stays in every training fold at every threshold, exactly as at 30, so
    the models being compared are the same models throughout.
    """
    records: list[dict[str, Any]] = []
    for threshold in ROW_THRESHOLDS:
        per_machine = hdb5.leave_one_tokamak_out(dataset, min_rows=threshold)
        wide = per_machine.pivot(index="tokamak", columns="model_name", values="rmsle")
        machines = list(wide.index)
        differences = wide["random_forest"] - wide["ridge_loglinear"]
        n_forest_worse = int((differences > 0).sum())
        row: dict[str, Any] = {
            "min_rows": threshold,
            "n_machines_scored": len(machines),
            "n_forest_worse_than_power_law": n_forest_worse,
            "mean_gap_forest_minus_power_law": float(differences.mean()),
            "exact_two_sided_p": _exact_two_sided_p(n_forest_worse, len(machines)),
        }
        for name in (REFERENCE, *CONTENDERS):
            row[f"mean_rmsle_{name}"] = float(wide[name].mean())
        records.append(row)
    return pd.DataFrame(records)


def repeated_grouped_cv(dataset: pd.DataFrame, seeds: tuple[int, ...] = CV_SEEDS) -> pd.DataFrame:
    """Grouped CV by discharge under shuffled fold assignments, one row per seed.

    ``shuffle=True`` permutes the discharges before dealing them into folds, so
    each seed is a different partition of the same rows into the same number of
    folds. Everything else, the models, the features and the scoring, is what
    Table 1 uses.
    """
    features = dataset[list(hdb5.BLIND_FEATURE_COLUMNS)]
    tau = dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float)
    log_tau = np.log(tau)
    groups = dataset[hdb5.GROUP_COLUMN].to_numpy()
    zoo = hdb5._assemble_zoo()

    records: list[dict[str, Any]] = []
    for seed in seeds:
        splitter = GroupKFold(n_splits=hdb5.N_CV_FOLDS, shuffle=True, random_state=seed)
        folds = list(splitter.split(features, log_tau, groups))
        scores: dict[str, float] = {}
        for name in CONTENDERS:
            predictions = np.empty_like(log_tau, dtype=float)
            with hdb5._suppress_benign_matmul_warnings():
                for train_index, test_index in folds:
                    model = hdb5.clone_pipeline(zoo[name])
                    hdb5.fit_pipeline(model, features.iloc[train_index], log_tau[train_index])
                    predictions[test_index] = model.predict(features.iloc[test_index])
            scores[name] = _rmsle(tau, np.exp(predictions))
        margin = 1.0 - scores["random_forest"] / scores["ridge_loglinear"]
        records.append(
            {
                "seed": seed,
                **{f"cv_rmsle_{name}": scores[name] for name in CONTENDERS},
                "forest_margin_over_power_law": margin,
                "forest_beats_power_law": bool(margin > 0.0),
            }
        )
    return pd.DataFrame(records)


def build_report(dataset: pd.DataFrame) -> dict[str, Any]:
    thresholds = threshold_sweep(dataset)
    repeated = repeated_grouped_cv(dataset)
    margins = repeated["forest_margin_over_power_law"].to_numpy(dtype=float)
    return {
        "dataset_sha256": hdb5.HDB5_STD5_SHA256,
        "n_rows": int(len(dataset)),
        "reported_min_rows": hdb5.MIN_HELD_OUT_ROWS,
        "threshold_sweep": thresholds.to_dict(orient="records"),
        "forest_loses_every_scored_machine_at_every_threshold": bool(
            (
                thresholds["n_forest_worse_than_power_law"] == thresholds["n_machines_scored"]
            ).all()
        ),
        "repeated_grouped_cv": {
            "n_partitions": int(len(repeated)),
            "n_folds": hdb5.N_CV_FOLDS,
            "seeds": list(CV_SEEDS),
            "deterministic_partition_margin": None,
            "mean_margin": float(margins.mean()),
            "min_margin": float(margins.min()),
            "max_margin": float(margins.max()),
            "forest_wins_every_partition": bool(repeated["forest_beats_power_law"].all()),
            "per_seed": repeated.to_dict(orient="records"),
        },
    }


def main() -> None:
    dataset = hdb5.prepare_dataset()
    report = build_report(dataset)

    # The deterministic partition Table 1 reports, scored the same way, so the
    # spread across seeds can be read against it rather than in isolation.
    features = dataset[list(hdb5.BLIND_FEATURE_COLUMNS)]
    tau = dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float)
    log_tau = np.log(tau)
    groups = dataset[hdb5.GROUP_COLUMN].to_numpy()
    zoo = hdb5._assemble_zoo()
    deterministic = {
        name: _rmsle(
            tau,
            np.exp(
                hdb5._grouped_cv_predictions(
                    zoo[name], features, log_tau, groups, hdb5.N_CV_FOLDS
                )
            ),
        )
        for name in ("ridge_loglinear", "random_forest")
    }
    report["repeated_grouped_cv"]["deterministic_partition_margin"] = float(
        1.0 - deterministic["random_forest"] / deterministic["ridge_loglinear"]
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(report["threshold_sweep"]).to_csv(
        RESULTS_DIR / "referee_threshold_sweep.csv", index=False
    )
    pd.DataFrame(report["repeated_grouped_cv"]["per_seed"]).to_csv(
        RESULTS_DIR / "referee_repeated_cv.csv", index=False
    )
    write_json_strict(RESULTS_DIR / "referee_robustness.json", report)

    print("--- leave-one-label-out at four row thresholds ---")
    print(f"  {'min rows':>9}{'scored':>8}{'forest worse':>14}{'mean gap':>11}{'p':>10}")
    for row in report["threshold_sweep"]:
        print(
            f"  {row['min_rows']:>9}{row['n_machines_scored']:>8}"
            f"{row['n_forest_worse_than_power_law']:>8}/{row['n_machines_scored']:<5}"
            f"{row['mean_gap_forest_minus_power_law']:>+11.3f}"
            f"{row['exact_two_sided_p']:>10.1e}"
        )

    repeated = report["repeated_grouped_cv"]
    print("\n--- grouped CV under shuffled discharge partitions ---")
    print(f"  deterministic partition: {repeated['deterministic_partition_margin']:+.3%}")
    print(
        f"  {repeated['n_partitions']} shuffled partitions: mean "
        f"{repeated['mean_margin']:+.3%}, range "
        f"[{repeated['min_margin']:+.3%}, {repeated['max_margin']:+.3%}]"
    )
    print(f"  forest wins every partition: {repeated['forest_wins_every_partition']}")
    print(f"\nWrote {RESULTS_DIR / 'referee_robustness.json'}")


if __name__ == "__main__":
    main()
