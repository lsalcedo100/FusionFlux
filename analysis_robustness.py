"""Does the ranking inversion survive the choices Table 1 leaves implicit?

Run ``python3 analysis_robustness.py`` to regenerate ``results/robustness.json``.

Table 1 compares a grouped-CV score against a leave-one-machine-out score and
attributes the difference to the split. Two things it does not hold fixed could
carry some of that difference instead, and a third makes the held-out unit less
independent than the phrase "a machine it has not seen" suggests:

    population   Grouped CV scores all 6228 rows; leave-one-out scores only the
                 13 machines with enough rows to hold out. Different rows.

    aggregation  Grouped CV pools out-of-fold predictions and takes one RMSLE
                 over every row, so JET and AUG dominate it. Leave-one-out
                 averages per-machine RMSLE, so every machine counts once.

    unit         JET and JET-ILW are one physical tokamak before and after its
                 wall change, as are AUG and AUGW. Holding out JET-ILW while
                 training on JET is not holding out an unseen device.

This script scores the three contenders under every combination: both
populations, both aggregations, and both definitions of a machine. It also runs
the exact paired sign test that the win count supports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

import hdb5
from storage import write_json_strict

RESULTS_DIR = Path(__file__).resolve().parent / "results"

CONTENDERS = ("ridge_loglinear", "random_forest", "hist_gradient_boosting")
REFERENCE = "ipb98y2_analytic"

# JET and AUG each appear twice: once before a wall change and once after. The
# database treats the two as separate labels, which is right for physics and
# wrong for the question "has this model seen this device".
PHYSICAL_DEVICE: dict[str, str] = {"JETILW": "JET", "AUGW": "AUG"}


def _rmsle(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.log(predicted) - np.log(actual)) ** 2)))


def _device_labels(dataset: pd.DataFrame) -> np.ndarray:
    labels = dataset[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()
    return np.array([PHYSICAL_DEVICE.get(str(label), str(label)) for label in labels])


def _cross_validate(dataset: pd.DataFrame, units: np.ndarray) -> dict[str, dict[str, Any]]:
    """Grouped CV by discharge, scored both pooled over rows and per unit."""
    features = dataset[list(hdb5.BLIND_FEATURE_COLUMNS)]
    tau = dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float)
    groups = dataset[hdb5.GROUP_COLUMN].to_numpy()
    n_splits = min(hdb5.N_CV_FOLDS, int(pd.Series(groups).nunique()))
    zoo = hdb5._assemble_zoo()

    predictions = {REFERENCE: dataset["ipb98y2_tau_s"].to_numpy(dtype=float)}
    for name in CONTENDERS:
        predictions[name] = np.exp(hdb5._grouped_cv_predictions(zoo[name], features, np.log(tau), groups, n_splits))
    return _both_aggregations(tau, units, predictions)


def _leave_one_unit_out(dataset: pd.DataFrame, units: np.ndarray) -> dict[str, dict[str, Any]]:
    """Hold out each unit in turn, scored both pooled over rows and per unit."""
    features = dataset[list(hdb5.BLIND_FEATURE_COLUMNS)]
    tau = dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float)
    log_tau = np.log(tau)
    reference = dataset["ipb98y2_tau_s"].to_numpy(dtype=float)

    eligible = [u for u in pd.unique(units) if (units == u).sum() >= hdb5.MIN_HELD_OUT_ROWS]
    predictions = {name: np.full(len(dataset), np.nan) for name in (*CONTENDERS, REFERENCE)}
    zoo = hdb5._assemble_zoo()
    for unit in eligible:
        held = units == unit
        train_rows = np.flatnonzero(~held)
        held_rows = np.flatnonzero(held)
        for name in CONTENDERS:
            model = hdb5.clone_pipeline(zoo[name])
            with hdb5._suppress_benign_matmul_warnings():
                hdb5.fit_pipeline(model, features.iloc[train_rows], log_tau[train_rows])
                predictions[name][held] = np.exp(model.predict(features.iloc[held_rows]))
        predictions[REFERENCE][held] = reference[held]

    scored = ~np.isnan(predictions["random_forest"])
    return _both_aggregations(
        tau[scored],
        units[scored],
        {name: value[scored] for name, value in predictions.items()},
    )


def _both_aggregations(
    actual: np.ndarray, units: np.ndarray, predictions: dict[str, np.ndarray]
) -> dict[str, dict[str, Any]]:
    """One RMSLE pooled over rows, and the mean of the per-unit RMSLEs."""
    distinct = list(pd.unique(units))
    out: dict[str, dict[str, Any]] = {}
    for name, predicted in predictions.items():
        per_unit = {str(unit): _rmsle(actual[units == unit], predicted[units == unit]) for unit in distinct}
        out[name] = {
            "pooled_rows": _rmsle(actual, predicted),
            "unit_equal": float(np.mean(list(per_unit.values()))),
            "n_units": len(per_unit),
            "per_unit": per_unit,
        }
    return out


def _sign_test(per_unit_a: dict[str, float], per_unit_b: dict[str, float]) -> dict[str, object]:
    """Exact two-sided binomial test on which model wins each unit."""
    from math import comb

    units = sorted(per_unit_a)
    losses = sum(per_unit_a[u] > per_unit_b[u] for u in units)
    n = len(units)
    extreme = max(losses, n - losses)
    tail = sum(comb(n, k) for k in range(extreme, n + 1)) / 2**n
    return {
        "n_units": n,
        "n_units_a_worse": int(losses),
        "mean_difference": float(np.mean([per_unit_a[u] - per_unit_b[u] for u in units])),
        "exact_two_sided_p": float(min(1.0, 2 * tail)),
    }


def analyze_robustness(dataset: pd.DataFrame) -> dict[str, object]:
    labels = dataset[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()
    devices = _device_labels(dataset)
    eligible = hdb5.eligible_tokamaks(dataset, min_rows=hdb5.MIN_HELD_OUT_ROWS)

    # The same rows under both splits: only machines leave-one-out can score.
    scored = dataset[np.isin(labels, eligible)].reset_index(drop=True)
    scored_labels = scored[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()

    arms = {
        "cv_all_rows": _cross_validate(dataset, labels),
        "cv_scored_machines_only": _cross_validate(scored, scored_labels),
        "lomo_by_database_label": _leave_one_unit_out(dataset, labels),
        "lomo_by_physical_device": _leave_one_unit_out(dataset, devices),
    }

    sign_tests = {
        arm: _sign_test(arms[arm]["random_forest"]["per_unit"], arms[arm]["ridge_loglinear"]["per_unit"])
        for arm in ("lomo_by_database_label", "lomo_by_physical_device")
    }

    def order(arm: str, key: str) -> list[str]:
        return sorted(CONTENDERS, key=lambda name: arms[arm][name][key])

    inversion = {
        f"{cv}|{lomo}|{key}": order(cv, key) == list(reversed(order(lomo, key)))
        for cv in ("cv_all_rows", "cv_scored_machines_only")
        for lomo in ("lomo_by_database_label", "lomo_by_physical_device")
        for key in ("pooled_rows", "unit_equal")
    }

    return {
        "n_rows": int(len(dataset)),
        "n_rows_scored_machines_only": int(len(scored)),
        "eligible_machines": list(eligible),
        "physical_device_map": PHYSICAL_DEVICE,
        "arms": arms,
        "sign_tests": sign_tests,
        "inversion_holds": inversion,
        "inversion_holds_everywhere": all(inversion.values()),
    }


def main() -> None:
    analysis = analyze_robustness(hdb5.prepare_dataset())
    write_json_strict(RESULTS_DIR / "robustness.json", analysis)

    arms = cast("dict[str, dict[str, Any]]", analysis["arms"])
    for arm, scores in arms.items():
        print(f"\n--- {arm} ---")
        for name in (*CONTENDERS, REFERENCE):
            row = scores[name]
            print(
                f"  {name:24s} pooled={row['pooled_rows']:.4f}  "
                f"unit-equal={row['unit_equal']:.4f}  (n={row['n_units']})"
            )
    print("\n--- forest against the blind log-linear power law ---")
    sign_tests = cast("dict[str, dict[str, Any]]", analysis["sign_tests"])
    for arm, test in sign_tests.items():
        print(
            f"  {arm:26s} forest worse on {test['n_units_a_worse']}/{test['n_units']}, "
            f"mean gap {test['mean_difference']:+.4f}, exact p={test['exact_two_sided_p']:.2e}"
        )
    print(f"\ninversion holds in every cell: {analysis['inversion_holds_everywhere']}")
    print(f"Wrote {RESULTS_DIR / 'robustness.json'}")


if __name__ == "__main__":
    main()
