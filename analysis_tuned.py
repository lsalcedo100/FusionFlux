"""Does the ranking inversion survive tuning the flexible models properly?

Run ``python3 analysis_tuned.py`` to regenerate ``results/tuned.json``.

Every model in Table 1 uses library defaults, which invites the obvious
objection: the trees lost because nobody tried to make them win. This script
removes that objection by giving the two ensembles a hyperparameter search
nested inside each training fold. The held-out unit, whether a discharge fold,
a machine, or everything above the size cut, never takes part in choosing a
hyperparameter.

    outer split     what the paper reports: grouped CV by discharge,
                    leave-one-machine-out, and the ITER-size-matched cut.

    inner search    two selection procedures, run over the *training* rows
                    only. ``discharge`` groups the inner folds by discharge,
                    which is the conventional way a practitioner would tune.
                    ``machine`` holds out one training machine at a time, which
                    is model selection matched to how the model will be
                    deployed. The winner is refitted on the whole training fold
                    and scored once on the held-out rows.

If the inversion survives this, "the baselines were untuned" stops being an
explanation for it.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold

import hdb5
from storage import write_json_strict

RESULTS_DIR = Path(__file__).resolve().parent / "results"

N_INNER_FOLDS = 3

# Deliberately small but real grids: enough to move each model meaningfully,
# small enough that the nested search finishes in the time a reviewer would
# expect a rerun to take.
RF_GRID: dict[str, list[Any]] = {
    "max_features": [1.0, 0.5, "sqrt"],
    "min_samples_leaf": [1, 5],
}
HGB_GRID: dict[str, list[Any]] = {
    "learning_rate": [0.05, 0.1],
    "max_leaf_nodes": [15, 31],
}


def _rmsle_log(actual_log: np.ndarray, predicted_log: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predicted_log - actual_log) ** 2)))


def _configs(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*grid.values())]


def _build(name: str, config: dict[str, Any]) -> Any:
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=300, random_state=hdb5.RANDOM_STATE, n_jobs=-1, **config)
    return HistGradientBoostingRegressor(random_state=hdb5.RANDOM_STATE, **config)


def _inner_folds(
    inner_unit: np.ndarray, train_rows: np.ndarray, log_target: np.ndarray, mode: str
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Inner splits for the search, grouped by discharge or by machine."""
    units = inner_unit[train_rows]
    if mode == "machine":
        # One inner fold per training machine large enough to score.
        counts = pd.Series(units).value_counts()
        keep = [u for u, n in counts.items() if n >= hdb5.MIN_HELD_OUT_ROWS]
        return [(np.flatnonzero(units != u), np.flatnonzero(units == u)) for u in keep]
    n_splits = min(N_INNER_FOLDS, int(pd.Series(units).nunique()))
    splitter = GroupKFold(n_splits=n_splits)
    return list(splitter.split(train_rows, log_target[train_rows], units))


def _tune_and_fit(
    name: str,
    grid: dict[str, list[Any]],
    features: pd.DataFrame,
    log_target: np.ndarray,
    groups: np.ndarray,
    train_rows: np.ndarray,
    *,
    inner_unit: np.ndarray | None = None,
    mode: str = "discharge",
) -> tuple[Any, dict[str, Any]]:
    """Pick a configuration inside ``train_rows``, then refit on all of them."""
    unit = groups if inner_unit is None else inner_unit
    inner = _inner_folds(unit, train_rows, log_target, mode)

    best_score, best_config = np.inf, None
    for config in _configs(grid):
        fold_scores = []
        for inner_train, inner_test in inner:
            rows_a, rows_b = train_rows[inner_train], train_rows[inner_test]
            model = _build(name, config)
            with hdb5._suppress_benign_matmul_warnings():
                model.fit(features.iloc[rows_a], log_target[rows_a])
                predicted = model.predict(features.iloc[rows_b])
            fold_scores.append(_rmsle_log(log_target[rows_b], predicted))
        score = float(np.mean(fold_scores))
        if score < best_score:
            best_score, best_config = score, config

    chosen = cast("dict[str, Any]", best_config)
    final = _build(name, chosen)
    with hdb5._suppress_benign_matmul_warnings():
        final.fit(features.iloc[train_rows], log_target[train_rows])
    return final, {"config": {k: str(v) for k, v in chosen.items()}, "inner_rmsle": best_score}


def analyze_tuned(dataset: pd.DataFrame) -> dict[str, Any]:
    features = dataset[list(hdb5.BLIND_FEATURE_COLUMNS)]
    log_target = np.log(dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float))
    groups = dataset[hdb5.GROUP_COLUMN].to_numpy()
    labels = dataset[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()
    eligible = hdb5.eligible_tokamaks(dataset, min_rows=hdb5.MIN_HELD_OUT_ROWS)
    cut = hdb5.iter_matched_split(dataset, hdb5.size_ordered_splits(dataset))
    above_cut = np.isin(labels, list(cut.test_machines))

    grids: dict[str, dict[str, list[Any]]] = {
        "random_forest": RF_GRID,
        "hist_gradient_boosting": HGB_GRID,
    }
    out: dict[str, Any] = {}
    for name, grid in grids.items():
        chosen: list[dict[str, Any]] = []

        # Outer grouped CV by discharge, tuning inside each training fold.
        outer = GroupKFold(n_splits=min(hdb5.N_CV_FOLDS, int(pd.Series(groups).nunique())))
        predicted_cv = np.empty_like(log_target)
        for train_idx, test_idx in outer.split(features, log_target, groups):
            model, record = _tune_and_fit(name, grid, features, log_target, groups, train_idx)
            with hdb5._suppress_benign_matmul_warnings():
                predicted_cv[test_idx] = model.predict(features.iloc[test_idx])
            chosen.append({"split": "cv", **record})

        # Leave one machine out, tuning inside the remaining machines. Two
        # selection procedures, differing only in what the inner folds hold out:
        # discharges, which is how a practitioner would normally tune, and whole
        # machines, which matches the way the model will be deployed.
        per_machine: dict[str, dict[str, float]] = {"discharge": {}, "machine": {}}
        for mode in ("discharge", "machine"):
            for machine in eligible:
                held = labels == machine
                model, record = _tune_and_fit(
                    name,
                    grid,
                    features,
                    log_target,
                    groups,
                    np.flatnonzero(~held),
                    inner_unit=labels if mode == "machine" else None,
                    mode=mode,
                )
                with hdb5._suppress_benign_matmul_warnings():
                    predicted = model.predict(features.iloc[np.flatnonzero(held)])
                per_machine[mode][str(machine)] = _rmsle_log(log_target[held], predicted)
                chosen.append({"split": f"lomo:{machine}", "inner": mode, **record})

        # The ITER-size-matched cut, tuning below the cut only.
        model, record = _tune_and_fit(name, grid, features, log_target, groups, np.flatnonzero(~above_cut))
        with hdb5._suppress_benign_matmul_warnings():
            cut_predicted = model.predict(features.iloc[np.flatnonzero(above_cut)])
        chosen.append({"split": "iter_matched_cut", **record})

        out[name] = {
            "cv": _rmsle_log(log_target, predicted_cv),
            "leave_one_machine_out": float(np.mean(list(per_machine["discharge"].values()))),
            "leave_one_machine_out_inner_machine": float(np.mean(list(per_machine["machine"].values()))),
            "iter_matched_cut": _rmsle_log(log_target[above_cut], cut_predicted),
            "per_machine": per_machine,
            "chosen_configurations": chosen,
        }
    return {
        "n_rows": int(len(dataset)),
        "n_inner_folds": N_INNER_FOLDS,
        "grids": {k: {p: [str(v) for v in vs] for p, vs in g.items()} for k, g in grids.items()},
        "tuned": out,
    }


def main() -> None:
    analysis = analyze_tuned(hdb5.prepare_dataset())
    write_json_strict(RESULTS_DIR / "tuned.json", analysis)
    print("--- nested-tuned ensembles ---")
    for name, row in cast("dict[str, Any]", analysis["tuned"]).items():
        print(
            f"  {name:24s} CV={row['cv']:.4f}  "
            f"LOMO(inner=discharge)={row['leave_one_machine_out']:.4f}  "
            f"LOMO(inner=machine)={row['leave_one_machine_out_inner_machine']:.4f}  "
            f"cut={row['iter_matched_cut']:.4f}"
        )
    print(f"\nWrote {RESULTS_DIR / 'tuned.json'}")


if __name__ == "__main__":
    main()
