"""Result 16: the reversal without the loss power in the feature set.

The thermal confinement time is defined as the thermal stored energy divided by
the thermal loss power, so ``log tau_th = log W_th - log P_LTH`` up to the
deposit's dW/dt and radiation conventions. ``PLTH`` is one of the nine features
every model in this study sees, which means one predictor is the denominator of
the response.

That admits a reading of Result 4 that nothing else in this repository rules
out. A log-linear model can carry the -1 coefficient on log P exactly, and carry
it outside the training range of P without error, because the relation is
definitional rather than fitted. A tree cannot: it approximates a -1 log-slope
by axis-aligned splits and, by the training-target bound of Result 4c, cannot
extrapolate it at all. Under that reading the power law transfers to an unseen
device because an identity is exactly log-linear, not because it has better
long-range structure, and "long-range saturation rather than flexibility" is the
wrong diagnosis of a real observation.

This module removes the term. Every model is refitted on the eight remaining
engineering features and scored under the same three comparisons: grouped
cross-validation by discharge, leave-one-label-out over the 13 eligible labels,
and leave-one-device-out over the 11 physical devices. If the reversal survives,
it does not depend on P being in the feature set and the objection is answered.

Retargeting to ``log W_th`` would be the cleaner control and is not available:
the STD5 deposit carries ``TAUTH`` and ``PLTH`` and no stored-energy column, so
the identity cannot be inverted from what is published.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

import gp
import hdb5
from storage import write_dataframe_csv_atomic, write_json_strict

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# The denominator of the target, and the only feature this module removes.
POWER_COLUMN = "log_p_loss_mw"

ABLATED_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    column for column in hdb5.BLIND_FEATURE_COLUMNS if column != POWER_COLUMN
)

# The three models the reversal is stated over, in the order Table 1 uses.
POWER_LAW = "ridge_loglinear"
FOREST = "random_forest"
BOOSTER = "hist_gradient_boosting"


def device_grouped(dataset: pd.DataFrame) -> pd.DataFrame:
    """The same rows, with the label column replaced by the physical device."""
    framed = hdb5.with_device_column(dataset).copy()
    framed[hdb5.TOKAMAK_LABEL_COLUMN] = framed[hdb5.DEVICE_COLUMN]
    return framed


def _win_count(per_unit: pd.DataFrame, worse: str, better: str) -> dict[str, Any]:
    """How often `worse` scores above `better` on a held-out unit, and by how much."""
    wide = per_unit.pivot(index="tokamak", columns="model_name", values="rmsle")
    paired = wide[[worse, better]].dropna()
    differences = paired[worse] - paired[better]
    return {
        "n_units": int(len(paired)),
        "n_worse": int((differences > 0).sum()),
        "mean_difference": float(differences.mean()),
        "units": sorted(paired.index),
    }


def _arm(dataset: pd.DataFrame, features: tuple[str, ...]) -> dict[str, Any]:
    cv = {score.model_name: score.cv_rmsle for score in hdb5.evaluate_models(dataset, feature_columns=features)}

    by_label = hdb5.leave_one_tokamak_out(dataset, feature_columns=features)
    by_device = hdb5.leave_one_tokamak_out(device_grouped(dataset), feature_columns=features)

    summary = hdb5.summarize_leave_one_tokamak_out(by_label).set_index("model_name")
    return {
        "n_features": len(features),
        "features": list(features),
        "cv_rmsle": {name: float(value) for name, value in cv.items()},
        "lolo_mean_rmsle": {name: float(summary.loc[name, "mean_rmsle"]) for name in summary.index},
        "forest_worse_by_label": _win_count(by_label, FOREST, POWER_LAW),
        "forest_worse_by_device": _win_count(by_device, FOREST, POWER_LAW),
        "booster_worse_by_label": _win_count(by_label, BOOSTER, POWER_LAW),
        "per_label": by_label,
        "per_device": by_device,
    }


def _interval_arm(dataset: pd.DataFrame, features: tuple[str, ...]) -> dict[str, Any]:
    """Does the interval collapse need P? It is the strongest result in the paper.

    Nothing about a conformal interval involves the target's definition, so the
    expectation is that it does not, but expecting is not measuring.
    """
    _, in_distribution = hdb5.conformal_coverage_grouped_cv(dataset, feature_columns=features)
    _, held_out = hdb5.conformal_coverage_leave_one_tokamak_out(dataset, feature_columns=features)

    def coverage(frame: pd.DataFrame) -> dict[str, float]:
        """The pooled row, not the per-machine ones: the paper quotes pooled."""
        pooled = frame[frame["scope"] == "__pooled__"].set_index("model_name")
        return {
            name: float(pooled.loc[name, "empirical_coverage"])
            for name in (POWER_LAW, FOREST, BOOSTER)
            if name in pooled.index
        }

    return {"grouped_cv": coverage(in_distribution), "leave_one_label_out": coverage(held_out)}


def _kernel_arm(dataset: pd.DataFrame, features: tuple[str, ...]) -> dict[str, Any]:
    """The mechanism claim, which is what the objection actually threatens.

    If the bounded kernel still fails and the unbounded one still does not, the
    long-range-saturation reading holds whether or not P is a feature.
    """
    models = gp.build_gp_models()
    per_label = hdb5.leave_one_tokamak_out(
        dataset, feature_columns=features, extra_models=models
    )
    summary = hdb5.summarize_leave_one_tokamak_out(per_label).set_index("model_name")
    return {
        name: float(summary.loc[name, "mean_rmsle"])
        for name in summary.index
        if name.startswith("gp_") or name in (POWER_LAW, FOREST)
    }


def analyze(dataset: pd.DataFrame, *, with_kernels: bool = True) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for name, features in (
        ("baseline", hdb5.BLIND_FEATURE_COLUMNS),
        ("ablated", ABLATED_FEATURE_COLUMNS),
    ):
        arm = _arm(dataset, features)
        arm["conformal_coverage"] = _interval_arm(dataset, features)
        if with_kernels:
            arm["kernel_ladder_lolo"] = _kernel_arm(dataset, features)
        arms[name] = arm
    return arms


def _report(arms: dict[str, Any]) -> None:
    for name, arm in arms.items():
        label = arm["forest_worse_by_label"]
        device = arm["forest_worse_by_device"]
        print(f"\n{name}: {arm['n_features']} features")
        print(f"  CV      power law {arm['cv_rmsle'][POWER_LAW]:.4f}   forest {arm['cv_rmsle'][FOREST]:.4f}")
        print(
            f"  LOLO    power law {arm['lolo_mean_rmsle'][POWER_LAW]:.4f}   forest {arm['lolo_mean_rmsle'][FOREST]:.4f}"
        )
        print(
            f"  forest worse on {label['n_worse']} of {label['n_units']} labels,"
            f" {device['n_worse']} of {device['n_units']} devices"
        )
        cover = arm.get("conformal_coverage", {}).get("leave_one_label_out", {})
        if cover:
            spelled = "  ".join(f"{k.split('_')[0]} {v:.0%}" for k, v in cover.items())
            print(f"  held-out coverage at nominal 90%: {spelled}")
        ladder = arm.get("kernel_ladder_lolo", {})
        for name in sorted(ladder):
            print(f"  LOLO {name:26s} {ladder[name]:.4f}")


def main() -> None:
    dataset = hdb5.prepare_dataset()
    arms = analyze(dataset)
    _report(arms)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "dataset_sha256": hdb5.fingerprint_file(hdb5.default_hdb5_path()).sha256,
        "removed_feature": POWER_COLUMN,
        "target_column": hdb5.TARGET_COLUMN,
    }
    for name, arm in arms.items():
        payload[name] = {key: value for key, value in arm.items() if not key.startswith("per_")}
        write_dataframe_csv_atomic(RESULTS_DIR / f"power_ablation_{name}_per_label.csv", arm["per_label"])
        write_dataframe_csv_atomic(RESULTS_DIR / f"power_ablation_{name}_per_device.csv", arm["per_device"])
    write_json_strict(RESULTS_DIR / "power_ablation.json", payload)
    print(f"\nwrote {RESULTS_DIR / 'power_ablation.json'}")


if __name__ == "__main__":
    main()
