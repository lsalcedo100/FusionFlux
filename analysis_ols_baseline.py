"""The headline comparison with plain least squares as the power law.

The power law the paper scores against the tree ensembles is a ridge
regression on the log features with the default penalty, alpha = 1 on
standardised columns. Confinement scalings are historically fitted by
unpenalised least squares, so a reader can ask whether the reversal depends
on the penalty. The constrained-fit module already solves the unpenalised
problem (the unconstrained rung of the Connor--Taylor hierarchy); this scores
that fit beside the forest and the booster under the two leave-one-out splits
and writes the per-unit scores, so the count of labels and devices on which
each ensemble loses to the unpenalised fit is a generated number rather than
an argument.

Run ``python3 analysis_ols_baseline.py`` to regenerate
``results/ols_baseline.json`` and ``results/ols_baseline_per_unit.csv``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

import analysis_device_arm as arm
import dimensional as dm
import hdb5
from storage import write_dataframe_csv_atomic, write_json_strict

RESULTS_DIR = Path(__file__).resolve().parent / "results"

OLS = "powerlaw_free"
RIDGE = "ridge_loglinear"
ENSEMBLES = ("random_forest", "hist_gradient_boosting")


def _ols_model() -> dict[str, Pipeline]:
    """The unconstrained least-squares rung, wrapped the way the device arm wraps it."""
    return {OLS: arm._wrapped(dm.build_constrained_models()[OLS])}


def _comparison(per_unit: pd.DataFrame, unit: str) -> dict[str, Any]:
    scores = per_unit.pivot(index="tokamak", columns="model_name", values="rmsle").astype(float)
    out: dict[str, Any] = {
        "n_units": int(len(scores)),
        "ols_mean_rmsle": float(scores[OLS].mean()),
        "ridge_mean_rmsle": float(scores[RIDGE].mean()),
        "ols_minus_ridge_mean": float((scores[OLS] - scores[RIDGE]).mean()),
    }
    for name in ENSEMBLES:
        gap = scores[name] - scores[OLS]
        ridge_gap = scores[name] - scores[RIDGE]
        # With eleven or thirteen dependent units the mean gap's bootstrap
        # interval is a loose summary; the median, and how far the mean moves
        # when any one unit is dropped, say the same thing with no resampling.
        drop_one = [float(ridge_gap.drop(unit).mean()) for unit in ridge_gap.index]
        out[name] = {
            "mean_rmsle": float(scores[name].mean()),
            "n_worse_than_ols": int((gap > 0).sum()),
            "mean_gap_vs_ols": float(gap.mean()),
            "n_worse_than_ridge": int((ridge_gap > 0).sum()),
            "mean_gap_vs_ridge": float(ridge_gap.mean()),
            "median_gap_vs_ridge": float(ridge_gap.median()),
            "min_gap_vs_ridge": float(ridge_gap.min()),
            "drop_one_unit_mean_gap_vs_ridge": {"min": min(drop_one), "max": max(drop_one)},
        }
    out["unit"] = unit
    return out


def analyze(dataset: pd.DataFrame | None = None) -> tuple[dict[str, Any], pd.DataFrame]:
    if dataset is None:
        dataset = hdb5.prepare_dataset()
    models = _ols_model()
    per_label = hdb5.extrapolation_report(dataset, extra_models=models)
    per_device = hdb5.extrapolation_report(arm.device_grouped(dataset), extra_models=models)
    keep = [OLS, RIDGE, *ENSEMBLES]
    per_label = per_label[per_label["model_name"].isin(keep)].assign(unit="label")
    per_device = per_device[per_device["model_name"].isin(keep)].assign(unit="device")
    report = {
        "dataset_sha256": hdb5.fingerprint_file(hdb5.default_hdb5_path()).sha256,
        "ols_model": OLS,
        "by_label": _comparison(per_label, "label"),
        "by_device": _comparison(per_device, "device"),
    }
    columns = ["unit", "model_name", "tokamak", "n_held_out_rows", "rmsle"]
    return report, pd.concat([per_label[columns], per_device[columns]], ignore_index=True)


def main() -> None:
    report, per_unit = analyze()
    RESULTS_DIR.mkdir(exist_ok=True)
    write_json_strict(RESULTS_DIR / "ols_baseline.json", report)
    write_dataframe_csv_atomic(RESULTS_DIR / "ols_baseline_per_unit.csv", per_unit)
    for arm_name in ("by_label", "by_device"):
        c = report[arm_name]
        print(
            f"--- {arm_name}: {c['n_units']} units; OLS {c['ols_mean_rmsle']:.3f}, "
            f"ridge {c['ridge_mean_rmsle']:.3f} (OLS - ridge {c['ols_minus_ridge_mean']:+.4f}) ---"
        )
        for name in ENSEMBLES:
            e = c[name]
            print(
                f"  {name:24s} {e['mean_rmsle']:.3f}; worse than OLS on "
                f"{e['n_worse_than_ols']}/{c['n_units']} (gap {e['mean_gap_vs_ols']:+.3f}); "
                f"worse than ridge on {e['n_worse_than_ridge']}/{c['n_units']}"
            )
    print(f"\nWrote {RESULTS_DIR / 'ols_baseline.json'}")


if __name__ == "__main__":
    main()
