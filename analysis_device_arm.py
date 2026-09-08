"""Result 19: the constructive models under leave-one-device-out.

The paper argues, in Sec. 4.2 and again in Limitations, that the 13 database
labels are not 13 devices and that the claim it rests on is the 11-device arm.
It then scores every model it *recommends* on the label arm and the size cut and
never on the device arm: the Gaussian-process ladder, the Connor--Taylor rungs
and the mean-function ablation are all reported as CV / held-out label / ITER
cut. So the strict split was applied to the models under criticism and withheld
from the models under recommendation, and the "does not reverse" claim for the
linear-plus-RBF process rests on 0.218 against the power law's 0.214, a
difference the paper's own resolution statement calls a tie.

This module closes that. It relabels the dataset by physical device, folding the
two JET wall eras and the two ASDEX Upgrade entries onto one machine each, and
scores every constructive model over the 11 devices that clear the row
threshold. Nothing else changes: same nine blind features, same estimators, same
seeds, same eligibility rule.

Run ``python3 analysis_device_arm.py`` to regenerate ``results/device_arm.json``
and ``results/device_arm_per_device.csv``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import analysis_mechanism as mech
import dimensional as dm
import gp
import hdb5
from storage import write_dataframe_csv_atomic, write_json_strict

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def device_grouped(dataset: pd.DataFrame) -> pd.DataFrame:
    """The same rows, with the held-out unit switched from label to device."""
    framed = hdb5.with_device_column(dataset).copy()
    framed[hdb5.TOKAMAK_LABEL_COLUMN] = framed[hdb5.DEVICE_COLUMN]
    return framed


def _wrapped(estimator: Any) -> Pipeline:
    if isinstance(estimator, Pipeline):
        return estimator
    if getattr(estimator, "scales_internally", False):
        return Pipeline([("model", estimator)])
    return Pipeline([("scale", StandardScaler()), ("model", estimator)])


def constructive_models() -> dict[str, Pipeline]:
    """Every model the paper offers as a repair, in one dictionary.

    The three Gaussian-process kernels, the four Connor--Taylor rungs including
    the unconstrained one they are measured against, and the three mean
    functions of the ablation. The tree ensembles and the plain power law come
    from the standard zoo and are not repeated here.
    """
    models: dict[str, Pipeline] = {}
    models.update(gp.build_gp_models())
    models.update({name: _wrapped(est) for name, est in dm.build_constrained_models().items()})
    models.update(
        {
            "mean_constant_rbf": _wrapped(mech.MeanPlusResidualGP(mean="constant")),
            "mean_powerlaw_rbf": _wrapped(mech.MeanPlusResidualGP(mean="powerlaw")),
            "mean_ipb98_rbf": _wrapped(mech.FixedLawMeanGP()),
        }
    )
    return models


def analyze(dataset: pd.DataFrame | None = None) -> dict[str, Any]:
    if dataset is None:
        dataset = hdb5.prepare_dataset()

    models = constructive_models()
    by_device = device_grouped(dataset)
    by_label = dataset

    per_device = hdb5.extrapolation_report(by_device, extra_models=models)
    per_label = hdb5.extrapolation_report(by_label, extra_models=models)

    device_summary = hdb5.summarize_leave_one_tokamak_out(per_device).set_index("model_name")
    label_summary = hdb5.summarize_leave_one_tokamak_out(per_label).set_index("model_name")

    baseline = "ridge_loglinear"
    devices = sorted(per_device["tokamak"].unique())

    def _scores(frame: pd.DataFrame, model: str) -> pd.Series:
        rows = frame[frame["model_name"] == model]
        return rows.set_index("tokamak")["rmsle"].astype(float)

    reference = _scores(per_device, baseline)
    comparison: dict[str, Any] = {}
    for name in list(models) + [baseline, "random_forest", "hist_gradient_boosting"]:
        if name not in device_summary.index:
            continue
        scores = _scores(per_device, name)
        difference = (scores - reference).dropna()
        comparison[name] = {
            "lodo_mean_rmsle": float(device_summary.loc[name, "mean_rmsle"]),
            "lolo_mean_rmsle": (float(label_summary.loc[name, "mean_rmsle"]) if name in label_summary.index else None),
            "n_devices": int(len(difference)),
            "n_worse_than_power_law": int((difference > 0).sum()),
            "mean_difference_vs_power_law": float(difference.mean()),
        }

    return {
        "n_devices": len(devices),
        "devices": devices,
        "baseline": baseline,
        "models": comparison,
        "dataset_sha256": hdb5.fingerprint_file(hdb5.default_hdb5_path()).sha256,
    }


def main() -> None:
    dataset = hdb5.prepare_dataset()
    payload = analyze(dataset)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([{"model_name": name, **values} for name, values in payload["models"].items()]).sort_values(
        "lodo_mean_rmsle"
    )
    write_dataframe_csv_atomic(RESULTS_DIR / "device_arm_per_device.csv", frame)
    write_json_strict(RESULTS_DIR / "device_arm.json", payload)

    print(f"\n{payload['n_devices']} physical devices: {', '.join(payload['devices'])}\n")
    print(f"{'model':26s} {'LOLO':>8s} {'LODO':>8s} {'vs power law':>14s} {'worse on':>10s}")
    for _, row in frame.iterrows():
        lolo = "n/a" if pd.isna(row["lolo_mean_rmsle"]) else f"{row['lolo_mean_rmsle']:.3f}"
        print(
            f"{row['model_name']:26s} {lolo:>8s} {row['lodo_mean_rmsle']:8.3f} "
            f"{row['mean_difference_vs_power_law']:+14.3f} "
            f"{int(row['n_worse_than_power_law']):5d} of {int(row['n_devices']):d}"
        )
    print(f"\nwrote {RESULTS_DIR / 'device_arm.json'}")


if __name__ == "__main__":
    main()
