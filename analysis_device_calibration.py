"""The same calibration over physical devices rather than database labels. Result 10b.

Run ``python3 analysis_device_calibration.py`` to regenerate
``results/device_calibration.json`` and its CSV.

Result 10 calibrates on held-out *labels*. That is not the same as calibrating
on held-out devices, because two physical tokamaks contribute two wall-era
labels each: with JET-ILW as the target, JET is still in the calibration set.
For a paper whose argument is that the held-out unit has to match the
deployment being claimed, that gap is exactly the one a reader should not have
to take on trust, so it is measured here instead.

Everything in ``conformal_shift`` keys off ``TOKAMAK_LABEL_COLUMN``, so
collapsing the wall variants onto the device and writing that back into the
label column makes training, calibration and scoring all use the device as the
unit. Nothing else changes: same schemes, same models, same alpha, same seed.

What comes out is not a confirmation. Plain split conformal is *worse* on a
genuinely unseen device than on an unseen label, which is what removing the
within-device overlap should do, and the machine-level repair recovers most of
that for the random forest but noticeably less for the gradient booster. The
label-level number therefore overstates how well that repair calibrates the
booster on a device it has never seen. Reporting that is the point of the
script.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

import conformal_shift as cs
import dimensional
import hdb5
from storage import write_json_strict

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# The four fitted models plus the historical reference, which is the same set
# Table 8 reports so the two tables can be read side by side.
MODELS = ("random_forest", "hist_gradient_boosting", "ridge_loglinear")
CONSTRAINED = "powerlaw_collisionless"


def device_grouped_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """The same rows, with the label column replaced by the physical device.

    Overwriting the label column rather than passing a group argument is
    deliberate: every split, calibration and scoring path in this repository
    reads that one column, so replacing it changes the unit everywhere at once
    and cannot leave one of them still working label-wise.
    """
    framed = hdb5.with_device_column(dataset).copy()
    framed[hdb5.TOKAMAK_LABEL_COLUMN] = framed[hdb5.DEVICE_COLUMN]
    return framed


def _zoo() -> dict[str, Any]:
    models = {name: model for name, model in hdb5._assemble_zoo().items() if name in MODELS}
    constrained = dimensional.build_constrained_models()
    if CONSTRAINED in constrained:
        models[CONSTRAINED] = constrained[CONSTRAINED]
    return models


def analyse(dataset: pd.DataFrame) -> dict[str, Any]:
    devices = device_grouped_dataset(dataset)
    zoo = _zoo()
    eligible = hdb5.eligible_tokamaks(devices)

    _, lomo = cs.coverage_leave_one_tokamak_out(devices, zoo)
    splits = hdb5.size_ordered_splits(devices)
    split = hdb5.iter_matched_split(devices, splits)
    _, cut = cs.coverage_size_split(devices, split, zoo)

    def _pooled(summary: pd.DataFrame) -> list[dict[str, Any]]:
        pooled = summary[summary["scope"] == "__pooled__"]
        return [
            {
                "model_name": str(row.model_name),
                "method": str(row.method),
                "empirical_coverage": float(row.empirical_coverage),
                "median_interval_factor": float(row.median_interval_factor),
                "n_rows": int(row.n_rows),
            }
            for row in pooled.itertuples()
        ]

    return {
        "dataset_sha256": hdb5.HDB5_STD5_SHA256,
        "n_rows": int(len(dataset)),
        "n_labels": int(dataset[hdb5.TOKAMAK_LABEL_COLUMN].nunique()),
        "n_devices": int(devices[hdb5.TOKAMAK_LABEL_COLUMN].nunique()),
        "eligible_devices": list(eligible),
        "nominal_coverage": 1.0 - hdb5.DEFAULT_CONFORMAL_ALPHA,
        "size_cut": {
            "n_train_units": len(split.train_machines),
            "n_test_units": len(split.test_machines),
            "size_ratio": float(split.size_ratio),
        },
        "leave_one_device_out": _pooled(lomo),
        "size_cut_coverage": _pooled(cut),
    }


def coverage_table(report: dict[str, Any], arm: str) -> pd.DataFrame:
    """Pooled coverage as the paper prints it: models down, schemes across."""
    frame = pd.DataFrame(report[arm])
    if frame.empty:
        return frame
    table = frame.pivot(
        index="model_name", columns="method", values=["empirical_coverage", "median_interval_factor"]
    )
    order = [*MODELS, CONSTRAINED]
    return table.reindex([m for m in order if m in table.index])


def main() -> None:
    dataset = hdb5.prepare_dataset()
    report = analyse(dataset)

    RESULTS_DIR.mkdir(exist_ok=True)
    rows = []
    for arm in ("leave_one_device_out", "size_cut_coverage"):
        for entry in report[arm]:
            rows.append({"arm": arm, **entry})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "device_calibration.csv", index=False)
    write_json_strict(RESULTS_DIR / "device_calibration.json", report)

    print(f"--- {report['n_labels']} labels collapse to {report['n_devices']} devices; "
          f"{len(report['eligible_devices'])} are scored ---")
    for arm, title in (
        ("leave_one_device_out", "leave-one-device-out"),
        ("size_cut_coverage", "size cut, device grouping"),
    ):
        table = coverage_table(report, arm)
        print(f"\n--- {title}: pooled coverage of nominal "
              f"{report['nominal_coverage']:.0%} intervals ---")
        if table.empty:
            print("  (nothing scored)")
            continue
        coverage = (table["empirical_coverage"] * 100).round(0).astype("Int64")
        width = table["median_interval_factor"].round(2)
        for model in coverage.index:
            cells = ", ".join(
                f"{method} {coverage.loc[model, method]}% ({width.loc[model, method]:.2f}x)"
                for method in coverage.columns
            )
            print(f"  {model:24} {cells}")
    print(f"\nWrote {RESULTS_DIR / 'device_calibration.json'}")


if __name__ == "__main__":
    main()
