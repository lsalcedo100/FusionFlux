"""What each model says about three machines with no data yet. Result 12.

Run ``python3 analysis_forecast.py`` to regenerate ``results/forecast.json``,
the locked record of what this repository predicts for SPARC, JT-60SA and ITER.

Everything else here is retrospective, and retrospection has a well-known
failure mode: a method that has only ever been scored against answers already in
the file has never been exposed to the one situation it exists for. So this
script writes the predictions down, with intervals, before the answers exist.

Why this is worth a result of its own
-------------------------------------
It converts Result 4c from a property into a checkable claim. A tree ensemble
predicts by averaging training targets, so its output is bounded above by the
largest target it was trained on, which in HDB5 STD5 is 1.321 s. IPB98(y,2) puts
ITER near 3.6 s. **The random forest that beats the published law by 41% under
cross-validation therefore cannot reach the physics prediction whatever it is
asked**, and what it actually returns is 0.435 s, lower by a factor of 8. This
script records that number rather than describing it.

Two of the three machines will settle it. JT-60SA is operating now and sits
inside the database's size range, so it is the nearest thing to a fair test and
the first likely to be checkable. SPARC is a field and density extrapolation
rather than a size one. ITER is the machine the whole field's scaling laws exist
to predict, and the one where the models disagree most.

On intervals
------------
The intervals come from Result 10's machine-level calibration with distance
scaling, not from plain split conformal. Publishing a split-conformal interval
here would mean publishing an interval Result 7 already measured covering 3% of
rows in exactly this situation. The scheme used instead is the best-calibrated
one this repository has out of distribution and is still not guaranteed at these
distances, which the printed output says rather than implies.

The narrative built on these numbers is in ``results/RESULTS.md``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import dimensional as dm
import forecast as fc
import hdb5
from storage import write_dataframe_csv_atomic, write_json_strict

RESULTS_DIR = Path(__file__).resolve().parent / "results"

MODEL_LABELS: dict[str, str] = {
    "ipb98y2_analytic": "IPB98(y,2), analytic",
    "powerlaw_collisionless": "power law, collisionless",
    "ridge_loglinear": "ridge, log-linear",
    "random_forest": "random forest",
    "hist_gradient_boosting": "hist gradient boosting",
}

# Order the table is printed in: the physics baseline, then the model Result 8
# selected, then the unconstrained law, then the two bounded families.
PRINT_ORDER: tuple[str, ...] = tuple(MODEL_LABELS)


def build_zoo() -> dict[str, object]:
    zoo = hdb5.build_model_zoo()
    del zoo["mean_baseline"]
    zoo.update(dm.build_constrained_models(("collisionless",)))
    return zoo


def main() -> None:
    dataset = hdb5.prepare_dataset()
    record = fc.build_forecast(dataset, build_zoo())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row.to_json() for row in record.forecasts])
    write_dataframe_csv_atomic(RESULTS_DIR / "forecast.csv", frame)
    write_json_strict(RESULTS_DIR / "forecast.json", record.to_json())

    print("--- Result 12: a locked prediction for three machines without data ---")
    print(
        f"  fitted on all {record.n_training_rows} rows of the pinned STD5 revision\n"
        f"  generated {record.generated_on}, content digest "
        f"{record.content_digest()[:16]}...\n"
        f"  largest thermal confinement time anywhere in the training data: "
        f"{record.train_tau_max_s:.3f} s"
    )

    for device in fc.DEVICES:
        subset = frame[frame["device"] == device.name].set_index("model_name")
        distance = float(subset["feature_mahalanobis"].iloc[0])
        print(f"\n  {device.name}  (R = {device.r_m} m, {device.status})")
        print(f"    {device.source}")
        print(
            f"    Mahalanobis distance from the training rows: {distance:.1f}"
            f"   [the 13 held-out machines of Result 4 span roughly 1 to 6]"
        )
        print(
            f"    {'model':<28}{'tau_E (s)':>11}"
            f"{f'{record.nominal_coverage:.0%} interval (s)':>24}{'bounded':>9}"
        )
        for model in PRINT_ORDER:
            if model not in subset.index:
                continue
            row = subset.loc[model]
            marker = " " if bool(row["is_blind"]) else "*"
            bounded = "yes" if bool(row["bounded_by_training_range"]) else ""
            interval = (
                f"{float(row['tau_interval_low_s']):.2f} to "
                f"{float(row['tau_interval_high_s']):.2f}"
            )
            print(
                f"  {marker} {MODEL_LABELS[model]:<27}"
                f"{float(row['tau_predicted_s']):>11.3f}{interval:>24}{bounded:>9}"
            )
        if device.published_ipb98_tau_s is not None:
            predicted = float(subset.loc["ipb98y2_analytic", "tau_predicted_s"])
            deviation = abs(predicted / device.published_ipb98_tau_s - 1.0)
            print(
                f"    parameter check: IPB98(y,2) on these inputs gives {predicted:.3f} s "
                f"against {device.published_ipb98_tau_s:.2f} s quoted in the source "
                f"({deviation * 100:.1f}%)"
            )

    print("\n  * not a blind baseline")
    print(
        "  'bounded' marks a model whose prediction cannot exceed the largest training\n"
        f"  target ({record.train_tau_max_s:.3f} s) whatever the inputs, which is Result 4c"
    )

    # The claim this file exists to make checkable, stated as a number.
    iter_rows = frame[frame["device"] == "ITER"].set_index("model_name")
    analytic = float(iter_rows.loc["ipb98y2_analytic", "tau_predicted_s"])
    bounded = iter_rows[iter_rows["bounded_by_training_range"]]
    if not bounded.empty:
        worst = float(bounded["tau_predicted_s"].max())
        print(
            f"\n  On ITER the analytic law says {analytic:.2f} s. The best any bounded model\n"
            f"  returns is {worst:.2f} s, a factor of {analytic / worst:.1f} lower, and no "
            "input could raise it:\n  the bound is a property of the model class, not of "
            "the tuning or the features."
        )

    print(
        "\n  These intervals use Result 10's machine-level calibration with distance\n"
        "  scaling. That is the best-calibrated scheme measured here out of distribution\n"
        "  and it still carries no guarantee at these distances; Result 10 reports what\n"
        "  it does deliver, which is well short of nominal at the ITER-matched cut."
    )


if __name__ == "__main__":
    main()
