"""Two ways the community fits that this paper does not. Result 4g.

Run ``python3 analysis_fitting_conventions.py`` to regenerate
``results/fitting_conventions.json`` and its CSV.

The reversal is measured on every row of STD5, fitted unweighted. Neither choice
matches how the ITPA scalings were produced, and both are the kind of thing a
referee asks about rather than a thing this paper argues for, so both are
measured here instead of defended.

    aspect ratio   IPB98(y,2) was fitted on a selection that excluded spherical
                   tokamaks. This database does not: MAST and NSTX are two of
                   the 13 scored labels and two of the four most distant, and
                   START sits below the row threshold. Sec. 6 controls for this
                   only at the size cut, where dropping them moves the forest
                   from 0.938 to 0.936. This arm drops them from training and
                   scoring alike and reruns the whole leave-one-label-out table.

    weighting      JET and ASDEX Upgrade supply 77% of rows, so an unweighted
                   fit is mostly a fit to those two devices. The Limitations
                   section reports an equal-label refit of the power law, which
                   makes it worse. Reporting that for the power law alone is
                   the wrong shape of control: it is the models the weighting
                   might rescue that need it, so this arm weights every model
                   the same way.

Neither arm is expected to overturn the reversal. They are here because a
control that is only run on the model it flatters is not a control.
"""

from __future__ import annotations

from math import comb
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import hdb5
from storage import write_json_strict

RESULTS_DIR = Path(__file__).resolve().parent / "results"

CONTENDERS = ("ridge_loglinear", "random_forest", "hist_gradient_boosting")
REFERENCE = "ipb98y2_analytic"


def _rmsle(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.log(predicted) - np.log(actual)) ** 2)))


def _exact_two_sided_p(n_wins: int, n_trials: int) -> float:
    if n_trials == 0:
        return float("nan")
    extreme = max(n_wins, n_trials - n_wins)
    tail = sum(comb(n_trials, k) for k in range(extreme, n_trials + 1))
    return float(min(1.0, 2.0 * tail / 2**n_trials))


def conventional_aspect_ratio_only(dataset: pd.DataFrame) -> pd.DataFrame:
    """Rows from labels whose median inverse aspect ratio is conventional.

    The cut is on the label's median rather than the row, so a label is either
    in or out and no machine is split across the boundary.
    """
    median = dataset.groupby(hdb5.TOKAMAK_LABEL_COLUMN)["inverse_aspect_ratio"].median()
    keep = median[median <= hdb5.MAX_CONVENTIONAL_INVERSE_ASPECT_RATIO].index
    return dataset[dataset[hdb5.TOKAMAK_LABEL_COLUMN].isin(keep)].reset_index(drop=True)


def equal_label_weights(labels: np.ndarray) -> np.ndarray:
    """One unit of influence per label, spread evenly over that label's rows.

    Normalised to sum to the row count so the weighted fit is on the same scale
    as the unweighted one and the two are comparable without rescaling.
    """
    counts = pd.Series(labels).value_counts()
    weights = np.array([1.0 / counts[label] for label in labels], dtype=float)
    return weights * (len(labels) / weights.sum())


def leave_one_label_out(
    dataset: pd.DataFrame, *, weighted: bool = False
) -> pd.DataFrame:
    """Per-label log-RMSE, optionally fitting every model under equal-label weights."""
    features = dataset[list(hdb5.BLIND_FEATURE_COLUMNS)]
    tau = dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float)
    log_tau = np.log(tau)
    labels = dataset[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()
    reference = dataset["ipb98y2_tau_s"].to_numpy(dtype=float)
    zoo = hdb5._assemble_zoo()

    records: list[dict[str, Any]] = []
    for label in hdb5.eligible_tokamaks(dataset):
        held = np.flatnonzero(labels == label)
        train = np.flatnonzero(labels != label)
        records.append(
            {
                "tokamak": label,
                "model_name": REFERENCE,
                "n_held_out_rows": len(held),
                "rmsle": _rmsle(tau[held], reference[held]),
            }
        )
        weights = equal_label_weights(labels[train]) if weighted else None
        for name in CONTENDERS:
            model = hdb5.clone_pipeline(zoo[name])
            with hdb5._suppress_benign_matmul_warnings():
                if weights is None:
                    model.fit(features.iloc[train], log_tau[train])
                else:
                    # Every pipeline in the zoo ends in a step called "model",
                    # and every estimator behind it takes sample_weight, so the
                    # weights reach the fit rather than being silently dropped.
                    model.fit(
                        features.iloc[train], log_tau[train], model__sample_weight=weights
                    )
                predicted = np.exp(model.predict(features.iloc[held]))
            records.append(
                {
                    "tokamak": label,
                    "model_name": name,
                    "n_held_out_rows": len(held),
                    "rmsle": _rmsle(tau[held], predicted),
                }
            )
    return pd.DataFrame(records)


def summarise(per_label: pd.DataFrame, arm: str) -> dict[str, Any]:
    wide = per_label.pivot(index="tokamak", columns="model_name", values="rmsle")
    gap = wide["random_forest"] - wide["ridge_loglinear"]
    worse = int((gap > 0).sum())
    return {
        "arm": arm,
        "n_labels_scored": int(len(wide)),
        "labels": list(wide.index),
        "mean_rmsle": {name: float(wide[name].mean()) for name in wide.columns},
        "n_forest_worse_than_power_law": worse,
        "mean_gap_forest_minus_power_law": float(gap.mean()),
        "exact_two_sided_p": _exact_two_sided_p(worse, len(wide)),
        "reversal_holds": bool(worse == len(wide)),
    }


def build_report(dataset: pd.DataFrame) -> dict[str, Any]:
    conventional = conventional_aspect_ratio_only(dataset)
    dropped = sorted(
        set(dataset[hdb5.TOKAMAK_LABEL_COLUMN]) - set(conventional[hdb5.TOKAMAK_LABEL_COLUMN])
    )
    arms = {
        "baseline": leave_one_label_out(dataset),
        "conventional_aspect_ratio": leave_one_label_out(conventional),
        "equal_label_weights": leave_one_label_out(dataset, weighted=True),
    }
    return {
        "dataset_sha256": hdb5.HDB5_STD5_SHA256,
        "n_rows": int(len(dataset)),
        "max_conventional_inverse_aspect_ratio": hdb5.MAX_CONVENTIONAL_INVERSE_ASPECT_RATIO,
        "spherical_labels_dropped": dropped,
        "n_rows_conventional": int(len(conventional)),
        "summaries": {arm: summarise(frame, arm) for arm, frame in arms.items()},
        "per_label": {arm: frame.to_dict(orient="records") for arm, frame in arms.items()},
    }


def main() -> None:
    dataset = hdb5.prepare_dataset()
    report = build_report(dataset)

    RESULTS_DIR.mkdir(exist_ok=True)
    rows = [
        {"arm": arm, **record}
        for arm, records in report["per_label"].items()
        for record in records
    ]
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "fitting_conventions.csv", index=False)
    write_json_strict(RESULTS_DIR / "fitting_conventions.json", report)

    print(f"dropped as spherical: {report['spherical_labels_dropped']} "
          f"({report['n_rows'] - report['n_rows_conventional']} rows)")
    for arm, summary in report["summaries"].items():
        print(f"\n--- {arm} ---")
        print(f"  labels scored: {summary['n_labels_scored']}")
        for name in (REFERENCE, *CONTENDERS):
            if name in summary["mean_rmsle"]:
                print(f"    {name:24} {summary['mean_rmsle'][name]:.3f}")
        print(f"  forest worse than power law on "
              f"{summary['n_forest_worse_than_power_law']}/{summary['n_labels_scored']}, "
              f"gap {summary['mean_gap_forest_minus_power_law']:+.3f}, "
              f"p={summary['exact_two_sided_p']:.1e}")
        print(f"  reversal holds: {summary['reversal_holds']}")
    print(f"\nWrote {RESULTS_DIR / 'fitting_conventions.json'}")


if __name__ == "__main__":
    main()
