"""Which tree ensembles are bounded by their training targets, and which merely are?

Run ``python3 analysis_boundedness.py`` to regenerate ``results/boundedness.json``.

The paper's mechanism for the ranking inversion is a bound: a model that cannot
emit a value above the largest target it was trained on cannot predict a machine
whose confinement times run above that value, however good its features are.
That bound is *structural* for a random forest and only *empirical* for a
gradient booster, and the distinction matters enough to measure rather than
assert.

    A random forest averages leaf values, and every leaf value is a mean of
    training targets, so every prediction it can emit lies in
    ``[min(y_train), max(y_train))``. No input reaches outside it.

    A gradient booster sums tree outputs onto an initial estimate. Nothing in
    that construction confines the sum to the training-target range, so the same
    guarantee is unavailable. Whether it *uses* the freedom is an empirical
    question about these rows, and this script answers it.

Scored on every split the paper's boundedness claims are made about: each
leave-one-tokamak-out fold, and the ITER-size-matched size cut.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

import hdb5
from storage import write_json_strict

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# The two ensembles whose boundedness the paper contrasts. Ridge is omitted
# deliberately: it is unbounded by construction and there is nothing to measure.
ENSEMBLES = ("random_forest", "hist_gradient_boosting")

# Guaranteed by the averaging construction, so a violation is a bug in this
# script or in the model, not a finding. Asserted rather than reported.
STRUCTURALLY_BOUNDED = ("random_forest",)


def _measure(
    dataset: pd.DataFrame,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    split: str,
    held_out: str,
    feature_columns: tuple[str, ...],
) -> list[dict[str, object]]:
    """Largest prediction each ensemble emits, against the training-target ceiling."""
    features = list(feature_columns)
    log_target = np.log(dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float))
    x_train = dataset.loc[train_mask, features]
    x_test = dataset.loc[test_mask, features]
    y_train = log_target[train_mask]
    y_test = log_target[test_mask]
    train_max = float(y_train.max())

    zoo = hdb5.build_model_zoo()
    rows: list[dict[str, object]] = []
    for name in ENSEMBLES:
        model = zoo[name]
        model.fit(x_train, y_train)
        predicted = np.asarray(model.predict(x_test), dtype=float)
        prediction_max = float(predicted.max())
        exceeds = prediction_max > train_max
        if name in STRUCTURALLY_BOUNDED and exceeds:
            raise AssertionError(f"{name} exceeded max(y_train) on {held_out}, which averaging forbids")
        rows.append(
            {
                "split": split,
                "held_out": held_out,
                "model_name": name,
                "structurally_bounded": name in STRUCTURALLY_BOUNDED,
                "n_train_rows": int(train_mask.sum()),
                "n_test_rows": int(test_mask.sum()),
                "log_train_target_max": train_max,
                "log_test_target_max": float(y_test.max()),
                "log_prediction_max": prediction_max,
                # Negative means the model stayed under the ceiling.
                "log_headroom_used": prediction_max - train_max,
                "fraction_predictions_above_train_max": float((predicted > train_max).mean()),
                # How many times higher the held-out machine's best shot is than
                # the largest number this model actually emitted for it.
                "best_shot_over_prediction_max": float(np.exp(y_test.max() - prediction_max)),
            }
        )
    return rows


def analyze_boundedness(
    dataset: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] = hdb5.BLIND_FEATURE_COLUMNS,
    min_rows: int = hdb5.MIN_HELD_OUT_ROWS,
) -> dict[str, object]:
    labels = dataset[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()
    counts = pd.Series(labels).value_counts()
    held_out_machines = sorted(str(name) for name, n in counts.items() if n >= min_rows)

    records: list[dict[str, object]] = []
    for tokamak in held_out_machines:
        test_mask = labels == tokamak
        records.extend(
            _measure(
                dataset,
                ~test_mask,
                test_mask,
                split="leave_one_machine_out",
                held_out=tokamak,
                feature_columns=feature_columns,
            )
        )

    splits = hdb5.size_ordered_splits(dataset)
    matched = hdb5.iter_matched_split(dataset, splits)
    test_mask = np.isin(labels, list(matched.test_machines))
    records.extend(
        _measure(
            dataset,
            ~test_mask,
            test_mask,
            split="iter_matched_cut",
            held_out="+".join(matched.test_machines),
            feature_columns=feature_columns,
        )
    )

    frame = pd.DataFrame(records)
    summary: dict[str, object] = {}
    for name in ENSEMBLES:
        subset = frame[frame["model_name"] == name]
        worst = subset.loc[subset["log_headroom_used"].idxmax()]
        summary[name] = {
            "structurally_bounded": bool(subset["structurally_bounded"].iloc[0]),
            "n_splits_scored": int(len(subset)),
            "n_splits_exceeding_train_max": int((subset["log_headroom_used"] > 0.0).sum()),
            # The closest any split came to the ceiling. Negative means it never
            # reached it on any split scored here.
            "closest_approach_to_train_max": float(worst["log_headroom_used"]),
            "closest_approach_split": str(worst["held_out"]),
        }

    return {
        "feature_columns": list(feature_columns),
        "n_rows": int(len(dataset)),
        "machines_held_out": held_out_machines,
        "iter_matched_cut": matched.to_json(),
        "summary": summary,
        "per_split": records,
    }


def main() -> None:
    dataset = hdb5.prepare_dataset()
    analysis = analyze_boundedness(dataset)
    write_json_strict(RESULTS_DIR / "boundedness.json", analysis)

    summary = cast("dict[str, dict[str, object]]", analysis["summary"])
    print("--- largest prediction against max(y_train), in logs ---")
    for name, stats in summary.items():
        guarantee = "structural" if stats["structurally_bounded"] else "not guaranteed"
        print(
            f"{name:24s} ({guarantee}): exceeded on "
            f"{stats['n_splits_exceeding_train_max']} of {stats['n_splits_scored']} splits, "
            f"closest approach {stats['closest_approach_to_train_max']:+.4f} "
            f"on {stats['closest_approach_split']}"
        )
    print(f"\nWrote {RESULTS_DIR / 'boundedness.json'}")


if __name__ == "__main__":
    main()
