"""The headline comparison without the reconstructed minor radius.

The nine engineering features carry one exact dependency, because the minor
radius is reconstructed as a = epsilon R and handed to the models beside R
and epsilon. A log-linear model cannot be hurt by that: the column is a
linear combination of two others and the rank audit (Sec. S12 of the
supplement) shows the fit lives on the eight independent directions. A tree
can split on a, R and epsilon separately, so a reader can ask whether the
redundant column helps or hurts the ensembles. This scores the three primary
models on the eight independent features under every split the paper uses,
so that question has a generated answer.

Run ``python3 analysis_eight_features.py`` to regenerate
``results/eight_features.json``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import analysis_dimensionless as dl
import hdb5
from storage import write_json_strict

RESULTS_DIR = Path(__file__).resolve().parent / "results"

MINOR_RADIUS = "log_a_m"
EIGHT: tuple[str, ...] = tuple(c for c in hdb5.BLIND_FEATURE_COLUMNS if c != MINOR_RADIUS)
MODELS = ("ridge_loglinear", "random_forest", "hist_gradient_boosting")


def _size_cut(dataset, columns: tuple[str, ...]) -> dict[str, float]:
    split = hdb5.iter_matched_split(dataset, hdb5.size_ordered_splits(dataset))
    scored = hdb5.score_size_split(
        dataset, split, feature_columns=columns, include_ipb98_reference=False
    ).set_index("model_name")["rmsle"]
    return {name: float(scored.loc[name]) for name in MODELS if name in scored.index}


def analyze() -> dict[str, Any]:
    dataset = hdb5.prepare_dataset()
    assert MINOR_RADIUS in hdb5.BLIND_FEATURE_COLUMNS and len(EIGHT) == 8
    arms = {}
    for name, columns in (("nine_features", hdb5.BLIND_FEATURE_COLUMNS), ("eight_features", EIGHT)):
        arm = dl._reversal_arm(dataset, columns)
        arm["size_cut_rmsle"] = _size_cut(dataset, columns)
        arms[name] = arm
    return {
        "dataset_sha256": hdb5.fingerprint_file(hdb5.default_hdb5_path()).sha256,
        "dropped_feature": MINOR_RADIUS,
        "arms": arms,
    }


def main() -> None:
    report = analyze()
    RESULTS_DIR.mkdir(exist_ok=True)
    write_json_strict(RESULTS_DIR / "eight_features.json", report)
    for name, arm in report["arms"].items():
        cv, cut = arm["cv_rmsle"], arm["size_cut_rmsle"]
        print(
            f"{name:15s} CV forest {cv[dl.FOREST]:.3f} / booster {cv[dl.BOOSTER]:.3f} / power law {cv[dl.POWER_LAW]:.3f}; "
            f"forest worse on {arm['by_label']['n_forest_worse']}/{arm['by_label']['n_units']} labels "
            f"(gap {arm['by_label']['mean_difference']:+.3f}), "
            f"{arm['by_device']['n_forest_worse']}/{arm['by_device']['n_units']} devices "
            f"(gap {arm['by_device']['mean_difference']:+.3f}); size cut forest {cut[dl.FOREST]:.3f}, "
            f"booster {cut[dl.BOOSTER]:.3f}, power law {cut[dl.POWER_LAW]:.3f}"
        )
    print(f"Wrote {RESULTS_DIR / 'eight_features.json'}")


if __name__ == "__main__":
    main()
