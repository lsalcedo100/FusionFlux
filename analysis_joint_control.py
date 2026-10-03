"""Both objections at once: stored energy as the target, within-device features only.

The paper answers two objections to the reversal separately. The loss power is
the target's denominator, so the stored-energy arm retargets the models on
W_th and leaves P as an ordinary predictor. The engineering features nearly
name the machine, so the within-device arm keeps only the four features that
move inside a device (current, field, density, loss power). Each arm removes
one confound and leaves the other in place, and the dimensionless arm that
would combine them reconstructs its temperature from W_th and so is not blind
to the target.

This arm removes both at once with nothing derived from the target: the target
is log W_th and the features are the four within-device columns, every one of
them known before a machine runs. It is the harshest control in the paper,
since it deletes the size dependence the power law extrapolates through as
well as the identity, and it is reported for what it shows.

Run ``python3 analysis_joint_control.py`` to regenerate
``results/joint_control.json``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import analysis_dimensionless as dl
import analysis_stored_energy as se
import hdb5
from storage import write_json_strict

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def analyze() -> dict[str, Any]:
    framed = se.dataset_with_stored_energy()
    arms = {
        # The two single controls on the same 6214 rows, for a like-for-like
        # reading beside the joint one.
        "stored_energy_nine_features": dl._reversal_arm(
            framed, hdb5.BLIND_FEATURE_COLUMNS, target=se.STORED_ENERGY_COLUMN
        ),
        "confinement_time_within_device_features": dl._reversal_arm(
            framed, dl.WITHIN_DEVICE_FEATURES
        ),
        "stored_energy_within_device_features": dl._reversal_arm(
            framed, dl.WITHIN_DEVICE_FEATURES, target=se.STORED_ENERGY_COLUMN
        ),
    }
    return {
        "dataset_sha256": hdb5.fingerprint_file(hdb5.default_hdb5_path()).sha256,
        "n_rows": int(len(framed)),
        "within_device_features": list(dl.WITHIN_DEVICE_FEATURES),
        "arms": arms,
    }


def main() -> None:
    report = analyze()
    RESULTS_DIR.mkdir(exist_ok=True)
    write_json_strict(RESULTS_DIR / "joint_control.json", report)
    print(f"--- {report['n_rows']} rows with a stored energy ---")
    for name, arm in report["arms"].items():
        cv = arm["cv_rmsle"]
        print(
            f"{name:42s} CV forest {cv[dl.FOREST]:.3f} / power law {cv[dl.POWER_LAW]:.3f}; "
            f"forest worse on {arm['by_label']['n_forest_worse']}/{arm['by_label']['n_units']} labels "
            f"(gap {arm['by_label']['mean_difference']:+.3f}) and "
            f"{arm['by_device']['n_forest_worse']}/{arm['by_device']['n_units']} devices "
            f"(gap {arm['by_device']['mean_difference']:+.3f})"
        )
    print(f"Wrote {RESULTS_DIR / 'joint_control.json'}")


if __name__ == "__main__":
    main()
