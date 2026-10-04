"""The committed joint control says what the paper says about it.

The arm removes the loss-power identity and the device-identifying features at
once, with nothing derived from the target, and the paper reports that no
reversal survives it. These check that reading of the committed file, and
that the arm's two single controls reproduce the numbers their own scripts
report, so the three are scored on the same rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parents[1] / "results"


@pytest.fixture(scope="module")
def committed() -> dict:
    path = RESULTS / "joint_control.json"
    if not path.exists():
        pytest.skip("no results/joint_control.json; run `python3 analysis_joint_control.py`")
    return json.loads(path.read_text())


def test_the_joint_arm_uses_only_prospective_features(committed: dict) -> None:
    arm = committed["arms"]["stored_energy_within_device_features"]
    assert arm["target"] == "w_th_j"
    assert set(arm["features"]) == {"log_ip_ma", "log_bt_t", "log_ne_line_1e19_m3", "log_p_loss_mw"}


def test_the_forest_still_wins_cross_validation_but_no_reversal_survives(committed: dict) -> None:
    arm = committed["arms"]["stored_energy_within_device_features"]
    assert arm["cv_rmsle"]["random_forest"] < arm["cv_rmsle"]["ridge_loglinear"]
    for unit in ("by_label", "by_device"):
        n, worse = arm[unit]["n_units"], arm[unit]["n_forest_worse"]
        assert worse <= n // 2 + 1, f"{unit}: the forest loses on {worse} of {n}, which is a reversal"
        assert abs(arm[unit]["mean_difference"]) < 0.1


def test_the_single_controls_reproduce_their_own_scripts(committed: dict) -> None:
    stored = json.loads((RESULTS / "stored_energy.json").read_text())
    joint_stored = committed["arms"]["stored_energy_nine_features"]
    own = stored["arms"]["stored_energy"]
    assert joint_stored["by_device"]["n_forest_worse"] == own["forest_worse_by_device"]["n_worse"]
    assert abs(joint_stored["by_device"]["mean_difference"] - own["forest_worse_by_device"]["mean_difference"]) < 1e-9
    dimensionless = json.loads((RESULTS / "dimensionless.json").read_text())
    joint_within = committed["arms"]["confinement_time_within_device_features"]
    own_within = dimensionless["device_identity"]["arms"]["within_device_features_only"]
    assert joint_within["by_device"]["n_forest_worse"] == own_within["by_device"]["n_forest_worse"]
