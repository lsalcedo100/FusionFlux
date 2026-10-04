"""The committed eight-feature ablation says what the paper says about it.

The reconstructed minor radius is an exact function of two other features, so
the question is whether the tree ensembles gain or lose from the redundant
column. The paper says the reversal is unchanged without it; this holds the
committed file to that.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parents[1] / "results" / "eight_features.json"


@pytest.fixture(scope="module")
def committed() -> dict:
    if not RESULTS.exists():
        pytest.skip("no results/eight_features.json; run `python3 analysis_eight_features.py`")
    return json.loads(RESULTS.read_text())


def test_the_dropped_column_is_the_minor_radius(committed: dict) -> None:
    assert committed["dropped_feature"] == "log_a_m"
    assert committed["arms"]["eight_features"]["n_features"] == 8
    assert "log_a_m" not in committed["arms"]["eight_features"]["features"]


def test_the_reversal_is_unchanged_without_it(committed: dict) -> None:
    eight, nine = committed["arms"]["eight_features"], committed["arms"]["nine_features"]
    for unit in ("by_label", "by_device"):
        assert eight[unit]["n_forest_worse"] == eight[unit]["n_units"] == nine[unit]["n_forest_worse"]
        assert abs(eight[unit]["mean_difference"] - nine[unit]["mean_difference"]) < 0.02
    assert abs(eight["cv_rmsle"]["random_forest"] - nine["cv_rmsle"]["random_forest"]) < 0.005
    assert eight["size_cut_rmsle"]["random_forest"] > 3 * eight["size_cut_rmsle"]["ridge_loglinear"]
