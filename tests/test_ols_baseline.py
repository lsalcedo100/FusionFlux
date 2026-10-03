"""The committed least-squares comparison says what the paper says about it.

The script exists to answer one objection, that the reversal might depend on
the ridge penalty, so the checks are the claims the paper makes from its
output: the two power laws are tied, and each ensemble loses to the unpenalised
fit on the same units it loses to ridge on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parents[1] / "results" / "ols_baseline.json"


@pytest.fixture(scope="module")
def committed() -> dict:
    if not RESULTS.exists():
        pytest.skip("no results/ols_baseline.json; run `python3 analysis_ols_baseline.py`")
    return json.loads(RESULTS.read_text())


def test_the_two_power_laws_are_tied(committed: dict) -> None:
    for arm in ("by_label", "by_device"):
        assert abs(committed[arm]["ols_minus_ridge_mean"]) < 0.001


def test_both_ensembles_lose_to_the_unpenalised_fit_as_they_lose_to_ridge(committed: dict) -> None:
    for arm in ("by_label", "by_device"):
        for model in ("random_forest", "hist_gradient_boosting"):
            entry = committed[arm][model]
            assert entry["n_worse_than_ols"] == entry["n_worse_than_ridge"]
            assert entry["mean_gap_vs_ols"] > 0.1


def test_the_forest_loses_on_every_device(committed: dict) -> None:
    arm = committed["by_device"]
    assert arm["n_units"] == 11
    assert arm["random_forest"]["n_worse_than_ols"] == 11
