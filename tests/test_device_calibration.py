"""The device-grouped calibration is actually grouped by device.

``analysis_device_calibration`` exists because Result 10 calibrates on database
labels, which is not the same as calibrating on physical devices: with JET-ILW
as the target, JET is still in the calibration set. The whole value of the
script is that it removes that overlap, so the test that matters is the one
asserting the overlap is gone rather than one re-deriving its coverage numbers.

The committed report is checked against what the paper says about it, in the
same spirit as ``tests/test_reported_numbers.py``: if a rerun moves a number,
the claim the paper makes about that number should stop passing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import analysis_device_calibration as adc
import hdb5

RESULTS = Path(__file__).resolve().parents[1] / "results" / "device_calibration.json"

# The two physical tokamaks that contribute two wall-era labels each. These are
# the entire reason this analysis exists.
WALL_VARIANTS = {"JETILW": "JET", "AUGW": "AUG"}


def _dataset_or_skip():
    if not hdb5.default_hdb5_path().exists():
        pytest.skip("HDB5 STD5 not downloaded; run `python3 hdb5.py download`.")
    return hdb5.prepare_dataset()


@pytest.fixture(scope="module")
def committed() -> dict:
    if not RESULTS.exists():
        pytest.skip("no results/device_calibration.json; run `python3 analysis_device_calibration.py`")
    return json.loads(RESULTS.read_text())


# --- the collapse actually happens -----------------------------------------


def test_wall_variants_are_folded_onto_one_unit() -> None:
    dataset = _dataset_or_skip()
    devices = adc.device_grouped_dataset(dataset)
    units = set(devices[hdb5.TOKAMAK_LABEL_COLUMN])
    for variant, device in WALL_VARIANTS.items():
        assert variant not in units, f"{variant} survived the collapse"
        assert device in units


def test_the_collapse_moves_rows_rather_than_dropping_them() -> None:
    """A collapse that silently lost rows would look like a cleaner result."""
    dataset = _dataset_or_skip()
    devices = adc.device_grouped_dataset(dataset)
    assert len(devices) == len(dataset)
    for variant, device in WALL_VARIANTS.items():
        moved = int((dataset[hdb5.TOKAMAK_LABEL_COLUMN] == variant).sum())
        before = int((dataset[hdb5.TOKAMAK_LABEL_COLUMN] == device).sum())
        after = int((devices[hdb5.TOKAMAK_LABEL_COLUMN] == device).sum())
        assert after == before + moved


def test_there_are_fewer_units_after_collapsing() -> None:
    dataset = _dataset_or_skip()
    devices = adc.device_grouped_dataset(dataset)
    assert devices[hdb5.TOKAMAK_LABEL_COLUMN].nunique() == (
        dataset[hdb5.TOKAMAK_LABEL_COLUMN].nunique() - len(WALL_VARIANTS)
    )


# --- the committed report says what the paper says --------------------------


def test_committed_report_is_device_grouped(committed: dict) -> None:
    assert committed["n_labels"] == 18
    assert committed["n_devices"] == 16
    assert len(committed["eligible_devices"]) == 11
    for variant in WALL_VARIANTS:
        assert variant not in committed["eligible_devices"]


def _coverage(committed: dict, arm: str, model: str, method: str) -> float:
    for row in committed[arm]:
        if row["model_name"] == model and row["method"] == method:
            return float(row["empirical_coverage"])
    raise AssertionError(f"no {model}/{method} row in {arm}")


def test_the_paper_s_claims_about_this_table_still_hold(committed: dict) -> None:
    """Sec. 10 says three things about these numbers. Each is asserted here.

    Tolerances are wide because the point is the claim, not the digit: the paper
    rounds these to whole percent, and a rerun that moved one by a point would
    not change a sentence.
    """
    arm = "leave_one_device_out"

    # 1. Plain split conformal fails badly for both tree ensembles.
    assert _coverage(committed, arm, "random_forest", "split") < 0.50
    assert _coverage(committed, arm, "hist_gradient_boosting", "split") < 0.50

    # 2. Calibrating by device recovers most of it for the forest.
    assert _coverage(committed, arm, "random_forest", "machine_cv") > 0.75

    # 3. And noticeably less for the booster, which is the finding that
    #    qualifies the label-level result rather than confirming it.
    booster = _coverage(committed, arm, "hist_gradient_boosting", "machine_cv")
    forest = _coverage(committed, arm, "random_forest", "machine_cv")
    assert booster < forest, "the booster is supposed to recover less than the forest"
    assert booster < 0.85

    # 4. Both linear models sit near nominal under every scheme.
    for model in ("ridge_loglinear", "powerlaw_collisionless"):
        for method in ("machine_cv", "machine_cv_distance"):
            assert 0.85 < _coverage(committed, arm, model, method) < 0.99


def test_the_size_cut_is_still_not_repaired(committed: dict) -> None:
    """The section's point is that changing the unit cannot fix a size shift."""
    arm = "size_cut_coverage"
    for model in ("random_forest", "hist_gradient_boosting"):
        for method in ("machine_cv", "machine_cv_distance"):
            assert _coverage(committed, arm, model, method) < 0.70
