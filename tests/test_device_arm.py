"""The device arm scores the constructive models on the split the paper rests on.

The point of the module under test is that it changes the held-out unit and
nothing else, so these are mostly checks that nothing else moved: the same rows,
the same eligibility rule, the same estimators, and a device count that is the
label count with the two wall-variant pairs folded together.
"""

from __future__ import annotations

import pandas as pd
import pytest

import analysis_device_arm as arm
import hdb5


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    return hdb5.prepare_dataset()


def test_regrouping_folds_the_wall_variants_and_keeps_every_row(dataset: pd.DataFrame) -> None:
    regrouped = arm.device_grouped(dataset)
    assert len(regrouped) == len(dataset)
    labels = set(dataset[hdb5.TOKAMAK_LABEL_COLUMN])
    devices = set(regrouped[hdb5.TOKAMAK_LABEL_COLUMN])
    assert {"JET", "JETILW"} <= labels
    assert "JETILW" not in devices, "the ILW era should have folded onto JET"
    assert len(devices) == len(labels) - 2, "two wall-variant pairs collapse, nothing else"


def test_the_scored_units_are_the_eleven_devices(dataset: pd.DataFrame) -> None:
    eligible = hdb5.eligible_tokamaks(arm.device_grouped(dataset), min_rows=hdb5.MIN_HELD_OUT_ROWS)
    assert len(eligible) == 11


def test_every_model_the_paper_recommends_is_scored() -> None:
    """A repair added to the paper and not to this dictionary is scored nowhere."""
    models = set(arm.constructive_models())
    assert {"gp_rbf", "gp_linear", "gp_linear_rbf"} <= models
    assert {"powerlaw_free", "powerlaw_kadomtsev", "powerlaw_collisionless"} <= models
    assert {"mean_constant_rbf", "mean_powerlaw_rbf", "mean_ipb98_rbf"} <= models


def test_the_constant_mean_arm_reproduces_the_rbf_kernel(dataset: pd.DataFrame) -> None:
    """The control the mean-function ablation rests on, checked on this split too.

    A constant mean plus an RBF residual is the RBF process rewritten, so the
    two have to land on the same number. If they ever diverge, the ablation is
    not the decomposition it claims to be.
    """
    import json
    from pathlib import Path

    payload = json.loads((Path(arm.RESULTS_DIR) / "device_arm.json").read_text())
    rbf = payload["models"]["gp_rbf"]["lodo_mean_rmsle"]
    constant = payload["models"]["mean_constant_rbf"]["lodo_mean_rmsle"]
    assert rbf == pytest.approx(constant, abs=1e-6)
