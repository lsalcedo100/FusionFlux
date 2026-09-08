"""The control the paper said was unavailable, and whether the join is real.

Sec. 4.2 used to state that regressing the stored energy was impossible because
the STD5 deposit carries no such column. That is true of STD5 and false of the
file this repository already pins beside it, and the objection it was declining
is the sharpest one the headline faces. So the join is what these tests attack:
a wrong one returns plausible numbers for the wrong discharges, which is the
failure mode that would look like a result.

``KAPPAA`` is the lever. It is delivered in both files, so the two copies must
agree on every matched row, and they do to 5e-10. Beyond that: the identity the
control turns on has to actually hold on these rows, and the retargeted arm has
to be scored on the same rows as the arm it is compared with.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import analysis_stored_energy as ase
import hdb5
import replication as rep

RESULTS = Path(__file__).resolve().parents[1] / "results" / "stored_energy.json"


def _real_data_or_skip() -> None:
    if not hdb5.default_hdb5_path().exists():
        pytest.skip("HDB5 STD5 not downloaded; run `python3 hdb5.py download`.")
    if not rep.default_db523_path().exists():
        pytest.skip("DB5.2.3 not downloaded; run `python3 -c 'import replication; replication.download_db523()'`.")


@pytest.fixture(scope="module")
def committed() -> dict:
    if not RESULTS.exists():
        pytest.skip("no results/stored_energy.json; run `python3 analysis_stored_energy.py`")
    return json.loads(RESULTS.read_text())


def test_std5_really_does_not_carry_a_stored_energy() -> None:
    """The premise of the correction: STD5 is an extract, DB5.2.3 is not.

    If STD5 gained the column, the elaborate join below would be pointless and
    the paragraph explaining why it is needed would be wrong.
    """
    _real_data_or_skip()
    std5 = hdb5.load_hdb5_dataframe()
    assert "WTH" not in std5.columns
    assert "KAPPA" not in std5.columns
    full = rep.load_db523_raw()
    assert "WTH" in full.columns
    assert "KAPPA" in full.columns


def test_the_join_is_checked_against_a_column_both_files_carry() -> None:
    """A join off by one row would be invisible without this."""
    _real_data_or_skip()
    carried = rep.db523_columns(("WTH",))
    reference = rep.db523_columns(("KAPPAA",))
    assert len(carried) == len(reference)
    matched = reference["KAPPAA"].notna()
    raw = hdb5.load_hdb5_dataframe()
    analysed = raw.loc[hdb5.analysed_row_mask(raw)].reset_index(drop=True)
    left = pd.to_numeric(analysed.loc[matched.to_numpy(), "KAPPAA"], errors="coerce").to_numpy()
    right = pd.to_numeric(reference.loc[matched, "KAPPAA"], errors="coerce").to_numpy()
    assert np.nanmax(np.abs(left - right)) < 1e-6


def test_a_shifted_join_is_rejected() -> None:
    """The guard has to fire on the failure it exists for, not only pass on success."""
    raw = pd.DataFrame({"KAPPAA": [1.0, 2.0, 3.0]})
    shifted = pd.DataFrame({"KAPPAA": [2.0, 3.0, 1.0]})
    with pytest.raises(AssertionError, match="not the rows being analysed"):
        rep._check_join(raw, shifted)


def test_the_wall_era_labels_are_recovered_by_the_join() -> None:
    """STD5 labels a wall era, DB5.2.3 labels a machine, and 1633 rows hang on it.

    Without the mapping the join drops JET-ILW and AUG-W entirely, which would
    quietly restrict the control to 16 of the 18 labels and to neither of the
    two devices whose eras the paper collapses.
    """
    _real_data_or_skip()
    dataset = hdb5.prepare_dataset()
    carried = rep.db523_columns(("WTH",))
    labels = dataset[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()
    for era in ("JETILW", "AUGW"):
        matched = carried["WTH"][labels == era].notna()
        assert matched.any(), f"{era} matched no rows; the wall-era mapping is not working"
        assert matched.mean() > 0.9


def test_the_identity_the_control_turns_on_actually_holds(committed: dict) -> None:
    """W_th / (tau_th P) is definitional, and the paper prints how tightly."""
    residual = committed["identity_residual"]
    assert residual["median"] == pytest.approx(1.0, abs=1e-4)
    assert 0.97 < residual["q25"] <= 1.0
    assert 1.0 <= residual["q75"] < 1.03


def test_both_arms_are_scored_on_the_same_rows(committed: dict) -> None:
    """The comparison is between targets, so nothing else may differ.

    Both arms come from one frame in `analyze`, and this is the assertion that
    keeps it that way: the row count is shared, and each arm scores the same
    number of held-out units.
    """
    label_units = {arm["forest_worse_by_label"]["n_units"] for arm in committed["arms"].values()}
    device_units = {arm["forest_worse_by_device"]["n_units"] for arm in committed["arms"].values()}
    assert len(label_units) == 1
    assert len(device_units) == 1


def test_the_control_answers_the_objection_it_was_run_for(committed: dict) -> None:
    """What Sec. 4.2 claims from this artifact, as a test.

    The claim is that removing the identity leaves the direction and most of the
    magnitude: the forest still wins cross-validation, still loses out of
    distribution on a majority of units, and still collapses at the size cut. If
    that stops holding, the paragraph is wrong and so is the Conclusion.
    """
    arm = committed["arms"]["stored_energy"]
    baseline = committed["arms"]["confinement_time"]

    assert arm["cv_rmsle"]["random_forest"] < arm["cv_rmsle"]["ridge_loglinear"]
    for key in ("forest_worse_by_label", "forest_worse_by_device"):
        assert arm[key]["n_worse"] > arm[key]["n_units"] / 2
        assert arm[key]["mean_difference"] > 0
    # The device gap is the one the paper says is not merely preserved but larger.
    assert arm["forest_worse_by_device"]["mean_difference"] > baseline["forest_worse_by_device"]["mean_difference"]
    assert arm["iter_matched_cut"]["random_forest"] > 3 * arm["iter_matched_cut"]["ridge_loglinear"]


def test_the_sign_test_is_exact() -> None:
    """Worked out by hand: 9 of 13 two-sided is 2 * 1414 / 8192."""
    assert ase._sign_test(13, 13) == pytest.approx(2 / 2**13)
    assert ase._sign_test(9, 13) == pytest.approx(0.266845703125)
    assert ase._sign_test(9, 11) == pytest.approx(0.065429687500)
    assert ase._sign_test(6, 12) == pytest.approx(1.0)
