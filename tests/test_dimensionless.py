"""Where the held-out machines sit in dimensionless space, and whether it matters.

``analysis_dimensionless`` answers Hall's objection to organising an
extrapolation audit by engineering variables, and it answers it in a way that
could have gone against the paper twice: the distance diagnostic of Sec. 4.1
could have turned out to be a shadow of a dimensionless one, and the
ITER-size-matched cut could have turned out to be a clean size boundary. The
first did not happen and the second did.

Since the whole thing rests on three group definitions and a temperature
recovered from stored energy, that is what these tests attack: the groups have
to scale the way ``dimensional.py`` derives its constraints from, and the
distance has to be the same construction as the engineering one or the two
correlations are not comparable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import analysis_dimensionless as ad
import dimensional
import hdb5

RESULTS = Path(__file__).resolve().parents[1] / "results" / "dimensionless.json"


@pytest.fixture(scope="module")
def committed() -> dict:
    if not RESULTS.exists():
        pytest.skip("no results/dimensionless.json; run `python3 analysis_dimensionless.py`")
    return json.loads(RESULTS.read_text())


def _framed():
    if not hdb5.default_hdb5_path().exists():
        pytest.skip("HDB5 STD5 not downloaded; run `python3 hdb5.py download`.")
    return ad.with_dimensionless_groups(hdb5.prepare_dataset())


# --- the groups scale the way the constraint derivation says they do --------
#
# `dimensional.DIMENSIONLESS_GROUPS` holds each group's exponents on the four
# independent scales (length, field, temperature, density). The columns built
# here have to obey the same exponents, or the placement is in different
# coordinates from the constraints and the two halves of the paper disagree
# about what rho* means.


@pytest.mark.parametrize("group", ["rho_star", "beta", "nu_star"])
def test_each_group_scales_as_the_constraint_derivation_requires(group: str) -> None:
    length, field, temperature, density = dimensional.DIMENSIONLESS_GROUPS[group]
    rng = np.random.default_rng(4)
    n = 64
    base = {
        "log_n": rng.normal(size=n),
        "log_b": rng.normal(size=n),
        "log_t": rng.normal(size=n),
        "log_a": rng.normal(size=n),
    }

    def build(scaled: dict) -> np.ndarray:
        if group == "rho_star":
            return 0.5 * scaled["log_t"] - scaled["log_b"] - scaled["log_a"]
        if group == "beta":
            return scaled["log_n"] + scaled["log_t"] - 2.0 * scaled["log_b"]
        return scaled["log_n"] + scaled["log_a"] - 2.0 * scaled["log_t"]

    # Scale each physical quantity by lam^(its own exponent) and the group must
    # move by lam^(the exponent dimensional.py records for it). Both sides are
    # computed independently: the left from the columns this module builds, the
    # right from the exponent vector the constraints are derived from.
    log_lam = 0.37
    scaled = {
        "log_n": base["log_n"] + density * log_lam,
        "log_b": base["log_b"] + field * log_lam,
        "log_t": base["log_t"] + temperature * log_lam,
        "log_a": base["log_a"] + length * log_lam,
    }
    moved = build(scaled) - build(base)
    from_columns = {
        "rho_star": 0.5 * temperature - field - length,
        "beta": density + temperature - 2 * field,
        "nu_star": density + length - 2 * temperature,
    }[group]
    assert moved == pytest.approx(np.full(n, from_columns * log_lam))


def test_the_temperature_is_recovered_from_the_stored_energy_and_not_a_feature() -> None:
    """T is not a delivered column; if it were, this module would be unnecessary."""
    dataset = hdb5.prepare_dataset() if hdb5.default_hdb5_path().exists() else None
    if dataset is None:
        pytest.skip("HDB5 STD5 not downloaded; run `python3 hdb5.py download`.")
    assert not any("temperature" in column for column in dataset.columns)
    framed = _framed()
    assert "log_temperature" in framed.columns
    assert np.isfinite(framed["log_temperature"]).all()


def test_the_distance_is_the_same_construction_as_the_engineering_one() -> None:
    """Otherwise the two correlations compare constructions, not coordinates."""
    rng = np.random.default_rng(11)
    train = rng.normal(size=(200, 4))
    held = rng.normal(size=(30, 4)) + 1.5
    assert ad._mahalanobis_of_mean(train, held) == pytest.approx(
        hdb5._mahalanobis_of_mean(train, held), rel=1e-12
    )


def test_the_distance_ignores_a_constant_offset_in_any_group() -> None:
    """Every group is defined up to a multiplicative constant, so this must hold.

    If a dropped constant could move the distance, the whole placement would
    depend on conventions the module deliberately does not fix.
    """
    rng = np.random.default_rng(3)
    train = rng.normal(size=(150, 3))
    held = rng.normal(size=(25, 3)) + 0.8
    shift = np.array([2.5, -1.0, 0.25])
    assert ad._mahalanobis_of_mean(train + shift, held + shift) == pytest.approx(
        ad._mahalanobis_of_mean(train, held), rel=1e-10
    )


def test_the_spearman_helper_matches_the_committed_engineering_correlation() -> None:
    """Pin the correlation helper against a number the repository already reports."""
    import pandas as pd

    path = Path(__file__).resolve().parents[1] / "results" / "extrapolation_per_machine.csv"
    if not path.exists():
        pytest.skip("no results/extrapolation_per_machine.csv")
    frame = pd.read_csv(path)
    forest = frame[frame["model_name"] == "random_forest"]
    computed = ad._spearman(
        forest["rmsle"].to_numpy(), forest["feature_mahalanobis"].to_numpy()
    )
    assert computed == pytest.approx(0.846, abs=5e-4)


# --- what the paper claims from the artifact --------------------------------


def test_the_diagnosis_survives_the_change_of_coordinates(committed: dict) -> None:
    """Sec. 4.1 says the reading translates. If it stops, the paragraph is wrong."""
    forest = committed["error_correlations"]["random_forest"]
    power_law = committed["error_correlations"]["ridge_loglinear"]
    assert forest["against_dimensionless"] > 0.6
    assert abs(forest["against_dimensionless"] - forest["against_engineering"]) < 0.1
    assert abs(power_law["against_dimensionless"]) < 0.15
    assert abs(power_law["against_engineering"]) < 0.15


def test_the_size_cut_is_reported_as_separated_in_dimensionless_space(committed: dict) -> None:
    """The half of this that goes against the paper, asserted so it cannot drift back.

    Sec. 5 now says the cut displaces rho* and nu* as well as size. That is a
    concession, and a concession is worth a test precisely because nothing else
    would notice it quietly reverting.
    """
    groups = committed["iter_matched_cut"]["groups"]
    assert groups["log_rho_star"]["iqr_overlap_fraction"] < 0.2
    assert groups["log_rho_star"]["fraction_of_a_training_sd"] > 0.8
    assert groups["log_nu_star"]["fraction_of_a_training_sd"] > 0.5
    # Beta is the one that does stay put, which is why it is named separately.
    assert groups["log_beta"]["iqr_overlap_fraction"] > 0.5


# --- does the reversal need a coordinate that names the machine? ------------
#
# This is the objection that would have explained Table 1 without any appeal to
# long-range behaviour, and the module answers it with three arms rather than
# one. The middle arm exists to be reported as inconclusive, so a test that only
# checked the conclusion would miss the point of it.


def test_the_engineering_features_really_do_name_the_machine(committed: dict) -> None:
    identity = committed["device_identity"]
    variance = identity["within_device_variance"]
    assert variance["log_r_m"] < 0.01
    assert variance["log_a_m"] < 0.05
    recovery = identity["device_recovery"]
    assert recovery["accuracy"] > 0.95
    assert recovery["accuracy"] > 2 * recovery["majority_baseline"]


def test_the_dimensionless_groups_do_not(committed: dict) -> None:
    """The premise of the third arm: these coordinates move inside a device."""
    variance = committed["device_identity"]["within_device_variance"]
    for group in ("log_rho_star", "log_beta", "log_nu_star", "log_q_cyl"):
        assert variance[group] > 0.2, group


def test_deleting_the_device_constant_features_is_inconclusive(committed: dict) -> None:
    """It kills the reversal, and it kills the power law's size dependence with it.

    Reported because it looks decisive and is not. The assertion is both halves:
    the reversal goes, and the power law gets much worse, which is why the arm
    cannot separate leakage from physics.
    """
    arms = committed["device_identity"]["arms"]
    trimmed = arms["within_device_features_only"]
    full = arms["nine_engineering_features"]
    assert trimmed["by_device"]["mean_difference"] < 0
    assert (
        trimmed["by_device"]["power_law_mean_rmsle"]
        > 2 * full["by_device"]["power_law_mean_rmsle"]
    )


def test_the_reversal_survives_in_dimensionless_coordinates(committed: dict) -> None:
    """The claim Sec. 4.2 makes, and the one the paper's headline now rests on.

    Direction and counts have to hold, and the margin is allowed to be smaller:
    the paper says explicitly that part of the engineering-space gap is device
    identity and that this is the conservative reading.
    """
    arms = committed["device_identity"]["arms"]
    groups = arms["dimensionless_groups"]
    full = arms["nine_engineering_features"]

    assert groups["cv_gain_of_forest"] > 0.2
    for unit in ("by_label", "by_device"):
        assert groups[unit]["mean_difference"] > 0
        assert groups[unit]["n_forest_worse"] > groups[unit]["n_units"] / 2
        assert groups[unit]["mean_difference"] < full[unit]["mean_difference"]


def test_the_dimensionless_target_is_the_dimensionless_confinement_time() -> None:
    """B*tau, not tau: regressing tau on dimensionless groups mixes the two."""
    if not hdb5.default_hdb5_path().exists():
        pytest.skip("HDB5 STD5 not downloaded; run `python3 hdb5.py download`.")
    framed = ad.with_dimensionless_regression_columns(_framed())
    expected = framed["bt_t"] * framed[hdb5.TARGET_COLUMN]
    assert framed[ad.DIMENSIONLESS_TARGET].to_numpy() == pytest.approx(expected.to_numpy())


def test_the_safety_factor_is_not_another_device_name() -> None:
    """q completes the group list and would be useless if it were device-constant."""
    if not hdb5.default_hdb5_path().exists():
        pytest.skip("HDB5 STD5 not downloaded; run `python3 hdb5.py download`.")
    framed = ad.with_dimensionless_regression_columns(_framed())
    share = ad.within_device_variance(framed, ("log_q_cyl",))["log_q_cyl"]
    assert share > 0.5
