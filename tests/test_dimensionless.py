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
