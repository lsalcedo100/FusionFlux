"""Tests for Result 13, the replication on a scaling law from another science.

This result has two ways to be silently wrong and neither raises.

The first is parsing. The deposit is tab separated with **carriage-return** line
endings and marks missing measurements with ``-9999`` rather than leaving the
field empty. A default ``read_csv`` returns one row of 623 concatenated records
instead of failing, and any check that tests for nulls passes the sentinel
straight through into the logs, where ``log(-9999)`` is a NaN that then
propagates into a score rather than into an error. Both are checked directly.

The second is the baseline. Kleiber's law is used here with its exponent fixed
at 3/4 and its intercept fitted, and the whole comparison is meaningless if the
exponent is not actually pinned, since the "constrained" model would then just
be the unconstrained one under another name. That is asserted rather than
assumed.

Tests needing the download skip without it, as the HDB5 ones do.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import allometry as al
import hdb5

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _dataset_or_skip() -> pd.DataFrame:
    if not al.default_allometry_path().exists():
        pytest.skip(
            "Allometry dataset not downloaded; run "
            "`python3 -c 'import allometry; allometry.download_allometry()'`."
        )
    return al.prepare_dataset()


# --- provenance -------------------------------------------------------------
def test_pinned_digest_matches_the_file_on_disk() -> None:
    if not al.default_allometry_path().exists():
        pytest.skip("Allometry dataset not downloaded.")
    fingerprint = al.verify_allometry_file(al.default_allometry_path())
    assert fingerprint.sha256 == al.ALLOMETRY_SHA256


def test_verification_rejects_a_different_file(tmp_path: Path) -> None:
    """A gate that accepts anything converts unverified data into verified data."""
    impostor = tmp_path / "not-the-deposit.txt"
    impostor.write_text("Order\tSpecies\n")
    with pytest.raises(hdb5.DatasetIntegrityError, match="integrity check failed"):
        al.verify_allometry_file(impostor)


def test_a_missing_file_says_how_to_get_it() -> None:
    with pytest.raises(FileNotFoundError, match="download_allometry"):
        al.load_allometry_raw("/nonexistent/allometry.txt")


# --- the two parsing traps --------------------------------------------------
def test_the_file_parses_into_many_rows_not_one() -> None:
    """Carriage-return line endings: a default read yields a single row.

    623 records collapsed into one is not an error, it is a DataFrame, and every
    number downstream would be computed from it.
    """
    raw = al.load_allometry_raw() if al.default_allometry_path().exists() else None
    if raw is None:
        pytest.skip("Allometry dataset not downloaded.")
    assert len(raw) > 500, "the file parsed into too few rows; check the line terminator"
    assert "Order" in raw.columns, "column names still carry whitespace"


def test_the_missing_sentinel_never_reaches_the_logs() -> None:
    """-9999 is a valid float, so it survives any null check."""
    dataset = _dataset_or_skip()
    assert (dataset[al.TARGET_COLUMN] > 0).all()
    assert (dataset[al.MASS_COLUMN] > 0).all()
    assert not (dataset[al.TARGET_COLUMN] == al.MISSING_SENTINEL).any()
    assert np.isfinite(dataset["log_bmr"]).all()
    assert np.isfinite(dataset["log_mass_g"]).all()


def test_cleaning_drops_the_sentinel_rows_it_should() -> None:
    """The raw file has rows with a BMR sentinel; they must not survive."""
    if not al.default_allometry_path().exists():
        pytest.skip("Allometry dataset not downloaded.")
    raw = al.load_allometry_raw()
    sentinels = pd.to_numeric(raw["BMR (mlO2/hour)"], errors="coerce") == al.MISSING_SENTINEL
    assert sentinels.any(), "the fixture assumption is wrong; there are no sentinel rows"
    assert len(al.prepare_dataset()) <= len(raw) - int(sentinels.sum())


def test_field_metabolic_rate_is_not_pooled_into_the_target() -> None:
    """FMR is a different measurement at a different mass; pooling would hide that."""
    dataset = _dataset_or_skip()
    assert "FMR (kJ/day)" not in dataset.columns
    assert set(dataset.columns) >= {al.GROUP_COLUMN, "species", al.TARGET_COLUMN, al.MASS_COLUMN}


# --- the baseline -----------------------------------------------------------
def test_kleiber_exponent_is_pinned_at_three_quarters() -> None:
    """If the exponent is not fixed, the constrained model is not constrained."""
    rng = np.random.default_rng(0)
    log_mass = rng.normal(3.0, 1.5, size=200)
    # Generate with a clearly different exponent; the fit must not chase it.
    log_bmr = 0.55 * log_mass + 2.0 + rng.normal(0.0, 0.1, size=200)
    fitted = al.fit_kleiber(log_mass, log_bmr)
    assert fitted.exponent == pytest.approx(0.75)


def test_kleiber_intercept_is_least_squares_for_the_fixed_exponent() -> None:
    rng = np.random.default_rng(1)
    log_mass = rng.normal(3.0, 1.5, size=200)
    log_bmr = 0.75 * log_mass + 1.4 + rng.normal(0.0, 0.05, size=200)
    fitted = al.fit_kleiber(log_mass, log_bmr)
    assert fitted.log_coefficient == pytest.approx(1.4, abs=0.02)
    # The residual mean is the least-squares intercept when the slope is fixed.
    predicted = fitted.predict_log(log_mass)
    assert float(np.mean(log_bmr - predicted)) == pytest.approx(0.0, abs=1e-12)


def test_the_free_refit_disagrees_with_the_published_exponent() -> None:
    """The analogue of Result 2: the data does not reproduce the published law.

    If these ever agreed, the constraint would be free and Result 13 would have
    nothing to measure.
    """
    dataset = _dataset_or_skip()
    slope = float(np.polyfit(dataset["log_mass_g"], dataset["log_bmr"], 1)[0])
    assert 0.6 < slope < 0.73
    assert abs(slope - al.KLEIBER_EXPONENT) > 0.02


# --- the groups -------------------------------------------------------------
def test_eligible_orders_respects_the_row_floor() -> None:
    dataset = _dataset_or_skip()
    orders = al.eligible_orders(dataset)
    counts = dataset.groupby(al.GROUP_COLUMN).size()
    assert orders, "no order cleared the floor"
    for order in orders:
        assert counts[order] >= al.MIN_HELD_OUT_ROWS
    dropped = set(counts.index) - set(orders)
    for order in dropped:
        assert counts[order] < al.MIN_HELD_OUT_ROWS


def test_orders_are_returned_lightest_first() -> None:
    """The ordered split reads this sequence, so its direction is load-bearing."""
    dataset = _dataset_or_skip()
    orders = al.eligible_orders(dataset)
    medians = al.order_mass_medians(dataset)
    values = [medians[order] for order in orders]
    assert values == sorted(values)


def test_there_is_a_real_mass_axis_to_extrapolate_along() -> None:
    """Without a wide spread of group masses the size analogue is not available."""
    dataset = _dataset_or_skip()
    medians = [al.order_mass_medians(dataset)[o] for o in al.eligible_orders(dataset)]
    assert max(medians) / min(medians) > 100


# --- the reported result ----------------------------------------------------
@pytest.fixture(scope="module")
def artifact() -> dict:
    path = RESULTS / "allometry.json"
    if not path.exists():
        pytest.skip("run `python3 analysis_allometry.py`")
    return json.loads(path.read_text())


def test_the_artifact_records_the_pin_it_was_computed_from(artifact: dict) -> None:
    assert artifact["dataset_sha256"] == al.ALLOMETRY_SHA256


def test_the_trees_lose_to_the_power_laws_out_of_distribution(artifact: dict) -> None:
    """The half of the finding that does reproduce."""
    scores = artifact["scores"]
    for tree in ("random_forest", "hist_gradient_boosting"):
        assert scores[tree]["loo_mean_rmsle"] > scores["kleiber"]["loo_mean_rmsle"]
        assert scores[tree]["mass_cut_rmsle"] > scores["kleiber"]["mass_cut_rmsle"]
    assert artifact["sweep_wins"]["power_laws_beat_trees"] == artifact["sweep_wins"]["n_cuts"]


def test_the_reversal_does_not_reproduce_and_the_reason_is_recorded(artifact: dict) -> None:
    """The half that does not, which is the more informative half.

    The premise of Results 4 and 11 is that the flexible model wins the easy
    split first. With a single predictor it does not, so there is no inflated
    margin to invert. This test exists so that the prose claim and the artifact
    cannot drift apart: if a future rerun made the trees win cross-validation,
    the write-up would need rewriting and this fails until it is.
    """
    assert artifact["trees_win_cv"] is False
    assert artifact["ranking_reversed"] is False
    scores = artifact["scores"]
    best_power_law_cv = min(scores["kleiber"]["cv_rmsle"], scores["ols_loglinear"]["cv_rmsle"])
    for tree in ("random_forest", "hist_gradient_boosting"):
        assert scores[tree]["cv_rmsle"] > best_power_law_cv


def test_the_constraint_costs_in_sample_and_pays_at_the_widest_cut(artifact: dict) -> None:
    """The Result 8 pattern, reproduced: pay under CV, win out of distribution."""
    kleiber = artifact["scores"]["kleiber"]
    free = artifact["scores"]["ols_loglinear"]
    assert kleiber["cv_rmsle"] > free["cv_rmsle"], "the constraint should cost something"
    assert kleiber["mass_cut_rmsle"] < free["mass_cut_rmsle"], "and buy something out there"


def test_the_constraint_record_is_reported_as_mixed(artifact: dict) -> None:
    """It wins at some cuts and not others, and the write-up says so.

    Guards against the write-up being tightened into "wins everywhere" by a
    later edit, which is the direction prose drifts.
    """
    wins = artifact["sweep_wins"]
    assert 0 < wins["kleiber_beats_free_power_law"] < wins["n_cuts"]


def test_the_distance_mechanism_points_the_same_way(artifact: dict) -> None:
    """Trees should track extrapolation distance more strongly than power laws."""
    scores = artifact["scores"]
    trees = min(
        scores["random_forest"]["distance_spearman"],
        scores["hist_gradient_boosting"]["distance_spearman"],
    )
    laws = max(
        scores["kleiber"]["distance_spearman"],
        scores["ols_loglinear"]["distance_spearman"],
    )
    assert trees > laws
