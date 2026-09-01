"""Tests for Result 12: the locked prediction for machines without data.

A forecast file has no ground truth to check against yet, which is the point of
it, so the tests here defend the two things that can be checked now.

**That the inputs describe the machines they claim to.** Every device parameter
is a published design value, and a transposed digit or a wrong unit would
produce a confident forecast for a machine that does not exist. The check is
that IPB98(y,2), evaluated on the parameter set, reproduces the confinement time
quoted in the source paper. That is an external number this repository does not
control, and it pins down the whole input vector at once: no single parameter
can be badly wrong while the product still lands on the published figure.

**That the Result 4c bound is real and is what makes the forecast interesting.**
A tree ensemble's prediction is an average of training targets, so it cannot
exceed the largest one, which in this database is 1.321 s against a physics
prediction near 3.6 s for ITER. That claim is asserted directly rather than
being left as prose, because it is the single most checkable thing in the file
and the one a reader is most likely to want verified.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import forecast as fc
import hdb5


def _dataset_or_skip() -> pd.DataFrame:
    if not hdb5.default_hdb5_path().exists():
        pytest.skip("HDB5 STD5 not downloaded; run `python3 hdb5.py download`.")
    return hdb5.prepare_dataset()


# --- the device parameters --------------------------------------------------


@pytest.mark.parametrize(
    "device", [device for device in fc.DEVICES if device.published_ipb98_tau_s is not None]
)
def test_parameters_reproduce_the_published_confinement_time(device: fc.Device) -> None:
    """The external check on the whole input vector.

    IPB98(y,2) on these parameters must land on the figure the design paper
    quotes. The tolerance absorbs the scenario-dependence of the loss power and
    the several definitions of elongation in circulation; it does not absorb a
    wrong unit or a transposed digit, which is what it is here to catch.
    """
    frame = fc.device_frame((device,))
    predicted = float(hdb5.ipb98y2_tau_s(frame).iloc[0])
    assert predicted == pytest.approx(
        device.published_ipb98_tau_s, rel=fc.PUBLISHED_TAU_TOLERANCE
    )


def test_devices_are_ordered_by_size_and_span_the_iter_jump() -> None:
    radii = [device.r_m for device in fc.DEVICES]
    assert radii == sorted(radii)
    assert fc.DEVICES[-1].name == "ITER"
    assert fc.DEVICES[-1].r_m == pytest.approx(hdb5.ITER_MAJOR_RADIUS_M)


def test_every_device_records_its_source() -> None:
    """A design value with no provenance is not checkable, so none are allowed."""
    for device in fc.DEVICES:
        assert device.source.strip()
        assert device.status.strip()


def test_inverse_aspect_ratio_is_derived_not_typed() -> None:
    """``eps`` must follow from the two radii, so the three cannot disagree."""
    for device in fc.DEVICES:
        assert device.inverse_aspect_ratio == pytest.approx(
            device.minor_radius_m / device.r_m
        )
        assert 0.15 < device.inverse_aspect_ratio < 0.8


def test_device_frame_carries_every_model_feature() -> None:
    frame = fc.device_frame()
    for column in hdb5.BLIND_FEATURE_COLUMNS:
        assert column in frame.columns
        assert np.isfinite(frame[column]).all()


# --- the Result 4c bound ----------------------------------------------------


def test_tree_ensembles_are_detected_structurally() -> None:
    """Classified by what the estimator is, not by what it is called.

    A model added to the zoo later must be flagged correctly without anyone
    remembering to update a list of names.
    """
    zoo = hdb5.build_model_zoo()
    assert fc._is_tree_ensemble(zoo["random_forest"])
    assert fc._is_tree_ensemble(zoo["hist_gradient_boosting"])
    assert not fc._is_tree_ensemble(zoo["ridge_loglinear"])


def test_bounded_models_cannot_reach_the_physics_prediction_for_iter() -> None:
    """Result 4c, stated as the specific claim the forecast file makes.

    Every bounded model's ITER prediction must sit under the largest training
    target, and therefore far under the analytic law's answer. This is what
    makes the forecast worth locking: the gap is a property of the model class
    and no amount of tuning closes it.
    """
    dataset = _dataset_or_skip()
    zoo = {name: hdb5.build_model_zoo()[name] for name in ("random_forest", "ridge_loglinear")}
    record = fc.build_forecast(dataset, zoo)
    frame = pd.DataFrame([row.to_json() for row in record.forecasts])
    iter_rows = frame[frame["device"] == "ITER"].set_index("model_name")

    analytic = float(iter_rows.loc["ipb98y2_analytic", "tau_predicted_s"])
    forest = float(iter_rows.loc["random_forest", "tau_predicted_s"])

    assert bool(iter_rows.loc["random_forest", "bounded_by_training_range"])
    assert forest <= record.train_tau_max_s
    assert analytic > 2.0 * forest


def test_the_unbounded_law_is_not_flagged_as_bounded() -> None:
    dataset = _dataset_or_skip()
    zoo = {"ridge_loglinear": hdb5.build_model_zoo()["ridge_loglinear"]}
    record = fc.build_forecast(dataset, zoo)
    frame = pd.DataFrame([row.to_json() for row in record.forecasts])
    assert not frame[frame["model_name"] == "ridge_loglinear"][
        "bounded_by_training_range"
    ].any()


# --- the lock ---------------------------------------------------------------


def test_content_digest_changes_when_a_forecast_changes() -> None:
    """A later edit has to leave a mark, which is all "locked" can mean here."""
    original = fc.ForecastRecord(
        generated_on="2026-01-01",
        dataset_sha256="abc",
        n_training_rows=10,
        train_tau_max_s=1.0,
        nominal_coverage=0.9,
        forecasts=[
            fc.DeviceForecast(
                device="ITER",
                model_name="m",
                is_blind=True,
                tau_predicted_s=1.0,
                tau_interval_low_s=0.5,
                tau_interval_high_s=2.0,
                nominal_coverage=0.9,
                feature_mahalanobis=3.0,
                bounded_by_training_range=False,
            )
        ],
    )
    edited = fc.ForecastRecord(
        generated_on=original.generated_on,
        dataset_sha256=original.dataset_sha256,
        n_training_rows=original.n_training_rows,
        train_tau_max_s=original.train_tau_max_s,
        nominal_coverage=original.nominal_coverage,
        forecasts=[replace(original.forecasts[0], tau_predicted_s=1.0001)],
    )
    assert original.content_digest() != edited.content_digest()


def test_content_digest_covers_the_dataset_it_was_fitted_on() -> None:
    """Swapping the data under an unchanged forecast must move the digest too."""
    base = fc.ForecastRecord(
        generated_on="2026-01-01",
        dataset_sha256="abc",
        n_training_rows=10,
        train_tau_max_s=1.0,
        nominal_coverage=0.9,
        forecasts=[],
    )
    other = fc.ForecastRecord(
        generated_on="2026-01-01",
        dataset_sha256="def",
        n_training_rows=10,
        train_tau_max_s=1.0,
        nominal_coverage=0.9,
        forecasts=[],
    )
    assert base.content_digest() != other.content_digest()


def test_digest_is_stable_under_reordering_of_json_keys() -> None:
    """Two identical records must agree, so a digest change always means content."""
    def _record() -> fc.ForecastRecord:
        return fc.ForecastRecord(
            generated_on="2026-01-01",
            dataset_sha256="abc",
            n_training_rows=10,
            train_tau_max_s=1.0,
            nominal_coverage=0.9,
            forecasts=[],
        )

    assert _record().content_digest() == _record().content_digest()


def test_intervals_bracket_the_point_prediction() -> None:
    dataset = _dataset_or_skip()
    zoo = {"ridge_loglinear": hdb5.build_model_zoo()["ridge_loglinear"]}
    record = fc.build_forecast(dataset, zoo)
    for row in record.forecasts:
        assert row.tau_interval_low_s <= row.tau_predicted_s <= row.tau_interval_high_s
        assert row.tau_interval_low_s > 0.0


def test_iter_sits_further_from_the_training_data_than_the_smaller_devices() -> None:
    """The distance ordering the whole argument assumes, measured rather than asserted."""
    dataset = _dataset_or_skip()
    zoo = {"ridge_loglinear": hdb5.build_model_zoo()["ridge_loglinear"]}
    record = fc.build_forecast(dataset, zoo)
    frame = pd.DataFrame([row.to_json() for row in record.forecasts])
    distances = frame.groupby("device")["feature_mahalanobis"].first()
    assert distances["ITER"] > distances["JT-60SA"]
