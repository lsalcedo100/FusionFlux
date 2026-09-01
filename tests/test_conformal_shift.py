"""Tests for Result 10: repairing the conformal collapse Result 7 measured.

Result 10 claims a repair, and a claimed repair is easier to fake than a claimed
failure: an interval can be made to cover anything at all simply by being wide.
So the tests here are arranged around the two ways this could look like a repair
without being one.

* **Leakage.** If the calibration set were built using rows from the test
  machine, coverage would improve for a reason that has nothing to do with
  exchangeability and would not transfer to a real device. The calibration is
  therefore checked to touch only training rows.

* **Width.** If the distance scaling merely inflated every interval uniformly,
  coverage would rise and nothing would have been learned. So the scaling is
  checked to be *selective*: the half-width must be a function of the row's
  distance rather than a constant, and it must widen with distance whenever the
  calibration residuals do. The sign is never assumed, because the scale is
  fitted rather than imposed.

The control arm matters as much as in Result 7: under a synthetic dataset with
no shift at all, the machine-level scheme must land near nominal rather than
overcovering, which is what shows the repair is calibrated rather than just
conservative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import conformal_shift as cshift
import hdb5


def _make_dataset(
    machines: dict[str, float] | None = None,
    n_per_machine: int = 150,
    seed: int = 23,
) -> pd.DataFrame:
    """An HDB5-shaped frame from an exact power law with log-normal noise."""
    machines = machines or {"A": 1.0, "B": 1.3, "C": 1.7, "D": 2.1, "E": 2.5, "F": 2.9}
    rng = np.random.default_rng(seed)
    frames = []
    for index, (machine, radius) in enumerate(machines.items()):
        n = n_per_machine
        ip = rng.uniform(0.5, 4.0, n)
        bt = rng.uniform(1.0, 5.0, n)
        nel = rng.uniform(2.0, 18.0, n)
        plth = rng.uniform(1.0, 20.0, n)
        rgeo = radius * rng.uniform(0.97, 1.03, n)
        eps = rng.uniform(0.25, 0.35, n)
        kappa = rng.uniform(1.2, 2.0, n)
        meff = rng.uniform(1.5, 2.5, n)
        tau = (
            0.0562
            * ip**0.93
            * bt**0.15
            * nel**0.41
            * plth**-0.69
            * rgeo**1.97
            * eps**0.58
            * kappa**0.78
            * meff**0.19
        ) * np.exp(rng.normal(0.0, 0.12, n))
        frames.append(
            pd.DataFrame(
                {
                    "TOK": machine,
                    "SHOT": rng.integers(index * 10_000, index * 10_000 + n // 4, n),
                    "TIME": rng.uniform(1.0, 5.0, n),
                    "TAUTH": tau,
                    "IP": ip,
                    "BT": bt,
                    "NEL": nel,
                    "PLTH": plth,
                    "RGEO": rgeo,
                    "DELTA1": rng.uniform(0.1, 0.5, n),
                    "KAPPAA": kappa,
                    "EPS": eps,
                    "MEFF": meff,
                }
            )
        )
    return hdb5.build_features(hdb5.map_to_canonical(pd.concat(frames, ignore_index=True)))


def _ridge() -> dict[str, Pipeline]:
    return {
        "ridge_loglinear": Pipeline(
            [("scale", StandardScaler()), ("model", Ridge(alpha=1.0, solver="svd"))]
        )
    }


# --- the distance measure ---------------------------------------------------


def test_mahalanobis_is_zero_at_the_training_mean_and_grows_outward() -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(size=(500, 3))
    mean = train.mean(axis=0, keepdims=True)
    distances = cshift.row_mahalanobis(train, np.vstack([mean, mean + 5.0]))
    assert distances[0] == pytest.approx(0.0, abs=1e-8)
    assert distances[1] > distances[0]


def test_mahalanobis_matches_the_euclidean_case_for_identity_covariance() -> None:
    """With unit, uncorrelated features the distance is just the Euclidean one."""
    rng = np.random.default_rng(1)
    train = rng.normal(size=(20_000, 2))
    query = np.array([[1.0, 0.0], [0.0, 2.0]]) + train.mean(axis=0)
    distances = cshift.row_mahalanobis(train, query)
    assert distances[0] == pytest.approx(1.0, rel=0.05)
    assert distances[1] == pytest.approx(2.0, rel=0.05)


# --- the distance scale -----------------------------------------------------


def test_distance_scale_recovers_a_known_exponential() -> None:
    distances = np.linspace(0.5, 4.0, 200)
    residuals = np.exp(-1.5 + 0.8 * distances)
    intercept, slope = cshift.fit_distance_scale(distances, residuals)
    assert intercept == pytest.approx(-1.5, abs=1e-8)
    assert slope == pytest.approx(0.8, abs=1e-8)


def test_distance_scale_falls_back_to_constant_when_it_cannot_fit() -> None:
    """Too few usable points must degrade to the unscaled method, not raise."""
    intercept, slope = cshift.fit_distance_scale(np.array([1.0]), np.array([0.5]))
    assert (intercept, slope) == (0.0, 0.0)
    assert cshift.distance_scale(np.array([9.0]), intercept, slope)[0] == pytest.approx(1.0)


def test_distance_scale_is_floored_so_an_interval_cannot_collapse() -> None:
    """A steeply negative slope must not drive the half-width to zero."""
    scale = cshift.distance_scale(np.array([50.0]), 0.0, -10.0)
    assert scale[0] == pytest.approx(cshift.MIN_DISTANCE_SCALE)


# --- no leakage -------------------------------------------------------------


def test_machine_cv_calibration_uses_only_training_rows() -> None:
    """The calibration must never see the machine being predicted.

    Checked by counting: the calibration set can hold at most as many scores as
    there are training rows, and holding out one machine at a time means it
    holds exactly the rows of the training machines that clear the row floor.
    """
    dataset = _make_dataset()
    labels = dataset[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()
    train_index = np.flatnonzero(labels != "F")
    calibration = cshift.machine_cv_calibration(
        dataset, _ridge()["ridge_loglinear"], train_index
    )
    assert calibration.absolute_residuals.size == train_index.size
    assert calibration.n_machines == 5


def test_machine_cv_calibration_needs_two_machines() -> None:
    dataset = _make_dataset(machines={"A": 1.0, "B": 1.5})
    labels = dataset[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()
    with pytest.raises(ValueError, match="at least two"):
        cshift.machine_cv_calibration(
            dataset, _ridge()["ridge_loglinear"], np.flatnonzero(labels == "A")
        )


# --- the control arm --------------------------------------------------------


def test_machine_cv_lands_near_nominal_when_there_is_no_shift() -> None:
    """The repair must be calibrated, not merely conservative.

    All six synthetic machines are drawn from the same generating law, so the
    machines really are exchangeable and machine-level calibration should hit
    its nominal level rather than overshoot it. A scheme that simply widened
    everything would sail past 90% here and fail this test.
    """
    dataset = _make_dataset()
    _, summary = cshift.coverage_leave_one_tokamak_out(
        dataset, _ridge(), methods=("machine_cv",), alpha=0.10
    )
    pooled = summary[
        (summary["scope"] == "__pooled__") & (summary["model_name"] == "ridge_loglinear")
    ]
    coverage = float(pooled["empirical_coverage"].iloc[0])
    assert 0.83 < coverage < 0.97


# --- the distance scaling is selective, not a blanket widening --------------


def test_distance_scaling_makes_width_a_function_of_distance() -> None:
    """Half-width must vary across rows and be determined by the distance.

    Result 7's specific complaint is that widths do not move out of
    distribution. Under the plain machine-level scheme every row shares one
    half-width, so this is the property that distinguishes the two repairs.

    The sign of the relationship is deliberately *not* asserted here. The scale
    is fitted on calibration residuals rather than assumed, and this synthetic
    dataset is drawn from a correctly specified power law with homoscedastic
    log-noise, so error genuinely does not grow with distance and the fitted
    slope is free to come out negative. That is the method behaving correctly.
    Whether the slope is positive on the real database is a measurement, and it
    is reported in ``results/conformal_shift.json`` rather than assumed here.
    """
    dataset = _make_dataset()
    labels = dataset[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()
    per_row = cshift.shifted_conformal_arm(
        dataset,
        train_index=np.flatnonzero(~np.isin(labels, ["E", "F"])),
        test_index=np.flatnonzero(np.isin(labels, ["E", "F"])),
        zoo=_ridge(),
        methods=("machine_cv", "machine_cv_distance"),
        include_ipb98_reference=False,
    )
    flat = per_row[per_row["method"] == "machine_cv"]
    scaled = per_row[per_row["method"] == "machine_cv_distance"]

    assert flat["half_width_log"].nunique() == 1
    assert scaled["half_width_log"].nunique() > 1
    correlation = np.corrcoef(
        scaled["distance"].to_numpy(dtype=float),
        scaled["half_width_log"].to_numpy(dtype=float),
    )[0, 1]
    assert abs(correlation) > 0.9


def test_widths_grow_with_distance_when_the_residuals_do() -> None:
    """The repaired property itself, on calibration data that exhibits it.

    ``fit_distance_scale`` is what turns "error grows with distance" into "the
    interval grows with distance", so this checks the two ends of that: given
    calibration residuals that grow, a more distant row must receive a strictly
    wider interval than a nearer one.
    """
    distances = np.linspace(0.5, 3.0, 300)
    growing = np.exp(-1.0 + 0.9 * distances)
    intercept, slope = cshift.fit_distance_scale(distances, growing)
    assert slope > 0.0

    calibration = cshift.CalibrationSet(growing, distances, n_machines=5)
    quantile = hdb5.split_conformal_half_width(
        calibration.scaled_scores(intercept, slope), alpha=0.10
    )
    near, far = cshift.distance_scale(np.array([1.0, 6.0]), intercept, slope) * quantile
    assert far > near


def test_split_method_is_delegated_unchanged() -> None:
    """The baseline arm must be Result 7's procedure, not a reimplementation."""
    dataset = _make_dataset()
    labels = dataset[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()
    train_index = np.flatnonzero(labels != "F")
    test_index = np.flatnonzero(labels == "F")
    mine = cshift.shifted_conformal_arm(
        dataset,
        train_index=train_index,
        test_index=test_index,
        zoo=_ridge(),
        methods=("split",),
        include_ipb98_reference=False,
    )
    theirs = hdb5._conformal_arm(
        dataset,
        train_index=train_index,
        test_index=test_index,
        zoo=_ridge(),
        feature_columns=hdb5.BLIND_FEATURE_COLUMNS,
        alpha=hdb5.DEFAULT_CONFORMAL_ALPHA,
        calibration_fraction=hdb5.DEFAULT_CALIBRATION_FRACTION,
        seed=hdb5.CONFORMAL_SEED,
        include_ipb98_reference=False,
    )
    assert mine["covered"].tolist() == theirs["covered"].tolist()
    assert mine["half_width_log"].iloc[0] == pytest.approx(theirs["half_width_log"].iloc[0])


def test_unknown_method_is_rejected() -> None:
    dataset = _make_dataset()
    labels = dataset[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()
    with pytest.raises(ValueError, match="Unknown methods"):
        cshift.shifted_conformal_arm(
            dataset,
            train_index=np.flatnonzero(labels != "F"),
            test_index=np.flatnonzero(labels == "F"),
            zoo=_ridge(),
            methods=("bootstrap",),
        )
