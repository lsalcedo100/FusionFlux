"""A locked, falsifiable prediction for three machines that do not have data yet. Result 12.

Everything else in this repository is retrospective. It holds out a machine that
was built decades ago, predicts it, and scores the prediction against a number
already in the database. That is the only way to measure a method, but it leaves
the central claim untested in the one direction that matters: **a next-step
device is not a held-out row, and nobody has checked these models against one.**

This module writes down what each model says about three real machines, before
the answer is known, in a form that can be checked later and cannot be quietly
revised. Two of the three will produce the number inside a decade.

    SPARC       R = 1.85 m. Commissioning, D-T planned. Smaller than JT-60U,
                so this is *not* a size extrapolation at all: it is a field and
                density extrapolation, at 12.2 T against a database maximum of
                about 4 T. The models should do well here and the interest is in
                whether they do.

    JT-60SA     R = 2.96 m. Operating. Sits inside the database's size range,
                which makes it the nearest thing to a fair in-distribution test
                and the first of the three likely to be checkable.

    ITER        R = 6.2 m. The machine every scaling law in this field exists to
                predict, 1.82x beyond the largest row in the database, which is
                the jump Result 5 reproduced inside the data.

The Result 4c bound, made concrete
----------------------------------
The largest thermal confinement time anywhere in HDB5 STD5 is 1.321 s. A tree
ensemble predicts by averaging training targets, so **its output cannot exceed
that value**, for any input whatsoever. IPB98(y,2) says ITER will reach about
3.6 s. So the random forest that beats the published law by 41% under
cross-validation is not merely expected to be wrong about ITER: it is
arithmetically incapable of returning the physics answer, whatever it is asked,
and this file records the number it returns instead.

That is Result 4c stated as a prediction rather than as a property. It is also
the single easiest thing in this repository for a reader to check by hand.

What "locked" means here
------------------------
The output carries the date, the SHA-256 of the dataset the models were fitted
on, and a digest over the forecast rows themselves. Regenerating it after
changing a model changes the digest, so a later edit is visible rather than
silent. It does not and cannot stop anyone from rewriting the file; it makes a
rewrite leave a mark.

On the inputs
-------------
Every device parameter below is a published design value with its source
recorded beside it, and the forecast is conditional on those values being what
the machine actually runs at. They will not be exactly right: operating points
move, and ``p_loss_mw`` in particular is a derived quantity that depends on the
scenario. The check on the parameter set is that it reproduces the published
IPB98(y,2) prediction for each device, which
``tests/test_forecast.py`` asserts: SPARC to under 1% and ITER to under 4% of
the figures quoted in the design papers. A parameter set that could not
reproduce the published number would be predicting a different machine.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

import conformal_shift as cshift
import hdb5


@dataclass(frozen=True)
class Device:
    """One machine's design operating point, and where the numbers came from."""

    name: str
    ip_ma: float
    bt_t: float
    ne_line_1e19_m3: float
    p_loss_mw: float
    r_m: float
    minor_radius_m: float
    kappa: float
    m_eff_amu: float
    source: str
    # The IPB98(y,2) confinement time quoted in that source, for the parameter
    # check described in the module docstring. None where the source does not
    # quote one.
    published_ipb98_tau_s: float | None
    status: str

    @property
    def inverse_aspect_ratio(self) -> float:
        return self.minor_radius_m / self.r_m

    def to_frame_row(self) -> dict[str, float]:
        return {
            "ip_ma": self.ip_ma,
            "bt_t": self.bt_t,
            "ne_line_1e19_m3": self.ne_line_1e19_m3,
            "p_loss_mw": self.p_loss_mw,
            "r_m": self.r_m,
            "inverse_aspect_ratio": self.inverse_aspect_ratio,
            "kappa": self.kappa,
            "m_eff_amu": self.m_eff_amu,
            "a_m": self.minor_radius_m,
        }


# Ordered by major radius, which is also roughly the order in which the answers
# will become available.
DEVICES: tuple[Device, ...] = (
    Device(
        name="SPARC",
        ip_ma=8.7,
        bt_t=12.2,
        ne_line_1e19_m3=31.0,
        p_loss_mw=29.0,
        r_m=1.85,
        minor_radius_m=0.57,
        kappa=1.75,
        m_eff_amu=2.5,
        source="Creely et al., J. Plasma Phys. 86 865860502 (2020), V2 primary reference discharge",
        published_ipb98_tau_s=0.77,
        status="under construction / commissioning",
    ),
    Device(
        name="JT-60SA",
        ip_ma=5.5,
        bt_t=2.25,
        ne_line_1e19_m3=6.3,
        p_loss_mw=37.0,
        r_m=2.96,
        minor_radius_m=1.18,
        kappa=1.75,
        m_eff_amu=2.0,
        source="JT-60SA Research Plan, full-current inductive scenario",
        published_ipb98_tau_s=None,
        status="operating",
    ),
    Device(
        name="ITER",
        ip_ma=15.0,
        bt_t=5.3,
        ne_line_1e19_m3=10.0,
        p_loss_mw=87.0,
        r_m=6.2,
        minor_radius_m=2.0,
        kappa=1.7,
        m_eff_amu=2.5,
        source="ITER Physics Basis, Nucl. Fusion 39 2175 (1999), Q=10 inductive baseline",
        published_ipb98_tau_s=3.7,
        status="under construction",
    ),
)

# How close the parameter set has to reproduce the published IPB98(y,2) figure
# before it is considered to describe the same machine. Loose enough to absorb
# the scenario-dependence of ``p_loss_mw`` and the several definitions of
# elongation in circulation, tight enough that a wrong unit or a transposed
# digit fails.
PUBLISHED_TAU_TOLERANCE = 0.05


def device_frame(devices: tuple[Device, ...] = DEVICES) -> pd.DataFrame:
    """The devices as a feature frame the models can be handed directly."""
    frame = pd.DataFrame([device.to_frame_row() for device in devices])
    frame.index = pd.Index([device.name for device in devices], name="device")
    featured = hdb5.build_features(frame)
    return featured


@dataclass(frozen=True)
class DeviceForecast:
    """One model's prediction for one machine, with its interval and its caveats."""

    device: str
    model_name: str
    is_blind: bool
    tau_predicted_s: float
    tau_interval_low_s: float
    tau_interval_high_s: float
    nominal_coverage: float
    # Mahalanobis distance of this device from the training rows, in the same
    # units as Result 4b's per-machine distances.
    feature_mahalanobis: float
    # True when the model is a tree ensemble and the prediction is therefore
    # pinned under the largest target in the training data, whatever the
    # features say. This is Result 4c, evaluated on a specific machine.
    bounded_by_training_range: bool

    def to_json(self) -> dict[str, object]:
        return {
            "device": self.device,
            "model_name": self.model_name,
            "is_blind": self.is_blind,
            "tau_predicted_s": self.tau_predicted_s,
            "tau_interval_low_s": self.tau_interval_low_s,
            "tau_interval_high_s": self.tau_interval_high_s,
            "nominal_coverage": self.nominal_coverage,
            "feature_mahalanobis": self.feature_mahalanobis,
            "bounded_by_training_range": self.bounded_by_training_range,
        }


@dataclass(frozen=True)
class ForecastRecord:
    """The locked record: forecasts, provenance, and a digest over both."""

    generated_on: str
    dataset_sha256: str
    n_training_rows: int
    train_tau_max_s: float
    nominal_coverage: float
    forecasts: list[DeviceForecast] = field(default_factory=list)

    def content_digest(self) -> str:
        """SHA-256 over the forecast rows and the data they were fitted on.

        Covers the predictions and the provenance together, so neither can be
        changed without the digest moving. Serialized with sorted keys and fixed
        separators so the digest is a function of the values rather than of
        dictionary ordering.
        """
        payload = {
            "dataset_sha256": self.dataset_sha256,
            "n_training_rows": self.n_training_rows,
            "forecasts": [row.to_json() for row in self.forecasts],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def to_json(self) -> dict[str, object]:
        return {
            "generated_on": self.generated_on,
            "dataset_sha256": self.dataset_sha256,
            "n_training_rows": self.n_training_rows,
            "train_tau_max_s": self.train_tau_max_s,
            "nominal_coverage": self.nominal_coverage,
            "content_digest_sha256": self.content_digest(),
            "devices": [
                {
                    "name": device.name,
                    "status": device.status,
                    "source": device.source,
                    "r_m": device.r_m,
                    "published_ipb98_tau_s": device.published_ipb98_tau_s,
                    **device.to_frame_row(),
                }
                for device in DEVICES
            ],
            "forecasts": [row.to_json() for row in self.forecasts],
        }


def _is_tree_ensemble(estimator: Any) -> bool:
    """Whether every prediction this estimator makes is an average of training targets.

    Checked structurally rather than by name so a model added to the zoo later
    is classified correctly without anyone remembering to update a list.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

    steps = getattr(estimator, "steps", None)
    candidates = [step for _, step in steps] if steps else [estimator]
    return any(
        isinstance(step, (RandomForestRegressor, HistGradientBoostingRegressor))
        for step in candidates
    )


def build_forecast(
    dataset: pd.DataFrame,
    zoo: dict[str, Any],
    *,
    devices: tuple[Device, ...] = DEVICES,
    feature_columns: tuple[str, ...] = hdb5.BLIND_FEATURE_COLUMNS,
    alpha: float = hdb5.DEFAULT_CONFORMAL_ALPHA,
    dataset_sha256: str | None = None,
) -> ForecastRecord:
    """Fit every model on the whole database and predict each device, with intervals.

    The intervals come from Result 10's ``machine_cv_distance`` scheme rather
    than from plain split conformal. That is not a preference: Result 7 measured
    plain split-conformal intervals covering 3% of rows across the ITER-matched
    cut, so using them here would be publishing an interval already known not to
    hold in exactly this situation. The machine-level calibration with distance
    scaling is the best-calibrated scheme this repository has out of
    distribution, and it is still not guaranteed, which the reported numbers say
    plainly.
    """
    columns = list(feature_columns)
    features = dataset[columns]
    tau = dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float)
    log_tau = np.log(tau)
    train_index = np.arange(len(dataset))

    device_features = device_frame(devices)[columns]
    distances = cshift.row_mahalanobis(
        features.to_numpy(dtype=float), device_features.to_numpy(dtype=float)
    )

    forecasts: list[DeviceForecast] = []

    def _record(
        model_name: str,
        predicted_log: np.ndarray,
        half_widths: np.ndarray,
        blind: bool,
        bounded: bool,
    ) -> None:
        for index, device in enumerate(devices):
            forecasts.append(
                DeviceForecast(
                    device=device.name,
                    model_name=model_name,
                    is_blind=blind,
                    tau_predicted_s=float(np.exp(predicted_log[index])),
                    tau_interval_low_s=float(np.exp(predicted_log[index] - half_widths[index])),
                    tau_interval_high_s=float(np.exp(predicted_log[index] + half_widths[index])),
                    nominal_coverage=1.0 - alpha,
                    feature_mahalanobis=float(distances[index]),
                    bounded_by_training_range=bounded,
                )
            )

    # The analytic law, which needs no fit. Its calibration is its own residuals
    # on each held-out machine, matching how every fitted model below is
    # calibrated so the intervals are comparable rather than merely coexisting.
    analytic_log = np.log(hdb5.ipb98y2_tau_s(device_frame(devices)).to_numpy(dtype=float))
    analytic_train = np.log(dataset["ipb98y2_tau_s"].to_numpy(dtype=float))
    labels = dataset[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()
    machines = hdb5.eligible_tokamaks(dataset)
    residuals, machine_distances = [], []
    for machine in machines:
        held = np.flatnonzero(labels == machine)
        inner = np.flatnonzero(labels != machine)
        residuals.append(np.abs(log_tau[held] - analytic_train[held]))
        machine_distances.append(
            cshift.row_mahalanobis(
                features.iloc[inner].to_numpy(dtype=float),
                features.iloc[held].to_numpy(dtype=float),
            )
        )
    analytic_calibration = cshift.CalibrationSet(
        np.concatenate(residuals), np.concatenate(machine_distances), len(machines)
    )
    intercept, slope = cshift.fit_distance_scale(
        analytic_calibration.distances, analytic_calibration.absolute_residuals
    )
    quantile = hdb5.split_conformal_half_width(
        analytic_calibration.scaled_scores(intercept, slope), alpha
    )
    _record(
        "ipb98y2_analytic",
        analytic_log,
        quantile * cshift.distance_scale(distances, intercept, slope),
        False,
        False,
    )

    for name, estimator in zoo.items():
        calibration = cshift.machine_cv_calibration(
            dataset, estimator, train_index, feature_columns=feature_columns
        )
        intercept, slope = cshift.fit_distance_scale(
            calibration.distances, calibration.absolute_residuals
        )
        quantile = hdb5.split_conformal_half_width(
            calibration.scaled_scores(intercept, slope), alpha
        )
        model = hdb5.clone_pipeline(estimator)
        with hdb5._suppress_benign_matmul_warnings():
            hdb5.fit_pipeline(model, features, log_tau)
            predicted = model.predict(device_features)
        _record(
            name,
            np.asarray(predicted, dtype=float),
            quantile * cshift.distance_scale(distances, intercept, slope),
            True,
            _is_tree_ensemble(estimator),
        )

    return ForecastRecord(
        generated_on=date.today().isoformat(),
        dataset_sha256=dataset_sha256 or hdb5.HDB5_STD5_SHA256,
        n_training_rows=int(len(dataset)),
        train_tau_max_s=float(tau.max()),
        nominal_coverage=1.0 - alpha,
        forecasts=forecasts,
    )
