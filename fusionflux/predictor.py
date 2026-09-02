"""A confinement predictor that knows when not to answer.

Every result in this repository says the same thing in a different way: a model
fitted on these 18 tokamaks is trustworthy on a nineteenth only to the extent
that the nineteenth resembles them, and the models that look best by
cross-validation are the ones that degrade worst when it does not. Result 4b
measures that degradation as a function of Mahalanobis distance, Result 4c shows
a tree ensemble cannot emit a value above its training range at all, Result 8
finds the constrained power law is the best blind model at the largest
extrapolation this database can pose, and Result 10 finds it is the only one
whose intervals survive there.

And then ``hdb5.predict_single_case`` ignores all of it. Hand it ITER's
parameters and it returns a random forest's point estimate, 0.435 s, with no
interval and no complaint, for a machine the same repository predicts at 3.6 s.
The study's central finding and the study's prediction path disagree, and the
prediction path is the one someone would actually call.

This module is that finding, made operational. :func:`predict` returns

* a point estimate from the model Result 8 selected,
* an interval from the calibration scheme Result 10 selected,
* the extrapolation distance that Result 4b showed predicts the error, and
* an explicit refusal when the query sits beyond anything this study validated.

The refusal is the product
--------------------------
Two conditions trigger it, and both are read off the study rather than chosen.

``beyond_validated_range``
    The query is farther from the training data than any of the 13 machines
    Result 4 actually held out and scored. Past that point the repository has no
    measurement of how any model behaves, so a number returned there is an
    extrapolation of an extrapolation.

``physics_exceeds_training_ceiling``
    The analytic law predicts a confinement time above the largest value in the
    training data. Result 4c is arithmetic: a tree ensemble averages training
    targets, so *no* range-bounded model can reach that answer, whatever the
    features say and whatever the tuning. This one is decidable before any model
    runs, from the inputs alone, and for ITER it fires: IPB98(y,2) gives 3.59 s
    against a training ceiling of 1.321 s.

Neither condition suppresses a number. Suppressing it would be its own kind of
dishonesty, since the caller may have good reasons this module cannot see. Both
attach a reason to every affected model and move the recommendation to one that
still holds, which is what a person reading the output actually needs.

Why a service card instead of a saved model
-------------------------------------------
Prediction here is arithmetic on nine coefficients, so the card
(``results/predictor.json``) carries the coefficients rather than a pickled
estimator: the constrained and ridge fits, the training feature mean and
pseudo-inverse covariance for the distance, the training ceiling, the validated
distance limit, and each model's conformal quantile and distance scale. It is a
few kilobytes of numbers, it is diffable, it is covered by the reproduce job like
every other artifact, and loading it cannot execute anything. A ``joblib``
artifact would be none of those.

The card carries the log-linear models only. Tree ensembles are the models this
whole document recommends against extrapolating with, and in distribution
``hdb5.predict_single_case`` already serves them; what this module owes the
caller about a tree is not its number but the fact that it is bounded, which
``training_ceiling_s`` states exactly.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# The card is resolved rather than fixed, because this module is read from two
# places that disagree about where it lives.
#
# In a checkout, ``results/predictor.json`` is the single source: `make results`
# regenerates it, the reproduce workflow diffs it against the raw data, and
# `site/build_page.py` reads the same file. Preferring it here means a rebuilt
# card takes effect immediately, with no copy to refresh.
#
# In an installed wheel there is no ``results/`` directory, so the build copies
# that same file to ``fusionflux/predictor.json`` as package data (see
# ``setup.py``) and this falls through to it. The copy is generated at build
# time and gitignored, so it cannot drift from the artifact it came from: there
# is nothing committed to go stale.
_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_CARD = _PACKAGE_DIR.parent / "results" / "predictor.json"
_PACKAGED_CARD = _PACKAGE_DIR / "predictor.json"

DEFAULT_CARD_PATH = _REPO_CARD if _REPO_CARD.exists() else _PACKAGED_CARD

# The model Result 8 selected, used as the recommendation whenever the query is
# outside what the learned models were validated on.
SAFE_MODEL = "powerlaw_collisionless"

# Models the card carries coefficients for, in the order they are reported.
CARD_MODELS: tuple[str, ...] = ("ipb98y2_analytic", SAFE_MODEL, "powerlaw_free")

MODEL_LABELS: dict[str, str] = {
    "ipb98y2_analytic": "IPB98(y,2), analytic",
    SAFE_MODEL: "power law, collisionless",
    "powerlaw_free": "power law, unconstrained",
}

# Quantile of the training rows' own distances that defines the hull. Not the
# maximum: a single outlying row would then set the boundary, and the point is to
# describe where the data actually lives.
HULL_QUANTILE = 0.99


@dataclass(frozen=True)
class ModelPrediction:
    """One model's answer for one operating point, and whether to believe it."""

    model_name: str
    tau_s: float
    interval_low_s: float
    interval_high_s: float
    nominal_coverage: float
    # False when the query falls outside what this model was validated on.
    trustworthy_here: bool
    note: str

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ConfinementPrediction:
    """A prediction, its interval, and everything qualifying it."""

    inputs: dict[str, float]
    # Mahalanobis distance of this operating point from the training rows, in the
    # same units as Result 4b's per-machine distances and Result 12's devices.
    extrapolation_distance: float
    outside_training_hull: bool
    beyond_validated_range: bool
    physics_exceeds_training_ceiling: bool
    training_ceiling_s: float
    recommended_model: str
    predictions: tuple[ModelPrediction, ...]
    warnings: tuple[str, ...]

    @property
    def recommended(self) -> ModelPrediction:
        for prediction in self.predictions:
            if prediction.model_name == self.recommended_model:
                return prediction
        raise AssertionError(f"recommended model {self.recommended_model} not in predictions")

    @property
    def tau_s(self) -> float:
        """Point estimate of the recommended model, for the one-line use."""
        return self.recommended.tau_s

    @property
    def interval_s(self) -> tuple[float, float]:
        return (self.recommended.interval_low_s, self.recommended.interval_high_s)

    def to_json(self) -> dict[str, object]:
        return {
            "inputs": self.inputs,
            "extrapolation_distance": self.extrapolation_distance,
            "outside_training_hull": self.outside_training_hull,
            "beyond_validated_range": self.beyond_validated_range,
            "physics_exceeds_training_ceiling": self.physics_exceeds_training_ceiling,
            "training_ceiling_s": self.training_ceiling_s,
            "recommended_model": self.recommended_model,
            "tau_s": self.tau_s,
            "interval_low_s": self.interval_s[0],
            "interval_high_s": self.interval_s[1],
            "predictions": [row.to_json() for row in self.predictions],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class ServiceCard:
    """Everything needed to predict, without refitting anything."""

    generated_on: str
    dataset_sha256: str
    n_training_rows: int
    nominal_coverage: float
    # Engineering log columns the linear models are stated on.
    model_feature_columns: tuple[str, ...]
    # Columns the distance is measured in. Deliberately the wider blind set, so
    # the number is comparable to Result 4b and Result 12 rather than being a
    # new quantity that happens to share a name.
    distance_feature_columns: tuple[str, ...]
    coefficients: dict[str, list[float]]
    interval_quantile: dict[str, float]
    interval_distance_intercept: dict[str, float]
    interval_distance_slope: dict[str, float]
    distance_mean: list[float]
    distance_precision: list[list[float]]
    training_ceiling_s: float
    training_hull_distance: float
    validated_distance_max: float
    validated_machine: str

    def to_json(self) -> dict[str, object]:
        payload = asdict(self)
        payload["model_feature_columns"] = list(self.model_feature_columns)
        payload["distance_feature_columns"] = list(self.distance_feature_columns)
        return payload

    @classmethod
    def from_json(cls, payload: dict) -> "ServiceCard":
        return cls(
            generated_on=str(payload["generated_on"]),
            dataset_sha256=str(payload["dataset_sha256"]),
            n_training_rows=int(payload["n_training_rows"]),
            nominal_coverage=float(payload["nominal_coverage"]),
            model_feature_columns=tuple(payload["model_feature_columns"]),
            distance_feature_columns=tuple(payload["distance_feature_columns"]),
            coefficients={k: list(map(float, v)) for k, v in payload["coefficients"].items()},
            interval_quantile={k: float(v) for k, v in payload["interval_quantile"].items()},
            interval_distance_intercept={
                k: float(v) for k, v in payload["interval_distance_intercept"].items()
            },
            interval_distance_slope={
                k: float(v) for k, v in payload["interval_distance_slope"].items()
            },
            distance_mean=[float(v) for v in payload["distance_mean"]],
            distance_precision=[[float(v) for v in row] for row in payload["distance_precision"]],
            training_ceiling_s=float(payload["training_ceiling_s"]),
            training_hull_distance=float(payload["training_hull_distance"]),
            validated_distance_max=float(payload["validated_distance_max"]),
            validated_machine=str(payload["validated_machine"]),
        )


def build_service_card(dataset: pd.DataFrame | None = None) -> ServiceCard:
    """Fit everything once and reduce it to numbers a prediction can be read from.

    Imports the study modules lazily so that :func:`predict` and
    :func:`load_card`, which are the hot path, do not pull in scikit-learn at all.

    Those modules are analysis scripts and the wheel does not install them, so
    this is the one entry point in the package that requires a checkout. It says
    so rather than surfacing a bare ``ModuleNotFoundError`` naming an internal
    module the caller has no reason to have heard of.
    """
    try:
        import conformal_shift as cshift
        import dimensional as dm
        import hdb5
    except ModuleNotFoundError as error:  # pragma: no cover - requires an installed wheel
        raise ModuleNotFoundError(
            "Rebuilding the predictor card needs the analysis modules "
            f"({error.name} is missing) and the HDB5 dataset. Neither ships in the "
            "wheel: `pip install fusionflux` installs the callable study, not the "
            "pipeline that generated it. Clone the repository and run "
            "`python3 -m fusionflux card` there. Prediction itself needs "
            "none of this and works from the installed package alone."
        ) from error

    if dataset is None:
        dataset = hdb5.prepare_dataset()

    distance_columns = tuple(hdb5.BLIND_FEATURE_COLUMNS)
    model_columns = tuple(f"log_{name}" for name in dm.CONSTRAINED_FEATURE_COLUMNS)

    features = dataset[list(distance_columns)]
    tau = dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float)
    log_tau = np.log(tau)
    train_index = np.arange(len(dataset))

    zoo = {
        SAFE_MODEL: dm.build_constrained_models(("collisionless",))[SAFE_MODEL],
        "powerlaw_free": dm.build_constrained_models(("free",))["powerlaw_free"],
    }

    # Coefficients, on the eight engineering log columns plus an intercept, so a
    # prediction is one dot product.
    #
    # The comparator is ``powerlaw_free``, the unconstrained least-squares fit,
    # and deliberately not the zoo's ``ridge_loglinear``. Those are different
    # models: ridge carries a penalty and is fitted on the wider blind feature
    # set standardized, so its coefficients do not exist on these eight columns
    # and quoting them here would be a different number under a familiar name.
    # ``powerlaw_free`` is also the better comparator, since it differs from the
    # recommended model by the constraint and by nothing else, which is exactly
    # the contrast Result 8 reports.
    coefficients: dict[str, list[float]] = {}
    for name, model in (
        (SAFE_MODEL, "collisionless"),
        ("powerlaw_free", "free"),
    ):
        fitted = dm.ConstrainedPowerLaw(model=model).fit(features, log_tau)
        coefficients[name] = [float(value) for value in fitted.coefficients_]

    # Intervals, from Result 10's machine-level calibration with distance scaling.
    quantiles: dict[str, float] = {}
    intercepts: dict[str, float] = {}
    slopes: dict[str, float] = {}
    alpha = hdb5.DEFAULT_CONFORMAL_ALPHA

    def _calibrate(name: str, calibration: cshift.CalibrationSet) -> None:
        intercept, slope = cshift.fit_distance_scale(
            calibration.distances, calibration.absolute_residuals
        )
        quantiles[name] = float(
            hdb5.split_conformal_half_width(calibration.scaled_scores(intercept, slope), alpha)
        )
        intercepts[name] = intercept
        slopes[name] = slope

    for name, estimator in zoo.items():
        _calibrate(
            name,
            cshift.machine_cv_calibration(
                dataset, estimator, train_index, feature_columns=distance_columns
            ),
        )

    # The analytic law is not fitted, so its machine-level calibration is simply
    # its residuals on each machine, assembled in the same shape.
    analytic = np.log(dataset["ipb98y2_tau_s"].to_numpy(dtype=float))
    labels = dataset[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()
    machines = hdb5.eligible_tokamaks(dataset)
    residuals, distances = [], []
    for machine in machines:
        held = np.flatnonzero(labels == machine)
        inner = np.flatnonzero(labels != machine)
        residuals.append(np.abs(log_tau[held] - analytic[held]))
        distances.append(
            cshift.row_mahalanobis(
                features.iloc[inner].to_numpy(dtype=float),
                features.iloc[held].to_numpy(dtype=float),
            )
        )
    _calibrate(
        "ipb98y2_analytic",
        cshift.CalibrationSet(
            np.concatenate(residuals), np.concatenate(distances), len(machines)
        ),
    )

    # The distance metric, frozen. Pseudo-inverse for the reason Result 4b gives:
    # ``log a_m`` is exactly ``log r_m + log inverse_aspect_ratio``, so the
    # covariance is singular by construction and a plain inverse would blow up.
    training = features.to_numpy(dtype=float)
    precision = np.linalg.pinv(np.atleast_2d(np.cov(training, rowvar=False)))
    row_distances = cshift.row_mahalanobis(training, training)

    # How far out this study actually measured anything.
    #
    # Deliberately the same statistic as ``training_hull_distance``: a quantile
    # of *per-row* distances. Result 4b's per-machine number is the distance of a
    # machine's mean from the training data, which is a different quantity, and
    # comparing a query's row distance against it would put the two thresholds on
    # scales that need not even be ordered. Here each held-out machine's rows are
    # measured against the machines that would have been trained on, which is the
    # situation Result 4 actually scored, so the maximum is "the furthest any row
    # we have a measured error for sat from the data used to predict it".
    machine_distances: dict[str, float] = {}
    for machine in machines:
        held = np.flatnonzero(labels == machine)
        inner = np.flatnonzero(labels != machine)
        machine_distances[machine] = float(
            np.quantile(
                cshift.row_mahalanobis(
                    features.iloc[inner].to_numpy(dtype=float),
                    features.iloc[held].to_numpy(dtype=float),
                ),
                HULL_QUANTILE,
            )
        )
    furthest = max(machine_distances, key=lambda name: machine_distances[name])

    return ServiceCard(
        generated_on=date.today().isoformat(),
        dataset_sha256=hdb5.HDB5_STD5_SHA256,
        n_training_rows=int(len(dataset)),
        nominal_coverage=1.0 - alpha,
        model_feature_columns=model_columns,
        distance_feature_columns=distance_columns,
        coefficients=coefficients,
        interval_quantile=quantiles,
        interval_distance_intercept=intercepts,
        interval_distance_slope=slopes,
        distance_mean=[float(value) for value in training.mean(axis=0)],
        distance_precision=[[float(value) for value in row] for row in precision],
        training_ceiling_s=float(tau.max()),
        training_hull_distance=float(np.quantile(row_distances, HULL_QUANTILE)),
        validated_distance_max=float(machine_distances[furthest]),
        validated_machine=str(furthest),
    )


def save_card(card: ServiceCard, path: Path | str = DEFAULT_CARD_PATH) -> Path:
    # Same checkout-only story as ``build_service_card``: the only caller that
    # reaches here is the card rebuild, which already needs the dataset.
    from storage import write_json_strict

    target = Path(path)
    write_json_strict(target, card.to_json())
    return target


_CARD_CACHE: dict[Path, ServiceCard] = {}


def load_card(path: Path | str = DEFAULT_CARD_PATH) -> ServiceCard:
    """Load the service card, cached by path so repeated predictions are free."""
    resolved = Path(path).expanduser().resolve()
    if resolved not in _CARD_CACHE:
        if not resolved.exists():
            raise FileNotFoundError(
                f"No predictor card at {resolved}. Build one with "
                "`python3 -m fusionflux card`, which needs the HDB5 dataset and a\n"
                "checkout: the analysis modules it reads are not installed by the wheel."
            )
        _CARD_CACHE[resolved] = ServiceCard.from_json(json.loads(resolved.read_text()))
    return _CARD_CACHE[resolved]


# The eight engineering inputs, in the order the card's coefficients expect.
REQUIRED_INPUTS: tuple[str, ...] = (
    "ip_ma",
    "bt_t",
    "ne_line_1e19_m3",
    "p_loss_mw",
    "r_m",
    "inverse_aspect_ratio",
    "kappa",
    "m_eff_amu",
)


def _ipb98y2(inputs: dict[str, float]) -> float:
    """The published law, evaluated directly so the card needs no coefficients for it."""
    return (
        0.0562
        * inputs["ip_ma"] ** 0.93
        * inputs["bt_t"] ** 0.15
        * inputs["ne_line_1e19_m3"] ** 0.41
        * inputs["p_loss_mw"] ** -0.69
        * inputs["r_m"] ** 1.97
        * inputs["inverse_aspect_ratio"] ** 0.58
        * inputs["kappa"] ** 0.78
        * inputs["m_eff_amu"] ** 0.19
    )


def _distance_features(inputs: dict[str, float], card: ServiceCard) -> np.ndarray:
    """The query in the card's distance coordinates.

    ``a_m`` is derived rather than requested, exactly as it is in cleaning, so a
    caller cannot supply a minor radius inconsistent with ``r_m`` and ``eps``.
    """
    values = dict(inputs)
    values["a_m"] = values["inverse_aspect_ratio"] * values["r_m"]
    return np.array(
        [np.log(values[column.removeprefix("log_")]) for column in card.distance_feature_columns]
    )


def predict(
    *,
    ip_ma: float,
    bt_t: float,
    ne_line_1e19_m3: float,
    p_loss_mw: float,
    r_m: float,
    inverse_aspect_ratio: float,
    kappa: float,
    m_eff_amu: float,
    card: ServiceCard | None = None,
) -> ConfinementPrediction:
    """Predict energy confinement time, with an interval and a refusal.

    Every argument is keyword-only and required. A confinement scaling law has
    eight inputs whose order nobody remembers, and a positional call that
    silently transposed the field and the density would return a plausible
    number for the wrong machine.

    ``a_m`` is not an argument: it is derived as ``inverse_aspect_ratio * r_m``
    the way cleaning derives it, which is also why Result 1 found the design
    matrix rank deficient.
    """
    inputs = {
        "ip_ma": float(ip_ma),
        "bt_t": float(bt_t),
        "ne_line_1e19_m3": float(ne_line_1e19_m3),
        "p_loss_mw": float(p_loss_mw),
        "r_m": float(r_m),
        "inverse_aspect_ratio": float(inverse_aspect_ratio),
        "kappa": float(kappa),
        "m_eff_amu": float(m_eff_amu),
    }
    bad = [name for name, value in inputs.items() if not np.isfinite(value) or value <= 0.0]
    if bad:
        raise ValueError(
            f"Every input must be finite and strictly positive; got non-positive {sorted(bad)}. "
            "These are logged before fitting, so zero or negative has no meaning here."
        )
    if card is None:
        card = load_card()

    query = _distance_features(inputs, card)
    difference = query - np.asarray(card.distance_mean)
    precision = np.asarray(card.distance_precision)
    distance = float(np.sqrt(max(float(difference @ precision @ difference), 0.0)))

    outside_hull = distance > card.training_hull_distance
    beyond_validated = distance > card.validated_distance_max

    analytic_tau = _ipb98y2(inputs)
    exceeds_ceiling = analytic_tau > card.training_ceiling_s

    design = np.concatenate(
        [[1.0], [np.log(inputs[column.removeprefix("log_")]) for column in card.model_feature_columns]]
    )

    predictions: list[ModelPrediction] = []
    for name in CARD_MODELS:
        if name == "ipb98y2_analytic":
            tau = analytic_tau
        else:
            tau = float(np.exp(design @ np.asarray(card.coefficients[name])))
        scale = np.exp(
            card.interval_distance_intercept[name] + card.interval_distance_slope[name] * distance
        )
        half_width = card.interval_quantile[name] * max(float(scale), 1e-3)

        trustworthy = not beyond_validated or name == SAFE_MODEL
        if beyond_validated and name == SAFE_MODEL:
            note = (
                "beyond the validated range, but this is the model Result 8 selected for "
                "exactly that case and the only one whose intervals held at the largest "
                "extrapolation measured"
            )
        elif beyond_validated:
            note = (
                f"beyond the validated range: this query sits {distance:.1f} from the training "
                f"data and the furthest machine ever scored ({card.validated_machine}) sits at "
                f"{card.validated_distance_max:.1f}"
            )
        elif outside_hull:
            note = "outside the bulk of the training data but inside the validated range"
        else:
            note = "inside the training distribution"
        predictions.append(
            ModelPrediction(
                model_name=name,
                tau_s=tau,
                interval_low_s=float(tau * np.exp(-half_width)),
                interval_high_s=float(tau * np.exp(half_width)),
                nominal_coverage=card.nominal_coverage,
                trustworthy_here=trustworthy,
                note=note,
            )
        )

    warnings: list[str] = []
    if beyond_validated:
        warnings.append(
            f"This operating point sits {distance:.1f} from the training data, beyond the "
            f"{card.validated_distance_max:.1f} of {card.validated_machine}, the furthest machine "
            "this study held out and scored. No model here has been measured this far out."
        )
    elif outside_hull:
        warnings.append(
            f"This operating point sits {distance:.1f} from the training data, outside the bulk "
            f"of it ({card.training_hull_distance:.1f}) but within the range Result 4 validated."
        )
    if exceeds_ceiling:
        recommended_tau = next(
            row.tau_s for row in predictions if row.model_name == SAFE_MODEL
        )
        warnings.append(
            f"Any range-bounded model is capped at {card.training_ceiling_s:.3f} s here, the "
            f"largest confinement time in the training data, which is a factor of "
            f"{recommended_tau / card.training_ceiling_s:.1f} below the {recommended_tau:.2f} s "
            "recommended above. By Result 4c a tree ensemble averages training targets, so no "
            "random forest or gradient booster can return the right answer for this machine "
            "whatever its inputs, features or tuning. It is structurally wrong here rather than "
            "merely uncertain, and that is decidable from the inputs alone."
        )
    if not warnings:
        warnings.append("Inside the training distribution; all models here are on measured ground.")

    return ConfinementPrediction(
        inputs=inputs,
        extrapolation_distance=distance,
        outside_training_hull=outside_hull,
        beyond_validated_range=beyond_validated,
        physics_exceeds_training_ceiling=exceeds_ceiling,
        training_ceiling_s=card.training_ceiling_s,
        recommended_model=SAFE_MODEL,
        predictions=tuple(predictions),
        warnings=tuple(warnings),
    )


def format_prediction(result: ConfinementPrediction) -> str:
    """Human-readable report, which is what the CLI prints."""
    lines = [
        f"  extrapolation distance     {result.extrapolation_distance:.2f}",
        f"  training ceiling           {result.training_ceiling_s:.3f} s",
        "",
        f"  {'model':<28}{'tau_E (s)':>11}{'interval (s)':>22}{'trust':>8}",
    ]
    for prediction in result.predictions:
        interval = f"{prediction.interval_low_s:.3f} to {prediction.interval_high_s:.3f}"
        mark = "yes" if prediction.trustworthy_here else "NO"
        star = "*" if prediction.model_name == result.recommended_model else " "
        lines.append(
            f"{star} {MODEL_LABELS[prediction.model_name]:<27}"
            f"{prediction.tau_s:>11.3f}{interval:>22}{mark:>8}"
        )
    if result.physics_exceeds_training_ceiling:
        lines.append(
            f"  {'any range-bounded ensemble':<28}{'<= ' + format(result.training_ceiling_s, '.3f'):>11}"
            f"{'(cannot exceed)':>22}{'NO':>8}"
        )
    lines.append("")
    lines.append(f"  * recommended: {MODEL_LABELS[result.recommended_model]}")
    for warning in result.warnings:
        lines.append("")
        for chunk in _wrap(warning, 76):
            lines.append(f"  {chunk}")
    return "\n".join(lines)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def main(argv: list[str] | None = None) -> None:
    """Rebuild the service card. Prediction lives in ``cli``.

    Deliberately build-only. ``fusionflux predict`` is the prediction front door
    and defining the same flags twice would be two parsers to keep in step; this
    entry point exists so ``make results`` and a checkout without the package
    installed can still regenerate the card. ``python3 -m fusionflux card`` is the
    same thing through the CLI and is what ``make results`` runs.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Rebuild the confinement predictor card.")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="Rebuild results/predictor.json from the dataset.")
    build.add_argument("--output", default=str(DEFAULT_CARD_PATH))

    args = parser.parse_args(argv)
    print(f"wrote {save_card(build_service_card(), args.output)}")


if __name__ == "__main__":
    main()
