"""Tests for the predictor: the study's conclusions, enforced at the call site.

``predictor.predict`` is the one function in this repository a person would call
without having read anything, so the risk here is different from the analysis
modules. Those can be wrong and produce a visibly odd table. This can be wrong
and produce a confident, plausible, well-formatted number for a machine nobody
should be predicting yet, which is precisely the failure the whole study is
about.

So the tests below fix the behaviour that makes it a *refusal* rather than a
calculator:

* the ceiling check fires for ITER and does not fire in distribution, and it is
  decidable from the inputs alone, before any model runs;
* the recommendation stays on the model Result 8 selected when the query leaves
  the validated range, rather than following whatever scores best nearby;
* the numbers agree with ``results/forecast.json``, so the tool and Result 12
  cannot drift apart while both claim to describe the same three machines;
* the distance agrees with ``conformal_shift.row_mahalanobis``, which is what
  Results 4b, 10 and 12 all report, so the number the caller is shown means the
  same thing there as it does in the prose.

Everything except the card-building tests runs from the committed card alone, so
a fresh checkout can predict without downloading anything. That is a property
worth having and therefore worth testing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fusionflux import predictor

# ITER's Q=10 inductive baseline, the case the refusal exists for.
ITER = {
    "ip_ma": 15.0,
    "bt_t": 5.3,
    "ne_line_1e19_m3": 10.0,
    "p_loss_mw": 87.0,
    "r_m": 6.2,
    "inverse_aspect_ratio": 2.0 / 6.2,
    "kappa": 1.7,
    "m_eff_amu": 2.5,
}

# A mid-sized conventional operating point well inside the database.
IN_RANGE = {
    "ip_ma": 2.5,
    "bt_t": 2.6,
    "ne_line_1e19_m3": 5.0,
    "p_loss_mw": 12.0,
    "r_m": 2.9,
    "inverse_aspect_ratio": 0.33,
    "kappa": 1.7,
    "m_eff_amu": 2.0,
}


def _card_or_skip() -> predictor.ServiceCard:
    if not predictor.DEFAULT_CARD_PATH.exists():
        pytest.skip("No predictor card; run `python3 -m fusionflux card`.")
    return predictor.load_card()


# --- the refusal ------------------------------------------------------------


def test_iter_trips_the_training_ceiling() -> None:
    """The headline case: no range-bounded model can be right about ITER.

    This is Result 4c evaluated on a specific machine, and it is decidable from
    the inputs alone, so it must fire without any model being consulted.
    """
    result = predictor.predict(**ITER, card=_card_or_skip())
    assert result.physics_exceeds_training_ceiling
    assert result.training_ceiling_s < 2.0
    assert any("range-bounded" in warning for warning in result.warnings)


def test_an_ordinary_operating_point_trips_nothing() -> None:
    """The control. A refusal that fires everywhere would be useless."""
    result = predictor.predict(**IN_RANGE, card=_card_or_skip())
    assert not result.physics_exceeds_training_ceiling
    assert not result.beyond_validated_range
    assert not result.outside_training_hull
    assert result.warnings == (
        "Inside the training distribution; all models here are on measured ground.",
    )


def test_a_machine_beyond_every_scored_one_is_flagged_and_not_trusted() -> None:
    """Past the furthest machine ever held out, nothing here has been measured."""
    absurd = {**ITER, "r_m": 30.0, "bt_t": 40.0, "ip_ma": 60.0, "p_loss_mw": 500.0}
    result = predictor.predict(**absurd, card=_card_or_skip())
    assert result.beyond_validated_range
    assert result.outside_training_hull
    untrusted = [row.model_name for row in result.predictions if not row.trustworthy_here]
    assert untrusted, "something must be marked untrustworthy this far out"
    # The model Result 8 selected is the one that stays recommended out there.
    assert predictor.SAFE_MODEL not in untrusted
    assert result.recommended_model == predictor.SAFE_MODEL


def test_the_recommendation_is_always_the_constrained_law() -> None:
    """Result 8's model, in and out of distribution, rather than a score chase."""
    card = _card_or_skip()
    for inputs in (ITER, IN_RANGE):
        assert predictor.predict(**inputs, card=card).recommended_model == predictor.SAFE_MODEL


# --- agreement with the study ----------------------------------------------


def test_predictions_agree_with_the_locked_forecast() -> None:
    """The tool and Result 12 must not drift apart.

    Both fit the same models on the same pinned rows, so a disagreement means
    one of them changed and the other did not, which is exactly the failure
    ``tests/test_reported_numbers.py`` guards for the prose.
    """
    forecast_path = Path(__file__).resolve().parents[1] / "results" / "forecast.json"
    if not forecast_path.exists():
        pytest.skip("No forecast artifact; run `python3 analysis_forecast.py`.")
    forecast = json.loads(forecast_path.read_text())

    devices = {row["name"]: row for row in forecast["devices"]}
    expected = {
        (row["device"], row["model_name"]): row["tau_predicted_s"]
        for row in forecast["forecasts"]
    }
    card = _card_or_skip()

    for name in ("SPARC", "JT-60SA", "ITER"):
        device = devices[name]
        result = predictor.predict(
            **{key: float(device[key]) for key in predictor.REQUIRED_INPUTS}, card=card
        )
        for prediction in result.predictions:
            key = (name, prediction.model_name)
            if key not in expected:
                continue
            assert prediction.tau_s == pytest.approx(expected[key], rel=1e-6), key


def test_the_ceiling_matches_the_forecast_artifact() -> None:
    forecast_path = Path(__file__).resolve().parents[1] / "results" / "forecast.json"
    if not forecast_path.exists():
        pytest.skip("No forecast artifact.")
    forecast = json.loads(forecast_path.read_text())
    assert _card_or_skip().training_ceiling_s == pytest.approx(
        float(forecast["train_tau_max_s"]), rel=1e-9
    )


def test_distance_matches_the_studys_own_measure() -> None:
    """The number shown to a caller must mean what Results 4b and 12 mean by it."""
    import hdb5

    if not hdb5.default_hdb5_path().exists():
        pytest.skip("HDB5 STD5 not downloaded.")
    import conformal_shift as cshift

    card = _card_or_skip()
    dataset = hdb5.prepare_dataset()
    training = dataset[list(card.distance_feature_columns)].to_numpy(dtype=float)

    values = dict(ITER)
    values["a_m"] = values["inverse_aspect_ratio"] * values["r_m"]
    query = np.array(
        [[np.log(values[c.removeprefix("log_")]) for c in card.distance_feature_columns]]
    )
    expected = float(cshift.row_mahalanobis(training, query)[0])
    assert predictor.predict(**ITER, card=card).extrapolation_distance == pytest.approx(
        expected, rel=1e-6
    )


# --- inputs -----------------------------------------------------------------


@pytest.mark.parametrize("bad", ["ip_ma", "bt_t", "r_m", "kappa"])
def test_non_positive_inputs_are_refused(bad: str) -> None:
    """These are logged before fitting, so zero has no meaning rather than a value."""
    with pytest.raises(ValueError, match="strictly positive"):
        predictor.predict(**{**ITER, bad: 0.0}, card=_card_or_skip())


def test_inputs_are_keyword_only() -> None:
    """Eight inputs whose order nobody remembers must not be passable positionally.

    A transposed field and density would otherwise return a confident number for
    a machine that does not exist.
    """
    with pytest.raises(TypeError):
        predictor.predict(*ITER.values())  # type: ignore[call-arg]


def test_minor_radius_is_derived_not_accepted() -> None:
    """``a_m`` follows from ``eps * r_m``, so a caller cannot contradict it."""
    assert "a_m" not in predictor.REQUIRED_INPUTS
    with pytest.raises(TypeError):
        predictor.predict(**ITER, a_m=1.0, card=_card_or_skip())  # type: ignore[call-arg]


# --- shape of the answer ----------------------------------------------------


def test_every_interval_brackets_its_point_estimate() -> None:
    result = predictor.predict(**ITER, card=_card_or_skip())
    for prediction in result.predictions:
        assert 0.0 < prediction.interval_low_s <= prediction.tau_s <= prediction.interval_high_s
        assert prediction.nominal_coverage == pytest.approx(0.9)


def test_convenience_accessors_agree_with_the_recommended_row() -> None:
    result = predictor.predict(**IN_RANGE, card=_card_or_skip())
    assert result.tau_s == result.recommended.tau_s
    assert result.interval_s == (
        result.recommended.interval_low_s,
        result.recommended.interval_high_s,
    )


def test_json_round_trips_and_carries_the_qualifiers() -> None:
    payload = predictor.predict(**ITER, card=_card_or_skip()).to_json()
    assert json.loads(json.dumps(payload))["physics_exceeds_training_ceiling"] is True
    assert payload["recommended_model"] == predictor.SAFE_MODEL
    predictions = payload["predictions"]
    assert isinstance(predictions, list)
    assert len(predictions) == len(predictor.CARD_MODELS)


def test_report_names_the_ceiling_when_it_applies() -> None:
    """The refusal has to be visible in the thing people actually read."""
    card = _card_or_skip()
    text = predictor.format_prediction(predictor.predict(**ITER, card=card))
    assert "cannot exceed" in text
    assert "recommended" in text
    assert "cannot exceed" not in predictor.format_prediction(
        predictor.predict(**IN_RANGE, card=card)
    )


# --- the card ---------------------------------------------------------------


def test_card_round_trips_through_json() -> None:
    card = _card_or_skip()
    assert predictor.ServiceCard.from_json(json.loads(json.dumps(card.to_json()))) == card


def test_card_is_pinned_to_the_dataset_it_was_built_from() -> None:
    import hdb5

    assert _card_or_skip().dataset_sha256 == hdb5.HDB5_STD5_SHA256


def test_missing_card_raises_a_useful_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="fusionflux card"):
        predictor.load_card(tmp_path / "absent.json")


def test_hull_sits_inside_the_validated_range() -> None:
    """The two thresholds must be ordered, or the warnings would contradict.

    The bulk of the data has to be nearer than the furthest machine ever scored;
    if it were not, a query could be "beyond everything validated" while still
    inside the hull, and both warnings would fire with opposite meanings.
    """
    card = _card_or_skip()
    assert 0.0 < card.training_hull_distance <= card.validated_distance_max


# --- building the card ------------------------------------------------------


def _synthetic_dataset(n_per_machine: int = 120, seed: int = 11) -> pd.DataFrame:
    """An HDB5-shaped frame drawn from an exact power law with log-normal noise.

    The card builder is exercised here rather than only against the real
    database, for the reason the rest of this suite is: a builder that only ever
    runs on one dataset is indistinguishable from one hard-wired to it, and this
    also lets the expensive path be covered without a download.
    """
    import hdb5

    rng = np.random.default_rng(seed)
    frames = []
    for index, (machine, radius) in enumerate(
        {"A": 0.9, "B": 1.3, "C": 1.8, "D": 2.4, "E": 3.0}.items()
    ):
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
            0.0562 * ip**0.93 * bt**0.15 * nel**0.41 * plth**-0.69
            * rgeo**1.97 * eps**0.58 * kappa**0.78 * meff**0.19
        ) * np.exp(rng.normal(0.0, 0.10, n))
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


def test_a_card_can_be_built_from_any_hdb5_shaped_frame() -> None:
    card = predictor.build_service_card(_synthetic_dataset())
    assert card.n_training_rows == 600
    assert set(card.coefficients) == {predictor.SAFE_MODEL, "powerlaw_free"}
    # One intercept plus the eight engineering exponents.
    assert all(len(values) == 9 for values in card.coefficients.values())
    assert set(card.interval_quantile) == set(predictor.CARD_MODELS)
    assert card.training_ceiling_s > 0.0
    assert 0.0 < card.training_hull_distance <= card.validated_distance_max
    assert card.validated_machine in {"A", "B", "C", "D", "E"}


def test_a_freshly_built_card_predicts_the_law_it_was_drawn_from() -> None:
    """End to end: build on data generated by IPB98(y,2), then recover it.

    The synthetic targets *are* the analytic law plus 10% log noise, so a
    correctly built card must predict close to it. This is the check that the
    coefficients, the design vector and the ordering of the eight columns all
    agree; a transposed column would fit the training data and still fail here.
    """
    card = predictor.build_service_card(_synthetic_dataset())
    inputs = {
        "ip_ma": 2.0, "bt_t": 3.0, "ne_line_1e19_m3": 8.0, "p_loss_mw": 9.0,
        "r_m": 1.9, "inverse_aspect_ratio": 0.30, "kappa": 1.6, "m_eff_amu": 2.0,
    }
    result = predictor.predict(**inputs, card=card)
    analytic = next(
        row.tau_s for row in result.predictions if row.model_name == "ipb98y2_analytic"
    )
    assert result.tau_s == pytest.approx(analytic, rel=0.15)


def test_save_and_reload_a_card_round_trips(tmp_path: Path) -> None:
    card = predictor.build_service_card(_synthetic_dataset())
    path = predictor.save_card(card, tmp_path / "card.json")
    assert path.exists()
    assert predictor.load_card(path) == card


def test_the_module_entry_point_builds_a_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """``python3 -m fusionflux card``, which is what ``make results`` runs."""
    monkeypatch.setattr(predictor, "build_service_card", lambda: predictor.load_card())
    target = tmp_path / "rebuilt.json"
    predictor.main(["build", "--output", str(target)])
    assert target.exists()
    assert "wrote" in capsys.readouterr().out


def test_the_module_entry_point_does_not_predict() -> None:
    """Prediction is ``fusionflux predict``; two parsers would drift apart."""
    with pytest.raises(SystemExit):
        predictor.main(["predict", "--ip-ma", "1.0"])
