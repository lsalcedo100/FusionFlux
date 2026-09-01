"""Tests for Results 8 and 9: physics as a constraint, and physics as a prior.

The load-bearing risk in ``dimensional.py`` is not a bug in the fitting, which
is ``scaling_law.solve_constrained_lstsq`` and already tested. It is that the
*derivation* is wrong: a transposed sign or a dropped term in the engineering
exponents would produce a constraint matrix that fits perfectly well, obeys
itself to machine precision, and encodes physics nobody intended. Nothing
downstream would notice, because a constrained fit looks exactly as healthy when
the constraint is wrong.

So the tests here attack the derivation from three independent directions:

* the transformations really do hold the dimensionless groups they claim to
  hold fixed, checked against the group definitions rather than against the
  transformation that produced them;
* the Kadomtsev transformation recovers the closed-form scaling written out by
  hand in the module docstring, ``B ~ lam^-5/4``, ``T ~ lam^-1/2``,
  ``n ~ lam^-2``, ``tau ~ lam^5/4``;
* IPB98(y,2), a law published in 1999 and not consulted while deriving any of
  this, lands on the first two constraint surfaces to inside the rounding of its
  own two-decimal exponents. That is the check no coincidence survives.

For ``spectral.py`` the risk is different and simpler: both ends of every sweep
are models that already exist elsewhere, so the endpoints are exactly testable.
An estimator that did not reproduce the unconstrained fit at one end and the
published exponents at the other would be shrinking toward something other than
what it says.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone, is_regressor

import dimensional as dm
import hdb5
import scaling_law as sl
import spectral as sp


def _make_dataset(n_per_machine: int = 90, seed: int = 5) -> pd.DataFrame:
    """A prepared HDB5-shaped frame drawn from an exact power law plus noise."""
    rng = np.random.default_rng(seed)
    frames = []
    for index, (machine, radius) in enumerate(
        {"A": 0.7, "B": 1.2, "C": 1.9, "D": 2.6, "E": 3.2}.items()
    ):
        n = n_per_machine
        ip = rng.uniform(0.4, 4.0, n)
        bt = rng.uniform(1.0, 5.0, n)
        nel = rng.uniform(1.5, 20.0, n)
        plth = rng.uniform(0.5, 25.0, n)
        rgeo = radius * rng.uniform(0.97, 1.03, n)
        eps = rng.uniform(0.25, 0.35, n)
        kappa = rng.uniform(1.1, 2.2, n)
        meff = rng.uniform(1.0, 3.0, n)
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
        ) * np.exp(rng.normal(0.0, 0.08, n))
        frames.append(
            pd.DataFrame(
                {
                    "TOK": machine,
                    "SHOT": rng.integers(index * 10_000, index * 10_000 + n // 3, n),
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


# --- the derivation ---------------------------------------------------------


@pytest.mark.parametrize("model", dm.CONSTRAINT_MODELS)
def test_transformations_hold_their_groups_fixed(model: str) -> None:
    """Every admissible transformation leaves the required groups invariant.

    Checked by evaluating the group definitions on the transformation, not by
    re-deriving it: this is the property the null-space construction is supposed
    to deliver, so it has to be verified independently of the construction.
    """
    for transform in dm.admissible_transformations(model):
        exponents = transform.group_exponents()
        for group in dm.MODEL_FIXED_GROUPS[model]:
            assert exponents[group] == pytest.approx(0.0, abs=1e-12)


def test_each_model_admits_the_expected_number_of_transformations() -> None:
    """Four scale freedoms minus the conditions imposed leaves the family size."""
    for model, groups in dm.MODEL_FIXED_GROUPS.items():
        expected = 4 - len(groups)
        assert len(dm.admissible_transformations(model)) == expected
        rows, rhs = dm.constraint_matrix(model, intercept=False)
        assert rows.shape == (expected, len(dm.CONSTRAINED_FEATURE_COLUMNS))
        assert rhs.shape == (expected,)


def test_kadomtsev_recovers_the_closed_form_similarity_transformation() -> None:
    """The one-parameter family must be the textbook Kadomtsev scaling.

    Normalising so the length exponent is 1, the module docstring's derivation
    gives ``B ~ lam^-5/4``, ``T ~ lam^-1/2``, ``n ~ lam^-2`` and
    ``tau_E ~ lam^5/4``. The SVD returns an arbitrary scaling of that vector and
    an arbitrary sign, so both are normalised away before comparing.
    """
    (transform,) = dm.admissible_transformations("kadomtsev")
    scale = 1.0 / transform.length
    assert transform.field * scale == pytest.approx(-1.25, abs=1e-12)
    assert transform.temperature * scale == pytest.approx(-0.5, abs=1e-12)
    assert transform.density * scale == pytest.approx(-2.0, abs=1e-12)
    assert transform.tau * scale == pytest.approx(1.25, abs=1e-12)

    engineering = transform.engineering_exponents() * scale
    # Ip ~ L B, Bt ~ B, ne ~ n, P ~ n T L^3 / tau, R ~ L, then three zeros.
    assert engineering == pytest.approx(
        np.array([-0.25, -1.25, -2.0, -0.75, 1.0, 0.0, 0.0, 0.0]), abs=1e-12
    )


def test_published_ipb98_lies_on_the_first_two_constraint_surfaces() -> None:
    """The external check on the whole derivation.

    IPB98(y,2)'s exponents are published to two decimal places, so a residual
    below 0.01 is inside the rounding of the law's own coefficients. Landing
    there on surfaces derived from the definitions of rho*, beta and nu* is not
    something a mistaken derivation does.
    """
    residuals = dm.constraint_residuals(sl.IPB98Y2_EXPONENTS).set_index("model")
    assert residuals.loc["kadomtsev", "residual_norm"] < 0.01
    assert residuals.loc["collisionless", "residual_norm"] < 0.01
    # And the third is genuinely violated, so the hierarchy discriminates rather
    # than being satisfied by everything.
    assert residuals.loc["electrostatic", "residual_norm"] > 0.1


def test_constraint_residual_rejects_a_wrong_length_vector() -> None:
    with pytest.raises(ValueError, match="Expected 8 exponents"):
        dm.constraint_residuals(np.ones(3))


# --- the constrained fit ----------------------------------------------------


@pytest.mark.parametrize("model", dm.CONSTRAINT_MODELS)
def test_constrained_fit_actually_satisfies_its_constraint(model: str) -> None:
    """``C b = d`` to machine precision, which the KKT solve should deliver exactly."""
    dataset = _make_dataset()
    fitted = dm.fit_constrained_power_law(dataset, hdb5.TARGET_COLUMN, model)
    vector = np.array([fitted[name] for name in dm.CONSTRAINED_FEATURE_COLUMNS])
    rows, rhs = dm.constraint_matrix(model, intercept=False)
    assert np.abs(rows @ vector - rhs).max() < 1e-9


def test_free_model_reproduces_the_unconstrained_least_squares_fit() -> None:
    """The anchor of the hierarchy is the fit Result 2 already reports."""
    dataset = _make_dataset()
    constrained = dm.fit_constrained_power_law(dataset, hdb5.TARGET_COLUMN, "free")
    plain = sl.fit_scaling_law(dataset, hdb5.TARGET_COLUMN)
    for name in dm.CONSTRAINED_FEATURE_COLUMNS:
        assert constrained[name] == pytest.approx(plain.exponents[name], rel=1e-8, abs=1e-10)


def test_constraining_never_improves_the_in_sample_fit() -> None:
    """A constrained optimum cannot beat the unconstrained one on its own rows.

    Elementary, and worth asserting anyway: it is the property that would break
    first if the KKT system were assembled with a sign error, which is otherwise
    hard to see because a sign-flipped constraint still produces a plausible fit.
    """
    dataset = _make_dataset()
    features = dataset[list(hdb5.BLIND_FEATURE_COLUMNS)]
    log_tau = np.log(dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float))

    def rmsle(model: str) -> float:
        estimator = dm.ConstrainedPowerLaw(model=model).fit(features, log_tau)
        return float(np.sqrt(np.mean((log_tau - estimator.predict(features)) ** 2)))

    free = rmsle("free")
    for model in dm.CONSTRAINT_MODELS:
        assert rmsle(model) >= free - 1e-12


def test_constrained_power_law_is_a_cloneable_regressor() -> None:
    estimator = dm.ConstrainedPowerLaw(model="collisionless")
    assert is_regressor(estimator)
    assert clone(estimator).get_params()["model"] == "collisionless"


def test_constrained_power_law_rejects_an_unknown_model() -> None:
    dataset = _make_dataset()
    features = dataset[list(hdb5.BLIND_FEATURE_COLUMNS)]
    log_tau = np.log(dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float))
    with pytest.raises(ValueError, match="Unknown model"):
        dm.ConstrainedPowerLaw(model="gyrobohm").fit(features, log_tau)


def test_constrained_power_law_needs_named_columns() -> None:
    """A bare array cannot say which column is which, so it is refused.

    Silently taking the first eight columns of an ndarray would fit a constraint
    on the wrong variables, which is the failure this whole file exists to make
    impossible.
    """
    dataset = _make_dataset()
    log_tau = np.log(dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float))
    with pytest.raises(TypeError, match="DataFrame"):
        dm.ConstrainedPowerLaw().fit(
            dataset[list(hdb5.BLIND_FEATURE_COLUMNS)].to_numpy(), log_tau
        )


# --- the prior-shrinkage family --------------------------------------------


@pytest.mark.parametrize("weighting", sp.WEIGHTINGS)
def test_zero_penalty_reproduces_the_unconstrained_fit(weighting: str) -> None:
    dataset = _make_dataset()
    features = dataset[list(hdb5.BLIND_FEATURE_COLUMNS)]
    log_tau = np.log(dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float))
    estimator = sp.SpectralPriorRidge(weighting=weighting, alpha=0.0).fit(features, log_tau)
    plain = sl.fit_scaling_law(dataset, hdb5.TARGET_COLUMN)
    for name in sp.PRIOR_FEATURE_COLUMNS:
        assert estimator.exponent_map_[name] == pytest.approx(
            plain.exponents[name], rel=1e-6, abs=1e-8
        )


@pytest.mark.parametrize("weighting", sp.WEIGHTINGS)
def test_infinite_penalty_reproduces_the_published_exponents(weighting: str) -> None:
    """The far end of every sweep must be IPB98(y,2) itself, not merely near it."""
    dataset = _make_dataset()
    features = dataset[list(hdb5.BLIND_FEATURE_COLUMNS)]
    log_tau = np.log(dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float))
    estimator = sp.SpectralPriorRidge(weighting=weighting, alpha=1e14).fit(features, log_tau)
    for name in sp.PRIOR_FEATURE_COLUMNS:
        assert estimator.exponent_map_[name] == pytest.approx(
            sl.IPB98Y2_EXPONENTS[name], abs=1e-6
        )


def test_truncation_endpoints_are_the_two_anchor_models() -> None:
    """Full rank is the data's fit; rank zero is the prior's exponents."""
    dataset = _make_dataset()
    features = dataset[list(hdb5.BLIND_FEATURE_COLUMNS)]
    log_tau = np.log(dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float))
    plain = sl.fit_scaling_law(dataset, hdb5.TARGET_COLUMN)

    full = sp.SpectralPriorRidge(
        weighting="truncated", n_data_directions=len(sp.PRIOR_FEATURE_COLUMNS)
    ).fit(features, log_tau)
    none = sp.SpectralPriorRidge(weighting="truncated", n_data_directions=0).fit(
        features, log_tau
    )
    for name in sp.PRIOR_FEATURE_COLUMNS:
        assert full.exponent_map_[name] == pytest.approx(plain.exponents[name], rel=1e-6, abs=1e-8)
        assert none.exponent_map_[name] == pytest.approx(sl.IPB98Y2_EXPONENTS[name], abs=1e-10)


def test_truncation_rank_is_range_checked() -> None:
    dataset = _make_dataset()
    features = dataset[list(hdb5.BLIND_FEATURE_COLUMNS)]
    log_tau = np.log(dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float))
    with pytest.raises(ValueError, match="n_data_directions"):
        sp.SpectralPriorRidge(weighting="truncated", n_data_directions=99).fit(features, log_tau)


def test_spectral_weighting_penalises_weak_directions_harder_than_isotropic() -> None:
    """The whole point of the targeting, as a property of the filter itself.

    At a penalty strength chosen so both leave the strongest direction almost
    untouched, the spectral filter must still have collapsed the weakest one.
    Otherwise "targeted" is a description rather than a behaviour.
    """
    singular_values = np.array([100.0, 50.0, 10.0, 1.0])
    alpha = 100.0
    isotropic = sp.direction_filters(singular_values, alpha, "isotropic")
    spectral = sp.direction_filters(singular_values, alpha, "spectral")

    assert isotropic[0] == pytest.approx(spectral[0], abs=0.02)
    assert spectral[-1] < isotropic[-1]
    # Both filters must be decreasing: a later direction is never trusted more.
    assert np.all(np.diff(isotropic) <= 1e-12)
    assert np.all(np.diff(spectral) <= 1e-12)


def test_direction_filters_reject_a_negative_penalty() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        sp.direction_filters(np.array([1.0]), -1.0, "isotropic")


def test_prior_model_names_round_trip_through_the_sweep() -> None:
    """The zoo keys the sweep parses must be the keys the builder produced."""
    models = sp.build_prior_shrinkage_models()
    assert sp.prior_model_name("spectral", alpha=1000.0) in models
    assert sp.prior_model_name("truncated", rank=0) in models
    assert sp.prior_model_name("truncated", rank=len(sp.PRIOR_FEATURE_COLUMNS)) in models
    with pytest.raises(ValueError, match="named by rank"):
        sp.prior_model_name("truncated")
    with pytest.raises(ValueError, match="named by alpha"):
        sp.prior_model_name("spectral")
