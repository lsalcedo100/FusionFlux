"""Six ways the headline could be an artifact of a choice rather than the data.

Each of these is an objection a referee would raise, turned into a measurement:
newer published scalings as baselines instead of IPB98(y,2) alone; whether the
distance correlation survives its own small-sample uncertainty; whether pooling
rows lets two machines carry the result; whether error on the *predictors*
rather than the target moves the exponents; whether splitting by discharge
rather than by row leaks; and whether the redundant ninth feature matters.

The risk in a file like this is that a check looks rigorous and tests nothing.
A permutation p-value computed against a reshuffled copy of the wrong array, a
Spearman that is really a Pearson, an ODR seeded so close to the OLS answer that
it cannot move: all pass silently and all would make the paper overclaim. These
tests pin the statistics against cases whose answers are known independently.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import analysis_sensitivity as asens
import hdb5

RESULTS = Path(__file__).resolve().parents[1] / "results" / "sensitivity.json"


@pytest.fixture(scope="module")
def committed() -> dict:
    if not RESULTS.exists():
        pytest.skip("no results/sensitivity.json; run `python3 analysis_sensitivity.py`")
    return json.loads(RESULTS.read_text())


# --- the rank correlation is a rank correlation -----------------------------


def test_spearman_is_one_for_a_monotone_nonlinear_relation() -> None:
    """The distinguishing property: Pearson would not give 1 here."""
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = a**5
    assert asens._spearman(a, b) == pytest.approx(1.0)
    assert np.corrcoef(a, b)[0, 1] < 0.95, "the test case must separate rank from linear"


def test_spearman_is_minus_one_when_reversed() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert asens._spearman(a, -a) == pytest.approx(-1.0)


def test_spearman_handles_ties_by_averaging_ranks() -> None:
    a = np.array([1.0, 1.0, 2.0, 3.0])
    b = np.array([1.0, 1.0, 2.0, 3.0])
    assert asens._spearman(a, b) == pytest.approx(1.0)


# --- the permutation test and the jackknife ---------------------------------


def _per_machine(error: list[float], distance: list[float], model: str = "m") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_name": [model] * len(error),
            "rmsle": error,
            "feature_mahalanobis": distance,
        }
    )


def test_a_perfect_correlation_gets_a_small_permutation_p() -> None:
    frame = _per_machine([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    result = asens.correlation_uncertainty(frame)["m"]

    assert result["spearman"] == pytest.approx(1.0)
    assert result["permutation_p_two_sided"] < 0.01
    assert result["n_machines"] == 7


def test_an_unrelated_pair_is_not_significant() -> None:
    rng = np.random.default_rng(0)
    error = list(rng.normal(size=13))
    distance = list(rng.normal(size=13))
    result = asens.correlation_uncertainty(_per_machine(error, distance))["m"]

    assert result["permutation_p_two_sided"] > 0.05


def test_the_jackknife_range_brackets_the_observed_correlation() -> None:
    """Leaving one machine out cannot move rho outside its own reported range."""
    rng = np.random.default_rng(4)
    distance = list(rng.uniform(0, 5, 13))
    error = [d * 0.4 + rng.normal(0, 0.3) for d in distance]
    result = asens.correlation_uncertainty(_per_machine(error, distance))["m"]

    assert result["jackknife_min"] <= result["jackknife_max"]
    assert result["jackknife_min"] <= result["spearman"] + 1e-9
    assert result["spearman"] <= result["jackknife_max"] + 1e-9


def test_the_permutation_null_is_actually_reshuffled() -> None:
    """A null built without permuting would give p = 1 for every input."""
    perfect = asens.correlation_uncertainty(
        _per_machine([1.0, 2, 3, 4, 5, 6, 7, 8], [1.0, 2, 3, 4, 5, 6, 7, 8])
    )["m"]
    scrambled = asens.correlation_uncertainty(
        _per_machine([1.0, 2, 3, 4, 5, 6, 7, 8], [5.0, 1, 8, 2, 7, 3, 6, 4])
    )["m"]
    assert perfect["permutation_p_two_sided"] < scrambled["permutation_p_two_sided"]


def test_the_permutation_is_seeded_so_the_artifact_reproduces() -> None:
    frame = _per_machine([1.0, 2, 3, 4, 5, 6, 7], [2.0, 1, 4, 3, 6, 5, 7])
    first = asens.correlation_uncertainty(frame)["m"]
    second = asens.correlation_uncertainty(frame)["m"]
    assert first == second


# --- the published scalings are transcribed, not fitted ---------------------


def test_both_published_scalings_carry_every_exponent_the_law_needs() -> None:
    required = {
        "coefficient",
        "ip_ma",
        "bt_t",
        "ne_line_1e19_m3",
        "p_loss_mw",
        "r_m",
        "one_plus_delta",
        "kappa",
        "inverse_aspect_ratio",
        "m_eff_amu",
    }
    for name, law in asens.PUBLISHED_SCALINGS.items():
        assert set(law) == required, f"{name} is missing {required - set(law)}"


def test_the_power_dependence_is_negative_in_both() -> None:
    """More heating gives worse confinement; a sign flip here would be a typo."""
    for name, law in asens.PUBLISHED_SCALINGS.items():
        assert law["p_loss_mw"] < 0, f"{name} has a non-negative power exponent"
        assert law["ip_ma"] > 0, f"{name} has a non-positive current exponent"


def test_the_two_laws_are_actually_different_laws() -> None:
    itpa20 = asens.PUBLISHED_SCALINGS["ITPA20"]
    itpa20_il = asens.PUBLISHED_SCALINGS["ITPA20-IL"]
    assert itpa20 != itpa20_il
    # The IL variant's defining features: stronger current, weaker size, no
    # aspect-ratio term at all.
    assert itpa20_il["ip_ma"] > itpa20["ip_ma"]
    assert itpa20_il["r_m"] < itpa20["r_m"]
    assert itpa20_il["inverse_aspect_ratio"] == 0.0


# --- the committed artifact -------------------------------------------------


def test_every_sensitivity_arm_is_present(committed: dict) -> None:
    for arm in (
        "published_scalings",
        "correlation_uncertainty",
        "machine_equal_weighting",
        "errors_in_variables",
        "discharge_disjoint",
        "redundant_feature",
    ):
        assert arm in committed, f"the artifact is missing the {arm} arm"


def test_the_published_baselines_include_the_newer_laws(committed: dict) -> None:
    """IPB98 alone would leave "you used an outdated baseline" unanswered."""
    assert {"ITPA20", "ITPA20-IL", "IPB98(y,2)"} <= set(committed["published_scalings"])


def test_odr_moves_the_exponents_at_all(committed: dict) -> None:
    """If ODR returned the OLS answer the check would be vacuous."""
    eiv = committed["errors_in_variables"]
    assert eiv["max_abs_exponent_shift"] > 0.0
    assert eiv["largest_shift_feature"] in eiv["feature_columns"]
    assert len(eiv["ols_exponents"]) == len(eiv["odr_exponents"])


def test_the_permutation_draw_count_is_recorded(committed: dict) -> None:
    """A p-value is uninterpretable without the number of draws behind it."""
    assert committed["permutation_draws"] == asens.PERMUTATION_DRAWS
    assert committed["permutation_draws"] >= 10_000

# --- the analysis functions themselves, on a small synthetic frame ---------


def _dataset(n_per_machine: int = 45, seed: int = 7) -> pd.DataFrame:
    """A prepared HDB5-shaped frame, with the triangularity column kept.

    `dataset_with_triangularity` exists because `map_to_canonical` drops
    DELTA1, which the ITPA20 laws need. Building the frame the same way here
    lets the published-scaling arms be exercised without the real download.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for index, (machine, radius) in enumerate(
        {"A": 0.8, "B": 1.4, "C": 2.1, "D": 2.9}.items()
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
        delta = rng.uniform(0.1, 0.5, n)
        tau = (
            0.0562 * ip**0.93 * bt**0.15 * nel**0.41 * plth**-0.69
            * rgeo**1.97 * eps**0.58 * kappa**0.78 * meff**0.19
        ) * np.exp(rng.normal(0.0, 0.08, n))
        frames.append(
            pd.DataFrame(
                {
                    "TOK": machine,
                    "SHOT": rng.integers(index * 10_000, index * 10_000 + n // 3, n),
                    "TIME": rng.uniform(1.0, 5.0, n),
                    "TAUTH": tau, "IP": ip, "BT": bt, "NEL": nel, "PLTH": plth,
                    "RGEO": rgeo, "DELTA1": delta, "KAPPAA": kappa, "EPS": eps, "MEFF": meff,
                }
            )
        )
    raw = pd.concat(frames, ignore_index=True)
    dataset = hdb5.build_features(hdb5.map_to_canonical(raw))
    dataset["one_plus_delta"] = 1.0 + raw["DELTA1"].to_numpy()[: len(dataset)]
    return dataset


def test_published_prediction_is_the_law_evaluated_not_fitted() -> None:
    """Recompute one row by hand; a fitted coefficient would not match."""
    dataset = _dataset()
    predicted = asens.published_prediction(dataset, "ITPA20")

    law = asens.PUBLISHED_SCALINGS["ITPA20"]
    expected = law["coefficient"]
    for column, exponent in law.items():
        if column == "coefficient" or exponent == 0.0:
            continue
        expected = expected * float(dataset[column].iloc[0]) ** exponent

    assert predicted[0] == pytest.approx(expected)
    assert len(predicted) == len(dataset)
    assert (predicted > 0).all()


def test_a_zero_exponent_contributes_nothing() -> None:
    """ITPA20-IL has no aspect-ratio term, so that column must not enter."""
    dataset = _dataset()
    baseline = asens.published_prediction(dataset, "ITPA20-IL")

    altered = dataset.copy()
    altered["inverse_aspect_ratio"] = altered["inverse_aspect_ratio"] * 2.0
    assert asens.published_prediction(altered, "ITPA20-IL") == pytest.approx(baseline)

    # The same change must move ITPA20, which does carry the term.
    assert asens.published_prediction(altered, "ITPA20") != pytest.approx(
        asens.published_prediction(dataset, "ITPA20")
    )


def test_score_published_scores_every_law_under_every_split() -> None:
    scores = asens.score_published(_dataset())

    assert {"ITPA20", "ITPA20-IL", "IPB98(y,2)"} <= set(scores)
    for name, arms in scores.items():
        for value in arms.values():
            if isinstance(value, float):
                assert value >= 0.0, f"{name} produced a negative RMSLE"


def test_errors_in_variables_returns_a_different_fit_from_ols() -> None:
    """ODR must actually move, or the sensitivity check is vacuous."""
    result = asens.errors_in_variables(_dataset())

    assert len(result["ols_exponents"]) == len(result["odr_exponents"])
    assert result["max_abs_exponent_shift"] > 0.0
    assert result["largest_shift_feature"] in result["feature_columns"]
    assert result["odr_in_sample_rmsle"] > 0.0


def test_machine_equal_weighting_reports_both_weightings() -> None:
    """Two machines supply 77% of the real rows, so the weighting is a real choice."""
    result = asens.machine_equal_weighting(_dataset())

    assert set(result) == {"unweighted", "machine_equal"}
    for arm in result.values():
        assert arm, "an empty weighting arm cannot support a comparison"
