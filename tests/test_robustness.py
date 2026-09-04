"""Does the inversion survive the analyst's discretionary choices?

The headline result compares one cross-validation arm against one
leave-one-out arm, and three choices inside that comparison were made by a
person rather than forced by the data: whether JET-ILW counts as JET or as its
own machine, whether error is pooled over rows or averaged over machines, and
whether cross-validation is scored on all rows or only the machines
leave-one-out can score. Each is defensible, and each could be the reason the
inversion appears.

This analysis runs the comparison under every combination. These tests pin the
part that matters: that the grid is actually a grid (no combination silently
dropped), that the sign test is exact rather than approximate, and that
``inversion_holds_everywhere`` cannot report true while one of its cells is
false.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import analysis_robustness as ar
import hdb5

RESULTS = Path(__file__).resolve().parents[1] / "results" / "robustness.json"


@pytest.fixture(scope="module")
def committed() -> dict:
    if not RESULTS.exists():
        pytest.skip("no results/robustness.json; run `python3 analysis_robustness.py`")
    return json.loads(RESULTS.read_text())


# --- the sign test, checked against values worked out by hand ---------------


def test_sign_test_is_exact_on_a_case_with_a_known_answer() -> None:
    """13 of 13 one way is 2/2^13, the number the paper's 13-of-13 claim rests on."""
    a = {str(i): 1.0 for i in range(13)}
    b = {str(i): 0.0 for i in range(13)}
    result = ar._sign_test(a, b)

    assert result["n_units"] == 13
    assert result["n_units_a_worse"] == 13
    assert result["exact_two_sided_p"] == pytest.approx(2 / 2**13)


def test_sign_test_of_an_even_split_is_not_significant() -> None:
    a = {str(i): (1.0 if i % 2 else 0.0) for i in range(10)}
    b = {str(i): (0.0 if i % 2 else 1.0) for i in range(10)}
    result = ar._sign_test(a, b)

    assert result["n_units_a_worse"] == 5
    assert result["exact_two_sided_p"] == pytest.approx(1.0)


def test_sign_test_p_value_is_a_probability() -> None:
    """The two-sided doubling must be clamped, or a near-even split exceeds one."""
    for wins in range(0, 14):
        a = {str(i): (1.0 if i < wins else 0.0) for i in range(13)}
        b = {str(i): (0.0 if i < wins else 1.0) for i in range(13)}
        p = ar._sign_test(a, b)["exact_two_sided_p"]
        assert isinstance(p, float)
        assert 0.0 < p <= 1.0


def test_sign_test_mean_difference_signs_with_the_first_argument() -> None:
    a = {"x": 0.5, "y": 0.7}
    b = {"x": 0.2, "y": 0.3}
    assert ar._sign_test(a, b)["mean_difference"] == pytest.approx(0.35)
    assert ar._sign_test(b, a)["mean_difference"] == pytest.approx(-0.35)


# --- both aggregations measure what they say --------------------------------


def test_pooled_and_unit_equal_differ_when_units_are_unbalanced() -> None:
    """The whole point of reporting both: one large unit can carry the pooled score."""
    actual = np.array([1.0] * 100 + [1.0] * 4)
    units = np.array(["big"] * 100 + ["small"] * 4)
    predicted = np.concatenate([np.full(100, 1.0), np.full(4, 4.0)])

    scores = ar._both_aggregations(actual, units, {"m": predicted})["m"]

    assert scores["n_units"] == 2
    assert scores["pooled_rows"] < scores["unit_equal"], (
        "a small badly-predicted unit must weigh more when units are equal-weighted"
    )
    assert scores["per_unit"]["big"] == pytest.approx(0.0, abs=1e-12)


def test_unit_equal_is_the_mean_of_the_per_unit_scores() -> None:
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    units = np.array(["a", "a", "b", "b"])
    predicted = np.array([1.1, 2.2, 2.7, 4.4])

    scores = ar._both_aggregations(actual, units, {"m": predicted})["m"]
    assert scores["unit_equal"] == pytest.approx(np.mean(list(scores["per_unit"].values())))


# --- the committed artifact -------------------------------------------------


def test_the_wall_variant_machines_are_the_ones_remapped(committed: dict) -> None:
    """JET-ILW and AUG-W are the same physical device as JET and AUG."""
    assert committed["physical_device_map"] == {"JETILW": "JET", "AUGW": "AUG"}


def test_every_arm_combination_is_present(committed: dict) -> None:
    """A dropped cell would let `holds everywhere` mean `holds where we looked`."""
    cv_arms = ["cv_all_rows", "cv_scored_machines_only"]
    lomo_arms = ["lomo_by_database_label", "lomo_by_physical_device"]
    for arm in cv_arms + lomo_arms:
        assert arm in committed["arms"]

    expected = {
        f"{cv}|{lomo}|{agg}"
        for cv in cv_arms
        for lomo in lomo_arms
        for agg in ("pooled_rows", "unit_equal")
    }
    assert set(committed["inversion_holds"]) == expected


def test_holds_everywhere_is_the_conjunction_of_its_cells(committed: dict) -> None:
    """It must not be able to report true while a cell is false."""
    cells = committed["inversion_holds"]
    assert committed["inversion_holds_everywhere"] is all(cells.values())


def test_the_inversion_survives_every_discretionary_choice(committed: dict) -> None:
    """The claim the analysis exists to make."""
    assert committed["inversion_holds_everywhere"] is True, (
        "the inversion does not hold under every choice; "
        f"failing cells: {[k for k, v in committed['inversion_holds'].items() if not v]}"
    )

# --- the analysis end to end, on a small synthetic frame -------------------


def _dataset(n_per_machine: int = 45, seed: int = 9) -> pd.DataFrame:
    """Four machines, two of them wall variants of the same physical device."""
    rng = np.random.default_rng(seed)
    frames = []
    for index, (machine, radius) in enumerate(
        {"JET": 2.9, "JETILW": 2.9, "AUG": 1.6, "C-Mod": 0.7}.items()
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
        ) * np.exp(rng.normal(0.0, 0.08, n))
        frames.append(
            pd.DataFrame(
                {
                    "TOK": machine,
                    "SHOT": rng.integers(index * 10_000, index * 10_000 + n // 3, n),
                    "TIME": rng.uniform(1.0, 5.0, n),
                    "TAUTH": tau, "IP": ip, "BT": bt, "NEL": nel, "PLTH": plth,
                    "RGEO": rgeo, "DELTA1": rng.uniform(0.1, 0.5, n),
                    "KAPPAA": kappa, "EPS": eps, "MEFF": meff,
                }
            )
        )
    return hdb5.build_features(hdb5.map_to_canonical(pd.concat(frames, ignore_index=True)))


def test_wall_variants_collapse_onto_one_physical_device() -> None:
    """JET and JET-ILW are one tokamak with two walls, which is the whole point."""
    dataset = _dataset()
    labels = dataset[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()
    devices = ar._device_labels(dataset)

    assert set(labels) >= {"JET", "JETILW"}
    assert "JETILW" not in set(devices)
    assert (devices[labels == "JETILW"] == "JET").all()
    assert len(set(devices)) < len(set(labels))


def test_cross_validate_scores_every_contender_both_ways() -> None:
    dataset = _dataset()
    labels = dataset[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()
    scores = ar._cross_validate(dataset, labels)

    for name in ar.CONTENDERS:
        assert name in scores
        assert scores[name]["pooled_rows"] > 0.0
        assert scores[name]["unit_equal"] > 0.0
        assert scores[name]["n_units"] == len(set(labels))


def test_leave_one_unit_out_holds_each_unit_out_in_turn() -> None:
    dataset = _dataset()
    devices = ar._device_labels(dataset)
    scores = ar._leave_one_unit_out(dataset, devices)

    for name in ar.CONTENDERS:
        assert set(scores[name]["per_unit"]) <= set(devices)
    assert ar.REFERENCE in scores, "the published law must be scored alongside"


def test_the_analysis_reports_every_cell_of_its_own_grid() -> None:
    """`inversion_holds_everywhere` has to be the conjunction it claims to be."""
    analysis = ar.analyze_robustness(_dataset())

    arms = analysis["arms"]
    assert isinstance(arms, dict)
    assert set(arms) == {
        "cv_all_rows",
        "cv_scored_machines_only",
        "lomo_by_database_label",
        "lomo_by_physical_device",
    }
    cells = analysis["inversion_holds"]
    assert isinstance(cells, dict)
    assert analysis["inversion_holds_everywhere"] is all(cells.values())
