"""Tests for Result 7: split-conformal intervals and where their coverage goes.

Scope split as elsewhere: the conformal primitives in ``hdb5.py`` are tested on
synthetic data, the analysis layer in ``analysis_conformal.py`` on top of them.

The result being defended is a *shortfall*, which is an unusual thing to test:
almost any bug in the interval construction would also produce a shortfall, and
would look exactly like the finding. So the tests below concentrate on the
control arm and on the construction itself, because those are what license the
interpretation:

* the conformal quantile uses the ``ceil((n+1)(1-alpha))`` rank, so the
  finite-sample guarantee actually holds rather than nearly holding;
* it returns an infinite half-width instead of a wrong finite one when the
  calibration sample is too small for the level to exist;
* calibration splits hold out whole *discharges*, so near-duplicate time slices
  cannot leak between the fit and the calibration set and shrink the interval;
* under exchangeability the coverage lands on nominal.

Only once those hold does a shortfall on a held-out machine mean what Result 7
says it means.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import analysis_conformal as ac
import hdb5


def _make_dataset(
    machines: dict[str, float] | None = None,
    n_per_machine: int = 120,
    seed: int = 17,
) -> pd.DataFrame:
    """A prepared HDB5-shaped dataset with several time slices per discharge."""
    machines = machines or {"S1": 0.6, "S2": 0.9, "M1": 1.4, "M2": 1.8, "L1": 2.8, "L2": 3.3}
    rng = np.random.default_rng(seed)
    frames = []
    for index, (machine, radius) in enumerate(machines.items()):
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
        ) * np.exp(rng.normal(0.0, 0.10, n))
        frames.append(
            pd.DataFrame(
                {
                    "TOK": machine,
                    # Roughly four time slices per discharge, which is what
                    # makes the by-discharge calibration split matter.
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
    raw = pd.concat(frames, ignore_index=True)
    return hdb5.build_features(hdb5.map_to_canonical(raw))


def _real_dataset_or_skip() -> pd.DataFrame:
    if not hdb5.default_hdb5_path().exists():
        pytest.skip("HDB5 STD5 not downloaded; run `python3 hdb5.py download`.")
    return hdb5.prepare_dataset()


# --- the conformal quantile -------------------------------------------------


def test_half_width_uses_the_finite_sample_rank_not_the_plain_quantile() -> None:
    """``ceil((n+1)(1-alpha))``, which is the whole guarantee.

    With n = 19 and alpha = 0.10 the rank is ceil(20 * 0.9) = 18, so the
    half-width is the 18th smallest score. ``numpy.quantile`` would return
    something between the 17th and 18th and undercover. The gap shrinks as 1/n,
    so it is invisible on the pooled arms and material on the small ones.
    """
    scores = np.arange(1.0, 20.0)  # 19 points, 1 through 19
    assert hdb5.split_conformal_half_width(scores, alpha=0.10) == pytest.approx(18.0)


def test_half_width_is_infinite_when_the_level_is_unattainable() -> None:
    """Too few calibration points means no finite interval has the guarantee.

    Returning a finite half-width there would be a number with no property,
    which is worse than an honest infinity.
    """
    # n = 5, alpha = 0.10: rank = ceil(6 * 0.9) = 6 > 5.
    assert np.isinf(hdb5.split_conformal_half_width(np.arange(5.0), alpha=0.10))
    assert np.isinf(hdb5.split_conformal_half_width(np.array([]), alpha=0.10))


def test_half_width_covers_at_least_the_nominal_share_of_its_own_sample() -> None:
    rng = np.random.default_rng(3)
    scores = np.abs(rng.normal(0.0, 0.2, 500))
    half_width = hdb5.split_conformal_half_width(scores, alpha=0.10)
    assert float(np.mean(scores <= half_width)) >= 0.90


def test_half_width_ignores_non_finite_scores() -> None:
    finite = np.array([1.0, 2.0, 3.0, 4.0] * 10)
    with_nan = np.concatenate([finite, [np.nan, np.inf]])
    assert hdb5.split_conformal_half_width(with_nan, alpha=0.10) == pytest.approx(
        hdb5.split_conformal_half_width(finite, alpha=0.10)
    )


# --- the calibration split --------------------------------------------------


def test_calibration_holds_out_whole_discharges() -> None:
    """No discharge may straddle the fit and the calibration set.

    Several time slices from one shot are near-duplicates. A row-level split
    would put a row's own near-twin in the training data, shrinking the
    calibration residuals and returning intervals that are too narrow. That
    would manufacture the CV arm's shortfall out of nothing.
    """
    groups = np.repeat(np.arange(40), 5)
    mask = hdb5._calibration_mask_by_group(groups, calibration_fraction=0.25, seed=0)
    calibration_groups = set(groups[mask])
    fit_groups = set(groups[~mask])
    assert not calibration_groups & fit_groups
    assert calibration_groups and fit_groups


def test_calibration_split_always_leaves_both_sides_non_empty() -> None:
    """Guards the rounding: a tiny fraction must not empty the calibration set."""
    groups = np.repeat(np.arange(6), 3)
    for fraction in (0.01, 0.5, 0.99):
        mask = hdb5._calibration_mask_by_group(
            groups, calibration_fraction=fraction, seed=1
        )
        assert 0 < int(mask.sum()) < mask.size


def test_calibration_split_rejects_a_single_discharge() -> None:
    with pytest.raises(ValueError, match="at least two discharges"):
        hdb5._calibration_mask_by_group(
            np.zeros(10), calibration_fraction=0.25, seed=0
        )


# --- coverage under the two splits ------------------------------------------


def test_grouped_cv_coverage_lands_on_nominal() -> None:
    """The control arm, and the test that licenses reading the others.

    Calibration and test rows are both held-out discharges from machines in the
    training fold, so they are exchangeable and the guarantee applies. If this
    ever failed, the shortfall reported in Result 7b would be a bug in the
    construction rather than a property of the split, and the whole result
    would evaporate.
    """
    dataset = _make_dataset()
    _, summary = hdb5.conformal_coverage_grouped_cv(dataset, n_splits=4)
    pooled = summary[summary["scope"] == "__pooled__"].set_index("model_name")
    for model_name in ("ridge_loglinear", "random_forest"):
        assert pooled.loc[model_name, "empirical_coverage"] == pytest.approx(0.90, abs=0.05)


def test_reported_intervals_carry_their_width() -> None:
    """Coverage without width is not a result: a wide enough interval covers all."""
    dataset = _make_dataset()
    _, summary = hdb5.conformal_coverage_grouped_cv(dataset, n_splits=4)
    assert (summary["median_half_width_log"] > 0).all()
    np.testing.assert_allclose(
        summary["median_interval_factor"].to_numpy(dtype=float),
        np.exp(summary["median_half_width_log"].to_numpy(dtype=float)),
    )


def test_leave_one_tokamak_out_scores_every_eligible_machine() -> None:
    dataset = _make_dataset()
    _, summary = hdb5.conformal_coverage_leave_one_tokamak_out(dataset, min_rows=10)
    machines = set(summary[summary["scope"] != "__pooled__"]["scope"])
    assert machines == set(hdb5.eligible_tokamaks(dataset, min_rows=10))


def test_size_cut_coverage_scores_only_machines_above_the_cut() -> None:
    """A machine the model trained on must never appear in the held-out scope."""
    # The earliest cut trains on three machines, so the dataset has to be big
    # enough that 25% of those discharges still clears MIN_CALIBRATION_ROWS.
    # On the real database the ITER-matched cut trains on 3498 rows.
    dataset = _make_dataset(n_per_machine=200)
    splits = hdb5.size_ordered_splits(dataset, min_train_machines=3, min_test_rows=10)
    split = splits[0]
    _, summary = hdb5.conformal_coverage_size_split(dataset, split)
    scopes = set(summary[summary["scope"] != "__pooled__"]["scope"])
    assert scopes <= set(split.test_machines)
    assert not scopes & set(split.train_machines)


def test_arm_refuses_to_run_on_too_few_calibration_rows() -> None:
    """Rather than silently reporting a quantile of a handful of points."""
    dataset = _make_dataset(machines={"A": 1.0, "B": 2.0}, n_per_machine=40)
    labels = dataset[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()
    with pytest.raises(ValueError, match="calibration rows"):
        hdb5._conformal_arm(
            dataset,
            train_index=np.flatnonzero(labels == "A"),
            test_index=np.flatnonzero(labels == "B"),
            zoo=hdb5.build_model_zoo(),
            feature_columns=hdb5.BLIND_FEATURE_COLUMNS,
            alpha=0.10,
            calibration_fraction=0.25,
            seed=0,
            include_ipb98_reference=False,
        )


# --- the analysis layer -----------------------------------------------------


def test_analysis_reports_every_model_under_all_three_splits() -> None:
    dataset = _make_dataset()
    analysis = ac.analyze_conformal(dataset)
    assert analysis.collapse
    for row in analysis.collapse:
        assert 0.0 <= row.cv_coverage <= 1.0
        assert 0.0 <= row.lomo_coverage <= 1.0
        assert 0.0 <= row.size_cut_coverage <= 1.0
        assert row.coverage_shortfall == pytest.approx(row.cv_coverage - row.lomo_coverage)


def test_analysis_attaches_the_same_distance_result_4b_uses() -> None:
    """Coverage and point error must be read against one common x axis."""
    dataset = _make_dataset()
    analysis = ac.analyze_conformal(dataset)
    scored = analysis.per_machine[analysis.per_machine["feature_mahalanobis"].notna()]
    for machine, distance in scored[["scope", "feature_mahalanobis"]].drop_duplicates().values:
        expected = hdb5.extrapolation_diagnostic(dataset, str(machine)).feature_mahalanobis
        assert float(distance) == pytest.approx(expected)


# --- against the real database ---------------------------------------------


def test_real_data_coverage_holds_in_distribution_and_collapses_out_of_it() -> None:
    """Result 7's headline: the control holds, the held-out arm does not.

    Both halves are asserted together on purpose. The shortfall only means
    something if the same construction hits nominal on the split where the
    guarantee applies, so a regression that broke the intervals everywhere
    would fail this test rather than strengthening the reported finding.
    """
    dataset = _real_dataset_or_skip()
    analysis = ac.analyze_conformal(dataset)
    by_model = {row.model_name: row for row in analysis.collapse}

    for row in analysis.collapse:
        assert row.cv_coverage == pytest.approx(0.90, abs=0.03)

    # The forest is confidently wrong on a machine it has not seen, and the
    # power law much less so.
    assert by_model["random_forest"].lomo_coverage < 0.60
    assert by_model["ridge_loglinear"].lomo_coverage > by_model["random_forest"].lomo_coverage
    # And across the ITER-matched size cut the trees cover almost nothing.
    assert by_model["random_forest"].size_cut_coverage < 0.10


def test_real_data_intervals_do_not_widen_to_compensate() -> None:
    """The failure mode is confidence, not width.

    If the half-widths grew out of distribution the models would merely be
    vague, which is a far less serious problem than being narrow and wrong.
    They do not grow: the calibration rows are drawn the same way in both arms,
    so the interval is the same size and simply misses.
    """
    dataset = _real_dataset_or_skip()
    analysis = ac.analyze_conformal(dataset)
    for row in analysis.collapse:
        assert row.lomo_interval_factor == pytest.approx(row.cv_interval_factor, rel=0.25)
