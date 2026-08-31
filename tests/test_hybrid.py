"""Tests for Result 6: the power law plus a shrunk correction on its residuals.

Scope split, matching ``tests/test_extrapolation.py``: the estimator and its
zoo wiring in ``hdb5.py`` are tested on synthetic data with a controllable
signal, and the analysis layer in ``analysis_hybrid.py`` on top of them.

Result 6's claims are structural rather than numerical, and each one gets a
test rather than a spot check on a score:

* shrinkage 0 is *exactly* the base model, which is what makes the sweep a
  frontier anchored on a model Result 4 already reported;
* the prediction is affine in the damping factor, so the sweep interpolates
  between two fits rather than between two unrelated models;
* the boosted-tree correction is bounded by its training range, which is the
  Result 4c property the whole mechanism rests on;
* the polynomial correction is *not* bounded, which is why the two families
  come apart across the size cut.

The last two are the ones worth having. If the tree correction ever stopped
being bounded, Result 6c's mechanism would be wrong and every number in it
would still look perfectly reasonable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

import analysis_hybrid as ah
import hdb5


def _make_dataset(
    machines: dict[str, float] | None = None,
    n_per_machine: int = 80,
    seed: int = 11,
) -> pd.DataFrame:
    """A prepared HDB5-shaped dataset spanning a range of machine sizes."""
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
        ) * np.exp(rng.normal(0.0, 0.08, n))
        frames.append(
            pd.DataFrame(
                {
                    "TOK": machine,
                    "SHOT": rng.integers(index * 10_000, index * 10_000 + n // 2, n),
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


def _xy(dataset: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    features = dataset[list(hdb5.BLIND_FEATURE_COLUMNS)]
    return features, np.log(dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float))


def _real_dataset_or_skip() -> pd.DataFrame:
    if not hdb5.default_hdb5_path().exists():
        pytest.skip("HDB5 STD5 not downloaded; run `python3 hdb5.py download`.")
    return hdb5.prepare_dataset()


# --- the estimator ----------------------------------------------------------


@pytest.mark.parametrize("correction", ["ridge", "gbm"])
def test_zero_shrinkage_is_exactly_the_base_ridge(correction: str) -> None:
    """Not "close to": the frontier is anchored on Result 4's ridge row.

    If this drifted, the sweep would start from a model no other result
    reports, and every comparison against "plain ridge" in Result 6 would be
    against something else.
    """
    dataset = _make_dataset()
    features, log_tau = _xy(dataset)

    # The zoo's ridge is called directly here rather than through the hybrid, so
    # the suppression the hybrid applies internally does not cover it. Same
    # spurious BLAS flags on a singular design matrix, same reason to ignore
    # them: see ``hdb5._suppress_benign_matmul_warnings``.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        base = hdb5.build_model_zoo()["ridge_loglinear"].fit(features, log_tau)
        base_prediction = base.predict(features)
    hybrid = hdb5.PowerLawResidualHybrid(correction=correction, shrinkage=0.0).fit(
        features, log_tau
    )
    np.testing.assert_allclose(hybrid.predict(features), base_prediction, atol=1e-12)


@pytest.mark.parametrize("correction", ["ridge", "gbm"])
def test_prediction_is_affine_in_the_damping_factor(correction: str) -> None:
    """pred(lambda) = base + lambda * correction, for every lambda.

    This is what makes the sweep a single family rather than nine unrelated
    fits, and it is the reason the frontier in Result 6a can be read as a
    trajectory instead of a scatter.
    """
    dataset = _make_dataset()
    features, log_tau = _xy(dataset)

    at_zero = hdb5.PowerLawResidualHybrid(correction=correction, shrinkage=0.0).fit(
        features, log_tau
    ).predict(features)
    at_one = hdb5.PowerLawResidualHybrid(correction=correction, shrinkage=1.0).fit(
        features, log_tau
    ).predict(features)

    for shrinkage in (0.25, 0.5, 0.75):
        predicted = hdb5.PowerLawResidualHybrid(
            correction=correction, shrinkage=shrinkage
        ).fit(features, log_tau).predict(features)
        np.testing.assert_allclose(
            predicted, at_zero + shrinkage * (at_one - at_zero), atol=1e-10
        )


def test_boosted_tree_correction_is_bounded_by_its_training_range() -> None:
    """The Result 4c bound, now working for the model rather than against it.

    A tree ensemble can only average training targets, so a correction fitted on
    residuals cannot output anything outside the residual range it saw. That is
    what keeps the hybrid's extrapolation behaviour close to the base power
    law's however far outside the data it is asked to predict, and Result 6c's
    whole mechanism depends on it.
    """
    dataset = _make_dataset()
    features = dataset[list(hdb5.BLIND_FEATURE_COLUMNS)]
    labels = dataset[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()
    log_tau = np.log(dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float))

    # Train on the small machines only, so the held-out rows are a genuine size
    # extrapolation rather than a random split.
    train = np.flatnonzero(np.isin(labels, ["S1", "S2", "M1", "M2"]))
    held = np.flatnonzero(np.isin(labels, ["L1", "L2"]))

    model = hdb5.PowerLawResidualHybrid(correction="gbm", shrinkage=1.0).fit(
        features.iloc[train], log_tau[train]
    )
    train_correction = model.correction_.predict(features.iloc[train])
    held_correction = model.correction_.predict(features.iloc[held])

    assert held_correction.min() >= train_correction.min() - 1e-9
    assert held_correction.max() <= train_correction.max() + 1e-9


def test_polynomial_correction_is_not_bounded_that_way() -> None:
    """The contrast that makes the two families a real comparison.

    Result 6c reports the polynomial correction getting *worse* across the size
    cut while the tree correction gets better. That difference is attributed to
    boundedness, so the polynomial half of the claim needs pinning down too: a
    degree-2 expansion extrapolates without bound, and on a size extrapolation
    it leaves the range it was fitted on.
    """
    dataset = _make_dataset()
    features = dataset[list(hdb5.BLIND_FEATURE_COLUMNS)]
    labels = dataset[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()
    log_tau = np.log(dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float))

    train = np.flatnonzero(np.isin(labels, ["S1", "S2", "M1", "M2"]))
    held = np.flatnonzero(np.isin(labels, ["L1", "L2"]))

    model = hdb5.PowerLawResidualHybrid(
        correction="ridge", shrinkage=1.0, ridge_correction_alpha=1.0
    ).fit(features.iloc[train], log_tau[train])
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        train_correction = model.correction_.predict(features.iloc[train])
        held_correction = model.correction_.predict(features.iloc[held])

    assert (
        held_correction.min() < train_correction.min()
        or held_correction.max() > train_correction.max()
    )


def test_unknown_correction_is_rejected() -> None:
    dataset = _make_dataset()
    features, log_tau = _xy(dataset)
    with pytest.raises(ValueError, match="Unknown correction"):
        hdb5.PowerLawResidualHybrid(correction="wavelets").fit(features, log_tau)


def test_hybrid_survives_sklearn_clone_with_its_parameters() -> None:
    """``leave_one_tokamak_out`` clones every estimator once per machine.

    A ``get_params`` contract violation would silently reset the damping factor
    to its default on every fold, and the sweep would report nine copies of one
    model without failing anywhere.
    """
    pipeline = hdb5.build_hybrid_models((0.3,), corrections=("gbm",))["hybrid_gbm_s0p3"]
    cloned = clone(pipeline)
    assert cloned.named_steps["model"].shrinkage == 0.3
    assert cloned.named_steps["model"].correction == "gbm"


def test_build_hybrid_models_names_match_the_helper() -> None:
    """The analysis layer looks rungs up by name; the two must not drift."""
    models = hdb5.build_hybrid_models((0.0, 0.5, 1.0), corrections=("ridge", "gbm"))
    for correction in ("ridge", "gbm"):
        for shrinkage in (0.0, 0.5, 1.0):
            assert hdb5.hybrid_model_name(correction, shrinkage) in models
    assert len(models) == 6


def test_hybrid_names_round_trip_through_the_parser() -> None:
    for correction in ("ridge", "gbm"):
        for shrinkage in hdb5.SHRINKAGE_GRID:
            name = hdb5.hybrid_model_name(correction, shrinkage)
            assert ah._parse_hybrid_name(name) == (correction, shrinkage)
    assert ah._parse_hybrid_name("random_forest") == ("", pytest.approx(float("nan"), nan_ok=True))


def test_hybrids_score_under_every_split_the_zoo_does() -> None:
    """A rung missing from one split would make the frontier's axes incomparable."""
    dataset = _make_dataset()
    hybrids = hdb5.build_hybrid_models((0.5,), corrections=("gbm",))

    lomo = hdb5.leave_one_tokamak_out(dataset, min_rows=10, extra_models=hybrids)
    cv = hdb5.evaluate_models(
        dataset, feature_columns=hdb5.BLIND_FEATURE_COLUMNS, extra_models=hybrids
    )
    split = hdb5.size_ordered_splits(dataset, min_train_machines=3, min_test_rows=10)[0]
    size = hdb5.score_size_split(dataset, split, extra_models=hybrids)

    assert "hybrid_gbm_s0p5" in set(lomo["model_name"])
    assert "hybrid_gbm_s0p5" in {score.model_name for score in cv}
    assert "hybrid_gbm_s0p5" in set(size["model_name"])


# --- the analysis layer -----------------------------------------------------


def test_selection_never_picks_the_rung_by_its_held_out_score() -> None:
    """The damping factor must be chosen on CV, which is all a practitioner has.

    Selecting on LOMO would be selecting on the test set and would make Result
    6b meaningless. This pins the protocol rather than the number it produces:
    the reported rung is the CV-minimising one, whatever its LOMO score.
    """
    dataset = _make_dataset()
    analysis = ah.analyze_hybrid(
        dataset, shrinkage_grid=(0.0, 0.5, 1.0), corrections=("gbm",), n_resamples=50
    )
    outcome = analysis.selection[0]
    rungs = [p for p in analysis.frontier if p.correction == "gbm"]
    best_cv = min(rungs, key=lambda p: p.cv_rmsle)
    assert outcome.cv_selected_shrinkage == best_cv.shrinkage
    assert outcome.cv_selected_cv_rmsle == pytest.approx(best_cv.cv_rmsle)


def test_frontier_carries_every_rung_under_all_three_splits() -> None:
    dataset = _make_dataset()
    analysis = ah.analyze_hybrid(
        dataset, shrinkage_grid=(0.0, 1.0), corrections=("gbm",), n_resamples=50
    )
    rungs = [point for point in analysis.frontier if point.is_hybrid]
    assert len(rungs) == 2
    for point in rungs:
        assert np.isfinite(point.cv_rmsle)
        assert np.isfinite(point.lomo_mean_rmsle)
        assert np.isfinite(point.size_cut_rmsle)


def test_mechanism_reports_the_bound_holding_on_the_held_out_machines() -> None:
    """Result 6c's mechanism, asserted rather than described.

    The narrative says the correction stays inside its training range on the
    held-out machines. That is the claim a reader cannot check from the score
    table, so it is checked here.
    """
    dataset = _make_dataset()
    splits = hdb5.size_ordered_splits(dataset, min_train_machines=3, min_test_rows=10)
    rows = ah.measure_correction_mechanism(dataset, splits[0], correction="gbm")
    assert all(row.correction_within_training_range for row in rows)
    assert {"__train__", "__held_out__"} <= {row.scope for row in rows}


def test_mechanism_reports_no_bias_fraction_where_there_is_no_bias() -> None:
    """On the training rows the base residual is ~0, so the ratio is undefined.

    Reporting a large finite number there would be a division artifact that
    reads as a real measurement.
    """
    dataset = _make_dataset()
    splits = hdb5.size_ordered_splits(dataset, min_train_machines=3, min_test_rows=10)
    rows = ah.measure_correction_mechanism(dataset, splits[0], correction="gbm")
    train_row = next(row for row in rows if row.scope == "__train__")
    assert train_row.base_residual_mean == pytest.approx(0.0, abs=1e-6)
    assert np.isnan(train_row.bias_fraction_corrected)


# --- against the real database ---------------------------------------------


def test_real_data_hybrid_beats_plain_ridge_across_the_iter_matched_cut() -> None:
    """Result 6c's headline, on the real database.

    Guarded loosely: the claim is the direction and a margin comfortably larger
    than any resampling wobble, not the third decimal place.
    """
    dataset = _real_dataset_or_skip()
    splits = hdb5.size_ordered_splits(dataset)
    iter_split = hdb5.iter_matched_split(dataset, splits)
    hybrid = hdb5.build_hybrid_models((1.0,), corrections=("gbm",))
    scores = hdb5.score_size_split(dataset, iter_split, extra_models=hybrid)
    pooled = scores[scores["scope"] == "__pooled__"].set_index("model_name")["rmsle"]

    assert pooled["hybrid_gbm_s1"] < pooled["ridge_loglinear"]
    assert pooled["hybrid_gbm_s1"] < 0.85 * pooled["ridge_loglinear"]
    # And far below the tree ensembles it is built out of.
    assert pooled["hybrid_gbm_s1"] < 0.5 * pooled["random_forest"]


# --- the robustness sweeps of Result 6e -------------------------------------


def test_size_cut_sweep_reports_losses_as_well_as_wins() -> None:
    """Result 6e's honesty depends on this not filtering to the wins.

    The headline gain is measured at one cut. If ``sweep_size_cuts`` ever
    silently dropped the rungs where the hybrid loses, Result 6e would read as
    a clean generalisation of Result 6c instead of the qualified one it is.
    """
    dataset = _make_dataset()
    rows = ah.sweep_size_cuts(dataset)
    assert rows
    splits = hdb5.size_ordered_splits(dataset)
    assert len(rows) == len(splits)
    for row in rows:
        assert row.hybrid_wins == (row.hybrid_rmsle < row.ridge_rmsle)
        assert row.well_powered == (row.n_train_rows >= ah.MIN_WELL_POWERED_TRAIN_ROWS)


def test_hyperparameter_sweep_covers_the_reported_setting() -> None:
    """The grid has to contain the point Result 6c actually reports.

    A grid that surrounded the headline without including it would compare two
    different models and could not say whether the headline was cherry-picked.
    """
    dataset = _make_dataset()
    splits = hdb5.size_ordered_splits(dataset, min_train_machines=3, min_test_rows=10)
    rows = ah.sweep_correction_hyperparameters(dataset, splits[0])
    assert len(rows) == len(ah.HYPERPARAMETER_GRID_DEPTHS) * len(
        ah.HYPERPARAMETER_GRID_ITERATIONS
    )
    reported = (hdb5.DEFAULT_GBM_CORRECTION_DEPTH, hdb5.DEFAULT_GBM_CORRECTION_ITERATIONS)
    assert reported in {(row.gbm_max_depth, row.gbm_max_iter) for row in rows}


def test_real_data_every_grid_setting_beats_ridge_at_the_matched_cut() -> None:
    """Result 6e's first claim: the gain is not an artifact of one setting."""
    dataset = _real_dataset_or_skip()
    splits = hdb5.size_ordered_splits(dataset)
    rows = ah.sweep_correction_hyperparameters(
        dataset, hdb5.iter_matched_split(dataset, splits)
    )
    assert all(row.beats_ridge for row in rows)
