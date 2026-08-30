"""Tests for the leave-one-tokamak-out extrapolation study (Result 4).

Two things here are worth more than the usual shape assertions:

* ``test_held_out_machine_is_never_in_its_own_training_set`` is a behavioural
  leakage test. A shape check cannot tell a correct hold-out from a broken one.
* ``test_forest_predictions_are_bounded_by_the_training_target_range`` pins the
  structural claim Result 4c rests on. If a future scikit-learn made tree
  ensembles able to extrapolate, that narrative would be wrong, and this test is
  what would say so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import analysis_extrapolation as ax
import hdb5


def _make_multi_machine_dataset(
    n_per_machine: int = 120,
    seed: int = 11,
    machines: tuple[str, ...] = ("JET", "AUG", "D3D", "NSTX"),
    offsets: dict[str, float] | None = None,
) -> pd.DataFrame:
    """A prepared, HDB5-shaped dataset with a controllable per-machine offset.

    ``offsets`` multiplies one machine's confinement time, which is how the
    leakage and truncation tests create a machine the others cannot explain.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for index, machine in enumerate(machines):
        n = n_per_machine
        ip = rng.uniform(0.4, 4.0, n)
        bt = rng.uniform(1.0, 5.0, n)
        nel = rng.uniform(1.5, 20.0, n)
        plth = rng.uniform(0.5, 25.0, n)
        rgeo = rng.uniform(0.5, 3.2, n)
        eps = rng.uniform(0.2, 0.7, n)
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
        tau = tau * (offsets or {}).get(machine, 1.0)
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


# --- spearman ---------------------------------------------------------------


def test_spearman_is_one_for_a_monotone_pair_and_minus_one_when_reversed() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert ax.spearman(values, values**3) == pytest.approx(1.0)
    assert ax.spearman(values, -values) == pytest.approx(-1.0)


def test_spearman_uses_midranks_so_ties_do_not_depend_on_input_order() -> None:
    """Ordinal ranking breaks ties arbitrarily; midranks do not."""
    a = np.array([1.0, 2.0, 2.0, 3.0])
    b = np.array([10.0, 20.0, 30.0, 40.0])
    forward = ax.spearman(a, b)
    # Swapping the two tied entries' partners must not move the correlation.
    swapped = ax.spearman(np.array([1.0, 2.0, 2.0, 3.0]), np.array([10.0, 30.0, 20.0, 40.0]))
    assert forward == pytest.approx(swapped)
    assert ax._midranks(a).tolist() == [1.0, 2.5, 2.5, 4.0]


def test_spearman_matches_scipy_on_a_tied_sample() -> None:
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(3)
    a = rng.integers(0, 5, 40).astype(float)  # integers force ties
    b = rng.integers(0, 5, 40).astype(float)
    assert ax.spearman(a, b) == pytest.approx(scipy_stats.spearmanr(a, b).statistic)


def test_spearman_is_nan_for_a_constant_vector_and_rejects_length_mismatch() -> None:
    assert np.isnan(ax.spearman(np.ones(5), np.arange(5.0)))
    assert np.isnan(ax.spearman(np.array([1.0]), np.array([2.0])))
    with pytest.raises(ValueError):
        ax.spearman(np.arange(3.0), np.arange(4.0))


# --- the leak guard ---------------------------------------------------------


def test_blind_feature_set_excludes_the_ipb98_prior() -> None:
    """The prior's exponents were fitted on this database, held-out machine included."""
    assert "log_ipb98y2_tau_s" in hdb5.MODEL_FEATURE_COLUMNS
    assert "log_ipb98y2_tau_s" not in hdb5.BLIND_FEATURE_COLUMNS
    assert set(hdb5.BLIND_FEATURE_COLUMNS) < set(hdb5.MODEL_FEATURE_COLUMNS)


def test_evaluate_models_honours_the_requested_feature_columns() -> None:
    """Both arms of Result 4 must be able to run on one shared feature set."""
    dataset = _make_multi_machine_dataset(n_per_machine=60)
    scores = hdb5.evaluate_models(
        dataset, n_splits=3, feature_columns=hdb5.BLIND_FEATURE_COLUMNS
    )
    assert {score.model_name for score in scores} >= {"ridge_loglinear", "random_forest"}
    assert all(np.isfinite(score.cv_rmsle) for score in scores)


def test_held_out_machine_is_never_in_its_own_training_set() -> None:
    """Give one machine an offset nothing else explains; it must stay unpredicted.

    If the hold-out leaked, a random forest would learn the offset and score
    well on it. Held out properly, it cannot: the offset is not a function of
    any feature.
    """
    offset = 6.0
    dataset = _make_multi_machine_dataset(offsets={"NSTX": offset})
    report = hdb5.leave_one_tokamak_out(dataset, min_rows=30, include_ipb98_reference=False)
    forest = report[report["model_name"] == "random_forest"].set_index("tokamak")["rmsle"]

    # RMSLE is the root mean square log ratio, so a model that learned none of
    # the offset scores almost exactly log(offset) on the held-out machine. A
    # model that leaked would score near zero instead. This pins the hold-out
    # far more tightly than "the error is large".
    assert forest["NSTX"] == pytest.approx(np.log(offset), rel=0.1)
    assert forest.idxmax() == "NSTX"


def test_forest_predictions_are_bounded_by_the_training_target_range() -> None:
    """The structural claim behind Result 4c, asserted directly.

    A tree ensemble predicts an average of training targets, so it cannot emit a
    value above ``max(y_train)`` however far the features point. This is why a
    machine whose confinement times run above the training range is unreachable
    rather than merely hard.
    """
    dataset = _make_multi_machine_dataset(offsets={"JET": 8.0})
    columns = list(hdb5.BLIND_FEATURE_COLUMNS)
    held = dataset[hdb5.TOKAMAK_LABEL_COLUMN] == "JET"
    log_tau = np.log(dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float))

    with hdb5._suppress_benign_matmul_warnings():
        forest = hdb5.clone_pipeline(hdb5.build_model_zoo()["random_forest"])
        forest.fit(dataset.loc[~held, columns], log_tau[~held.to_numpy()])
        predicted = forest.predict(dataset.loc[held, columns])

    train_max = log_tau[~held.to_numpy()].max()
    assert predicted.max() <= train_max + 1e-9
    # The held-out machine genuinely reaches above that ceiling, so the bound bites.
    assert log_tau[held.to_numpy()].max() > train_max

    # A log-linear power law carries no such bound.
    with hdb5._suppress_benign_matmul_warnings():
        ridge = hdb5.clone_pipeline(hdb5.build_model_zoo()["ridge_loglinear"])
        ridge.fit(dataset.loc[~held, columns], log_tau[~held.to_numpy()])
        assert np.isfinite(ridge.predict(dataset.loc[held, columns])).all()


# --- diagnostics ------------------------------------------------------------


def test_diagnostic_distance_grows_when_a_machine_sits_off_the_distribution() -> None:
    dataset = _make_multi_machine_dataset()
    similar = hdb5.extrapolation_diagnostic(dataset, "AUG")

    shifted = dataset.copy()
    outlier = shifted[hdb5.TOKAMAK_LABEL_COLUMN] == "NSTX"
    shifted.loc[outlier, "log_r_m"] = shifted.loc[outlier, "log_r_m"] + 4.0
    assert hdb5.extrapolation_diagnostic(shifted, "NSTX").feature_mahalanobis > (
        similar.feature_mahalanobis
    )


def test_diagnostic_reports_target_headroom_only_when_the_machine_runs_high() -> None:
    high = hdb5.extrapolation_diagnostic(
        _make_multi_machine_dataset(offsets={"JET": 8.0}), "JET"
    )
    assert high.log_target_headroom > 0.0
    assert high.target_above_train_max_fraction > 0.0

    low = hdb5.extrapolation_diagnostic(_make_multi_machine_dataset(), "D3D")
    assert low.log_target_headroom < 0.0
    assert low.target_above_train_max_fraction == 0.0


def test_diagnostic_rejects_an_absent_or_solitary_machine() -> None:
    dataset = _make_multi_machine_dataset(machines=("JET", "AUG"))
    with pytest.raises(ValueError, match="No rows"):
        hdb5.extrapolation_diagnostic(dataset, "NOT_A_TOKAMAK")
    only_jet = dataset[dataset[hdb5.TOKAMAK_LABEL_COLUMN] == "JET"]
    with pytest.raises(ValueError, match="only machine"):
        hdb5.extrapolation_diagnostic(only_jet, "JET")


def test_eligible_tokamaks_filters_by_row_count_and_orders_by_size() -> None:
    dataset = _make_multi_machine_dataset(n_per_machine=40, machines=("JET", "AUG"))
    trimmed = pd.concat(
        [
            dataset[dataset[hdb5.TOKAMAK_LABEL_COLUMN] == "JET"],
            dataset[dataset[hdb5.TOKAMAK_LABEL_COLUMN] == "AUG"].head(5),
        ]
    )
    assert hdb5.eligible_tokamaks(trimmed, min_rows=30) == ["JET"]
    assert hdb5.eligible_tokamaks(trimmed, min_rows=1) == ["JET", "AUG"]
    with pytest.raises(ValueError, match="nothing can be held out"):
        hdb5.leave_one_tokamak_out(trimmed, min_rows=10_000)


# --- the analysis end to end ------------------------------------------------


def test_analysis_scores_both_splits_on_one_shared_feature_set() -> None:
    """The comparison is only meaningful if the split is the sole difference."""
    dataset = _make_multi_machine_dataset()
    analysis = ax.analyze_extrapolation(dataset, min_rows=30, n_splits=3)
    assert analysis.feature_columns == list(hdb5.BLIND_FEATURE_COLUMNS)
    assert "log_ipb98y2_tau_s" not in analysis.feature_columns
    assert analysis.n_machines_held_out == 4
    assert analysis.n_rows == len(dataset)


def test_analysis_reports_a_degradation_factor_consistent_with_its_own_scores() -> None:
    dataset = _make_multi_machine_dataset()
    analysis = ax.analyze_extrapolation(dataset, min_rows=30, n_splits=3)
    for transfer in analysis.transfers:
        assert transfer.degradation_factor == pytest.approx(
            transfer.lomo_mean_rmsle / transfer.cv_rmsle
        )
        assert transfer.lomo_median_rmsle <= transfer.lomo_worst_rmsle
        assert transfer.worst_machine in analysis.machines_held_out


def test_analysis_marks_the_published_law_as_not_blind() -> None:
    """It was fitted on this database, so it is a reference, not a competitor."""
    dataset = _make_multi_machine_dataset()
    analysis = ax.analyze_extrapolation(dataset, min_rows=30, n_splits=3)
    by_name = {transfer.model_name: transfer for transfer in analysis.transfers}
    assert by_name["ipb98y2_analytic"].is_blind is False
    assert by_name["random_forest"].is_blind is True
    assert "ipb98y2_analytic" not in ax.CONTENDER_MODELS
    assert "mean_baseline" not in ax.CONTENDER_MODELS


def test_analysis_ranks_are_a_permutation_within_each_split() -> None:
    dataset = _make_multi_machine_dataset()
    analysis = ax.analyze_extrapolation(dataset, min_rows=30, n_splits=3)
    n = len(analysis.transfers)
    assert sorted(t.cv_rank for t in analysis.transfers) == list(range(1, n + 1))
    assert sorted(t.lomo_rank for t in analysis.transfers) == list(range(1, n + 1))


def test_analysis_reports_truncation_only_for_machines_above_the_training_range() -> None:
    # The offset has to clear the spread of the other machines, not just their
    # median, before a material share of rows becomes unreachable.
    dataset = _make_multi_machine_dataset(offsets={"JET": 50.0})
    analysis = ax.analyze_extrapolation(dataset, min_rows=30, n_splits=3)
    assert [finding.tokamak for finding in analysis.truncation] == ["JET"]
    finding = analysis.truncation[0]
    assert finding.log_headroom > 0.0
    assert finding.headroom_ratio == pytest.approx(np.exp(finding.log_headroom))

    assert finding.fraction_above_train_max >= ax.MIN_TRUNCATED_ROW_FRACTION

    # One machine happening to hold the database record is not truncation; the
    # threshold exists so a single extreme row does not raise the finding.
    clean = ax.analyze_extrapolation(_make_multi_machine_dataset(), min_rows=30, n_splits=3)
    assert clean.truncation == []


def test_analysis_json_is_serializable_and_carries_the_headline_fields() -> None:
    import json

    dataset = _make_multi_machine_dataset()
    payload = ax.analyze_extrapolation(dataset, min_rows=30, n_splits=3).to_json()
    round_tripped = json.loads(json.dumps(payload))
    assert round_tripped["n_contenders"] == len(ax.CONTENDER_MODELS)
    assert isinstance(round_tripped["ranking_exactly_reversed"], bool)
    assert round_tripped["contender_models"] == list(ax.CONTENDER_MODELS)


# --- the control model ------------------------------------------------------


def test_control_model_extrapolates_where_a_tree_ensemble_cannot() -> None:
    """The premise of Result 4d: the control is unbounded, the forest is not.

    This is what makes ``ridge_log_quadratic`` discriminating. It is far more
    flexible than plain ridge, yet unlike the trees it can emit a value above
    the training target range, so any gap between it and plain ridge cannot be
    explained away by the Result 4c bound.

    Probed with a machine one e-fold larger than anything trained on. Since
    ``tau ~ R^1.97``, a form that is linear in the logs must answer above the
    training maximum, and a tree ensemble cannot.
    """
    dataset = _make_multi_machine_dataset()
    columns = list(hdb5.BLIND_FEATURE_COLUMNS)
    log_tau = np.log(dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float))
    train_max = log_tau.max()

    # A device an e-fold larger than any in the training set. ``a = eps * R``, so
    # the minor radius has to move with the major radius to stay a real machine.
    probe = dataset.nlargest(5, "log_r_m")[columns].copy()
    probe["log_r_m"] = probe["log_r_m"] + 1.0
    probe["log_a_m"] = probe["log_a_m"] + 1.0

    predictions = {}
    with hdb5._suppress_benign_matmul_warnings():
        for name, estimator in (
            ("ridge_log_quadratic", hdb5.build_control_models()["ridge_log_quadratic"]),
            ("ridge_loglinear", hdb5.build_model_zoo()["ridge_loglinear"]),
            ("random_forest", hdb5.build_model_zoo()["random_forest"]),
        ):
            model = hdb5.clone_pipeline(estimator)
            model.fit(dataset[columns], log_tau)
            predictions[name] = model.predict(probe)

    assert predictions["ridge_loglinear"].max() > train_max
    assert predictions["ridge_log_quadratic"].max() > train_max
    assert predictions["random_forest"].max() <= train_max + 1e-9


def test_controls_are_scored_under_both_splits_or_neither() -> None:
    """A control only discriminates if it appears in both arms of the comparison."""
    dataset = _make_multi_machine_dataset()

    with_controls = ax.analyze_extrapolation(
        dataset, min_rows=30, n_splits=3, include_controls=True
    )
    names = {transfer.model_name for transfer in with_controls.transfers}
    assert set(ax.CONTROL_MODELS) <= names
    for name in ax.CONTROL_MODELS:
        transfer = next(t for t in with_controls.transfers if t.model_name == name)
        assert np.isfinite(transfer.cv_rmsle)
        assert np.isfinite(transfer.lomo_mean_rmsle)

    without = ax.analyze_extrapolation(dataset, min_rows=30, n_splits=3, include_controls=False)
    assert not set(ax.CONTROL_MODELS) & {t.model_name for t in without.transfers}


def test_controls_are_excluded_from_the_ranking_claim() -> None:
    """The reversal claim is about the three contenders, not about the control."""
    assert not set(ax.CONTROL_MODELS) & set(ax.CONTENDER_MODELS)
    dataset = _make_multi_machine_dataset()
    analysis = ax.analyze_extrapolation(dataset, min_rows=30, n_splits=3, include_controls=True)
    assert analysis.n_contenders == len(ax.CONTENDER_MODELS)


def test_evaluate_models_control_flag_only_adds_the_control() -> None:
    dataset = _make_multi_machine_dataset(n_per_machine=60)
    base = hdb5.evaluate_models(dataset, n_splits=3, feature_columns=hdb5.BLIND_FEATURE_COLUMNS)
    extended = hdb5.evaluate_models(
        dataset,
        n_splits=3,
        feature_columns=hdb5.BLIND_FEATURE_COLUMNS,
        include_controls=True,
    )
    added = {s.model_name for s in extended} - {s.model_name for s in base}
    assert added == set(ax.CONTROL_MODELS)
