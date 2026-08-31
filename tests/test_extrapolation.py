"""Tests for ``analysis_extrapolation``: the Result 4 study built on top of hdb5.

Scope split: ``tests/test_hdb5.py`` owns the extrapolation primitives in
``hdb5.py`` (the diagnostic, the hold-out loop, the report join). This file owns
the analysis layer, plus the two behavioural claims that need a dataset with a
controllable per-machine offset:

* ``test_held_out_machine_is_never_in_its_own_training_set`` is a leakage test.
  A shape check cannot tell a correct hold-out from a broken one; this pins the
  held-out error to the size of an offset nothing in the features explains.
* ``test_control_model_extrapolates_where_a_tree_ensemble_cannot`` pins the
  premise of Result 4d. The control only discriminates if it is genuinely
  unbounded, so a gap between it and plain ridge cannot be blamed on Result 4c.
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


# --- the shared feature set -------------------------------------------------


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


def test_eligible_tokamaks_orders_machines_by_descending_size() -> None:
    """The documented ordering. ``tests/test_hdb5.py`` covers the row-count filter."""
    dataset = _make_multi_machine_dataset(n_per_machine=40, machines=("JET", "AUG", "D3D"))
    trimmed = pd.concat(
        [
            dataset[dataset[hdb5.TOKAMAK_LABEL_COLUMN] == "JET"],
            dataset[dataset[hdb5.TOKAMAK_LABEL_COLUMN] == "AUG"].head(20),
            dataset[dataset[hdb5.TOKAMAK_LABEL_COLUMN] == "D3D"].head(10),
        ]
    )
    assert hdb5.eligible_tokamaks(trimmed, min_rows=1) == ["JET", "AUG", "D3D"]


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

    without = ax.analyze_extrapolation(
        dataset, min_rows=30, n_splits=3, include_controls=False, include_ladder=False
    )
    assert not set(ax.CONTROL_MODELS) & {t.model_name for t in without.transfers}


def test_control_and_ladder_constants_match_their_factories() -> None:
    """A new rung added to one factory but not its constant would go unreported."""
    assert set(ax.CONTROL_MODELS) == set(hdb5.build_control_models())
    assert set(ax.LADDER_MODELS) == set(ax.build_flexibility_ladder())
    assert not set(ax.CONTROL_MODELS) & set(ax.LADDER_MODELS)


def test_controls_are_excluded_from_the_ranking_claim() -> None:
    """The reversal claim is about the three contenders, not about the controls."""
    assert not (set(ax.CONTROL_MODELS) | set(ax.LADDER_MODELS)) & set(ax.CONTENDER_MODELS)
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


# --- the machine-level bootstrap --------------------------------------------


def _fake_per_machine(values: dict[str, list[float]], machines: list[str]) -> pd.DataFrame:
    """A minimal per-machine report, so bootstrap maths is tested on known input."""
    rows = [
        {"tokamak": machine, "model_name": model, "rmsle": series[index]}
        for model, series in values.items()
        for index, machine in enumerate(machines)
    ]
    return pd.DataFrame(rows)


def test_bootstrap_interval_brackets_the_mean_and_is_deterministic() -> None:
    machines = [f"M{i}" for i in range(8)]
    report = _fake_per_machine({"a": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}, machines)
    first = ax.bootstrap_over_machines(report, n_resamples=500)
    second = ax.bootstrap_over_machines(report, n_resamples=500)
    assert [i.to_json() for i in first] == [i.to_json() for i in second]

    interval = first[0]
    assert interval.mean_rmsle == pytest.approx(0.45)
    assert interval.ci_low < interval.mean_rmsle < interval.ci_high


def test_bootstrap_interval_collapses_when_every_machine_agrees() -> None:
    """No spread between machines means nothing for the resampling to vary."""
    machines = [f"M{i}" for i in range(6)]
    report = _fake_per_machine({"a": [0.3] * 6}, machines)
    interval = ax.bootstrap_over_machines(report, n_resamples=300)[0]
    assert interval.ci_low == pytest.approx(0.3)
    assert interval.ci_high == pytest.approx(0.3)


def test_paired_difference_detects_a_gap_the_marginals_would_hide() -> None:
    """The reason the paired statistic exists, on data built to make the point.

    Model ``b`` is worse than ``a`` on every machine by a constant, but the
    machines differ so much in difficulty that each model's own spread dwarfs
    the gap. Marginal intervals overlap heavily; the paired one must not.
    """
    machines = [f"M{i}" for i in range(10)]
    difficulty = [0.1, 0.3, 0.5, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2]
    report = _fake_per_machine(
        {"a": difficulty, "b": [value + 0.15 for value in difficulty]}, machines
    )
    marginals = {i.model_name: i for i in ax.bootstrap_over_machines(report, n_resamples=1000)}
    assert marginals["a"].ci_high > marginals["b"].ci_low  # the intervals overlap

    gap = ax.bootstrap_paired_difference(report, "b", "a", n_resamples=1000)
    assert gap.mean_difference == pytest.approx(0.15)
    assert gap.excludes_zero
    assert gap.n_machines_a_worse == gap.n_machines == 10


def test_paired_difference_is_antisymmetric_and_rejects_unscored_models() -> None:
    machines = [f"M{i}" for i in range(6)]
    report = _fake_per_machine(
        {"a": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7], "b": [0.3, 0.3, 0.5, 0.4, 0.8, 0.6]}, machines
    )
    forward = ax.bootstrap_paired_difference(report, "a", "b", n_resamples=400)
    backward = ax.bootstrap_paired_difference(report, "b", "a", n_resamples=400)
    assert forward.mean_difference == pytest.approx(-backward.mean_difference)
    assert forward.n_machines_a_worse + backward.n_machines_a_worse <= forward.n_machines

    with pytest.raises(ValueError, match="was not scored"):
        ax.bootstrap_paired_difference(report, "a", "absent", n_resamples=100)


# --- the flexibility ladder ---------------------------------------------------


def test_flexibility_ladder_is_ordered_and_fully_scored() -> None:
    dataset = _make_multi_machine_dataset()
    analysis = ax.analyze_extrapolation(dataset, min_rows=30, n_splits=3, include_ladder=True)
    scored = {transfer.model_name for transfer in analysis.transfers}
    ladder = [name for name, _ in ax.FLEXIBILITY_LADDER]
    assert set(ladder) <= scored
    # Degree 1 is plain ridge and the trees are the far end; the ladder must
    # start at the constrained form or the Result 4d comparison is meaningless.
    assert ladder[0] == "ridge_loglinear"
    assert ladder[-1] == "random_forest"


def test_ladder_can_be_switched_off() -> None:
    dataset = _make_multi_machine_dataset()
    analysis = ax.analyze_extrapolation(dataset, min_rows=30, n_splits=3, include_ladder=False)
    assert "ridge_log_cubic" not in {t.model_name for t in analysis.transfers}


def test_extra_models_may_not_silently_replace_a_zoo_entry() -> None:
    """A name collision would overwrite a model in one split and not the other."""
    from sklearn.dummy import DummyRegressor
    from sklearn.pipeline import Pipeline

    clash = {"random_forest": Pipeline([("model", DummyRegressor())])}
    with pytest.raises(ValueError, match="silently replace"):
        hdb5._assemble_zoo(extra_models=clash)

    combined = hdb5._assemble_zoo(include_controls=True, extra_models=ax.build_flexibility_ladder())
    assert "ridge_log_cubic" in combined
    assert set(hdb5.build_model_zoo()) <= set(combined)
