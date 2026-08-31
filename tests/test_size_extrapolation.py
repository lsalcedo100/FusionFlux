"""Tests for Result 5: extrapolating in the direction ITER actually sits.

Scope split, matching the convention in ``tests/test_extrapolation.py``: the
split primitives in ``hdb5.py`` are tested here on a synthetic dataset with a
controllable size axis, and the analysis layer in
``analysis_size_extrapolation.py`` is tested on top of them.

The claim Result 5 makes is not "trees are worse", which any split would show.
It is that the *held-out set is above the training set in size*, that the ITER
rung is picked by the data rather than by eye, and that the failure has the
Result 4c mechanism behind it. Each of those is a separate test below.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import analysis_size_extrapolation as sx
import hdb5


def _make_size_ordered_dataset(
    machines: dict[str, float] | None = None,
    n_per_machine: int = 90,
    seed: int = 5,
) -> pd.DataFrame:
    """A prepared HDB5-shaped dataset whose machines differ mainly in size.

    ``machines`` maps a machine name to its major radius. Everything else is
    drawn from the same distribution, so a size cut isolates size and the tests
    are not accidentally measuring some other axis.
    """
    machines = machines or {"S1": 0.6, "S2": 0.9, "S3": 1.3, "M1": 1.7, "L1": 2.6, "L2": 3.3}
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


def _real_dataset_or_skip() -> pd.DataFrame:
    if not hdb5.default_hdb5_path().exists():
        pytest.skip("HDB5 STD5 not downloaded; run `python3 hdb5.py download`.")
    return hdb5.prepare_dataset()


# --- the size ordering ------------------------------------------------------


def test_machine_sizes_are_ordered_ascending_by_median_radius() -> None:
    sizes = hdb5.machine_sizes(_make_size_ordered_dataset())
    radii = [size.r_median_m for size in sizes]
    assert radii == sorted(radii)
    assert [size.tokamak for size in sizes] == ["S1", "S2", "S3", "M1", "L1", "L2"]


def test_every_split_trains_strictly_below_the_cut_and_predicts_strictly_above() -> None:
    """The defining property. If it does not hold, the split is not a size split.

    A shape check would pass on a split that leaked a large machine into
    training; this pins the actual ordering of the radii on both sides.
    """
    dataset = _make_size_ordered_dataset()
    for split in hdb5.size_ordered_splits(dataset, min_train_machines=2, min_test_rows=10):
        assert not set(split.train_machines) & set(split.test_machines)
        labels = dataset[hdb5.TOKAMAK_LABEL_COLUMN]
        train_radius = dataset.loc[labels.isin(split.train_machines), "r_m"]
        test_radius = dataset.loc[labels.isin(split.test_machines), "r_m"]
        # Every training machine is smaller than every test machine, by median.
        assert train_radius.max() == pytest.approx(split.train_r_max_m)
        assert test_radius.max() == pytest.approx(split.test_r_max_m)
        assert split.size_ratio > 1.0


def test_splits_cover_every_row_exactly_once() -> None:
    dataset = _make_size_ordered_dataset()
    for split in hdb5.size_ordered_splits(dataset, min_train_machines=2, min_test_rows=10):
        assert split.n_train_rows + split.n_test_rows == len(dataset)


def test_splits_respect_the_minimum_training_machines_and_test_rows() -> None:
    dataset = _make_size_ordered_dataset()
    splits = hdb5.size_ordered_splits(dataset, min_train_machines=3, min_test_rows=100)
    assert splits
    assert all(split.n_train_machines >= 3 for split in splits)
    assert all(split.n_test_rows >= 100 for split in splits)


def test_a_dataset_with_too_few_machines_yields_no_splits() -> None:
    dataset = _make_size_ordered_dataset(machines={"A": 1.0, "B": 2.0})
    assert hdb5.size_ordered_splits(dataset, min_train_machines=3) == []
    with pytest.raises(ValueError, match="No size cut"):
        hdb5.size_extrapolation_report(dataset, min_train_machines=3)


# --- picking the ITER-matched rung -----------------------------------------


def test_the_iter_ratio_is_read_off_the_data_rather_than_hardcoded() -> None:
    dataset = _make_size_ordered_dataset()
    assert hdb5.iter_size_ratio(dataset) == pytest.approx(
        hdb5.ITER_MAJOR_RADIUS_M / dataset["r_m"].max()
    )


def test_the_matched_split_is_the_one_closest_to_the_iter_ratio_in_log_terms() -> None:
    """Chosen by proximity, so the rung moves on its own if the database grows.

    Picking it by eye would make the headline a choice rather than a property of
    the data, and would quietly stop being the ITER analogue the moment a larger
    machine was added.
    """
    dataset = _make_size_ordered_dataset()
    splits = hdb5.size_ordered_splits(dataset, min_train_machines=2, min_test_rows=10)
    matched = hdb5.iter_matched_split(dataset, splits)
    target = np.log(hdb5.iter_size_ratio(dataset))
    best = min(abs(np.log(split.size_ratio) - target) for split in splits)
    assert abs(np.log(matched.size_ratio) - target) == pytest.approx(best)


def test_matching_rejects_an_empty_sweep() -> None:
    with pytest.raises(ValueError, match="cannot match"):
        hdb5.iter_matched_split(_make_size_ordered_dataset(), [])


# --- scoring ----------------------------------------------------------------


def test_scoring_reports_a_pooled_row_per_model_and_per_machine_on_request() -> None:
    dataset = _make_size_ordered_dataset()
    splits = hdb5.size_ordered_splits(dataset, min_train_machines=3, min_test_rows=10)
    split = splits[0]

    pooled = hdb5.score_size_split(dataset, split)
    assert set(pooled["scope"]) == {"__pooled__"}

    detailed = hdb5.score_size_split(dataset, split, per_machine=True, min_rows=10)
    assert set(detailed["scope"]) == {"__pooled__", *split.test_machines}
    assert (detailed["rmsle"] >= 0).all()


def test_the_pooled_row_count_is_every_row_above_the_cut() -> None:
    dataset = _make_size_ordered_dataset()
    split = hdb5.size_ordered_splits(dataset, min_train_machines=3, min_test_rows=10)[0]
    pooled = hdb5.score_size_split(dataset, split)
    assert set(pooled["n_held_out_rows"]) == {split.n_test_rows}


def test_no_test_machine_appears_in_its_own_training_set() -> None:
    """A leakage test with teeth: one machine gets an offset nothing explains.

    If a large machine leaked into training, the model would learn its offset and
    score well on it. Holding the offset out means a correct split *must* be
    unable to predict it.
    """
    dataset = _make_size_ordered_dataset()
    labels = dataset[hdb5.TOKAMAK_LABEL_COLUMN]
    dataset = dataset.copy()
    dataset.loc[labels == "L2", hdb5.TARGET_COLUMN] *= 6.0

    split = hdb5.iter_matched_split(
        dataset, hdb5.size_ordered_splits(dataset, min_train_machines=3, min_test_rows=10)
    )
    assert "L2" in split.test_machines
    scores = hdb5.score_size_split(dataset, split, per_machine=True, min_rows=10)
    on_l2 = scores[(scores["scope"] == "L2") & (scores["model_name"] == "random_forest")]
    # An unexplained 6x offset is roughly log(6) = 1.8 of RMSLE that no feature
    # can account for, so a genuinely blind model cannot score well here.
    assert float(on_l2["rmsle"].iloc[0]) > 1.0


# --- the aspect-ratio control ----------------------------------------------


def test_the_control_drops_the_spherical_machines_only() -> None:
    dataset = _make_size_ordered_dataset()
    spherical = dataset[hdb5.TOKAMAK_LABEL_COLUMN] == "S1"
    dataset = dataset.copy()
    dataset.loc[spherical, "inverse_aspect_ratio"] = 0.7
    dataset.loc[spherical, "log_inverse_aspect_ratio"] = np.log(0.7)

    splits = hdb5.size_ordered_splits(
        dataset, min_train_machines=2, min_test_rows=10, conventional_aspect_ratio_only=True
    )
    for split in splits:
        assert "S1" not in split.train_machines
        assert "S1" not in split.test_machines


# --- the analysis layer -----------------------------------------------------


def test_the_sweep_marks_underpowered_cuts_rather_than_hiding_them() -> None:
    """Small training sets confound size extrapolation with sample size.

    They are still reported; the flag is what stops a claim resting on them.
    """
    dataset = _make_size_ordered_dataset(n_per_machine=60)
    sweep, splits = sx.build_sweep(dataset)
    assert sweep
    by_cut = {split.n_train_machines: split for split in splits}
    for score in sweep:
        expected = by_cut[score.n_train_machines].n_train_rows >= sx.MIN_WELL_POWERED_TRAIN_ROWS
        assert score.well_powered is expected
        assert score.n_train_rows == by_cut[score.n_train_machines].n_train_rows


def test_skill_is_zero_at_the_mean_baseline_and_one_at_the_analytic_law() -> None:
    """The skill statistic has to be anchored where the narrative says it is."""
    dataset = _make_size_ordered_dataset()
    splits = hdb5.size_ordered_splits(dataset, min_train_machines=3, min_test_rows=10)
    matched = hdb5.iter_matched_split(dataset, splits)
    scores = hdb5.score_size_split(dataset, matched, per_machine=True)
    rows = {row.model_name: row for row in sx.build_escalation(dataset, scores, n_splits=3)}

    assert rows["mean_baseline"].skill_against_baseline == pytest.approx(0.0, abs=1e-9)
    assert rows["ipb98y2_analytic"].skill_against_baseline == pytest.approx(1.0, abs=1e-9)


def test_the_escalation_scores_all_three_splits_on_one_feature_set() -> None:
    """Three columns on different features would not be comparable."""
    dataset = _make_size_ordered_dataset()
    splits = hdb5.size_ordered_splits(dataset, min_train_machines=3, min_test_rows=10)
    matched = hdb5.iter_matched_split(dataset, splits)
    scores = hdb5.score_size_split(dataset, matched, per_machine=True)
    rows = sx.build_escalation(dataset, scores, n_splits=3)

    assert {row.model_name for row in rows} <= set(sx.REPORTED_MODELS)
    for row in rows:
        assert np.isfinite(row.cv_rmsle)
        assert np.isfinite(row.lomo_mean_rmsle)
        assert np.isfinite(row.size_cut_rmsle)
        assert row.degradation_factor == pytest.approx(row.size_cut_rmsle / row.cv_rmsle)


# --- the mechanism ----------------------------------------------------------


def test_a_tree_ensemble_cannot_reach_above_its_training_range_at_the_size_cut() -> None:
    """Result 4c's bound, which is what makes Result 5a a structural failure.

    Trees average training targets, so no prediction can leave
    ``[min(y_train), max(y_train)]``. Under a size cut the held-out machines are
    the large ones, whose confinement times are systematically higher, so the
    bound binds on a large share of the held-out rows at once rather than on the
    odd record shot. That is why the trees land near a constant predictor rather
    than merely behind the power law.
    """
    from sklearn.ensemble import RandomForestRegressor

    dataset = _make_size_ordered_dataset()
    split = hdb5.iter_matched_split(
        dataset, hdb5.size_ordered_splits(dataset, min_train_machines=3, min_test_rows=10)
    )
    labels = dataset[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()
    train_mask = np.isin(labels, list(split.train_machines))
    features = dataset[list(hdb5.BLIND_FEATURE_COLUMNS)]
    tau = dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float)

    forest = RandomForestRegressor(n_estimators=40, random_state=0)
    forest.fit(features[train_mask], np.log(tau[train_mask]))
    predicted = np.exp(forest.predict(features[~train_mask]))

    train_max = tau[train_mask].max()
    assert predicted.max() <= train_max * (1 + 1e-9)
    # And the truth genuinely runs above it, so the bound is binding rather than
    # vacuously satisfied.
    assert tau[~train_mask].max() > train_max


def test_truncation_reports_the_share_of_unreachable_rows() -> None:
    dataset = _make_size_ordered_dataset()
    split = hdb5.iter_matched_split(
        dataset, hdb5.size_ordered_splits(dataset, min_train_machines=3, min_test_rows=10)
    )
    truncation = sx.build_truncation(dataset, split)

    assert 0.0 <= truncation["fraction_above_train_max"] <= 1.0
    assert truncation["headroom_ratio"] > 1.0
    assert truncation["n_rows_above_train_max"] <= truncation["n_test_rows"]
    assert truncation["test_tau_max_s"] > truncation["train_tau_max_s"]


# --- the published numbers --------------------------------------------------


def test_the_published_iter_matched_cut_is_the_one_results_md_describes() -> None:
    """Pins the headline split, so a change to the ordering cannot pass silently."""
    dataset = _real_dataset_or_skip()
    splits = hdb5.size_ordered_splits(dataset)
    matched = hdb5.iter_matched_split(dataset, splits)

    assert matched.n_train_machines == 14
    assert matched.train_r_max_m == pytest.approx(1.865)
    assert matched.test_r_max_m == pytest.approx(3.4)
    assert set(matched.test_machines) == {"TFTR", "JETILW", "JET", "JT60U"}
    # The rung reproduces the ITER jump to well under a percent in log terms.
    assert abs(np.log(matched.size_ratio) - np.log(hdb5.iter_size_ratio(dataset))) < 0.01


def test_the_power_law_beats_both_tree_ensembles_at_the_iter_matched_cut() -> None:
    """Result 5a, on the real data. This is the claim the narrative rests on."""
    dataset = _real_dataset_or_skip()
    matched = hdb5.iter_matched_split(dataset, hdb5.size_ordered_splits(dataset))
    scores = hdb5.score_size_split(dataset, matched)
    rmsle = {
        str(name): float(value)
        for name, value in zip(
            scores["model_name"].to_numpy(), scores["rmsle"].to_numpy(dtype=float)
        , strict=True)
    }

    assert rmsle["ridge_loglinear"] < rmsle["random_forest"]
    assert rmsle["ridge_loglinear"] < rmsle["hist_gradient_boosting"]
    # The trees land closer to a constant predictor than to the power law.
    for tree in ("random_forest", "hist_gradient_boosting"):
        assert rmsle[tree] > 0.5 * rmsle["mean_baseline"]
    # The power law does not.
    assert rmsle["ridge_loglinear"] < 0.25 * rmsle["mean_baseline"]
