"""The deployment-matched estimator, and whether it is fitted correctly.

``analysis_mixed_model`` answers an objection rather than adding a model the
paper recommends: the clustered-validation literature the split design cites
pairs that design with a mixed model carrying a random intercept per device, and
Limitations conceded it was never scored. Since the whole point is to have
fitted it properly, these tests check the fit rather than the conclusion.

The GLS solve is a closed form specific to a one-way random intercept, and a
wrong theta would still produce plausible-looking numbers, so it is checked
against a brute-force solve that builds the covariance matrix and inverts it.
The REML criterion is checked the same way. Beyond that: the model must reduce
to the pooled fit at a zero variance ratio, since every claim made from the
sweep rests on that limit being the power law and not merely near it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import linalg

import analysis_mixed_model as mm
import hdb5

RESULTS = Path(__file__).resolve().parents[1] / "results" / "mixed_model.json"


@pytest.fixture(scope="module")
def committed() -> dict:
    if not RESULTS.exists():
        pytest.skip("no results/mixed_model.json; run `python3 analysis_mixed_model.py`")
    return json.loads(RESULTS.read_text())


def _synthetic(seed: int = 0):
    """A small grouped problem with known structure, sized for a dense inverse."""
    rng = np.random.default_rng(seed)
    sizes = (7, 9, 4, 11, 6)
    groups = np.concatenate([[f"g{i}"] * n for i, n in enumerate(sizes)])
    features = rng.normal(size=(len(groups), 3))
    design = np.column_stack([np.ones(len(groups)), features])
    intercepts = rng.normal(scale=0.8, size=len(sizes))
    response = (
        design @ rng.normal(size=design.shape[1])
        + np.repeat(intercepts, sizes)
        + rng.normal(scale=0.3, size=len(groups))
    )
    return features, design, response, groups


def _brute_force(design, response, groups, variance_ratio):
    """GLS and the REML criterion with the covariance built and inverted whole."""
    same = groups[:, None] == groups[None, :]
    covariance = np.eye(len(response)) + variance_ratio * same
    inverse = np.linalg.inv(covariance)
    information = design.T @ inverse @ design
    coefficients = np.linalg.solve(information, design.T @ inverse @ response)
    residual = response - design @ coefficients
    degrees = len(response) - design.shape[1]
    scale = float(residual @ inverse @ residual) / degrees
    criterion = degrees * np.log(scale) + np.linalg.slogdet(covariance)[1] + np.linalg.slogdet(information)[1]
    return coefficients, criterion


@pytest.mark.parametrize("variance_ratio", [1e-6, 0.05, 0.5, 2.0, 20.0])
def test_whitening_reproduces_a_dense_gls_solve(variance_ratio: float) -> None:
    _, design, response, groups = _synthetic()
    blocks = mm._group_slices(groups)
    expected, expected_criterion = _brute_force(design, response, groups, variance_ratio)

    whitened_design, whitened_response, _ = mm._whiten(design, response, blocks, variance_ratio)
    coefficients, *_ = linalg.lstsq(whitened_design, whitened_response)
    _, _, criterion = mm._solve(design, response, blocks, variance_ratio)

    assert coefficients == pytest.approx(expected, abs=1e-10)
    assert criterion == pytest.approx(expected_criterion, rel=1e-12)


def test_a_zero_variance_ratio_is_the_pooled_fit() -> None:
    """The sweep's lambda = 0 row has to be ordinary least squares exactly.

    Every comparison in the paper that says the mixed model does not beat the
    pooled power law is really the statement that its optimum sits at or beside
    this row, so the row has to be the pooled fit rather than an approximation
    of it.
    """
    features, design, response, groups = _synthetic()
    fit = mm.fit_random_intercept(features, response, groups, variance_ratio=0.0)
    expected, *_ = linalg.lstsq(design, response)
    assert fit.coefficients == pytest.approx(expected, abs=1e-10)
    assert fit.group_variance == 0.0
    assert fit.intraclass_correlation == 0.0


def test_group_slices_partition_every_row_exactly_once() -> None:
    _, _, _, groups = _synthetic()
    blocks = mm._group_slices(groups)
    assert sorted(np.concatenate(blocks).tolist()) == list(range(len(groups)))
    for block in blocks:
        assert len(set(groups[block])) == 1


def test_reml_recovers_a_known_variance_ratio_on_many_groups() -> None:
    """With enough groups the profiled optimum should find the ratio it was given.

    Eleven devices is not enough groups for this to be sharp, which is the
    reason the analysis sweeps the ratio as well as fitting it. Here the point
    is only that the estimator is unbiased in the regime where it is well posed,
    so an unstable fold in the real data reads as small-sample noise rather than
    a bug.
    """
    rng = np.random.default_rng(11)
    n_groups, per_group, true_ratio = 200, 12, 2.0
    groups = np.repeat([f"g{i}" for i in range(n_groups)], per_group)
    features = rng.normal(size=(len(groups), 2))
    response = (
        features @ np.array([0.7, -0.4])
        + np.repeat(rng.normal(scale=np.sqrt(true_ratio) * 0.5, size=n_groups), per_group)
        + rng.normal(scale=0.5, size=len(groups))
    )
    fit = mm.fit_random_intercept(features, response, groups)
    assert fit.variance_ratio == pytest.approx(true_ratio, rel=0.35)


def test_independent_columns_drops_the_exact_dependency() -> None:
    """log a = log R + log epsilon holds by construction, so one column must go."""
    if not hdb5.default_hdb5_path().exists():
        pytest.skip("HDB5 STD5 not downloaded; run `python3 hdb5.py download`.")
    dataset = hdb5.prepare_dataset()
    columns = list(hdb5.BLIND_FEATURE_COLUMNS)
    features = dataset[columns].to_numpy(dtype=float)

    kept = mm._independent_columns(features)
    assert len(kept) == len(columns) - 1
    design = np.column_stack([np.ones(len(features)), features[:, list(kept)]])
    assert np.linalg.matrix_rank(design) == design.shape[1]


def test_a_held_out_prediction_ignores_which_dependent_column_was_kept() -> None:
    """The dependency holds on held-out rows too, so the choice cannot move a number."""
    if not hdb5.default_hdb5_path().exists():
        pytest.skip("HDB5 STD5 not downloaded; run `python3 hdb5.py download`.")
    dataset = hdb5.prepare_dataset()
    columns = list(hdb5.BLIND_FEATURE_COLUMNS)
    features = dataset[columns].to_numpy(dtype=float)
    response = np.log(dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float))
    groups = hdb5.with_device_column(dataset)[hdb5.DEVICE_COLUMN].to_numpy()

    held = groups == "JET"
    fit = mm.fit_random_intercept(features[~held], response[~held], groups[~held], variance_ratio=0.0)
    baseline = fit.predict_population(features[held])

    # Refit on the same rows with the redundant column removed by hand instead.
    dropped = tuple(i for i in range(len(columns)) if columns[i] != "log_a_m")
    trimmed = features[:, list(dropped)]
    other = mm.fit_random_intercept(trimmed[~held], response[~held], groups[~held], variance_ratio=0.0)
    assert other.predict_population(trimmed[held]) == pytest.approx(baseline, abs=1e-8)


def test_committed_report_says_the_pooled_limit_is_at_least_as_good(committed: dict) -> None:
    """The claim the paper makes from this artifact, stated as a test.

    The mixed model is offered as an answer to an objection, and the answer is
    that it does not beat the pooled power law on an unseen device even when its
    one free quantity is chosen against the held-out score it is being judged
    on. If that stops holding, the paragraph in the paper is wrong.
    """
    for arm in ("leave_one_label_out", "leave_one_device_out"):
        sweep = committed[arm]["variance_ratio_sweep"]
        assert sweep["best_mean_rmsle"] <= sweep["pooled_limit_rmsle"] + 1e-12
        assert sweep["pooled_limit_rmsle"] - sweep["best_mean_rmsle"] < 0.01
        assert committed[arm]["reml"]["mean_rmsle"] > sweep["pooled_limit_rmsle"]


def test_committed_sweep_covers_the_pooled_limit(committed: dict) -> None:
    for arm in ("leave_one_label_out", "leave_one_device_out"):
        ratios = [row["variance_ratio"] for row in committed[arm]["variance_ratio_sweep"]["grid"]]
        assert 0.0 in ratios
        assert ratios == sorted(ratios)
