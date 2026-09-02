"""Result 14: the kernel ladder, its control, and the bound it demonstrates.

The risk in this result is not that a Gaussian process is fitted wrongly. It is
that the three rungs stop being the same experiment. The whole claim rests on
only one thing differing between them, the kernel's behaviour far from the data,
so these tests pin the things that must stay equal (the optimizer, the feature
handling, the rows each fold sees) and the one property that must differ.

The second risk is the one that nearly sank the result during development. A
hand-picked RBF length scale, an order of magnitude longer than the data
supports, made the flexible-and-unbounded rung look like a failure. Marginal
likelihood fixes that, so ``test_learned_kernel_is_not_sensitive_to_the_subsample``
guards the shortcut that makes marginal likelihood affordable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone, is_regressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import gp
import hdb5


def _make_dataset(n_per_machine: int = 80, seed: int = 11) -> pd.DataFrame:
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
            0.0562 * ip**0.93 * bt**0.15 * nel**0.41 * plth**-0.69
            * rgeo**1.97 * eps**0.58 * kappa**0.78 * meff**0.19
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


def _xy(dataset: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    features = dataset[list(hdb5.BLIND_FEATURE_COLUMNS)]
    return features, np.log(dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float))


# --- the ladder is one family with one thing varied -------------------------


def test_the_ladder_has_exactly_the_three_rungs() -> None:
    models = gp.build_gp_models()
    assert set(models) == {"gp_rbf", "gp_linear", "gp_linear_rbf"}


@pytest.mark.parametrize("name", gp.KERNEL_NAMES)
def test_every_rung_is_a_regressor_and_clones(name: str) -> None:
    """sklearn's contract, which the shared splits rely on to refit per fold."""
    estimator = gp.SubsampledGaussianProcess(kernel_name=name)
    assert is_regressor(estimator)
    assert clone(estimator).get_params() == estimator.get_params()


def test_unknown_kernel_is_refused() -> None:
    with pytest.raises(ValueError, match="Unknown kernel"):
        gp.build_kernel("matern")


def test_rungs_differ_only_in_their_long_range_terms() -> None:
    """The linear and RBF rungs are each a strict subset of the combined one.

    This is the design claim: a difference between the three cannot come from
    the noise model or the starting values, because those are shared.
    """
    rbf = str(gp.build_kernel("rbf"))
    linear = str(gp.build_kernel("linear"))
    combined = str(gp.build_kernel("linear_rbf"))

    assert "DotProduct" in linear and "DotProduct" not in rbf
    assert "RBF" in rbf and "RBF" not in linear
    assert "DotProduct" in combined and "RBF" in combined
    for kernel in (rbf, linear, combined):
        assert "WhiteKernel(noise_level=0.05)" in kernel


# --- Result 14a: the control ------------------------------------------------


def test_linear_kernel_reproduces_the_log_linear_power_law() -> None:
    """A dot-product kernel is Bayesian linear regression in the log features.

    If this drifts, the ladder is no longer anchored to a model whose behaviour
    Results 4 and 8 already establish, and the other two rungs mean nothing.
    """
    dataset = _make_dataset()
    features, log_tau = _xy(dataset)

    gp_linear = gp.build_gp_models(n_tuning_rows=200)["gp_linear"].fit(features, log_tau)
    ridge = Pipeline(
        [("scale", StandardScaler()), ("model", Ridge(alpha=1.0, solver="svd"))]
    ).fit(features, log_tau)

    gap = float(np.sqrt(np.mean((gp_linear.predict(features) - ridge.predict(features)) ** 2)))
    assert gap < 0.02, f"linear-kernel GP and ridge disagree by {gap:.4f} in log space"


# --- Result 14b: the bound --------------------------------------------------


def test_bounded_kernel_reverts_to_the_training_mean_far_from_the_data() -> None:
    """The RBF rung's failure mode, asserted directly rather than inferred.

    Result 4c makes the equivalent assertion for tree ensembles: no tree can
    output a value above the largest training target. The GP analogue is that an
    RBF kernel decays to zero with distance, so the posterior returns to its
    prior mean. Far enough away, the prediction is the training mean whatever
    the features say, and that is what makes it a hard limit rather than a
    shortfall that more data or more tuning could close.
    """
    dataset = _make_dataset()
    features, log_tau = _xy(dataset)
    scaled = StandardScaler().fit_transform(features)

    model = gp.SubsampledGaussianProcess(kernel_name="rbf", n_tuning_rows=200)
    model.fit(scaled, log_tau)

    # "Far" has to be measured against the length scale the fit actually chose,
    # not against a fixed number of standard deviations. On data this smooth the
    # optimizer can select a very long scale, and a query 50 standard deviations
    # out is then still well inside the kernel's reach. The claim is asymptotic,
    # so the test places the query where the kernel has provably decayed.
    length_scale = max(
        float(np.exp(theta))
        for name, theta in zip(
            [hp.name for hp in model.kernel_.hyperparameters],
            model.kernel_.theta,
            strict=True,
        )
        if "length_scale" in name
    )
    far = np.full((1, scaled.shape[1]), 30.0 * length_scale)
    prediction = float(model.predict(far)[0])

    assert prediction == pytest.approx(float(log_tau.mean()), abs=1e-3)


def test_unbounded_kernel_keeps_moving_far_from_the_data() -> None:
    """The contrast that makes the previous test a property of the kernel.

    Same family, same optimizer, same rows. Adding a dot-product term is the
    only change, and it is enough that the prediction no longer collapses.
    """
    dataset = _make_dataset()
    features, log_tau = _xy(dataset)

    model = gp.build_gp_models(n_tuning_rows=200)["gp_linear_rbf"].fit(features, log_tau)

    near = features.mean().to_frame().T
    far = features.mean().to_frame().T + 50.0 * features.std().to_frame().T
    moved = abs(float(model.predict(far)[0]) - float(model.predict(near)[0]))

    assert moved > 1.0, f"the unbounded rung moved only {moved:.3f} in log space"


def test_reversion_diagnostic_is_zero_for_a_perfect_spread() -> None:
    log_actual = np.array([0.1, 0.5, 1.0, 1.7, 2.4])
    diagnostic = gp.reversion_diagnostic("x", log_actual, log_actual.copy(), 0.0)
    assert diagnostic.reversion == pytest.approx(0.0)


def test_reversion_diagnostic_is_one_for_a_constant_prediction() -> None:
    log_actual = np.array([0.1, 0.5, 1.0, 1.7, 2.4])
    constant = np.full_like(log_actual, log_actual.mean())
    diagnostic = gp.reversion_diagnostic("x", log_actual, constant, float(log_actual.mean()))
    assert diagnostic.reversion == pytest.approx(1.0)
    assert diagnostic.predicted_mean_offset == pytest.approx(0.0)


# --- the shortcut that makes marginal likelihood affordable -----------------


def test_learned_kernel_is_not_sensitive_to_the_subsample() -> None:
    """Tuning on a subsample must not be tuning on the subsample's accidents.

    The hyperparameters are fitted on ``n_tuning_rows`` rather than every row,
    because marginal-likelihood optimization pays an O(n^3) factorization at
    every step. That is only legitimate if the answer is stable in the size of
    the subsample, so this fits the same rung twice at different sizes and
    compares the learned kernel.
    """
    dataset = _make_dataset(n_per_machine=140)
    features, log_tau = _xy(dataset)

    small = gp.SubsampledGaussianProcess(kernel_name="linear_rbf", n_tuning_rows=200)
    large = gp.SubsampledGaussianProcess(kernel_name="linear_rbf", n_tuning_rows=500)
    scaled = StandardScaler().fit_transform(features)
    small.fit(scaled, log_tau)
    large.fit(scaled, log_tau)

    small_theta = np.exp(small.kernel_.theta)
    large_theta = np.exp(large.kernel_.theta)
    relative = np.abs(small_theta - large_theta) / np.maximum(np.abs(large_theta), 1e-12)
    assert relative.max() < 0.5, (
        f"learned kernel moved by {relative.max():.1%} between subsample sizes: "
        f"{small.kernel_} against {large.kernel_}"
    )


def test_tuning_uses_only_the_rows_it_was_given() -> None:
    """No held-out row can reach the hyperparameters.

    The estimator is handed one fold at a time by the shared splits, so this is
    the property that keeps the tuning honest: everything it sees arrives
    through ``fit``, and it retains nothing between calls.
    """
    dataset = _make_dataset()
    features, log_tau = _xy(dataset)
    scaled = StandardScaler().fit_transform(features)

    half = len(scaled) // 2
    first = gp.SubsampledGaussianProcess(n_tuning_rows=150).fit(scaled[:half], log_tau[:half])
    second = gp.SubsampledGaussianProcess(n_tuning_rows=150).fit(scaled[:half], log_tau[:half])
    assert np.allclose(first.kernel_.theta, second.kernel_.theta)

    other = gp.SubsampledGaussianProcess(n_tuning_rows=150).fit(scaled[half:], log_tau[half:])
    assert not np.allclose(first.kernel_.theta, other.kernel_.theta), (
        "two disjoint folds produced the same kernel; the fit is not using its rows"
    )


def test_tuning_subsample_is_capped_by_the_fold_size() -> None:
    dataset = _make_dataset(n_per_machine=20)
    features, log_tau = _xy(dataset)
    model = gp.SubsampledGaussianProcess(n_tuning_rows=10_000).fit(
        StandardScaler().fit_transform(features), log_tau
    )
    assert model.n_tuning_rows_ == len(features)


def test_predict_returns_a_standard_deviation_when_asked() -> None:
    """Result 14d reads the posterior interval, so the estimator has to expose it."""
    dataset = _make_dataset()
    features, log_tau = _xy(dataset)
    scaled = StandardScaler().fit_transform(features)
    model = gp.SubsampledGaussianProcess(n_tuning_rows=200).fit(scaled, log_tau)

    mean, std = model.predict(scaled[:20], return_std=True)
    assert mean.shape == (20,)
    assert std.shape == (20,)
    assert np.all(std > 0.0)


# --- the ladder runs through the shared splits ------------------------------


def test_the_ladder_scores_through_the_existing_size_split() -> None:
    """The rungs must be scorable by the same machinery as every other model.

    Result 14's whole comparison depends on the GP rows and the tree rows coming
    out of one call, on one split, with one scorer. A parallel pipeline would
    make the table incomparable in exactly the way Result 4 was designed to
    avoid.
    """
    dataset = _make_dataset()
    splits = hdb5.size_ordered_splits(dataset, min_train_machines=2, min_test_rows=20)
    if not splits:
        pytest.skip("synthetic dataset yielded no usable size cut")

    scores = hdb5.score_size_split(
        dataset, splits[0], extra_models=gp.build_gp_models(n_tuning_rows=150)
    )
    scored = set(scores["model_name"])
    assert {"gp_rbf", "gp_linear", "gp_linear_rbf"} <= scored
    assert {"random_forest", "ridge_loglinear"} <= scored
    assert scores["rmsle"].notna().all()
